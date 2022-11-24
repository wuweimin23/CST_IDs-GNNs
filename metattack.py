import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from gcn import GCN,GCN_head
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
from tqdm import tqdm
import utils
import math
import scipy.sparse as sp
from sam import SAM
from torch.optim import SGD

class BaseMeta(Module):

    def __init__(self, nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters, attack_features, lambda_, device, Gibbs,  average , sam ,with_bias=False, lr=0.01, with_relu=False):
        super(BaseMeta, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.nfeat = nfeat
        self.nclass = nclass
        self.with_bias = with_bias
        self.with_relu = with_relu
        self.Gibbs = Gibbs
        self.average = average
        self.sam = sam

        ### need to modify
        self.gcn_feature = GCN(nfeat=nfeat,
                       nhid=hidden_sizes[0],
                       nclass=nclass,
                       dropout=0.5,
                       with_relu=False)
        self.gcn_head_source = GCN_head(nfeat=nfeat,
                       nhid=hidden_sizes[0],
                       nclass=nclass,
                       dropout=0.5,
                       with_relu=False)
        self.gcn_head_target = GCN_head(nfeat=nfeat,
                       nhid=hidden_sizes[0],
                       nclass=nclass,
                       dropout=0.5,
                       with_relu=False)
        

        self.train_iters = train_iters
 
        if self.sam == True:
            base_optimizer = SGD
            self.surrogate_optimizer_feature = SAM(self.gcn_feature.parameters(), base_optimizer, lr = 0.005,
                                                momentum=0.9, weight_decay=1e-3, adaptive = True, rho = 0.5)
            self.surrogate_optimizer_source = SAM(self.gcn_head_source.parameters(), base_optimizer, lr = 0.005,
                                                momentum=0.9, weight_decay=1e-3, adaptive = True, rho = 0.5)
            self.surrogate_optimizer_target = SAM(self.gcn_head_target.parameters(), base_optimizer, lr = 0.005,
                                                momentum=0.9, weight_decay=1e-3, adaptive = True, rho = 0.5)
        else:
            self.surrogate_optimizer_feature = optim.Adam(self.gcn_feature.parameters(), lr=lr, weight_decay=5e-4)
            self.surrogate_optimizer_source = optim.Adam(self.gcn_head_source.parameters(), lr=lr, weight_decay=5e-4)
            self.surrogate_optimizer_target = optim.Adam(self.gcn_head_target.parameters(), lr=lr, weight_decay=5e-4)

        if self.Gibbs == 2:
            self.ts_loss = TsallisEntropy(temperature=2.0, alpha = 1.9)

        self.attack_features = attack_features
        self.lambda_ = lambda_
        self.device = device
        self.nnodes = nnodes

        self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes))
        self.adj_changes.data.fill_(0)

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.

        Returns
        -------
        torch.Tensor shape [N, N], float with ones everywhere except the entries of potential singleton nodes,
        where the returned tensor has value 0.

        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()

        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask

    def train_surrogate(self, features, adj, labels, idx_train, idx_unlabeled, train_iters=500):
        print('=== training surrogate model to predict unlabled data for self-training')
        surrogate = self.gcn_feature
        surrogate_head_source = self.gcn_head_source
        surrogate_head_target = self.gcn_head_target
        surrogate.initialize()
        surrogate_head_source.initialize()
        surrogate_head_target.initialize()

        adj_norm = utils.normalize_adj_tensor(adj)
        surrogate.train()
        surrogate_head_source.train()
        surrogate_head_target.train()
        
        if self.sam == 1:
            for i in range(train_iters):
                #Forward step
                output = surrogate(features, adj_norm)
                _, output = surrogate_head_source(output, adj_norm)
                pseudo_unlabeled = output.detach().argmax(1)[idx_unlabeled]
                #Reverse step
                #(1)
                output = surrogate(features, adj_norm)
                _, output = surrogate_head_target(output, adj_norm)
                loss_train = F.nll_loss(output[idx_unlabeled], pseudo_unlabeled)
                loss_train.backward()
                self.surrogate_optimizer_target.first_step(zero_grad=True)
                
                output = surrogate(features, adj_norm)
                _, output = surrogate_head_target(output, adj_norm)
                loss_train = F.nll_loss(output[idx_unlabeled], pseudo_unlabeled)
                loss_train.backward()
                self.surrogate_optimizer_target.second_step(zero_grad=True)

                #(2)
                output1 = surrogate(features, adj_norm)
                output_feature, output1 = surrogate_head_source(output1, adj_norm)
                output2 = surrogate(features, adj_norm)
                _, output2 = surrogate_head_target(output2, adj_norm)
                if self.Gibbs == 1:
                    loss_train_1 = F.nll_loss(output1[idx_train], labels[idx_train]) 
                    loss_train_2 = F.nll_loss(output2[idx_train], labels[idx_train]) 
                    loss_train_3 = entropy(output_feature[idx_unlabeled])
                    loss_train = loss_train_1 + loss_train_2 + (loss_train_1.detach()/loss_train_3.detach()) * loss_train_3
                elif self.Gibbs == 2:  #ts_loss
                    loss_train_1 = F.nll_loss(output1[idx_train], labels[idx_train]) 
                    loss_train_2 = F.nll_loss(output2[idx_train], labels[idx_train]) 
                    loss_train_3 = self.ts_loss(output_feature[idx_unlabeled])
                    loss_train = loss_train_1 + loss_train_2 + (loss_train_1.detach()/loss_train_3.detach()) * loss_train_3
                else:
                    loss_train = F.nll_loss(output1[idx_train], labels[idx_train]) + F.nll_loss(output2[idx_train], labels[idx_train])
                loss_train.backward()
                self.surrogate_optimizer_source.first_step(zero_grad=True)
                self.surrogate_optimizer_feature.first_step(zero_grad=True)

                output1 = surrogate(features, adj_norm)
                output_feature, output1 = surrogate_head_source(output1, adj_norm)
                output2 = surrogate(features, adj_norm)
                _, output2 = surrogate_head_target(output2, adj_norm)
                if self.Gibbs == 1:
                    loss_train_1 = F.nll_loss(output1[idx_train], labels[idx_train]) 
                    loss_train_2 = F.nll_loss(output2[idx_train], labels[idx_train]) 
                    loss_train_3 = entropy(output_feature[idx_unlabeled])
                    loss_train = loss_train_1 + loss_train_2 + (loss_train_1.detach()/loss_train_3.detach()) * loss_train_3
                elif self.Gibbs == 2:
                    loss_train_1 = F.nll_loss(output1[idx_train], labels[idx_train]) 
                    loss_train_2 = F.nll_loss(output2[idx_train], labels[idx_train]) 
                    loss_train_3 = self.ts_loss(output_feature[idx_unlabeled])
                    loss_train = loss_train_1 + loss_train_2 + (loss_train_1.detach()/loss_train_3.detach()) * loss_train_3
                else:
                    loss_train = F.nll_loss(output1[idx_train], labels[idx_train]) + F.nll_loss(output2[idx_train], labels[idx_train])
                loss_train.backward()
                self.surrogate_optimizer_source.second_step(zero_grad=True)
                self.surrogate_optimizer_feature.second_step(zero_grad=True)
        else:
            for i in range(train_iters):
                #Forward step
                output = surrogate(features, adj_norm)
                _, output = surrogate_head_source(output, adj_norm)
                pseudo_unlabeled = output.detach().argmax(1)[idx_unlabeled]
                #Reverse step
                #(1)
                self.surrogate_optimizer_target.zero_grad()
                output = surrogate(features, adj_norm)
                _, output = surrogate_head_target(output, adj_norm)
                loss_train = F.nll_loss(output[idx_unlabeled], pseudo_unlabeled)
                loss_train.backward()
                self.surrogate_optimizer_target.step()

                #(2)
                self.surrogate_optimizer_feature.zero_grad()
                self.surrogate_optimizer_source.zero_grad()
                output1 = surrogate(features, adj_norm)
                output_feature, output1 = surrogate_head_source(output1, adj_norm)
                output2 = surrogate(features, adj_norm)
                _, output2 = surrogate_head_target(output2, adj_norm)
                if self.Gibbs == 1:
                    loss_train_1 = F.nll_loss(output1[idx_train], labels[idx_train]) 
                    loss_train_2 = F.nll_loss(output2[idx_train], labels[idx_train]) 
                    loss_train_3 = entropy(output_feature[idx_unlabeled])
                    loss_train = loss_train_1 + loss_train_2 + (loss_train_1.detach()/loss_train_3.detach()) * loss_train_3
                elif self.Gibbs == 2:
                    loss_train_1 = F.nll_loss(output1[idx_train], labels[idx_train]) 
                    loss_train_2 = F.nll_loss(output2[idx_train], labels[idx_train]) 
                    loss_train_3 = self.ts_loss(output_feature[idx_unlabeled])
                    loss_train = loss_train_1 + loss_train_2 + (loss_train_1.detach()/loss_train_3.detach()) * loss_train_3
                else:
                    loss_train = F.nll_loss(output1[idx_train], labels[idx_train]) + F.nll_loss(output2[idx_train], labels[idx_train])
                loss_train.backward()
                self.surrogate_optimizer_source.step()
                self.surrogate_optimizer_feature.step()
                
                #select confident pseudo labels
                with torch.no_grad():
                    output = surrogate(features, adj_norm)
                    prediction_s, output_s = surrogate_head_source(output, adj_norm)
                    prediction_t, output_t = surrogate_head_target(output, adj_norm)
                    max_output_s = torch.max(F.softmax(prediction_s, dim=1), dim = 1)[0]
                    max_output_t = torch.max(F.softmax(prediction_t, dim=1), dim = 1)[0]
                    pseudo_max = 0.90
                    add_index = np.array(torch.from_numpy(idx_unlabeled)[max_output_s[idx_unlabeled].cpu() > pseudo_max])
                    add_index = np.array(torch.from_numpy(add_index)[max_output_t[add_index].cpu() > pseudo_max])
                    if (type(add_index) == np.int64):
                        add_index = np.array([add_index])
                    idx_train = np.concatenate((idx_train, add_index))
                    idx_unlabeled = np.setdiff1d(idx_unlabeled, add_index)


        if self.average == 1:
            torch.save(surrogate_head_source.state_dict(), 'source_state_Gibbs_{0}_SAM_{1}'.format(self.Gibbs,self.sam))
            torch.save(surrogate_head_target.state_dict(), 'target_state_Gibbs_{0}_SAM_{1}'.format(self.Gibbs,self.sam))
            source_state = torch.load('source_state_Gibbs_{0}_SAM_{1}'.format(self.Gibbs,self.sam))
            target_state = torch.load('target_state_Gibbs_{0}_SAM_{1}'.format(self.Gibbs,self.sam))
            for i in source_state:
                source_state[i] = (source_state[i]+target_state[i]) * 0.5
            surrogate_head_source.load_state_dict(source_state)

        # Predict the labels of the unlabeled nodes to use them for self-training.
        surrogate.eval()
        surrogate_head_source.eval()
        surrogate_head_target.eval()
        output = surrogate(features, adj_norm)
        _, output = surrogate_head_source(output, adj_norm)
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        # reset parameters for later updating
        surrogate.initialize()
        surrogate_head_source.initialize()
        surrogate_head_target.initialize()
        return labels_self_training


    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.
        """

        t_d_min = torch.tensor(2.0).to(self.device)
        t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff)

        return allowed_mask, current_ratio


class Metattack(BaseMeta):

    def __init__(self, nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters,
                 attack_features, device, Gibbs,  average, sam ,lambda_=0.5, with_relu=False, with_bias=False, lr=0.1, momentum=0.9):

        super(Metattack, self).__init__(nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters, attack_features, lambda_, device, Gibbs,  average, sam ,with_bias=with_bias, with_relu=with_relu)

        self.momentum = momentum
        self.lr = lr

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        previous_size = nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            bias = Parameter(torch.FloatTensor(nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            b_velocity = torch.zeros(bias.shape).to(device)
            previous_size = nhid

            self.weights.append(weight)
            self.biases.append(bias)
            self.w_velocities.append(w_velocity)
            self.b_velocities.append(b_velocity)

        output_weight = Parameter(torch.FloatTensor(previous_size, nclass).to(device))
        output_bias = Parameter(torch.FloatTensor(nclass).to(device))
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        output_b_velocity = torch.zeros(output_bias.shape).to(device)

        self.weights.append(output_weight)
        self.biases.append(output_bias)
        self.w_velocities.append(output_w_velocity)
        self.b_velocities.append(output_b_velocity)

        self._initialize()

    def _initialize(self):

        for w, b in zip(self.weights, self.biases):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)


    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()

        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w.to(features.device, non_blocking = True)) + b
                else:
                    hidden = adj_norm @ hidden @ w.to(features.device, non_blocking = True) + b
                if self.with_relu:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=False)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=False)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]
        del hidden, output
        torch.cuda.empty_cache()
                

    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):

        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                w = w.to(hidden.device)
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)

        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled


        adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=False)[0]
        print(f'GCN loss on unlabled data: {loss_test_val.item()}')
        print(f'GCN acc on unlabled data: {utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()}')
        print(f'attack loss: {attack_loss.item()}')
        
        del hidden, output
        torch.cuda.empty_cache()

        return adj_grad


    def forward(self, features, ori_adj, labels, idx_train, idx_unlabeled, perturbations, device_1, device_2, ll_constraint=True, ll_cutoff=0.004):
        self.sparse_features = sp.issparse(features)

        labels_self_training = self.train_surrogate(features, ori_adj, labels, idx_train, idx_unlabeled)
        features = features.to(device_1, non_blocking = True)
        labels = labels.to(device_1, non_blocking = True)
        labels_self_training = labels_self_training.to(device_1, non_blocking = True)
        
        features2 = features.to(device_2, non_blocking = True)
        labels2 = labels.to(device_2, non_blocking = True)
        labels_self_training = labels_self_training.to(device_2, non_blocking = True)

        for i in tqdm(range(perturbations), desc="Perturbing graph"):
            self.adj_changes.data = self.adj_changes.data.to(device_1, non_blocking = True)
            ori_adj = ori_adj.to(device_1, non_blocking = True)
            adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
            ind = np.diag_indices(self.adj_changes.shape[0])
            adj_changes_symm = torch.clamp(adj_changes_square + torch.transpose(adj_changes_square, 1, 0), -1, 1)

            modified_adj = (adj_changes_symm + ori_adj).to(device_1, non_blocking = True)
            del adj_changes_symm
            adj_norm = utils.normalize_adj_tensor(modified_adj).to(device_1, non_blocking = True)

            self.inner_train(features, adj_norm, idx_train, idx_unlabeled, labels)
            adj_grad = self.get_meta_grad(features2, adj_norm.to(device_2, non_blocking = True), idx_train, idx_unlabeled, labels2, labels_self_training).to(device_1, non_blocking = True)
            
            with torch.no_grad():
                adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
                del adj_grad
                adj_meta_grad -= adj_meta_grad.min()
                adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
                singleton_mask = self.filter_potential_singletons(modified_adj)
                torch.cuda.empty_cache()
                adj_meta_grad = adj_meta_grad *  singleton_mask
                torch.cuda.empty_cache()
                if ll_constraint:
                    allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff).to(device_1, non_blocking = True)
                    adj_meta_grad = adj_meta_grad * allowed_mask

                # Get argmax of the meta gradients.
                adj_meta_argmax = torch.argmax(adj_meta_grad)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)

                # modified_adj = modified_adj.to(self.device, non_blocking = True)
                # ori_adj = ori_adj.to(self.device, non_blocking = True)
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)

                if self.attack_features:
                    pass
                torch.cuda.empty_cache()

        return self.adj_changes + ori_adj


class MetaApprox(BaseMeta):

    def __init__(self, nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters, attack_features, lambda_, device, Gibbs ,  average , sam, with_relu=False, with_bias=False, lr=0.01):
        super(MetaApprox, self).__init__(nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters, attack_features, lambda_, device,Gibbs ,  average , sam, with_bias=with_bias, with_relu=with_relu)
 
        self.lr = lr
        self.adj_meta_grad = None
        self.features_meta_grad = None

        self.grad_sum = torch.zeros(nnodes, nnodes).to(device)

        self.weights = []
        self.biases = []
        previous_size = nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            bias = Parameter(torch.FloatTensor(nhid).to(device))
            previous_size = nhid

            self.weights.append(weight)
            self.biases.append(bias)

        output_weight = Parameter(torch.FloatTensor(previous_size, nclass).to(device))
        output_bias = Parameter(torch.FloatTensor(nclass).to(device))
        self.weights.append(output_weight)
        self.biases.append(output_bias)
        self._initialize()

    def _initialize(self):
        for w, b in zip(self.weights, self.biases):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)

        self.optimizer = optim.Adam(self.weights + self.biases, lr=self.lr) # , weight_decay=5e-4)

    def inner_train(self, features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training):
        adj_norm = utils.normalize_adj_tensor(modified_adj)
        for j in range(self.train_iters):
            hidden = features
            for w, b in zip(self.weights, self.biases):
                b = b if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                if self.with_relu:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])

            if self.lambda_ == 1:
                attack_loss = loss_labeled
            elif self.lambda_ == 0:
                attack_loss = loss_unlabeled
            else:
                attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

            self.optimizer.zero_grad()
            loss_labeled.backward(retain_graph=True)

            self.adj_changes.grad.zero_()
            self.grad_sum += torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
            self.optimizer.step()

        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        print(f'GCN loss on unlabled data: {loss_test_val.item()}')
        print(f'GCN acc on unlabled data: {utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()}')


    def forward(self, features, ori_adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=True, ll_cutoff=0.004):
        labels_self_training = self.train_surrogate(features, ori_adj, labels, idx_train, idx_unlabeled)
        self.sparse_features = sp.issparse(features)

        for i in tqdm(range(perturbations), desc="Perturbing graph"):

            adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
            ind = np.diag_indices(self.adj_changes.shape[0])
            adj_changes_symm = torch.clamp(adj_changes_square + torch.transpose(adj_changes_square, 1, 0), -1, 1)
            modified_adj = adj_changes_symm + ori_adj

            self._initialize()
            self.grad_sum.data.fill_(0)
            self.inner_train(features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training)

            adj_meta_grad = self.grad_sum * (-2 * modified_adj + 1)
            adj_meta_grad -= adj_meta_grad.min()
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_meta_grad = adj_meta_grad *  singleton_mask

            if ll_constraint:
                allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
                allowed_mask = allowed_mask.to(self.device)
                adj_meta_grad = adj_meta_grad * allowed_mask

            # Get argmax of the approximate meta gradients.
            adj_meta_approx_argmax = torch.argmax(adj_meta_grad)
            row_idx, col_idx = utils.unravel_index(adj_meta_approx_argmax, ori_adj.shape)

            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)

            if self.attack_features:
                pass

        return self.adj_changes + ori_adj


def visualize(your_var):
    from graphviz import Digraph
    import torch
    from torch.autograd import Variable
    from torchviz import make_dot
    make_dot(your_var).view()

def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    predictions = F.softmax(predictions, dim=1)
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    return H.mean()

def entropy_copy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    return H

class TsallisEntropy(nn.Module):
    
    def __init__(self, temperature: float, alpha: float):
        super(TsallisEntropy, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        N, C = logits.shape
        
        pred = F.softmax(logits / self.temperature, dim=1) 
        entropy_weight = entropy_copy(pred).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (N * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  
        
        sum_dim = torch.sum(pred * entropy_weight, dim = 0).unsqueeze(dim=0)
      
        return 1 / (self.alpha - 1) * torch.sum((1 / torch.mean(sum_dim) - torch.sum(pred ** self.alpha / sum_dim * entropy_weight, dim = -1)))