from distutils.command.clean import clean
from symbol import parameters
import torch
from gcn import GCN_C
from utils import *
import argparse
import numpy as np
from metattack import MetaApprox, Metattack
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='citeseer',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--ite_train', type=int, default=100,  help='training numbers')
parser.add_argument('--Gibbs', type = int, default=1,  help='whether using Gibbs')
parser.add_argument('--average', type = int, default=1,  help='whether average the parameters of f_l and f_u')
parser.add_argument('--sam',  type = int, default=1,  help='whether use sam')
parser.add_argument('--model', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self'], help='model variant')

args = parser.parse_args()
cuda = torch.cuda.is_available()
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

# === loading dataset
adj, features, labels = load_data(dataset=args.dataset)
nclass = max(labels) + 1

val_size = 0.1
test_size = 0.8
train_size = 1 - test_size - val_size

idx = np.arange(adj.shape[0])
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)
idx_unlabeled = np.union1d(idx_val, idx_test)
perturbations = int(args.ptb_rate * (adj.sum()//2))

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

# set up attack model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5


if 'A' in args.model:
    model = MetaApprox(nfeat=features.shape[1], hidden_sizes=[args.hidden],
                       nnodes=adj.shape[0], nclass=nclass, dropout=0.5,
                       train_iters=args.ite_train, attack_features=False, lambda_=lambda_, device=device, Gibbs = args.Gibbs, average = args.average, sam = args.sam)

else:
    model = Metattack(nfeat=features.shape[1], hidden_sizes=[args.hidden],
                       nnodes=adj.shape[0], nclass=nclass, dropout=0.5,
                       train_iters=args.ite_train, attack_features=False, lambda_=lambda_, device=device, Gibbs = args.Gibbs, average = args.average, sam = args.sam)

if device != 'cpu':
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    model = model.to(device)


def test(adj):
    ''' test on GCN '''

    adj = normalize_adj_tensor(adj)
    gcn = GCN_C(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=0.5)

    if device != 'cpu':
        gcn = gcn.to(device)

    optimizer = optim.Adam(gcn.parameters(),
                           lr=args.lr, weight_decay=5e-4)

    gcn.train()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output = gcn(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

    gcn.eval()
    output = gcn(features, adj)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
    device_1 = torch.device("cuda:4")
    device_2 = torch.device("cuda:5")
    modified_adj = model(features, adj, labels, idx_train,
                         idx_unlabeled, perturbations, device_1, device_2, ll_constraint=False)
    modified_adj = modified_adj.detach()

    runs = 10
    clean_acc = []
    attacked_acc = []
    print('=== testing GCN on original(clean) graph ===')
    for i in range(runs):
        clean_acc.append(test(adj))
    clean_acc_mean = np.mean(np.array(clean_acc))

    print('=== testing GCN on attacked graph ===')
    for i in range(runs):
        attacked_acc.append(test(modified_adj))
    attacked_acc_mean = np.mean(np.array(attacked_acc))

    print("results of clean and attact:{},{}".format(clean_acc_mean, attacked_acc_mean))

if __name__ == '__main__':
    main()

