# <div align="center">Cyclic Self-training For Adversarial Attack on Graph Neural Networks</div>

## Introduction 

Semi-supervised node classification: given a single (attributed) network and a subset of nodes whose class labels are known, the goal is to infer the classes of the unlabeled nodes. We can view the labeled nodes as source domain, and the unlabeled nodes as target domain. Then we apply the methods in domain adaptation to semi-supervised node classification.

Firstly, we try to get more accurate pseudo labels for unlabeled nodes. And then with the pseudo labels, our method can be applied into relative tasks, include poisoning attacks that is able to compromise the global node classification performance of a model, evasion attack that uses surrogate models to add permutation to the original graph and node features.

We have the following insights. Intuitively, the labels on the labeled nodes (source domain) contain both useful information that can transfer to the unlabeled nodes (target domain) and harmful information that can make pseudo-labels incorrect. Similarly, reliable pseudo-labels on the unlabeled nodes can transfer to the labeled nodes in turn, while models trained with incorrect pseudo-labels on the unlabeled nodes cannot transfer to the labeled nodes. In this sense, if we explicitly train the model to make pseudo-labels of unlabeled nodes informative of the labeled nodes, we can gradually make the pseudo-labels more reliable. 

## Algorithm
$\mathcal{V}_{L}$, $\mathcal{V}_{U}$ denote the labeled, unlabeled nodes respectively. $C_{L}$ is the true label for $\mathcal{V}_{L}$. Initially, we construct two same models, denoted as $f_{\theta_{L}, \alpha}$, $f_{\theta_{U}, \alpha}$, they are constructed for labeled nodes, unlabeled nodes classification respectively. And the parameters $\alpha$ of feature extractor in them are shared, while the parameters $\theta_{L}, \theta_{U}$ of classifiers (the last neural layer) are different. $\mathcal{L}_{\text {atk}}$ denotes the loss function the attacker aims to optimize. 

<div align="center">
  <img width="100%" alt="Algorithm of Cyclic Self-training for Adversarial Attack on GNNs" src="https://github.com/wuweimin23/CST-GNNs/blob/master/fig/1.png">
</div>

## Training

Use the command in code.sh, like

`python test_metattack.py --dataset citeseer --ite_train 100 --Gibbs 0 --average 0 --sam 0 --model Meta-Self`

## Results
Applying Cyclic Self-Training Model to base attack model named Graph Structure Poisoning via Meta-Learning, the attack
performance relatively increases 27%, 28%, and 6% on datasets CORA, CITESEER, and POLBLOGS respectively.

## Acknowledgement
Our implementation is based on the [ADVERSARIAL ATTACKS ON GRAPH NEURAL NETWORKS VIA META LEARNING](https://github.com/ChandlerBang/pytorch-gnn-meta-attack). 
