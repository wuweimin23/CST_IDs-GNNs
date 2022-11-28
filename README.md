# <div align="center">Cyclic Self-training For Adversarial Attack on Graph Neural Networks</div>

## Introduction 

Graph neural networks (GNNs) have been widely applied to various domains, such as biology and social networks. There have been many works examining the robustness of GNNs. For the adversarial attacks on GNNs in the semi-supervised node classification task, most existing works directly use the simplest generation methods for pseudo-labels, failing to consider more reliable pseudo-labels which may bring better performance. This paper proposes to view labeled nodes as the source domain and unlabeled nodes as the target domain. Then we introduce the domain adaptation method Cyclic Self-Training (CST) to the task. CST cycles between a source-trained classifier and a target-trained classifier, which boosts the reliability of pseudo labels. 

Intuitively, the labeled node labels contain both useful information that can transfer to the unlabeled nodes and harmful information that can make unlabeled node pseudo-labels incorrect. For the unlabeled node pseudo-labels, this is similar. In this sense, if we explicitly train the model to make unlabeled node pseudo-labels informative of the labeled nodes, we can gradually make the pseudo-labels more reliable. As for the concrete methods in our model, CST generates unlabeled node pseudo-labels with a source-trained classifier at first. Then CST trains a target-trained classifier using unlabeled node pseudo-labels, and updates the shared representations to make the target-trained classifier perform well on the labeled nodes. At last, we use the source-trained classifier to generate pseudo-labels for the unlabeled nodes.

To evaluate our method, we implement the CST model based on the adversarial attack method Graph Structure Poisoning via Meta-Learning.


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
