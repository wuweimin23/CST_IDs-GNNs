# <div align="center">Cycle Self-Training and IDs Updating for Adversarial Attack on GNNs ([Paper](https://github.com/wuweimin23/wuweimin23.github.io/blob/master/files/paper3.pdf))</div>

## Introduction 

Graph neural networks (GNNs) have been widely applied to various domains, such as biology and social networks. There have been many works examining the robustness of GNNs, which is important in the real application. One of the focuses is the poisoning adversarial attacks on GNNs in the semi-supervised node classification task. Several existing works involve the pseudo labels of unlabeled nodes. However, they generate the pseudo labels by training classifier on labeled nodes with cross entropy directly, failing to consider more reliable pseudo labels which may bring better performance of the attacker. This paper introduces the domain adaptation into the task, viewing the labeled nodes as the source domain and unlabeled nodes as the target domain. Then we apply the Cycle Self-Training (CST) method to boost the transferability of the model between two domains. Furthermore, we also propose confidence-based IDs updating method to select the node IDs with reliable pseudo labels, passing the reliable information of unlabeled nodes to the model to raise the generalization of the model across domains. To evaluate our methods, we implement our methods based on the method Adversarial Attack on GNNs Via Meta Learning on three datasets CITESEER, CORA-ML and POLBLOGS.

## Training

Use the command in code.sh, like

`python test_metattack.py --dataset citeseer --ite_train 100 --Gibbs 0 --average 0 --sam 0 --model Meta-Self`

## Results

The following table reports the results of the relative increased percentage for semi-supervised node classification task on three datasets: CITESEER, CORA-ML, POLBLOGS.

<div align="center">
<table>
        <tr>
            <th>Dataset</th>
            <th>CITESEER</th>
            <th>CORA-ML</th>
            <th>POLBLOGS</th>
        </tr>
        <tr>
            <td> Relative Increase (in %)</td>
            <td>2.1</td>
            <td>0.8</td>
            <td>0.7</td>
        </tr>
</table>
</div>

The following table reports the results of the relative increased percentage for our method based on Adversarial Attack on GNNs Via Meta Learning

<div align="center">
<table>
        <tr>
            <th>Dataset</th>
            <th>CITESEER</th>
            <th>CORA-ML</th>
            <th>POLBLOGS</th>
        </tr>
        <tr>
            <td> Relative Increase (in %)</td>
            <td>28.9</td>
            <td>15.2</td>
            <td>8.8</td>
        </tr>
</table>
</div>

## Acknowledgement
1. Our implementation is based on the [Adversarial Attacks on Graph Neural Networks Via Meta Learning](https://github.com/ChandlerBang/pytorch-gnn-meta-attack) and [Cycle Self-Training for Domain Adaptation](https://proceedings.neurips.cc/paper/2021/hash/c1fea270c48e8079d8ddf7d06d26ab52-Abstract.html).
2. In this paper, we just present part of our works temporarily. We will dig deeper and explore more robust methods in the future.
