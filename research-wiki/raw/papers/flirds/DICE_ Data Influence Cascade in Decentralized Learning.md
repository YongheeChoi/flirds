---
title: "DICE: Data Influence Cascade in Decentralized Learning"
source: "https://arxiv.org/html/2507.06931v1"
author:
published:
created: 2026-05-05
description:
tags:
  - "clippings"
---
propositiontheorem proposition

Tongtian Zhu, Wenhao Li & Can Wang  
Zhejiang University  
{raiden,wenhao-li,wcan}@zju.edu.cn  
&Fengxiang He  
University of Edinburgh  
F.He@ed.ac.uk

###### Abstract

Decentralized learning offers a promising approach to crowdsource data consumptions and computational workloads across geographically distributed compute interconnected through peer-to-peer networks, accommodating the exponentially increasing demands. However, proper incentives are still in absence, considerably discouraging participation. Our vision is that a fair incentive mechanism relies on fair attribution of contributions to participating nodes, which faces non-trivial challenges arising from the localized connections making influence “cascade” in a decentralized network. To overcome this, we design the first method to estimate Data Influence CascadE (DICE) in a decentralized environment. Theoretically, the framework derives tractable approximations of influence cascade over arbitrary neighbor hops, suggesting the influence cascade is determined by an interplay of data, communication topology, and the curvature of loss landscape. DICE also lays the foundations for applications including selecting suitable collaborators and identifying malicious behaviors. Project page is available at [DICE](https://raiden-zhu.github.io/blog/2025/DICE/).

## 1 Introduction

Large language models (LLMs) have seen remarkable progress in recent years [^36] [^84] [^23] [^19] [^2], surpassing human on key benchmarks [^76]. The compute scaling, highlighted by [^46] as a major reason of the successes, is estimated by Epoch AI to increase four to five times annually in cutting-edge models [^96]. This dramatic computational demand requires substantial financial investments; for example, training OpenAI’s GPT-4 requires approximately $\$$ 78 million in compute costs [^76]. Such exorbitant expenses are far beyond the affordability of most smaller players, making tech giants increasingly dominant.

Currently, large-scale training and inference processes are primarily performed in expensive data centers. Decentralized training, echoing swarm intelligence [^6] [^77], offers a cost-efficient alternative avenue by crowd-sourcing computational workload to decentralized compute nodes [^127] [^51]. One notable example showcasing decentralized computing’s computational potential is the Bitcoin eco-system which virtually distributes jobs requiring instantaneous 16 GW power consumption [^11].

Despite profound potential advantages, contributing to decentralized training incurs non-negligible costs for participants, raising a natural question: What motivates edge participants to engage in decentralized training?

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/influence_extended3_lower.png)

Figure 1: A semantic visualization of influence cascade in decentralized learning with ResNet-18 on CIFAR-10. The illustration depicts a 16-node communication topology (see Figure D.25 ), where node sizes represent DICE-E influence scores (see Theorem 1 ). Influence originates from the stem node and propagates through the network, weakening over distance, akin to “ripples in water”. This highlights how data contribution extends beyond local nodes, shaped by the communication topology.

Game theory suggests that when appropriate incentives exist, self-interested (rational) players can be keen to contribute for socially desirable outcomes. It is thus essential to design a proper incentive mechanism to unleash the collectively massive computational potential of decentralized nodes. Our vision is that such incentive mechanism relies on accurate quantification of contributions from players. This leads to the following problem:

<svg height="33.58" id="S1.p5.pic1" overflow="visible" version="1.1" width="600"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,33.58) matrix(1 0 0 -1 0 0)"><g fill="#BFBFBF" fill-opacity="1.0"><path d="M 0 6.7 L 0 26.88 C 0 30.58 3 33.58 6.7 33.58 L 593.3 33.58 C 597 33.58 600 30.58 600 26.88 L 600 6.7 C 600 3 597 0 593.3 0 L 6.7 0 C 3 0 0 3 0 6.7 Z" style="stroke:none"></path></g><g fill="#FFFAFA" fill-opacity="1.0"><path d="M 2.77 6.7 L 2.77 26.88 C 2.77 29.05 4.53 30.81 6.7 30.81 L 593.3 30.81 C 595.47 30.81 597.23 29.05 597.23 26.88 L 597.23 6.7 C 597.23 4.53 595.47 2.77 593.3 2.77 L 6.7 2.77 C 4.53 2.77 2.77 4.53 2.77 6.7 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 8.67 10.64)"><foreignObject color="#000000" height="12.3" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="581.87"><span id="S1.p5.pic1.1.1.1.1.1" style="width:420.5pt;"><span id="S1.p5.pic1.1.1.1.1.1.1"><span id="S1.p5.pic1.1.1.1.1.1.1.1">How to quantify individual contributions in decentralized learning?</span></span></span></foreignObject></g></g></svg>

Quantifying the contributions (or “influence”) to the learned model has been well studied in the centralized paradigm [^55] [^88]. However, it is still largely untouched in understanding and measuring data influence in fully decentralized environments. Unlike centralized scenarios, where data influence is computed on a single model statically, decentralized learning relies on localized, indirect communications on a perhaps sparse network – the influence of data on one node impacts its own model, propagates to its neighbors through iterative parameter exchanges, and cascades to multi-hop neighbors. We term this mechanism as cascading influence (see Figure 1). Existing data influence estimators tailored for centralized settings assume that influence is computed within a single model and do not account for its recursive propagation through parameter exchange. As a result, they are not applicable to decentralized learning, where data influence extends beyond direct neighbors to multi-hop neighbors.

To address this challenge, we propose a Data Influence CascadE (DICE), the first work for measuring the influence in a decentralized learning environment. We make contributions as follows:

- Conceptual contributions: DICE introduces the concept of a ground-truth data influence for decentralized learning, integrating direct and indirect contributions to capture influence propagation across multiple hops during training (see Definition 3).
- Theoretical contributions: Building on this foundation, we derive tractable approximations of ground-truth DICE for an arbitrary number of neighbor hops, establishing a foundational framework to systematically characterize the flow of influence across decentralized networks. These theoretical results uncover, for the first time, that data influence in decentralized learning extends beyond the data itself and the local model, as seen in centralized training. Instead, it is a joint product of three critical factors: the original data, the topological importance of the data keeper, and the curvature information of intermediate nodes mediating propagation (see Theorem 1).

We anticipate that our DICE framework will pave the way for novel incentive mechanism designs and the establishment of economic opportunities for decentralized learning, such as data and parameter markets. DICE also holds significant potential to address critical challenges of identifying new suitable collaborators, and detecting free-riders. We envision these applications will contribute to a scalable, autonomous, and reciprocal decentralized learning eco-system.

## 2 Related work

Data Influence Estimation. As high-quality data becomes increasingly critical in modern machine learning [^47] [^87] [^66] [^109], understanding its influence has emerged as a key research direction [^100] [^34]. Data influence estimation quantifies the contribution of training data to model predictions [^14] [^50], supporting incentive mechanisms and applications in few-shot learning [^86], dataset pruning [^121], distillation [^70], fairness [^65], machine unlearning [^95], explainability [^56] [^34], and security [^20] [^39].

Existing methods fall into static and dynamic categories. Static approaches, including retraining-based methods (e.g., leave-one-out [^17], Shapley value [^97], Datamodels [^49]) and one-point methods (e.g., influence functions [^56]), estimate influence post-training. While theoretically grounded, these methods cannot characterize dynamic influence during training. Dynamic approaches address this limitation by tracking model parameter evolution [^12]. Notable methods include TracIn [^88] and In-Run Data Shapley [^113], which average gradient similarities over time. Recent advances [^83] leverage memory-perturbation equations to extend dynamic influence estimation to various optimization algorithms. For a more detailed background, please refer to Subsection A.1.

However, existing methods primarily focus on centralized training. To the best of our knowledge, the most closely related work is by [^106], who propose a decentralized hyper-gradient method and offer novel insights into using hyper-gradients to compute data influence. Nevertheless, their estimation method is static and cannot capture the influence cascade in decentralized training. In contrast, our framework, DICE, is specifically designed for fully decentralized environments, providing a fine-grained characterization of influence propagation unique to these settings.

Incentivized Decentralized Learning. Most existing incentive mechanisms for collaborative learning are designed for federated learning [^133]. For instance, [^116] propose an Incentive Collaboration Learning (ICL) framework to promote collaboration. Their focus is on mechanism design rather than the precise quantification of individual contributions. In federated learning, the Shapley value has been effectively utilized to quantify participant contributions [^53] [^31] [^112] [^115]. Our approach differs fundamentally in two key aspects: first, we focus on fully decentralized settings without central servers, although our framework supports federated learning scenarios (see Algorithm 1); second, our work considers influence cascade between participants, an completely new perspective that has not been explored in existing literature. Regrading decentralized learning, we are only aware of the work by [^126] presenting a blockchain-based incentive mechanism for fully decentralized learning. However, their mechanism relies on smart contracts and differs from ours.

## 3 Notations and Preliminaries

This section introduces notations and essential preliminaries for decentralized learning. This work focuses on the most studied form of decentralized learning: data parallelism with only peer-level communication. For more detailed background, please refer to Subsection A.2.

We consider a general personalized distributed optimization problem over a connected graph $G=(\mathcal{V},\mathcal{E})$, where $\mathcal{V}$ represents the set of participants and $\mathcal{E}$ denotes the communication links between them. The participants collaboratively minimize a weighted sum of local personalized objectives [^103] [^43]:

$$
\displaystyle\min_{\bm{\theta}=\{\bm{\theta}_{k}\in\mathbb{R}^{d}\}_{k\in%
\mathcal{V}}}\left[L(\bm{\theta})\triangleq\sum_{k\in\mathcal{V}}q_{k}L_{k}(%
\bm{\theta}_{k})\right],
$$

where $q_{k}\geq 0$ with $\sum_{k\in\mathcal{V}}q_{k}=1$, and each local objective $L_{k}(\bm{\theta}_{k})=\mathbb{E}_{\bm{z}_{k}\sim\mathcal{D}_{k}}\left[L(\bm{%
\theta}_{k};\bm{z}_{k})\right]$ is defined by the expectation over the local data distribution $\mathcal{D}_{k}$. Empirical risk minimization involves optimizing the sample average approximation:

$$
\displaystyle\hat{L}(\bm{\theta})=\sum_{k\in\mathcal{V}}q_{k}\hat{L}_{k}(\bm{%
\theta}_{k})\quad\text{where}\quad\hat{L}_{k}(\bm{\theta}_{k})=\frac{1}{n_{k}}%
\sum_{i=1}^{n_{k}}L(\bm{\theta}_{k};\bm{z}_{k_{i}}).
$$

Here, $n_{k}$ is the number of samples in participant $k$, and $\{\bm{z}_{k_{i}}\}_{i=1}^{n_{k}}$ are drawn from $\mathcal{D}_{k}$.

![Refer to caption](https://arxiv.org/html/2507.06931v1/x1.png)

Figure 2: A comparative illustration of server-based learning versus decentralized learning.

Decentralized learning aims to minimize the global objective $l(\bm{\theta})=\sum_{k\in\mathcal{V}}q_{k}l_{k}(\bm{\theta}_{k})$ with only local computations and gossip communications among neighboring participants [^107] [^82]. The communication protocol is governed by a weighted adjacency matrix $\bm{W}\in[0,1]^{n\times n}$, where $\bm{W}_{k,j}\geq 0$ represents the strength of connection from participant $j$ to participant $k$, with $\bm{W}_{k,j}>0$ if $(k,j)\in\mathcal{E}$. This matrix characterizes the communication topology, thereby defining how information propagates through the network (see Figure A.1). In this paper, $\bm{W}$ is designed to be row-stochastic, satisfying $\sum_{j=1}^{n}\bm{W}_{k,j}=1$ for all $i\in\mathcal{V}$ <sup>1</sup>. Decentralized learning alternates between local optimization and gossip-based parameter aggregation, as shown below:

Algorithm 1 Decentralized Learning with Flexible Gossip and Optimization

$G=(\mathcal{V},\mathcal{E})$, $\{\bm{\theta}_{k}^{0}\}_{k\in\mathcal{V}}$, optimizer $\mathcal{O}_{k}$, number of communication rounds $T$, and mixing matrix distributions $\mathcal{W}^{t}\ (\forall t\in[T])$

for $t=1$ to $T$ do in parallel for all participants $k\in\mathcal{V}$

Local Update:

Sample $\bm{z}_{k}^{t}\sim\mathcal{D}_{k}$, update parameters with optimizer $\mathcal{O}_{k}$: $\bm{\theta}_{k}^{t+\frac{1}{2}}\leftarrow\mathcal{O}_{k}(\bm{\theta}_{k}^{t},%
\bm{z}_{k}^{t})$

Gossip Averaging:

Send $\bm{\theta}_{k}^{t+\frac{1}{2}}$ to $\{l\mid\bm{W}_{l,k}>0\}$ and receive $\bm{\theta}_{j}^{t+\frac{1}{2}}$ from $\{j\mid\bm{W}_{k,j}>0\}$.

Sample $\bm{W}^{t}\sim\mathcal{W}^{t}$, perform gossip averaging: $\bm{\theta}_{k}^{t+1}\leftarrow\sum_{j\in\mathcal{N}_{\text{in}}(k)}\bm{W}_{k,%
j}^{t}\bm{\theta}_{j}^{t+\frac{1}{2}}$ End for

###### Remark 1.

Algorithm 1 provides a flexible framework for decentralized learning with arbitrary optimizers and randomized gossip. A special case of this framework is decentralized stochastic gradient descent (DSGD) [^128] [^67] [^57], where the optimizer $\mathcal{O}_{k}$ performs a simple stochastic gradient step: $\bm{\theta}_{k}^{t+\frac{1}{2}}\leftarrow\bm{\theta}_{k}^{t}-\eta^{t}\nabla L(%
\bm{\theta}_{k}^{t};\bm{z}_{k}^{t}),$ with $\eta^{t}$ being the learning rate. Another notable special case is FedAVG [^78], which corresponds to the standard server-based learning setting, where a central server collects and averages model updates from all participants in each round. Mathematically, this is equivalent to using a fully connected and uniform mixing matrix in Algorithm 1, i.e., $\bm{W}_{k,j}^{t}\equiv\frac{1}{n}\mathbf{1}\mathbf{1}^{T}$ (see Figure 2 for a visual comparison and its connection to decentralized learning). Therefore, the framework is applicable to both federated and decentralized learning paradigms, although our primary focus remains on fully decentralized learning without a central server.

## 4 Data Influence Cascades

In this section, we introduce DICE, a comprehensive framework for measuring data influence in decentralized environments. Subsection 4.1 introduces the ground-truth influence measures designed for decentralized learning and Subsection 4.2 provides their dynamic gradient-based estimations.

### 4.1 Ground-truth Influence in Decentralized Learning

To ensure a logical and coherent flow, we first introduce the fundamental concepts of data influence in centralized settings and then discuss the significant challenges involved in extending these ideas to decentralized environments. In conventional centralized setups, the influence of an individual data instance can be assessed by evaluating the counterfactual change in learning performance through leave-one-out retraining (LOO) [^17], defined as follows:

###### Definition 1 (Leave-one-out Influence).

$$
\displaystyle\mathcal{I}_{\text{LOO}}(\bm{z},\bm{z}^{\prime})=
$$
 
$$
\displaystyle L(\bm{\theta^{*}};\bm{z}^{\prime})-L(\bm{\theta^{*}}_{\setminus z%
};\bm{z}^{\prime}),
$$

where $\bm{z}$ denotes the training data instance under influence assessment, $\bm{z}^{\prime}$ is the loss-evaluating instance, $\bm{\theta^{*}}$ and $\bm{\theta^{*}}_{\setminus\bm{z}}$ are the models trained on the entire dataset $\mathcal{S}$ and $\mathcal{S}\setminus\{z\}$, respectively.

Intuitively, Eq. 3 quantifies the influence of $\bm{z}$ by its individual impact on test loss reduction. A smaller LOO value indicates a significant contribution to learning, which aligns with the concept that the data influence is reflected in its ability to enhance model performance. LOO influence is often considered as the “gold standard” for evaluating how well influence estimators approximate the ground-truth influence in the data influence literature [^56] [^4].

However, extending LOO to decentralized scenarios introduces non-trivial challenges due to the distributed nature and the localized connections in decentralized learning, reflected in Eq. 2 and Algorithm 1. In centralized setups, the core idea of LOO is to link data influence to variations in loss or parameter outcomes. In contrast, decentralized learning systems involve multiple participants sharing model parameters through inter-participant communications. As a result, alterations in model parameters caused by a data-level modification propagate throughout the whole network.

A natural way to measure the influence of one participant in such collaborative environments is through evaluating its contribution to the whole community [^115] [^125], which aligns with the customer-centric principle [^22] in determining value <sup>2</sup>. In decentralized learning, when a participant transmits its training assets (e.g., model parameters or gradients) to neighboring participants—akin to offering a product—the recipients derive possible utility from these training assets and may provide reciprocal feedback, such as sharing their own assets in return. This dynamic positions the neighbors as “customers”, thereby entrusting them with the rights to determine the value of the assets provided by the supplier. With these insights in mind, we recognize that assessing data influence in decentralized scenarios is far more complex, as summarized below:

<svg height="68.33" id="S4.SS1.p5.pic1" overflow="visible" version="1.1" width="600"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,68.33) matrix(1 0 0 -1 0 0)"><g fill="#BFBFBF" fill-opacity="1.0"><path d="M 0 6.7 L 0 61.62 C 0 65.33 3 68.33 6.7 68.33 L 593.3 68.33 C 597 68.33 600 65.33 600 61.62 L 600 6.7 C 600 3 597 0 593.3 0 L 6.7 0 C 3 0 0 3 0 6.7 Z" style="stroke:none"></path></g><g fill="#F9FEFE" fill-opacity="1.0"><path d="M 2.77 6.7 L 2.77 61.62 C 2.77 63.8 4.53 65.56 6.7 65.56 L 593.3 65.56 C 595.47 65.56 597.23 63.8 597.23 61.62 L 597.23 6.7 C 597.23 4.53 595.47 2.77 593.3 2.77 L 6.7 2.77 C 4.53 2.77 2.77 4.53 2.77 6.7 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 8.67 10.64)"><foreignObject color="#000000" height="47.05" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="581.87"><span id="S4.SS1.p5.pic1.1.1.1.1.1" style="width:420.5pt;"><span id="S4.SS1.p5.pic1.1.1.1.1.1.1"><span id="S4.SS1.p5.pic1.1.1.1.1.1.1.1">Key observations</span>: <span id="S4.SS1.p5.pic1.1.1.1.1.1.1.2">In decentralized learning,<br>1) neighbors who serves as customers hold the rights to determine data influence;<br>2) data influence is not static but spreads across participants through gossips during training.</span></span></span></foreignObject></g></g></svg>

Unfortunately, existing static estimators only calculate the loss change after training and thus cannot characterize the dynamic transmission of data influence within the whole decentralized learning community. Based on the above discussion, we posit that a “gold-standard” influence measure in decentralized scenarios should satisfy the following requirements:

- Quantify community-level influence: Measure the impact of training data instances on the collective utility of the community.
- Depend on training dynamics: Measure the influence based on the training process to characterize the propagation of influence on decentralized networks.

In the following, we introduce the ground-truth influence measures tailored to the requirements of decentralized environments, termed as the ground-truth influence cascade (DICE-GT).

###### Definition 2 (One-hop Ground-truth Influence).

The one-hop DICE-GT value quantifies the influence of a data instance $\bm{z}_{j}^{t}$ from participant $j$ on a loss-evaluating instance $\bm{z}^{\prime}$ within itself and its immediate neighbors. Formally, for a given participant $j\in\mathcal{V}$:

$$
\displaystyle\mathcal{I}_{\text{DICE-GT}}^{(1)}(\bm{z}_{j}^{t},\bm{z}^{\prime}%
)=\underbrace{q_{j}\left(L(\bm{\theta}^{t+\frac{1}{2}}_{j};\bm{z}^{\prime})-L(%
\bm{\theta}^{t}_{j};\bm{z}^{\prime})\right)}_{\text{direct marginal %
contribution of $\bm{z}_{j}^{t}$ to }j}+\underbrace{\sum_{k\in\mathcal{N}_{%
\text{out}}^{(1)}(j)}q_{k}\left(L(\bm{\theta}^{t+1}_{k};\bm{z}^{\prime})-L(\bm%
{\theta}^{t+1}_{k\setminus\bm{z}_{j}^{t}};\bm{z}^{\prime})\right)}_{\text{%
indirect marginal contribution of $\bm{z}_{j}^{t}$ to one-hop neighbors}},
$$

where $\bm{\theta}^{t+\frac{1}{2}}_{j}$ denotes the updated model parameters of $j$ after training on $\bm{z}_{j}^{t}$ at iteration $t$ (see Algorithm 1). For each one-hop out-neighbor $k\in\mathcal{N}_{\text{out}}^{(1)}(j)$, $\bm{\theta}^{t+1}_{k}$ denotes the averaged model parameters after receiving updated parameters $\{\bm{\theta}^{t+\frac{1}{2}}_{l}|\bm{W}_{k,l}>0\}$ influenced by $\bm{z}_{j}^{t}$, while $\bm{\theta}^{t+1}_{k\setminus\bm{z}_{j}^{t}}$ represents the model parameters of $k$ without the influence from $\bm{z}_{j}^{t}$, i.e., $\bm{\theta}^{t+1}_{k\setminus\bm{z}_{j}^{t}}=\sum_{l\in\mathcal{N}_{\text{out}%
}(k)\setminus j}\bm{W}_{k,l}^{t}\bm{\theta}_{l}^{t+\frac{1}{2}}+\bm{W}_{k,j}^{%
t}\bm{\theta}_{l}^{t}$.

The economic intuition behind the DICE-GT value is that it captures both the direct marginal contribution of a data instance to itself and its subsequent impact on immediate neighbors. Specially, the first term $L(\bm{z}^{\prime};\bm{\theta}^{t+\frac{1}{2}}_{j})-L(\bm{z}^{\prime};\bm{%
\theta}^{t}_{j})$ captures the inter-node direct influence of training data instance $\bm{z}_{j}^{t}$ on the test loss change at node $j$, which corresponds to the TracInIdeal influence in [^88] designed for centralized scenarios. The second term $\sum_{k\in\mathcal{N}_{\text{out}}^{(1)}(j)}(L(\bm{z}^{\prime};\bm{\theta}^{t+%
1}_{k})-L(\bm{z}^{\prime};\bm{\theta}^{t}_{k\setminus\bm{z}_{j}^{t}}))$ measures the intra-node influence unique in decentralized learning, which aggregates the indirect influences on all one-hop neighbors, i.e., direct neighbors, of node $j$.

In decentralized learning environments, data influence propagates not only to immediate neighbors but also to multi-hop neighbors through the communication topology. To characterize this multi-hop influence, we extend the ground-truth influence cascade measure to arbitrary $r$ -hop neighbors.

###### Definition 3 (Multi-hop Ground-truth Influence).

The multi-hop DICE-GT value quantifies the cumulative influence of a data instance $\bm{z}$ on a loss-evaluating instance $\bm{z}^{\prime}$ across all nodes within $r$ -hop neighborhoods of participant $j$. Formally, for a given participant $j\in\mathcal{V}$:

$$
\displaystyle\mathcal{I}_{\text{DICE-GT}}^{(r)}(\bm{z}_{j}^{t},\bm{z}^{\prime}%
)=q_{j}\left(L(\bm{\theta}^{t+\frac{1}{2}}_{j};\bm{z}^{\prime})-L(\bm{\theta}^%
{t}_{j};\bm{z}^{\prime})\right)+\sum_{s=1}^{r}\sum_{k\in\mathcal{N}_{\text{out%
}}^{(s)}(j)}q_{k}\left(L(\bm{\theta}^{t+s}_{k};\bm{z}^{\prime})-L(\bm{\theta}^%
{t+s}_{k\setminus\bm{z}_{j}^{t}};\bm{z}^{\prime})\right),
$$

where $\mathcal{N}_{\text{out}}^{(s)}(j)$ denotes the set of $s$ -hop out-neighbors of $j$ (please refer to Subsection A.3 for details of high-order neighbors). Here $\bm{\theta}^{t+s}_{k}$ and $\bm{\theta}^{t+s}_{k\setminus\bm{z}_{j}^{t}}$ represents the parameters of node $k$ at iteration $t+s$ when the influence from $\bm{z}_{j}^{t}$ are included and excluded, respectively.

Analogous to Definition 2, the first term captures the direct influence of data $\bm{z}_{j}^{t}$ on the loss at node $j$. The subsequent summation aggregates the indirect influences on all multi-hop neighbors up to $r$ steps away from node $j$ <sup>3</sup>. The reason to measure test loss change at the $t+s$ step is that the impact of $\bm{z}_{j}^{t}$ propagating to $k\in\mathcal{N}_{\text{out}}^{(s)}(j)$ requires $s$ steps. This layered formulation accounts for the multi-hop cascading effects through the network up to the specified order $r$.

### 4.2 Dynamic Gradient-based Estimations

To meet the second aforementioned requirement of decentralized learning, we design dynamic gradient-based estimators for DICE-GT, called the influence cascade estimations (DICE-E).

<svg height="86.3" id="S4.SS2.p2.pic1" overflow="visible" version="1.1" width="600"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,86.3) matrix(1 0 0 -1 0 0)"><g fill="#BFBFBF" fill-opacity="1.0"><path d="M 0 6.7 L 0 79.6 C 0 83.3 3 86.3 6.7 86.3 L 593.3 86.3 C 597 86.3 600 83.3 600 79.6 L 600 6.7 C 600 3 597 0 593.3 0 L 6.7 0 C 3 0 0 3 0 6.7 Z" style="stroke:none"></path></g><g fill="#F9FEFE" fill-opacity="1.0"><path d="M 2.77 6.7 L 2.77 79.6 C 2.77 81.77 4.53 83.54 6.7 83.54 L 593.3 83.54 C 595.47 83.54 597.23 81.77 597.23 79.6 L 597.23 6.7 C 597.23 4.53 595.47 2.77 593.3 2.77 L 6.7 2.77 C 4.53 2.77 2.77 4.53 2.77 6.7 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 8.67 10.64)"><foreignObject color="#000000" height="65.02" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="581.87"><span id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1" style="width:420.5pt;"><span id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1"><h6>Proposition (Approximation of One-hop DICE-GT).</h6><span id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1"><span id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.2">The one-hop DICE-GT value (see Definition&nbsp;2) can be linearly approximated as follow:</span> <span id="A4.EGx6"><span id="S4.Ex3"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\displaystyle\mathcal{I}_{\text{DICE-E}}^{(1)}(\bm{z}_{j}^{t},\bm{z}^{\prime})="><semantics id="S4.Ex3.m1.3a"><mrow id="S4.Ex3.m1.3.3" xref="S4.Ex3.m1.3.3.cmml"><mrow id="S4.Ex3.m1.3.3.2" xref="S4.Ex3.m1.3.3.2.cmml"><msubsup id="S4.Ex3.m1.3.3.2.4" xref="S4.Ex3.m1.3.3.2.4.cmml"><mi id="S4.Ex3.m1.3.3.2.4.2.2" xref="S4.Ex3.m1.3.3.2.4.2.2.cmml">ℐ</mi> <mtext id="S4.Ex3.m1.3.3.2.4.2.3" xref="S4.Ex3.m1.3.3.2.4.2.3a.cmml">DICE-E</mtext> <mrow id="S4.Ex3.m1.1.1.1.3" xref="S4.Ex3.m1.3.3.2.4.cmml"><mo id="S4.Ex3.m1.1.1.1.3.1" stretchy="false" xref="S4.Ex3.m1.3.3.2.4.cmml">(</mo><mn id="S4.Ex3.m1.1.1.1.1" xref="S4.Ex3.m1.1.1.1.1.cmml">1</mn><mo id="S4.Ex3.m1.1.1.1.3.2" stretchy="false" xref="S4.Ex3.m1.3.3.2.4.cmml">)</mo></mrow></msubsup> <mo id="S4.Ex3.m1.3.3.2.3" xref="S4.Ex3.m1.3.3.2.3.cmml">⁢</mo> <mrow id="S4.Ex3.m1.3.3.2.2.2" xref="S4.Ex3.m1.3.3.2.2.3.cmml"><mo id="S4.Ex3.m1.3.3.2.2.2.3" stretchy="false" xref="S4.Ex3.m1.3.3.2.2.3.cmml">(</mo><msubsup id="S4.Ex3.m1.2.2.1.1.1.1" xref="S4.Ex3.m1.2.2.1.1.1.1.cmml"><mi id="S4.Ex3.m1.2.2.1.1.1.1.2.2" xref="S4.Ex3.m1.2.2.1.1.1.1.2.2.cmml">𝒛</mi> <mi id="S4.Ex3.m1.2.2.1.1.1.1.2.3" xref="S4.Ex3.m1.2.2.1.1.1.1.2.3.cmml">j</mi> <mi id="S4.Ex3.m1.2.2.1.1.1.1.3" xref="S4.Ex3.m1.2.2.1.1.1.1.3.cmml">t</mi></msubsup><mo id="S4.Ex3.m1.3.3.2.2.2.4" xref="S4.Ex3.m1.3.3.2.2.3.cmml">,</mo><msup id="S4.Ex3.m1.3.3.2.2.2.2" xref="S4.Ex3.m1.3.3.2.2.2.2.cmml"><mi id="S4.Ex3.m1.3.3.2.2.2.2.2" xref="S4.Ex3.m1.3.3.2.2.2.2.2.cmml">𝒛</mi> <mo id="S4.Ex3.m1.3.3.2.2.2.2.3" xref="S4.Ex3.m1.3.3.2.2.2.2.3.cmml">′</mo></msup><mo id="S4.Ex3.m1.3.3.2.2.2.5" stretchy="false" xref="S4.Ex3.m1.3.3.2.2.3.cmml">)</mo></mrow></mrow> <mo id="S4.Ex3.m1.3.3.3" xref="S4.Ex3.m1.3.3.3.cmml">=</mo></mrow> <annotation-xml encoding="MathML-Content" id="S4.Ex3.m1.3b"><apply id="S4.Ex3.m1.3.3.cmml" xref="S4.Ex3.m1.3.3"><apply id="S4.Ex3.m1.3.3.2.cmml" xref="S4.Ex3.m1.3.3.2"><apply id="S4.Ex3.m1.3.3.2.4.cmml" xref="S4.Ex3.m1.3.3.2.4"><csymbol cd="ambiguous" id="S4.Ex3.m1.3.3.2.4.1.cmml" xref="S4.Ex3.m1.3.3.2.4">superscript</csymbol> <apply id="S4.Ex3.m1.3.3.2.4.2.cmml" xref="S4.Ex3.m1.3.3.2.4"><csymbol cd="ambiguous" id="S4.Ex3.m1.3.3.2.4.2.1.cmml" xref="S4.Ex3.m1.3.3.2.4">subscript</csymbol> <ci id="S4.Ex3.m1.3.3.2.4.2.2.cmml" xref="S4.Ex3.m1.3.3.2.4.2.2">ℐ</ci> <ci id="S4.Ex3.m1.3.3.2.4.2.3a.cmml" xref="S4.Ex3.m1.3.3.2.4.2.3"><mtext id="S4.Ex3.m1.3.3.2.4.2.3.cmml" mathsize="70%" xref="S4.Ex3.m1.3.3.2.4.2.3">DICE-E</mtext></ci></apply> <cn id="S4.Ex3.m1.1.1.1.1.cmml" type="integer" xref="S4.Ex3.m1.1.1.1.1">1</cn></apply> <interval closure="open" id="S4.Ex3.m1.3.3.2.2.3.cmml" xref="S4.Ex3.m1.3.3.2.2.2"><apply id="S4.Ex3.m1.2.2.1.1.1.1.cmml" xref="S4.Ex3.m1.2.2.1.1.1.1"><csymbol cd="ambiguous" id="S4.Ex3.m1.2.2.1.1.1.1.1.cmml" xref="S4.Ex3.m1.2.2.1.1.1.1">superscript</csymbol> <apply id="S4.Ex3.m1.2.2.1.1.1.1.2.cmml" xref="S4.Ex3.m1.2.2.1.1.1.1"><csymbol cd="ambiguous" id="S4.Ex3.m1.2.2.1.1.1.1.2.1.cmml" xref="S4.Ex3.m1.2.2.1.1.1.1">subscript</csymbol> <ci id="S4.Ex3.m1.2.2.1.1.1.1.2.2.cmml" xref="S4.Ex3.m1.2.2.1.1.1.1.2.2">𝒛</ci> <ci id="S4.Ex3.m1.2.2.1.1.1.1.2.3.cmml" xref="S4.Ex3.m1.2.2.1.1.1.1.2.3">𝑗</ci></apply> <ci id="S4.Ex3.m1.2.2.1.1.1.1.3.cmml" xref="S4.Ex3.m1.2.2.1.1.1.1.3">𝑡</ci></apply> <apply id="S4.Ex3.m1.3.3.2.2.2.2.cmml" xref="S4.Ex3.m1.3.3.2.2.2.2"><csymbol cd="ambiguous" id="S4.Ex3.m1.3.3.2.2.2.2.1.cmml" xref="S4.Ex3.m1.3.3.2.2.2.2">superscript</csymbol> <ci id="S4.Ex3.m1.3.3.2.2.2.2.2.cmml" xref="S4.Ex3.m1.3.3.2.2.2.2.2">𝒛</ci> <ci id="S4.Ex3.m1.3.3.2.2.2.2.3.cmml" xref="S4.Ex3.m1.3.3.2.2.2.2.3">′</ci></apply></interval></apply> <csymbol cd="latexml" id="S4.Ex3.m1.3.3.4.cmml" xref="S4.Ex3.m1.3.3.4">absent</csymbol></apply></annotation-xml> <annotation encoding="application/x-tex" id="S4.Ex3.m1.3c">\displaystyle\mathcal{I}_{\text{DICE-E}}^{(1)}(\bm{z}_{j}^{t},\bm{z}^{\prime})=</annotation> <annotation encoding="application/x-llamapun" id="S4.Ex3.m1.3d">caligraphic_I start_POSTSUBSCRIPT DICE-E end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( 1 ) end_POSTSUPERSCRIPT ( bold_italic_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, bold_italic_z start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) =</annotation></semantics></math> <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\displaystyle\;-q_{j}\,\nabla L(\bm{\theta}_{j}^{t};\bm{z}^{\prime})^{\top}%
\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})-\sum_{k\in\mathcal{N}_{\text{%
out}}^{(1)}(j)}q_{k}\,\bm{W}_{k,j}^{t}\,\nabla L(\bm{\theta}_{k}^{t+1};\bm{z}^%
{\prime})^{\top}\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t}),"><semantics id="S4.Ex3.m2.5a"><mrow id="S4.Ex3.m2.5.5.1" xref="S4.Ex3.m2.5.5.1.1.cmml"><mrow id="S4.Ex3.m2.5.5.1.1" xref="S4.Ex3.m2.5.5.1.1.cmml"><mrow id="S4.Ex3.m2.5.5.1.1.4" xref="S4.Ex3.m2.5.5.1.1.4.cmml"><mo id="S4.Ex3.m2.5.5.1.1.4a" xref="S4.Ex3.m2.5.5.1.1.4.cmml">−</mo> <mrow id="S4.Ex3.m2.5.5.1.1.4.4" xref="S4.Ex3.m2.5.5.1.1.4.4.cmml"><msub id="S4.Ex3.m2.5.5.1.1.4.4.6" xref="S4.Ex3.m2.5.5.1.1.4.4.6.cmml"><mi id="S4.Ex3.m2.5.5.1.1.4.4.6.2" xref="S4.Ex3.m2.5.5.1.1.4.4.6.2.cmml">q</mi> <mi id="S4.Ex3.m2.5.5.1.1.4.4.6.3" xref="S4.Ex3.m2.5.5.1.1.4.4.6.3.cmml">j</mi></msub> <mo id="S4.Ex3.m2.5.5.1.1.4.4.5" lspace="0.167em" xref="S4.Ex3.m2.5.5.1.1.4.4.5.cmml">⁢</mo> <mrow id="S4.Ex3.m2.5.5.1.1.4.4.7" xref="S4.Ex3.m2.5.5.1.1.4.4.7.cmml"><mo id="S4.Ex3.m2.5.5.1.1.4.4.7.1" rspace="0.167em" xref="S4.Ex3.m2.5.5.1.1.4.4.7.1.cmml">∇</mo> <mi id="S4.Ex3.m2.5.5.1.1.4.4.7.2" xref="S4.Ex3.m2.5.5.1.1.4.4.7.2.cmml">L</mi></mrow> <mo id="S4.Ex3.m2.5.5.1.1.4.4.5a" xref="S4.Ex3.m2.5.5.1.1.4.4.5.cmml">⁢</mo> <msup id="S4.Ex3.m2.5.5.1.1.2.2.2" xref="S4.Ex3.m2.5.5.1.1.2.2.2.cmml"><mrow id="S4.Ex3.m2.5.5.1.1.2.2.2.2.2" xref="S4.Ex3.m2.5.5.1.1.2.2.2.2.3.cmml"><mo id="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.3" stretchy="false" xref="S4.Ex3.m2.5.5.1.1.2.2.2.2.3.cmml">(</mo><msubsup id="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1" xref="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.cmml"><mi id="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.2.2" xref="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.2.2.cmml">𝜽</mi> <mi id="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.2.3" xref="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.2.3.cmml">j</mi> <mi id="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.3" xref="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.3.cmml">t</mi></msubsup><mo id="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.4" xref="S4.Ex3.m2.5.5.1.1.2.2.2.2.3.cmml">;</mo><msup id="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.2" xref="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.2.cmml"><mi id="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.2.2" xref="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.2.2.cmml">𝒛</mi> <mo id="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.2.3" xref="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.2.3.cmml">′</mo></msup><mo id="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.5" stretchy="false" xref="S4.Ex3.m2.5.5.1.1.2.2.2.2.3.cmml">)</mo></mrow> <mo id="S4.Ex3.m2.5.5.1.1.2.2.2.4" xref="S4.Ex3.m2.5.5.1.1.2.2.2.4.cmml">⊤</mo></msup> <mo id="S4.Ex3.m2.5.5.1.1.4.4.5b" xref="S4.Ex3.m2.5.5.1.1.4.4.5.cmml">⁢</mo> <msub id="S4.Ex3.m2.5.5.1.1.4.4.8" xref="S4.Ex3.m2.5.5.1.1.4.4.8.cmml"><mi id="S4.Ex3.m2.5.5.1.1.4.4.8.2" mathvariant="normal" xref="S4.Ex3.m2.5.5.1.1.4.4.8.2.cmml">Δ</mi> <mi id="S4.Ex3.m2.5.5.1.1.4.4.8.3" xref="S4.Ex3.m2.5.5.1.1.4.4.8.3.cmml">j</mi></msub> <mo id="S4.Ex3.m2.5.5.1.1.4.4.5c" xref="S4.Ex3.m2.5.5.1.1.4.4.5.cmml">⁢</mo> <mrow id="S4.Ex3.m2.5.5.1.1.4.4.4.2" xref="S4.Ex3.m2.5.5.1.1.4.4.4.3.cmml"><mo id="S4.Ex3.m2.5.5.1.1.4.4.4.2.3" stretchy="false" xref="S4.Ex3.m2.5.5.1.1.4.4.4.3.cmml">(</mo><msubsup id="S4.Ex3.m2.5.5.1.1.3.3.3.1.1" xref="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.cmml"><mi id="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.2.2" xref="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.2.2.cmml">𝜽</mi> <mi id="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.2.3" xref="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.2.3.cmml">j</mi> <mi id="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.3" xref="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.3.cmml">t</mi></msubsup><mo id="S4.Ex3.m2.5.5.1.1.4.4.4.2.4" xref="S4.Ex3.m2.5.5.1.1.4.4.4.3.cmml">,</mo><msubsup id="S4.Ex3.m2.5.5.1.1.4.4.4.2.2" xref="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.cmml"><mi id="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.2.2" xref="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.2.2.cmml">𝒛</mi> <mi id="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.2.3" xref="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.2.3.cmml">j</mi> <mi id="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.3" xref="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.3.cmml">t</mi></msubsup><mo id="S4.Ex3.m2.5.5.1.1.4.4.4.2.5" stretchy="false" xref="S4.Ex3.m2.5.5.1.1.4.4.4.3.cmml">)</mo></mrow></mrow></mrow> <mo id="S4.Ex3.m2.5.5.1.1.9" xref="S4.Ex3.m2.5.5.1.1.9.cmml">−</mo> <mrow id="S4.Ex3.m2.5.5.1.1.8" xref="S4.Ex3.m2.5.5.1.1.8.cmml"><mstyle displaystyle="true" id="S4.Ex3.m2.5.5.1.1.8.5" xref="S4.Ex3.m2.5.5.1.1.8.5.cmml"><munder id="S4.Ex3.m2.5.5.1.1.8.5a" xref="S4.Ex3.m2.5.5.1.1.8.5.cmml"><mo id="S4.Ex3.m2.5.5.1.1.8.5.2" movablelimits="false" xref="S4.Ex3.m2.5.5.1.1.8.5.2.cmml">∑</mo> <mrow id="S4.Ex3.m2.2.2.2" xref="S4.Ex3.m2.2.2.2.cmml"><mi id="S4.Ex3.m2.2.2.2.4" xref="S4.Ex3.m2.2.2.2.4.cmml">k</mi> <mo id="S4.Ex3.m2.2.2.2.3" xref="S4.Ex3.m2.2.2.2.3.cmml">∈</mo> <mrow id="S4.Ex3.m2.2.2.2.5" xref="S4.Ex3.m2.2.2.2.5.cmml"><msubsup id="S4.Ex3.m2.2.2.2.5.2" xref="S4.Ex3.m2.2.2.2.5.2.cmml"><mi id="S4.Ex3.m2.2.2.2.5.2.2.2" xref="S4.Ex3.m2.2.2.2.5.2.2.2.cmml">𝒩</mi> <mtext id="S4.Ex3.m2.2.2.2.5.2.2.3" xref="S4.Ex3.m2.2.2.2.5.2.2.3a.cmml">out</mtext> <mrow id="S4.Ex3.m2.1.1.1.1.1.3" xref="S4.Ex3.m2.2.2.2.5.2.cmml"><mo id="S4.Ex3.m2.1.1.1.1.1.3.1" stretchy="false" xref="S4.Ex3.m2.2.2.2.5.2.cmml">(</mo><mn id="S4.Ex3.m2.1.1.1.1.1.1" xref="S4.Ex3.m2.1.1.1.1.1.1.cmml">1</mn><mo id="S4.Ex3.m2.1.1.1.1.1.3.2" stretchy="false" xref="S4.Ex3.m2.2.2.2.5.2.cmml">)</mo></mrow></msubsup> <mo id="S4.Ex3.m2.2.2.2.5.1" xref="S4.Ex3.m2.2.2.2.5.1.cmml">⁢</mo> <mrow id="S4.Ex3.m2.2.2.2.5.3.2" xref="S4.Ex3.m2.2.2.2.5.cmml"><mo id="S4.Ex3.m2.2.2.2.5.3.2.1" stretchy="false" xref="S4.Ex3.m2.2.2.2.5.cmml">(</mo><mi id="S4.Ex3.m2.2.2.2.2" xref="S4.Ex3.m2.2.2.2.2.cmml">j</mi><mo id="S4.Ex3.m2.2.2.2.5.3.2.2" stretchy="false" xref="S4.Ex3.m2.2.2.2.5.cmml">)</mo></mrow></mrow></mrow></munder></mstyle> <mrow id="S4.Ex3.m2.5.5.1.1.8.4" xref="S4.Ex3.m2.5.5.1.1.8.4.cmml"><msub id="S4.Ex3.m2.5.5.1.1.8.4.6" xref="S4.Ex3.m2.5.5.1.1.8.4.6.cmml"><mi id="S4.Ex3.m2.5.5.1.1.8.4.6.2" xref="S4.Ex3.m2.5.5.1.1.8.4.6.2.cmml">q</mi> <mi id="S4.Ex3.m2.5.5.1.1.8.4.6.3" xref="S4.Ex3.m2.5.5.1.1.8.4.6.3.cmml">k</mi></msub> <mo id="S4.Ex3.m2.5.5.1.1.8.4.5" xref="S4.Ex3.m2.5.5.1.1.8.4.5.cmml">⁢</mo> <msubsup id="S4.Ex3.m2.5.5.1.1.8.4.7" xref="S4.Ex3.m2.5.5.1.1.8.4.7.cmml"><mi id="S4.Ex3.m2.5.5.1.1.8.4.7.2.2" xref="S4.Ex3.m2.5.5.1.1.8.4.7.2.2.cmml">𝑾</mi> <mrow id="S4.Ex3.m2.4.4.2.4" xref="S4.Ex3.m2.4.4.2.3.cmml"><mi id="S4.Ex3.m2.3.3.1.1" xref="S4.Ex3.m2.3.3.1.1.cmml">k</mi><mo id="S4.Ex3.m2.4.4.2.4.1" xref="S4.Ex3.m2.4.4.2.3.cmml">,</mo><mi id="S4.Ex3.m2.4.4.2.2" xref="S4.Ex3.m2.4.4.2.2.cmml">j</mi></mrow> <mi id="S4.Ex3.m2.5.5.1.1.8.4.7.3" xref="S4.Ex3.m2.5.5.1.1.8.4.7.3.cmml">t</mi></msubsup> <mo id="S4.Ex3.m2.5.5.1.1.8.4.5a" lspace="0.167em" xref="S4.Ex3.m2.5.5.1.1.8.4.5.cmml">⁢</mo> <mrow id="S4.Ex3.m2.5.5.1.1.8.4.8" xref="S4.Ex3.m2.5.5.1.1.8.4.8.cmml"><mo id="S4.Ex3.m2.5.5.1.1.8.4.8.1" rspace="0.167em" xref="S4.Ex3.m2.5.5.1.1.8.4.8.1.cmml">∇</mo> <mi id="S4.Ex3.m2.5.5.1.1.8.4.8.2" xref="S4.Ex3.m2.5.5.1.1.8.4.8.2.cmml">L</mi></mrow> <mo id="S4.Ex3.m2.5.5.1.1.8.4.5b" xref="S4.Ex3.m2.5.5.1.1.8.4.5.cmml">⁢</mo> <msup id="S4.Ex3.m2.5.5.1.1.6.2.2" xref="S4.Ex3.m2.5.5.1.1.6.2.2.cmml"><mrow id="S4.Ex3.m2.5.5.1.1.6.2.2.2.2" xref="S4.Ex3.m2.5.5.1.1.6.2.2.2.3.cmml"><mo id="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.3" stretchy="false" xref="S4.Ex3.m2.5.5.1.1.6.2.2.2.3.cmml">(</mo><msubsup id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.cmml"><mi id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.2.2" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.2.2.cmml">𝜽</mi> <mi id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.2.3" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.2.3.cmml">k</mi> <mrow id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.3" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.3.cmml"><mi id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.3.2" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.3.2.cmml">t</mi> <mo id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.3.1" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.3.1.cmml">+</mo> <mn id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.3.3" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.3.3.cmml">1</mn></mrow></msubsup><mo id="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.4" xref="S4.Ex3.m2.5.5.1.1.6.2.2.2.3.cmml">;</mo><msup id="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.2" xref="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.2.cmml"><mi id="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.2.2" xref="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.2.2.cmml">𝒛</mi> <mo id="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.2.3" xref="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.2.3.cmml">′</mo></msup><mo id="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.5" stretchy="false" xref="S4.Ex3.m2.5.5.1.1.6.2.2.2.3.cmml">)</mo></mrow> <mo id="S4.Ex3.m2.5.5.1.1.6.2.2.4" xref="S4.Ex3.m2.5.5.1.1.6.2.2.4.cmml">⊤</mo></msup> <mo id="S4.Ex3.m2.5.5.1.1.8.4.5c" xref="S4.Ex3.m2.5.5.1.1.8.4.5.cmml">⁢</mo> <msub id="S4.Ex3.m2.5.5.1.1.8.4.9" xref="S4.Ex3.m2.5.5.1.1.8.4.9.cmml"><mi id="S4.Ex3.m2.5.5.1.1.8.4.9.2" mathvariant="normal" xref="S4.Ex3.m2.5.5.1.1.8.4.9.2.cmml">Δ</mi> <mi id="S4.Ex3.m2.5.5.1.1.8.4.9.3" xref="S4.Ex3.m2.5.5.1.1.8.4.9.3.cmml">j</mi></msub> <mo id="S4.Ex3.m2.5.5.1.1.8.4.5d" xref="S4.Ex3.m2.5.5.1.1.8.4.5.cmml">⁢</mo> <mrow id="S4.Ex3.m2.5.5.1.1.8.4.4.2" xref="S4.Ex3.m2.5.5.1.1.8.4.4.3.cmml"><mo id="S4.Ex3.m2.5.5.1.1.8.4.4.2.3" stretchy="false" xref="S4.Ex3.m2.5.5.1.1.8.4.4.3.cmml">(</mo><msubsup id="S4.Ex3.m2.5.5.1.1.7.3.3.1.1" xref="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.cmml"><mi id="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.2.2" xref="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.2.2.cmml">𝜽</mi> <mi id="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.2.3" xref="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.2.3.cmml">j</mi> <mi id="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.3" xref="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.3.cmml">t</mi></msubsup><mo id="S4.Ex3.m2.5.5.1.1.8.4.4.2.4" xref="S4.Ex3.m2.5.5.1.1.8.4.4.3.cmml">,</mo><msubsup id="S4.Ex3.m2.5.5.1.1.8.4.4.2.2" xref="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.cmml"><mi id="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.2.2" xref="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.2.2.cmml">𝒛</mi> <mi id="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.2.3" xref="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.2.3.cmml">j</mi> <mi id="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.3" xref="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.3.cmml">t</mi></msubsup><mo id="S4.Ex3.m2.5.5.1.1.8.4.4.2.5" stretchy="false" xref="S4.Ex3.m2.5.5.1.1.8.4.4.3.cmml">)</mo></mrow></mrow></mrow></mrow><mo id="S4.Ex3.m2.5.5.1.2" xref="S4.Ex3.m2.5.5.1.1.cmml">,</mo></mrow><annotation-xml encoding="MathML-Content" id="S4.Ex3.m2.5b"><apply id="S4.Ex3.m2.5.5.1.1.cmml" xref="S4.Ex3.m2.5.5.1"><apply id="S4.Ex3.m2.5.5.1.1.4.cmml" xref="S4.Ex3.m2.5.5.1.1.4"><apply id="S4.Ex3.m2.5.5.1.1.4.4.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4"><apply id="S4.Ex3.m2.5.5.1.1.4.4.6.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.6"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.4.4.6.1.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.6">subscript</csymbol> <ci id="S4.Ex3.m2.5.5.1.1.4.4.6.2.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.6.2">𝑞</ci> <ci id="S4.Ex3.m2.5.5.1.1.4.4.6.3.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.6.3">𝑗</ci></apply> <apply id="S4.Ex3.m2.5.5.1.1.4.4.7.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.7"><ci id="S4.Ex3.m2.5.5.1.1.4.4.7.1.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.7.1">∇</ci> <ci id="S4.Ex3.m2.5.5.1.1.4.4.7.2.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.7.2">𝐿</ci></apply> <apply id="S4.Ex3.m2.5.5.1.1.2.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.2.2.2"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.2.2.2.3.cmml" xref="S4.Ex3.m2.5.5.1.1.2.2.2">superscript</csymbol> <list id="S4.Ex3.m2.5.5.1.1.2.2.2.2.3.cmml" xref="S4.Ex3.m2.5.5.1.1.2.2.2.2.2"><apply id="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.cmml" xref="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.1.cmml" xref="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1">superscript</csymbol> <apply id="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.2.cmml" xref="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.2.1.cmml" xref="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1">subscript</csymbol> <ci id="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.2.2">𝜽</ci> <ci id="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.2.3.cmml" xref="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.2.3">𝑗</ci></apply> <ci id="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.3.cmml" xref="S4.Ex3.m2.5.5.1.1.1.1.1.1.1.1.3">𝑡</ci></apply> <apply id="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.2"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.2.1.cmml" xref="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.2">superscript</csymbol> <ci id="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.2.2">𝒛</ci> <ci id="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.2.3.cmml" xref="S4.Ex3.m2.5.5.1.1.2.2.2.2.2.2.3">′</ci></apply></list> <csymbol cd="latexml" id="S4.Ex3.m2.5.5.1.1.2.2.2.4.cmml" xref="S4.Ex3.m2.5.5.1.1.2.2.2.4">top</csymbol></apply> <apply id="S4.Ex3.m2.5.5.1.1.4.4.8.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.8"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.4.4.8.1.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.8">subscript</csymbol> <ci id="S4.Ex3.m2.5.5.1.1.4.4.8.2.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.8.2">Δ</ci> <ci id="S4.Ex3.m2.5.5.1.1.4.4.8.3.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.8.3">𝑗</ci></apply> <interval closure="open" id="S4.Ex3.m2.5.5.1.1.4.4.4.3.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.4.2"><apply id="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.cmml" xref="S4.Ex3.m2.5.5.1.1.3.3.3.1.1"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.1.cmml" xref="S4.Ex3.m2.5.5.1.1.3.3.3.1.1">superscript</csymbol> <apply id="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.2.cmml" xref="S4.Ex3.m2.5.5.1.1.3.3.3.1.1"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.2.1.cmml" xref="S4.Ex3.m2.5.5.1.1.3.3.3.1.1">subscript</csymbol> <ci id="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.2.2">𝜽</ci> <ci id="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.2.3.cmml" xref="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.2.3">𝑗</ci></apply> <ci id="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.3.cmml" xref="S4.Ex3.m2.5.5.1.1.3.3.3.1.1.3">𝑡</ci></apply> <apply id="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.4.2.2"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.1.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.4.2.2">superscript</csymbol> <apply id="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.4.2.2"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.2.1.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.4.2.2">subscript</csymbol> <ci id="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.2.2">𝒛</ci> <ci id="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.2.3.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.2.3">𝑗</ci></apply> <ci id="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.3.cmml" xref="S4.Ex3.m2.5.5.1.1.4.4.4.2.2.3">𝑡</ci></apply></interval></apply></apply> <apply id="S4.Ex3.m2.5.5.1.1.8.cmml" xref="S4.Ex3.m2.5.5.1.1.8"><apply id="S4.Ex3.m2.5.5.1.1.8.5.cmml" xref="S4.Ex3.m2.5.5.1.1.8.5"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.8.5.1.cmml" xref="S4.Ex3.m2.5.5.1.1.8.5">subscript</csymbol> <apply id="S4.Ex3.m2.2.2.2.cmml" xref="S4.Ex3.m2.2.2.2"><ci id="S4.Ex3.m2.2.2.2.4.cmml" xref="S4.Ex3.m2.2.2.2.4">𝑘</ci> <apply id="S4.Ex3.m2.2.2.2.5.cmml" xref="S4.Ex3.m2.2.2.2.5"><apply id="S4.Ex3.m2.2.2.2.5.2.cmml" xref="S4.Ex3.m2.2.2.2.5.2"><csymbol cd="ambiguous" id="S4.Ex3.m2.2.2.2.5.2.1.cmml" xref="S4.Ex3.m2.2.2.2.5.2">superscript</csymbol> <apply id="S4.Ex3.m2.2.2.2.5.2.2.cmml" xref="S4.Ex3.m2.2.2.2.5.2"><csymbol cd="ambiguous" id="S4.Ex3.m2.2.2.2.5.2.2.1.cmml" xref="S4.Ex3.m2.2.2.2.5.2">subscript</csymbol> <ci id="S4.Ex3.m2.2.2.2.5.2.2.2.cmml" xref="S4.Ex3.m2.2.2.2.5.2.2.2">𝒩</ci> <ci id="S4.Ex3.m2.2.2.2.5.2.2.3a.cmml" xref="S4.Ex3.m2.2.2.2.5.2.2.3"><mtext id="S4.Ex3.m2.2.2.2.5.2.2.3.cmml" mathsize="50%" xref="S4.Ex3.m2.2.2.2.5.2.2.3">out</mtext></ci></apply> <cn id="S4.Ex3.m2.1.1.1.1.1.1.cmml" type="integer" xref="S4.Ex3.m2.1.1.1.1.1.1">1</cn></apply> <ci id="S4.Ex3.m2.2.2.2.2.cmml" xref="S4.Ex3.m2.2.2.2.2">𝑗</ci></apply></apply></apply> <apply id="S4.Ex3.m2.5.5.1.1.8.4.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4"><apply id="S4.Ex3.m2.5.5.1.1.8.4.6.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.6"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.8.4.6.1.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.6">subscript</csymbol> <ci id="S4.Ex3.m2.5.5.1.1.8.4.6.2.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.6.2">𝑞</ci> <ci id="S4.Ex3.m2.5.5.1.1.8.4.6.3.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.6.3">𝑘</ci></apply> <apply id="S4.Ex3.m2.5.5.1.1.8.4.7.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.7"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.8.4.7.1.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.7">superscript</csymbol> <apply id="S4.Ex3.m2.5.5.1.1.8.4.7.2.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.7"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.8.4.7.2.1.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.7">subscript</csymbol> <ci id="S4.Ex3.m2.5.5.1.1.8.4.7.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.7.2.2">𝑾</ci> <list id="S4.Ex3.m2.4.4.2.3.cmml" xref="S4.Ex3.m2.4.4.2.4"><ci id="S4.Ex3.m2.3.3.1.1.cmml" xref="S4.Ex3.m2.3.3.1.1">𝑘</ci> <ci id="S4.Ex3.m2.4.4.2.2.cmml" xref="S4.Ex3.m2.4.4.2.2">𝑗</ci></list></apply> <ci id="S4.Ex3.m2.5.5.1.1.8.4.7.3.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.7.3">𝑡</ci></apply> <apply id="S4.Ex3.m2.5.5.1.1.8.4.8.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.8"><ci id="S4.Ex3.m2.5.5.1.1.8.4.8.1.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.8.1">∇</ci> <ci id="S4.Ex3.m2.5.5.1.1.8.4.8.2.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.8.2">𝐿</ci></apply> <apply id="S4.Ex3.m2.5.5.1.1.6.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.6.2.2"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.6.2.2.3.cmml" xref="S4.Ex3.m2.5.5.1.1.6.2.2">superscript</csymbol> <list id="S4.Ex3.m2.5.5.1.1.6.2.2.2.3.cmml" xref="S4.Ex3.m2.5.5.1.1.6.2.2.2.2"><apply id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.cmml" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.1.cmml" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1">superscript</csymbol> <apply id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.2.cmml" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.2.1.cmml" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1">subscript</csymbol> <ci id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.2.2">𝜽</ci> <ci id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.2.3.cmml" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.2.3">𝑘</ci></apply> <apply id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.3.cmml" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.3"><ci id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.3.2.cmml" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.3.2">𝑡</ci> <cn id="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.3.3.cmml" type="integer" xref="S4.Ex3.m2.5.5.1.1.5.1.1.1.1.1.3.3">1</cn></apply></apply> <apply id="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.2"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.2.1.cmml" xref="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.2">superscript</csymbol> <ci id="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.2.2">𝒛</ci> <ci id="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.2.3.cmml" xref="S4.Ex3.m2.5.5.1.1.6.2.2.2.2.2.3">′</ci></apply></list> <csymbol cd="latexml" id="S4.Ex3.m2.5.5.1.1.6.2.2.4.cmml" xref="S4.Ex3.m2.5.5.1.1.6.2.2.4">top</csymbol></apply> <apply id="S4.Ex3.m2.5.5.1.1.8.4.9.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.9"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.8.4.9.1.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.9">subscript</csymbol> <ci id="S4.Ex3.m2.5.5.1.1.8.4.9.2.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.9.2">Δ</ci> <ci id="S4.Ex3.m2.5.5.1.1.8.4.9.3.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.9.3">𝑗</ci></apply> <interval closure="open" id="S4.Ex3.m2.5.5.1.1.8.4.4.3.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.4.2"><apply id="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.cmml" xref="S4.Ex3.m2.5.5.1.1.7.3.3.1.1"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.1.cmml" xref="S4.Ex3.m2.5.5.1.1.7.3.3.1.1">superscript</csymbol> <apply id="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.2.cmml" xref="S4.Ex3.m2.5.5.1.1.7.3.3.1.1"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.2.1.cmml" xref="S4.Ex3.m2.5.5.1.1.7.3.3.1.1">subscript</csymbol> <ci id="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.2.2">𝜽</ci> <ci id="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.2.3.cmml" xref="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.2.3">𝑗</ci></apply> <ci id="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.3.cmml" xref="S4.Ex3.m2.5.5.1.1.7.3.3.1.1.3">𝑡</ci></apply> <apply id="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.4.2.2"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.1.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.4.2.2">superscript</csymbol> <apply id="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.4.2.2"><csymbol cd="ambiguous" id="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.2.1.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.4.2.2">subscript</csymbol> <ci id="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.2.2.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.2.2">𝒛</ci> <ci id="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.2.3.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.2.3">𝑗</ci></apply> <ci id="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.3.cmml" xref="S4.Ex3.m2.5.5.1.1.8.4.4.2.2.3">𝑡</ci></apply></interval></apply></apply></apply></annotation-xml> <annotation encoding="application/x-tex" id="S4.Ex3.m2.5c">\displaystyle\;-q_{j}\,\nabla L(\bm{\theta}_{j}^{t};\bm{z}^{\prime})^{\top}% \Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})-\sum_{k\in\mathcal{N}_{\text{% out}}^{(1)}(j)}q_{k}\,\bm{W}_{k,j}^{t}\,\nabla L(\bm{\theta}_{k}^{t+1};\bm{z}^% {\prime})^{\top}\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t}),</annotation><annotation encoding="application/x-llamapun" id="S4.Ex3.m2.5d">- italic_q start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ∇ italic_L ( bold_italic_θ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT; bold_italic_z start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT roman_Δ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ( bold_italic_θ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, bold_italic_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT ) - ∑ start_POSTSUBSCRIPT italic_k ∈ caligraphic_N start_POSTSUBSCRIPT out end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( 1 ) end_POSTSUPERSCRIPT ( italic_j ) end_POSTSUBSCRIPT italic_q start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_k, italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT ∇ italic_L ( bold_italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t + 1 end_POSTSUPERSCRIPT; bold_italic_z start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT roman_Δ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ( bold_italic_θ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, bold_italic_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT ),</annotation></semantics></math></span></span> <span id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1">where <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})=\mathcal{O}_{j}(\bm{\theta}_{j}%
^{t},\bm{z}_{j}^{t})-\bm{\theta}_{j}^{t}"><semantics id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4a"><mrow id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.cmml"><mrow id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.cmml"><msub id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.4" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.4.cmml"><mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.4.2" mathvariant="normal" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.4.2.cmml">Δ</mi> <mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.4.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.4.3.cmml">j</mi></msub> <mo id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.3.cmml">⁢</mo> <mrow id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.3.cmml"><mo id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.3" stretchy="false" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.3.cmml">(</mo><msubsup id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.cmml"><mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.2.2" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.2.2.cmml">𝜽</mi> <mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.2.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.2.3.cmml">j</mi> <mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.3.cmml">t</mi></msubsup><mo id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.4" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.3.cmml">,</mo><msubsup id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.cmml"><mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.2.2" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.2.2.cmml">𝒛</mi> <mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.2.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.2.3.cmml">j</mi> <mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.3.cmml">t</mi></msubsup><mo id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.5" stretchy="false" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.3.cmml">)</mo></mrow></mrow> <mo id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.5" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.5.cmml">=</mo> <mrow id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.cmml"><mrow id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.cmml"><msub id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.4" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.4.cmml"><mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.4.2" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.4.2.cmml">𝒪</mi> <mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.4.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.4.3.cmml">j</mi></msub> <mo id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.3.cmml">⁢</mo> <mrow id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.3.cmml"><mo id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.3" stretchy="false" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.3.cmml">(</mo><msubsup id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.cmml"><mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.2.2" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.2.2.cmml">𝜽</mi> <mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.2.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.2.3.cmml">j</mi> <mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.3.cmml">t</mi></msubsup><mo id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.4" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.3.cmml">,</mo><msubsup id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.cmml"><mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.2.2" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.2.2.cmml">𝒛</mi> <mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.2.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.2.3.cmml">j</mi> <mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.3.cmml">t</mi></msubsup><mo id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.5" stretchy="false" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.3.cmml">)</mo></mrow></mrow> <mo id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.3.cmml">−</mo> <msubsup id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.cmml"><mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.2.2" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.2.2.cmml">𝜽</mi> <mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.2.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.2.3.cmml">j</mi> <mi id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.3" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.3.cmml">t</mi></msubsup></mrow></mrow> <annotation-xml encoding="MathML-Content" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4b"><apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4"><apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2"><apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.4.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.4"><csymbol cd="ambiguous" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.4.1.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.4">subscript</csymbol> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.4.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.4.2">Δ</ci> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.4.3.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.4.3">𝑗</ci></apply> <interval closure="open" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.3.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2"><apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1"><csymbol cd="ambiguous" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.1.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1">superscript</csymbol> <apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1"><csymbol cd="ambiguous" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.2.1.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1">subscript</csymbol> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.2.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.2.2">𝜽</ci> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.2.3.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.2.3">𝑗</ci></apply> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.3.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.1.1.1.1.1.1.3">𝑡</ci></apply> <apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2"><csymbol cd="ambiguous" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.1.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2">superscript</csymbol> <apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2"><csymbol cd="ambiguous" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.2.1.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2">subscript</csymbol> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.2.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.2.2">𝒛</ci> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.2.3.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.2.3">𝑗</ci></apply> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.3.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.2.2.2.2.2.2.3">𝑡</ci></apply></interval></apply> <apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4"><apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2"><apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.4.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.4"><csymbol cd="ambiguous" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.4.1.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.4">subscript</csymbol> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.4.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.4.2">𝒪</ci> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.4.3.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.4.3">𝑗</ci></apply> <interval closure="open" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.3.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2"><apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1"><csymbol cd="ambiguous" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.1.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1">superscript</csymbol> <apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1"><csymbol cd="ambiguous" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.2.1.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1">subscript</csymbol> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.2.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.2.2">𝜽</ci> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.2.3.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.2.3">𝑗</ci></apply> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.3.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.3.3.3.1.1.1.1.3">𝑡</ci></apply> <apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2"><csymbol cd="ambiguous" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.1.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2">superscript</csymbol> <apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2"><csymbol cd="ambiguous" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.2.1.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2">subscript</csymbol> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.2.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.2.2">𝒛</ci> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.2.3.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.2.3">𝑗</ci></apply> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.3.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.2.2.2.2.3">𝑡</ci></apply></interval></apply> <apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4"><csymbol cd="ambiguous" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.1.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4">superscript</csymbol> <apply id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4"><csymbol cd="ambiguous" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.2.1.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4">subscript</csymbol> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.2.2.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.2.2">𝜽</ci> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.2.3.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.2.3">𝑗</ci></apply> <ci id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.3.cmml" xref="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4.4.4.4.3">𝑡</ci></apply></apply></apply></annotation-xml> <annotation encoding="application/x-tex" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4c">\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})=\mathcal{O}_{j}(\bm{\theta}_{j}% ^{t},\bm{z}_{j}^{t})-\bm{\theta}_{j}^{t}</annotation> <annotation encoding="application/x-llamapun" id="S4.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.p1.1.m1.4d">roman_Δ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ( bold_italic_θ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, bold_italic_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT ) = caligraphic_O start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ( bold_italic_θ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, bold_italic_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT ) - bold_italic_θ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT</annotation></semantics></math>. The proof is included in Subsection&nbsp;C.1.</span></span></span></span></foreignObject></g></g></svg>

Additivity. The one-hop DICE-E influence measure is additive over training instances. Specifically, for a mini-batch $\mathcal{B}_{j}^{t}$ from participant $j$, the total influence is the sum of individual influences:

$$
\displaystyle\mathcal{I}_{\text{DICE-E}}^{(1)}(\mathcal{B}_{j}^{t},\bm{z}^{%
\prime})=\sum_{\bm{z}_{j}^{t}\in\mathcal{B}_{j}^{t}}\mathcal{I}_{\text{DICE-E}%
}^{(1)}(\bm{z}_{j}^{t},\bm{z}^{\prime}).
$$

The additivity provides guarantees for efficient computation of DICE-E score for large mini-batches.

We can then extend the influence approximation to multi-hop neighbors in decentralized learning and show how the influence of a data instance cascades over the decentralized learning network.

<svg height="168.68" id="S4.SS2.p5.pic1" overflow="visible" version="1.1" width="600"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,168.68) matrix(1 0 0 -1 0 0)"><g fill="#BFBFBF" fill-opacity="1.0"><path d="M 0 6.7 L 0 161.97 C 0 165.68 3 168.68 6.7 168.68 L 593.3 168.68 C 597 168.68 600 165.68 600 161.97 L 600 6.7 C 600 3 597 0 593.3 0 L 6.7 0 C 3 0 0 3 0 6.7 Z" style="stroke:none"></path></g><g fill="#F9FEFE" fill-opacity="1.0"><path d="M 2.77 6.7 L 2.77 161.97 C 2.77 164.15 4.53 165.91 6.7 165.91 L 593.3 165.91 C 595.47 165.91 597.23 164.15 597.23 161.97 L 597.23 6.7 C 597.23 4.53 595.47 2.77 593.3 2.77 L 6.7 2.77 C 4.53 2.77 2.77 4.53 2.77 6.7 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 8.67 10.64)"><foreignObject color="#000000" height="147.39" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="581.87"><span id="S4.SS2.p5.pic1.1.1.1.1.1" style="width:420.5pt;"><span id="Thmtheorem1"><h6>Theorem 1 (Approximation of r𝑟ritalic_r-hop DICE-GT).</h6><span id="Thmtheorem1.p1"><span id="Thmtheorem1.p1.2">The <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="r"><semantics id="Thmtheorem1.p1.1.m1.1a"><mi id="Thmtheorem1.p1.1.m1.1.1" xref="Thmtheorem1.p1.1.m1.1.1.cmml">r</mi> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.1.m1.1b"><ci id="Thmtheorem1.p1.1.m1.1.1.cmml" xref="Thmtheorem1.p1.1.m1.1.1">𝑟</ci></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.1.m1.1c">r</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.1.m1.1d">italic_r</annotation></semantics></math> -hop DICE-GT influence <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\mathcal{I}_{\text{DICE-GT}}^{(r)}(\bm{z}_{j}^{t},\bm{z}^{\prime})"><semantics id="Thmtheorem1.p1.2.m2.3a"><mrow id="Thmtheorem1.p1.2.m2.3.3" xref="Thmtheorem1.p1.2.m2.3.3.cmml"><msubsup id="Thmtheorem1.p1.2.m2.3.3.4" xref="Thmtheorem1.p1.2.m2.3.3.4.cmml"><mi id="Thmtheorem1.p1.2.m2.3.3.4.2.2" xref="Thmtheorem1.p1.2.m2.3.3.4.2.2.cmml">ℐ</mi> <mtext id="Thmtheorem1.p1.2.m2.3.3.4.2.3" xref="Thmtheorem1.p1.2.m2.3.3.4.2.3a.cmml">DICE-GT</mtext> <mrow id="Thmtheorem1.p1.2.m2.1.1.1.3" xref="Thmtheorem1.p1.2.m2.3.3.4.cmml"><mo id="Thmtheorem1.p1.2.m2.1.1.1.3.1" stretchy="false" xref="Thmtheorem1.p1.2.m2.3.3.4.cmml">(</mo><mi id="Thmtheorem1.p1.2.m2.1.1.1.1" xref="Thmtheorem1.p1.2.m2.1.1.1.1.cmml">r</mi><mo id="Thmtheorem1.p1.2.m2.1.1.1.3.2" stretchy="false" xref="Thmtheorem1.p1.2.m2.3.3.4.cmml">)</mo></mrow></msubsup> <mo id="Thmtheorem1.p1.2.m2.3.3.3" xref="Thmtheorem1.p1.2.m2.3.3.3.cmml">⁢</mo> <mrow id="Thmtheorem1.p1.2.m2.3.3.2.2" xref="Thmtheorem1.p1.2.m2.3.3.2.3.cmml"><mo id="Thmtheorem1.p1.2.m2.3.3.2.2.3" stretchy="false" xref="Thmtheorem1.p1.2.m2.3.3.2.3.cmml">(</mo><msubsup id="Thmtheorem1.p1.2.m2.2.2.1.1.1" xref="Thmtheorem1.p1.2.m2.2.2.1.1.1.cmml"><mi id="Thmtheorem1.p1.2.m2.2.2.1.1.1.2.2" xref="Thmtheorem1.p1.2.m2.2.2.1.1.1.2.2.cmml">𝒛</mi> <mi id="Thmtheorem1.p1.2.m2.2.2.1.1.1.2.3" xref="Thmtheorem1.p1.2.m2.2.2.1.1.1.2.3.cmml">j</mi> <mi id="Thmtheorem1.p1.2.m2.2.2.1.1.1.3" xref="Thmtheorem1.p1.2.m2.2.2.1.1.1.3.cmml">t</mi></msubsup><mo id="Thmtheorem1.p1.2.m2.3.3.2.2.4" xref="Thmtheorem1.p1.2.m2.3.3.2.3.cmml">,</mo><msup id="Thmtheorem1.p1.2.m2.3.3.2.2.2" xref="Thmtheorem1.p1.2.m2.3.3.2.2.2.cmml"><mi id="Thmtheorem1.p1.2.m2.3.3.2.2.2.2" xref="Thmtheorem1.p1.2.m2.3.3.2.2.2.2.cmml">𝒛</mi> <mo id="Thmtheorem1.p1.2.m2.3.3.2.2.2.3" xref="Thmtheorem1.p1.2.m2.3.3.2.2.2.3.cmml">′</mo></msup><mo id="Thmtheorem1.p1.2.m2.3.3.2.2.5" stretchy="false" xref="Thmtheorem1.p1.2.m2.3.3.2.3.cmml">)</mo></mrow></mrow> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.2.m2.3b"><apply id="Thmtheorem1.p1.2.m2.3.3.cmml" xref="Thmtheorem1.p1.2.m2.3.3"><apply id="Thmtheorem1.p1.2.m2.3.3.4.cmml" xref="Thmtheorem1.p1.2.m2.3.3.4"><csymbol cd="ambiguous" id="Thmtheorem1.p1.2.m2.3.3.4.1.cmml" xref="Thmtheorem1.p1.2.m2.3.3.4">superscript</csymbol> <apply id="Thmtheorem1.p1.2.m2.3.3.4.2.cmml" xref="Thmtheorem1.p1.2.m2.3.3.4"><csymbol cd="ambiguous" id="Thmtheorem1.p1.2.m2.3.3.4.2.1.cmml" xref="Thmtheorem1.p1.2.m2.3.3.4">subscript</csymbol> <ci id="Thmtheorem1.p1.2.m2.3.3.4.2.2.cmml" xref="Thmtheorem1.p1.2.m2.3.3.4.2.2">ℐ</ci> <ci id="Thmtheorem1.p1.2.m2.3.3.4.2.3a.cmml" xref="Thmtheorem1.p1.2.m2.3.3.4.2.3"><mtext id="Thmtheorem1.p1.2.m2.3.3.4.2.3.cmml" mathsize="70%" xref="Thmtheorem1.p1.2.m2.3.3.4.2.3">DICE-GT</mtext></ci></apply> <ci id="Thmtheorem1.p1.2.m2.1.1.1.1.cmml" xref="Thmtheorem1.p1.2.m2.1.1.1.1">𝑟</ci></apply> <interval closure="open" id="Thmtheorem1.p1.2.m2.3.3.2.3.cmml" xref="Thmtheorem1.p1.2.m2.3.3.2.2"><apply id="Thmtheorem1.p1.2.m2.2.2.1.1.1.cmml" xref="Thmtheorem1.p1.2.m2.2.2.1.1.1"><csymbol cd="ambiguous" id="Thmtheorem1.p1.2.m2.2.2.1.1.1.1.cmml" xref="Thmtheorem1.p1.2.m2.2.2.1.1.1">superscript</csymbol> <apply id="Thmtheorem1.p1.2.m2.2.2.1.1.1.2.cmml" xref="Thmtheorem1.p1.2.m2.2.2.1.1.1"><csymbol cd="ambiguous" id="Thmtheorem1.p1.2.m2.2.2.1.1.1.2.1.cmml" xref="Thmtheorem1.p1.2.m2.2.2.1.1.1">subscript</csymbol> <ci id="Thmtheorem1.p1.2.m2.2.2.1.1.1.2.2.cmml" xref="Thmtheorem1.p1.2.m2.2.2.1.1.1.2.2">𝒛</ci> <ci id="Thmtheorem1.p1.2.m2.2.2.1.1.1.2.3.cmml" xref="Thmtheorem1.p1.2.m2.2.2.1.1.1.2.3">𝑗</ci></apply> <ci id="Thmtheorem1.p1.2.m2.2.2.1.1.1.3.cmml" xref="Thmtheorem1.p1.2.m2.2.2.1.1.1.3">𝑡</ci></apply> <apply id="Thmtheorem1.p1.2.m2.3.3.2.2.2.cmml" xref="Thmtheorem1.p1.2.m2.3.3.2.2.2"><csymbol cd="ambiguous" id="Thmtheorem1.p1.2.m2.3.3.2.2.2.1.cmml" xref="Thmtheorem1.p1.2.m2.3.3.2.2.2">superscript</csymbol> <ci id="Thmtheorem1.p1.2.m2.3.3.2.2.2.2.cmml" xref="Thmtheorem1.p1.2.m2.3.3.2.2.2.2">𝒛</ci> <ci id="Thmtheorem1.p1.2.m2.3.3.2.2.2.3.cmml" xref="Thmtheorem1.p1.2.m2.3.3.2.2.2.3">′</ci></apply></interval></apply></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.2.m2.3c">\mathcal{I}_{\text{DICE-GT}}^{(r)}(\bm{z}_{j}^{t},\bm{z}^{\prime})</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.2.m2.3d">caligraphic_I start_POSTSUBSCRIPT DICE-GT end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_r ) end_POSTSUPERSCRIPT ( bold_italic_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, bold_italic_z start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT )</annotation></semantics></math> (see Definition&nbsp;3) can be approximated as follows:</span> <span id="A4.EGx8"><span id="S4.Ex4"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\displaystyle\mathcal{I}_{\text{DICE-E}}^{(r)}(\bm{z}_{j}^{t},\bm{z}^{\prime})=-"><semantics id="S4.Ex4.m1.3a"><mrow id="S4.Ex4.m1.3.3" xref="S4.Ex4.m1.3.3.cmml"><mrow id="S4.Ex4.m1.3.3.2" xref="S4.Ex4.m1.3.3.2.cmml"><msubsup id="S4.Ex4.m1.3.3.2.4" xref="S4.Ex4.m1.3.3.2.4.cmml"><mi id="S4.Ex4.m1.3.3.2.4.2.2" xref="S4.Ex4.m1.3.3.2.4.2.2.cmml">ℐ</mi> <mtext id="S4.Ex4.m1.3.3.2.4.2.3" xref="S4.Ex4.m1.3.3.2.4.2.3a.cmml">DICE-E</mtext> <mrow id="S4.Ex4.m1.1.1.1.3" xref="S4.Ex4.m1.3.3.2.4.cmml"><mo id="S4.Ex4.m1.1.1.1.3.1" stretchy="false" xref="S4.Ex4.m1.3.3.2.4.cmml">(</mo><mi id="S4.Ex4.m1.1.1.1.1" xref="S4.Ex4.m1.1.1.1.1.cmml">r</mi><mo id="S4.Ex4.m1.1.1.1.3.2" stretchy="false" xref="S4.Ex4.m1.3.3.2.4.cmml">)</mo></mrow></msubsup> <mo id="S4.Ex4.m1.3.3.2.3" xref="S4.Ex4.m1.3.3.2.3.cmml">⁢</mo> <mrow id="S4.Ex4.m1.3.3.2.2.2" xref="S4.Ex4.m1.3.3.2.2.3.cmml"><mo id="S4.Ex4.m1.3.3.2.2.2.3" stretchy="false" xref="S4.Ex4.m1.3.3.2.2.3.cmml">(</mo><msubsup id="S4.Ex4.m1.2.2.1.1.1.1" xref="S4.Ex4.m1.2.2.1.1.1.1.cmml"><mi id="S4.Ex4.m1.2.2.1.1.1.1.2.2" xref="S4.Ex4.m1.2.2.1.1.1.1.2.2.cmml">𝒛</mi> <mi id="S4.Ex4.m1.2.2.1.1.1.1.2.3" xref="S4.Ex4.m1.2.2.1.1.1.1.2.3.cmml">j</mi> <mi id="S4.Ex4.m1.2.2.1.1.1.1.3" xref="S4.Ex4.m1.2.2.1.1.1.1.3.cmml">t</mi></msubsup><mo id="S4.Ex4.m1.3.3.2.2.2.4" xref="S4.Ex4.m1.3.3.2.2.3.cmml">,</mo><msup id="S4.Ex4.m1.3.3.2.2.2.2" xref="S4.Ex4.m1.3.3.2.2.2.2.cmml"><mi id="S4.Ex4.m1.3.3.2.2.2.2.2" xref="S4.Ex4.m1.3.3.2.2.2.2.2.cmml">𝒛</mi> <mo id="S4.Ex4.m1.3.3.2.2.2.2.3" xref="S4.Ex4.m1.3.3.2.2.2.2.3.cmml">′</mo></msup><mo id="S4.Ex4.m1.3.3.2.2.2.5" stretchy="false" xref="S4.Ex4.m1.3.3.2.2.3.cmml">)</mo></mrow></mrow> <mo id="S4.Ex4.m1.3.3.3" rspace="0em" xref="S4.Ex4.m1.3.3.3.cmml">=</mo> <mo id="S4.Ex4.m1.3.3.4" lspace="0em" xref="S4.Ex4.m1.3.3.4.cmml">−</mo></mrow> <annotation-xml encoding="MathML-Content" id="S4.Ex4.m1.3b"><apply id="S4.Ex4.m1.3.3.cmml" xref="S4.Ex4.m1.3.3"><apply id="S4.Ex4.m1.3.3.2.cmml" xref="S4.Ex4.m1.3.3.2"><apply id="S4.Ex4.m1.3.3.2.4.cmml" xref="S4.Ex4.m1.3.3.2.4"><csymbol cd="ambiguous" id="S4.Ex4.m1.3.3.2.4.1.cmml" xref="S4.Ex4.m1.3.3.2.4">superscript</csymbol> <apply id="S4.Ex4.m1.3.3.2.4.2.cmml" xref="S4.Ex4.m1.3.3.2.4"><csymbol cd="ambiguous" id="S4.Ex4.m1.3.3.2.4.2.1.cmml" xref="S4.Ex4.m1.3.3.2.4">subscript</csymbol> <ci id="S4.Ex4.m1.3.3.2.4.2.2.cmml" xref="S4.Ex4.m1.3.3.2.4.2.2">ℐ</ci> <ci id="S4.Ex4.m1.3.3.2.4.2.3a.cmml" xref="S4.Ex4.m1.3.3.2.4.2.3"><mtext id="S4.Ex4.m1.3.3.2.4.2.3.cmml" mathsize="70%" xref="S4.Ex4.m1.3.3.2.4.2.3">DICE-E</mtext></ci></apply> <ci id="S4.Ex4.m1.1.1.1.1.cmml" xref="S4.Ex4.m1.1.1.1.1">𝑟</ci></apply> <interval closure="open" id="S4.Ex4.m1.3.3.2.2.3.cmml" xref="S4.Ex4.m1.3.3.2.2.2"><apply id="S4.Ex4.m1.2.2.1.1.1.1.cmml" xref="S4.Ex4.m1.2.2.1.1.1.1"><csymbol cd="ambiguous" id="S4.Ex4.m1.2.2.1.1.1.1.1.cmml" xref="S4.Ex4.m1.2.2.1.1.1.1">superscript</csymbol> <apply id="S4.Ex4.m1.2.2.1.1.1.1.2.cmml" xref="S4.Ex4.m1.2.2.1.1.1.1"><csymbol cd="ambiguous" id="S4.Ex4.m1.2.2.1.1.1.1.2.1.cmml" xref="S4.Ex4.m1.2.2.1.1.1.1">subscript</csymbol> <ci id="S4.Ex4.m1.2.2.1.1.1.1.2.2.cmml" xref="S4.Ex4.m1.2.2.1.1.1.1.2.2">𝒛</ci> <ci id="S4.Ex4.m1.2.2.1.1.1.1.2.3.cmml" xref="S4.Ex4.m1.2.2.1.1.1.1.2.3">𝑗</ci></apply> <ci id="S4.Ex4.m1.2.2.1.1.1.1.3.cmml" xref="S4.Ex4.m1.2.2.1.1.1.1.3">𝑡</ci></apply> <apply id="S4.Ex4.m1.3.3.2.2.2.2.cmml" xref="S4.Ex4.m1.3.3.2.2.2.2"><csymbol cd="ambiguous" id="S4.Ex4.m1.3.3.2.2.2.2.1.cmml" xref="S4.Ex4.m1.3.3.2.2.2.2">superscript</csymbol> <ci id="S4.Ex4.m1.3.3.2.2.2.2.2.cmml" xref="S4.Ex4.m1.3.3.2.2.2.2.2">𝒛</ci> <ci id="S4.Ex4.m1.3.3.2.2.2.2.3.cmml" xref="S4.Ex4.m1.3.3.2.2.2.2.3">′</ci></apply></interval></apply></apply></annotation-xml> <annotation encoding="application/x-tex" id="S4.Ex4.m1.3c">\displaystyle\mathcal{I}_{\text{DICE-E}}^{(r)}(\bm{z}_{j}^{t},\bm{z}^{\prime})=-</annotation> <annotation encoding="application/x-llamapun" id="S4.Ex4.m1.3d">caligraphic_I start_POSTSUBSCRIPT DICE-E end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_r ) end_POSTSUPERSCRIPT ( bold_italic_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, bold_italic_z start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) = -</annotation></semantics></math> <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\displaystyle\sum_{\rho=0}^{r}\sum_{(k_{1},\dots,k_{\rho})\in P_{j}^{(\rho)}}%
\eta^{t}q_{k_{\rho}}\underbrace{\hbox{\pagecolor{green!10}$\displaystyle\left(%
\prod_{s=1}^{\rho}\bm{W}_{k_{s},k_{s-1}}^{t+s-1}\right)$}}_{{\text{%
communication graph-related term}}}\underbrace{\hbox{\pagecolor{Mulberry!10}$%
\displaystyle\vphantom{\prod_{s=1}^{\rho}}\nabla L\bigl{(}\bm{\theta}_{k_{\rho%
}}^{t+\rho};\bm{z}^{\prime}\bigr{)}^{\top}$}}_{{\text{test gradient}}}"><semantics id="S4.Ex4.m2.11a"><mrow id="S4.Ex4.m2.11.12" xref="S4.Ex4.m2.11.12.cmml"><mstyle displaystyle="true" id="S4.Ex4.m2.11.12.1" xref="S4.Ex4.m2.11.12.1.cmml"><munderover id="S4.Ex4.m2.11.12.1a" xref="S4.Ex4.m2.11.12.1.cmml"><mo id="S4.Ex4.m2.11.12.1.2.2" movablelimits="false" xref="S4.Ex4.m2.11.12.1.2.2.cmml">∑</mo> <mrow id="S4.Ex4.m2.11.12.1.2.3" xref="S4.Ex4.m2.11.12.1.2.3.cmml"><mi id="S4.Ex4.m2.11.12.1.2.3.2" xref="S4.Ex4.m2.11.12.1.2.3.2.cmml">ρ</mi> <mo id="S4.Ex4.m2.11.12.1.2.3.1" xref="S4.Ex4.m2.11.12.1.2.3.1.cmml">=</mo> <mn id="S4.Ex4.m2.11.12.1.2.3.3" xref="S4.Ex4.m2.11.12.1.2.3.3.cmml">0</mn></mrow> <mi id="S4.Ex4.m2.11.12.1.3" xref="S4.Ex4.m2.11.12.1.3.cmml">r</mi></munderover></mstyle> <mrow id="S4.Ex4.m2.11.12.2" xref="S4.Ex4.m2.11.12.2.cmml"><mstyle displaystyle="true" id="S4.Ex4.m2.11.12.2.1" xref="S4.Ex4.m2.11.12.2.1.cmml"><munder id="S4.Ex4.m2.11.12.2.1a" xref="S4.Ex4.m2.11.12.2.1.cmml"><mo id="S4.Ex4.m2.11.12.2.1.2" movablelimits="false" xref="S4.Ex4.m2.11.12.2.1.2.cmml">∑</mo> <mrow id="S4.Ex4.m2.6.6.4" xref="S4.Ex4.m2.6.6.4.cmml"><mrow id="S4.Ex4.m2.6.6.4.4.2" xref="S4.Ex4.m2.6.6.4.4.3.cmml"><mo id="S4.Ex4.m2.6.6.4.4.2.3" stretchy="false" xref="S4.Ex4.m2.6.6.4.4.3.cmml">(</mo><msub id="S4.Ex4.m2.5.5.3.3.1.1" xref="S4.Ex4.m2.5.5.3.3.1.1.cmml"><mi id="S4.Ex4.m2.5.5.3.3.1.1.2" xref="S4.Ex4.m2.5.5.3.3.1.1.2.cmml">k</mi> <mn id="S4.Ex4.m2.5.5.3.3.1.1.3" xref="S4.Ex4.m2.5.5.3.3.1.1.3.cmml">1</mn></msub><mo id="S4.Ex4.m2.6.6.4.4.2.4" xref="S4.Ex4.m2.6.6.4.4.3.cmml">,</mo><mi id="S4.Ex4.m2.4.4.2.2" mathvariant="normal" xref="S4.Ex4.m2.4.4.2.2.cmml">…</mi><mo id="S4.Ex4.m2.6.6.4.4.2.5" xref="S4.Ex4.m2.6.6.4.4.3.cmml">,</mo><msub id="S4.Ex4.m2.6.6.4.4.2.2" xref="S4.Ex4.m2.6.6.4.4.2.2.cmml"><mi id="S4.Ex4.m2.6.6.4.4.2.2.2" xref="S4.Ex4.m2.6.6.4.4.2.2.2.cmml">k</mi> <mi id="S4.Ex4.m2.6.6.4.4.2.2.3" xref="S4.Ex4.m2.6.6.4.4.2.2.3.cmml">ρ</mi></msub><mo id="S4.Ex4.m2.6.6.4.4.2.6" stretchy="false" xref="S4.Ex4.m2.6.6.4.4.3.cmml">)</mo></mrow> <mo id="S4.Ex4.m2.6.6.4.5" xref="S4.Ex4.m2.6.6.4.5.cmml">∈</mo> <msubsup id="S4.Ex4.m2.6.6.4.6" xref="S4.Ex4.m2.6.6.4.6.cmml"><mi id="S4.Ex4.m2.6.6.4.6.2.2" xref="S4.Ex4.m2.6.6.4.6.2.2.cmml">P</mi> <mi id="S4.Ex4.m2.6.6.4.6.2.3" xref="S4.Ex4.m2.6.6.4.6.2.3.cmml">j</mi> <mrow id="S4.Ex4.m2.3.3.1.1.1.3" xref="S4.Ex4.m2.6.6.4.6.cmml"><mo id="S4.Ex4.m2.3.3.1.1.1.3.1" stretchy="false" xref="S4.Ex4.m2.6.6.4.6.cmml">(</mo><mi id="S4.Ex4.m2.3.3.1.1.1.1" xref="S4.Ex4.m2.3.3.1.1.1.1.cmml">ρ</mi><mo id="S4.Ex4.m2.3.3.1.1.1.3.2" stretchy="false" xref="S4.Ex4.m2.6.6.4.6.cmml">)</mo></mrow></msubsup></mrow></munder></mstyle> <mrow id="S4.Ex4.m2.11.12.2.2" xref="S4.Ex4.m2.11.12.2.2.cmml"><msup id="S4.Ex4.m2.11.12.2.2.2" xref="S4.Ex4.m2.11.12.2.2.2.cmml"><mi id="S4.Ex4.m2.11.12.2.2.2.2" xref="S4.Ex4.m2.11.12.2.2.2.2.cmml">η</mi> <mi id="S4.Ex4.m2.11.12.2.2.2.3" xref="S4.Ex4.m2.11.12.2.2.2.3.cmml">t</mi></msup> <mo id="S4.Ex4.m2.11.12.2.2.1" xref="S4.Ex4.m2.11.12.2.2.1.cmml">⁢</mo> <msub id="S4.Ex4.m2.11.12.2.2.3" xref="S4.Ex4.m2.11.12.2.2.3.cmml"><mi id="S4.Ex4.m2.11.12.2.2.3.2" xref="S4.Ex4.m2.11.12.2.2.3.2.cmml">q</mi> <msub id="S4.Ex4.m2.11.12.2.2.3.3" xref="S4.Ex4.m2.11.12.2.2.3.3.cmml"><mi id="S4.Ex4.m2.11.12.2.2.3.3.2" xref="S4.Ex4.m2.11.12.2.2.3.3.2.cmml">k</mi> <mi id="S4.Ex4.m2.11.12.2.2.3.3.3" xref="S4.Ex4.m2.11.12.2.2.3.3.3.cmml">ρ</mi></msub></msub> <mo id="S4.Ex4.m2.11.12.2.2.1a" xref="S4.Ex4.m2.11.12.2.2.1.cmml">⁢</mo> <munder id="S4.Ex4.m2.11.12.2.2.4" xref="S4.Ex4.m2.11.12.2.2.4.cmml"><munder accentunder="true" id="S4.Ex4.m2.9.9" xref="S4.Ex4.m2.9.9.cmml"><mrow id="S4.Ex4.m2.9.9.4.4.4" xref="S4.Ex4.m2.9.9.4.4.4.1.cmml"><mo id="S4.Ex4.m2.9.9.4.4.4.2" mathbackground="#E6FFE6" xref="S4.Ex4.m2.9.9.4.4.4.1.cmml">(</mo><mrow id="S4.Ex4.m2.9.9.4.4.4.1" xref="S4.Ex4.m2.9.9.4.4.4.1.cmml"><mstyle displaystyle="true" id="S4.Ex4.m2.9.9.4.4.4.1.1" xref="S4.Ex4.m2.9.9.4.4.4.1.1.cmml"><munderover id="S4.Ex4.m2.9.9.4.4.4.1.1a" xref="S4.Ex4.m2.9.9.4.4.4.1.1.cmml"><mo id="S4.Ex4.m2.9.9.4.4.4.1.1.2.2" mathbackground="#E6FFE6" movablelimits="false" xref="S4.Ex4.m2.9.9.4.4.4.1.1.2.2.cmml">∏</mo> <mrow id="S4.Ex4.m2.9.9.4.4.4.1.1.2.3" xref="S4.Ex4.m2.9.9.4.4.4.1.1.2.3.cmml"><mi id="S4.Ex4.m2.9.9.4.4.4.1.1.2.3.2" mathbackground="#E6FFE6" xref="S4.Ex4.m2.9.9.4.4.4.1.1.2.3.2.cmml">s</mi> <mo id="S4.Ex4.m2.9.9.4.4.4.1.1.2.3.1" mathbackground="#E6FFE6" xref="S4.Ex4.m2.9.9.4.4.4.1.1.2.3.1.cmml">=</mo> <mn id="S4.Ex4.m2.9.9.4.4.4.1.1.2.3.3" mathbackground="#E6FFE6" xref="S4.Ex4.m2.9.9.4.4.4.1.1.2.3.3.cmml">1</mn></mrow> <mi id="S4.Ex4.m2.9.9.4.4.4.1.1.3" mathbackground="#E6FFE6" xref="S4.Ex4.m2.9.9.4.4.4.1.1.3.cmml">ρ</mi></munderover></mstyle> <msubsup id="S4.Ex4.m2.9.9.4.4.4.1.2" xref="S4.Ex4.m2.9.9.4.4.4.1.2.cmml"><mi id="S4.Ex4.m2.9.9.4.4.4.1.2.2.2" mathbackground="#E6FFE6" xref="S4.Ex4.m2.9.9.4.4.4.1.2.2.2.cmml">𝑾</mi> <mrow id="S4.Ex4.m2.8.8.3.3.3.2.2" xref="S4.Ex4.m2.8.8.3.3.3.2.3.cmml"><msub id="S4.Ex4.m2.7.7.2.2.2.1.1.1" xref="S4.Ex4.m2.7.7.2.2.2.1.1.1.cmml"><mi id="S4.Ex4.m2.7.7.2.2.2.1.1.1.2" mathbackground="#E6FFE6" xref="S4.Ex4.m2.7.7.2.2.2.1.1.1.2.cmml">k</mi> <mi id="S4.Ex4.m2.7.7.2.2.2.1.1.1.3" mathbackground="#E6FFE6" xref="S4.Ex4.m2.7.7.2.2.2.1.1.1.3.cmml">s</mi></msub><mo id="S4.Ex4.m2.8.8.3.3.3.2.2.3" mathbackground="#E6FFE6" xref="S4.Ex4.m2.8.8.3.3.3.2.3.cmml">,</mo><msub id="S4.Ex4.m2.8.8.3.3.3.2.2.2" xref="S4.Ex4.m2.8.8.3.3.3.2.2.2.cmml"><mi id="S4.Ex4.m2.8.8.3.3.3.2.2.2.2" mathbackground="#E6FFE6" xref="S4.Ex4.m2.8.8.3.3.3.2.2.2.2.cmml">k</mi> <mrow id="S4.Ex4.m2.8.8.3.3.3.2.2.2.3" xref="S4.Ex4.m2.8.8.3.3.3.2.2.2.3.cmml"><mi id="S4.Ex4.m2.8.8.3.3.3.2.2.2.3.2" mathbackground="#E6FFE6" xref="S4.Ex4.m2.8.8.3.3.3.2.2.2.3.2.cmml">s</mi> <mo id="S4.Ex4.m2.8.8.3.3.3.2.2.2.3.1" mathbackground="#E6FFE6" xref="S4.Ex4.m2.8.8.3.3.3.2.2.2.3.1.cmml">−</mo> <mn id="S4.Ex4.m2.8.8.3.3.3.2.2.2.3.3" mathbackground="#E6FFE6" xref="S4.Ex4.m2.8.8.3.3.3.2.2.2.3.3.cmml">1</mn></mrow></msub></mrow> <mrow id="S4.Ex4.m2.9.9.4.4.4.1.2.3" xref="S4.Ex4.m2.9.9.4.4.4.1.2.3.cmml"><mrow id="S4.Ex4.m2.9.9.4.4.4.1.2.3.2" xref="S4.Ex4.m2.9.9.4.4.4.1.2.3.2.cmml"><mi id="S4.Ex4.m2.9.9.4.4.4.1.2.3.2.2" mathbackground="#E6FFE6" xref="S4.Ex4.m2.9.9.4.4.4.1.2.3.2.2.cmml">t</mi> <mo id="S4.Ex4.m2.9.9.4.4.4.1.2.3.2.1" mathbackground="#E6FFE6" xref="S4.Ex4.m2.9.9.4.4.4.1.2.3.2.1.cmml">+</mo> <mi id="S4.Ex4.m2.9.9.4.4.4.1.2.3.2.3" mathbackground="#E6FFE6" xref="S4.Ex4.m2.9.9.4.4.4.1.2.3.2.3.cmml">s</mi></mrow> <mo id="S4.Ex4.m2.9.9.4.4.4.1.2.3.1" mathbackground="#E6FFE6" xref="S4.Ex4.m2.9.9.4.4.4.1.2.3.1.cmml">−</mo> <mn id="S4.Ex4.m2.9.9.4.4.4.1.2.3.3" mathbackground="#E6FFE6" xref="S4.Ex4.m2.9.9.4.4.4.1.2.3.3.cmml">1</mn></mrow></msubsup></mrow><mo id="S4.Ex4.m2.9.9.4.4.4.3" mathbackground="#E6FFE6" xref="S4.Ex4.m2.9.9.4.4.4.1.cmml">)</mo></mrow> <mo id="S4.Ex4.m2.9.9.5" xref="S4.Ex4.m2.9.9.5.cmml">⏟</mo></munder> <mtext id="S4.Ex4.m2.11.12.2.2.4.2" xref="S4.Ex4.m2.11.12.2.2.4.2a.cmml">communication graph-related term</mtext></munder> <mo id="S4.Ex4.m2.11.12.2.2.1b" lspace="0.167em" xref="S4.Ex4.m2.11.12.2.2.1.cmml">⁢</mo> <munder id="S4.Ex4.m2.11.12.2.2.5" xref="S4.Ex4.m2.11.12.2.2.5.cmml"><munder accentunder="true" id="S4.Ex4.m2.11.11" xref="S4.Ex4.m2.11.11.cmml"><mrow id="S4.Ex4.m2.11.11.3" xref="S4.Ex4.m2.11.11.3.cmml"><mrow id="S4.Ex4.m2.11.11.3.5" xref="S4.Ex4.m2.11.11.3.5.cmml"><mo id="S4.Ex4.m2.11.11.3.5.1" mathbackground="#E6E6E6" rspace="0.167em" xref="S4.Ex4.m2.11.11.3.5.1.cmml">∇</mo> <mi id="S4.Ex4.m2.11.11.3.5.2" mathbackground="#E6E6E6" xref="S4.Ex4.m2.11.11.3.5.2.cmml">L</mi></mrow> <mo id="S4.Ex4.m2.11.11.3.4" xref="S4.Ex4.m2.11.11.3.4.cmml">⁢</mo> <msup id="S4.Ex4.m2.11.11.3.3.3" xref="S4.Ex4.m2.11.11.3.3.3.cmml"><mrow id="S4.Ex4.m2.11.11.3.3.3.2.2" xref="S4.Ex4.m2.11.11.3.3.3.2.3.cmml"><mo id="S4.Ex4.m2.11.11.3.3.3.2.2.3" mathbackground="#E6E6E6" maxsize="120%" minsize="120%" xref="S4.Ex4.m2.11.11.3.3.3.2.3.cmml">(</mo><msubsup id="S4.Ex4.m2.10.10.2.2.2.1.1.1" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.cmml"><mi id="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.2" mathbackground="#E6E6E6" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.2.cmml">𝜽</mi> <msub id="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.3" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.3.cmml"><mi id="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.3.2" mathbackground="#E6E6E6" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.3.2.cmml">k</mi> <mi id="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.3.3" mathbackground="#E6E6E6" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.3.3.cmml">ρ</mi></msub> <mrow id="S4.Ex4.m2.10.10.2.2.2.1.1.1.3" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.3.cmml"><mi id="S4.Ex4.m2.10.10.2.2.2.1.1.1.3.2" mathbackground="#E6E6E6" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.3.2.cmml">t</mi> <mo id="S4.Ex4.m2.10.10.2.2.2.1.1.1.3.1" mathbackground="#E6E6E6" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.3.1.cmml">+</mo> <mi id="S4.Ex4.m2.10.10.2.2.2.1.1.1.3.3" mathbackground="#E6E6E6" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.3.3.cmml">ρ</mi></mrow></msubsup><mo id="S4.Ex4.m2.11.11.3.3.3.2.2.4" mathbackground="#E6E6E6" xref="S4.Ex4.m2.11.11.3.3.3.2.3.cmml">;</mo><msup id="S4.Ex4.m2.11.11.3.3.3.2.2.2" xref="S4.Ex4.m2.11.11.3.3.3.2.2.2.cmml"><mi id="S4.Ex4.m2.11.11.3.3.3.2.2.2.2" mathbackground="#E6E6E6" xref="S4.Ex4.m2.11.11.3.3.3.2.2.2.2.cmml">𝒛</mi> <mo id="S4.Ex4.m2.11.11.3.3.3.2.2.2.3" mathbackground="#E6E6E6" xref="S4.Ex4.m2.11.11.3.3.3.2.2.2.3.cmml">′</mo></msup><mo id="S4.Ex4.m2.11.11.3.3.3.2.2.5" mathbackground="#E6E6E6" maxsize="120%" minsize="120%" xref="S4.Ex4.m2.11.11.3.3.3.2.3.cmml">)</mo></mrow> <mo id="S4.Ex4.m2.11.11.3.3.3.4" mathbackground="#E6E6E6" xref="S4.Ex4.m2.11.11.3.3.3.4.cmml">⊤</mo></msup></mrow> <mo id="S4.Ex4.m2.11.11.4" xref="S4.Ex4.m2.11.11.4.cmml">⏟</mo></munder> <mtext id="S4.Ex4.m2.11.12.2.2.5.2" xref="S4.Ex4.m2.11.12.2.2.5.2a.cmml">test gradient</mtext></munder></mrow></mrow></mrow> <annotation-xml encoding="MathML-Content" id="S4.Ex4.m2.11b"><apply id="S4.Ex4.m2.11.12.cmml" xref="S4.Ex4.m2.11.12"><apply id="S4.Ex4.m2.11.12.1.cmml" xref="S4.Ex4.m2.11.12.1"><csymbol cd="ambiguous" id="S4.Ex4.m2.11.12.1.1.cmml" xref="S4.Ex4.m2.11.12.1">superscript</csymbol> <apply id="S4.Ex4.m2.11.12.1.2.cmml" xref="S4.Ex4.m2.11.12.1"><csymbol cd="ambiguous" id="S4.Ex4.m2.11.12.1.2.1.cmml" xref="S4.Ex4.m2.11.12.1">subscript</csymbol> <apply id="S4.Ex4.m2.11.12.1.2.3.cmml" xref="S4.Ex4.m2.11.12.1.2.3"><ci id="S4.Ex4.m2.11.12.1.2.3.2.cmml" xref="S4.Ex4.m2.11.12.1.2.3.2">𝜌</ci> <cn id="S4.Ex4.m2.11.12.1.2.3.3.cmml" type="integer" xref="S4.Ex4.m2.11.12.1.2.3.3">0</cn></apply></apply> <ci id="S4.Ex4.m2.11.12.1.3.cmml" xref="S4.Ex4.m2.11.12.1.3">𝑟</ci></apply> <apply id="S4.Ex4.m2.11.12.2.cmml" xref="S4.Ex4.m2.11.12.2"><apply id="S4.Ex4.m2.11.12.2.1.cmml" xref="S4.Ex4.m2.11.12.2.1"><csymbol cd="ambiguous" id="S4.Ex4.m2.11.12.2.1.1.cmml" xref="S4.Ex4.m2.11.12.2.1">subscript</csymbol> <apply id="S4.Ex4.m2.6.6.4.cmml" xref="S4.Ex4.m2.6.6.4"><vector id="S4.Ex4.m2.6.6.4.4.3.cmml" xref="S4.Ex4.m2.6.6.4.4.2"><apply id="S4.Ex4.m2.5.5.3.3.1.1.cmml" xref="S4.Ex4.m2.5.5.3.3.1.1"><csymbol cd="ambiguous" id="S4.Ex4.m2.5.5.3.3.1.1.1.cmml" xref="S4.Ex4.m2.5.5.3.3.1.1">subscript</csymbol> <ci id="S4.Ex4.m2.5.5.3.3.1.1.2.cmml" xref="S4.Ex4.m2.5.5.3.3.1.1.2">𝑘</ci> <cn id="S4.Ex4.m2.5.5.3.3.1.1.3.cmml" type="integer" xref="S4.Ex4.m2.5.5.3.3.1.1.3">1</cn></apply> <ci id="S4.Ex4.m2.4.4.2.2.cmml" xref="S4.Ex4.m2.4.4.2.2">…</ci> <apply id="S4.Ex4.m2.6.6.4.4.2.2.cmml" xref="S4.Ex4.m2.6.6.4.4.2.2"><csymbol cd="ambiguous" id="S4.Ex4.m2.6.6.4.4.2.2.1.cmml" xref="S4.Ex4.m2.6.6.4.4.2.2">subscript</csymbol> <ci id="S4.Ex4.m2.6.6.4.4.2.2.2.cmml" xref="S4.Ex4.m2.6.6.4.4.2.2.2">𝑘</ci> <ci id="S4.Ex4.m2.6.6.4.4.2.2.3.cmml" xref="S4.Ex4.m2.6.6.4.4.2.2.3">𝜌</ci></apply></vector> <apply id="S4.Ex4.m2.6.6.4.6.cmml" xref="S4.Ex4.m2.6.6.4.6"><csymbol cd="ambiguous" id="S4.Ex4.m2.6.6.4.6.1.cmml" xref="S4.Ex4.m2.6.6.4.6">superscript</csymbol> <apply id="S4.Ex4.m2.6.6.4.6.2.cmml" xref="S4.Ex4.m2.6.6.4.6"><csymbol cd="ambiguous" id="S4.Ex4.m2.6.6.4.6.2.1.cmml" xref="S4.Ex4.m2.6.6.4.6">subscript</csymbol> <ci id="S4.Ex4.m2.6.6.4.6.2.2.cmml" xref="S4.Ex4.m2.6.6.4.6.2.2">𝑃</ci> <ci id="S4.Ex4.m2.6.6.4.6.2.3.cmml" xref="S4.Ex4.m2.6.6.4.6.2.3">𝑗</ci></apply> <ci id="S4.Ex4.m2.3.3.1.1.1.1.cmml" xref="S4.Ex4.m2.3.3.1.1.1.1">𝜌</ci></apply></apply></apply> <apply id="S4.Ex4.m2.11.12.2.2.cmml" xref="S4.Ex4.m2.11.12.2.2"><apply id="S4.Ex4.m2.11.12.2.2.2.cmml" xref="S4.Ex4.m2.11.12.2.2.2"><csymbol cd="ambiguous" id="S4.Ex4.m2.11.12.2.2.2.1.cmml" xref="S4.Ex4.m2.11.12.2.2.2">superscript</csymbol> <ci id="S4.Ex4.m2.11.12.2.2.2.2.cmml" xref="S4.Ex4.m2.11.12.2.2.2.2">𝜂</ci> <ci id="S4.Ex4.m2.11.12.2.2.2.3.cmml" xref="S4.Ex4.m2.11.12.2.2.2.3">𝑡</ci></apply> <apply id="S4.Ex4.m2.11.12.2.2.3.cmml" xref="S4.Ex4.m2.11.12.2.2.3"><csymbol cd="ambiguous" id="S4.Ex4.m2.11.12.2.2.3.1.cmml" xref="S4.Ex4.m2.11.12.2.2.3">subscript</csymbol> <ci id="S4.Ex4.m2.11.12.2.2.3.2.cmml" xref="S4.Ex4.m2.11.12.2.2.3.2">𝑞</ci> <apply id="S4.Ex4.m2.11.12.2.2.3.3.cmml" xref="S4.Ex4.m2.11.12.2.2.3.3"><csymbol cd="ambiguous" id="S4.Ex4.m2.11.12.2.2.3.3.1.cmml" xref="S4.Ex4.m2.11.12.2.2.3.3">subscript</csymbol> <ci id="S4.Ex4.m2.11.12.2.2.3.3.2.cmml" xref="S4.Ex4.m2.11.12.2.2.3.3.2">𝑘</ci> <ci id="S4.Ex4.m2.11.12.2.2.3.3.3.cmml" xref="S4.Ex4.m2.11.12.2.2.3.3.3">𝜌</ci></apply></apply> <apply id="S4.Ex4.m2.11.12.2.2.4.cmml" xref="S4.Ex4.m2.11.12.2.2.4"><csymbol cd="ambiguous" id="S4.Ex4.m2.11.12.2.2.4.1.cmml" xref="S4.Ex4.m2.11.12.2.2.4">subscript</csymbol> <apply id="S4.Ex4.m2.9.9.cmml" xref="S4.Ex4.m2.9.9"><ci id="S4.Ex4.m2.9.9.5.cmml" xref="S4.Ex4.m2.9.9.5">⏟</ci> <apply id="S4.Ex4.m2.9.9.4.4.4.1.cmml" xref="S4.Ex4.m2.9.9.4.4.4"><apply id="S4.Ex4.m2.9.9.4.4.4.1.1.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.1"><csymbol cd="ambiguous" id="S4.Ex4.m2.9.9.4.4.4.1.1.1.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.1">superscript</csymbol> <apply id="S4.Ex4.m2.9.9.4.4.4.1.1.2.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.1"><csymbol cd="ambiguous" id="S4.Ex4.m2.9.9.4.4.4.1.1.2.1.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.1">subscript</csymbol> <csymbol cd="latexml" id="S4.Ex4.m2.9.9.4.4.4.1.1.2.2.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.1.2.2">product</csymbol> <apply id="S4.Ex4.m2.9.9.4.4.4.1.1.2.3.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.1.2.3"><ci id="S4.Ex4.m2.9.9.4.4.4.1.1.2.3.2.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.1.2.3.2">𝑠</ci> <cn id="S4.Ex4.m2.9.9.4.4.4.1.1.2.3.3.cmml" type="integer" xref="S4.Ex4.m2.9.9.4.4.4.1.1.2.3.3">1</cn></apply></apply> <ci id="S4.Ex4.m2.9.9.4.4.4.1.1.3.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.1.3">𝜌</ci></apply> <apply id="S4.Ex4.m2.9.9.4.4.4.1.2.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.2"><csymbol cd="ambiguous" id="S4.Ex4.m2.9.9.4.4.4.1.2.1.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.2">superscript</csymbol> <apply id="S4.Ex4.m2.9.9.4.4.4.1.2.2.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.2"><csymbol cd="ambiguous" id="S4.Ex4.m2.9.9.4.4.4.1.2.2.1.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.2">subscript</csymbol> <ci id="S4.Ex4.m2.9.9.4.4.4.1.2.2.2.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.2.2.2">𝑾</ci> <list id="S4.Ex4.m2.8.8.3.3.3.2.3.cmml" xref="S4.Ex4.m2.8.8.3.3.3.2.2"><apply id="S4.Ex4.m2.7.7.2.2.2.1.1.1.cmml" xref="S4.Ex4.m2.7.7.2.2.2.1.1.1"><csymbol cd="ambiguous" id="S4.Ex4.m2.7.7.2.2.2.1.1.1.1.cmml" xref="S4.Ex4.m2.7.7.2.2.2.1.1.1">subscript</csymbol> <ci id="S4.Ex4.m2.7.7.2.2.2.1.1.1.2.cmml" xref="S4.Ex4.m2.7.7.2.2.2.1.1.1.2">𝑘</ci> <ci id="S4.Ex4.m2.7.7.2.2.2.1.1.1.3.cmml" xref="S4.Ex4.m2.7.7.2.2.2.1.1.1.3">𝑠</ci></apply> <apply id="S4.Ex4.m2.8.8.3.3.3.2.2.2.cmml" xref="S4.Ex4.m2.8.8.3.3.3.2.2.2"><csymbol cd="ambiguous" id="S4.Ex4.m2.8.8.3.3.3.2.2.2.1.cmml" xref="S4.Ex4.m2.8.8.3.3.3.2.2.2">subscript</csymbol> <ci id="S4.Ex4.m2.8.8.3.3.3.2.2.2.2.cmml" xref="S4.Ex4.m2.8.8.3.3.3.2.2.2.2">𝑘</ci> <apply id="S4.Ex4.m2.8.8.3.3.3.2.2.2.3.cmml" xref="S4.Ex4.m2.8.8.3.3.3.2.2.2.3"><ci id="S4.Ex4.m2.8.8.3.3.3.2.2.2.3.2.cmml" xref="S4.Ex4.m2.8.8.3.3.3.2.2.2.3.2">𝑠</ci> <cn id="S4.Ex4.m2.8.8.3.3.3.2.2.2.3.3.cmml" type="integer" xref="S4.Ex4.m2.8.8.3.3.3.2.2.2.3.3">1</cn></apply></apply></list></apply> <apply id="S4.Ex4.m2.9.9.4.4.4.1.2.3.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.2.3"><apply id="S4.Ex4.m2.9.9.4.4.4.1.2.3.2.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.2.3.2"><ci id="S4.Ex4.m2.9.9.4.4.4.1.2.3.2.2.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.2.3.2.2">𝑡</ci> <ci id="S4.Ex4.m2.9.9.4.4.4.1.2.3.2.3.cmml" xref="S4.Ex4.m2.9.9.4.4.4.1.2.3.2.3">𝑠</ci></apply> <cn id="S4.Ex4.m2.9.9.4.4.4.1.2.3.3.cmml" type="integer" xref="S4.Ex4.m2.9.9.4.4.4.1.2.3.3">1</cn></apply></apply></apply></apply> <ci id="S4.Ex4.m2.11.12.2.2.4.2a.cmml" xref="S4.Ex4.m2.11.12.2.2.4.2"><mtext id="S4.Ex4.m2.11.12.2.2.4.2.cmml" mathsize="70%" xref="S4.Ex4.m2.11.12.2.2.4.2">communication graph-related term</mtext></ci></apply> <apply id="S4.Ex4.m2.11.12.2.2.5.cmml" xref="S4.Ex4.m2.11.12.2.2.5"><csymbol cd="ambiguous" id="S4.Ex4.m2.11.12.2.2.5.1.cmml" xref="S4.Ex4.m2.11.12.2.2.5">subscript</csymbol> <apply id="S4.Ex4.m2.11.11.cmml" xref="S4.Ex4.m2.11.11"><ci id="S4.Ex4.m2.11.11.4.cmml" xref="S4.Ex4.m2.11.11.4">⏟</ci> <apply id="S4.Ex4.m2.11.11.3.cmml" xref="S4.Ex4.m2.11.11.3"><apply id="S4.Ex4.m2.11.11.3.5.cmml" xref="S4.Ex4.m2.11.11.3.5"><ci id="S4.Ex4.m2.11.11.3.5.1.cmml" xref="S4.Ex4.m2.11.11.3.5.1">∇</ci> <ci id="S4.Ex4.m2.11.11.3.5.2.cmml" xref="S4.Ex4.m2.11.11.3.5.2">𝐿</ci></apply> <apply id="S4.Ex4.m2.11.11.3.3.3.cmml" xref="S4.Ex4.m2.11.11.3.3.3"><csymbol cd="ambiguous" id="S4.Ex4.m2.11.11.3.3.3.3.cmml" xref="S4.Ex4.m2.11.11.3.3.3">superscript</csymbol> <list id="S4.Ex4.m2.11.11.3.3.3.2.3.cmml" xref="S4.Ex4.m2.11.11.3.3.3.2.2"><apply id="S4.Ex4.m2.10.10.2.2.2.1.1.1.cmml" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1"><csymbol cd="ambiguous" id="S4.Ex4.m2.10.10.2.2.2.1.1.1.1.cmml" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1">superscript</csymbol> <apply id="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.cmml" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1"><csymbol cd="ambiguous" id="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.1.cmml" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1">subscript</csymbol> <ci id="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.2.cmml" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.2">𝜽</ci> <apply id="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.3.cmml" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.3"><csymbol cd="ambiguous" id="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.3.1.cmml" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.3">subscript</csymbol> <ci id="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.3.2.cmml" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.3.2">𝑘</ci> <ci id="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.3.3.cmml" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.2.3.3">𝜌</ci></apply></apply> <apply id="S4.Ex4.m2.10.10.2.2.2.1.1.1.3.cmml" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.3"><ci id="S4.Ex4.m2.10.10.2.2.2.1.1.1.3.2.cmml" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.3.2">𝑡</ci> <ci id="S4.Ex4.m2.10.10.2.2.2.1.1.1.3.3.cmml" xref="S4.Ex4.m2.10.10.2.2.2.1.1.1.3.3">𝜌</ci></apply></apply> <apply id="S4.Ex4.m2.11.11.3.3.3.2.2.2.cmml" xref="S4.Ex4.m2.11.11.3.3.3.2.2.2"><csymbol cd="ambiguous" id="S4.Ex4.m2.11.11.3.3.3.2.2.2.1.cmml" xref="S4.Ex4.m2.11.11.3.3.3.2.2.2">superscript</csymbol> <ci id="S4.Ex4.m2.11.11.3.3.3.2.2.2.2.cmml" xref="S4.Ex4.m2.11.11.3.3.3.2.2.2.2">𝒛</ci> <ci id="S4.Ex4.m2.11.11.3.3.3.2.2.2.3.cmml" xref="S4.Ex4.m2.11.11.3.3.3.2.2.2.3">′</ci></apply></list> <csymbol cd="latexml" id="S4.Ex4.m2.11.11.3.3.3.4.cmml" xref="S4.Ex4.m2.11.11.3.3.3.4">top</csymbol></apply></apply></apply> <ci id="S4.Ex4.m2.11.12.2.2.5.2a.cmml" xref="S4.Ex4.m2.11.12.2.2.5.2"><mtext id="S4.Ex4.m2.11.12.2.2.5.2.cmml" mathsize="70%" xref="S4.Ex4.m2.11.12.2.2.5.2">test gradient</mtext></ci></apply></apply></apply></apply></annotation-xml> <annotation encoding="application/x-tex" id="S4.Ex4.m2.11c">\displaystyle\sum_{\rho=0}^{r}\sum_{(k_{1},\dots,k_{\rho})\in P_{j}^{(\rho)}}% \eta^{t}q_{k_{\rho}}\underbrace{\hbox{\pagecolor{green!10}$\displaystyle\left(% \prod_{s=1}^{\rho}\bm{W}_{k_{s},k_{s-1}}^{t+s-1}\right)$}}_{{\text{% communication graph-related term}}}\underbrace{\hbox{\pagecolor{Mulberry!10}$% \displaystyle\vphantom{\prod_{s=1}^{\rho}}\nabla L\bigl{(}\bm{\theta}_{k_{\rho% }}^{t+\rho};\bm{z}^{\prime}\bigr{)}^{\top}$}}_{{\text{test gradient}}}</annotation> <annotation encoding="application/x-llamapun" id="S4.Ex4.m2.11d">∑ start_POSTSUBSCRIPT italic_ρ = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_r end_POSTSUPERSCRIPT ∑ start_POSTSUBSCRIPT ( italic_k start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT, …, italic_k start_POSTSUBSCRIPT italic_ρ end_POSTSUBSCRIPT ) ∈ italic_P start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_ρ ) end_POSTSUPERSCRIPT end_POSTSUBSCRIPT italic_η start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT italic_q start_POSTSUBSCRIPT italic_k start_POSTSUBSCRIPT italic_ρ end_POSTSUBSCRIPT end_POSTSUBSCRIPT under⏟ start_ARG ( ∏ start_POSTSUBSCRIPT italic_s = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_ρ end_POSTSUPERSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_k start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT, italic_k start_POSTSUBSCRIPT italic_s - 1 end_POSTSUBSCRIPT end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t + italic_s - 1 end_POSTSUPERSCRIPT ) end_ARG start_POSTSUBSCRIPT communication graph-related term end_POSTSUBSCRIPT under⏟ start_ARG ∇ italic_L ( bold_italic_θ start_POSTSUBSCRIPT italic_k start_POSTSUBSCRIPT italic_ρ end_POSTSUBSCRIPT end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t + italic_ρ end_POSTSUPERSCRIPT; bold_italic_z start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT end_ARG start_POSTSUBSCRIPT test gradient end_POSTSUBSCRIPT</annotation></semantics></math> </span><span id="S4.Ex5"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\displaystyle\times\underbrace{\hbox{\pagecolor{myorange!30}$\displaystyle%
\left(\prod_{s=2}^{\rho}\left(\bm{I}-\eta^{t+s-1}\bm{H}(\bm{\theta}_{k_{s}}^{t%
+s-1};\bm{z}_{k_{s}}^{t+s-1})\right)\right)$}}_{{\text{curvature-related term}%
}}\underbrace{\hbox{\pagecolor{cyan!10}$\displaystyle\vphantom{\prod_{s=2}^{%
\rho}}\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})$}}_{{\text{optimization-%
related term}}}."><semantics id="S4.Ex5.m1.6a"><mrow id="S4.Ex5.m1.6.6.1" xref="S4.Ex5.m1.6.6.1.1.cmml"><mrow id="S4.Ex5.m1.6.6.1.1" xref="S4.Ex5.m1.6.6.1.1.cmml"><mo id="S4.Ex5.m1.6.6.1.1.1" lspace="0.222em" rspace="0.222em" xref="S4.Ex5.m1.6.6.1.1.1.cmml">×</mo> <mrow id="S4.Ex5.m1.6.6.1.1.3" xref="S4.Ex5.m1.6.6.1.1.3.cmml"><munder id="S4.Ex5.m1.6.6.1.1.3.2" xref="S4.Ex5.m1.6.6.1.1.3.2.cmml"><munder accentunder="true" id="S4.Ex5.m1.3.3" xref="S4.Ex5.m1.3.3.cmml"><mrow id="S4.Ex5.m1.3.3.2.2.2" xref="S4.Ex5.m1.3.3.2.2.2.1.cmml"><mo id="S4.Ex5.m1.3.3.2.2.2.2" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.cmml">(</mo><mrow id="S4.Ex5.m1.3.3.2.2.2.1" xref="S4.Ex5.m1.3.3.2.2.2.1.cmml"><mstyle displaystyle="true" id="S4.Ex5.m1.3.3.2.2.2.1.2" xref="S4.Ex5.m1.3.3.2.2.2.1.2.cmml"><munderover id="S4.Ex5.m1.3.3.2.2.2.1.2a" xref="S4.Ex5.m1.3.3.2.2.2.1.2.cmml"><mo id="S4.Ex5.m1.3.3.2.2.2.1.2.2.2" mathbackground="#FFF5D9" movablelimits="false" xref="S4.Ex5.m1.3.3.2.2.2.1.2.2.2.cmml">∏</mo> <mrow id="S4.Ex5.m1.3.3.2.2.2.1.2.2.3" xref="S4.Ex5.m1.3.3.2.2.2.1.2.2.3.cmml"><mi id="S4.Ex5.m1.3.3.2.2.2.1.2.2.3.2" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.2.2.3.2.cmml">s</mi> <mo id="S4.Ex5.m1.3.3.2.2.2.1.2.2.3.1" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.2.2.3.1.cmml">=</mo> <mn id="S4.Ex5.m1.3.3.2.2.2.1.2.2.3.3" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.2.2.3.3.cmml">2</mn></mrow> <mi id="S4.Ex5.m1.3.3.2.2.2.1.2.3" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.2.3.cmml">ρ</mi></munderover></mstyle> <mrow id="S4.Ex5.m1.3.3.2.2.2.1.1.1" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.cmml"><mo id="S4.Ex5.m1.3.3.2.2.2.1.1.1.2" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.cmml">(</mo><mrow id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.cmml"><mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.4" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.4.cmml">𝑰</mi> <mo id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.3" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.3.cmml">−</mo> <mrow id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.cmml"><msup id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.cmml"><mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.2" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.2.cmml">η</mi> <mrow id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.cmml"><mrow id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.2" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.2.cmml"><mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.2.2" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.2.2.cmml">t</mi> <mo id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.2.1" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.2.1.cmml">+</mo> <mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.2.3" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.2.3.cmml">s</mi></mrow> <mo id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.1" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.1.cmml">−</mo> <mn id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.3" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.3.cmml">1</mn></mrow></msup> <mo id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.3" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.3.cmml">⁢</mo> <mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.5" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.5.cmml">𝑯</mi> <mo id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.3a" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.3.cmml">⁢</mo> <mrow id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.3.cmml"><mo id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.3" mathbackground="#FFF5D9" stretchy="false" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.3.cmml">(</mo><msubsup id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.cmml"><mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.2" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.2.cmml">𝜽</mi> <msub id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.3" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.3.cmml"><mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.3.2" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.3.2.cmml">k</mi> <mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.3.3" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.3.3.cmml">s</mi></msub> <mrow id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.cmml"><mrow id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.2" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.2.cmml"><mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.2.2" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.2.2.cmml">t</mi> <mo id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.2.1" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.2.1.cmml">+</mo> <mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.2.3" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.2.3.cmml">s</mi></mrow> <mo id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.1" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.1.cmml">−</mo> <mn id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.3" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.3.cmml">1</mn></mrow></msubsup><mo id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.4" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.3.cmml">;</mo><msubsup id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.cmml"><mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.2" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.2.cmml">𝒛</mi> <msub id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.3" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.3.cmml"><mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.3.2" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.3.2.cmml">k</mi> <mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.3.3" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.3.3.cmml">s</mi></msub> <mrow id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.cmml"><mrow id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.2" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.2.cmml"><mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.2.2" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.2.2.cmml">t</mi> <mo id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.2.1" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.2.1.cmml">+</mo> <mi id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.2.3" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.2.3.cmml">s</mi></mrow> <mo id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.1" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.1.cmml">−</mo> <mn id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.3" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.3.cmml">1</mn></mrow></msubsup><mo id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.5" mathbackground="#FFF5D9" stretchy="false" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.3.cmml">)</mo></mrow></mrow></mrow><mo id="S4.Ex5.m1.3.3.2.2.2.1.1.1.3" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.cmml">)</mo></mrow></mrow><mo id="S4.Ex5.m1.3.3.2.2.2.3" mathbackground="#FFF5D9" xref="S4.Ex5.m1.3.3.2.2.2.1.cmml">)</mo></mrow> <mo id="S4.Ex5.m1.3.3.3" xref="S4.Ex5.m1.3.3.3.cmml">⏟</mo></munder> <mtext id="S4.Ex5.m1.6.6.1.1.3.2.2" xref="S4.Ex5.m1.6.6.1.1.3.2.2a.cmml">curvature-related term</mtext></munder> <mo id="S4.Ex5.m1.6.6.1.1.3.1" xref="S4.Ex5.m1.6.6.1.1.3.1.cmml">⁢</mo> <munder id="S4.Ex5.m1.6.6.1.1.3.3" xref="S4.Ex5.m1.6.6.1.1.3.3.cmml"><munder accentunder="true" id="S4.Ex5.m1.5.5" xref="S4.Ex5.m1.5.5.cmml"><mrow id="S4.Ex5.m1.5.5.3" xref="S4.Ex5.m1.5.5.3.cmml"><msub id="S4.Ex5.m1.5.5.3.5" xref="S4.Ex5.m1.5.5.3.5.cmml"><mi id="S4.Ex5.m1.5.5.3.5.2" mathbackground="#E6FFFF" mathvariant="normal" xref="S4.Ex5.m1.5.5.3.5.2.cmml">Δ</mi> <mi id="S4.Ex5.m1.5.5.3.5.3" mathbackground="#E6FFFF" xref="S4.Ex5.m1.5.5.3.5.3.cmml">j</mi></msub> <mo id="S4.Ex5.m1.5.5.3.4" xref="S4.Ex5.m1.5.5.3.4.cmml">⁢</mo> <mrow id="S4.Ex5.m1.5.5.3.3.3.2" xref="S4.Ex5.m1.5.5.3.3.3.3.cmml"><mo id="S4.Ex5.m1.5.5.3.3.3.2.3" mathbackground="#E6FFFF" stretchy="false" xref="S4.Ex5.m1.5.5.3.3.3.3.cmml">(</mo><msubsup id="S4.Ex5.m1.4.4.2.2.2.1.1" xref="S4.Ex5.m1.4.4.2.2.2.1.1.cmml"><mi id="S4.Ex5.m1.4.4.2.2.2.1.1.2.2" mathbackground="#E6FFFF" xref="S4.Ex5.m1.4.4.2.2.2.1.1.2.2.cmml">𝜽</mi> <mi id="S4.Ex5.m1.4.4.2.2.2.1.1.2.3" mathbackground="#E6FFFF" xref="S4.Ex5.m1.4.4.2.2.2.1.1.2.3.cmml">j</mi> <mi id="S4.Ex5.m1.4.4.2.2.2.1.1.3" mathbackground="#E6FFFF" xref="S4.Ex5.m1.4.4.2.2.2.1.1.3.cmml">t</mi></msubsup><mo id="S4.Ex5.m1.5.5.3.3.3.2.4" mathbackground="#E6FFFF" xref="S4.Ex5.m1.5.5.3.3.3.3.cmml">,</mo><msubsup id="S4.Ex5.m1.5.5.3.3.3.2.2" xref="S4.Ex5.m1.5.5.3.3.3.2.2.cmml"><mi id="S4.Ex5.m1.5.5.3.3.3.2.2.2.2" mathbackground="#E6FFFF" xref="S4.Ex5.m1.5.5.3.3.3.2.2.2.2.cmml">𝒛</mi> <mi id="S4.Ex5.m1.5.5.3.3.3.2.2.2.3" mathbackground="#E6FFFF" xref="S4.Ex5.m1.5.5.3.3.3.2.2.2.3.cmml">j</mi> <mi id="S4.Ex5.m1.5.5.3.3.3.2.2.3" mathbackground="#E6FFFF" xref="S4.Ex5.m1.5.5.3.3.3.2.2.3.cmml">t</mi></msubsup><mo id="S4.Ex5.m1.5.5.3.3.3.2.5" mathbackground="#E6FFFF" stretchy="false" xref="S4.Ex5.m1.5.5.3.3.3.3.cmml">)</mo></mrow></mrow> <mo id="S4.Ex5.m1.5.5.4" xref="S4.Ex5.m1.5.5.4.cmml">⏟</mo></munder> <mtext id="S4.Ex5.m1.6.6.1.1.3.3.2" xref="S4.Ex5.m1.6.6.1.1.3.3.2a.cmml">optimization-related term</mtext></munder></mrow></mrow><mo id="S4.Ex5.m1.6.6.1.2" lspace="0em" xref="S4.Ex5.m1.6.6.1.1.cmml">.</mo></mrow><annotation-xml encoding="MathML-Content" id="S4.Ex5.m1.6b"><apply id="S4.Ex5.m1.6.6.1.1.cmml" xref="S4.Ex5.m1.6.6.1"><csymbol cd="latexml" id="S4.Ex5.m1.6.6.1.1.2.cmml" xref="S4.Ex5.m1.6.6.1.1.2">absent</csymbol> <apply id="S4.Ex5.m1.6.6.1.1.3.cmml" xref="S4.Ex5.m1.6.6.1.1.3"><apply id="S4.Ex5.m1.6.6.1.1.3.2.cmml" xref="S4.Ex5.m1.6.6.1.1.3.2"><csymbol cd="ambiguous" id="S4.Ex5.m1.6.6.1.1.3.2.1.cmml" xref="S4.Ex5.m1.6.6.1.1.3.2">subscript</csymbol> <apply id="S4.Ex5.m1.3.3.cmml" xref="S4.Ex5.m1.3.3"><ci id="S4.Ex5.m1.3.3.3.cmml" xref="S4.Ex5.m1.3.3.3">⏟</ci> <apply id="S4.Ex5.m1.3.3.2.2.2.1.cmml" xref="S4.Ex5.m1.3.3.2.2.2"><apply id="S4.Ex5.m1.3.3.2.2.2.1.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.2"><csymbol cd="ambiguous" id="S4.Ex5.m1.3.3.2.2.2.1.2.1.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.2">superscript</csymbol> <apply id="S4.Ex5.m1.3.3.2.2.2.1.2.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.2"><csymbol cd="ambiguous" id="S4.Ex5.m1.3.3.2.2.2.1.2.2.1.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.2">subscript</csymbol> <csymbol cd="latexml" id="S4.Ex5.m1.3.3.2.2.2.1.2.2.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.2.2.2">product</csymbol> <apply id="S4.Ex5.m1.3.3.2.2.2.1.2.2.3.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.2.2.3"><ci id="S4.Ex5.m1.3.3.2.2.2.1.2.2.3.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.2.2.3.2">𝑠</ci> <cn id="S4.Ex5.m1.3.3.2.2.2.1.2.2.3.3.cmml" type="integer" xref="S4.Ex5.m1.3.3.2.2.2.1.2.2.3.3">2</cn></apply></apply> <ci id="S4.Ex5.m1.3.3.2.2.2.1.2.3.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.2.3">𝜌</ci></apply> <apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1"><ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.4.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.4">𝑰</ci> <apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2"><apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4"><csymbol cd="ambiguous" id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.1.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4">superscript</csymbol> <ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.2">𝜂</ci> <apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3"><apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.2"><ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.2.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.2.2">𝑡</ci> <ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.2.3.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.2.3">𝑠</ci></apply> <cn id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.3.cmml" type="integer" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.4.3.3">1</cn></apply></apply> <ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.5.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.5">𝑯</ci> <list id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.3.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2"><apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1"><csymbol cd="ambiguous" id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.1.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1">superscript</csymbol> <apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1"><csymbol cd="ambiguous" id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.1.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1">subscript</csymbol> <ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.2">𝜽</ci> <apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.3.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.3"><csymbol cd="ambiguous" id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.3.1.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.3">subscript</csymbol> <ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.3.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.3.2">𝑘</ci> <ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.3.3.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.2.3.3">𝑠</ci></apply></apply> <apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3"><apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.2"><ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.2.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.2.2">𝑡</ci> <ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.2.3.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.2.3">𝑠</ci></apply> <cn id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.3.cmml" type="integer" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.1.1.1.1.3.3">1</cn></apply></apply> <apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2"><csymbol cd="ambiguous" id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.1.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2">superscript</csymbol> <apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2"><csymbol cd="ambiguous" id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.1.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2">subscript</csymbol> <ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.2">𝒛</ci> <apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.3.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.3"><csymbol cd="ambiguous" id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.3.1.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.3">subscript</csymbol> <ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.3.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.3.2">𝑘</ci> <ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.3.3.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.2.3.3">𝑠</ci></apply></apply> <apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3"><apply id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.2"><ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.2.2.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.2.2">𝑡</ci> <ci id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.2.3.cmml" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.2.3">𝑠</ci></apply> <cn id="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.3.cmml" type="integer" xref="S4.Ex5.m1.3.3.2.2.2.1.1.1.1.2.2.2.2.3.3">1</cn></apply></apply></list></apply></apply></apply></apply> <ci id="S4.Ex5.m1.6.6.1.1.3.2.2a.cmml" xref="S4.Ex5.m1.6.6.1.1.3.2.2"><mtext id="S4.Ex5.m1.6.6.1.1.3.2.2.cmml" mathsize="70%" xref="S4.Ex5.m1.6.6.1.1.3.2.2">curvature-related term</mtext></ci></apply> <apply id="S4.Ex5.m1.6.6.1.1.3.3.cmml" xref="S4.Ex5.m1.6.6.1.1.3.3"><csymbol cd="ambiguous" id="S4.Ex5.m1.6.6.1.1.3.3.1.cmml" xref="S4.Ex5.m1.6.6.1.1.3.3">subscript</csymbol> <apply id="S4.Ex5.m1.5.5.cmml" xref="S4.Ex5.m1.5.5"><ci id="S4.Ex5.m1.5.5.4.cmml" xref="S4.Ex5.m1.5.5.4">⏟</ci> <apply id="S4.Ex5.m1.5.5.3.cmml" xref="S4.Ex5.m1.5.5.3"><apply id="S4.Ex5.m1.5.5.3.5.cmml" xref="S4.Ex5.m1.5.5.3.5"><csymbol cd="ambiguous" id="S4.Ex5.m1.5.5.3.5.1.cmml" xref="S4.Ex5.m1.5.5.3.5">subscript</csymbol> <ci id="S4.Ex5.m1.5.5.3.5.2.cmml" xref="S4.Ex5.m1.5.5.3.5.2">Δ</ci> <ci id="S4.Ex5.m1.5.5.3.5.3.cmml" xref="S4.Ex5.m1.5.5.3.5.3">𝑗</ci></apply> <interval closure="open" id="S4.Ex5.m1.5.5.3.3.3.3.cmml" xref="S4.Ex5.m1.5.5.3.3.3.2"><apply id="S4.Ex5.m1.4.4.2.2.2.1.1.cmml" xref="S4.Ex5.m1.4.4.2.2.2.1.1"><csymbol cd="ambiguous" id="S4.Ex5.m1.4.4.2.2.2.1.1.1.cmml" xref="S4.Ex5.m1.4.4.2.2.2.1.1">superscript</csymbol> <apply id="S4.Ex5.m1.4.4.2.2.2.1.1.2.cmml" xref="S4.Ex5.m1.4.4.2.2.2.1.1"><csymbol cd="ambiguous" id="S4.Ex5.m1.4.4.2.2.2.1.1.2.1.cmml" xref="S4.Ex5.m1.4.4.2.2.2.1.1">subscript</csymbol> <ci id="S4.Ex5.m1.4.4.2.2.2.1.1.2.2.cmml" xref="S4.Ex5.m1.4.4.2.2.2.1.1.2.2">𝜽</ci> <ci id="S4.Ex5.m1.4.4.2.2.2.1.1.2.3.cmml" xref="S4.Ex5.m1.4.4.2.2.2.1.1.2.3">𝑗</ci></apply> <ci id="S4.Ex5.m1.4.4.2.2.2.1.1.3.cmml" xref="S4.Ex5.m1.4.4.2.2.2.1.1.3">𝑡</ci></apply> <apply id="S4.Ex5.m1.5.5.3.3.3.2.2.cmml" xref="S4.Ex5.m1.5.5.3.3.3.2.2"><csymbol cd="ambiguous" id="S4.Ex5.m1.5.5.3.3.3.2.2.1.cmml" xref="S4.Ex5.m1.5.5.3.3.3.2.2">superscript</csymbol> <apply id="S4.Ex5.m1.5.5.3.3.3.2.2.2.cmml" xref="S4.Ex5.m1.5.5.3.3.3.2.2"><csymbol cd="ambiguous" id="S4.Ex5.m1.5.5.3.3.3.2.2.2.1.cmml" xref="S4.Ex5.m1.5.5.3.3.3.2.2">subscript</csymbol> <ci id="S4.Ex5.m1.5.5.3.3.3.2.2.2.2.cmml" xref="S4.Ex5.m1.5.5.3.3.3.2.2.2.2">𝒛</ci> <ci id="S4.Ex5.m1.5.5.3.3.3.2.2.2.3.cmml" xref="S4.Ex5.m1.5.5.3.3.3.2.2.2.3">𝑗</ci></apply> <ci id="S4.Ex5.m1.5.5.3.3.3.2.2.3.cmml" xref="S4.Ex5.m1.5.5.3.3.3.2.2.3">𝑡</ci></apply></interval></apply></apply> <ci id="S4.Ex5.m1.6.6.1.1.3.3.2a.cmml" xref="S4.Ex5.m1.6.6.1.1.3.3.2"><mtext id="S4.Ex5.m1.6.6.1.1.3.3.2.cmml" mathsize="70%" xref="S4.Ex5.m1.6.6.1.1.3.3.2">optimization-related term</mtext></ci></apply></apply></apply></annotation-xml> <annotation encoding="application/x-tex" id="S4.Ex5.m1.6c">\displaystyle\times\underbrace{\hbox{\pagecolor{myorange!30}$\displaystyle% \left(\prod_{s=2}^{\rho}\left(\bm{I}-\eta^{t+s-1}\bm{H}(\bm{\theta}_{k_{s}}^{t% +s-1};\bm{z}_{k_{s}}^{t+s-1})\right)\right)$}}_{{\text{curvature-related term}% }}\underbrace{\hbox{\pagecolor{cyan!10}$\displaystyle\vphantom{\prod_{s=2}^{% \rho}}\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})$}}_{{\text{optimization-% related term}}}.</annotation><annotation encoding="application/x-llamapun" id="S4.Ex5.m1.6d">× under⏟ start_ARG ( ∏ start_POSTSUBSCRIPT italic_s = 2 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_ρ end_POSTSUPERSCRIPT ( bold_italic_I - italic_η start_POSTSUPERSCRIPT italic_t + italic_s - 1 end_POSTSUPERSCRIPT bold_italic_H ( bold_italic_θ start_POSTSUBSCRIPT italic_k start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t + italic_s - 1 end_POSTSUPERSCRIPT; bold_italic_z start_POSTSUBSCRIPT italic_k start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t + italic_s - 1 end_POSTSUPERSCRIPT ) ) ) end_ARG start_POSTSUBSCRIPT curvature-related term end_POSTSUBSCRIPT under⏟ start_ARG roman_Δ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ( bold_italic_θ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, bold_italic_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT ) end_ARG start_POSTSUBSCRIPT optimization-related term end_POSTSUBSCRIPT.</annotation></semantics></math></span></span> <span id="Thmtheorem1.p1.15">where <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})=\mathcal{O}_{j}(\bm{\theta}_{j}%
^{t},\bm{z}_{j}^{t})-\bm{\theta}_{j}^{t}"><semantics id="Thmtheorem1.p1.3.m1.4a"><mrow id="Thmtheorem1.p1.3.m1.4.4" xref="Thmtheorem1.p1.3.m1.4.4.cmml"><mrow id="Thmtheorem1.p1.3.m1.2.2.2" xref="Thmtheorem1.p1.3.m1.2.2.2.cmml"><msub id="Thmtheorem1.p1.3.m1.2.2.2.4" xref="Thmtheorem1.p1.3.m1.2.2.2.4.cmml"><mi id="Thmtheorem1.p1.3.m1.2.2.2.4.2" mathvariant="normal" xref="Thmtheorem1.p1.3.m1.2.2.2.4.2.cmml">Δ</mi> <mi id="Thmtheorem1.p1.3.m1.2.2.2.4.3" xref="Thmtheorem1.p1.3.m1.2.2.2.4.3.cmml">j</mi></msub> <mo id="Thmtheorem1.p1.3.m1.2.2.2.3" xref="Thmtheorem1.p1.3.m1.2.2.2.3.cmml">⁢</mo> <mrow id="Thmtheorem1.p1.3.m1.2.2.2.2.2" xref="Thmtheorem1.p1.3.m1.2.2.2.2.3.cmml"><mo id="Thmtheorem1.p1.3.m1.2.2.2.2.2.3" stretchy="false" xref="Thmtheorem1.p1.3.m1.2.2.2.2.3.cmml">(</mo><msubsup id="Thmtheorem1.p1.3.m1.1.1.1.1.1.1" xref="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.cmml"><mi id="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.2.2" xref="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.2.2.cmml">𝜽</mi> <mi id="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.2.3" xref="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.2.3.cmml">j</mi> <mi id="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.3" xref="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.3.cmml">t</mi></msubsup><mo id="Thmtheorem1.p1.3.m1.2.2.2.2.2.4" xref="Thmtheorem1.p1.3.m1.2.2.2.2.3.cmml">,</mo><msubsup id="Thmtheorem1.p1.3.m1.2.2.2.2.2.2" xref="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.cmml"><mi id="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.2.2" xref="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.2.2.cmml">𝒛</mi> <mi id="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.2.3" xref="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.2.3.cmml">j</mi> <mi id="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.3" xref="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.3.cmml">t</mi></msubsup><mo id="Thmtheorem1.p1.3.m1.2.2.2.2.2.5" stretchy="false" xref="Thmtheorem1.p1.3.m1.2.2.2.2.3.cmml">)</mo></mrow></mrow> <mo id="Thmtheorem1.p1.3.m1.4.4.5" xref="Thmtheorem1.p1.3.m1.4.4.5.cmml">=</mo> <mrow id="Thmtheorem1.p1.3.m1.4.4.4" xref="Thmtheorem1.p1.3.m1.4.4.4.cmml"><mrow id="Thmtheorem1.p1.3.m1.4.4.4.2" xref="Thmtheorem1.p1.3.m1.4.4.4.2.cmml"><msub id="Thmtheorem1.p1.3.m1.4.4.4.2.4" xref="Thmtheorem1.p1.3.m1.4.4.4.2.4.cmml"><mi id="Thmtheorem1.p1.3.m1.4.4.4.2.4.2" xref="Thmtheorem1.p1.3.m1.4.4.4.2.4.2.cmml">𝒪</mi> <mi id="Thmtheorem1.p1.3.m1.4.4.4.2.4.3" xref="Thmtheorem1.p1.3.m1.4.4.4.2.4.3.cmml">j</mi></msub> <mo id="Thmtheorem1.p1.3.m1.4.4.4.2.3" xref="Thmtheorem1.p1.3.m1.4.4.4.2.3.cmml">⁢</mo> <mrow id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.3.cmml"><mo id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.3" stretchy="false" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.3.cmml">(</mo><msubsup id="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1" xref="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.cmml"><mi id="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.2.2" xref="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.2.2.cmml">𝜽</mi> <mi id="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.2.3" xref="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.2.3.cmml">j</mi> <mi id="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.3" xref="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.3.cmml">t</mi></msubsup><mo id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.4" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.3.cmml">,</mo><msubsup id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.cmml"><mi id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.2.2" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.2.2.cmml">𝒛</mi> <mi id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.2.3" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.2.3.cmml">j</mi> <mi id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.3" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.3.cmml">t</mi></msubsup><mo id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.5" stretchy="false" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.3.cmml">)</mo></mrow></mrow> <mo id="Thmtheorem1.p1.3.m1.4.4.4.3" xref="Thmtheorem1.p1.3.m1.4.4.4.3.cmml">−</mo> <msubsup id="Thmtheorem1.p1.3.m1.4.4.4.4" xref="Thmtheorem1.p1.3.m1.4.4.4.4.cmml"><mi id="Thmtheorem1.p1.3.m1.4.4.4.4.2.2" xref="Thmtheorem1.p1.3.m1.4.4.4.4.2.2.cmml">𝜽</mi> <mi id="Thmtheorem1.p1.3.m1.4.4.4.4.2.3" xref="Thmtheorem1.p1.3.m1.4.4.4.4.2.3.cmml">j</mi> <mi id="Thmtheorem1.p1.3.m1.4.4.4.4.3" xref="Thmtheorem1.p1.3.m1.4.4.4.4.3.cmml">t</mi></msubsup></mrow></mrow> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.3.m1.4b"><apply id="Thmtheorem1.p1.3.m1.4.4.cmml" xref="Thmtheorem1.p1.3.m1.4.4"><apply id="Thmtheorem1.p1.3.m1.2.2.2.cmml" xref="Thmtheorem1.p1.3.m1.2.2.2"><apply id="Thmtheorem1.p1.3.m1.2.2.2.4.cmml" xref="Thmtheorem1.p1.3.m1.2.2.2.4"><csymbol cd="ambiguous" id="Thmtheorem1.p1.3.m1.2.2.2.4.1.cmml" xref="Thmtheorem1.p1.3.m1.2.2.2.4">subscript</csymbol> <ci id="Thmtheorem1.p1.3.m1.2.2.2.4.2.cmml" xref="Thmtheorem1.p1.3.m1.2.2.2.4.2">Δ</ci> <ci id="Thmtheorem1.p1.3.m1.2.2.2.4.3.cmml" xref="Thmtheorem1.p1.3.m1.2.2.2.4.3">𝑗</ci></apply> <interval closure="open" id="Thmtheorem1.p1.3.m1.2.2.2.2.3.cmml" xref="Thmtheorem1.p1.3.m1.2.2.2.2.2"><apply id="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.cmml" xref="Thmtheorem1.p1.3.m1.1.1.1.1.1.1"><csymbol cd="ambiguous" id="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.1.cmml" xref="Thmtheorem1.p1.3.m1.1.1.1.1.1.1">superscript</csymbol> <apply id="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.2.cmml" xref="Thmtheorem1.p1.3.m1.1.1.1.1.1.1"><csymbol cd="ambiguous" id="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.2.1.cmml" xref="Thmtheorem1.p1.3.m1.1.1.1.1.1.1">subscript</csymbol> <ci id="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.2.2.cmml" xref="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.2.2">𝜽</ci> <ci id="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.2.3.cmml" xref="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.2.3">𝑗</ci></apply> <ci id="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.3.cmml" xref="Thmtheorem1.p1.3.m1.1.1.1.1.1.1.3">𝑡</ci></apply> <apply id="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.cmml" xref="Thmtheorem1.p1.3.m1.2.2.2.2.2.2"><csymbol cd="ambiguous" id="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.1.cmml" xref="Thmtheorem1.p1.3.m1.2.2.2.2.2.2">superscript</csymbol> <apply id="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.2.cmml" xref="Thmtheorem1.p1.3.m1.2.2.2.2.2.2"><csymbol cd="ambiguous" id="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.2.1.cmml" xref="Thmtheorem1.p1.3.m1.2.2.2.2.2.2">subscript</csymbol> <ci id="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.2.2.cmml" xref="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.2.2">𝒛</ci> <ci id="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.2.3.cmml" xref="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.2.3">𝑗</ci></apply> <ci id="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.3.cmml" xref="Thmtheorem1.p1.3.m1.2.2.2.2.2.2.3">𝑡</ci></apply></interval></apply> <apply id="Thmtheorem1.p1.3.m1.4.4.4.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4"><apply id="Thmtheorem1.p1.3.m1.4.4.4.2.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.2"><apply id="Thmtheorem1.p1.3.m1.4.4.4.2.4.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.2.4"><csymbol cd="ambiguous" id="Thmtheorem1.p1.3.m1.4.4.4.2.4.1.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.2.4">subscript</csymbol> <ci id="Thmtheorem1.p1.3.m1.4.4.4.2.4.2.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.2.4.2">𝒪</ci> <ci id="Thmtheorem1.p1.3.m1.4.4.4.2.4.3.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.2.4.3">𝑗</ci></apply> <interval closure="open" id="Thmtheorem1.p1.3.m1.4.4.4.2.2.3.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.2"><apply id="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.cmml" xref="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1"><csymbol cd="ambiguous" id="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.1.cmml" xref="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1">superscript</csymbol> <apply id="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.2.cmml" xref="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1"><csymbol cd="ambiguous" id="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.2.1.cmml" xref="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1">subscript</csymbol> <ci id="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.2.2.cmml" xref="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.2.2">𝜽</ci> <ci id="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.2.3.cmml" xref="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.2.3">𝑗</ci></apply> <ci id="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.3.cmml" xref="Thmtheorem1.p1.3.m1.3.3.3.1.1.1.1.3">𝑡</ci></apply> <apply id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2"><csymbol cd="ambiguous" id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.1.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2">superscript</csymbol> <apply id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.2.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2"><csymbol cd="ambiguous" id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.2.1.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2">subscript</csymbol> <ci id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.2.2.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.2.2">𝒛</ci> <ci id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.2.3.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.2.3">𝑗</ci></apply> <ci id="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.3.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.2.2.2.2.3">𝑡</ci></apply></interval></apply> <apply id="Thmtheorem1.p1.3.m1.4.4.4.4.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.4"><csymbol cd="ambiguous" id="Thmtheorem1.p1.3.m1.4.4.4.4.1.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.4">superscript</csymbol> <apply id="Thmtheorem1.p1.3.m1.4.4.4.4.2.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.4"><csymbol cd="ambiguous" id="Thmtheorem1.p1.3.m1.4.4.4.4.2.1.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.4">subscript</csymbol> <ci id="Thmtheorem1.p1.3.m1.4.4.4.4.2.2.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.4.2.2">𝜽</ci> <ci id="Thmtheorem1.p1.3.m1.4.4.4.4.2.3.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.4.2.3">𝑗</ci></apply> <ci id="Thmtheorem1.p1.3.m1.4.4.4.4.3.cmml" xref="Thmtheorem1.p1.3.m1.4.4.4.4.3">𝑡</ci></apply></apply></apply></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.3.m1.4c">\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})=\mathcal{O}_{j}(\bm{\theta}_{j}% ^{t},\bm{z}_{j}^{t})-\bm{\theta}_{j}^{t}</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.3.m1.4d">roman_Δ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ( bold_italic_θ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, bold_italic_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT ) = caligraphic_O start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ( bold_italic_θ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, bold_italic_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT ) - bold_italic_θ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT</annotation></semantics></math>, <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="k_{0}=j"><semantics id="Thmtheorem1.p1.4.m2.1a"><mrow id="Thmtheorem1.p1.4.m2.1.1" xref="Thmtheorem1.p1.4.m2.1.1.cmml"><msub id="Thmtheorem1.p1.4.m2.1.1.2" xref="Thmtheorem1.p1.4.m2.1.1.2.cmml"><mi id="Thmtheorem1.p1.4.m2.1.1.2.2" xref="Thmtheorem1.p1.4.m2.1.1.2.2.cmml">k</mi> <mn id="Thmtheorem1.p1.4.m2.1.1.2.3" xref="Thmtheorem1.p1.4.m2.1.1.2.3.cmml">0</mn></msub> <mo id="Thmtheorem1.p1.4.m2.1.1.1" xref="Thmtheorem1.p1.4.m2.1.1.1.cmml">=</mo> <mi id="Thmtheorem1.p1.4.m2.1.1.3" xref="Thmtheorem1.p1.4.m2.1.1.3.cmml">j</mi></mrow> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.4.m2.1b"><apply id="Thmtheorem1.p1.4.m2.1.1.cmml" xref="Thmtheorem1.p1.4.m2.1.1"><apply id="Thmtheorem1.p1.4.m2.1.1.2.cmml" xref="Thmtheorem1.p1.4.m2.1.1.2"><csymbol cd="ambiguous" id="Thmtheorem1.p1.4.m2.1.1.2.1.cmml" xref="Thmtheorem1.p1.4.m2.1.1.2">subscript</csymbol> <ci id="Thmtheorem1.p1.4.m2.1.1.2.2.cmml" xref="Thmtheorem1.p1.4.m2.1.1.2.2">𝑘</ci> <cn id="Thmtheorem1.p1.4.m2.1.1.2.3.cmml" type="integer" xref="Thmtheorem1.p1.4.m2.1.1.2.3">0</cn></apply> <ci id="Thmtheorem1.p1.4.m2.1.1.3.cmml" xref="Thmtheorem1.p1.4.m2.1.1.3">𝑗</ci></apply></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.4.m2.1c">k_{0}=j</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.4.m2.1d">italic_k start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT = italic_j</annotation></semantics></math>. Here <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="P_{j}^{(\rho)}"><semantics id="Thmtheorem1.p1.5.m3.1a"><msubsup id="Thmtheorem1.p1.5.m3.1.2" xref="Thmtheorem1.p1.5.m3.1.2.cmml"><mi id="Thmtheorem1.p1.5.m3.1.2.2.2" xref="Thmtheorem1.p1.5.m3.1.2.2.2.cmml">P</mi> <mi id="Thmtheorem1.p1.5.m3.1.2.2.3" xref="Thmtheorem1.p1.5.m3.1.2.2.3.cmml">j</mi> <mrow id="Thmtheorem1.p1.5.m3.1.1.1.3" xref="Thmtheorem1.p1.5.m3.1.2.cmml"><mo id="Thmtheorem1.p1.5.m3.1.1.1.3.1" stretchy="false" xref="Thmtheorem1.p1.5.m3.1.2.cmml">(</mo><mi id="Thmtheorem1.p1.5.m3.1.1.1.1" xref="Thmtheorem1.p1.5.m3.1.1.1.1.cmml">ρ</mi><mo id="Thmtheorem1.p1.5.m3.1.1.1.3.2" stretchy="false" xref="Thmtheorem1.p1.5.m3.1.2.cmml">)</mo></mrow></msubsup> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.5.m3.1b"><apply id="Thmtheorem1.p1.5.m3.1.2.cmml" xref="Thmtheorem1.p1.5.m3.1.2"><csymbol cd="ambiguous" id="Thmtheorem1.p1.5.m3.1.2.1.cmml" xref="Thmtheorem1.p1.5.m3.1.2">superscript</csymbol> <apply id="Thmtheorem1.p1.5.m3.1.2.2.cmml" xref="Thmtheorem1.p1.5.m3.1.2"><csymbol cd="ambiguous" id="Thmtheorem1.p1.5.m3.1.2.2.1.cmml" xref="Thmtheorem1.p1.5.m3.1.2">subscript</csymbol> <ci id="Thmtheorem1.p1.5.m3.1.2.2.2.cmml" xref="Thmtheorem1.p1.5.m3.1.2.2.2">𝑃</ci> <ci id="Thmtheorem1.p1.5.m3.1.2.2.3.cmml" xref="Thmtheorem1.p1.5.m3.1.2.2.3">𝑗</ci></apply> <ci id="Thmtheorem1.p1.5.m3.1.1.1.1.cmml" xref="Thmtheorem1.p1.5.m3.1.1.1.1">𝜌</ci></apply></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.5.m3.1c">P_{j}^{(\rho)}</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.5.m3.1d">italic_P start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_ρ ) end_POSTSUPERSCRIPT</annotation></semantics></math> denotes the set of all sequences <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="(k_{1},\dots,k_{\rho})"><semantics id="Thmtheorem1.p1.6.m4.3a"><mrow id="Thmtheorem1.p1.6.m4.3.3.2" xref="Thmtheorem1.p1.6.m4.3.3.3.cmml"><mo id="Thmtheorem1.p1.6.m4.3.3.2.3" stretchy="false" xref="Thmtheorem1.p1.6.m4.3.3.3.cmml">(</mo><msub id="Thmtheorem1.p1.6.m4.2.2.1.1" xref="Thmtheorem1.p1.6.m4.2.2.1.1.cmml"><mi id="Thmtheorem1.p1.6.m4.2.2.1.1.2" xref="Thmtheorem1.p1.6.m4.2.2.1.1.2.cmml">k</mi> <mn id="Thmtheorem1.p1.6.m4.2.2.1.1.3" xref="Thmtheorem1.p1.6.m4.2.2.1.1.3.cmml">1</mn></msub><mo id="Thmtheorem1.p1.6.m4.3.3.2.4" xref="Thmtheorem1.p1.6.m4.3.3.3.cmml">,</mo><mi id="Thmtheorem1.p1.6.m4.1.1" mathvariant="normal" xref="Thmtheorem1.p1.6.m4.1.1.cmml">…</mi><mo id="Thmtheorem1.p1.6.m4.3.3.2.5" xref="Thmtheorem1.p1.6.m4.3.3.3.cmml">,</mo><msub id="Thmtheorem1.p1.6.m4.3.3.2.2" xref="Thmtheorem1.p1.6.m4.3.3.2.2.cmml"><mi id="Thmtheorem1.p1.6.m4.3.3.2.2.2" xref="Thmtheorem1.p1.6.m4.3.3.2.2.2.cmml">k</mi> <mi id="Thmtheorem1.p1.6.m4.3.3.2.2.3" xref="Thmtheorem1.p1.6.m4.3.3.2.2.3.cmml">ρ</mi></msub><mo id="Thmtheorem1.p1.6.m4.3.3.2.6" stretchy="false" xref="Thmtheorem1.p1.6.m4.3.3.3.cmml">)</mo></mrow> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.6.m4.3b"><vector id="Thmtheorem1.p1.6.m4.3.3.3.cmml" xref="Thmtheorem1.p1.6.m4.3.3.2"><apply id="Thmtheorem1.p1.6.m4.2.2.1.1.cmml" xref="Thmtheorem1.p1.6.m4.2.2.1.1"><csymbol cd="ambiguous" id="Thmtheorem1.p1.6.m4.2.2.1.1.1.cmml" xref="Thmtheorem1.p1.6.m4.2.2.1.1">subscript</csymbol> <ci id="Thmtheorem1.p1.6.m4.2.2.1.1.2.cmml" xref="Thmtheorem1.p1.6.m4.2.2.1.1.2">𝑘</ci> <cn id="Thmtheorem1.p1.6.m4.2.2.1.1.3.cmml" type="integer" xref="Thmtheorem1.p1.6.m4.2.2.1.1.3">1</cn></apply> <ci id="Thmtheorem1.p1.6.m4.1.1.cmml" xref="Thmtheorem1.p1.6.m4.1.1">…</ci> <apply id="Thmtheorem1.p1.6.m4.3.3.2.2.cmml" xref="Thmtheorem1.p1.6.m4.3.3.2.2"><csymbol cd="ambiguous" id="Thmtheorem1.p1.6.m4.3.3.2.2.1.cmml" xref="Thmtheorem1.p1.6.m4.3.3.2.2">subscript</csymbol> <ci id="Thmtheorem1.p1.6.m4.3.3.2.2.2.cmml" xref="Thmtheorem1.p1.6.m4.3.3.2.2.2">𝑘</ci> <ci id="Thmtheorem1.p1.6.m4.3.3.2.2.3.cmml" xref="Thmtheorem1.p1.6.m4.3.3.2.2.3">𝜌</ci></apply></vector></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.6.m4.3c">(k_{1},\dots,k_{\rho})</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.6.m4.3d">( italic_k start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT, …, italic_k start_POSTSUBSCRIPT italic_ρ end_POSTSUBSCRIPT )</annotation></semantics></math> such that <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="k_{s}\in\mathcal{N}_{\text{out}}^{(1)}(k_{s-1})"><semantics id="Thmtheorem1.p1.7.m5.2a"><mrow id="Thmtheorem1.p1.7.m5.2.2" xref="Thmtheorem1.p1.7.m5.2.2.cmml"><msub id="Thmtheorem1.p1.7.m5.2.2.3" xref="Thmtheorem1.p1.7.m5.2.2.3.cmml"><mi id="Thmtheorem1.p1.7.m5.2.2.3.2" xref="Thmtheorem1.p1.7.m5.2.2.3.2.cmml">k</mi> <mi id="Thmtheorem1.p1.7.m5.2.2.3.3" xref="Thmtheorem1.p1.7.m5.2.2.3.3.cmml">s</mi></msub> <mo id="Thmtheorem1.p1.7.m5.2.2.2" xref="Thmtheorem1.p1.7.m5.2.2.2.cmml">∈</mo> <mrow id="Thmtheorem1.p1.7.m5.2.2.1" xref="Thmtheorem1.p1.7.m5.2.2.1.cmml"><msubsup id="Thmtheorem1.p1.7.m5.2.2.1.3" xref="Thmtheorem1.p1.7.m5.2.2.1.3.cmml"><mi id="Thmtheorem1.p1.7.m5.2.2.1.3.2.2" xref="Thmtheorem1.p1.7.m5.2.2.1.3.2.2.cmml">𝒩</mi> <mtext id="Thmtheorem1.p1.7.m5.2.2.1.3.2.3" xref="Thmtheorem1.p1.7.m5.2.2.1.3.2.3a.cmml">out</mtext> <mrow id="Thmtheorem1.p1.7.m5.1.1.1.3" xref="Thmtheorem1.p1.7.m5.2.2.1.3.cmml"><mo id="Thmtheorem1.p1.7.m5.1.1.1.3.1" stretchy="false" xref="Thmtheorem1.p1.7.m5.2.2.1.3.cmml">(</mo><mn id="Thmtheorem1.p1.7.m5.1.1.1.1" xref="Thmtheorem1.p1.7.m5.1.1.1.1.cmml">1</mn><mo id="Thmtheorem1.p1.7.m5.1.1.1.3.2" stretchy="false" xref="Thmtheorem1.p1.7.m5.2.2.1.3.cmml">)</mo></mrow></msubsup> <mo id="Thmtheorem1.p1.7.m5.2.2.1.2" xref="Thmtheorem1.p1.7.m5.2.2.1.2.cmml">⁢</mo> <mrow id="Thmtheorem1.p1.7.m5.2.2.1.1.1" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.cmml"><mo id="Thmtheorem1.p1.7.m5.2.2.1.1.1.2" stretchy="false" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.cmml">(</mo><msub id="Thmtheorem1.p1.7.m5.2.2.1.1.1.1" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.cmml"><mi id="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.2" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.2.cmml">k</mi> <mrow id="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.3" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.3.cmml"><mi id="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.3.2" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.3.2.cmml">s</mi> <mo id="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.3.1" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.3.1.cmml">−</mo> <mn id="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.3.3" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.3.3.cmml">1</mn></mrow></msub><mo id="Thmtheorem1.p1.7.m5.2.2.1.1.1.3" stretchy="false" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.cmml">)</mo></mrow></mrow></mrow> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.7.m5.2b"><apply id="Thmtheorem1.p1.7.m5.2.2.cmml" xref="Thmtheorem1.p1.7.m5.2.2"><apply id="Thmtheorem1.p1.7.m5.2.2.3.cmml" xref="Thmtheorem1.p1.7.m5.2.2.3"><csymbol cd="ambiguous" id="Thmtheorem1.p1.7.m5.2.2.3.1.cmml" xref="Thmtheorem1.p1.7.m5.2.2.3">subscript</csymbol> <ci id="Thmtheorem1.p1.7.m5.2.2.3.2.cmml" xref="Thmtheorem1.p1.7.m5.2.2.3.2">𝑘</ci> <ci id="Thmtheorem1.p1.7.m5.2.2.3.3.cmml" xref="Thmtheorem1.p1.7.m5.2.2.3.3">𝑠</ci></apply> <apply id="Thmtheorem1.p1.7.m5.2.2.1.cmml" xref="Thmtheorem1.p1.7.m5.2.2.1"><apply id="Thmtheorem1.p1.7.m5.2.2.1.3.cmml" xref="Thmtheorem1.p1.7.m5.2.2.1.3"><csymbol cd="ambiguous" id="Thmtheorem1.p1.7.m5.2.2.1.3.1.cmml" xref="Thmtheorem1.p1.7.m5.2.2.1.3">superscript</csymbol> <apply id="Thmtheorem1.p1.7.m5.2.2.1.3.2.cmml" xref="Thmtheorem1.p1.7.m5.2.2.1.3"><csymbol cd="ambiguous" id="Thmtheorem1.p1.7.m5.2.2.1.3.2.1.cmml" xref="Thmtheorem1.p1.7.m5.2.2.1.3">subscript</csymbol> <ci id="Thmtheorem1.p1.7.m5.2.2.1.3.2.2.cmml" xref="Thmtheorem1.p1.7.m5.2.2.1.3.2.2">𝒩</ci> <ci id="Thmtheorem1.p1.7.m5.2.2.1.3.2.3a.cmml" xref="Thmtheorem1.p1.7.m5.2.2.1.3.2.3"><mtext id="Thmtheorem1.p1.7.m5.2.2.1.3.2.3.cmml" mathsize="70%" xref="Thmtheorem1.p1.7.m5.2.2.1.3.2.3">out</mtext></ci></apply> <cn id="Thmtheorem1.p1.7.m5.1.1.1.1.cmml" type="integer" xref="Thmtheorem1.p1.7.m5.1.1.1.1">1</cn></apply> <apply id="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.cmml" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1"><csymbol cd="ambiguous" id="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.1.cmml" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1">subscript</csymbol> <ci id="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.2.cmml" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.2">𝑘</ci> <apply id="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.3.cmml" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.3"><ci id="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.3.2.cmml" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.3.2">𝑠</ci> <cn id="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.3.3.cmml" type="integer" xref="Thmtheorem1.p1.7.m5.2.2.1.1.1.1.3.3">1</cn></apply></apply></apply></apply></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.7.m5.2c">k_{s}\in\mathcal{N}_{\text{out}}^{(1)}(k_{s-1})</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.7.m5.2d">italic_k start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT ∈ caligraphic_N start_POSTSUBSCRIPT out end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( 1 ) end_POSTSUPERSCRIPT ( italic_k start_POSTSUBSCRIPT italic_s - 1 end_POSTSUBSCRIPT )</annotation></semantics></math> for <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="s=1,\dots,\rho"><semantics id="Thmtheorem1.p1.8.m6.3a"><mrow id="Thmtheorem1.p1.8.m6.3.4" xref="Thmtheorem1.p1.8.m6.3.4.cmml"><mi id="Thmtheorem1.p1.8.m6.3.4.2" xref="Thmtheorem1.p1.8.m6.3.4.2.cmml">s</mi> <mo id="Thmtheorem1.p1.8.m6.3.4.1" xref="Thmtheorem1.p1.8.m6.3.4.1.cmml">=</mo> <mrow id="Thmtheorem1.p1.8.m6.3.4.3.2" xref="Thmtheorem1.p1.8.m6.3.4.3.1.cmml"><mn id="Thmtheorem1.p1.8.m6.1.1" xref="Thmtheorem1.p1.8.m6.1.1.cmml">1</mn><mo id="Thmtheorem1.p1.8.m6.3.4.3.2.1" xref="Thmtheorem1.p1.8.m6.3.4.3.1.cmml">,</mo><mi id="Thmtheorem1.p1.8.m6.2.2" mathvariant="normal" xref="Thmtheorem1.p1.8.m6.2.2.cmml">…</mi><mo id="Thmtheorem1.p1.8.m6.3.4.3.2.2" xref="Thmtheorem1.p1.8.m6.3.4.3.1.cmml">,</mo><mi id="Thmtheorem1.p1.8.m6.3.3" xref="Thmtheorem1.p1.8.m6.3.3.cmml">ρ</mi></mrow></mrow> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.8.m6.3b"><apply id="Thmtheorem1.p1.8.m6.3.4.cmml" xref="Thmtheorem1.p1.8.m6.3.4"><ci id="Thmtheorem1.p1.8.m6.3.4.2.cmml" xref="Thmtheorem1.p1.8.m6.3.4.2">𝑠</ci> <list id="Thmtheorem1.p1.8.m6.3.4.3.1.cmml" xref="Thmtheorem1.p1.8.m6.3.4.3.2"><cn id="Thmtheorem1.p1.8.m6.1.1.cmml" type="integer" xref="Thmtheorem1.p1.8.m6.1.1">1</cn> <ci id="Thmtheorem1.p1.8.m6.2.2.cmml" xref="Thmtheorem1.p1.8.m6.2.2">…</ci> <ci id="Thmtheorem1.p1.8.m6.3.3.cmml" xref="Thmtheorem1.p1.8.m6.3.3">𝜌</ci></list></apply></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.8.m6.3c">s=1,\dots,\rho</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.8.m6.3d">italic_s = 1, …, italic_ρ</annotation></semantics></math> (see Definition&nbsp;A.7) and <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\bm{H}(\bm{\theta}_{k_{s}}^{t+s};\bm{z}_{k_{s}}^{t+s})"><semantics id="Thmtheorem1.p1.9.m7.2a"><mrow id="Thmtheorem1.p1.9.m7.2.2" xref="Thmtheorem1.p1.9.m7.2.2.cmml"><mi id="Thmtheorem1.p1.9.m7.2.2.4" xref="Thmtheorem1.p1.9.m7.2.2.4.cmml">𝑯</mi> <mo id="Thmtheorem1.p1.9.m7.2.2.3" xref="Thmtheorem1.p1.9.m7.2.2.3.cmml">⁢</mo> <mrow id="Thmtheorem1.p1.9.m7.2.2.2.2" xref="Thmtheorem1.p1.9.m7.2.2.2.3.cmml"><mo id="Thmtheorem1.p1.9.m7.2.2.2.2.3" stretchy="false" xref="Thmtheorem1.p1.9.m7.2.2.2.3.cmml">(</mo><msubsup id="Thmtheorem1.p1.9.m7.1.1.1.1.1" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.cmml"><mi id="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.2" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.2.cmml">𝜽</mi> <msub id="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.3" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.3.cmml"><mi id="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.3.2" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.3.2.cmml">k</mi> <mi id="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.3.3" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.3.3.cmml">s</mi></msub> <mrow id="Thmtheorem1.p1.9.m7.1.1.1.1.1.3" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.3.cmml"><mi id="Thmtheorem1.p1.9.m7.1.1.1.1.1.3.2" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.3.2.cmml">t</mi> <mo id="Thmtheorem1.p1.9.m7.1.1.1.1.1.3.1" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.3.1.cmml">+</mo> <mi id="Thmtheorem1.p1.9.m7.1.1.1.1.1.3.3" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.3.3.cmml">s</mi></mrow></msubsup><mo id="Thmtheorem1.p1.9.m7.2.2.2.2.4" xref="Thmtheorem1.p1.9.m7.2.2.2.3.cmml">;</mo><msubsup id="Thmtheorem1.p1.9.m7.2.2.2.2.2" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.cmml"><mi id="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.2" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.2.cmml">𝒛</mi> <msub id="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.3" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.3.cmml"><mi id="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.3.2" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.3.2.cmml">k</mi> <mi id="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.3.3" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.3.3.cmml">s</mi></msub> <mrow id="Thmtheorem1.p1.9.m7.2.2.2.2.2.3" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.3.cmml"><mi id="Thmtheorem1.p1.9.m7.2.2.2.2.2.3.2" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.3.2.cmml">t</mi> <mo id="Thmtheorem1.p1.9.m7.2.2.2.2.2.3.1" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.3.1.cmml">+</mo> <mi id="Thmtheorem1.p1.9.m7.2.2.2.2.2.3.3" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.3.3.cmml">s</mi></mrow></msubsup><mo id="Thmtheorem1.p1.9.m7.2.2.2.2.5" stretchy="false" xref="Thmtheorem1.p1.9.m7.2.2.2.3.cmml">)</mo></mrow></mrow> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.9.m7.2b"><apply id="Thmtheorem1.p1.9.m7.2.2.cmml" xref="Thmtheorem1.p1.9.m7.2.2"><ci id="Thmtheorem1.p1.9.m7.2.2.4.cmml" xref="Thmtheorem1.p1.9.m7.2.2.4">𝑯</ci> <list id="Thmtheorem1.p1.9.m7.2.2.2.3.cmml" xref="Thmtheorem1.p1.9.m7.2.2.2.2"><apply id="Thmtheorem1.p1.9.m7.1.1.1.1.1.cmml" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1"><csymbol cd="ambiguous" id="Thmtheorem1.p1.9.m7.1.1.1.1.1.1.cmml" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1">superscript</csymbol> <apply id="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.cmml" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1"><csymbol cd="ambiguous" id="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.1.cmml" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1">subscript</csymbol> <ci id="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.2.cmml" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.2">𝜽</ci> <apply id="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.3.cmml" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.3"><csymbol cd="ambiguous" id="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.3.1.cmml" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.3">subscript</csymbol> <ci id="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.3.2.cmml" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.3.2">𝑘</ci> <ci id="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.3.3.cmml" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.2.3.3">𝑠</ci></apply></apply> <apply id="Thmtheorem1.p1.9.m7.1.1.1.1.1.3.cmml" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.3"><ci id="Thmtheorem1.p1.9.m7.1.1.1.1.1.3.2.cmml" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.3.2">𝑡</ci> <ci id="Thmtheorem1.p1.9.m7.1.1.1.1.1.3.3.cmml" xref="Thmtheorem1.p1.9.m7.1.1.1.1.1.3.3">𝑠</ci></apply></apply> <apply id="Thmtheorem1.p1.9.m7.2.2.2.2.2.cmml" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2"><csymbol cd="ambiguous" id="Thmtheorem1.p1.9.m7.2.2.2.2.2.1.cmml" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2">superscript</csymbol> <apply id="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.cmml" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2"><csymbol cd="ambiguous" id="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.1.cmml" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2">subscript</csymbol> <ci id="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.2.cmml" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.2">𝒛</ci> <apply id="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.3.cmml" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.3"><csymbol cd="ambiguous" id="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.3.1.cmml" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.3">subscript</csymbol> <ci id="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.3.2.cmml" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.3.2">𝑘</ci> <ci id="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.3.3.cmml" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.2.3.3">𝑠</ci></apply></apply> <apply id="Thmtheorem1.p1.9.m7.2.2.2.2.2.3.cmml" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.3"><ci id="Thmtheorem1.p1.9.m7.2.2.2.2.2.3.2.cmml" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.3.2">𝑡</ci> <ci id="Thmtheorem1.p1.9.m7.2.2.2.2.2.3.3.cmml" xref="Thmtheorem1.p1.9.m7.2.2.2.2.2.3.3">𝑠</ci></apply></apply></list></apply></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.9.m7.2c">\bm{H}(\bm{\theta}_{k_{s}}^{t+s};\bm{z}_{k_{s}}^{t+s})</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.9.m7.2d">bold_italic_H ( bold_italic_θ start_POSTSUBSCRIPT italic_k start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t + italic_s end_POSTSUPERSCRIPT; bold_italic_z start_POSTSUBSCRIPT italic_k start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t + italic_s end_POSTSUPERSCRIPT )</annotation></semantics></math> is the Hessian matrix of <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="L"><semantics id="Thmtheorem1.p1.10.m8.1a"><mi id="Thmtheorem1.p1.10.m8.1.1" xref="Thmtheorem1.p1.10.m8.1.1.cmml">L</mi> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.10.m8.1b"><ci id="Thmtheorem1.p1.10.m8.1.1.cmml" xref="Thmtheorem1.p1.10.m8.1.1">𝐿</ci></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.10.m8.1c">L</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.10.m8.1d">italic_L</annotation></semantics></math> with respect to <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\bm{\theta}"><semantics id="Thmtheorem1.p1.11.m9.1a"><mi id="Thmtheorem1.p1.11.m9.1.1" xref="Thmtheorem1.p1.11.m9.1.1.cmml">𝜽</mi> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.11.m9.1b"><ci id="Thmtheorem1.p1.11.m9.1.1.cmml" xref="Thmtheorem1.p1.11.m9.1.1">𝜽</ci></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.11.m9.1c">\bm{\theta}</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.11.m9.1d">bold_italic_θ</annotation></semantics></math> evaluated at <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\bm{\theta}_{k_{s}}^{t+s}"><semantics id="Thmtheorem1.p1.12.m10.1a"><msubsup id="Thmtheorem1.p1.12.m10.1.1" xref="Thmtheorem1.p1.12.m10.1.1.cmml"><mi id="Thmtheorem1.p1.12.m10.1.1.2.2" xref="Thmtheorem1.p1.12.m10.1.1.2.2.cmml">𝜽</mi> <msub id="Thmtheorem1.p1.12.m10.1.1.2.3" xref="Thmtheorem1.p1.12.m10.1.1.2.3.cmml"><mi id="Thmtheorem1.p1.12.m10.1.1.2.3.2" xref="Thmtheorem1.p1.12.m10.1.1.2.3.2.cmml">k</mi> <mi id="Thmtheorem1.p1.12.m10.1.1.2.3.3" xref="Thmtheorem1.p1.12.m10.1.1.2.3.3.cmml">s</mi></msub> <mrow id="Thmtheorem1.p1.12.m10.1.1.3" xref="Thmtheorem1.p1.12.m10.1.1.3.cmml"><mi id="Thmtheorem1.p1.12.m10.1.1.3.2" xref="Thmtheorem1.p1.12.m10.1.1.3.2.cmml">t</mi> <mo id="Thmtheorem1.p1.12.m10.1.1.3.1" xref="Thmtheorem1.p1.12.m10.1.1.3.1.cmml">+</mo> <mi id="Thmtheorem1.p1.12.m10.1.1.3.3" xref="Thmtheorem1.p1.12.m10.1.1.3.3.cmml">s</mi></mrow></msubsup> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.12.m10.1b"><apply id="Thmtheorem1.p1.12.m10.1.1.cmml" xref="Thmtheorem1.p1.12.m10.1.1"><csymbol cd="ambiguous" id="Thmtheorem1.p1.12.m10.1.1.1.cmml" xref="Thmtheorem1.p1.12.m10.1.1">superscript</csymbol> <apply id="Thmtheorem1.p1.12.m10.1.1.2.cmml" xref="Thmtheorem1.p1.12.m10.1.1"><csymbol cd="ambiguous" id="Thmtheorem1.p1.12.m10.1.1.2.1.cmml" xref="Thmtheorem1.p1.12.m10.1.1">subscript</csymbol> <ci id="Thmtheorem1.p1.12.m10.1.1.2.2.cmml" xref="Thmtheorem1.p1.12.m10.1.1.2.2">𝜽</ci> <apply id="Thmtheorem1.p1.12.m10.1.1.2.3.cmml" xref="Thmtheorem1.p1.12.m10.1.1.2.3"><csymbol cd="ambiguous" id="Thmtheorem1.p1.12.m10.1.1.2.3.1.cmml" xref="Thmtheorem1.p1.12.m10.1.1.2.3">subscript</csymbol> <ci id="Thmtheorem1.p1.12.m10.1.1.2.3.2.cmml" xref="Thmtheorem1.p1.12.m10.1.1.2.3.2">𝑘</ci> <ci id="Thmtheorem1.p1.12.m10.1.1.2.3.3.cmml" xref="Thmtheorem1.p1.12.m10.1.1.2.3.3">𝑠</ci></apply></apply> <apply id="Thmtheorem1.p1.12.m10.1.1.3.cmml" xref="Thmtheorem1.p1.12.m10.1.1.3"><ci id="Thmtheorem1.p1.12.m10.1.1.3.2.cmml" xref="Thmtheorem1.p1.12.m10.1.1.3.2">𝑡</ci> <ci id="Thmtheorem1.p1.12.m10.1.1.3.3.cmml" xref="Thmtheorem1.p1.12.m10.1.1.3.3">𝑠</ci></apply></apply></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.12.m10.1c">\bm{\theta}_{k_{s}}^{t+s}</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.12.m10.1d">bold_italic_θ start_POSTSUBSCRIPT italic_k start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t + italic_s end_POSTSUPERSCRIPT</annotation></semantics></math> and data <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\bm{z}_{k_{s}}^{t+s}"><semantics id="Thmtheorem1.p1.13.m11.1a"><msubsup id="Thmtheorem1.p1.13.m11.1.1" xref="Thmtheorem1.p1.13.m11.1.1.cmml"><mi id="Thmtheorem1.p1.13.m11.1.1.2.2" xref="Thmtheorem1.p1.13.m11.1.1.2.2.cmml">𝒛</mi> <msub id="Thmtheorem1.p1.13.m11.1.1.2.3" xref="Thmtheorem1.p1.13.m11.1.1.2.3.cmml"><mi id="Thmtheorem1.p1.13.m11.1.1.2.3.2" xref="Thmtheorem1.p1.13.m11.1.1.2.3.2.cmml">k</mi> <mi id="Thmtheorem1.p1.13.m11.1.1.2.3.3" xref="Thmtheorem1.p1.13.m11.1.1.2.3.3.cmml">s</mi></msub> <mrow id="Thmtheorem1.p1.13.m11.1.1.3" xref="Thmtheorem1.p1.13.m11.1.1.3.cmml"><mi id="Thmtheorem1.p1.13.m11.1.1.3.2" xref="Thmtheorem1.p1.13.m11.1.1.3.2.cmml">t</mi> <mo id="Thmtheorem1.p1.13.m11.1.1.3.1" xref="Thmtheorem1.p1.13.m11.1.1.3.1.cmml">+</mo> <mi id="Thmtheorem1.p1.13.m11.1.1.3.3" xref="Thmtheorem1.p1.13.m11.1.1.3.3.cmml">s</mi></mrow></msubsup> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.13.m11.1b"><apply id="Thmtheorem1.p1.13.m11.1.1.cmml" xref="Thmtheorem1.p1.13.m11.1.1"><csymbol cd="ambiguous" id="Thmtheorem1.p1.13.m11.1.1.1.cmml" xref="Thmtheorem1.p1.13.m11.1.1">superscript</csymbol> <apply id="Thmtheorem1.p1.13.m11.1.1.2.cmml" xref="Thmtheorem1.p1.13.m11.1.1"><csymbol cd="ambiguous" id="Thmtheorem1.p1.13.m11.1.1.2.1.cmml" xref="Thmtheorem1.p1.13.m11.1.1">subscript</csymbol> <ci id="Thmtheorem1.p1.13.m11.1.1.2.2.cmml" xref="Thmtheorem1.p1.13.m11.1.1.2.2">𝒛</ci> <apply id="Thmtheorem1.p1.13.m11.1.1.2.3.cmml" xref="Thmtheorem1.p1.13.m11.1.1.2.3"><csymbol cd="ambiguous" id="Thmtheorem1.p1.13.m11.1.1.2.3.1.cmml" xref="Thmtheorem1.p1.13.m11.1.1.2.3">subscript</csymbol> <ci id="Thmtheorem1.p1.13.m11.1.1.2.3.2.cmml" xref="Thmtheorem1.p1.13.m11.1.1.2.3.2">𝑘</ci> <ci id="Thmtheorem1.p1.13.m11.1.1.2.3.3.cmml" xref="Thmtheorem1.p1.13.m11.1.1.2.3.3">𝑠</ci></apply></apply> <apply id="Thmtheorem1.p1.13.m11.1.1.3.cmml" xref="Thmtheorem1.p1.13.m11.1.1.3"><ci id="Thmtheorem1.p1.13.m11.1.1.3.2.cmml" xref="Thmtheorem1.p1.13.m11.1.1.3.2">𝑡</ci> <ci id="Thmtheorem1.p1.13.m11.1.1.3.3.cmml" xref="Thmtheorem1.p1.13.m11.1.1.3.3">𝑠</ci></apply></apply></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.13.m11.1c">\bm{z}_{k_{s}}^{t+s}</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.13.m11.1d">bold_italic_z start_POSTSUBSCRIPT italic_k start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t + italic_s end_POSTSUPERSCRIPT</annotation></semantics></math>. For the cases when <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\rho=0"><semantics id="Thmtheorem1.p1.14.m12.1a"><mrow id="Thmtheorem1.p1.14.m12.1.1" xref="Thmtheorem1.p1.14.m12.1.1.cmml"><mi id="Thmtheorem1.p1.14.m12.1.1.2" xref="Thmtheorem1.p1.14.m12.1.1.2.cmml">ρ</mi> <mo id="Thmtheorem1.p1.14.m12.1.1.1" xref="Thmtheorem1.p1.14.m12.1.1.1.cmml">=</mo> <mn id="Thmtheorem1.p1.14.m12.1.1.3" xref="Thmtheorem1.p1.14.m12.1.1.3.cmml">0</mn></mrow> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.14.m12.1b"><apply id="Thmtheorem1.p1.14.m12.1.1.cmml" xref="Thmtheorem1.p1.14.m12.1.1"><ci id="Thmtheorem1.p1.14.m12.1.1.2.cmml" xref="Thmtheorem1.p1.14.m12.1.1.2">𝜌</ci> <cn id="Thmtheorem1.p1.14.m12.1.1.3.cmml" type="integer" xref="Thmtheorem1.p1.14.m12.1.1.3">0</cn></apply></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.14.m12.1c">\rho=0</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.14.m12.1d">italic_ρ = 0</annotation></semantics></math> and <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\rho=1"><semantics id="Thmtheorem1.p1.15.m13.1a"><mrow id="Thmtheorem1.p1.15.m13.1.1" xref="Thmtheorem1.p1.15.m13.1.1.cmml"><mi id="Thmtheorem1.p1.15.m13.1.1.2" xref="Thmtheorem1.p1.15.m13.1.1.2.cmml">ρ</mi> <mo id="Thmtheorem1.p1.15.m13.1.1.1" xref="Thmtheorem1.p1.15.m13.1.1.1.cmml">=</mo> <mn id="Thmtheorem1.p1.15.m13.1.1.3" xref="Thmtheorem1.p1.15.m13.1.1.3.cmml">1</mn></mrow> <annotation-xml encoding="MathML-Content" id="Thmtheorem1.p1.15.m13.1b"><apply id="Thmtheorem1.p1.15.m13.1.1.cmml" xref="Thmtheorem1.p1.15.m13.1.1"><ci id="Thmtheorem1.p1.15.m13.1.1.2.cmml" xref="Thmtheorem1.p1.15.m13.1.1.2">𝜌</ci> <cn id="Thmtheorem1.p1.15.m13.1.1.3.cmml" type="integer" xref="Thmtheorem1.p1.15.m13.1.1.3">1</cn></apply></annotation-xml> <annotation encoding="application/x-tex" id="Thmtheorem1.p1.15.m13.1c">\rho=1</annotation> <annotation encoding="application/x-llamapun" id="Thmtheorem1.p1.15.m13.1d">italic_ρ = 1</annotation></semantics></math>, the relevant product expressions are defined as identity matrices, thereby ensuring that the r-hop DICE-E remains well-defined. Full proof is deferred to Subsection&nbsp;C.3.</span></span></span></span></foreignObject></g></g></svg>

Multi-hop DICE-E characterizes the cascading effects of data influence through multiple “layers” of communication. In this context, the influence of a data instance from participant $j$ can propagate through a sequence of intermediate nodes, reaching participants that are $\rho$ hops away.

Influence Dynamics: Exponential Decay and Topological Dependency. Theorem 1 demonstrates that the multi-hop influence of a data instance $\bm{z}j^{t}$ is governed by the product of communication weights $\prod{s=1}^{\rho}\bm{W}{k_{s},k{s-1}}^{t+s-1}$ and Hessian-related terms $\prod_{s=2}^{\rho}(\bm{I}-\eta^{t+s-1}\bm{H}{k_{s}}^{t+s-1})$. This indicates that data influence in decentralized learning depends on the curvature of intermediate nodes and decays exponentially with each additional hop. Nodes with higher topological importance (e.g., node $j$ with large $\sum{j=1}^{n}\bm{W}_{j,k}$) propagate their data influence more widely and with greater impact on global utility (see Figure 1). These characteristics underscore the interplay between the original data, the loss landscape curvature, and communication topology in shaping data influence.

### 4.3 Practical Applications

In idealized scenarios, participants may seek to estimate the influence of their high-order neighbors on their local utility improvement. In practice, one-hop DICE-E emerges as a more suitable choice due to its computational efficiency. Based on Subsection 4.2, we derive the peer-level contribution, which we refer to as the proximal influence.

###### Definition 4 (Proximal Influence).

The proximal influence of a data instance $\bm{z}_{j}^{t}$ from participant $j$ on participant $k$ at iteration $t$ is defined as follows:

$$
\displaystyle\mathcal{I}_{\text{DICE-E}}^{k,j}(\bm{z}_{j}^{t},\bm{z}^{\prime})%
=-\eta^{t}\bm{W}_{k,j}^{t}q_{k}\nabla L(\bm{\theta}_{j}^{t};\bm{z}_{j}^{t})^{%
\top}\nabla L(\bm{\theta}_{k}^{t+1};\bm{z}^{\prime}).
$$

This term quantifies the influence of the data instance $\bm{z}_{j}^{t}$ from participant $j$ on the loss reduction experienced by its immediate neighbor $k$. Importantly, under the information sharing protocol defined in Algorithm 1, participant $k$ has access to $q_{k}$, $\bm{W}_{k,j}^{t}$, $\nabla L(\bm{\theta}_{j}^{t};\bm{z}_{j}^{t})$, and $\nabla L(\bm{\theta}_{k}^{t+1};\bm{z}^{\prime})$.<sup>4</sup> Therefore, each participant can compute the proximal contributions of its neighbors. The proximal influence can be utilized in the following scenarios:

Collaborator Selection. In decentralized learning, local data remains private and only local parameter communication is permitted. The absence of a central authority complicates the problem of selecting the most suitable neighbors with high-quality data. Fortunately, DICE offers a mechanism for participants to efficiently estimate the contributions of their neighbors with proximal influence. By assessing the proximal influence of their neighbors, participants can identify the potential collaborators that have the most significant positive impact on their learning process.

To ensure reciprocal collaboration [^33] [^26] [^102], participants can compute reciprocity factors, which evaluate the mutual balance of influence.

###### Definition 5 (Reciprocity Factors).

The reciprocity factor is defined in two forms:

1\. Proximal Reciprocity Factor: The reciprocity factor between participants $j$ and $k$ at iteration $t$ is

$$
\displaystyle R_{k,j}^{t}=\frac{q_{k}\bm{W}_{k,j}^{t}\nabla L(\bm{\theta}_{k}^%
{t+1};\bm{z}^{\prime})^{\top}\nabla L(\bm{\theta}_{j}^{t};\bm{z}_{k}^{t})}{q_{%
j}\bm{W}_{j,k}^{t}\nabla L(\bm{\theta}_{j}^{t+1};\bm{z}^{\prime})^{\top}\nabla
L%
(\bm{\theta}_{k}^{t};\bm{z}_{j}^{t})}.
$$

2\. Neighborhood Reciprocity Factor: To evaluate reciprocity at the community level, the neighborhood reciprocity factor for participant $j$ at iteration $t$ is defined as:

$$
\displaystyle R_{j}^{t}=\frac{\sum_{k\in\mathcal{N}_{\text{out}}^{(1)}(j)}q_{k%
}\bm{W}_{k,j}^{t}\nabla L(\bm{\theta}_{k}^{t+1};\bm{z}^{\prime})^{\top}\nabla L%
(\bm{\theta}_{j}^{t};\bm{z}_{j}^{t})}{\sum_{l\in\mathcal{N}_{\text{in}}^{(1)}(%
j)}q_{l}\bm{W}_{j,l}^{t}\nabla L(\bm{\theta}_{j}^{t+1};\bm{z}^{\prime})^{\top}%
\nabla L(\bm{\theta}_{l}^{t};\bm{z}_{j}^{t})}.
$$

The proximal reciprocity factor measures the balance of influence between two participants, with values near unity indicating equitable mutual contributions. Significant deviations suggest an imbalance, helping participants refine their collaboration strategies. The neighborhood reciprocity factor extends this concept to a participant’s local community, evaluating the balance between influence inflow and outflow. These metrics support participants in adjusting their engagement and aids the community in managing membership, such as admitting new members or excluding underperforming participants.

## 5 Experiments

This section presents the experimental results, with implementation details outlined in Subsection D.1.

##### Influence Alignment

We evaluate the alignment between one-hop DICE-GT (see Definition 2) and its first-order approximation, one-hop DICE-E (see Subsection 4.2). One-hop DICE-E $\scriptstyle{\mathcal{I}_{\text{DICE-E}}^{(1)}(\mathcal{B}_{j}^{t},\bm{z}^{%
\prime})}$ is computed as the sum of one-sample DICE-E within the mini-batch $\mathcal{B}_{j}^{t}$ thanks to the additivity (see Eq. 4). DICE-GT $\scriptstyle{\mathcal{I}{\text{DICE-GT}}^{(1)}(\mathcal{B}_{j}^{t},\bm{z}^{%
\prime})}$ is calculated by measuring the loss reduction after removing $\mathcal{B}_{j}^{t}$ from node $j$ at the $t$ -th iteration. As shown in Figure 3, each plot contains 30 points, with each point representing the result of a single comparison of the ground-truth and estimated influence. We can observe from Figure 3 that DICE-E closely tracks DICE-GT under different settings. The alignment becomes even stronger on simpler data set including CIFAR-10 and CIFAR-100, as detailed in Subsection D.2. These results demonstrate that DICE-E provides a strong approximation of DICE-GT, achieving consistent alignment across datasets (CIFAR-10, CIFAR-100 and Tiny ImageNet) and model architectures (CNN and MLP). Further validation of this alignment is provided in Subsection D.2 to corroborate the robustness of one-hop DICE-E under changing batch sizes, learning rates, and training epochs.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E1-Rebuttal-TinyImageNet/size32_tiny_batchsize128.png)

Figure 3: Alignment between one-hop DICE-GT (vertical axis) and DICE-E (horizontal axis) on a 32-node ring graph. Each node uses a 512-sample subset of Tiny ImageNet. Models are trained for 5 epochs with a batch size of 128 and a learning rate of 0.1.

##### Anomaly Detection

DICE identifies malicious neighbors, referred to as anomalies, by evaluating their proximal influence, which estimates the reduction in test loss caused by a single neighbor. A high proximal influence score indicates that a neighbor increases the test loss, negatively impacting the learning process. In our setup, anomalies are generated through random label flipping or by adding random Gaussian noise to features, please kindly refer to [^134]. Figure 3 illustrates that the most anomalies (in red) are readily detectable with proximal influence values. Additional results in Subsection D.3 further validate the reliability of this approach.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E2-Rebuttal-0.1/cifar10_epochs10_data512_batchsize64_modeexponential_size32__noniidfalse_chooseepoch5_pretrained0.png)

Figure 4: Anomaly detection on exponential graph with 32 nodes. Each node uses a 512-sample subset of CIFAR-10. Models are trained for 5 epochs with a batch size of 128 and a learning rate of 0.1. In a 32-node exponential graph, each participant connects with 5 neighbors, where the neighbor in red is set as an anomaly by random label flipping, while the other four are normal participants.

##### Influence Cascade

The topological dependency of DICE-E in our theory reveals the “power asymmetries” [^5] [^73] in decentralized learning. To support the theoretical finding, we examine the one-hop DICE-E values of the same batch on participants with vastly different topological importance. Figure 1 illustrates the one-hop DICE-E influence scores of an identical data batch across participants during decentralized training of a ResNet-18 model on the CIFAR-10 dataset. Node sizes represent the one-hop DICE-E influence scores, quantifying how a single batch impacts other participants in the network. The dominant nodes (e.g., those with larger outgoing communication weights in $\bm{W}$) exhibit significantly higher influence, as shown in Subsection D.4. These visualizations underscore the critical role of topological properties in shaping data influence in decentralized learning, demonstrating how the structure of the communication matrix $\bm{W}$ determines the asymmetries in influence.

## 6 Conclusion and Future Work

In this paper, we introduce DICE, the first comprehensive framework for quantifying data influence in fully decentralized learning environments. By modeling influence propagation across multiple hops, DICE reveals how local data contributions extend beyond immediate neighbors to reach non-adjacent neighbors. Mathematically, DICE formalizes how data influence cascades through the communication network, uncovering for the first time the intricate interplay between original data, communication topology, and the curvature of the optimization landscape in shaping data influence.

Future Work. Beyond its theoretical contributions, DICE holds significant potential for practical applications. By identifying influential contributors, DICE provides a foundation for designing fair incentive schemes that ensure equitable attribution of contributions, thereby promoting broader participation in decentralized learning. Moreover, DICE could contribute to the development of decentralized markets, including data markets [^48], parameter markets [^25], and compute markets [^59], which serve as fundamental building blocks of a broader decentralized learning ecosystem.

## Acknowledgment

This work is supported by the National Natural Science Foundation of China (No. 62476244), ZJU-China Unicom Digital Security Joint Laboratory and the advanced computing resources provided by the Supercomputing Center of Hangzhou City University.

## References

## Appendix A Background

### A.1 Data Influence

Data Influence Estimation. As high-quality data becomes increasingly critical in modern machine learning [^47] [^87] [^66] [^69] [^109], understanding the influence of data has emerged as a crucial research direction [^100] [^34] [^117]. Data influence estimation quantifies the contribution of training data to model predictions [^14] [^50]. It enables fair attribution of data source contributions, serving as the foundation for incentive mechanisms. Besides serving as an incentive, data influence has been extensively applied in various machine learning domains, including few-shot learning [^86], dataset pruning [^100] [^121], distillation [^70], fairness improving [^65], machine unlearning [^35] [^95], explainability [^56] [^42] [^34], as well as training-set attacks [^20] [^52] and defenses [^39].

Data influence estimators are broadly categorized into static and dynamic approaches <sup>5</sup>. Specifically, static approaches include both retraining-based and one-point methods. Retraining-based methods, such as leave-one-out [^17], Shapley value [^97], and Datamodels [^49], assess data influence by retraining the model on (subsets of) the training data. These methods offer a conceptually straightforward computation of data influence and are grounded in theoretical foundations, but they are often computationally expensive due to the requirement for retraining.

In contrast, one-point influence methods approximate the effect of retraining using a single trained model. A well-established one-point method is the canonical influence function [^56], developed from statistics [^41] [^13], which examines how infinitesimal perturbations of a training example affect the empirical risk minimizers (ERM) [^108]. The influence function has been extended to incorporate higher-order information [^3] and scaled for larger models [^37] [^94], including LLMs [^34]. [^27] extend influence function to Bayesian inference. While these static influence measures have elegant theoretical foundations, they are limited in characterizing how training data influences the training process.

Alternatively, dynamic methods enhance influence estimation by considering the evolution of model parameters across training iterations [^12]. Notable examples in this category include TracIn [^88] and In-Run Data Shapley [^113], which track the influence of training data points by averaging gradient similarities over time. The practicality of dynamic influence estimators is demonstrated by their applications in improving training processes in modern setups [^117]. Recently, [^83] adopt a novel memory-perturbation equation framework to derive dynamic influence estimation of model trained under different centralized optimization algorithms, including SGD, RMSprop and Adam.

However, the existing static and dynamic influence estimation methods primarily target centralized scenarios, and little progress has been made in analyzing data influence in fully decentralized environments. To the best of our knowledge, the only closely related work is by [^106], who proposed a decentralized hyper-gradient method and provided novel insights on applying hyper-gradients to compute a centralized formulation of data influence. Nevertheless, their estimation method is static and cannot capture the influence cascade arising from gossip communication during decentralized training. In contrast, our framework, DICE, is specifically designed for fully decentralized environments, allowing it to provide a fine-grained characterization of the unique influence cascade inherent in these settings.

### A.2 Decentralized learning

Currently, large-scale training and inference processes are primarily performed in expensive data centers. Decentralized training, echoing swarm intelligence [^6] [^77], offers a cost-efficient alternative avenue by crowd-sourcing computational workload to geographically decentralized compute nodes, without the control of central servers [^127] [^9] [^51]. One notable example showcasing decentralized computing’s computational potential is the Bitcoin eco-system which virtually distributes jobs requiring instantaneous 16 GW power consumption [^11] – this has been triple of the estimated 5 GW of the world’s largest planned cluster for AI [^30] [^85].

In the following, we provide an overview of the algorithmic and theoretical advancements in decentralized learning. While our discussion touches on several notable contributions, it is far from exhaustive. For a more comprehensive survey, we refer readers to [^75] [^99] [^131].

Algorithmic Development of Decentralized Learning. The advancement of decentralized learning algorithms has been driven by the need for communication-efficient optimization methods in practical distributed learning scenarios. These algorithms have adapted to accommodate dynamic network structures [^81] [^57] [^124] [^104], asynchronous communication [^68] [^120] [^80] [^7] [^24], data heterogeneity [^105] [^110] [^62], and Byzantine adversaries [^45] [^123]. Furthermore, their applicability has extended beyond conventional optimization problem to more complex problem formulations, including compositional [^28], minimax [^118] [^135] [^15], and bi-level [^122] [^29] [^16] optimization problems. Additionally, privacy concerns in decentralized learning are also critical, with efforts focusing on differentially privacy [^18] [^1] and data reconstruction attacks [^79].

Theoretical Development of Decentralized Learning. In terms of optimization, earlier works on decentralized optimization [^82] [^93] [^129] [^67] lay the groundwork for understanding convergence. [^72] present a systematic framework for federated and decentralized learning by categorizing decentralization into three distinct layers. [^57] unify synchronous decentralized gradient descent algorithms across various communication topologies, while [^24] extend this framework to accommodate asynchronous scenarios. Building on these efforts, [^132] further develops existing frameworks to consider the sporadicity of both communications and local computations. Regarding generalization, [^89] establishes generalization bounds of decentralized SGD in convex settings via uniform stability. [^101] extend this to non-convex settings, revealing an additional ${O}(\frac{1}{\rho})$ dependence on graph topology, though later empirical studies suggest this gap might be overstated [^58]. To refine this, [^136] introduce a Gaussian weight difference assumption, improving the $\rho$ dependence to $O((1-\rho)^{2})$. [^63] further show that in convex settings, the generalization error of local models in decentralized SGD matches that of standard SGD, while in non-convex settings, decentralization primarily affects worst-case generalization. To explain previously unexplained phenomena in decentralized learning [^58] [^38] [^111], [^137] later link decentralized SGD to random sharpness-aware minimization, uncovering a flatness bias in decentralized training. Complementing this, [^10] further analyze the flatness properties of DSGD and its role in escaping local minima.

Decentralized Training of Foundation Models. DT-FM [^127] introduces tasklet scheduling for training Transformers in decentralized settings with low-bandwidth networks, optimizing resource utilization in distributed environments. SWARM Parallelism [^91] enhances scalability through fault-tolerant pipelines and dynamic node rebalancing. CocktailSGD [^114] combines decentralization with sparsification and quantization enhances communication efficiency in fine-tuning LLMs. On the inference side, Petal [^8] leverages swarm parallelism to amortize inference costs across heterogeneous resources. Recently, Intellect [^51] built on Diloco [^21] has employed a combination of data parallel and model parallel to collaboratively train large models with up to billions of parameters. For a comprehensive overview of large-scale deep learning training, including data, model architecture, optimization strategies, budget constraints, and system design, see [^98].

The following figure presents a comparison between server-based learning and decentralized learning.

![Refer to caption](https://arxiv.org/html/2507.06931v1/x2.png)

Figure A.1: A comparative illustration of server-based learning versus decentralized learning.

We summarize some commonly used notions regarding decentralized training as follows:

###### Definition A.1 (Doubly Stochastic Matrix).

Let $\mathcal{G}=(\mathcal{V},\mathcal{E})$ represent a decentralized communication topology, where $\mathcal{V}$ is the set of $n$ nodes and $\mathcal{E}$ is the set of edges. For any $\mathcal{G}=(\mathcal{V},\mathcal{E})$, the doubly stochastic gossip matrix $\bm{W}=[\bm{W}_{j,k}]\in\mathbb{R}^{n\times n}$ is defined on the edge set $\mathcal{E}$ and satisfies:

- If $j\neq k$ and $(j,k)\notin\mathcal{E}$, then $\bm{W}_{j,k}=0$; otherwise, $\bm{W}_{j,k}>0$.
- $\bm{W}_{j,k}\in[0,1]$ for all $j,k$, and $\sum_{k}\bm{W}_{k,j}=\sum_{j}\bm{W}_{j,k}=1$.

Intuitively, the doubly stochastic property ensures a balanced flow of information during gossip communication, a common assumption in decentralized learning literature. However, in the scenarios we consider, participants may occupy different roles within the network. Influential nodes might have higher outgoing weights, i.e., $\sum_{j=1}^{n}\bm{W}_{j,k}>1$.

To accommodate such cases while still ensuring the convergence of decentralized SGD [^130] [^119], we introduce a relaxed condition:

###### Definition A.2 (Row Stochastic Matrix).

Let $\mathcal{G}=(\mathcal{V},\mathcal{E})$ denote a decentralized communication topology, where $\mathcal{V}$ is the set of $n$ nodes and $\mathcal{E}$ is the set of edges. For any $\mathcal{G}=(\mathcal{V},\mathcal{E})$, the row stochastic gossip matrix $\bm{W}=[\bm{W}_{j,k}]\in\mathbb{R}^{n\times n}$ is defined on the edge set $\mathcal{E}$ and satisfies:

- If $j\neq k$ and $(j,k)\notin\mathcal{E}$, then $\bm{W}_{j,k}=0$; otherwise, $\bm{W}_{j,k}>0$.
- $\bm{W}_{j,k}\in[0,1]$ for all $j,k$, and $\sum_{j}\bm{W}_{k,j}=1$.

The weighted adjacency matrix $\bm{W}$ in Algorithm 1 can vary across iterations, resulting in time-varying collaborations among participants. Additionally, FedAVG [^78] is a special case of Algorithm 1 where the averaging step is performed globally. This demonstrates that our framework accommodates decentralized learning with dynamic communication topologies and is applicable to both federated and decentralized learning paradigms, even though the primary focus is on fully decentralized learning without central servers.

### A.3 Multi-hop Neighbors

In graph theory, the concept of neighborhoods is fundamental for understanding the structure and dynamics of graphs. To ensure a coherent and comprehensive flow in Section 4, we provide formal definitions of multi-hop neighborhoods.

The adjacency matrix serves as a powerful tool for representing and analyzing the structure of a graph. Multi-hop neighbors can be precisely defined using the adjacency matrix.

###### Definition A.3 (Adjacency Matrix).

The adjacency matrix $A$ of a graph $G=(\mathcal{V},\mathcal{E})$ is an $n\times n$ square matrix (where $n=|\mathcal{V}|$) defined by:

$$
\displaystyle A_{jk}=\begin{cases}1&\text{if }(j,k)\in\mathcal{E},\\
0&\text{otherwise}.\end{cases}
$$

The adjacency matrix enables the determination of $r$ -hop neighbors through matrix exponentiation. Specifically, the $(j,k)$ -entry of $A^{r}$, denoted as $(A^{r})_{jk}$, corresponds to the number of distinct paths of length $r$ from node $j$ to node $k$.

###### Definition A.4 (r𝑟ritalic\_r-hop Neighbor via Adjacency Matrix).

The set of $r$ -hop neighbors is formally defined using the adjacency matrix $A$ as:

$$
\displaystyle\mathcal{N}^{(r)}(j)=\left\{k\in\mathcal{V}\,\bigg{|}\,(A^{r})_{%
jk}>0\text{ and }\forall s<r,\,(A^{s})_{jk}=0\right\}.
$$

This definition indicates that there exists at least one path of length $r$ connecting nodes $j$ and $k$, and no shorter path exists between them.

Multi-hop neighbors can also be defined via the shortest path length between two nodes.

###### Definition A.5 (Shortest Path Length).

In a connected graph $G=(\mathcal{V},\mathcal{E})$, the shortest path length $d(j,k)$ between nodes $j\in\mathcal{V}$ and $k\in\mathcal{V}$ is the minimum number of edges that must be traversed to travel from $j$ to $k$.

Building upon this, the set of $r$ -hop neighbors is defined as follows:

###### Definition A.6 (r𝑟ritalic\_r-hop Neighbor via Shortest Path Length).

For any node $j\in\mathcal{V}$ and a positive integer $r\geq 1$, the set of $r$ -hop neighbors, denoted by $\mathcal{N}^{(r)}(j)$, consists of all nodes that are at a distance of exactly $r$ from node $j$. Formally,

$$
\displaystyle\mathcal{N}^{(r)}(j)=\{k\in\mathcal{V}\mid d(j,k)=r\},
$$

where $d(j,k)$ represents the shortest path length between nodes $j$ and $k$ in the graph $G$.

Furthermore, an alternative perspective on $r$ -hop neighborhoods involves characterizing them through sequences of nodes, which provides a formal framework aligned with influence propagation in decentralized learning.

###### Definition A.7 (r𝑟ritalic\_r-hop Neighbor via Node Sequences).

For any node $j\in\mathcal{V}$ and a positive integer $r\geq 1$, let $P_{j}^{(r)}$ denote the set of all sequences $(k_{1},\dots,k_{r})$ such that for each $s=1,\dots,r$, the node $k_{s}$ is an out-neighbor of $k_{s-1}$, with $k_{0}=j$. Formally,

$$
P_{j}^{(r)}=\left\{(k_{1},\dots,k_{r})\mid k_{s}\in\mathcal{N}_{\text{out}}^{(%
1)}(k_{s-1})\text{ for }s=1,\dots,r\right\}.
$$

This definition ensures that each node in the $r$ -hop neighborhood is reachable from node $j$ through a sequence of consecutive immediate out-neighbors within $\rho\leq r$ steps.

This sequence-based characterization of $r$ -hop neighborhoods provides a granular understanding of the pathways through which influence or information can propagate within the network, complementing the previous definitions based on adjacency matrices and shortest path lengths.

## Appendix B Discussions

### B.1 Practical applications of DICE

Decentralized Machine Unlearning. As concerns about data privacy and the right to be forgotten increase, the ability to remove specific data contributions from a trained model becomes important [^35] [^95]. In decentralized settings, retraining the model from scratch is often impractical for edge users with limited compute. The proximal influence measure enables participants to estimate the impact of removing a particular data instance from its neighbor. For example, by assessing the influence of $\bm{z}_{j}^{t}$ on neighbors, participants can adjust their local models to mitigate the effects of $\bm{z}_{j}^{t}$ without requesting full retraining of the whole decentralized learning system. This approach facilitates efficient and targeted unlearning procedures, avoiding costly system-wide retraining while respecting individual data privacy requests.

### B.2 Additional related work

Clustered Federated Learning. Clustered Federated Learning (CFL) addresses the challenge of data heterogeneity by grouping clients with similar data distributions and training separate models for each cluster [^74] [^32] [^92] [^54]. Gradient-based CFL methods [^92] [^54] use client gradient similarities to form clusters, with [^92] employing cosine similarity to recursively partition clients after convergence and [^54] dynamically applying spectral clustering to organize clients based on gradient features during training. These methods effectively capture direct, peer-to-peer gradient relationships to cluster clients with similar data-generating distributions. Both gradient-based CFL and the one-hop DICE estimator (see Subsection 4.2) utilize gradient similarity information. However, CFL is inherently limited to local interactions, as its gradient similarity metrics are confined to pairwise relationships. In contrast, DICE extends far beyond this scope by quantifying the propagation of influence across multiple hops in a decentralized network. Mathematically, Theorem 1 highlights how DICE generalizes peer-level gradient similarity into a non-trivial extension for decentralized networks. This includes incorporating key factors including network topology and curvature information, enabling a deeper understanding of how influence flows through the whole decentralized learning systems. A promising future direction is to explore the potential of DICE-E as a more advanced high-order gradient similarity metric for effectively clustering participants in decentralized federated learning.

## Appendix C Proof

### C.1 Proof of Subsection 4.2

###### Proposition (Approximation of One-hop DICE-GT).

The one-hop DICE-GT value (see Definition 2) can be linearly approximated as follows:

$$
\displaystyle\mathcal{I}_{\text{DICE-E}}^{(1)}(\bm{z}_{j}^{t},\bm{z}^{\prime})=
$$
 
$$
\displaystyle\;-q_{j}\,\nabla L(\bm{\theta}_{j}^{t};\bm{z}^{\prime})^{\top}%
\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})-\sum_{k\in\mathcal{N}_{\text{%
out}}^{(1)}(j)}q_{k}\,\bm{W}_{k,j}^{t}\,\nabla L(\bm{\theta}_{k}^{t+1};\bm{z}^%
{\prime})^{\top}\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t}),
$$

where $\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})=\mathcal{O}_{j}(\bm{\theta}_{j}%
^{t},\bm{z}_{j}^{t})-\bm{\theta}_{j}^{t}$. The proof is given below.

###### Proof.

Recall from Definition 2 that the one-hop DICE-GT is defined by

$$
\displaystyle\mathcal{I}_{\text{DICE-GT}}^{(1)}(\bm{z}_{j}^{t},\bm{z}^{\prime})=
$$
 
$$
\displaystyle\;q_{j}\Bigl{(}L(\bm{\theta}_{j}^{t+\frac{1}{2}};\bm{z}^{\prime})%
-L(\bm{\theta}_{j}^{t};\bm{z}^{\prime})\Bigr{)}+\sum_{k\in\mathcal{N}_{\text{%
out}}^{(1)}(j)}q_{k}\Bigl{(}L(\bm{\theta}_{k}^{t+1};\bm{z}^{\prime})-L(\bm{%
\theta}_{k\setminus\bm{z}_{j}^{t}}^{t+1};\bm{z}^{\prime})\Bigr{)}.
$$

We proceed by applying a first-order Taylor expansion to each term.

First term: Using Taylor expansion, we write

$$
\displaystyle L(\bm{\theta}_{j}^{t+\frac{1}{2}};\bm{z}^{\prime})-L(\bm{\theta}%
_{j}^{t};\bm{z}^{\prime})
$$
 
$$
\displaystyle\approx\nabla L(\bm{\theta}_{j}^{t};\bm{z}^{\prime})^{\top}\Bigl{%
(}\bm{\theta}_{j}^{t+\frac{1}{2}}-\bm{\theta}_{j}^{t}\Bigr{)}.
$$

Under the new update rule, we have

$$
\bm{\theta}_{j}^{t+\frac{1}{2}}=\mathcal{O}_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}%
^{t}),
$$

so that

$$
\displaystyle\bm{\theta}_{j}^{t+\frac{1}{2}}-\bm{\theta}_{j}^{t}=\mathcal{O}_{%
j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})-\bm{\theta}_{j}^{t}.
$$

Thus, the first term is approximated by

$$
\displaystyle L(\bm{\theta}_{j}^{t+\frac{1}{2}};\bm{z}^{\prime})-L(\bm{\theta}%
_{j}^{t};\bm{z}^{\prime})\approx\nabla L(\bm{\theta}_{j}^{t};\bm{z}^{\prime})^%
{\top}\Bigl{(}\mathcal{O}_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})-\bm{\theta}_%
{j}^{t}\Bigr{)}.
$$

Second term: For each $k\in\mathcal{N}_{\text{out}}^{(1)}(j)$, we similarly have

$$
\displaystyle L(\bm{\theta}_{k}^{t+1};\bm{z}^{\prime})-L(\bm{\theta}_{k%
\setminus\bm{z}_{j}^{t}}^{t+1};\bm{z}^{\prime})\approx\nabla L(\bm{\theta}_{k}%
^{t+1};\bm{z}^{\prime})^{\top}\Bigl{(}\bm{\theta}_{k}^{t+1}-\bm{\theta}_{k%
\setminus\bm{z}_{j}^{t}}^{t+1}\Bigr{)}.
$$

By the gossip averaging step in the algorithm, we have

$$
\displaystyle\bm{\theta}_{k}^{t+1}
$$
 
$$
\displaystyle=\sum_{l\in\mathcal{N}_{\text{in}}(k)}\bm{W}_{k,l}^{t}\,\bm{%
\theta}_{l}^{t+\frac{1}{2}},
$$
$$
\displaystyle\bm{\theta}_{k\setminus\bm{z}_{j}^{t}}^{t+1}
$$
 
$$
\displaystyle=\bm{W}_{k,j}^{t}\,\bm{\theta}_{j}^{t}+\sum_{l\in\mathcal{N}_{%
\text{in}}(k)\setminus\{j\}}\bm{W}_{k,l}^{t}\,\bm{\theta}_{l}^{t+\frac{1}{2}}.
$$

It then follows that

$$
\displaystyle\bm{\theta}_{k}^{t+1}-\bm{\theta}_{k\setminus\bm{z}_{j}^{t}}^{t+1%
}=\bm{W}_{k,j}^{t}\Bigl{(}\bm{\theta}_{j}^{t+\frac{1}{2}}-\bm{\theta}_{j}^{t}%
\Bigr{)}=\bm{W}_{k,j}^{t}\Bigl{(}\mathcal{O}_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j%
}^{t})-\bm{\theta}_{j}^{t}\Bigr{)}.
$$

Thus, the second term becomes

$$
\displaystyle L(\bm{\theta}_{k}^{t+1};\bm{z}^{\prime})-L(\bm{\theta}_{k%
\setminus\bm{z}_{j}^{t}}^{t+1};\bm{z}^{\prime})\approx\bm{W}_{k,j}^{t}\,\nabla
L%
(\bm{\theta}_{k}^{t+1};\bm{z}^{\prime})^{\top}\Bigl{(}\mathcal{O}_{j}(\bm{%
\theta}_{j}^{t},\bm{z}_{j}^{t})-\bm{\theta}_{j}^{t}\Bigr{)}.
$$

Combining the Approximations: Substituting the above approximations into the definition of $\mathcal{I}_{\text{DICE-GT}}^{(1)}$ yields

$$
\displaystyle\mathcal{I}_{\text{DICE-E}}^{(1)}(\bm{z}_{j}^{t},\bm{z}^{\prime})\approx
$$
 
$$
\displaystyle\;-q_{j}\,\nabla L(\bm{\theta}_{j}^{t};\bm{z}^{\prime})^{\top}%
\Bigl{(}\mathcal{O}_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})-\bm{\theta}_{j}^{t%
}\Bigr{)}
$$
 
$$
\displaystyle\;-\sum_{k\in\mathcal{N}_{\text{out}}^{(1)}(j)}q_{k}\,\bm{W}_{k,j%
}^{t}\,\nabla L(\bm{\theta}_{k}^{t+1};\bm{z}^{\prime})^{\top}\Bigl{(}\mathcal{%
O}_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})-\bm{\theta}_{j}^{t}\Bigr{)}.
$$

This completes the proof. ∎

### C.2 Proof of Two-hop DICE-E Approximation

###### Proposition (Approximation of Two-hop DICE-GT).

The two-hop DICE-GT influence $\mathcal{I}_{\text{DICE-E}}^{(2)}(\bm{z}_{j}^{t},\bm{z}^{\prime})$ (see Definition 3) can be approximated as

$$
\displaystyle\mathcal{I}_{\text{DICE-E}}^{(2)}(\bm{z}_{j}^{t},\bm{z}^{\prime})%
=\mathcal{I}_{\text{DICE-E}}^{(1)}(\bm{z}_{j}^{t},\bm{z}^{\prime})
$$
 
$$
\displaystyle-\sum_{k\in\mathcal{N}_{\text{out}}^{(1)}(j)}\sum_{l\in\mathcal{N%
}_{\text{out}}^{(1)}(k)}\eta^{t}q_{l}\bm{W}_{l,k}^{t+1}\bm{W}_{k,j}^{t}\nabla L%
(\bm{\theta}_{l}^{t+2};\bm{z}^{\prime})^{\top}(\bm{I}-\eta^{t+1}\bm{H}(\bm{%
\theta}_{k}^{t+1};\bm{z}_{k}^{t+1}))\Delta_{j}(\bm{\theta}_{j}^{t};\bm{z}_{j}^%
{t}),
$$

where $\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})\triangleq\mathcal{O}_{j}(\bm{%
\theta}_{j}^{t},\bm{z}_{j}^{t})-\bm{\theta}_{j}^{t}.$ and $\bm{H}(\bm{\theta}_{k}^{t+1};\bm{z}_{k}^{t+1})$ denotes the Hessian matrix of $L$ with respect to $\bm{\theta}_{k}^{t+1}$ evaluated at $\bm{z}_{k}^{t+1}$.

###### Proof.

We begin from the definition in Definition 3 where the two-hop DICE-GT influence is given by

$$
\displaystyle\mathcal{I}_{\text{DICE-GT}}^{(2)}(\bm{z}_{j}^{t},\bm{z}^{\prime}%
)=\sum_{k\in\mathcal{N}_{\text{out}}^{(1)}(j)}\sum_{l\in\mathcal{N}_{\text{out%
}}^{(1)}(k)}q_{l}\,\Bigl{[}L(\bm{\theta}_{l}^{t+2};\bm{z}^{\prime})-L\bigl{(}%
\bm{\theta}_{l\setminus\bm{z}_{j}^{t}}^{t+2};\bm{z}^{\prime}\bigr{)}\Bigr{]}.
$$

Subtracting the one-hop influence from both sides yields

$$
\displaystyle\mathcal{I}_{\text{DICE-GT}}^{(2)}(\bm{z}_{j}^{t},\bm{z}^{\prime}%
)-\mathcal{I}_{\text{DICE-GT}}^{(1)}(\bm{z}_{j}^{t},\bm{z}^{\prime})=\sum_{k%
\in\mathcal{N}_{\text{out}}^{(1)}(j)}\sum_{l\in\mathcal{N}_{\text{out}}^{(1)}(%
k)}q_{l}\,\Bigl{[}L(\bm{\theta}_{l}^{t+2};\bm{z}^{\prime})-L\bigl{(}\bm{\theta%
}_{l\setminus\bm{z}_{j}^{t}}^{t+2};\bm{z}^{\prime}\bigr{)}\Bigr{]}.
$$

A first-order Taylor expansion gives

$$
L(\bm{\theta}_{l}^{t+2};\bm{z}^{\prime})-L\bigl{(}\bm{\theta}_{l\setminus\bm{z%
}_{j}^{t}}^{t+2};\bm{z}^{\prime}\bigr{)}\approx\nabla L\bigl{(}\bm{\theta}_{l}%
^{t+2};\bm{z}^{\prime}\bigr{)}^{\top}\Bigl{(}\bm{\theta}_{l}^{t+2}-\bm{\theta}%
_{l\setminus\bm{z}_{j}^{t}}^{t+2}\Bigr{)}.
$$

Next, the update rule in Algorithm 1 implies that

$$
\bm{\theta}_{l}^{t+2}=\sum_{m\in\mathcal{N}_{\text{in}}(l)}\bm{W}_{l,m}^{t+1}%
\,\bm{\theta}_{m}^{t+\frac{3}{2}},
$$

and similarly for $\bm{\theta}_{l\setminus\bm{z}_{j}^{t}}^{t+2}$. Since the influence of $\bm{z}_{j}^{t}$ reaches $l$ only via intermediate nodes, we may write

$$
\bm{\theta}_{l}^{t+2}-\bm{\theta}_{l\setminus\bm{z}_{j}^{t}}^{t+2}=\sum_{k\in%
\mathcal{N}_{\text{out}}^{(1)}(j)}\bm{W}_{l,k}^{t+1}\,\Bigl{(}\bm{\theta}_{k}^%
{t+\frac{3}{2}}-\bm{\theta}_{k\setminus\bm{z}_{j}^{t}}^{t+\frac{3}{2}}\Bigr{)}.
$$

At an intermediate node $k$, the local update in iteration $t+1$ gives

$$
\bm{\theta}_{k}^{t+\frac{3}{2}}=\bm{\theta}_{k}^{t+1}-\eta^{t+1}\,\nabla L%
\bigl{(}\bm{\theta}_{k}^{t+1};\bm{z}_{k}^{t+1}\bigr{)}.
$$

Therefore, the difference between the actual and perturbed updates is

$$
\bm{\theta}_{k}^{t+\frac{3}{2}}-\bm{\theta}_{k\setminus\bm{z}_{j}^{t}}^{t+%
\frac{3}{2}}\approx\Bigl{(}\bm{I}-\eta^{t+1}\,\bm{H}\bigl{(}\bm{\theta}_{k}^{t%
+1};\bm{z}_{k}^{t+1}\bigr{)}\Bigr{)}\Bigl{(}\bm{\theta}_{k}^{t+1}-\bm{\theta}_%
{k\setminus\bm{z}_{j}^{t}}^{t+1}\Bigr{)}.
$$

At node $k$, the difference between the parameters updated with and without the influence from $z_{j}^{t}$ is given by

$$
\bm{\theta}_{k}^{t+1}-\bm{\theta}_{k\setminus\bm{z}_{j}^{t}}^{t+1}=\bm{W}_{k,j%
}^{t}\Bigl{(}\bm{\theta}_{j}^{t+\frac{1}{2}}-\bm{\theta}_{j}^{t}\Bigr{)}%
\approx-\eta^{t}\,\bm{W}_{k,j}^{t}\,\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^%
{t}).
$$

Hence, we obtain

$$
\bm{\theta}_{k}^{t+\frac{3}{2}}-\bm{\theta}_{k\setminus\bm{z}_{j}^{t}}^{t+%
\frac{3}{2}}\approx-\eta^{t}\,\Bigl{(}\bm{I}-\eta^{t+1}\,\bm{H}\bigl{(}\bm{%
\theta}_{k}^{t+1};\bm{z}_{k}^{t+1}\bigr{)}\Bigr{)}\bm{W}_{k,j}^{t}\,\Delta_{j}%
(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t}).
$$

Substituting back into the expression for node $l$, we have

$$
\bm{\theta}_{l}^{t+2}-\bm{\theta}_{l\setminus\bm{z}_{j}^{t}}^{t+2}\approx-\eta%
^{t}\,\sum_{k\in\mathcal{N}_{\text{out}}^{(1)}(j)}\bm{W}_{l,k}^{t+1}\,\bm{W}_{%
k,j}^{t}\,\Bigl{(}\bm{I}-\eta^{t+1}\,\bm{H}\bigl{(}\bm{\theta}_{k}^{t+1};\bm{z%
}_{k}^{t+1}\bigr{)}\Bigr{)}\,\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t}).
$$

Plugging this into the Taylor expansion for the loss difference and multiplying by $q_{l}$ yields

$$
\displaystyle L(\bm{\theta}_{l}^{t+2};\bm{z}^{\prime})-L\bigl{(}\bm{\theta}_{l%
\setminus\bm{z}_{j}^{t}}^{t+2};\bm{z}^{\prime}\bigr{)}
$$
 
$$
\displaystyle\approx-\eta^{t}\,\nabla L\bigl{(}\bm{\theta}_{l}^{t+2};\bm{z}^{%
\prime}\bigr{)}^{\top}\,\sum_{k\in\mathcal{N}_{\text{out}}^{(1)}(j)}\bm{W}_{l,%
k}^{t+1}\,\bm{W}_{k,j}^{t}\,\Bigl{(}\bm{I}-\eta^{t+1}\,\bm{H}\bigl{(}\bm{%
\theta}_{k}^{t+1};\bm{z}_{k}^{t+1}\bigr{)}\Bigr{)}\,\Delta_{j}(\bm{\theta}_{j}%
^{t},\bm{z}_{j}^{t}).
$$

Finally, summing over all intermediate nodes and multiplying by $q_{l}$, we obtain

$$
\displaystyle\mathcal{I}_{\text{DICE-E}}^{(2)}(\bm{z}_{j}^{t},\bm{z}^{\prime})%
-\mathcal{I}_{\text{DICE-E}}^{(1)}(\bm{z}_{j}^{t},\bm{z}^{\prime})
$$
 
$$
\displaystyle\approx-\sum_{k\in\mathcal{N}_{\text{out}}^{(1)}(j)}\sum_{l\in%
\mathcal{N}_{\text{out}}^{(1)}(k)}\eta^{t}\,q_{l}\,\bm{W}_{l,k}^{t+1}\,\bm{W}_%
{k,j}^{t}\,\nabla L\bigl{(}\bm{\theta}_{l}^{t+2};\bm{z}^{\prime}\bigr{)}^{\top%
}\,\Bigl{(}\bm{I}-\eta^{t+1}\,\bm{H}\bigl{(}\bm{\theta}_{k}^{t+1};\bm{z}_{k}^{%
t+1}\bigr{)}\Bigr{)}\,\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t}).
$$

This completes the proof. ∎

### C.3 Proof of Theorem 1

Theorem 2 (Approximation of Multi-hop DICE-GT). The $r$ -hop DICE-GT value (see Definition 3) can be approximated as

$$
\displaystyle\mathcal{I}_{\text{DICE-E}}^{(r)}(\bm{z}_{j}^{t},\bm{z}^{\prime})%
=-\sum_{\rho=0}^{r}\sum_{(k_{1},\dots,k_{\rho})\in P_{j}^{(\rho)}}\eta^{t}\,q_%
{k_{\rho}}\,\left(\prod_{s=1}^{\rho}\bm{W}_{k_{s},k_{s-1}}^{t+s-1}\right)%
\nabla L\bigl{(}\bm{\theta}_{k_{\rho}}^{t+\rho};\bm{z}^{\prime}\bigr{)}^{\top}
$$
 
$$
\displaystyle\quad\times\left(\prod_{s=2}^{\rho}\Bigl{(}\bm{I}-\eta^{t+s-1}\,%
\bm{H}\bigl{(}\bm{\theta}_{k_{s}}^{t+s-1};\bm{z}_{k_{s}}^{t+s-1}\bigr{)}\Bigr{%
)}\right)\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t}),
$$

where

$$
\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t})\triangleq\mathcal{O}_{j}(\bm{%
\theta}_{j}^{t},\bm{z}_{j}^{t})-\bm{\theta}_{j}^{t},
$$

where $k_{0}=j$, $P_{j}^{(\rho)}$ denotes the set of all sequences $(k_{1},\dots,k_{\rho})$ such that $k_{s}\in\mathcal{N}_{\text{out}}^{(1)}(k_{s-1})$ for $s=1,\dots,\rho$ (see Definition A.7) and $\bm{H}(\bm{\theta}_{k_{s}}^{t+s};\bm{z}_{k_{s}}^{t+s})$ is the Hessian matrix of $L$ with respect to $\bm{\theta}$ evaluated at $\bm{\theta}_{k_{s}}^{t+s}$ and data $\bm{z}_{k_{s}}^{t+s}$. For the cases when $\rho=0$ and $\rho=1$, the relevant product expressions are defined as identity matrices, thereby ensuring that the r-hop DICE-E remains well-defined.

###### Proof.

From the definition in Definition 3, the $r$ -hop influence is

$$
\displaystyle\mathcal{I}_{\text{DICE-GT}}^{(r)}(\bm{z}_{j}^{t},\bm{z}^{\prime}%
)=\sum_{\rho=0}^{r}\sum_{(k_{1},\dots,k_{\rho})\in P_{j}^{(\rho)}}q_{k_{\rho}}%
\Bigl{(}L(\bm{\theta}_{k_{\rho}}^{t+\rho};\bm{z}^{\prime})-L(\bm{\theta}_{k_{%
\rho}\setminus\bm{z}_{j}^{t}}^{t+\rho};\bm{z}^{\prime})\Bigr{)}.
$$

Here the $\rho=0$ term (with $k_{0}=j$) corresponds to the direct influence on node $j$. For any $\rho\geq 1$, define the incremental influence as

$$
\Delta\mathcal{I}_{\text{DICE-GT}}^{(\rho)}(\bm{z}_{j}^{t},\bm{z}^{\prime})=%
\mathcal{I}_{\text{DICE-GT}}^{(\rho)}(\bm{z}_{j}^{t},\bm{z}^{\prime})-\mathcal%
{I}_{\text{DICE-GT}}^{(\rho-1)}(\bm{z}_{j}^{t},\bm{z}^{\prime}).
$$

Thus,

$$
\Delta\mathcal{I}_{\text{DICE-GT}}^{(\rho)}(\bm{z}_{j}^{t},\bm{z}^{\prime})=%
\sum_{(k_{1},\dots,k_{\rho})\in P_{j}^{(\rho)}}q_{k_{\rho}}\left[L(\bm{\theta}%
_{k_{\rho}}^{t+\rho};\bm{z}^{\prime})-L(\bm{\theta}_{k_{\rho}\setminus\bm{z}_{%
j}^{t}}^{t+\rho};\bm{z}^{\prime})\right].
$$

A first-order Taylor expansion gives

$$
L(\bm{\theta}_{k_{\rho}}^{t+\rho};\bm{z}^{\prime})-L(\bm{\theta}_{k_{\rho}%
\setminus\bm{z}_{j}^{t}}^{t+\rho};\bm{z}^{\prime})\approx\nabla L\bigl{(}\bm{%
\theta}_{k_{\rho}}^{t+\rho};\bm{z}^{\prime}\bigr{)}^{\top}\Bigl{[}\bm{\theta}_%
{k_{\rho}}^{t+\rho}-\bm{\theta}_{k_{\rho}\setminus\bm{z}_{j}^{t}}^{t+\rho}%
\Bigr{]}.
$$

Our goal is to express the parameter change $\Delta\bm{\theta}_{k_{\rho}}\triangleq\bm{\theta}_{k_{\rho}}^{t+\rho}-\bm{%
\theta}_{k_{\rho}\setminus\bm{z}_{j}^{t}}^{t+\rho}$ in terms of the propagated perturbation from node $j$.

According to the gossip update in Algorithm 1, for any node $k_{\rho}$ we have

$$
\displaystyle\bm{\theta}_{k_{\rho}}^{t+\rho}
$$
 
$$
\displaystyle=\sum_{m\in\mathcal{N}_{\text{in}}(k_{\rho})}\bm{W}_{k_{\rho},m}^%
{t+\rho-1}\,\bm{\theta}_{m}^{t+\rho-\frac{1}{2}},
$$
$$
\displaystyle\bm{\theta}_{k_{\rho}\setminus\bm{z}_{j}^{t}}^{t+\rho}
$$
 
$$
\displaystyle=\sum_{m\in\mathcal{N}_{\text{in}}(k_{\rho})}\bm{W}_{k_{\rho},m}^%
{t+\rho-1}\,\bm{\theta}_{m\setminus\bm{z}_{j}^{t}}^{t+\rho-\frac{1}{2}}.
$$

Since only the predecessor $k_{\rho-1}$ is affected by the perturbation from $\bm{z}_{j}^{t}$, we obtain

$$
\displaystyle\Delta\bm{\theta}_{k_{\rho}}
$$
 
$$
\displaystyle=\bm{W}_{k_{\rho},k_{\rho-1}}^{t+\rho-1}\Bigl{(}\bm{\theta}_{k_{%
\rho-1}}^{t+\rho-\frac{1}{2}}-\bm{\theta}_{k_{\rho-1}\setminus\bm{z}_{j}^{t}}^%
{t+\rho-\frac{1}{2}}\Bigr{)}.
$$

At node $k_{\rho-1}$, using the local update rule,

$$
\bm{\theta}_{k_{\rho-1}}^{t+\rho-\frac{1}{2}}=\mathcal{O}_{k_{\rho-1}}(\bm{%
\theta}_{k_{\rho-1}}^{t+\rho-1},\bm{z}_{k_{\rho-1}}^{t+\rho-1}),
$$

the difference can be written as

$$
\displaystyle\bm{\theta}_{k_{\rho-1}}^{t+\rho-\frac{1}{2}}-\bm{\theta}_{k_{%
\rho-1}\setminus\bm{z}_{j}^{t}}^{t+\rho-\frac{1}{2}}
$$
 
$$
\displaystyle=\Bigl{(}\bm{\theta}_{k_{\rho-1}}^{t+\rho-1}-\bm{\theta}_{k_{\rho%
-1}\setminus\bm{z}_{j}^{t}}^{t+\rho-1}\Bigr{)}
$$
 
$$
\displaystyle\quad-\eta^{t+\rho-1}\Bigl{(}\nabla L(\bm{\theta}_{k_{\rho-1}}^{t%
+\rho-1};\bm{z}_{k_{\rho-1}}^{t+\rho-1})-\nabla L(\bm{\theta}_{k_{\rho-1}%
\setminus\bm{z}_{j}^{t}}^{t+\rho-1};\bm{z}_{k_{\rho-1}}^{t+\rho-1})\Bigr{)}.
$$

A further first-order Taylor expansion approximates

$$
\nabla L(\bm{\theta}_{k_{\rho-1}}^{t+\rho-1};\bm{z}_{k_{\rho-1}}^{t+\rho-1})-%
\nabla L(\bm{\theta}_{k_{\rho-1}\setminus\bm{z}_{j}^{t}}^{t+\rho-1};\bm{z}_{k_%
{\rho-1}}^{t+\rho-1})\approx\bm{H}_{k_{\rho-1}}^{t+\rho-1}\Bigl{(}\bm{\theta}_%
{k_{\rho-1}}^{t+\rho-1}-\bm{\theta}_{k_{\rho-1}\setminus\bm{z}_{j}^{t}}^{t+%
\rho-1}\Bigr{)}.
$$

Thus,

$$
\Delta\bm{\theta}_{k_{\rho}}\approx\bm{W}_{k_{\rho},k_{\rho-1}}^{t+\rho-1}%
\Bigl{(}\bm{I}-\eta^{t+\rho-1}\bm{H}_{k_{\rho-1}}^{t+\rho-1}\Bigr{)}\Bigl{(}%
\bm{\theta}_{k_{\rho-1}}^{t+\rho-1}-\bm{\theta}_{k_{\rho-1}\setminus\bm{z}_{j}%
^{t}}^{t+\rho-1}\Bigr{)}.
$$

By recursively unrolling this relation from $s=\rho$ down to $s=1$, we deduce

$$
\displaystyle\bm{\theta}_{k_{\rho}}^{t+\rho}-\bm{\theta}_{k_{\rho}\setminus\bm%
{z}_{j}^{t}}^{t+\rho}\approx\left(\prod_{s=1}^{\rho}\bm{W}_{k_{s},k_{s-1}}^{t+%
s-1}\prod_{s=2}^{\rho}\Bigl{(}\bm{I}-\eta^{t+s-1}\bm{H}_{k_{s-1}}^{t+s-1}\Bigr%
{)}\right)\Bigl{(}\bm{\theta}_{k_{1}}^{t+1}-\bm{\theta}_{k_{1}\setminus\bm{z}_%
{j}^{t}}^{t+1}\Bigr{)}.
$$

At the base level, the local update at node $j$ gives

$$
\bm{\theta}_{k_{1}}^{t+1}-\bm{\theta}_{k_{1}\setminus\bm{z}_{j}^{t}}^{t+1}=-%
\eta^{t}\,\bm{W}_{k_{1},j}^{t}\,\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t}).
$$

Hence,

$$
\displaystyle\bm{\theta}_{k_{\rho}}^{t+\rho}-\bm{\theta}_{k_{\rho}\setminus\bm%
{z}_{j}^{t}}^{t+\rho}\approx-\eta^{t}\,\left(\prod_{s=1}^{\rho}\bm{W}_{k_{s},k%
_{s-1}}^{t+s-1}\right)\Biggl{(}\prod_{s=2}^{\rho}\Bigl{(}\bm{I}-\eta^{t+s-1}%
\bm{H}_{k_{s-1}}^{t+s-1}\Bigr{)}\Biggr{)}\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}%
_{j}^{t}).
$$

Substituting this back into the Taylor expansion for the loss difference, we have

$$
\displaystyle\Delta\mathcal{I}_{\text{DICE-GT}}^{(\rho)}(\bm{z}_{j}^{t},\bm{z}%
^{\prime})
$$
 
$$
\displaystyle\approx-\sum_{(k_{1},\dots,k_{\rho})\in P_{j}^{(\rho)}}\eta^{t}\,%
q_{k_{\rho}}\,\left(\prod_{s=1}^{\rho}\bm{W}_{k_{s},k_{s-1}}^{t+s-1}\right)%
\nabla L\bigl{(}\bm{\theta}_{k_{\rho}}^{t+\rho};\bm{z}^{\prime}\bigr{)}^{\top}
$$
 
$$
\displaystyle\quad\times\left(\prod_{s=2}^{\rho}\Bigl{(}\bm{I}-\eta^{t+s-1}\bm%
{H}_{k_{s-1}}^{t+s-1}\Bigr{)}\right)\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^%
{t}).
$$

Summing over $\rho=0$ to $r$ (with the $\rho=0$ term accounting for the direct influence at node $j$) yields

$$
\displaystyle\mathcal{I}_{\text{DICE-E}}^{(r)}(\bm{z}_{j}^{t},\bm{z}^{\prime})%
=-\sum_{\rho=0}^{r}\sum_{(k_{1},\dots,k_{\rho})\in P_{j}^{(\rho)}}\eta^{t}\,q_%
{k_{\rho}}\,\left(\prod_{s=1}^{\rho}\bm{W}_{k_{s},k_{s-1}}^{t+s-1}\right)%
\nabla L\bigl{(}\bm{\theta}_{k_{\rho}}^{t+\rho};\bm{z}^{\prime}\bigr{)}^{\top}
$$
 
$$
\displaystyle\quad\times\left(\prod_{s=2}^{\rho}\Bigl{(}\bm{I}-\eta^{t+s-1}\bm%
{H}\bigl{(}\bm{\theta}_{k_{s}}^{t+s-1};\bm{z}_{k_{s}}^{t+s-1}\bigr{)}\Bigr{)}%
\right)\Delta_{j}(\bm{\theta}_{j}^{t},\bm{z}_{j}^{t}).
$$

This concludes the proof. ∎

## Appendix D Additional Experiments

### D.1 Details of Experimental Setup

Computational Resources. The experiments are conducted on a computing facility equipped with 80 GB NVIDIA <sup>®</sup> A100 <sup>™</sup> GPUs.

We employ the vanilla mini-batch Adapt-Then-Communicate version of Decentralized SGD ([^71], see Algorithm 1) with commonly used network topologies [^124] to train three-layer MLPs [^90], three-layer CNNs [^64], and ResNet-18 [^44] on subsets of MNIST [^64], CIFAR-10, CIFAR-100 [^60], and Tiny ImageNet [^61]. The number of participants (one GPU as a participant) is set to 16 and 32, with each participant holding 512 samples. For sensitivity analysis, we evaluate the stability of results under hyperparameter adjustments. The local batch size is varied as 16, 64, and 128 per participant, while the learning rate is set as 0.1 and 0.01 without decay. The code will be made publicly available.

### D.2 Influence Alignment

In this experiments, we evaluate the alignment between one-hop DICE-GT (see Definition 2) and its first-order approximation, one-hop DICE-E (see Subsection 4.2). One-hop DICE-E $\scriptstyle{\mathcal{I}_{\text{DICE-E}}^{(1)}(\mathcal{B}_{j}^{t},\bm{z}^{%
\prime})}$ is computed as the sum of one-sample DICE-E within the mini-batch $\mathcal{B}_{j}^{t}$ thanks to the additivity (see Eq. 4). DICE-GT $\scriptstyle{\mathcal{I}{\text{DICE-GT}}^{(1)}(\mathcal{B}_{j}^{t},\bm{z}^{%
\prime})}$ is calculated by measuring the loss reduction after removing $\mathcal{B}_{j}^{t}$ from node $j$ at the $t$ -th iteration. In the following Figures, each plot contains 30 points, with each point representing the result of a single comparison of one-hop DICE-GT and the estimated influence DICE-E. Strong alignments of DICE-GT and DICE-E are observed across datasets (CIFAR-10, CIFAR-100 and Tiny ImageNet) and model architectures (CNN and MLP).

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E1-Rebuttal-0.1/size16_exponential_batchsize128.png)

Figure D.1: Alignment between one-hop DICE-GT (vertical axis) and DICE-E (horizontal axis) on a 16-node exponential graph. Each node uses a 512-sample subset of CIFAR-10 or CIFAR-100. Models are trained for 5 epochs with a batch size of 128 and a learning rate of 0.1.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E1-Rebuttal-0.1/size32_exponential_batchsize128.png)

Figure D.2: Alignment between one-hop DICE-GT (vertical axis) and DICE-E (horizontal axis) on a 32-node exponential graph. Each node uses a 512-sample subset of CIFAR-10 or CIFAR-100. Models are trained for 5 epochs with a batch size of 128 and a learning rate of 0.1.

We conduct additional sensitivity analysis experiments to evaluate the robustness of DICE-E under varying hyperparameters, including learning rate, batch size, and training epoch. These results demonstrate that DICE-E provides a strong approximation of DICE-GT, achieving consistent alignment across datasets (CIFAR-10 and CIFAR-100) and model architectures (CNN and MLP) under different batch sizes, learning rates, and training epochs.

#### D.2.1 Sensitivity Analysis on Batch Size

We conduct experiments to evaluate the robustness of DICE-E under varying batch sizes.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E1-Rebuttal-0.1/size32_ring_batchsize16.png)

Figure D.3: Alignment between one-hop DICE-GT (vertical axis) and DICE-E (horizontal axis) on a 32-node ring graph. Each node uses a 512-sample subset of CIFAR-10 or CIFAR-100. Models are trained for 5 epochs with a batch size of 16 and a learning rate of 0.1.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E1-Rebuttal-0.1/size32_ring_batchsize64.png)

Figure D.4: Alignment between one-hop DICE-GT (vertical axis) and DICE-E (horizontal axis) on a 32-node ring graph. Each node uses a 512-sample subset of CIFAR-10 or CIFAR-100. Models are trained for 5 epochs with a batch size of 64 and a learning rate of 0.1.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E1-Rebuttal-0.1/size32_ring_batchsize128.png)

Figure D.5: Alignment between one-hop DICE-GT (vertical axis) and DICE-E (horizontal axis) on a 32-node ring graph. Each node uses a 512-sample subset of CIFAR-10 or CIFAR-100. Models are trained for 5 epochs with a batch size of 128 and a learning rate of 0.1.

#### D.2.2 Sensitivity Analysis on Learning Rate and the Number of Nodes

We also condcut experiments to evaluate the robustness of DICE-E under varying learning rates and the number of nodes.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E1-Rebuttal-0.1/size16_ring_batchsize64.png)

Figure D.6: Alignment between one-hop DICE-GT (vertical axis) and DICE-E (horizontal axis) on a 16-node ring graph. Each node uses a 512-sample subset of CIFAR-10 or CIFAR-100. Models are trained for 5 epochs with a batch size of 64 and a learning rate of 0.1.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E1-Rebuttal-0.01/size16_ring_batchsize64.png)

Figure D.7: Alignment between one-hop DICE-GT (vertical axis) and DICE-E (horizontal axis) on a 16-node ring graph. Each node uses a 512-sample subset of CIFAR-10 or CIFAR-100. Models are trained for 5 epochs with a batch size of 64 and a learning rate of 0.01.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E1-Rebuttal-0.1/size32_ring_batchsize64.png)

Figure D.8: Alignment between one-hop DICE-GT (vertical axis) and DICE-E (horizontal axis) on a 32-node ring graph. Each node uses a 512-sample subset of CIFAR-10 or CIFAR-100. Models are trained for 5 epochs with a batch size of 64 and a learning rate of 0.1.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E1-Rebuttal-0.01/size32_ring_batchsize64.png)

Figure D.9: Alignment between one-hop DICE-GT (vertical axis) and DICE-E (horizontal axis) on a 32-node ring graph. Each node uses a 512-sample subset of CIFAR-10 or CIFAR-100. Models are trained for 5 epochs with a batch size of 64 and a learning rate of 0.01.

#### D.2.3 Sensitivity Analysis on Training Epochs

We conduct experiments to evaluate the robustness of DICE-E under varying training epochs.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E1-Rebuttal-0.1-Tiny-Epochs/10epochs.jpg)

Figure D.10: Alignment between one-hop DICE-GT (vertical axis) and DICE-E (horizontal axis) on 16 and 32-node exponential graphs. Each node uses a 8192-sample subset of Tiny ImageNet. Models are trained for 10 epochs with a batch size of 128 and a learning rate of 0.1.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E1-Rebuttal-0.1-Tiny-Epochs/20epochs.jpg)

Figure D.11: Alignment between one-hop DICE-GT (vertical axis) and DICE-E (horizontal axis) on 16 and 32-node exponential graphs. Each node uses a 8192-sample subset of Tiny ImageNet. Models are trained for 20 epochs with a batch size of 128 and a learning rate of 0.1.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E1-Rebuttal-0.1-Tiny-Epochs/30epochs.jpg)

Figure D.12: Alignment between one-hop DICE-GT (vertical axis) and DICE-E (horizontal axis) on a 16 and 32-node exponential graph. Each node uses a 8192-sample subset of Tiny ImageNet. Models are trained for 30 epochs with a batch size of 128 and a learning rate of 0.1.

### D.3 Anomaly Detection

We can also use the proximal influence metric to effectively detect anomalies. Specifically, anomalies are identified by observing significantly higher or lower proximal influence values compared to normal data instances. In our setup, anomalies are generated through random label flipping or by adding random Gaussian noise to features. The following Figures illustrates that the most anomalies (in red) is detectable with proximal influence values.

#### D.3.1 Random label flipping

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E2-Rebuttal-0.1/e2-lr0.1-batch16-exponential-size32.png)

Figure D.13: Anomaly detection on exponential graph with 32 nodes. Each node uses a 512-sample subset of CIFAR-10. Models are trained for 5 epochs with a batch size of 16 and a learning rate of 0.1. In a 32-node exponential graph, each participant connects with 5 neighbors, where the neighbor in red is set as an anomaly by random label flipping, while the other four are normal participants.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E2-Rebuttal-0.1/e2-lr0.1-batch64-exponential-size32-.png)

Figure D.14: Anomaly detection on exponential graph with 32 nodes. Each node uses a 512-sample subset of CIFAR-10. Models are trained for 5 epochs with a batch size of 64 and a learning rate of 0.1. In a 32-node exponential graph, each participant connects with 5 neighbors, where the neighbor in red is set as an anomaly by random label flipping, while the other four are normal participants.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E2-Rebuttal-0.1/e2-lr0.1-batch128-exponential-size32-.png)

Figure D.15: Anomaly detection on exponential graph with 32 nodes. Each node uses a 512-sample subset of CIFAR-10. Models are trained for 5 epochs with a batch size of 128 and a learning rate of 0.1. In a 32-node exponential graph, each participant connects with 5 neighbors, where the neighbor in red is set as an anomaly by random label flipping, while the other four are normal participants.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E2-Rebuttal-0.01/e2-lr0.01-batch16-exponential-size32-.png)

Figure D.16: Anomaly detection on exponential graph with 32 nodes. Each node uses a 512-sample subset of CIFAR-10. Models are trained for 5 epochs with a batch size of 16 and a learning rate of 0.01. In a 32-node exponential graph, each participant connects with 5 neighbors, where the neighbor in red is set as an anomaly by random label flipping, while the other four are normal participants.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E2-Rebuttal-0.01/e2-lr0.01-batch64-exponential-size32-.png)

Figure D.17: Anomaly detection on exponential graph with 32 nodes. Each node uses a 512-sample subset of CIFAR-10. Models are trained for 5 epochs with a batch size of 64 and a learning rate of 0.01. In a 32-node exponential graph, each participant connects with 5 neighbors, where the neighbor in red is set as an anomaly by random label flipping, while the other four are normal participants.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E2-Rebuttal-0.01/e2-lr0.01-batch128-exponential-size32-.png)

Figure D.18: Anomaly detection on exponential graph with 32 nodes. Each node uses a 512-sample subset of CIFAR-10. Models are trained for 5 epochs with a batch size of 128 and a learning rate of 0.01. In a 32-node exponential graph, each participant connects with 5 neighbors, where the neighbor in red is set as an anomaly by random label flipping, while the other four are normal participants.

We can conclude from these experiments that anomalies introduced through random label flipping are readily detectable by analyzing their proximal influence.

#### D.3.2 Feature Perturbations

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E2-Rebuttal-noise/cifar10_epochs50_data512_batchsize128_modeexponential_size32__noniidfalse_chooseepoch5_pretrained0.png)

Figure D.19: Anomaly detection on exponential graph with 32 nodes. Each node uses a 512-sample subset of CIFAR-10. Models are trained for 5 epochs with a batch size of 128 and a learning rate of 0.1. In a 32-node exponential graph, each participant connects with 5 neighbors, where the neighbor in red is set as an anomaly by adding zero-mean Gaussian noise with variance equals 100 on each feature, while the other four are normal participants.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E2-Rebuttal-noise/cifar10_epochs50_data512_batchsize64_modeexponential_size32__noniidfalse_chooseepoch5_pretrained0.png)

Figure D.20: Anomaly detection on exponential graph with 32 nodes. Each node uses a 512-sample subset of CIFAR-10. Models are trained for 5 epochs with a batch size of 64 and a learning rate of 0.01. In a 32-node exponential graph, each participant connects with 5 neighbors, where the neighbor in red is set as an anomaly by adding zero-mean Gaussian noise with variance equals 100 on each feature, while the other four are normal participants.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/E2-Rebuttal-noise/cifar10_epochs50_data512_batchsize16_modeexponential_size32__noniidfalse_chooseepoch5_pretrained0.png)

Figure D.21: Anomaly detection on exponential graph with 32 nodes. Each node uses a 512-sample subset of CIFAR-10. Models are trained for 5 epochs with a batch size of 16 and a learning rate of 0.1. In a 32-node exponential graph, each participant connects with 5 neighbors, where the neighbor in red is set as an anomaly by adding zero-mean Gaussian noise with variance equals 100 on each feature, while the other four are normal participants.

We can conclude from Figure D.19, Figure D.20 and Figure D.21 that most anomalies introduced through adding zero-mean Gaussian noise with high variance are readily detectable by analyzing their proximal influence, which significantly deviates from that of normal data participants.

### D.4 Influence cascade

#### D.4.1 One-hop Influence cascade

The topological dependency of DICE-E in our theory reveals the “power asymmetries” [^5] [^73] in decentralized learning. To support the theoretical finding, we examine the one-hop DICE-E values of the same batch on participants with vastly different topological importance. Figure 1 illustrates the one-hop DICE-E influence scores of an identical data batch across participants during decentralized training of a ResNet-18 model on the CIFAR-10 dataset. Node sizes represent the one-hop DICE-E influence scores, quantifying how a single batch impacts other participants in the network. The dominant nodes (e.g., those with larger outgoing communication weights in $\bm{W}$) exhibit significantly higher influence, as shown in Figure 1 and further detailed in Figure D.23 and Figure D.24. These visualizations underscore the critical role of topological properties in shaping data influence in decentralized learning, demonstrating how the structure of the communication matrix $\bm{W}$ determines the asymmetries in influence.

To better observe and showcase the “influence cascade” phenomenon, we design a communication matrix with one “dominant” participant (node $00$), two “subdominant” participants (nodes $7$ and $10$), and several other common participants. Figure D.22 (Left) visualizes the communication topology, where node sizes indicate out-degree, reflecting their influence, and edge thickness represents the strength of communication links. Node $00$ stands out as the dominant participant with the largest size, while nodes $7$ and $10$ serve as subdominant intermediaries. Figure D.22 (Right) complements this by showing the adjacency matrix $\bm{W}$ as a heatmap, where the color intensity highlights the magnitude of connection strengths, with the dominant participant exhibiting strong outgoing links across the network. Together, these visualizations highlight the hierarchical structure and asymmetries in the communication matrix, crucial for understanding topological influences in decentralized learning.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/Graph.png)

Figure D.22: Left: Visualization of the communication topology used in Section 5, where each node represents a participant, and edges indicate communication links. Node sizes are proportional to their out-degree (sum of outgoing edge weights), reflecting their communication influence within the community. Edge thickness corresponds to the strength of connection (i.e., weight), with directional arrows capturing the flow of information between participants. Self-loops are omitted for simplicity. Right: Heatmap representation of the weighted adjacency matrix 𝑾 \\bm{W} bold\_italic\_W used in, where each entry k, j subscript 𝑘 𝑗 \\bm{W}\_{k,j} bold\_italic\_W start\_POSTSUBSCRIPT italic\_k, italic\_j end\_POSTSUBSCRIPT quantifies the communication strength from participant italic\_j to italic\_k. The color intensity represents the magnitude of the weights.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/e3-mlp-mnist.png)

Figure D.23: Visualization of one-hop influence cascade during decentralized trainingg with MLP on MNIST (left) and CIFAR-10 (right) under a designed communication matrix (see Figure D.22 ). The thickness of edges represents the strength of communication links (i.e., weights in 𝑾 \\bm{W} bold\_italic\_W ), while node sizes correspond to the relative one-hop DICE-E influence scores (see Subsection 4.2 ) computed for the same data batch across different participants. The numerical labels on the nodes indicate the corresponding participants, aligning with the participant indices in.

![Refer to caption](https://arxiv.org/html/2507.06931v1/extracted/6609070/Section/Figure/e3-resnet-cifar10.png)

Figure D.24: Visualization of one-hop influence cascade during decentralized trainingg with ResNet-18 on CIFAR-10 (left) and CIFAR-100 (right) under a designed communication matrix (see Figure D.22 ). The thickness of edges represents the strength of communication links (i.e., weights in 𝑾 \\bm{W} bold\_italic\_W ), while node sizes correspond to the relative one-hop DICE-E influence scores (see Subsection 4.2 ) computed for the same data batch across different participants. The numerical labels on the nodes indicate the corresponding participants, aligning with the participant indices in.

#### D.4.2 Multi-hop Influence cascade

To better illustrate the communication structure underlying the influence cascade phenomenon in multi-hop decentralized learning (see Figure 1), following the setup in Subsection D.4.1, with the only modification being the use of a different mixing matrix. This modification is specifically designed to refine the visualization for better geographic representation, making the spatial relationships of decentralized participants more apparent. The heatmap in Figure D.25 visualizes the corresponding mixing matrix (i.e., weighted adjacency matrix) $\bm{W}$ for the 16-node topology, where each entry $W_{j,k}$ represents the communication strength from participant $j$ to $k$. The color intensity encodes the magnitude of these weights, with warmer colors indicating stronger connections.

![Refer to caption](https://arxiv.org/html/2507.06931v1/x3.png)

Figure D.25: Heatmap representation of the weighted adjacency matrix 𝑾 \\bm{W} bold\_italic\_W used in Figure 1, where each entry k, j subscript 𝑘 𝑗 \\bm{W}\_{k,j} bold\_italic\_W start\_POSTSUBSCRIPT italic\_k, italic\_j end\_POSTSUBSCRIPT quantifies the communication strength from participant italic\_j to italic\_k. The color intensity represents the magnitude of the weights.

[^1]: Youssef Allouah, Anastasia Koloskova, Aymane El Firdoussi, Martin Jaggi, and Rachid Guerraoui. The privacy power of correlated noise in decentralized learning. In *Proceedings of the 41st International Conference on Machine Learning*, volume 235, pp. 1115–1143, 2024.

[^2]: Anthropic. Introducing claude 3.5 sonnet. *Anthropic News*, 2024. URL [https://www.anthropic.com/news/claude-3-5-sonnet](https://www.anthropic.com/news/claude-3-5-sonnet).

[^3]: Samyadeep Basu, Xuchen You, and Soheil Feizi. On second-order group influence functions for black-box predictions. In *Proceedings of the 37th International Conference on Machine Learning*, 2020.

[^4]: Samyadeep Basu, Phil Pope, and Soheil Feizi. Influence functions in deep learning are fragile. In *International Conference on Learning Representations*, 2021.

[^5]: Peter M. Blau. Exchange and power in social life. 1964.

[^6]: Eric Bonabeau, Marco Dorigo, and Guy Theraulaz. *Swarm Intelligence: From Natural to Artificial Systems*. Oxford University Press, 1999.

[^7]: Marco Bornstein, Tahseen Rabbani, Evan Z Wang, Amrit Bedi, and Furong Huang. SWIFT: Rapid decentralized federated learning via wait-free model communication. In *The Eleventh International Conference on Learning Representations*, 2023.

[^8]: Alexander Borzunov, Dmitry Baranchuk, Tim Dettmers, Maksim Riabinin, Younes Belkada, Artem Chumachenko, Pavel Samygin, and Colin Raffel. Petals: Collaborative inference and fine-tuning of large models. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)*, pp. 558–568. Association for Computational Linguistics, 2023a.

[^9]: Alexander Borzunov, Max Ryabinin, Artem Chumachenko, Dmitry Baranchuk, Tim Dettmers, Younes Belkada, Pavel Samygin, and Colin A Raffel. Distributed inference and fine-tuning of large language models over the internet. In *Advances in Neural Information Processing Systems*, 2023b.

[^10]: Ying Cao, Zhaoxian Wu, Kun Yuan, and Ali H Sayed. On the trade-off between flatness and optimization in distributed learning. *arXiv preprint arXiv:2406.20006*, 2024.

[^11]: CCAF. Cambridge bitcoin electricity consumption index (CBECI). [https://ccaf.io/cbnsi/cbeci](https://ccaf.io/cbnsi/cbeci), 2023.

[^12]: Guillaume Charpiat, Nicolas Girard, Loris Felardos, and Yuliya Tarabalka. Input similarity from the neural network perspective. In *Advances in Neural Information Processing Systems*, 2019.

[^13]: Samprit Chatterjee, R. Dennis Cook, and Sanford Weisberg. Residuals and influence in regression. 1982.

[^14]: Daiwei Chen, Jane Zhang, and Ramya Korlakai Vinayak. Unraveling the impact of training samples. In *ICLR Blogposts 2024*, 2024a. URL [https://iclr-blogposts.github.io/2024/blog/unraveling-the-impact-of-training-samples/](https://iclr-blogposts.github.io/2024/blog/unraveling-the-impact-of-training-samples/). https://iclr-blogposts.github.io/2024/blog/unraveling-the-impact-of-training-samples/.

[^15]: Lesi Chen, Haishan Ye, and Luo Luo. An efficient stochastic algorithm for decentralized nonconvex-strongly-concave minimax optimization. *International Conference on Artificial Intelligence and Statistics*, 2024b.

[^16]: Xuxing Chen, Minhui Huang, Shiqian Ma, and Krishna Balasubramanian. Decentralized stochastic bilevel optimization with improved per-iteration complexity. In *Proceedings of the 40th International Conference on Machine Learning*, volume 202, pp. 4641–4671. PMLR, 2023.

[^17]: R. Dennis Cook. Detection of influential observation in linear regression. *Technometrics*, 19(1):15–18, 1977.

[^18]: Edwige Cyffers, Aurélien Bellet, and Jalaj Upadhyay. Differentially private decentralized learning with random walks. In *Proceedings of the 41st International Conference on Machine Learning*, volume 235, pp. 9762–9783, 2024.

[^19]: Google DeepMind. Introducing gemini 2.0: our new ai model for the agentic era. *The Keyword*, 2024. URL [https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/#gemini-2-0-flash](https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/#gemini-2-0-flash).

[^20]: Ambra Demontis, Marco Melis, Maura Pintor, Matthew Jagielski, Battista Biggio, Alina Oprea, Cristina Nita-Rotaru, and Fabio Roli. Why do adversarial attacks transfer? explaining transferability of evasion and poisoning attacks. In *28th USENIX Security Symposium (USENIX Security 19)*, 2019.

[^21]: Arthur Douillard, Qixuan Feng, Andrei A Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc’Aurelio Ranzato, Arthur Szlam, and Jiajun Shen. Diloco: Distributed low-communication training of language models. *arXiv preprint arXiv:2311.08105*, 2023.

[^22]: Peter F. Drucker. *Innovation and Entrepreneurship*. Perennial Library, 1985.

[^23]: Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.

[^24]: Mathieu Even, Anastasia Koloskova, and Laurent Massoulie. Asynchronous SGD on graphs: a unified framework for asynchronous decentralized and federated optimization. In *Proceedings of The 27th International Conference on Artificial Intelligence and Statistics*, 2024.

[^25]: Alireza Fallah, Michael I Jordan, Ali Makhdoumi, and Azarakhsh Malekian. On three-layer data markets. *arXiv preprint arXiv:2402.09697*, 2024.

[^26]: Ernst Fehr and Simon Gächter. Fairness and retaliation: The economics of reciprocity. *Journal of Economic Perspectives*, 14(3):159–181, 2000.

[^27]: Shaopeng Fu, Fengxiang He, and Dacheng Tao. Knowledge removal in sampling-based bayesian inference. In *International Conference on Learning Representations*, 2022.

[^28]: Hongchang Gao and Heng Huang. Fast training method for stochastic compositional optimization problems. *Advances in Neural Information Processing Systems*, 34:25334–25345, 2021.

[^29]: Hongchang Gao, Bin Gu, and My T. Thai. On the convergence of distributed stochastic bilevel optimization algorithms over a network. In *Proceedings of The 26th International Conference on Artificial Intelligence and Statistics*, volume 206, pp. 9238–9281. PMLR, 2023.

[^30]: Anissa Gardizy and Amir Efrati. Microsoft and OpenAI plot $100 billion stargate AI supercomputer. *The Information*, 2024. URL [https://www.theinformation.com/articles/microsoft-and-openai-plot-100-billion-stargate-ai-supercomputer](https://www.theinformation.com/articles/microsoft-and-openai-plot-100-billion-stargate-ai-supercomputer).

[^31]: Amirata Ghorbani and James Zou. Data shapley: Equitable valuation of data for machine learning. In *Proceedings of the 36th International Conference on Machine Learning*, 2019.

[^32]: Avishek Ghosh, Jichan Chung, Dong Yin, and Kannan Ramchandran. An efficient framework for clustered federated learning. In *Advances in Neural Information Processing Systems*, 2020.

[^33]: Alvin W. Gouldner. The norm of reciprocity: A preliminary statement. *American Sociological Review*, 25(2):161–178, 1960.

[^34]: Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, et al. Studying large language model generalization with influence functions. *arXiv preprint arXiv:2308.03296*, 2023.

[^35]: Chuan Guo, Tom Goldstein, Awni Hannun, and Laurens Van Der Maaten. Certified data removal from machine learning models. In *Proceedings of the 37th International Conference on Machine Learning*, 2020.

[^36]: Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

[^37]: Han Guo, Nazneen Rajani, Peter Hase, Mohit Bansal, and Caiming Xiong. FastIF: Scalable influence functions for efficient model interpretation and debugging. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, 2021.

[^38]: Mert Gurbuzbalaban, Yuanhan Hu, Umut Simsekli, Kun Yuan, and Lingjiong Zhu. Heavy-tail phenomenon in decentralized sgd. *arXiv preprint arXiv:2205.06689*, 2022.

[^39]: Zayd Hammoudeh and Daniel Lowd. Identifying a training-set attack’s target using renormalized influence estimation. In *Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security*, 2022.

[^40]: Zayd Hammoudeh and Daniel Lowd. Training data influence analysis and estimation: a survey. *Machine Learning*, 113(5):2351–2403, 2024.

[^41]: Frank R. Hampel. The influence curve and its role in robust estimation. *Journal of the American Statistical Association*, 69(346):383–393, 1974.

[^42]: Xiaochuang Han, Byron C. Wallace, and Yulia Tsvetkov. Explaining black box predictions and unveiling data artifacts through influence functions. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 2020.

[^43]: Filip Hanzely and Peter Richtárik. Federated learning of a mixture of global and local models. *arXiv preprint arXiv:2002.05516*, 2020.

[^44]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In *European conference on computer vision*, 2016.

[^45]: Lie He, Sai Praneeth Karimireddy, and Martin Jaggi. Byzantine-robust decentralized learning via clippedgossip. *arXiv preprint arXiv:2202.01545*, 2022.

[^46]: Anson Ho, Tamay Besiroglu, Ege Erdil, David Owen, Robi Rahman, Zifan Carl Guo, David Atkinson, Neil Thompson, and Jaime Sevilla. Algorithmic progress in language models. *arXiv preprint arXiv:2403.05812*, 2024.

[^47]: Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Thomas Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karén Simonyan, Erich Elsen, Oriol Vinyals, Jack Rae, and Laurent Sifre. An empirical analysis of compute-optimal large language model training. In *Advances in Neural Information Processing Systems*, 2022.

[^48]: Tzu-Heng Huang, Harit Vishwakarma, and Frederic Sala. Train ’n trade: Foundations of parameter markets. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.

[^49]: Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. Datamodels: Understanding predictions with data and data with predictions. In *Proceedings of the 39th International Conference on Machine Learning*, 2022.

[^50]: Andrew Ilyas, Kristian Georgiev, Logan Engstrom, and Sung Min (Sam) Park. Data attribution at scale, 2024. URL [https://ml-data-tutorial.org/](https://ml-data-tutorial.org/). ICML 2024 Tutorial.

[^51]: Sami Jaghouar, Jack Min Ong, Manveer Basra, Fares Obeid, Jannik Straube, Michael Keiblinger, Elie Bakouch, Lucas Atkins, Maziyar Panahi, Charles Goddard, et al. Intellect-1 technical report. *arXiv preprint arXiv:2412.01152*, 2024.

[^52]: Matthew Jagielski, Giorgio Severi, Niklas Pousette Harger, and Alina Oprea. Subpopulation data poisoning attacks. In *Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security*, 2021.

[^53]: Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nick Hynes, Nezihe Merve Gürel, Bo Li, Ce Zhang, Dawn Song, and Costas J. Spanos. Towards efficient data valuation based on the shapley value. In *Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics*, 2019.

[^54]: Heasung Kim, Hyeji Kim, and Gustavo De Veciana. Clustered federated learning via gradient-based partitioning. In *Proceedings of the 41st International Conference on Machine Learning*, 2024.

[^55]: Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In *International conference on machine learning*, pp. 1885–1894. PMLR, 2017a.

[^56]: Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In *Proceedings of the 34th International Conference on Machine Learning*, 2017b.

[^57]: Anastasia Koloskova, Nicolas Loizou, Sadra Boreiri, Martin Jaggi, and Sebastian Stich. A unified theory of decentralized SGD with changing topology and local updates. In *International Conference on Machine Learning*, 2020.

[^58]: Lingjing Kong, Tao Lin, Anastasia Koloskova, Martin Jaggi, and Sebastian Stich. Consensus control for decentralized deep learning. In *Proceedings of the 38th International Conference on Machine Learning*, 2021.

[^59]: Jesper Kristensen, David Wender, and Carl Anthony. Commodification of compute. *arXiv preprint arXiv:2406.19261*, 2024.

[^60]: Alex Krizhevsky, G Hinton, et al. Learning multiple layers of features from tiny images (tech. rep.). *University of Toronto*, 2009.

[^61]: Ya Le and Xuan Yang. Tiny imagenet visual recognition challenge. *CS 231N*, 2015.

[^62]: Batiste Le Bars, Aur’elien Bellet, Marc Tommasi, Erick Lavoie, and Anne-Marie Kermarrec. Refined convergence and topology learning for decentralized sgd with heterogeneous data. In *Proceedings of The 26th International Conference on Artificial Intelligence and Statistics*, volume 206, pp. 1672–1702. PMLR, 25–27 Apr 2023.

[^63]: Batiste Le Bars, Aurélien Bellet, Marc Tommasi, Kevin Scaman, and Giovanni Neglia. Improved stability and generalization guarantees of the decentralized SGD algorithm. In *Proceedings of the 41st International Conference on Machine Learning*, 2024.

[^64]: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998.

[^65]: Peizhao Li and Hongfu Liu. Achieving fairness at no utility cost via data reweighing with influence. In *Proceedings of the 39th International Conference on Machine Learning*, 2022.

[^66]: Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. *arXiv preprint arXiv:2309.05463*, 2023.

[^67]: Xiangru Lian, Ce Zhang, Huan Zhang, Cho-Jui Hsieh, Wei Zhang, and Ji Liu. Can decentralized algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic gradient descent. In *Advances in Neural Information Processing Systems*, 2017.

[^68]: Xiangru Lian, Wei Zhang, Ce Zhang, and Ji Liu. Asynchronous decentralized parallel stochastic gradient descent. In *International Conference on Machine Learning*, 2018.

[^69]: Shayne Longpre, Robert Mahari, Ariel Lee, Campbell Lund, Hamidah Oderinwale, William Brannon, Nayan Saxena, Naana Obeng-Marnu, Tobin South, Cole Hunter, et al. Consent in crisis: The rapid decline of the ai data commons. *arXiv preprint arXiv:2407.14933*, 2024.

[^70]: Noel Loo, Ramin Hasani, Mathias Lechner, and Daniela Rus. Dataset distillation with convexified implicit gradients. In *Proceedings of the 40th International Conference on Machine Learning*, pp. 22649–22674, 2023.

[^71]: Cassio G. Lopes and Ali H. Sayed. Diffusion least-mean squares over adaptive networks: Formulation and performance analysis. *IEEE Transactions on Signal Processing*, 56(7), 2008.

[^72]: Yucheng Lu and Christopher De Sa. Optimal complexity in decentralized training. In *Proceedings of the 38th International Conference on Machine Learning*, 2021.

[^73]: Joe C Magee and Adam D Galinsky. Social hierarchy: The self-reinforcing nature of power and status. *Academy of Management Annals*, 2(1):351–398, 2008.

[^74]: Yishay Mansour, Mehryar Mohri, Jae Ro, and Ananda Theertha Suresh. Three approaches for personalization with applications to federated learning. *arXiv preprint arXiv:2002.10619*, 2020.

[^75]: Enrique Tomás Martínez Beltrán, Mario Quiles Pérez, Pedro Miguel Sánchez Sánchez, Sergio López Bernal, Gérôme Bovet, Manuel Gil Pérez, Gregorio Martínez Pérez, and Alberto Huertas Celdrán. Decentralized federated learning: Fundamentals, state of the art, frameworks, trends, and challenges. *IEEE Communications Surveys & Tutorials*, 25(4):2983–3013, 2023.

[^76]: Nestor Maslej, Loredana Fattorini, Raymond Perrault, Vanessa Parli, Anka Reuel, Erik Brynjolfsson, John Etchemendy, Katrina Ligett, Terah Lyons, James Manyika, Juan Carlos Niebles, Yoav Shoham, Russell Wald, and Jack Clark. The AI index 2024 annual report. Technical report, AI Index Steering Committee, Institute for Human-Centered AI, Stanford University, 2024.

[^77]: Michalis Mavrovouniotis, Changhe Li, and Shengxiang Yang. A survey of swarm intelligence for dynamic optimization: Algorithms and applications. *Swarm and Evolutionary Computation*, 33:1–17, 2017.

[^78]: Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics*, 2017.

[^79]: Abdellah El Mrini, Edwige Cyffers, and Aurélien Bellet. Privacy attacks in decentralized learning. In *Proceedings of the 41st International Conference on Machine Learning*, 2024.

[^80]: Giorgi Nadiradze, Amirmojtaba Sabour, Peter Davies, Shigang Li, and Dan Alistarh. Asynchronous decentralized sgd with quantized and local updates. *Advances in Neural Information Processing Systems*, 2021.

[^81]: Angelia Nedi’c and Alex Olshevsky. Distributed optimization over time-varying directed graphs. volume 60, pp. 601–615. IEEE, 2014.

[^82]: Angelia Nedic and Asuman Ozdaglar. Distributed subgradient methods for multi-agent optimization. *IEEE Transactions on Automatic Control*, 54(1):48–61, 2009.

[^83]: Peter Nickl, Lu Xu, Dharmesh Tailor, Thomas Möllenhoff, and Mohammad Emtiyaz E Khan. The memory-perturbation equation: Understanding model's sensitivity to data. In *Advances in Neural Information Processing Systems*, 2023.

[^84]: OpenAI. Learning to reason with llms. *OpenAI Blog*, 2024. URL [https://openai.com/index/learning-to-reason-with-llms/](https://openai.com/index/learning-to-reason-with-llms/).

[^85]: OpenAI. Announcing the stargate project. [https://openai.com/index/announcing-the-stargate-project/](https://openai.com/index/announcing-the-stargate-project/), 2025.

[^86]: Seulki Park, Jongin Lim, Younghan Jeon, and Jin Young Choi. Influence-balanced loss for imbalanced visual classification. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021.

[^87]: Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon LLM: Outperforming curated corpora with web data only. In *Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track*, 2023.

[^88]: Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. In *Advances in Neural Information Processing Systems*, 2020.

[^89]: Dominic Richards et al. Graph-dependent implicit regularisation for distributed stochastic subgradient descent. *Journal of Machine Learning Research*, 2020.

[^90]: David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. *Learning internal representations by error propagation*. MIT Press, 1986.

[^91]: Max Ryabinin, Tim Dettmers, Michael Diskin, and Alexander Borzunov. SWARM parallelism: Training large models can be surprisingly communication-efficient. In *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pp. 29416–29440. PMLR, 2023.

[^92]: Felix Sattler, Klaus-Robert Müller, and Wojciech Samek. Clustered federated learning: Model-agnostic distributed multitask optimization under privacy constraints. *IEEE Transactions on Neural Networks and Learning Systems*, 32(8):3710–3722, 2021.

[^93]: Ali H. Sayed. *Adaptation, Learning, and Optimization over Networks*. Now Publishers, 2014.

[^94]: Andrea Schioppa, Polina Zablotskaia, David Vilar, and Artem Sokolov. Scaling up influence functions. *Proceedings of the AAAI Conference on Artificial Intelligence*, 2022.

[^95]: Ayush Sekhari, Jayadev Acharya, Gautam Kamath, and Ananda Theertha Suresh. Remember what you want to forget: Algorithms for machine unlearning. In *Advances in Neural Information Processing Systems*, 2021.

[^96]: Jaime Sevilla and Edu Roldán. Training compute of frontier ai models grows by 4-5x per year, 2024. URL [https://epochai.org/blog/training-compute-of-frontier-ai-models-grows-by-4-5x-per-year](https://epochai.org/blog/training-compute-of-frontier-ai-models-grows-by-4-5x-per-year).

[^97]: Lloyd S. Shapley. A value for n-person games. In *Contributions to the Theory of Games, Volume II*, chapter 17. Princeton University Press, 1953.

[^98]: Li Shen, Yan Sun, Zhiyuan Yu, Liang Ding, Xinmei Tian, and Dacheng Tao. On efficient training of large-scale deep learning models. *ACM Comput. Surv.*, 57(3), 2024.

[^99]: Abhishek Singha, Charles Lua, Gauri Guptaa, Ayush Chopraa, Jonas Blanca, Tzofi Klinghoffera, Kushagra Tiwarya, and Ramesh Raskara. A perspective on decentralizing ai. 2024.

[^100]: Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari Morcos. Beyond neural scaling laws: beating power law scaling via data pruning. In *Advances in Neural Information Processing Systems*, 2022.

[^101]: Tao Sun, Dongsheng Li, and Bao Wang. Stability and generalization of decentralized stochastic gradient descent. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 2021.

[^102]: Mukund Sundararajan and Walid Krichene. Inflow, outflow, and reciprocity in machine learning. In *Proceedings of the 40th International Conference on Machine Learning*, 2023.

[^103]: Canh T. Dinh, Nguyen Tran, and Josh Nguyen. Personalized federated learning with moreau envelopes. In *Advances in Neural Information Processing Systems*, 2020.

[^104]: Yuki Takezawa, Ryoma Sato, Han Bao, Kenta Niwa, and Makoto Yamada. Beyond exponential graph: Communication-efficient topologies for decentralized learning via finite-time convergence. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.

[^105]: Hanlin Tang, Xiangru Lian, Ming Yan, Ce Zhang, and Ji Liu. D2: Decentralized training over decentralized data. In *International Conference on Machine Learning*. PMLR, 2018.

[^106]: Naoyuki Terashita and Satoshi Hara. Decentralized hyper-gradient computation over time-varying directed networks. *arXiv preprint arXiv:2210.02129*, 2022.

[^107]: J. Tsitsiklis, D. Bertsekas, and M. Athans. Distributed asynchronous deterministic and stochastic gradient optimization algorithms. *IEEE Transactions on Automatic Control*, 31(9):803–812, 1986.

[^108]: Vladimir Vapnik and Alexey Chervonenkis. *Theory of Pattern Recognition*. Nauka, Moscow, 1974.

[^109]: Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, and Marius Hobbhahn. Position: Will we run out of data? Limits of LLM scaling based on human-generated data. In *Proceedings of the 41st International Conference on Machine Learning*, 2024.

[^110]: Thijs Vogels, Lie He, Anastasiia Koloskova, Sai Praneeth Karimireddy, Tao Lin, Sebastian U Stich, and Martin Jaggi. Relaysum for decentralized deep learning on heterogeneous data. *Advances in Neural Information Processing Systems*, 34:28004–28015, 2021.

[^111]: Thijs Vogels, Hadrien Hendrikx, and Martin Jaggi. Beyond spectral gap: The role of the topology in decentralized learning. *Journal of Machine Learning Research*, 24(355):1–31, 2023.

[^112]: Guan Wang, Charlie Xiaoqian Dang, and Ziye Zhou. Measure contribution of participants in federated learning. In *2019 IEEE International Conference on Big Data (Big Data)*, pp. 2597–2604, 2019.

[^113]: Jiachen T Wang, Prateek Mittal, Dawn Song, and Ruoxi Jia. Data shapley in one training run. *arXiv preprint arXiv:2406.11011*, 2024.

[^114]: Jue Wang, Yucheng Lu, Binhang Yuan, Beidi Chen, Percy Liang, Christopher De Sa, Christopher Re, and Ce Zhang. CocktailSGD: Fine-tuning foundation models over 500Mbps networks. In *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pp. 36058–36076. PMLR, 2023a.

[^115]: Tianhao Wang, Johannes Rausch, Ce Zhang, Ruoxi Jia, and Dawn Song. *A Principled Approach to Data Valuation for Federated Learning*, pp. 153–167. Springer International Publishing, 2020.

[^116]: Xinran Wang, Qi Le, Ahmad Faraz Khan, Jie Ding, and Ali Anwar. A framework for incentivized collaborative learning. *arXiv preprint arXiv:2305.17052*, 2023b.

[^117]: Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. LESS: Selecting influential data for targeted instruction tuning. In *Proceedings of the 41st International Conference on Machine Learning*, pp. 54104–54132, 2024.

[^118]: Wenhan Xian, Feihu Huang, Yanfu Zhang, and Heng Huang. A faster decentralized algorithm for nonconvex minimax problems. *Advances in Neural Information Processing Systems*, 34:25865–25877, 2021.

[^119]: Ran Xin, Chenguang Xi, and Usman A. Khan. Frost—fast row-stochastic optimization with uncoordinated step-sizes. *EURASIP Journal on Advances in Signal Processing*, 2019(1):1, 2019.

[^120]: Jie Xu, Wei Zhang, and Fei Wang. A(dp) <sup>2</sup> sgd: Asynchronous decentralized parallel stochastic gradient descent with differential privacy. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2021.

[^121]: Shuo Yang, Zeke Xie, Hanyu Peng, Min Xu, Mingming Sun, and Ping Li. Dataset pruning: Reducing training data by examining generalization influence. In *The Eleventh International Conference on Learning Representations*, 2023.

[^122]: Shuoguang Yang, Xuezhou Zhang, and Mengdi Wang. Decentralized gossip-based stochastic bilevel optimization over communication networks. *Advances in Neural Information Processing Systems*, 35:238–252, 2022.

[^123]: Haoxiang Ye and Qing Ling. Generalization error matters in decentralized learning under Byzantine attacks. *IEEE Transactions on Signal Processing*, 2025.

[^124]: Bicheng Ying, Kun Yuan, Yiming Chen, Hanbin Hu, Pan Pan, and Wotao Yin. Exponential graph is provably efficient for decentralized deep training. In *Advances in Neural Information Processing Systems*, 2021.

[^125]: Han Yu, Zelei Liu, Yang Liu, Tianjian Chen, Mingshu Cong, Xi Weng, Dusit Niyato, and Qiang Yang. A fairness-aware incentive scheme for federated learning. In *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society*, 2020.

[^126]: Haoxiang Yu, Hsiao-Yuan Chen, Sangsu Lee, Sriram Vishwanath, Xi Zheng, and Christine Julien. idml: Incentivized decentralized machine learning. *arXiv preprint arXiv:2304.05354*, 2023.

[^127]: Binhang Yuan, Yongjun He, Jared Quincy Davis, Tianyi Zhang, Tri Dao, Beidi Chen, Percy Liang, Christopher Re, and Ce Zhang. Decentralized training of foundation models in heterogeneous environments. *Advances in Neural Information Processing Systems*, 2022.

[^128]: Kun Yuan, Qing Ling, and Wotao Yin. On the convergence of decentralized gradient descent. *SIAM Journal on Optimization*, 26(3):1835–1854, 2016a.

[^129]: Kun Yuan, Qing Ling, and Wotao Yin. On the convergence of decentralized gradient descent. *SIAM Journal on Optimization*, 26(3):1835–1854, 2016b.

[^130]: Kun Yuan, Bicheng Ying, Xiaochuan Zhao, and Ali H. Sayed. Exact diffusion for distributed optimization and learning—part i: Algorithm development. *IEEE Transactions on Signal Processing*, 67(3):708–723, 2019.

[^131]: Liangqi Yuan, Ziran Wang, Lichao Sun, Philip S. Yu, and Christopher G. Brinton. Decentralized federated learning: A survey and perspective. *IEEE Internet of Things Journal*, pp. 1–1, 2024.

[^132]: Shahryar Zehtabi, Dong-Jun Han, Rohit Parasnis, Seyyedali Hosseinalipour, and Christopher G Brinton. Decentralized sporadic federated learning: A unified algorithmic framework with convergence guarantees. In *The Thirteenth International Conference on Learning Representations*, 2025.

[^133]: Rongfei Zeng, Chao Zeng, Xingwei Wang, Bo Li, and Xiaowen Chu. A comprehensive survey of incentive mechanism for federated learning. *arXiv preprint arXiv:2106.15406*, 2021.

[^134]: Chang Zhang, Shunkun Yang, Lingfeng Mao, and Huansheng Ning. Anomaly detection and defense techniques in federated learning: a comprehensive review. *Artificial Intelligence Review*, 57(6):150, 2024.

[^135]: Miaoxi Zhu, Li Shen, Bo Du, and Dacheng Tao. Stability and generalization of the decentralized stochastic gradient descent ascent algorithm. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023a.

[^136]: Tongtian Zhu, Fengxiang He, Lan Zhang, Zhengyang Niu, Mingli Song, and Dacheng Tao. Topology-aware generalization of decentralized SGD. In *International Conference on Machine Learning*. PMLR, 2022.

[^137]: Tongtian Zhu, Fengxiang He, Kaixuan Chen, Mingli Song, and Dacheng Tao. Decentralized SGD and average-direction SAM are asymptotically equivalent. In *Proceedings of the 40th International Conference on Machine Learning*, 2023b.