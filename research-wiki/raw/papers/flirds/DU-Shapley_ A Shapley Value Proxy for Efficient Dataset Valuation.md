---
title: "DU-Shapley: A Shapley Value Proxy for Efficient Dataset Valuation"
source: "https://arxiv.org/html/2306.02071v3"
author:
published:
created: 2026-05-05
description:
tags:
  - "clippings"
---
corollarytheorem corollary propositiontheorem proposition definitiontheorem definition remarktheorem remark

Felipe Garrido-Lucero\*  
Inria, Fairplay joint team  
Palaiseau, France  
felipe.garrido-lucero@irit.fr  
\* Equal contribution  
&Benjamin Heymann\*  
Criteo AI Lab  
Paris, France  
b.heymann@criteo.com  
\* Equal contribution  
&Maxime Vono\*  
Criteo AI Lab  
Paris, France  
m.vono@criteo.com  
\* Equal contribution  
&Patrick Loiseau  
Inria, Fairplay joint team  
Palaiseau, France  
patrick.loiseau@inria.fr  
&Vianney Perchet  
ENSAE, FairPlay joint team  
Palaiseau, France  
vianney@ensae.fr

###### Abstract

We consider the dataset valuation problem, that is, the problem of quantifying the incremental gain, to some relevant pre-defined utility of a machine learning task, of aggregating an individual dataset to others. The Shapley value is a natural tool to perform dataset valuation due to its formal axiomatic justification, which can be combined with Monte Carlo integration to overcome the computational tractability challenges. Such generic approximation methods, however, remain expensive in some cases. In this paper, we exploit the knowledge about the structure of the dataset valuation problem to devise more efficient Shapley value estimators. We propose a novel approximation, referred to as discrete uniform Shapley, which is expressed as an expectation under a discrete uniform distribution with support of reasonable size. We justify the relevancy of the proposed framework via asymptotic and non-asymptotic theoretical guarantees and illustrate its benefits via an extensive set of numerical experiments.

## 1 Introduction

One of the main challenges for training machine learning (ML) models with enough generalization capabilities is to access a sufficiently large set of labeled training data. These data often exist but are commonly spread across many parties, impairing their usage in a direct and simple way. Real world examples range from the advertising industry, where different retailers hold sets of observations with either similar or complementary features from consented data about browsing and shopping habits of individual users; to the medical sector where hospitals may improve their diagnostics accuracy by sharing their data. By collaborating with each other and pooling their individual datasets together, these dataset owners could learn better ML models for their applications. Naturally, many questions raise from such collaborations. Federated learning [^8] [^9], for example, addresses the issues related to the practical ways that dataset owners can share their data. We consider a complementary problem to the one in federated learning: measuring the additional value each party would obtain by participating in the joint ML effort. In order to compute or estimate compensating rewards allowing to incentivize parties to share their data, a first stage that is commonly considered in the literature is to perform so-called *dataset valuation* [^1] [^42] [^44].

Motivated by natural properties expected for fair valuation, different solution concepts from cooperative game theory [^3] have been considered, the Shapley value [^40] being arguably the most broadly studied valuation scheme in ML due to its axiomatic justification. Agarwal et al. [^1] designed a data marketplace and used the Shapley value to allocate the data among buyers. Tay et al. [^44] considered a cooperative environment where agents can jointly train a generative model, from which synthetic data are drawn and distributed to the parties according to their Shapley values. Sim et al. [^42] rewarded parties based on the Shapley value and information gain on model parameters. The critical challenge when using the Shapley value is its well-known computational intractability. To cope with it, [^1] [^44] considered Monte Carlo (MC) approximations, while [^42] worked with a small set of three players. This approximation methods, however, remain expensive whenever computing the marginal contributions involve retraining. Moreover, they are generic and do not use the specific structure of the dataset valuation problem at stake, leaving open the possibility to find more adapted approximations for that problem.

The Shapley value was also used in the related problem of data valuation. Data valuation measures the contribution of a single data point within a dataset in the training of a given prediction model. Several solution concepts based on the Shapley value have been proposed for the data valuation problem including Data Shapley [^10] [^16], DShapley [^11] [^24], Beta Shapley [^23] or CS-Shapley [^38], together with different MC variants to cope with the computational intractability issue. For the data valuation problem, the structure was exploited to give easier-to-compute solutions in certain cases, in particular for the $k$ -nearest neighbor problem [^12] [^15] [^25] [^26] [^35] [^41] [^46]. Unlike data valuation, however, dataset valuation aims at quantifying the marginal contribution of a whole dataset to a given ML task with respect to (w.r.t.) the datasets brought by other dataset owners. Although data and dataset valuation are related problems, they are different and the techniques developed for data valuation cannot be used for the dataset valuation problem that we study (we further develop this point in Section 2.3).

Contributions. We consider the dataset valuation problem. Following the ML literature, we model it as a cooperative game whose value function relates to the considered ML task, and aim at estimating the Shapley value to measure the dataset owners contribution. We propose a new way to address the computational intractability issue of the Shapley value. Instead of relying on generic MC approximation schemes, our approximation method leverages the structure of the dataset valuation problem as well as a convergence result for a key random variable of the problem. Our approximation behaves well in many cases, both theoretically and empirically. More specifically, our main contributions can be summarized as follows:

1. We propose DU-Shapley (Section 3.2), a novel Shapley value approximation that exponentially reduces the number of utility function valuations required for the computation. This is the first dataset valuation approach leveraging the specific structure of the utility function.
2. Based on three different use-cases, we establish asymptotic and non-asymptotic theoretical guarantees for DU-Shapley, showing notably that it converges almost surely to the Shapley value as the number of dataset owners grows.
3. We assess the benefits of the proposed methodology using extensive numerical experiments on both Shapley value approximation and dataset valuation use-cases. We show, in particular, that DU-Shapley outperforms all considered MC approximations of the Shapley value.

Additional Related Work. Cooperative game theory has been applied to solve multi-agents ML problems beyond data and dataset valuation [^6] [^18] [^29] [^47]. In particular, the Shapley value has been used to solve several problems including variable selection [^5], feature importance [^7] [^27] [^28], or model interpretation [^4]. In these problems, similarly to the data and dataset valuation problems, the computational intractability issue of the Shapley value is usually addressed via MC [^2] [^31] [^32].

## 2 Problem Formulation and Main Concepts Involved

This section presents the dataset valuation problem we aim to solve, along with preliminaries including the definition and classical approximations of the Shapley value. For $n\in\mathbb{N}$ and $A$, we denote $[n]:=\{1,..,n\}$ and $\mathrm{U}(A)$ the uniform distribution with support on $A$.

### 2.1 Generic Model

We consider a collaborative ML setting involving a set $\mathcal{I}$ of $I=|\mathcal{I}|\in\mathbb{N}^{*}$ dataset owners, also referred to as players in the sequel, who are willing to cooperate in order to solve a common ML problem. Each player $i\in\mathcal{I}$ is assumed to possess an individual dataset $\mathrm{D}_{i}=\{(x_{i}^{(j)},y_{i}^{(j)})\}_{j\in[n_{i}]}$ where $x_{i}^{(j)}\in\mathcal{X}\subset\mathbb{R}^{d}$ stands for a feature vector, $y_{i}^{(j)}\in\mathcal{Y}$ is a label, $n_{i}=|\mathrm{D}_{i}|$ refers to the number of data points in $\mathrm{D}_{i}$, and samples are drawn independently from a player-dependent distribution $p_{i}$, i.e., $(x_{i}^{(j)},y_{i}^{(j)})\sim p_{i}$, for all $j\in[n_{i}]$ and $i\in\mathcal{I}$.

Our basic motivation is to quantify the incremental contribution that a given player $i\in\mathcal{I}$ brings by sharing her dataset $\mathrm{D}_{i}$ with other players towards solving some ML task. Hence, we are interested in scenarios in which, even though the data distribution might differ across players, they face a similar ML task, for instance the minimization of the expectation (with respect to $p_{i}$) of some loss function $\ell(\hat{Y},Y)$, where $\hat{Y}$ denotes a prediction of $Y$. In such cases, players can usually learn from others’ datasets, in the sense that given some $X$, the optimal prediction $\hat{Y}$ that minimizes $\mathbb{E}[\ell(\hat{Y},Y)|X]$ is the same for all player. This holds, *e.g.*, if the conditional distributions (or, in many cases, simply the conditional expectation) of $y^{(j)}$ given $x^{(j)}$ are the same but the marginal distributions of $x^{(j)}$ differ.

To model this problem with full generality, we assume that the players $i\in\mathcal{I}$ collaborate in solving an ML task whose success is measured through some abstract metric $u$ that maps any dataset to a real number (say, the prediction accuracy in a classification problem). With a slight abuse of notation, for any coalition of players $\mathcal{S}\subseteq\mathcal{I}$, we define $u(\mathcal{S})=u(\mathrm{D}_{\mathcal{S}})$, where $\mathrm{D}_{\mathcal{S}}:=\cup_{i\in\mathcal{S}}\mathrm{D}_{i}$. Hence, $u:2^{\mathcal{I}}\rightarrow\mathbb{R}$ can be seen as a game-theoretical utility function that quantifies how well coalitions of players can solve the considered ML task based on the union of their datasets.

The following subsections provide three theoretical use-cases that instantiate the generic model and give specific utility functions $u$ to illustrate the dataset valuation problem. Using different tools and techniques, Section 3 provides theoretical guarantees in each of them. These theoretical results are then complemented in Section 4 by numerical evidence of our proposed approach in more intricate practical problems on real data.

#### 2.1.1 Theoretical use-case 1: Non-parametric Regression

The first use case we shall investigate is quite generic and consists in non-parametric regression. We assume the existence of a function $f^{*}$ such that $y^{(j)}_{i}=f^{*}(x_{i}^{(j)})+\eta^{(j)}_{i}$ with $\eta^{(j)}_{i}$ i.i.d., and a quadratic loss function. Without regularity assumption on $f^{*}(\cdot)$, learning can be arbitrarily slow; hence it is usually assumed that this mapping is Lipschitz (or at least $\beta$ -Hölder [^13] [^45]).

The standard estimation method of $f^{*}$ we shall consider is called the regressogram or binning (also applied in [^13] to study local differential privacy within regression) and consists in learning optimal piece-wise constant functions. More precisely, given some parameter $B\in\mathbb{N}$ —chosen exogeneously as a function of the function regularity $\beta$, the ambient dimension $d$ and the total number $n$ of datapoints, typically $B\simeq n^{\nicefrac{{d}}{{(d+2\beta)}}}$ —, the feature space $\mathcal{X}$ is partitioned into $B$ cubic bins. The excess risk of learning $f^{*}$ can then be decomposed into

$$
\displaystyle\mathbb{E}\bigl{[}(\hat{f}(x)-f^{*}(x))^{2}\bigr{]}=\mathbb{E}%
\bigl{[}(\hat{f}(x)-\bar{f}(x))^{2}\bigr{]}+\mathbb{E}\bigl{[}(\bar{f}(x)-f^{*%
}(x))^{2}\bigr{]},\,
$$

where $\hat{f}$ is the estimator of $f^{*}$, $\bar{f}(x):=\sum_{b\in[B]}\bar{f}_{b}\mathbbm{1}\{x\in b\}$, and $\bar{f}_{b}$ is any value that $f^{*}$ can take on the bin $b$. The second term in (1) being agnostic to the players’ datasets, the problem of measuring the contributions of the players to estimating $f^{*}$ can be decomposed into measuring their contributions to estimating each $\bar{f}_{b}$. In particular, the utility $u(\mathcal{S})$ of a coalition $\mathcal{S}$ can be defined, and split into the sum of $B$ sub-utilities $u_{b}(\mathcal{S})$ functions, as follows

$$
\displaystyle u(\mathcal{S}):=-\mathbb{E}\bigl{[}(\hat{f}_{\mathcal{S}}(x)-%
\bar{f}(x))^{2}\bigr{]}=\sum\nolimits_{b\in[B]}-\mathbb{E}\bigl{[}(\hat{f}_{%
\mathcal{S},b}-\bar{f}_{b})^{2}\bigr{]}\mathbb{P}(x\in b)=:\sum\nolimits_{b\in%
[B]}u_{b}(\mathcal{S})\mathbb{P}(x\in b),
$$

where $\hat{f}_{\mathcal{S}}$ is the estimator of $\bar{f}$ when using the datasets of all players in $\mathcal{S}$ and $\hat{f}_{\mathcal{S},b}$ is the estimator $\bar{f}_{b}$ when using, for all players in $\mathcal{S}$, the datasets of points in the bin $b$. Interestingly, after this reduction, the problem is decomposed into $B$ independent sub-problems—one per bin—, where the utility is a sole function of the number of data points used to estimate $\bar{f}_{b}$, i.e., we can write $u_{b}(\mathcal{S})=w_{b}(\sum_{i\in\mathcal{S}}n_{i,b})$ for some function $w_{b}:\mathbb{N}\to\mathbb{R}$, where $n_{i,b}$ is the number of data points that player $i$ has in the bin $b$. This last property motivates our second theoretical use-case.

#### 2.1.2 Theoretical use-case 2: Homogeneous case

The second theoretical setting considers a general learning problem (not necessarily restricted to regression) and supposes that all players have the same sampling distribution, i.e., it takes $p_{i}=p$ for all $i\in\mathcal{I}$. This homogeneity on the players allows to reduce the problem of measuring the contribution of the players to just counting the number of data points contributed by each of them. Formally, and similarly to the previous use-case, we suppose the existence of a function $w:\mathbb{N}\to\mathbb{R}$ such that $u(\mathcal{S})=w(\sum_{i\in\mathcal{S}}n_{i})$.

#### 2.1.3 Theoretical use-case 3: Heterogeneous Linear Regression - Local Differential Privacy

The third theoretical setting we consider is linear regression with random design and different variance of the features and labels per player. Although the setting is more general, one of the motivations behind it is standard linear regression with homogeneous data between players, but where players can purposely add noise when sharing their dataset (in order to provide Local Differential Privacy, for instance). Formally, for any $i\in\mathcal{I}$, we consider the following linear model that generates the dataset $\mathrm{D}_{i}$ of size $n_{i}$:

$$
\displaystyle y_{i}^{(j)}=x_{i}^{(j)}\theta+\eta_{i}^{(j)}\,,\text{ where }%
\eta_{i}^{(j)}\sim\mathrm{N}(0,\varepsilon_{i}^{2})\,,\text{ and }x_{i}^{(j)}%
\sim\mathrm{N}(0_{d},\sigma_{i}^{2}\mathrm{I}_{d})\,,\text{ for any }j\in[n_{i%
}],\,
$$

with $\theta\in\mathbb{R}^{d}$ a ground-truth parameter, $\sigma_{i}$ positive and known, and $\varepsilon_{i}$ the differential privacy level chosen by player $i$. Under the linear regression framework defined in (3), and following [^8], the utility function of a set $\mathcal{S}\subseteq\mathcal{I}$ of players is defined by the negative expected mean square error over a hold-out dataset, i.e.,

$$
u(\mathcal{S})=-\mathbb{E}\bigl{[}\bigl{(}x^{\top}\hat{\theta}_{\mathcal{S}}-x%
^{\top}\theta\bigr{)}^{2}\bigr{]}\,,
$$

where the expectation is taken over the distribution $p_{\mathrm{test}}$ of a hold-out testing datum $x\in\mathbb{R}^{d}$, the sampling distributions $\mathrm{N}(0,\sigma_{i}^{2}\mathrm{I}_{d})$ for all $i\in\mathcal{S}$, and the linear regression error distributions $\mathrm{N}(0,\varepsilon_{i}^{2}),\forall i\in\mathcal{S},j\in[n_{i}]$, and $\hat{\theta}_{\mathcal{S}}$ stands for the generalized least square estimator defined by $\hat{\theta}_{\mathcal{S}}=(X_{\mathcal{S}}^{\top}\Sigma_{\mathcal{S}}^{-1}X_{%
\mathcal{S}})^{-1}X_{\mathcal{S}}^{\top}\Sigma_{\mathcal{S}}^{-1}Y_{\mathcal{S%
}},\text{ where }\Sigma_{\mathcal{S}}=\mathrm{diag}((\varepsilon^{2}_{i})_{i%
\in\mathcal{S}})\in\mathbb{R}^{|\mathcal{S}|\times|\mathcal{S}|}.$ The notations $X_{\mathcal{S}}$ and $Y_{\mathcal{S}}$ refer to the concatenation of $\{X_{i}\}_{i\in\mathcal{S}}$ and $\{Y_{i}\}_{i\in\mathcal{S}}$, respectively, and $X_{i}\in\mathbb{R}^{n_{i}\times d}$ is defined by $X_{i}=([x_{i}^{(1)}]^{\top},\ldots,[x_{i}^{(n_{i})}]^{\top})^{\top}$ while $Y_{i}\in\mathbb{R}^{n_{i}}$ is defined by $Y_{i}=(y_{i}^{(1)},\ldots,y_{i}^{(n_{i})})^{\top}$.

The following result provides a close-form expression for the utility function in this case:

###### Proposition.

Let $\mathcal{S}$ be a coalition of players and consider the value function as above. It follows,

$$
\displaystyle u(\mathcal{S})=\frac{-\mathrm{Tr}\bigl{[}\mathbb{E}\bigl{[}xx^{%
\top}\bigr{]}\bigr{]}}{q({\mathcal{S}})-d-1},\text{ where }q(\mathcal{S}):=%
\left\lfloor\frac{\bigl{(}\sum_{i\in\mathcal{S}}(\nicefrac{{\sigma_{i}}}{{%
\varepsilon_{i}}})n_{i}\bigr{)}^{2}}{\sum_{i\in\mathcal{S}}\bigl{(}\nicefrac{{%
\sigma_{i}}}{{\varepsilon_{i}}}\bigr{)}^{2}n_{i}}\right\rfloor,\text{ with the%
 convention }q(\varnothing)=0.
$$

In particular, considering $p_{\mathrm{test}}=\mathrm{N}(0,\mathrm{I}_{d})$, we get $u(\mathcal{S})=\frac{d}{d+1-q({\mathcal{S}})}.$

Section 2.1.3 shows that, in this use-case, the utility function can be written as a function $w(q(\mathcal{S}))$ of a scalar quantity $q(\mathcal{S})$ that captures the datasets heterogeneity. Notice that in this use-case, if we add the homogeneity assumption that $\sigma_{i}/\varepsilon_{i}=\sigma/\varepsilon$, for all $i\in\mathcal{I}$, then the term $q(\mathcal{S})$ becomes $\sum_{i\in\mathcal{S}}n_{i}$ and, as a consequence, we get

$$
\displaystyle u(\mathcal{S})=w(q(\mathcal{S}))=w\left(\sum\nolimits_{i\in%
\mathcal{S}}n_{i}\right)=\frac{d}{d+1-\sum_{i\in\mathcal{S}}n_{i}}.
$$

Recall that, in the non-parametric regression use-case, it holds $u(\mathcal{S})=\sum_{b\in[B]}\mathbb{P}(x\in b)w_{b}(q_{b}(\mathcal{S}))$ where $q_{b}(\mathcal{S})=\sum_{i\in\mathcal{S}}n_{i,b}$. Therefore, in our three uses-cases, the utility of a coalition can be summarized as the function of some scalar quantity of interest. This observation will be useful to state later our theoretical results.

### 2.2 Shapley Value

The Shapley value [^40] is a classical solution concept in cooperative game theory to fairly allocate the total gains generated by a coalition of players. Given a utility function $u$, the Shapley value of a player $i$ is defined as the average marginal contribution of her dataset $\mathrm{D}_{i}$ to all possible subsets of $\{\mathrm{D}_{j}\}_{j\in\mathcal{I}\setminus\{i\}}$, built by aggregating the datasets of the other players. Formally, the Shapley value $\varphi_{i}$ of player $i$ writes

$$
\varphi_{i}(u)=\frac{1}{|\Pi(\mathcal{I})|}\sum\nolimits_{\pi\in\Pi(\mathcal{I%
})}[u(\mathcal{P}_{i}^{\pi}\cup\{i\})-u(\mathcal{P}_{i}^{\pi})]\,,
$$

where $\Pi(\mathcal{I})$ refers to the set of permutations over $\mathcal{I}$ and $\mathcal{P}_{i}^{\pi}$ to the set of predecessors of player $i\in\mathcal{I}$ in permutation $\pi\in\Pi(\mathcal{I})$. The Shapley value of player $i$ is equivalently expressed as

$$
\displaystyle\varphi_{i}(u)=\frac{1}{I}\sum\nolimits_{\mathcal{S}\subseteq%
\mathcal{I}\setminus\{i\}}\binom{I-1}{|\mathcal{S}|}^{-1}[u(\mathcal{S}\cup\{i%
\})-u(\mathcal{S})]\,.
$$

The Shapley value has been commonly used in ML and cooperative game theory as it uniquely satisfies the following set of desirable properties.

1. *Efficiency.* $\sum_{i=1}^{I}\varphi_{i}(u)=u(\mathcal{I})$, i.e, the sum of all Shapley values is equal to the value of $\mathcal{I}$.
2. *Symmetry.* If, for any $\mathcal{S}\subseteq\mathcal{I}\setminus\{i_{1},i_{2}\}$, $u(\mathcal{S}\cup\{i_{1}\})=u(\mathcal{S}\cup\{i_{2}\})$, then $\varphi_{i_{1}}(u)=\varphi_{i_{2}}(u)$, i.e., whenever two players have the same marginal contributions, their Shapley values coincide.
3. *Dummy.* If, for any $\mathcal{S}\subseteq\mathcal{I}\setminus\{i\}$, $u(\mathcal{S}\cup\{i\})=u(\mathcal{S})$, then $\varphi_{i}(u)=0$, i.e., whenever a player has null marginal contributions, her Shapley value is zero.
4. *Linearity.* $\varphi_{i}(u_{1}+u_{2})=\varphi_{i}(u_{1})+\varphi_{i}(u_{2})$, i.e., the Shapley value of sums of games is the sum of the Shapley values of the respective games.

MC approximation of the Shapley Value. Evaluating the Shapley value is unfortunately computationally expensive in general. As a consequence, many MC approximations have been considered by sampling with replacement $T$ terms from the sum of either (7) or (8). Regarding (7), this boils down to considering the estimator

$$
\hat{\varphi}_{i}(u)=\frac{1}{T}\sum\nolimits_{t=1}^{T}[u(\mathcal{P}_{i}^{\pi%
_{t}}\cup\{i\})-u(\mathcal{P}_{i}^{\pi_{t}})]\,,\text{where $\pi_{t}\sim%
\mathrm{U}(\Pi(\mathcal{I}))$.}
$$

### 2.3 Data valuation vs Dataset valuation

A tentative, but naive, approach to solve the dataset valuation problem could be to run an auxiliary data-valuation algorithm on all the data and to assign to each dataset the sum of the values of its datapoints. We highlight the cons of this idea on a very simple, yet insightful example. Consider two datapoints $x_{1}$ and $x_{2}$, three datasets $\mathrm{D}_{1}=\{x_{1}\}$, $\mathrm{D}_{2}=\{x_{2}\}$, $\mathrm{D}_{3}=\{x_{2},x_{2}\}$, and the following toy utility function $u(\mathrm{D})=\mathbbm{1}{\{x_{1},x_{2}\in\mathrm{D}\}}$. In data valuation, any point $x_{2}$ shall have the same value, as they are identical. In particular, a naive summation would value $\mathrm{D}_{3}$ twice the value of $\mathrm{D}_{2}$. In dataset valuation, and for this toy problem at hand, it is quite clear that both datasets should have the same value. Moreover, the Shapley values are $1/6$ for $\mathrm{D}_{2}$ and $\mathrm{D}_{3}$ versus $2/3$ for $\mathrm{D}_{1}$.

The message here is twofold. Data valuation and dataset valuation are two fundamentally different concepts and one cannot directly reduce the latter to the former. This is actually true, and this is the second message, because the utility function $u$ is highly non-linear (even for the regression task).

## 3 Discrete Uniform Shapley Value

This section introduces and studies our approximation scheme for the Shapley value. Section 3.1 shows an asymptotic property that gives the general intuition behind our approximation. The result holds for the three use-cases of Sections 2.1.1, 2.1.2 and 2.1.3. Section 3.2 presents a general approximation methodology for dataset valuation and shows its almost surely convergence as the number of players grows for our three uses-cases. Section 3.3 studies the rate of convergence, first for the homogeneous setting (Section 2.1.2), and then leverages this result to obtain a similar one for the non-parametric regression setting (Section 2.1.1). All proofs are postponed to the supplementary material.

### 3.1 Insights behind DU-Shapley

The Shapley value, by re-arranging the coalitions $\mathcal{S}\subseteq\mathcal{I}\setminus\{i\}$ by their cardinality in the sum in (8), can be equivalently expressed as

$$
\displaystyle\varphi_{i}(u)=\mathbb{E}_{K\sim\mathrm{U}(\{0,...,I-1\})}\mathbb%
{E}_{\mathcal{S}\sim\mathrm{U}\bigl{(}2^{\mathcal{I}\setminus\{i\}}_{K}\bigr{)%
}}\left[{u({\mathcal{S}}\cup\{i\})-u({\mathcal{S}})}\right]\,,
$$

where $2^{\mathcal{I}\setminus\{i\}}_{K}$ denotes the subsets of $\mathcal{I}\setminus\{i\}$ of cardinality $K$. In our three uses-cases, it follows that

$$
\displaystyle\varphi_{i}(u)
$$
 
$$
\displaystyle=\varphi_{i}(w)=\mathbb{E}_{K\sim\mathrm{U}(\{0,...,I-1\})}%
\mathbb{E}_{\mathcal{S}\sim\mathrm{U}\bigl{(}2^{\mathcal{I}\setminus\{i\}}_{K}%
\bigr{)}}\left[{w(q(\mathcal{S}\cup\{i\}))-w(q(\mathcal{S}))}\right]\,,
$$

where $w:\mathbb{R}_{+}\to\mathbb{R}$ is such that $u(\mathcal{S})=w(q(\mathcal{S}))$ for any $\mathcal{S}\subseteq\mathcal{I}$, and $q(\mathcal{S})$ is the scalar quantity of interest identified in Sections 2.1.1, 2.1.2 and 2.1.3 for each use-case:

$$
\displaystyle q(\mathcal{S}):=\biggl{\lfloor}\frac{\bigl{(}\sum_{i\in\mathcal{%
S}}\gamma_{i}n_{i}\bigr{)}^{2}}{\sum_{i\in\mathcal{S}}\gamma_{i}^{2}n_{i}}%
\biggr{\rfloor},\text{ where, for any }i\in\mathcal{I},\gamma_{i}=\left\{%
\begin{array}[]{cl}1&\text{for the second use-case},\\
\nicefrac{{\sigma_{i}}}{{\varepsilon_{i}}}&\text{for the third use-case},\end{%
array}\right.
$$

and for the first use-case, $q_{b}(\mathcal{S})$ is analogously defined at every bin, with $\gamma_{i}^{b}=1$ for all players and all bins. We remark that the definition of $q(\mathcal{S})$ in the first and second use-cases is not restricted to linear regression. Equation (11) explicitly reveals a key random variable, namely $q(\mathcal{S})$. Interestingly, Figure 1 suggests that $q(\mathcal{S})$ converges in distribution to a uniform random variable as the number of players increases (with i.i.d. datasets sizes). Theorem 1 proves this result formally for any $(\gamma_{i})_{i\in\mathcal{I}}$.

###### Theorem 1.

Let $\{n_{i},\gamma_{i}\}_{i\in[I]}$ be two sequences of positive numbers such that the following limits

$$
\displaystyle\lim_{I\to\infty}\frac{1}{I}\sum\nolimits_{i\in[I]}n_{i}\gamma_{i%
}=\mu_{A},\quad\lim_{I\to\infty}\frac{1}{I}\sum\nolimits_{i\in[I]}(n_{i}\gamma%
_{i}-\mu_{A})^{2}=\sigma^{2}_{A},
$$
$$
\displaystyle\lim_{I\to\infty}\frac{1}{I}\sum\nolimits_{i\in[I]}n_{i}\gamma_{i%
}^{2}=\mu_{B},\quad\lim_{I\to\infty}\frac{1}{I}\sum\nolimits_{i\in[I]}(n_{i}%
\gamma_{i}^{2}-\mu_{B})^{2}=\sigma^{2}_{B}\,,
$$

all exist, for some constants $\mu_{A},\mu_{B},\sigma_{A},\sigma_{B}>0$. Let $K\sim\mathrm{U}(\{0,\ldots,I\})$, $\mathcal{S}_{K}\sim\mathrm{U}([2^{\mathcal{I}}_{K}])$. Then, almost surely, $\frac{q(\mathcal{S}_{K})}{q(\mathcal{I})}\xrightarrow{I\to\infty}\mathrm{U}([0%
,1])$.

![Refer to caption](https://arxiv.org/html/2306.02071v3/extracted/5976742/figure/heterogeneous_case/Img7.png)

Figure 1: Distribution of q ⁢ ( 𝒮 ) / ℐ 𝑞 q(\\mathcal{S})/q(\\mathcal{I}) italic\_q ( caligraphic\_S ) / italic\_q ( caligraphic\_I ) when \\mathcal{S} caligraphic\_S is sampled as in ( 11 ) (i.e., first sample a size K 𝐾 italic\_K uniformly, then sample a coalition of size uniformly). (left) I = 10 𝐼 I=10 italic\_I = 10, (middle) 50 I=50 italic\_I = 50, (right) 500 I=500 italic\_I = 500. We considered 4 superscript 10^{4} 10 start\_POSTSUPERSCRIPT 4 end\_POSTSUPERSCRIPT samples for each random variable, and the third use-case with n i ∼ U \[ 100 \] similar-to subscript 𝑛 𝑖 delimited-\[\] n\_{i}\\sim\\mathrm{U}(\[100\]) italic\_n start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ∼ roman\_U ( \[ 100 \] ) and σ ε 𝜎 𝜀 \\sigma\_{i}/\\varepsilon\_{i}\\sim\\mathrm{U}(\[10\]) italic\_σ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT / italic\_ε start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ∼ roman\_U ( \[ 10 \] ) for each ∈ i\\in\\mathcal{I} italic\_i ∈ caligraphic\_I.

### 3.2 Discrete Uniform Shapley value

The Shapley value re-arrangement in (10) exposes the main tool behind our approximation: it is enough to approximate the distribution of the random variable $\mathrm{D}_{\mathcal{S}}$ that takes values on the subsets of $\mathrm{D}_{-i}:=\cup_{j\in\mathcal{I}\setminus\{i\}}\mathrm{D}_{j}$ (recall that $u(\mathcal{S})=u(\mathrm{D}_{\mathcal{S}})$). Theorem 1, taking the example of the second use-case for intuition, indicates that these datasets have uniformly distributed numbers of points in the limit. Generalizing this intuition, we propose to approximate $\mathrm{D}_{\mathcal{S}}$ by taking $I$ samples of increasing size from the pool $\mathrm{D}_{-i}$ by sampling data points uniformly. This leads to the following definition of DU-Shapley for our generic model:

###### Definition.

\[DU-Shapley\] For any $i\in\mathcal{I}$, the discrete uniform Shapley value (DU-Shapley) of the $i$ -th player, denoted by $\psi_{i}$, is given by

$$
\displaystyle\psi_{i}(u):=\frac{1}{I}\sum\nolimits_{k=0}^{I-1}u(\mathrm{D}^{(k%
)}\cup\mathrm{D}_{i})-u(\mathrm{D}^{(k)}),
$$

where $\mathrm{D}^{(k)}$ is a set of data points uniformly sampled without replacement from $\mathrm{D}_{-i}$ of size $k\mu_{-i}$, with $\mu_{-i}=\frac{1}{(I-1)}|\mathrm{D}_{-i}|$.

Compared to the Shapley value defined in (8), which involves $2^{I}$ terms to compute, note that DU-Shapley only involves $I$ terms and hence it presents an exponential reduction of the number of utility function evaluations. Of course, these computational savings come at the cost of some bias. The latter is precisely quantified in Section 3.3 for our first two use-cases.

By definition, DU-Shapley is a random variable which depends on the sampled data points. However, whenever $u(\mathcal{S})=w(q(\mathcal{S}))$, with $q(\mathcal{S})$ some scalar quantify of interest, as in our use-cases, we can get rid of the stochastic nature of DU-Shapley by considering $I$ real values from well chosen intervals. In particular, in our uses-cases, DU-Shapley boils down to:

$$
\displaystyle\psi_{i}(w)=\frac{1}{I}\sum\nolimits_{k=0}^{I-1}w(\bar{q}_{i}^{k}%
)-w(\bar{q}_{-i}^{k}),
$$

where

$$
\bar{q}_{i}^{k}:=\biggl{\lfloor}\frac{(\gamma_{i}n_{i}+\frac{k}{I-1}\sum_{j\in%
\mathcal{I}\setminus\{i\}}\gamma_{j}n_{j})^{2}}{\gamma_{i}^{2}n_{i}+\frac{k}{I%
-1}\sum_{j\in\mathcal{I}\setminus\{i\}}\gamma_{j}^{2}n_{j}}\biggr{\rfloor}%
\text{ and }\bar{q}_{-i}^{k}:=\biggl{\lfloor}\frac{k}{I-1}\cdot\frac{(\sum_{j%
\in\mathcal{I}\setminus\{i\}}\gamma_{j}n_{j})^{2}}{\sum_{j\in\mathcal{I}%
\setminus\{i\}}\gamma_{j}^{2}n_{j}}\biggr{\rfloor}.
$$

We remark the notation abuse as we should write $\psi(w\circ q)$. For simplicity, we omit the composition and only write $\psi(w)$. Equation 18 coincides exactly with Section 3.2 in the first two use-cases, i.e., when $\gamma_{j}=\gamma$ for all $j\in\mathcal{I}$. Indeed, as the random datasets $\mathrm{D}^{(k)}$ have a fixed size and the value function only looks at the number of data points within the coalition, we obtain,

$$
\displaystyle\psi_{i}(u)
$$
 
$$
\displaystyle=\frac{1}{I}\sum_{k=0}^{I-1}u(\mathrm{D}^{(k)}\cup\mathrm{D}_{i})%
-u(\mathrm{D}^{(k)})=\frac{1}{I}\sum_{k=0}^{I-1}w(|\mathrm{D}^{(k)}\cup\mathrm%
{D}_{i}|)-w(|\mathrm{D}^{(k)}|)
$$
 
$$
\displaystyle=\frac{1}{I}\sum_{k=0}^{I-1}w(k\mu_{-i}+n_{i})-w(k\mu_{-i})=\psi_%
{i}(w).
$$

For the third use-case, Equation 18 is an approximation that comes from assuming that, for any $j\in\mathcal{I}\setminus\{i\}$, $|\mathrm{D}_{j}\cap\mathrm{D}^{(k)}|=k\cdot\frac{n_{j}}{I-1}$, which holds with high probability for large values of $I$, since

$$
\displaystyle q(\mathrm{D}^{(k)}\cup\mathrm{D}_{i})=\biggl{\lfloor}\frac{\bigl%
{(}\gamma_{i}n_{i}+\sum_{j\in\mathcal{I}\setminus\{i\}}\gamma_{j}\cdot|\mathrm%
{D}_{j}\cap\mathrm{D}^{(k)}|\bigr{)}^{2}}{\gamma_{i}^{2}n_{i}+\sum_{j\in%
\mathcal{I}\setminus\{i\}}\gamma_{j}^{2}\cdot|\mathrm{D}_{j}\cap\mathrm{D}^{(k%
)}|}\biggr{\rfloor}.
$$

Theorem 1 implies the following result.

###### Corollary.

Let $\varphi_{i}$ and $\psi_{i}$ be, respectively, the Shapley value (8) and the DU-Shapley (18) of player $i$. Then, in our three uses-cases, it holds, $\lim_{I\to\infty}|\varphi_{i}-\psi_{i}|=0$ almost surely.

While our theoretical results are based on Equation (18) for the cases where $u(\mathcal{S})=w(q(\mathcal{S}))$, we will see through numerical experiments that Section 3.2 gives good results in more general cases.

### 3.3 Non-Asymptotic Theoretical Guarantees

Section 3.2 states asymptotic guarantees for DU-Shapley. In this section, we show non-asymptotic results that give the convergence rate for the first two uses-cases.<sup>1</sup> Recall that in non-parametric estimation, the utility writes as $u(\mathcal{S})=\sum\nolimits_{b\in[B]}u_{b}(\mathcal{S})\mathbb{P}(x\in b)$, and therefore, by the linearity axiom of the Shapley value, for any $i\in\mathcal{I},\varphi_{i}(u)=\sum\nolimits_{b\in[B]}\varphi_{i}(u_{b})%
\mathbb{P}(x\in b)$. As a consequence, in order to estimate $\varphi_{i}(u)$, it is enough to compute each $\varphi_{i}(u_{b})$. In particular, the Shapley value approximation error over the whole feature space becomes a simple aggregation of the Shapley value approximation errors over the bins. We focus firstly on bounding the bias of our method in the homogeneous use-case to then extend it to the non-parametric regression case.

As in the homogeneous use-case the utility function writes as $u(\mathcal{S})=w(\sum_{i\in\mathcal{I}}n_{i})$, we consider the following regularity assumptions on $w$.

###### H 1.

The function $w:\mathbb{R}_{+}\rightarrow\mathbb{R}$ is increasing, twice continuously differentiable, and such that $\lim_{n\to\infty}n^{2}|w^{(2)}(n)|\operatorname*{<}\infty$ (where $w^{(2)}$ represents the second derivative).

Monotonicity is a natural assumption in our framework as, the more data, the more precise the ML prediction is expected to be. The condition over the limit aims at controlling the growth behavior of the utility function and it is automatically satisfied whenever $w$ is bounded and $w^{(2)}$ is monotone, by the mean value theorem. Theorem 2 bounds the bias of DU-Shapley for the homogeneous use-case.

###### Theorem 2.

Under Assumption H1, there exists a constant $\kappa>0$, such that, for any $i\in\mathcal{I}$, it holds,

$$
\displaystyle\bigl{|}\varphi_{i}-\psi_{i}\bigr{|}\leq\frac{\kappa}{(I-1)\mu_{-%
i}^{2}}\left(\sigma_{-i}^{2}(1+\ln(I-1))+\zeta_{-i}\right),
$$

where $\varphi_{i}$ and $\psi_{i}$ are respectively the Shapley value and the DU-Shapley of player $i$, $\mu_{-i}=\frac{1}{(I-1)}|\mathrm{D}_{-i}|$ is the average dataset size of all players but $i$, $\sigma^{2}_{-i}=\frac{1}{I-1}\sum_{j\in\mathcal{I}\setminus\{i\}}(n_{j}-\mu_{-%
i})^{2}$ their empirical variance, and $\zeta_{-i}$ measures the variability of the dataset sizes across players. Formally, it is defined as $\zeta_{-i}:={R_{-i}^{2}\tau_{-i}^{2}}/{4{n}^{\mathrm{max}}_{-i}}$ where $R_{-i}:=\max_{j\in\mathcal{I}\setminus\{i\}}|n_{j}-\mu_{-i}|$, ${n}^{\max}_{-i}:=\max_{j\in\mathcal{I}\setminus\{i\}}n_{j}$, and $\tau_{-i}:={{n}^{\max}_{-i}}/{\min_{j\in\mathcal{I}\setminus\{i\}}n_{j}}$.

The full proof of Theorem 2 is included in Section C.3 and it relies on controlling the absolute value of $\mathbb{E}[w(\mu_{-i}K)-w(\sum_{j\in\mathcal{S}}n_{j})]$, where $K\sim\mathrm{U}(\{0,...,I-1\})$ and $\mathcal{S}$ is the random variable in (10). Using a second order Taylor expansion, the problem is reduced to controlling the term related to the second derivative of $w$ by using the regularity assumptions in H1.

As advertised before, Theorem 2 can be directly generalized to the non-parametric use-case, since,

$$
\displaystyle u(\mathcal{S})=\sum\nolimits_{b\in[B]}w_{b}\biggl{(}\sum%
\nolimits_{i\in\mathcal{S}}n_{i,b}\biggr{)}\mathbb{P}(x\in b),\ \text{ for }n_%
{i,b}=|\{(x,y)\in\mathrm{D}_{j},x\in b\}|.
$$

###### Corollary.

Under Assumption H1 for all functions $w_{b}$, there exist constants $\kappa_{b}>0$, such that, for any $i\in\mathcal{I}$, it holds that

$$
\displaystyle\bigl{|}\varphi_{i}-\psi_{i}\bigr{|}\leq\sum\nolimits_{b\in[B]}%
\frac{\kappa_{b}\mathbb{P}(x\in b)}{(I-1)\mu_{-i,b}^{2}}\left(\sigma_{-i,b}^{2%
}(1+\ln(I-1))+2\zeta_{-i,b}\right),
$$

where $\varphi_{i}$ and $\psi_{i}$ are respectively the Shapley value and the DU-Shapley of player $i$, and all terms are equivalently defined to Theorem 2 at each bin $b\in[B]$.

The upper bound in (24) depends on natural quantities related to the dataset valuation problem described in Section 2.1 at each bin, such as the first two moments $\mu_{-i,b}$ and $\sigma_{-i,b}$ of the datasets’ size distribution. More precisely, the error increases when there are some outlier players with a very small or large dataset size. This behavior is expected since, in this particular setting, the random variable inside of the Shapley value differs from a uniform random variable. As showcased in Theorem 1, the error vanishes when the number of players $I$ tends towards infinity.

## 4 Numerical Experiments

We illustrate the benefits of DU-Shapley by measuring numerically three properties: (1) how well DU-Shapley approximates the Shapley value in real data, (2) how many (theoretical) iterations need other methods to achieve the same accuracy level than DU-Shapley, and (3) how well DU-Shapley performs in classical dataset valuation tasks with real data. Section A.1 complements the results by a complexity comparison between our method and SVARM [^22] and Section A.2 by experiments on synthetic data. The experiments strongly suggest that DU-Shapley performs well in all tasks.

### 4.1 Approximating the Shapley Value in Real-World Data

We consider the real-world datasets in [^32], whose details are provided in Table 3 in the appendix. To tackle these problems we consider logistic regression models and gradient-boosted decision trees (GBDT). For classification tasks, the utility function has been taken as the expected accuracy of the trained logistic regression model over a hold-out testing set while for regression tasks, the utility function corresponds to the averaged MSE over a hold-out testing set. In both cases we took a hold-out testing set with 10% of the size of the training dataset. For each dataset, we considered two worst-case scenarios for our method, namely $I=10$ players and $I=20$ players.

Starting from the datasets in Table 3, we heterogeneously allocate datasets to the players. We compare ourselves with two approaches, referred to as MC-Shapley, for the standard MC approximation defined in (9), and MC-anti-Shapley that considers, in addition, antithetic sampling [^32]. We compute the averaged MSE across all players between the true Shapley value and each estimator.

Since computing the marginal contributions in this experiment requires re-training, which is clearly not feasible for a large number of epochs, we chose to restrict ourselves to 20 steps of stochastic gradient descent for logistic regression and 20 boosting iterations for GBDTs. For MC-based approaches, we considered $I$ samples to compare those approximations with the proposed methodology on a fair basis, i.e., associated to the same computational budget.

Table 1 depicts the results. We clearly see that, even in the worst-case scenario where the number of players is small and far from the theoretical assumptions from Section 3.3, DU-Shapley competes favorably with the MC-based methods.

Table 1: Worst-case comparison between DU-Shapley and competitors, for real-world datasets considered in Table 3. We report the averaged MSE across all players w.r.t. the exact Shapley value.

<table><thead><tr><th>Dataset</th><th colspan="2">adult</th><th colspan="2">breast-cancer</th><th colspan="2">bank</th><th colspan="2">cal-housing</th></tr><tr><th>Players</th><th>10</th><th>20</th><th>10</th><th>20</th><th>10</th><th>20</th><th>10</th><th>20</th></tr></thead><tbody><tr><th>DU-Shapley</th><td><math><semantics><msup><mn>2.10</mn> <mrow><mo>−</mo> <mn>𝟑</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>2.10</cn> <apply><cn>3</cn></apply></apply> <annotation>\mathbf{2.10^{-3}}</annotation> <annotation>bold_2.10 start_POSTSUPERSCRIPT - bold_3 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>6.10</mn> <mrow><mo>−</mo> <mn>𝟒</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>6.10</cn> <apply><cn>4</cn></apply></apply> <annotation>\mathbf{6.10^{-4}}</annotation> <annotation>bold_6.10 start_POSTSUPERSCRIPT - bold_4 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>3.10</mn> <mrow><mo>−</mo> <mn>𝟑</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>3.10</cn> <apply><cn>3</cn></apply></apply> <annotation>\mathbf{3.10^{-3}}</annotation> <annotation>bold_3.10 start_POSTSUPERSCRIPT - bold_3 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>1.10</mn> <mrow><mo>−</mo> <mn>𝟒</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>1.10</cn> <apply><cn>4</cn></apply></apply> <annotation>\mathbf{1.10^{-4}}</annotation> <annotation>bold_1.10 start_POSTSUPERSCRIPT - bold_4 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>5.10</mn> <mrow><mo>−</mo> <mn>𝟐</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>5.10</cn> <apply><cn>2</cn></apply></apply> <annotation>\mathbf{5.10^{-2}}</annotation> <annotation>bold_5.10 start_POSTSUPERSCRIPT - bold_2 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>4.10</mn> <mrow><mo>−</mo> <mn>𝟑</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>4.10</cn> <apply><cn>3</cn></apply></apply> <annotation>\mathbf{4.10^{-3}}</annotation> <annotation>bold_4.10 start_POSTSUPERSCRIPT - bold_3 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>1.10</mn> <mrow><mo>−</mo> <mn>𝟐</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>1.10</cn> <apply><cn>2</cn></apply></apply> <annotation>\mathbf{1.10^{-2}}</annotation> <annotation>bold_1.10 start_POSTSUPERSCRIPT - bold_2 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>3.10</mn> <mrow><mo>−</mo> <mn>𝟑</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>3.10</cn> <apply><cn>3</cn></apply></apply> <annotation>\mathbf{3.10^{-3}}</annotation> <annotation>bold_3.10 start_POSTSUPERSCRIPT - bold_3 end_POSTSUPERSCRIPT</annotation></semantics></math></td></tr><tr><th>MC-Shapley</th><td><math><semantics><msup><mn>1.10</mn> <mrow><mo>−</mo> <mn>2</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>1.10</cn> <apply><cn>2</cn></apply></apply> <annotation>1.10^{-2}</annotation> <annotation>1.10 start_POSTSUPERSCRIPT - 2 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>4.10</mn> <mrow><mo>−</mo> <mn>3</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>4.10</cn> <apply><cn>3</cn></apply></apply> <annotation>4.10^{-3}</annotation> <annotation>4.10 start_POSTSUPERSCRIPT - 3 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>3.10</mn> <mrow><mo>−</mo> <mn>2</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>3.10</cn> <apply><cn>2</cn></apply></apply> <annotation>3.10^{-2}</annotation> <annotation>3.10 start_POSTSUPERSCRIPT - 2 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>1.10</mn> <mrow><mo>−</mo> <mn>3</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>1.10</cn> <apply><cn>3</cn></apply></apply> <annotation>1.10^{-3}</annotation> <annotation>1.10 start_POSTSUPERSCRIPT - 3 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>9.10</mn> <mrow><mo>−</mo> <mn>2</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>9.10</cn> <apply><cn>2</cn></apply></apply> <annotation>9.10^{-2}</annotation> <annotation>9.10 start_POSTSUPERSCRIPT - 2 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>6.10</mn> <mrow><mo>−</mo> <mn>2</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>6.10</cn> <apply><cn>2</cn></apply></apply> <annotation>6.10^{-2}</annotation> <annotation>6.10 start_POSTSUPERSCRIPT - 2 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>5.10</mn> <mrow><mo>−</mo> <mn>2</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>5.10</cn> <apply><cn>2</cn></apply></apply> <annotation>5.10^{-2}</annotation> <annotation>5.10 start_POSTSUPERSCRIPT - 2 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>2.10</mn> <mrow><mo>−</mo> <mn>2</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>2.10</cn> <apply><cn>2</cn></apply></apply> <annotation>2.10^{-2}</annotation> <annotation>2.10 start_POSTSUPERSCRIPT - 2 end_POSTSUPERSCRIPT</annotation></semantics></math></td></tr><tr><th>MC-anti-Shapley</th><td><math><semantics><msup><mn>8.10</mn> <mrow><mo>−</mo> <mn>3</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>8.10</cn> <apply><cn>3</cn></apply></apply> <annotation>8.10^{-3}</annotation> <annotation>8.10 start_POSTSUPERSCRIPT - 3 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>2.10</mn> <mrow><mo>−</mo> <mn>3</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>2.10</cn> <apply><cn>3</cn></apply></apply> <annotation>2.10^{-3}</annotation> <annotation>2.10 start_POSTSUPERSCRIPT - 3 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>1.10</mn> <mrow><mo>−</mo> <mn>2</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>1.10</cn> <apply><cn>2</cn></apply></apply> <annotation>1.10^{-2}</annotation> <annotation>1.10 start_POSTSUPERSCRIPT - 2 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>8.10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>8.10</cn> <apply><cn>4</cn></apply></apply> <annotation>8.10^{-4}</annotation> <annotation>8.10 start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>8.10</mn> <mrow><mo>−</mo> <mn>2</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>8.10</cn> <apply><cn>2</cn></apply></apply> <annotation>8.10^{-2}</annotation> <annotation>8.10 start_POSTSUPERSCRIPT - 2 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>4.10</mn> <mrow><mo>−</mo> <mn>2</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>4.10</cn> <apply><cn>2</cn></apply></apply> <annotation>4.10^{-2}</annotation> <annotation>4.10 start_POSTSUPERSCRIPT - 2 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>3.10</mn> <mrow><mo>−</mo> <mn>2</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>3.10</cn> <apply><cn>2</cn></apply></apply> <annotation>3.10^{-2}</annotation> <annotation>3.10 start_POSTSUPERSCRIPT - 2 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>1.10</mn> <mrow><mo>−</mo> <mn>2</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>1.10</cn> <apply><cn>2</cn></apply></apply> <annotation>1.10^{-2}</annotation> <annotation>1.10 start_POSTSUPERSCRIPT - 2 end_POSTSUPERSCRIPT</annotation></semantics></math></td></tr></tbody></table>

<table><thead><tr><th>Dataset</th><th colspan="2">make-regression</th><th colspan="2">year</th></tr><tr><th>Players</th><th>10</th><th>20</th><th>10</th><th>20</th></tr></thead><tbody><tr><th>DU-Shapley</th><td><math><semantics><msup><mn>9.10</mn> <mrow><mo>−</mo> <mn>𝟐</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>9.10</cn> <apply><cn>2</cn></apply></apply> <annotation>\mathbf{9.10^{-2}}</annotation> <annotation>bold_9.10 start_POSTSUPERSCRIPT - bold_2 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>2.10</mn> <mrow><mo>−</mo> <mn>𝟐</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>2.10</cn> <apply><cn>2</cn></apply></apply> <annotation>\mathbf{2.10^{-2}}</annotation> <annotation>bold_2.10 start_POSTSUPERSCRIPT - bold_2 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>1.10</mn> <mrow><mo>−</mo> <mn>𝟑</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>1.10</cn> <apply><cn>3</cn></apply></apply> <annotation>\mathbf{1.10^{-3}}</annotation> <annotation>bold_1.10 start_POSTSUPERSCRIPT - bold_3 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>7.10</mn> <mrow><mo>−</mo> <mn>𝟒</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>7.10</cn> <apply><cn>4</cn></apply></apply> <annotation>\mathbf{7.10^{-4}}</annotation> <annotation>bold_7.10 start_POSTSUPERSCRIPT - bold_4 end_POSTSUPERSCRIPT</annotation></semantics></math></td></tr><tr><th>MC-Shapley</th><td><math><semantics><msup><mn>4.10</mn> <mrow><mo>−</mo> <mn>1</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>4.10</cn> <apply><cn>1</cn></apply></apply> <annotation>4.10^{-1}</annotation> <annotation>4.10 start_POSTSUPERSCRIPT - 1 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>3.10</mn> <mrow><mo>−</mo> <mn>1</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>3.10</cn> <apply><cn>1</cn></apply></apply> <annotation>3.10^{-1}</annotation> <annotation>3.10 start_POSTSUPERSCRIPT - 1 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>5.10</mn> <mrow><mo>−</mo> <mn>3</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>5.10</cn> <apply><cn>3</cn></apply></apply> <annotation>5.10^{-3}</annotation> <annotation>5.10 start_POSTSUPERSCRIPT - 3 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>1.10</mn> <mrow><mo>−</mo> <mn>3</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>1.10</cn> <apply><cn>3</cn></apply></apply> <annotation>1.10^{-3}</annotation> <annotation>1.10 start_POSTSUPERSCRIPT - 3 end_POSTSUPERSCRIPT</annotation></semantics></math></td></tr><tr><th>MC-anti-Shapley</th><td><math><semantics><msup><mn>4.10</mn> <mrow><mo>−</mo> <mn>1</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>4.10</cn> <apply><cn>1</cn></apply></apply> <annotation>4.10^{-1}</annotation> <annotation>4.10 start_POSTSUPERSCRIPT - 1 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>2.10</mn> <mrow><mo>−</mo> <mn>1</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>2.10</cn> <apply><cn>1</cn></apply></apply> <annotation>2.10^{-1}</annotation> <annotation>2.10 start_POSTSUPERSCRIPT - 1 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>5.10</mn> <mrow><mo>−</mo> <mn>3</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>5.10</cn> <apply><cn>3</cn></apply></apply> <annotation>5.10^{-3}</annotation> <annotation>5.10 start_POSTSUPERSCRIPT - 3 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td><math><semantics><msup><mn>1.10</mn> <mrow><mo>−</mo> <mn>3</mn></mrow></msup> <apply><csymbol>superscript</csymbol> <cn>1.10</cn> <apply><cn>3</cn></apply></apply> <annotation>1.10^{-3}</annotation> <annotation>1.10 start_POSTSUPERSCRIPT - 3 end_POSTSUPERSCRIPT</annotation></semantics></math></td></tr></tbody></table>

### 4.2 Complexity of Computing the Shapley Values of all Players

We have looked at the number of iterations that DataShapley and the Improved Group Testing-Based method [^46] (IGTB) require to achieve DU-Shapley’s accumulated bias, formally given by

$$
\displaystyle\mathrm{DUbias}(I):=\frac{\kappa}{I-1}\biggl{(}\sum\nolimits_{i%
\in\mathcal{I}}\frac{(9\sigma_{-i}^{2}(1+\log(I-1))+\zeta_{-i})^{2}}{(\mu_{-i}%
)^{4}}\biggr{)}^{1/2}.
$$

To do so, we have replaced $\varepsilon=\mathrm{DUbias}(I)$, respectively, in the formula in Section 4.1 in [^16] and Equation 5 in [^46], with a value function motivated from our third use-case under the homogeneity assumption $\sigma_{i}/\varepsilon_{i}=\sigma/\varepsilon$ for all $i\in\mathcal{I}$. The results are illustrated in Figure 2. Remark DU-Shapley requires $I^{2}$ iterations to compute all Shapley values. We observe that in all tested instances, both methods require a higher number of iterations to achieve the same error than DU-Shapley.

![Refer to caption](https://arxiv.org/html/2306.02071v3/extracted/5976742/New_images/Complexity_delta_0.01_and_0.1.png)

Figure 2: Iterations required by DataShapley and the Improved Group Testing-Based method to achieve DU-Shapley’s accumulated bias with function w ⁢ ( n S ) = 1 − 10 k ℐ + 𝑤 subscript 𝑛 𝑆 superscript 𝑘 w(n\_{S})=1-\\frac{10^{k(\\mathcal{I})}}{10^{k(\\mathcal{I})}+n\_{S}} italic\_w ( italic\_n start\_POSTSUBSCRIPT italic\_S end\_POSTSUBSCRIPT ) = 1 - divide start\_ARG 10 start\_POSTSUPERSCRIPT italic\_k ( caligraphic\_I ) end\_POSTSUPERSCRIPT end\_ARG start\_ARG 10 start\_POSTSUPERSCRIPT italic\_k ( caligraphic\_I ) end\_POSTSUPERSCRIPT + italic\_n start\_POSTSUBSCRIPT italic\_S end\_POSTSUBSCRIPT end\_ARG, where n\_{S} italic\_n start\_POSTSUBSCRIPT italic\_S end\_POSTSUBSCRIPT is the number of data points of the coalition ⊆ S\\subseteq\\mathcal{I} italic\_S ⊆ caligraphic\_I, and:= ⌊ log ⁡ ∑ i ∈ ⌋ assign 𝑖 k(\\mathcal{I}):=\\lfloor\\log(\\sum\_{i\\in\\mathcal{I}}n\_{i})\\rfloor-1 italic\_k ( caligraphic\_I ):= ⌊ roman\_log ( ∑ start\_POSTSUBSCRIPT italic\_i ∈ caligraphic\_I end\_POSTSUBSCRIPT italic\_n start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) ⌋ - 1 is a normalization factor. (top) δ 0.01 𝛿 \\delta=0.01 italic\_δ = 0.01, (bottom) 0.1 \\delta=0.1 italic\_δ = 0.1, (left) max n\_{\\mathrm{max}}=10 italic\_n start\_POSTSUBSCRIPT roman\_max end\_POSTSUBSCRIPT = 10, (middle) 50 n\_{\\mathrm{max}}=50 italic\_n start\_POSTSUBSCRIPT roman\_max end\_POSTSUBSCRIPT = 50, (right) 100 n\_{\\mathrm{max}}=100 italic\_n start\_POSTSUBSCRIPT roman\_max end\_POSTSUBSCRIPT = 100.

### 4.3 Applying DU-Shapley to dataset valuation problems

We considered non-tabular datasets used in [^17], namely bbc-embedding, IMDB-embedding, both text datasets, and CIFAR10-embedding, an image dataset. Feature embedding have been generated using pretrained DistilBERT and ResNet50 models, respectively. In addition we have adapted three baselines from data valuation to our setting: Leave-One-Out (LOO), DataShapley, and KNN-Shapley. Section B.2 gives the implementations details. For these datasets associated to classification problems, we used a multi-layer perceptron classifier as prediction model.

We have considered three dataset valuation problems, none of them needing the real Shapley values, which allows us to increase the number of players w.r.t. the experiments in Section 4.1. We investigated noisy label detection (NLD), dataset removal (DR), and dataset addition (DA) [^17]. For NLD, we used as a metric the F1-score (the larger the better). For DR, we used the testing accuracy (the lesser the better). For DA, we used the testing accuracy (the lesser the better).

We considered splitting the dataset across $I=100$ players. The results are summarized in Table 2. We observe that DU-Shapley has competitive results compared to classical baselines despite of the fact that none of the considered cases verifies the structural assumptions from Section 3.3. In addition, we can see that DU-Shapley tends to have similar and even better results as Data Shapley (which is a MC based method). This is in line with our theory as, for larger number of players, DU-Shapley tends to better estimate the true Shapley value.

Table 2: Comparison between DU-Shapley and competitors for real-world datasets considered in [^17] in Noisy label detection, Dataset Removal and Dataset Addition.

<table><thead><tr><th>Dataset</th><th colspan="6">CIFAR 10</th><th colspan="6">BBC</th></tr><tr><th rowspan="2">Problem</th><th colspan="2">NLD</th><th colspan="2">DR</th><th colspan="2">DA</th><th colspan="2">NLD</th><th colspan="2">DR</th><th colspan="2">DA</th></tr><tr><th>5%</th><th>15%</th><th>5%</th><th>15%</th><th>5%</th><th>15%</th><th>5%</th><th>15%</th><th>5%</th><th>15%</th><th>5%</th><th>15%</th></tr></thead><tbody><tr><th>Random</th><td>0.11</td><td>0.19</td><td>0.61</td><td>0.60</td><td>0.25</td><td>0.41</td><td>0.11</td><td>0.19</td><td>0.90</td><td>0.88</td><td>0.68</td><td>0.81</td></tr><tr><th>LOO</th><td>0.13</td><td>0.18</td><td>0.62</td><td>0.60</td><td>0.15</td><td>0.32</td><td>0.11</td><td>0.17</td><td>0.90</td><td>0.88</td><td>0.61</td><td>0.77</td></tr><tr><th>DataShapley</th><td>0.13</td><td>0.25</td><td>0.61</td><td>0.59</td><td>0.12</td><td>0.18</td><td>0.12</td><td>0.20</td><td>0.89</td><td>0.87</td><td>0.08</td><td>0.12</td></tr><tr><th>KNN-Shapley</th><td>0.14</td><td>0.28</td><td>0.60</td><td>0.57</td><td>0.12</td><td>0.15</td><td>0.19</td><td>0.29</td><td>0.88</td><td>0.86</td><td>0.13</td><td>0.12</td></tr><tr><th>DU-Shapley</th><td>0.14</td><td>0.30</td><td>0.61</td><td>0.55</td><td>0.11</td><td>0.14</td><td>0.18</td><td>0.34</td><td>0.89</td><td>0.85</td><td>0.07</td><td>0.11</td></tr></tbody></table>

<table><thead><tr><th>Dataset</th><th colspan="6">IMBD</th></tr><tr><th rowspan="2">Problem</th><th colspan="2">NLD</th><th colspan="2">DR</th><th colspan="2">DA</th></tr><tr><th>5%</th><th>15%</th><th>5%</th><th>15%</th><th>5%</th><th>15%</th></tr></thead><tbody><tr><th>Random</th><td>0.10</td><td>0.16</td><td>0.77</td><td>0.75</td><td>0.62</td><td>0.68</td></tr><tr><th>LOO</th><td>0.11</td><td>0.18</td><td>0.77</td><td>0.74</td><td>0.53</td><td>0.59</td></tr><tr><th>DataShapley</th><td>0.17</td><td>0.28</td><td>0.75</td><td>0.69</td><td>0.36</td><td>0.33</td></tr><tr><th>KNN-Shapley</th><td>0.18</td><td>0.29</td><td>0.76</td><td>0.68</td><td>0.41</td><td>0.37</td></tr><tr><th>DU-Shapley</th><td>0.18</td><td>0.32</td><td>0.76</td><td>0.66</td><td>0.33</td><td>0.34</td></tr></tbody></table>

## 5 Conclusion

We model the dataset valuation problem as a cooperative game and design a Shapley value approximation, named DU-Shapley, that exploits the underlying structure of the utility function and exponentially reduces the number of functions valuations required for the computation. In three different uses-cases, DU-Shapley is proved to almost surely converge to the real Shapley value as the number of players grows. Moreover, we find the rate of convergence, which depends only on natural parameters of dataset valuation. Numerical experiments showcase that DU-Shapley performs well in approximating the Shapley value and performing dataset valuation tasks, even when the assumptions needed for the theoretical guarantees do not hold, and it has a good complexity when computing the Shapley values of all players.

Limitations of our method. Our non-asymptotic bound for the non-parametric regression setting in Section 3.3 indicates that DU-Shapley works better when agents’ datasets are regular in the sense that they have similar sizes. Hence, a limitation of our approximation is that it may work poorly in settings where some players have large datasets compared to others, as the distribution of the random variable within the Shapley value drives apart from being uniform. Moreover, our convergence result in Theorem 1 (for all use-cases) assume the existence of limits, which roughly requires that heterogeneity between players—in terms of both dataset size and variance—can be bounded. This also indicates that convergence may be not be guaranteed if the heterogeneity is arbitrarily high.

## Acknowledgments

This research was supported in part by the French National Research Agency (ANR) in the framework of the PEPR IA FOUNDRY project (ANR-23-PEIA-0003) and through the grant DOOM ANR-23-CE23-0002. It was also funded by the European Union (ERC, Ocean, 101071601). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## References

## Appendix A Complementary Numerical results

All experiments were executed on a laptop running macOS 13.3.1 and equipped with Apple M1 chip with 16GB of RAM. The minimum amount of compute was roughly 5 minutes while the maximum one roughly 10 hours.

### A.1 DU-Shapley vs SVARM

We have looked at the probability at which SVARM (Theorem 4 [^22]) can ensure, after $I^{2}$ iterations (without considering the warm up as part of the budget), an error equal to DU-Shapley’s accumulated bias. We have considered the same value function than in Section 4.2 with $n_{max}\in\{2\cdot 10^{3},3\cdot 10^{3},5\cdot 10^{3},10^{4}\}$ and 100 simulations of sets of players at each time. Figure 3 shows the results. We observe how SVARM cannot ensure, with high enough probability, an approximation error equal to the one of DU-Shapley.

![Refer to caption](https://arxiv.org/html/2306.02071v3/extracted/5976742/New_images/Proba_SVARM.png)

Figure 3: Probability that SVARM guarantees an error equal to DU-Shapley’s bias

### A.2 Approximating the Shapley value in Synthetic Data

We consider a toy dataset valuation problem associated to our heterogeneous linear regression with local differential privacy use-case (Section 2.1.3) and we measure the value of a coalition $\mathcal{S}$ with the utility function in close-form from Section 2.1.3. We consider $d=10$.

In order to benchmark the performances of DU-Shapley, we consider four competitive approaches, relying on Monte Carlo (MC) approximation strategies [^32]. The first one, referred to as MC-Shapley is the standard MC approximation defined in (9). The second one, coined MC-anti-Shapley is a variance-reduced version of MC-Shapley that considers antithetic sampling. The third one coined Owen-Shapley stands for the multilinear extension of [^34] which represents the Shapley value as two nested expectations (further explained in Section B.3). Finally, the fourth approach, coined Orthogonal-Shapley, relies on efficient permutation sampling techniques on the hypersphere to draw permutations in (7) in a dependent way. To assess the performance of the aforementioned Shapley value estimators, we used the mean square error (MSE) averaged over all players. DU-Shapley is computed exactly by using (18) while, for each MC-based estimator, we performed 25 Shapley value estimations to compute the MSE, and did it 10 times to obtain confidence intervals for the MSE.

Figure 4 compares DU-Shapley (the horizontal line which does not depend on the sampling budget as we compute it exactly) and the MC-based methods, which are computed at several different budgets. The x-axis corresponds to the sampling budget allowed to the MC-bases methods w.r.t. DU-Shapley, i.e., $10^{-1}$ means a budget equal to $10$ % the one of DU-Shapley, $10^{0}$ means same budget (indicated by the black vertical line), and $10^{1}$ means 10 times the DU-Shapley budget. Remark that, even when the MC-methods use 10 times the budget of DU-Shapley, our method keeps approximating better the Shapley value.

![Refer to caption](https://arxiv.org/html/2306.02071v3/extracted/5976742/New_images/MSE_syntetic_data_Gamma_10_10players.png)

Figure 4: Worst-case comparison between DU-Shapley and MC-based approximations with different budgets on synthetic datasets. From left to right, I = 10 𝐼 I=10 italic\_I = 10 and 20 I=20 italic\_I = 20, n i ∼ U ⁢ ( \[ 3 \] ) ∀ ∈ ℐ formulae-sequence similar-to subscript 𝑛 𝑖 delimited-\[\] superscript for-all n\_{i}\\sim\\mathrm{U}(\[10^{3}\]),\\forall i\\in\\mathcal{I} italic\_n start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ∼ roman\_U ( \[ 10 start\_POSTSUPERSCRIPT 3 end\_POSTSUPERSCRIPT \] ), ∀ italic\_i ∈ caligraphic\_I. (top) Scenario with small heterogeneity, σ / ε 0 𝜎 𝜀 \\nicefrac{{\\sigma\_{i}}}{{\\varepsilon\_{i}}}\\sim\\mathrm{U}(\[0,10\]),\\forall i\\in% \\mathcal{I} / start\_ARG italic\_σ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_ARG start\_ARG italic\_ε start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_ARG ∼ roman\_U ( \[ 0, 10 \] ), ∀ italic\_i ∈ caligraphic\_I, (bottom) scenario with high heterogeneity, 100 \\nicefrac{{\\sigma\_{i}}}{{\\varepsilon\_{i}}}\\sim\\mathrm{U}(\[0,100\]),\\forall i\\in% \\mathcal{I} / start\_ARG italic\_σ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_ARG start\_ARG italic\_ε start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_ARG ∼ roman\_U ( \[ 0, 100 \] ), ∀ italic\_i ∈ caligraphic\_I.

## Appendix B Further details about numerical implementations

### B.1 Datasets considered in Section 4.1.

Table 3 summarizes the real-world datasets considered in Section 4.1.

Table 3: Datasets considered in Section 4.1.

| Dataset | Size | $d$ | Task |
| --- | --- | --- | --- |
| adult [^21] | 48,842 | 107 | classification |
| breast-cancer [^30] | 699 | 30 | classification |
| bank [^33] | 45,211 | 16 | classification |
| cal-housing [^19] | 20,640 | 8 | regression |
| make-regression [^36] | 1,000 | 10 | regression |
| year [^36] | 515,345 | 90 | regression |

### B.2 OpenDataVal implementations

In this section we describe more in detail the implementations of DataShapley, Leave-One-Out (LOO), and KNN-Shapley for our numerical results in Section 4.3.

DataShapley, applied to the dataset valuation problem, simply corresponds to the method coined MC in Section 4.1. Therefore, we sample datasets and output the averaged marginal contribution.

Regarding LOO, notice that it corresponds to compute just one marginal contribution, usually computed on the big coalition, i.e.,

$$
\displaystyle\mathrm{LOO}_{i}:=u(\mathcal{I})-u(\mathcal{I}\setminus\{i\}).
$$

As players’ marginal contributions to large datasets tend to be small, we have preferred to sample one dataset $\mathrm{D}$ from $\mathrm{D}_{-i}$ and to output

$$
\displaystyle\mathrm{LOO}_{i}:=u(\mathrm{D}\cup\mathrm{D}_{i})-u(\mathrm{D}).
$$

Finally, regarding KNN-Shapley, we refer the reader to [^15], Section E.3 of the appendix who explain how to adapt the method to dataset valuation.

### B.3 Owen’s Shapley value approximation

In Section A.2, we considered the Shapley value approximation referred to as Owen-Shapley as a state-of-the-art competitor to DU-Shapley. We provide in the following additional details regarding Owen-Shapley. For the other competitors, we directly refer the interested reader to [^32]. Owen [^34] studied the multilinear extension of a cooperative game and an alternative way to express the Shapley value. Formally, a cooperative game $G=(\mathcal{I},u)$ consists on a set of $I$ players $\mathcal{I}=\{1,2,...,I\}$ and a value function $u:2^{\mathcal{I}}\to\mathbb{R}$ such that, for any $S\subseteq\mathcal{I}$, $u(S)$ corresponds to the value generated by the coalition $S$. The multilinear extension of $G$, denoted $\bar{G}=(\mathcal{I},\bar{u})$, is obtained when considering the value function $\bar{u}:[0,1]^{\mathcal{I}}\to\mathbb{R}$ given by,

$$
\displaystyle\bar{u}(x_{1},x_{2},...,x_{I})=\sum_{S\subseteq\mathcal{I}}\prod_%
{i\in S}x_{i}\prod_{j\notin S}(1-x_{i})u(S).
$$

Intuitively, $\bar{u}(x_{1},x_{2},...,x_{I})$ corresponds to the expected value of a coalition when each player $i\in\mathcal{I}$ joins the coalition with probability $x_{i}$. Theorem 5 in [^34] gives an alternative way to compute the Shapley value $\varphi_{i}(u)$ of player $i$ in game $G$, namely,

$$
\displaystyle\varphi_{i}(u)
$$
 
$$
\displaystyle=\int_{0}^{1}\frac{\partial\bar{u}}{\partial x_{i}}(\tau,...,\tau%
)\mathrm{d}\tau=\int_{0}^{1}\sum_{S\subseteq\mathcal{I}\setminus\{i\}}\tau^{|S%
|}(1-\tau)^{I-|S|-1}[u(S\cup\{i\})-u(S)]\mathrm{d}\tau
$$
 
$$
\displaystyle=\int_{0}^{1}\mathbb{E}\bigl{[}u(\mathcal{E}_{i}(\tau)\cup i)-u(%
\mathcal{E}_{i}(\tau))\bigr{]}\mathrm{d}\tau=\mathbb{E}_{\tau\sim\mathrm{U}([0%
,1])}\biggl{[}\mathbb{E}\bigl{[}u(\mathcal{E}_{i}(\tau)\cup i)-u(\mathcal{E}_{%
i}(\tau))\bigr{]}\biggr{]},
$$

where $\mathcal{E}_{i}(\tau)$ is a random subset of $\mathcal{I}\setminus\{i\}$, such that, $\forall j\in\mathcal{I}\setminus\{i\}$, $j$ is included in $\mathcal{E}_{i}(\tau)$ with probability $\tau$. In words, the Shapley value of player $i$ corresponds to her expected marginal contribution to the random set $\mathcal{E}_{i}(\tau)$, when $\tau$ is uniformly distributed on $[0,1]$. This brings an alternative way to use Monte Carlo to approximate the Shapley value $\varphi_{i}(u)$, coined Owen-Shapley, as,

$$
\displaystyle\hat{\varphi}_{i}^{\text{Owen}}(u)=\frac{1}{T}\sum_{t=1}^{T}u(%
\mathcal{E}_{i}(\tau_{t})\cup i)-u(\mathcal{E}_{i}^{t}(\tau_{t})),
$$

where for each $t\in\{1,...,T\}$, we draw $\tau_{t}$ independently and uniformly in $[0,1]$ and then, create a random set $\mathcal{E}_{i}(\tau_{t})$ by adding each player $j\in\mathcal{I}\setminus\{i\}$ to it with probability $\tau_{t}$.

## Appendix C Missing proofs

### C.1 Proof of Section 2.1.3

Section 2.1.3. Let $\mathcal{S}\subseteq\mathcal{I}$ be a coalition of players and consider the value function $u$ as in (4). It follows,

$$
\displaystyle u(\mathcal{S})=\frac{-\mathrm{Tr}\bigl{[}\mathbb{E}\bigl{[}xx^{%
\top}\bigr{]}\bigr{]}}{q({\mathcal{S}})-d-1},\text{ where }q(\mathcal{S}):=%
\left\lfloor\frac{\bigl{(}\sum\limits_{i\in\mathcal{S}}\frac{\sigma_{i}}{%
\varepsilon_{i}}n_{i}\bigr{)}^{2}}{\sum\limits_{i\in\mathcal{S}}\bigl{(}\frac{%
\sigma_{i}}{\varepsilon_{i}}\bigr{)}^{2}n_{i}}\right\rfloor,\text{with the %
convention }q(\varnothing)=0.
$$

In particular, considering $p_{\mathrm{test}}=\mathrm{N}(0,\mathrm{I}_{d})$, we get,

$$
\displaystyle u(\mathcal{S})=\frac{d}{d+1-q({\mathcal{S}})}.
$$

###### Proof.

Let $\mathcal{S}\subseteq\mathcal{I}$ be a coalition of players and $X_{\mathcal{S}},Y_{\mathcal{S}}$ be the concatenation of their datasets. The linear model can be rewritten in matrix form as

$$
\displaystyle Y_{\mathcal{S}}=X_{\mathcal{S}}\theta+\eta_{\mathcal{S}},
$$

where $\eta_{\mathcal{S}}$ is the concatenation of $\eta_{i}^{(j)}$ for all $i\in\mathcal{S}$ and $j\in[n_{i}]$. Take $\hat{\theta}_{\mathcal{S}}=(X_{\mathcal{S}}^{\top}\Sigma_{\mathcal{S}}^{-1}X_{%
\mathcal{S}})^{-1}X_{\mathcal{S}}^{\top}\Sigma_{\mathcal{S}}^{-1}Y_{\mathcal{S}}$ where $\Sigma_{\mathcal{S}}=\mathrm{diag}((\varepsilon^{2}_{i})_{i\in\mathcal{S}})$, and let $x\sim p_{\mathrm{test}}$ be a hold-out testing datum in $\mathbb{R}^{d}$. It follows,

$$
\displaystyle\bigl{(}x^{\top}
$$
 
$$
\displaystyle(\theta-\hat{\theta}_{\mathcal{S}})\bigr{)}^{2}=\biggl{(}\sum_{i%
\in\mathcal{S}}\eta_{i}\varepsilon_{i}^{-2}X_{i}\biggr{)}\biggl{(}\sum_{i\in%
\mathcal{S}}X_{i}^{\top}\varepsilon_{i}^{-2}X_{i}\biggr{)}^{-1}xx^{\top}\biggl%
{(}\sum_{i\in\mathcal{S}}X_{i}^{\top}\varepsilon_{i}^{-2}X_{i}\biggr{)}^{-1}%
\biggl{(}\sum_{i\in\mathcal{S}}\eta_{i}\varepsilon_{i}^{-2}X_{i}\biggr{)}
$$
 
$$
\displaystyle=\mathrm{Tr}\biggl{[}\biggl{(}\sum_{i\in\mathcal{S}}\eta_{i}%
\varepsilon_{i}^{-2}X_{i}\biggr{)}\biggl{(}\sum_{i\in\mathcal{S}}X_{i}^{\top}%
\varepsilon_{i}^{-2}X_{i}\biggr{)}^{-1}xx^{\top}\biggl{(}\sum_{i\in\mathcal{S}%
}X_{i}^{\top}\varepsilon_{i}^{-2}X_{i}\biggr{)}^{-1}\biggl{(}\sum_{i\in%
\mathcal{S}}\eta_{i}\varepsilon_{i}^{-2}X_{i}\biggr{)}\biggr{]}
$$
 
$$
\displaystyle=\mathrm{Tr}\biggl{[}xx^{\top}\biggl{(}\sum_{i\in\mathcal{S}}X_{i%
}^{\top}\varepsilon_{i}^{-2}X_{i}\biggr{)}^{-1}\biggl{(}\sum_{i\in\mathcal{S}}%
\eta_{i}\varepsilon_{i}^{-2}X_{i}\biggr{)}\biggl{(}\sum_{i\in\mathcal{S}}\eta_%
{i}\varepsilon_{i}^{-2}X_{i}\biggr{)}\biggl{(}\sum_{i\in\mathcal{S}}X_{i}^{%
\top}\varepsilon_{i}^{-2}X_{i}\biggr{)}^{-1}\biggr{]}
$$
 
$$
\displaystyle=\mathrm{Tr}\biggl{[}xx^{\top}\biggl{(}\sum_{i\in\mathcal{S}}X_{i%
}^{\top}\varepsilon_{i}^{-2}X_{i}\biggr{)}^{-1}\biggl{(}\sum_{i\in\mathcal{S}}%
\sum_{j\in\mathcal{S}}X_{i}^{\top}\varepsilon_{i}^{-2}\eta_{i}\eta_{j}^{\top}%
\varepsilon_{j}^{-2}X_{j}\biggr{)}\biggl{(}\sum_{i\in\mathcal{S}}X_{i}^{\top}%
\varepsilon_{i}^{-2}X_{i}\biggr{)}^{-1}\biggr{]}
$$

We take expectation with respect to the different stochastic terms. Since $\eta_{i}^{(k)}\sim\mathrm{N}(0,\varepsilon_{i}^{2})$ for any $i\in\mathcal{S},k\in[n_{i}]$, it holds,

$$
\displaystyle\mathbb{E}_{(\eta_{i}^{(k)}\sim\mathrm{N}(0,\varepsilon_{i}^{2}))%
_{i\in\mathcal{S}}^{k\in[n_{i}]}}
$$
 
$$
\displaystyle\bigl{[}\bigl{(}x^{\top}(\theta-\hat{\theta}_{\mathcal{S}})\bigr{%
)}^{2}\bigr{]}
$$
 
$$
\displaystyle=\mathrm{Tr}\biggl{[}xx^{\top}\biggl{(}\sum_{i\in\mathcal{S}}X_{i%
}^{\top}\varepsilon_{i}^{-2}X_{i}\biggr{)}^{-1}\biggl{(}\sum_{i\in\mathcal{S}}%
X_{i}^{\top}\varepsilon_{i}^{-2}X_{i}\biggr{)}\biggl{(}\sum_{i\in\mathcal{S}}X%
_{i}^{\top}\varepsilon_{i}^{-2}X_{i}\biggr{)}^{-1}\biggr{]}
$$
 
$$
\displaystyle=\mathrm{Tr}\biggl{[}xx^{\top}\biggl{(}\sum_{i\in\mathcal{S}}X_{i%
}^{\top}\varepsilon_{i}^{-2}X_{i}\biggr{)}^{-1}\biggr{]}
$$

Since players distributions differ on their variances, $\sum_{i\in\mathcal{S}}X_{i}^{\top}\varepsilon_{i}^{-2}X_{i}$ corresponds to a semi-correlated Wishart random variable where each $\frac{1}{\varepsilon_{i}}X_{i}\sim\mathrm{N}(0,(\frac{\sigma_{i}}{\varepsilon_%
{i}})^{2}\mathrm{I}_{d})$. In particular, the semi-correlated Wishart random variable can be approximated by a central Wishart distribution [^37] [^43], whose precision depends on the homogeneity of the coefficients $\sigma_{i}/\varepsilon_{i}$ over all $i\in\mathcal{I}$, as showed in [^20]. It follows,

$$
\displaystyle\mathbb{E}_{(X_{i}\sim\mathrm{N}(0,\sigma_{i}^{2}\mathrm{I}_{d}))%
_{i\in\mathcal{S}}}\left[\biggl{(}\sum_{i\in\mathcal{S}}X_{i}^{\top}%
\varepsilon_{i}^{-2}X_{i}\biggr{)}^{-1}\right]\approx\frac{\mathrm{I}_{d}}{(q(%
{\mathcal{S}})-d-1)}\,,
$$

where

$$
\displaystyle q(\mathcal{S}):=\left\lfloor\frac{\bigl{(}\sum\limits_{i\in%
\mathcal{S}}\frac{\sigma_{i}}{\varepsilon_{i}}n_{i}\bigr{)}^{2}}{\sum\limits_{%
i\in\mathcal{S}}\bigl{(}\frac{\sigma_{i}}{\varepsilon_{i}}\bigr{)}^{2}n_{i}}%
\right\rfloor.
$$

With all this in mind, it follows,

$$
\displaystyle\mathbb{E}_{\begin{subarray}{c}(\eta_{i}^{(j)}\sim\mathrm{N}(0,%
\varepsilon_{i}^{2}))_{i\in\mathcal{S}}^{j\in[n_{i}]}\\
(X_{i}\sim\mathrm{N}(0,\sigma_{i}^{2}\mathrm{I}_{d}))_{i\in\mathcal{S}}\end{%
subarray}}\bigl{[}\bigl{(}x^{\top}(\theta-\hat{\theta}_{\mathcal{S}})\bigr{)}^%
{2}\bigr{]}
$$
 
$$
\displaystyle=\mathrm{Tr}\biggl{[}xx^{\top}\frac{\mathrm{I}_{d}}{(q({\mathcal{%
S}})-d-1)}\biggr{]}=\frac{1}{q({\mathcal{S}})-d-1}\mathrm{Tr}\bigl{[}xx^{\top}%
\bigr{]}.
$$

In particular, considering $p_{\mathrm{test}}=\mathrm{N}(0,\mathrm{I}_{d})$, we get,

$$
\displaystyle u(\mathcal{S})=\frac{d}{d+1-q({\mathcal{S}})}.
$$

∎

### C.2 Proof of Theorem 1

Theorem 1. Let $\{n_{i},\gamma_{i}\}_{i\in[I]}$ be two sequences of positive numbers such that the following limits

$$
\displaystyle\lim_{I\to\infty}\frac{1}{I}\sum\nolimits_{i\in[I]}n_{i}\gamma_{i%
}=\mu_{A},\quad\lim_{I\to\infty}\frac{1}{I}\sum\nolimits_{i\in[I]}(n_{i}\gamma%
_{i}-\mu_{A})^{2}=\sigma^{2}_{A},
$$
$$
\displaystyle\lim_{I\to\infty}\frac{1}{I}\sum\nolimits_{i\in[I]}n_{i}\gamma_{i%
}^{2}=\mu_{B},\quad\lim_{I\to\infty}\frac{1}{I}\sum\nolimits_{i\in[I]}(n_{i}%
\gamma_{i}^{2}-\mu_{B})^{2}=\sigma^{2}_{B}\,,
$$

all exist, for some constants $\mu_{A},\mu_{B},\sigma_{A},\sigma_{B}>0$. Let $\mathbf{K}\sim\mathrm{U}(\{0,\ldots,I\})$, $\mathcal{S}_{\mathbf{K}}\sim\mathrm{U}([2^{\mathcal{I}}_{\mathbf{K}}])$, and define $q(\mathcal{S}_{\mathbf{K}})$ as in (14) for the third use-case. Then, almost surely, $\nicefrac{{q(\mathcal{S}_{\mathbf{K}})}}{{q(\mathcal{I})}}\xrightarrow{I\to%
\infty}\mathrm{U}([0,1])$.

###### Proof.

Introduce, for any $t,t_{0}\in(0,1)$ and any $s\operatorname*{>}0$,

$$
\displaystyle\mu_{A}(I)=\frac{1}{I}\sum_{i\in[I]}n_{i}\gamma_{i},\quad\mu_{B}(%
I)=\frac{1}{I}\sum_{i\in[I]}n_{i}\gamma_{i}^{2},
$$
$$
\displaystyle Y_{A}(t,I)=\sum_{i\in\mathcal{S}_{{\lfloor It\rfloor}}}n_{i}%
\gamma_{i},\quad Y_{B}(t,I)=\sum_{i\in\mathcal{S}_{{\lfloor It\rfloor}}}n_{i}%
\gamma_{i}^{2}.
$$
$$
\displaystyle R_{A}(I,t_{0},s)=\mathbb{P}\biggl{(}\sup_{t>t_{0}}\biggl{|}\frac%
{Y_{A}(t,I)}{\lfloor It\rfloor}-\mu_{A}(I)\biggr{|}>s\biggr{)},
$$
$$
\displaystyle R_{B}(I,t_{0},s)=\mathbb{P}\biggl{(}\sup_{t>t_{0}}\biggl{|}\frac%
{Y_{B}(t,I)}{\lfloor It\rfloor}-\mu_{B}(I)\biggr{|}>s\biggr{)}.
$$

By construction, $Y_{A}(t,I)$ and $Y_{B}(t,I)$ are sums of sampling without replacement of $\lfloor It\rfloor$ elements. Therefore, by Corollary 1.3 in [^39], for $s$ fixed, there exists $I_{0}^{A},I_{0}^{B}\in\mathbb{N}$ such that,

$$
\displaystyle R_{A}(I,t_{0},s)\leq\frac{(1-t_{0})\sigma_{A}^{2}}{\lfloor It_{0%
}\rfloor s^{2}},\forall I\geq I_{0}^{A}\text{ and }R_{B}(I,t_{0},s)\leq\frac{(%
1-t_{0})\sigma_{B}^{2}}{\lfloor It_{0}\rfloor s^{2}},\forall I\geq I_{0}^{B}.
$$

In other words, for any $s\operatorname*{>}0$ and $I$ large enough, almost surely, it holds,

$$
\displaystyle\biggl{|}\frac{Y_{A}(t,I)}{{\lfloor It\rfloor}}-\mu_{A}(I)\biggr{%
|}\leq s\text{ and }\biggl{|}\frac{Y_{B}(t,I)}{{\lfloor It\rfloor}}-\mu_{B}(I)%
\biggr{|}\leq s.
$$

It follows,

$$
\displaystyle\biggl{|}\frac{q(\mathcal{S}_{\lfloor It\rfloor})}{\lfloor It%
\rfloor}-\frac{\mu_{A}(I)^{2}}{\mu_{B}(I)}\biggr{|}
$$
 
$$
\displaystyle=\biggl{|}\frac{1}{\lfloor It\rfloor}\cdot\frac{Y_{A}(t,I)^{2}}{Y%
_{B}(t,I)}-\frac{\mu_{A}(I)^{2}}{\mu_{B}(I)}\biggr{|}
$$
 
$$
\displaystyle=\biggl{|}\biggl{(}\frac{Y_{A}(t,I)^{2}}{\lfloor It\rfloor^{2}}-%
\mu_{A}(I)^{2}+\mu_{A}(I)^{2}\biggr{)}\biggl{(}\frac{\lfloor It\rfloor}{Y_{B}(%
t,I)}-\frac{1}{\mu_{B}(I)}\biggr{)}
$$
 
$$
\displaystyle\quad+\biggl{(}\frac{Y_{A}(t,I)^{2}}{\lfloor It\rfloor^{2}}-\mu_{%
A}(I)^{2}\biggr{)}\frac{1}{\mu_{B}(I)}\biggr{|}
$$
 
$$
\displaystyle\leq s\biggl{(}s+\mu_{A}(I)+\frac{1}{\mu_{B}(I)}\biggr{)},
$$

which is arbitrarily small as $\mu_{A}(I),\mu_{B}(I)\to\mu_{A},\mu_{B}\operatorname*{<}\infty$. Therefore, almost surely,

$$
\displaystyle\lim_{I\to\infty}\frac{q(\mathcal{S}_{\lfloor It\rfloor})}{I\mu_{%
A}(I)^{2}/\mu_{B}(I)}=t.
$$

The proof concludes noticing that

$$
\displaystyle q(\mathcal{I})=\frac{I\mu_{A}(I)^{2}}{\mu_{B}(I)},
$$

and that $\mathbf{K}=\lfloor IU\rfloor$ with $U\sim\mathrm{U}([0,1])$. ∎

### C.3 Proof of Theorem 2

To prove Theorem 2, we need two preliminary results: Lemma 3, which itself needs two supplementary results (Lemmas 1 and 2), and Lemma 4, which is directly proved.

#### C.3.1 Technical lemmata

###### Lemma 1.

Consider a set of $I$ values $N=\{n_{1},\ldots,n_{I}\}$. Let $X_{1},\ldots,X_{k}$ and $Y_{1},\ldots,Y_{k}$ denote, respectively, $k$ random samples with and without replacement from $N$. For any continuous and convex function $f$, it follows,

$$
\displaystyle\mathbb{E}\biggl{[}f\biggl{(}\sum_{i=1}^{k}Y_{i}\biggr{)}\biggr{]%
}\leq\mathbb{E}\biggl{[}f\biggl{(}\sum_{i=1}^{k}X_{i}\biggr{)}\biggr{]}
$$

###### Proof.

The proof follows from [^14]. ∎

###### Lemma 2.

Let $I\in\mathbb{N}$, $N:=\{n_{1},\ldots,n_{I}\}\in\mathbb{R}_{+}^{I}$, $\mu=\frac{1}{I}\sum_{i=1}^{I}n_{i}$ be their mean value and $\sigma^{2}=\frac{1}{I}\sum_{i=1}^{I}(n_{i}-\mu)^{2}$ be their variance. For $k\in\{0,\ldots,n\}$, let $\mathcal{S}_{k}\sim\mathrm{U}(\{S_{k}\subseteq[I]:|S_{k}|=k\})$ be a uniform random variable on the subsets of $\{1,\ldots,I\}$ of size $k$, and $n_{\mathcal{S}_{k}}=\sum_{i\in\mathcal{S}_{k}}n_{i}$ be the random variable defined by the sum of the elements of $\mathcal{S}_{k}$. Let $\mathbf{K}\sim\mathrm{U}(\{0,\ldots,I\})$ and define $\mathbf{Y}=n_{\mathcal{S}_{\mathbf{K}}}$. Then,

$$
\displaystyle\mathbb{E}[\mathbf{Y}-\mu\mathbf{K}\mid\mathbf{K}=k]=0,
$$
$$
\displaystyle\mathbb{E}\bigl{[}\left(\mathbf{Y}-\mu\mathbf{K}\right)^{2}\mid%
\mathbf{K}=k\bigr{]}\leq k\sigma^{2}.
$$

###### Proof.

We prove (61) directly.

$$
\displaystyle\mathbb{E}[\mathbf{Y}\mid\mathbf{K}=k]
$$
 
$$
\displaystyle=\sum_{S_{k}\subseteq[I]:|S_{k}|=k}n_{S_{k}}\frac{1}{\binom{I}{k}%
}=\frac{1}{\binom{I}{k}}\sum_{S_{k}\subseteq[I]:|S_{k}|=k}\sum_{i\in S_{k}}n_{i}
$$
 
$$
\displaystyle=\frac{1}{\binom{I}{k}}\sum_{i\in[I]}\sum_{\begin{subarray}{c}S_{%
k}\subseteq[I]:|S_{k}|=k\\
i\in S_{k}\end{subarray}}n_{i}
$$
 
$$
\displaystyle=\frac{1}{\binom{I}{k}}\sum_{i\in[I]}n_{i}\binom{I-1}{k-1}
$$
 
$$
\displaystyle=\frac{(I-k)!k!}{I!}\cdot\frac{(I-1)!}{(k-1)!(I-k)!}\sum_{i\in[I]%
}n_{i}=\mu k.
$$

Thus, (61) follows as $\mathbb{E}[\mu\mathbf{K}\mid\mathbf{K}=k]=\mu k$. To prove (62), let $(\mathbf{X}_{i})_{i=1}^{k}$ be $k$ independent samples from the set $N$. From Lemma 1 it holds,

$$
\displaystyle\mathbb{E}\bigl{[}\left(\mathbf{Y}-\mu\mathbf{K}\right)^{2}\mid%
\mathbf{K}=k\bigr{]}
$$
 
$$
\displaystyle\leq\mathbb{E}\biggl{[}\bigl{(}\mu\mathbf{K}-\sum_{i=1}^{\mathbf{%
K}}\mathbf{X}_{i}\bigr{)}^{2}\mid\mathbf{K}=k\biggr{]}=\mathbb{E}\biggl{[}%
\biggl{(}\sum_{i=1}^{\mathbf{K}}\left(\mu-\mathbf{X}_{i}\right)\biggr{)}^{2}%
\mid\mathbf{K}=k\biggr{]}.
$$

Therefore,

$$
\displaystyle\mathbb{E}\bigl{[}(\mathbf{Y}-
$$
 
$$
\displaystyle\mu\mathbf{K})^{2}\mid\mathbf{K}=k\bigr{]}\leq\mathbb{E}\biggl{[}%
\biggl{(}\sum_{i=1}^{\mathbf{K}}\sum_{j=1}^{\mathbf{K}}\left(\mu-\mathbf{X}_{i%
}\right)\left(\mu-\mathbf{X}_{j}\right)\biggr{)}\mid\mathbf{K}=k\biggr{]}
$$
 
$$
\displaystyle=\mathbb{E}\biggl{[}\biggl{(}\sum_{i=1}^{\mathbf{K}}\sum_{j=1}^{%
\mathbf{K}}\left(\mu^{2}-\mu(\mathbf{X}_{i}+\mathbf{X}_{j})+\mathbf{X}_{i}%
\mathbf{X}_{j}\right)\biggr{)}\mid\mathbf{K}=k\biggr{]}
$$
 
$$
\displaystyle=\sum_{i=1}^{k}\sum_{j=1}^{k}\left(\mu^{2}-\mu(\mathbb{E}[\mathbf%
{X}_{i}\mid\mathbf{K}=k]+\mathbb{E}[\mathbf{X}_{j}\mid\mathbf{K}=k])+\mathbb{E%
}[\mathbf{X}_{i}\mathbf{X}_{j}\mid\mathbf{K}=k]\right)
$$
 
$$
\displaystyle=\sum_{i=1}^{k}\sum_{j=1}^{k}\left(\mu^{2}-\mu(\mathbb{E}[\mathbf%
{X}_{i}]+\mathbb{E}[\mathbf{X}_{j}])+\mathbb{E}[\mathbf{X}_{i}\mathbf{X}_{j}]\right)
$$
 
$$
\displaystyle=\sum_{i=1}^{k}\left(\mu^{2}-2\mu\mathbb{E}[\mathbf{X}_{i}]+%
\mathbb{E}[\mathbf{X}_{i}^{2}]\right)+\sum_{i=1}^{k}\sum_{\begin{subarray}{c}j%
=1\\
j\neq i\end{subarray}}^{k}\left(\mu^{2}-\mu(\mathbb{E}[\mathbf{X}_{i}]+\mathbb%
{E}[\mathbf{X}_{j}])+\mathbb{E}[\mathbf{X}_{i}]\mathbb{E}[\mathbf{X}_{j}]\right)
$$
 
$$
\displaystyle=\sum_{i=1}^{k}\mathbb{E}\bigl{[}\left(\mu-\mathbf{X}_{i}\right)^%
{2}\bigr{]}+\sum_{i=1}^{k}\sum_{\begin{subarray}{c}j=1\\
j\neq i\end{subarray}}^{k}\left(\mu^{2}-2\mu^{2}+\mu^{2}\right)
$$
 
$$
\displaystyle=\sum_{i=1}^{k}\mathbb{E}\bigl{[}\left(\mu-\mathbf{X}_{i}\right)^%
{2}\bigr{]}=\sum_{i=1}^{k}\text{Var}\left(\mu-\mathbf{X}_{i}\right)=k\sigma^{2}.
$$

The steps come from rearranging the terms, using the independence of $\mathbf{X}_{i}$ with respect to $\mathbf{K}$, the independence of $\mathbf{X}_{i},\mathbf{X}_{j}$ for $i\neq j$, and finally that $\mathbb{E}[\mathbf{X}_{i}]=\mu$ and $\text{Var}\left(\mathbf{X}_{i}\right)=\sigma^{2}$. ∎

###### Lemma 3.

Let $I\in\mathbb{N}$, $N:=\{n_{1},\ldots,n_{I}\}\in\mathbb{R}_{+}^{I}$, and define,

$$
\displaystyle\mu=\frac{1}{I}\sum_{i=1}^{I}n_{i},\quad\sigma^{2}=\frac{1}{I}%
\sum_{i=1}^{I}(n_{i}-\mu)^{2},\quad{n}^{\text{max}}=\max_{i\in\mathcal{I}}n_{i},
$$
$$
\displaystyle R:=\max_{i\in[I]}|n_{i}-\mu|,\quad\tau=\max_{i\in[I]}n_{i}/\min_%
{i\in[I]}n_{i}.
$$

Consider $\mathcal{S}_{\mathbf{K}}$, $n_{\mathcal{S}_{\mathbf{K}}}$, $\mathbf{K}$, and $\mathbf{Y}$ as in Lemma 2. Let $w:\mathbb{R}_{+}\to\mathbb{R}$ be a function in $\mathcal{C}^{2}$, increasing, and suppose there exists $\kappa\in\mathbb{R}_{+}$, such that,

$$
\displaystyle\bigl{|}w^{(2)}(n)\bigr{|}\leq\frac{\kappa}{n^{2}},\forall n%
\operatorname*{>}0,
$$

where $w^{(k)}$ is the k-th derivative of $w$. Then, it holds,

$$
\displaystyle\bigl{|}\mathbb{E}[w(\mu\mathbf{K})-w(\mathbf{Y})]\bigr{|}\leq%
\frac{\kappa}{2\mu^{2}I}\left(9\sigma^{2}(1+\ln(I))+\frac{2R^{2}\tau^{2}}{{n}^%
{\text{max}}}\right).
$$

###### Proof.

The proof considers a second-order Taylor extension of $w$ at $\mu k$ to recover the expected value of $\mathbb{E}[w(\mu\mathbf{K})-w(\mathbf{Y})]$. Noticing that the first derivative has a null expected value, the upper bound stated on the Lemma comes from bounding the expected value of the second derivative.

The Taylor-Lagrange Theorem on $w$ at $\mu k>0$ provides,

$$
\displaystyle w(y)=w(\mu k)+w^{(1)}(\mu k)(\mu k-y)+w^{(2)}(\tau)\frac{(\mu k-%
y)^{2}}{2},
$$

for some $\tau$ between $y$ and $\mu k$. Therefore, there exists a random variable $\mathrm{T}$, almost surely between $\mu\mathbf{K}_{+}$ and $\mathbf{Y}$, such that,

$$
\displaystyle\mathbb{E}[w(\mathbf{Y})-w(\mu\mathbf{K}_{+})]=\mathbb{E}\biggl{[%
}w^{(1)}(\mu\mathbf{K}_{+})(\mu\mathbf{K}_{+}-\mathbf{Y})+\frac{1}{2}w^{(2)}(%
\mathrm{T})(\mu\mathbf{K}_{+}-\mathbf{Y})^{2}\biggr{]},
$$

where $\mathbf{K}_{+}$ corresponds to $\mathbf{K}$ conditioned to be positive. To avoid overcharging the notation, we drop the index from $\mathbf{K}_{+}$. We observe that,

$$
\displaystyle\mathbb{E}\biggl{[}w^{(1)}(\mu\mathbf{K})(\mu\mathbf{K}-\mathbf{Y%
})\biggr{]}
$$
 
$$
\displaystyle=\mathbb{E}\biggl{[}\mathbb{E}\bigl{[}w^{(1)}(\mu\mathbf{K})(\mu%
\mathbf{K}-\mathbf{Y})\mid\mathbf{K}=k\bigr{]}\biggr{]}
$$
 
$$
\displaystyle=\mathbb{E}\biggl{[}w^{(1)}(\mu k)\mathbb{E}\bigl{[}(\mu\mathbf{K%
}-\mathbf{Y})\mid\mathbf{K}=k\bigr{]}\biggr{]}=0,
$$

by Lemma 2, Equation (61). Therefore,

$$
\displaystyle\bigl{|}\mathbb{E}[w(\mathbf{Y})-w(\mu\mathbf{K})]\bigr{|}
$$
 
$$
\displaystyle=\frac{1}{2}\bigl{|}\mathbb{E}\bigl{[}w^{(2)}(\mathrm{T})(\mu%
\mathbf{K}-\mathbf{Y})^{2}\bigr{]}\bigr{|}
$$
 
$$
\displaystyle\leq\frac{1}{2}\mathbb{E}\bigl{[}\bigl{|}w^{(2)}(\mathrm{T})\bigr%
{|}(\mu\mathbf{K}-\mathbf{Y})^{2}\bigr{]}
$$
 
$$
\displaystyle\leq\frac{1}{2}\mathbb{E}\biggl{[}\frac{\kappa}{\mathrm{T}^{2}}(%
\mu\mathbf{K}-\mathbf{Y})^{2}\biggr{]}=\frac{\kappa}{2}\mathbb{E}\biggl{[}%
\frac{1}{\mathrm{T}^{2}}(\mu\mathbf{K}-\mathbf{Y})^{2}\biggr{]}.
$$

Setting $\mathbf{I}:=\bigl{\{}|\mu\mathbf{K}-\mathbf{Y}|\leq\frac{1}{2}(\mu\mathbf{K}+%
\mathbf{Y})\bigr{\}}$, the previous expected value can be expressed as,

$$
\displaystyle\mathbb{E}\biggl{[}\frac{1}{\mathrm{T}^{2}}(\mu\mathbf{K}-\mathbf%
{Y})^{2}\biggr{]}=\mathbb{E}\biggl{[}\frac{1}{\mathrm{T}^{2}}(\mu\mathbf{K}-%
\mathbf{Y})^{2}\cdot\mathbf{I}\biggr{]}+\mathbb{E}\biggl{[}\frac{1}{\mathrm{T}%
^{2}}(\mu\mathbf{K}-\mathbf{Y})^{2}\cdot\mathbf{I}^{c}\biggr{]}.
$$

We deal with each term separately. Notice that, as $\mathrm{T}$ is almost surely between $\mathbf{Y}$ and $\mu\mathbf{K}$,

$$
\displaystyle|\mu\mathbf{K}-\mathbf{Y}|\leq\frac{1}{2}(\mu\mathbf{K}+\mathbf{Y%
})\Longrightarrow\mathrm{T}\geq\frac{1}{3}\mu\mathbf{K}.
$$

Thus,

$$
\displaystyle\mathbb{E}\biggl{[}\frac{(\mu\mathbf{K}-\mathbf{Y})^{2}}{\mathrm{%
T}^{2}}\cdot\mathbf{I}\biggr{]}
$$
 
$$
\displaystyle\leq\mathbb{E}\left[\frac{(\mu\mathbf{K}-\mathbf{Y})^{2}}{(\frac{%
\mu\mathbf{K}}{3})^{2}}\cdot\mathbf{I}\right]=\frac{9}{\mu^{2}}\sum_{k=1}^{I}%
\frac{1}{I}\cdot\mathbb{E}\left[\frac{(\mu k-\mathbf{Y})^{2}}{k^{2}}\cdot%
\mathbf{I}\mid\mathbf{K}=k\right]
$$
 
$$
\displaystyle\leq\frac{9}{I\mu^{2}}\sum_{k=1}^{I}\mathbb{E}\left[\frac{(\mu k-%
\mathbf{Y})^{2}}{k^{2}}\mid\mathbf{K}=k\right]
$$
 
$$
\displaystyle\leq\frac{9}{I\mu^{2}}\sum_{k=1}^{I}\frac{k\sigma^{2}}{k^{2}}
$$
 
$$
\displaystyle\leq\frac{9\sigma^{2}}{I\mu^{2}}\cdot(1+\ln(I)).
$$

Regarding the second term, denote $\overline{n}:=\max_{i\in\mathcal{I}}n_{i}$ and $\underline{n}:=\min_{i\in\mathcal{I}}n_{i}$. As $\mathbf{K}\underline{n}\leq\min\{\mu\mathbf{K},\mathbf{Y}\}\leq\mathrm{T}$, we have,

$$
\displaystyle\mathbb{E}\biggl{[}\frac{(\mu\mathbf{K}-\mathbf{Y})^{2}}{\mathrm{%
T}^{2}}\cdot\mathbf{I}^{c}\biggr{]}
$$
 
$$
\displaystyle\leq\mathbb{E}\left[\frac{(R\mathbf{K})^{2}}{\mathrm{T}^{2}}\cdot%
\mathbf{I}^{c}\right]\leq\mathbb{E}\left[\frac{(R\mathbf{K})^{2}}{(\underline{%
n}\mathbf{K})^{2}}\cdot\mathbf{I}^{c}\right]
$$
 
$$
\displaystyle=\frac{R^{2}}{I\underline{n}^{2}}\sum_{k=1}^{I}\mathbb{E}\left[%
\frac{1}{k^{2}}k^{2}\cdot\mathbf{I}^{c}\mid\mathbf{K}=k\right]
$$
 
$$
\displaystyle=\frac{R^{2}}{I\underline{n}^{2}}\sum_{k=1}^{I}\mathbb{P}\left(|%
\mu k-\mathbf{Y}|>\frac{1}{2}(\mu k+\mathbf{Y})\mid\mathbf{K}=k\right)
$$
 
$$
\displaystyle\leq\frac{R^{2}}{I\underline{n}^{2}}\sum_{k=1}^{I}\mathbb{P}\left%
(|\mu k-\mathbf{Y}|>\frac{\mu k}{2}\mid\mathbf{K}=k\right)
$$
 
$$
\displaystyle\leq\frac{R^{2}}{I\underline{n}^{2}}\sum_{k=1}^{I}\exp\biggl{(}-%
\frac{\mu^{2}k}{2\overline{n}}\biggr{)}
$$
 
$$
\displaystyle=\frac{2R^{2}\tau^{2}}{I\mu^{2}\overline{n}}\sum_{k=1}^{I}\frac{%
\mu^{2}}{2\overline{n}}\exp\biggl{(}-\frac{\mu^{2}k}{2\overline{n}}\biggr{)}
$$
 
$$
\displaystyle\leq\frac{2R^{2}\tau^{2}}{I\mu^{2}\overline{n}}\int_{0}^{\infty}%
\frac{\mu^{2}}{2\overline{n}}\exp\biggl{(}-\frac{\mu^{2}k}{2\overline{n}}%
\biggr{)}dk=\frac{2R^{2}\tau^{2}}{I\mu^{2}\overline{n}},
$$

as the integral corresponds to the cumulative distribution function of an exponential random variable of parameter $\lambda=\mu^{2}/2\overline{n}$. The upper bound on the theorem’s statement is obtained when gathering all together. ∎

###### Lemma 4.

Let $w:\mathbb{R}_{+}\to\mathbb{R}_{+}$ be a smooth and increasing function such that

$$
\lim_{n\to\infty}n^{2}|w^{(2)}(n)|\operatorname*{<}\infty.
$$

Then, there exists $\kappa\operatorname*{>}0$ such that $n^{2}|w^{(2)}(n)|\leq\kappa$.

###### Proof.

Notice that the assumptions imply, in particular, that $|w^{(2)}(n)|$ is bounded. We argue by contradiction. Suppose that for any $m\operatorname*{>}0$, there exists $n_{m}$ such that

$$
n_{m}^{2}|w^{(2)}(n_{m})|>m.
$$

Suppose the sequence $(n_{m})_{m}$ converges to a point $n^{*}$. Then,

$$
\displaystyle\lim_{m\to\infty}n_{m}^{2}|w^{(2)}(n_{m})|\operatorname*{>}\lim_{%
m\to\infty}m=\infty,
$$

which is a contradiction with $|w^{(2)}(n)|$ being bounded. Therefore, necessarily $(n_{m})_{m}$ has to diverge. However, this implies,

$$
\displaystyle\lim_{n\to\infty}n^{2}|w^{(2)}(n)|
$$
 
$$
\displaystyle=\lim_{m\to\infty}n_{m}^{2}|w^{(2)}(n_{m})|\operatorname*{>}\lim_%
{m\to\infty}m=\infty,
$$

obtaining again a contradiction. ∎

#### C.3.2 Proof of Theorem 2

We are ready to prove Theorem 2.

Theorem 2. Under Assumption H1, there exists a constant $\kappa>0$, such that, for any $i\in\mathcal{I}$, it holds,

$$
\displaystyle\bigl{|}\varphi_{i}-\psi_{i}\bigr{|}\leq\frac{\kappa}{(I-1)\mu_{-%
i}^{2}}\left(\sigma_{-i}^{2}(1+\ln(I-1))+\zeta_{-i}\right),
$$

where $\varphi_{i}$ and $\psi_{i}$ are respectively the Shapley value and the DU-Shapley of player $i$, $\mu_{-i}=\frac{1}{I-1}\sum_{j\in\mathcal{I}\setminus\{i\}}n_{j}$ is the average dataset size of other players, $\sigma^{2}_{-i}=\frac{1}{I-1}\sum_{j\in\mathcal{I}\setminus\{i\}}(n_{j}-\mu_{-%
i})^{2}$ its empirical variance, and $\zeta_{-i}$ measures the variability of the dataset sizes across players. Formally, it is defined as

$$
\zeta_{-i}:=R_{-i}^{2}\frac{\tau_{-i}^{2}}{4{n}^{\mathrm{max}}_{-i}}
$$

where $R_{-i}:=\max_{j\in\mathcal{I}\setminus\{i\}}|n_{j}-\mu_{-i}|$, ${n}^{\max}_{-i}:=\max_{j\in\mathcal{I}\setminus\{i\}}n_{j}$, and $\tau_{-i}:=\frac{{n}^{\max}_{-i}}{\min_{j\in\mathcal{I}\setminus\{i\}}n_{j}}$.

###### Proof.

Under Assumption H1, Lemma 4 implies the existence of $\kappa\operatorname*{>}0$ such that the value function $w$ satisfies all assumptions from Lemma 3. Theorem 2 comes from (a) noticing that

$$
\displaystyle\varphi_{i}=\mathbb{E}[w(\mathbf{Y}_{-i}+n_{i})-w(\mathbf{Y}_{-i}%
)],\quad\psi_{i}=\mathbb{E}[w(\mathbf{K}\mu_{-i}+n_{i})-w(\mathbf{K}\mu_{-i})],
$$

where $\mathbf{K}\sim\mathrm{U}([I-1])$ and $\mathbf{Y}_{-i}=n_{\mathcal{S}^{(i)}_{\mathbf{K}}}$ with $\mathcal{S}^{(i)}_{\mathbf{K}}$ taking values on the subsets of $\mathcal{I}\setminus\{i\}$ of size $\mathbf{K}$, (b) writing

$$
\displaystyle|\varphi_{i}-\psi_{i}|\leq\
$$
 
$$
\displaystyle|\mathbb{E}[w(\mathbf{Y}+n_{i})-w(\mathbf{K}\mu_{-i}+n_{i})]|+|%
\mathbb{E}[w(\mathbf{Y})-w(\mathbf{K}\mu_{-i})]|,
$$

and (c) applying Lemma 3 to each of the expected values, as the function $n\to w(n+n_{i})$ also satisfies H1. ∎

[^1]: Anish Agarwal, Munther Dahleh, and Tuhin Sarkar. A Marketplace for Data: An Algorithmic Solution. In *Proceedings of the ACM Conference on Economics and Computation*, page 701–726, 2019.

[^2]: Javier Castro, Daniel Gómez, and Juan Tejada. Polynomial calculation of the Shapley value based on sampling. *Computers & Operations Research*, 36(5):1726–1730, 2009.

[^3]: Georgios Chalkiadakis, Edith Elkind, and Michael J. Wooldridge. *Computational Aspects of Cooperative Game Theory*. Morgan & Claypool Publishers, 2011.

[^4]: Jianbo Chen, Le Song, Martin J. Wainwright, and Michael I. Jordan. L-Shapley and C-Shapley: Efficient Model Interpretation for Structured Data. In *International Conference on Learning Representations*, 2019.

[^5]: Shay Cohen, Eytan Ruppin, and Gideon Dror. Feature Selection Based on the Shapley Value. In *International Joint Conference on Artificial Intelligence*, page 665–670, 2005.

[^6]: Mingshu Cong, Han Yu, Xi Weng, and Siu Ming Yiu. A game-theoretic framework for incentive mechanism design in federated learning. *Federated Learning: Privacy and Incentive*, pages 205–222, 2020.

[^7]: Ian C. Covert, Scott Lundberg, and Su-In Lee. Understanding Global Feature Contributions with Additive Importance Measures. In *Advances in Neural Information Processing Systems*, 2020.

[^8]: Kate Donahue and Jon Kleinberg. Model-sharing games: Analyzing federated learning under voluntary participation. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(6):5303–5311, May 2021a. URL [https://ojs.aaai.org/index.php/AAAI/article/view/16669](https://ojs.aaai.org/index.php/AAAI/article/view/16669).

[^9]: Kate Donahue and Jon Kleinberg. Optimality and Stability in Federated Learning: A Game-theoretic Approach. In *Advances in Neural Information Processing Systems*, volume 34, 2021b.

[^10]: Amirata Ghorbani and James Zou. Data Shapley: Equitable Valuation of Data for Machine Learning. In *International Conference on Machine Learning*, 2019.

[^11]: Amirata Ghorbani, Michael Kim, and James Zou. A Distributional Framework For Data Valuation. In *International Conference on Machine Learning*, 2020.

[^12]: Amirata Ghorbani, James Zou, and Andre Esteva. Data shapley valuation for efficient batch active learning. In *2022 56th Asilomar Conference on Signals, Systems, and Computers*, pages 1456–1462. IEEE, 2022.

[^13]: László Györfi and Martin Kroll. On rate optimal private regression under local differential privacy. *arXiv preprint arXiv:2206.00114*, 2022.

[^14]: Wassily Hoeffding. Probability Inequalities for Sums of Bounded Random Variables. *Journal of the American Statistical Association*, 58(301):13–30, 1963.

[^15]: Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nezihe Merve Gurel, Bo Li, Ce Zhang, Costas Spanos, and Dawn Song. Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms. *Proc. VLDB Endow.*, 12(11):1610–1623, 2019a.

[^16]: Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nick Hynes, Nezihe Merve Gürel, Bo Li, Ce Zhang, Dawn Song, and Costas J. Spanos. Towards Efficient Data Valuation Based on the Shapley Value. In *International Conference on Artificial Intelligence and Statistics*, 2019b.

[^17]: Kevin Jiang, Weixin Liang, James Y Zou, and Yongchan Kwon. Opendataval: a unified benchmark for data valuation. *Advances in Neural Information Processing Systems*, 36, 2023.

[^18]: Jiawen Kang, Zehui Xiong, Dusit Niyato, Shengli Xie, and Junshan Zhang. Incentive mechanism for reliable federated learning: A joint optimization approach to combining reputation and contract theory. *IEEE Internet of Things Journal*, 6(6):10700–10714, 2019.

[^19]: R. Kelley Pace and Ronald Barry. Sparse spatial autoregressions. *Statistics & Probability Letters*, 33(3):291–297, 1997.

[^20]: Andre I Khuri, Thomas Mathew, and Daan G Nel. A test to determine closeness of multivariate satterthwaite’s approximation. *Journal of Multivariate Analysis*, 51(1):201–209, 1994.

[^21]: Ron Kohavi. Scaling up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid. In *International Conference on Knowledge Discovery and Data Mining*, 1996.

[^22]: Patrick Kolpaczki, Viktor Bengs, Maximilian Muschalik, and Eyke Hüllermeier. Approximating the Shapley value without marginal contributions. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pages 13246–13255, 2024.

[^23]: Yongchan Kwon and James Zou. Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for Machine Learning. In *International Conference on Artificial Intelligence and Statistics*, pages 8780–8802, 2022.

[^24]: Yongchan Kwon, Manuel A. Rivas, and James Zou. Efficient Computation and Analysis of Distributional Shapley Values. In *International Conference on Artificial Intelligence and Statistics*, pages 793–801, 2021.

[^25]: Weixin Liang, James Zou, and Zhou Yu. Beyond user self-reported likert scale ratings: A comparison model for automatic dialog evaluation. *arXiv preprint arXiv:2005.10716*, 2020.

[^26]: Weixin Liang, Kai-Hui Liang, and Zhou Yu. Herald: an annotation efficient method to detect user disengagement in social conversations. *arXiv preprint arXiv:2106.00162*, 2021.

[^27]: Scott M. Lundberg and Su-In Lee. A Unified Approach to Interpreting Model Predictions. In *Advances in Neural Information Processing Systems*, page 4768–4777, 2017.

[^28]: Scott M. Lundberg, Gabriel Erion, Hugh Chen, Alex DeGrave, Jordan M. Prutkin, Bala Nair, Ronit Katz, Jonathan Himmelfarb, Nisha Bansal, and Su-In Lee. From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1):56–67, 2020.

[^29]: Lingjuan Lyu, Xinyi Xu, Qian Wang, and Han Yu. Collaborative fairness in federated learning. *Federated Learning: Privacy and Incentive*, pages 189–204, 2020.

[^30]: Olvi L. Mangasarian, W. Nick Street, and William H. Wolberg. Breast Cancer Diagnosis and Prognosis Via Linear Programming. *Operations Research*, 43(4):570–577, 1995.

[^31]: Irwin Mann and Lloyd S. Shapley. *Values of Large Games, IV: Evaluating the Electoral College by Montecarlo Techniques*. RAND Corporation, Santa Monica, CA, 1960.

[^32]: Rory Mitchell, Joshua Cooper, Eibe Frank, and Geoffrey Holmes. Sampling Permutations for Shapley Value Estimation. *Journal of Machine Learning Research*, 23(43):1–46, 2022.

[^33]: Sérgio Moro, Paulo Cortez, and Paulo Rita. A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62:22–31, 2014.

[^34]: Guillermo Owen. Multilinear Extensions of Games. *Management Science*, 18(5):64–79, 1972.

[^35]: Konstantin D Pandl, Fabian Feiland, Scott Thiebes, and Ali Sunyaev. Trustworthy machine learning for health care: scalable data valuation with the shapley value. In *Proceedings of the Conference on Health, Inference, and Learning*, pages 47–57, 2021.

[^36]: Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Édouard Duchesnay. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12(85):2825–2830, 2011.

[^37]: Gabriel Fernando Pivaro, Santosh Kumar, Gustavo Fraidenraich, and Claudio Ferreira Dias. On the exact and approximate eigenvalue distribution for sum of wishart matrices. *IEEE Transactions on Vehicular Technology*, 66(11):10537–10541, 2017.

[^38]: Stephanie Schoch, Haifeng Xu, and Yangfeng Ji. CS-shapley: Class-wise shapley values for data valuation in classification. In *Advances in Neural Information Processing Systems*, 2022.

[^39]: Robert J Serfling. Probability inequalities for the sum in sampling without replacement. *The Annals of Statistics*, pages 39–48, 1974.

[^40]: Lloyd S. Shapley. *A Value for N-Person Games*. RAND Corporation, Santa Monica, CA, 1952.

[^41]: Dongsub Shim, Zheda Mai, Jihwan Jeong, Scott Sanner, Hyunwoo Kim, and Jongseong Jang. Online class-incremental continual learning with adversarial shapley value. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 35, pages 9630–9638, 2021.

[^42]: Rachael Hwee Ling Sim, Yehong Zhang, Mun Choon Chan, and Bryan Kian Hsiang Low. Collaborative machine learning with incentive-aware model rewards. In *Proceedings of the 37th International Conference on Machine Learning*, ICML’20. JMLR.org, 2020.

[^43]: WY Tan and RP Gupta. On approximating a linear combination of central wishart matrices with positive coefficients. *Communications in Statistics-Theory and Methods*, 12(22):2589–2600, 1983.

[^44]: Sebastian Shenghong Tay, Xinyi Xu, Chuan-Sheng Foo, and Bryan Kian Hsiang Low. Incentivizing collaboration in machine learning via synthetic data rewards. In *AAAI Conference on Artificial Intelligence*, 2021.

[^45]: Alexandre B. Tsybakov. *Introduction to Nonparametric Estimation*. Springer Publishing Company, Incorporated, 1st edition, 2008. ISBN 0387790519.

[^46]: Jiachen T Wang, Yuqing Zhu, Yu-Xiang Wang, Ruoxi Jia, and Prateek Mittal. Threshold knn-shapley: A linear-time and privacy-friendly approach to data valuation. *arXiv preprint arXiv:2308.15709*, 2023.

[^47]: Han Yu, Zelei Liu, Yang Liu, Tianjian Chen, Mingshu Cong, Xi Weng, Dusit Niyato, and Qiang Yang. A fairness-aware incentive scheme for federated learning. In *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society*, pages 393–399, 2020.