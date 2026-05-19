---
title: "Rethinking Data Value: Asymmetric Data Shapley for Structure-Aware Valuation in Data Markets and Machine Learning Pipelines"
source: "https://arxiv.org/html/2511.12863v1"
author:
published:
created: 2026-05-05
description:
tags:
  - "clippings"
---
Xi Zheng  
University of Washington; email: xzheng01@uw.edu.    Yinghui Huang Xi’an Jiaotong University; email: yinghui.huang@xjtu.edu.cn.    Xiangyu Chang Xi’an Jiaotong University; email: xiangyuchang@xjtu.edu.cn.    Ruoxi Jia  
Virginia Tech; email: ruoxijia@vt.edu.    Yong Tan University of Washington; email: ytan@uw.edu.

###### Abstract

Rigorous valuation of individual data sources is critical for fair compensation in data markets, informed data acquisition, and transparent development of ML/AI models. Classical Data Shapley (DS) provides a essential axiomatic framework for data valuation but is constrained by its symmetry axiom that assumes interchangeability of data sources. This assumption fails to capture the directional and temporal dependencies prevalent in modern ML/AI workflows, including the reliance of duplicated or augmented data on original sources and the order-specific contributions in sequential pipelines such as federated learning and multi-stage LLM fine tuning. To address these limitations, we introduce *Asymmetric Data Shapley (ADS)*, a structure-aware data valuation framework for modern ML/AI pipelines. ADS relaxes symmetry by averaging marginal contributions only over permutations consistent with an application-specific ordering of data groups. It preserves efficiency and linearity, maintains within group symmetry and directional precedence across groups, and reduces to DS when the ordering collapses to a single group. We develop two complementary computational procedures for ADS: (i) a Monte Carlo estimator (MC-ADS) with finite-sample accuracy guarantees, and (ii) a $k$ -nearest neighbor surrogate (KNN-ADS) that is exact and efficient for KNN predictors. Across representative settings with directional and temporal dependence, ADS consistently outperforms benchmark methods by distinguishing novel from redundant contributions and respecting the sequential nature of training. These results establish ADS as a principled and practical approach to equitable data valuation in data markets and complex ML/AI pipelines.

## 1 Introduction

In today’s digital economy, firms increasingly operate on continuous streams of individual- and enterprise-level data generated through search engines, social media, mobile applications, connected devices, and transactional platforms (meyer2014machine; yoganarasimhan2020search; wang2024learning). The increasing scale and persistence of digital traces have expanded opportunities to collect, integrate, and commercialize information via intermediaries and platforms, fostering new markets for data access and model-based services (pei2020survey; mehta2021sell; agarwal2019marketplace). Meanwhile, empirical evidence underscores the substantial economic value of individual-level data, as well as the frictions arising from opaque collection practices and insufficient compensation, thereby motivating transparent, consent-based exchange and procurement mechanisms (birkhead2025algorithms). As pricing, access, and compensation increasingly depend on outputs of models constructed via collected datasets, there is a need for a principled mechanism that translates downstream model utility into equitable payments to data contributors. This challenge is particularly salient in modern artificial intelligence systems, where data are costly, distinctive, and decisive for model performance. Accordingly, rigorous data valuation and incentive-aligned sharing have become central to sustaining competitive advantage.

A defining feature of today’s digital ecosystem is that a large portion of individual-level data has historically been collected by third-party brokers that monitor user behavior across websites, mobile applications, social media platforms, and e-commerce sites, often without the explicit awareness or fully informed consent of data contributors. These opaque practices fuel downstream machine learning applications while offering minimal transparency or compensation, and may further introduce bias if privacy-conscious users opt out of participation. Recent studies document these frictions and propose market mechanisms designed to elicit user consent, compensate for privacy loss, and procure representative data samples (birkhead2025algorithms). Against this backdrop, platform-mediated data marketplaces have emerged to coordinate data contributors that supply raw data, brokers that aggregate and engineer the data, and buyers who consume model outputs or data-driven services (xing2024contract; birkhead2025algorithms).

![Refer to caption](https://arxiv.org/html/2511.12863v1/figures/llm_fig/data_mkt.png)

Figure 1: Overview of data market involving multiple data contributors, a data broker, and model buyers.

Figure 1 illustrates a typical model-as-a-service (MaaS) workflow (chen2019towards; agarwal2019marketplace; tian2022data; gan2023model). In this framework, data contributors supply raw datasets that brokers clean, integrate, and use to train machine learning models, subsequently commercialized as products or services. The resulting revenues or other economic benefits are distributed back to contributors, thereby creating incentives for high-quality and sustained participation. As these markets expand, a central challenge is to quantify each participant’s contribution in a transparent and utility-aligned manner, ensuring that compensation fairly reflects the proper marginal impact of each contributor’s data (ghorbani2019data; jia2019towards; jia2019efficient; tian2022data).

In response to this pressing market challenge, we develop a principled, utility-aligned rule for compensating data contributors that can be implemented within modern MaaS pipelines. The line of research most closely aligned with this objective is Data Shapley (DS), which adapts the Shapley value from cooperative game theory to quantify each training data source’s contribution to supervised learning performance (ghorbani2019data; jia2019efficient; jia2019towards). Under DS, the value of a data source is defined as its average marginal contribution to model utility across all subsets of the remaining sources. The rule is uniquely characterized by efficiency, linearity, nullity, and symmetry, and extensive empirical evidence shows that DS can effectively identify valuable, as well as mislabeled or noisy data sources and inform data acquisition decisions (ghorbani2019data). These properties make DS a natural starting point for marketplace compensation, since it links downstream model utility to contributor-level payments. However, research on two-sided data markets shows that classical Shaply’s notion of fairness can fail when data are freely replicable and combinatorial: duplicating an informational signal can distort revenue splits under the symmetry axiom of standard Shapley allocation (agarwal2019marketplace). To address this, agarwal2019marketplace introduce robustness-to-replication as an additional fairness requirement for markets that trade freely replicable goods: the aggregate payment to a seller and any replicas must not exceed the payment that would be assigned to the seller’s original signal in the absence of replication.

Beyond replication, the symmetry axiom is often violated in modern data exchanges and training pipelines that exhibit complex dependencies among data sources. Two pervasive cases are directional dependence between original and synthetic data, and temporal dependence in sequential training pipelines (see Examples 1.1, 1.2, and 1.3 below for detailed discussions). When such dependencies are present, valuations that assume interchangeability among all data sources can misalign incentives, misallocate compensation, and undermine the sustainability of data markets. We therefore present three motivating examples that demonstrate why classical DS can yield unreasonable valuations under directional or temporal dependence, and we use these examples to motivate the structure-aware data valuation approach developed in the remainder of the paper.

###### Example 1.1 (Synthetic Data Valuation).

Data augmentation and synthesis are widely used to expand training datasets and improve generalization (shorten2019survey). Each augmented or synthetic instance is derived from one or more originals; its information is therefore conditional on those sources rather than independent. However, the symmetry axiom in DS treats original and derivative instances as interchangeable and cannot encode this directional dependence (see Lemma 3.1 in Section 3.2). The issue is amplified in the era of generative AI, where both human-created originals and outputs from generative models enter training; valuing them equally can obscure the foundational role of human-created content and raise copyright concerns (henderson2023foundation; grynbaum2023times; wang2024economic). A reasonable valuation should therefore measure the incremental, model-relevant information contributed by derivative data and distinguish replication from true informational novelty (agarwal2019marketplace).

###### Example 1.2 (Participant Valuation in Federated Learning).

Federated learning is a decentralized and iterative training paradigm in which a central server aggregates local model updates from multiple distributed data contributors to jointly train a global model (kairouz2021advances). In each communication round, the server samples a subset of contributors. Each contributor trains locally and sends only model updates, such as gradients, so raw data stays on the device. The server aggregates these updates to refine the global parameters and stores the resulting model after each round. This procedure produces a realized global model trajectory that records the sequence of applied updates.

The classical DS evaluates each contributor’s value as the average marginal contribution across all possible subsets of the remaining contributors. When applied to federated learning, this formulation would require constructing numerous counterfactual global model trajectories by permuting contributors across rounds (see Lemma 3.2 in Section 3.2). Each such trajectory entails recomputing local updates under alternative global models, redistributing these models to contributors, and re-aggregating the resulting updates across multiple rounds. The required communication and computation, together with bandwidth constraints and limited device resources, render this approach infeasible in practice (asad2023limitations; wang2020principled). The key challenge, therefore, is to design a valuation framework that leverages the sequential structure of federated learning, respects the realized global trajectory, and circumvents the need to evaluate prohibitively many counterfactual training paths.

###### Example 1.3 (Dataset Procurement in Multi-Stage LLM Fine-Tuning).

Pretrained large language models (LLMs) are increasingly adapted to domain-specific applications (zhao2023beyond), typically through a multi-stage fine-tuning process that promotes effective and stable learning (guan2025multi). Firms often acquire task-specific datasets over several stages and must determine, at each stage, which candidate dataset to purchase and how to compensate the contributor in proportion to the incremental value added to the evolving model at the current round. Two practical challenges complicate this decision. First, assessing data value by repeatedly retraining the model with the entire historical corpus together with candidate datasets is computationally prohibitive, even when the scope is limited to fine-tuning. Second, when the initial checkpoint is obtained from an external source, the earlier training data and intermediate model states are typically unavailable, which makes standard valuation methods that rely on counterfactual retraining with full historical access inapplicable. A practical compensation framework must therefore evaluate each candidate dataset’s contribution relative to the current model state, account for the sequential order of data acquisition, and avoid constructing counterfactual retraining trajectories using the entire historical corpus.

These examples illustrate that, in modern data exchanges and training pipelines, the value of a data source reflects not only its intrinsic quality but also when it enters the pipeline and how it interacts with other data sources. However, classical DS imposes symmetry, treating sources as interchangeable whenever they deliver identical marginal gains across all subsets of the remaining sources, irrespective of directional or temporal dependencies inherent in the pipeline. In real-world machine learning and AI systems where such dependencies are pervasive, this assumption breaks down and can produce misleading valuations. Building on insights from the literature on data markets and Shapley-based data valuation, we introduce Asymmetric Data Shapley (ADS). ADS retains the Shapley foundation while relaxing symmetry so that valuations respect the ordered grouping structure common in modern data exchanges and training pipelines. The resulting rule is replication-aware and structure-aware, preserves efficiency and linearity, and internalizes directional and temporal dependence into data values. Our main contributions are summarized as follows:

- Conceptually, we identify the symmetry axiom as the key limitation of classical DS in workflows with directional or temporal dependence. Through concrete examples and formal lemmas, we show that symmetry can systematically misallocate value by treating original sources and their duplicated or augmented derivatives as interchangeable and by ignoring the effects of temporal order in sequential training pipelines. These directional and temporal dependencies yield biased payouts, motivating an asymmetric valuation rule that respects inherent structure of the workflow. To address this, we propose ADS, an structure-aware data valuation framework that extend the weighted Shapley value from cooperative game theory. By incorporating application-specific precedence structure, ADS provide a principled solution for equitable data valuation in complex machine learning workflows that exhibit directional or temporal dependence.
- Theoretically, we give an axiomatic characterization showing that ADS equals the expected one-step marginal contribution under the uniform distribution over permutations that respect the pre-specified group order. We further prove a *group efficiency* property: for each group, the sum of assigned values equals its incremental utility relative to the union of the preceding groups. Taken together, these results establish ADS as a practical, structure-aware valuation framework for modern machine learning workflows.
- Computationally, we develop two practical estimators for ADS. First, a Monte Carlo estimator (MC-ADS) that, with probability at least $1-\delta$, achieves additive error at most $\epsilon$ in time $O\!\big(n\,\epsilon^{-2}\log(n/\delta)\big)$. Second, a $K$ -nearest-neighbor surrogate (KNN-ADS) that is exact when the downstream predictor is a KNN classifier, with per–test–point complexity $O(n\log n)$.
- Empirically, we evaluate ADS across three representative machine learning and data market workflows that feature directional or temporal dependence: valuing synthetic data, compensating contributors in federated learning, and procuring datasets for multi-stage LLM fine-tuning. Across all settings, ADS consistently serves as a fair and practical valuation rule, assigning higher value to informative sources and lower value to redundant, low-quality, or mislabeled ones relative to benchmark methods. In turn, this delivers contributor compensation that more faithfully reflects each source’s incremental, model-relevant utility within complex training pipelines.

The remainder of the paper is organized as follows. Section 2 reviews the literature on data markets and data valuation. Section 3 formalizes the limitations of classical DS in two representative settings: (i) directional dependence between original and synthetic data, and (ii) temporal dependence across sequential rounds. Section 4 introduces ADS: we define ordered data groups and ordered permutations, develop state-conditioned marginal contributions, and present the axiomatic characterization. Section 5 proposes two scalable algorithms for computing ADS. Section 6 presents extensive empirical results for ADS across three representative applications in machine learning and data markets. Section 7 discusses implications for marketplace design and outlines directions for future work.

## 2 Related Work

This section is structured around two interconnected strands of literature that motivate our work. First, we review the data market literature to ground our approach in practical context and requirements. Second, we survey the data valuation literature to highlight the strengths and limitations of existing approaches, thereby justifying the necessity of our framework in complex machine learning workflows.

### 2.1 Data Markets and Personal Data Exchange

Digital data are traded at scale in both business to business exchanges and platform-mediated personal data ecosystems. Three streams organize the core insights. First, pricing and bargaining for datasets and information products are shaped by bundling, screening, and downstream competition: bundling heterogeneous information goods can raise profits and in many settings improve efficiency; a monopolist seller of information screens buyers with menus of experiments; platforms balance exclusive and shared access; and competition in downstream markets determines whether to sell precise signals broadly or restrict precision and access (bakos1999bundling; bergemann2018design; bhargava2020optimal; bimpikis2019information). For dataset monetization, simple quantity based price schedules can be optimal or near optimal, and informative demonstrations during bargaining can shift negotiated prices and the division of surplus (mehta2021sell; ray2020bargaining).

Second, platform-mediated personal data markets study consent, privacy costs, and regulation. Fixed payments and centralized procurement can induce adverse selection on privacy costs and bias samples; a mechanism that truthfully elicits privacy concerns can procure low cost, unbiased data while improving transparency and compensation (birkhead2025algorithms). Regulatory models show how rights to opt in, erasure, and portability, together with security mandates, reshape data availability, market outcomes, and welfare (ke2023privacy; choe2025bright). Complementary work designs acquisition mechanisms for privacy aware individuals, including prior free procurement for unbiased estimation and Bayesian optimal mechanisms with differential privacy (chen2019prior; fallah2024optimal).

Third, and most relevant to our setting, marketplaces for training data in machine learning tie value and price to downstream predictive performance. DS aligns payouts with each data source’s average marginal contribution to model utility and has been shown to surface low quality or mislabeled data and to guide acquisition (ghorbani2019data). Market design must also contend with replication and combinatorial value; replication robust allocation and pricing mitigate distorted revenue splits when signals are duplicated (agarwal2019marketplace). Complementary mechanisms translate valuation into deployable market practices, including model-based pricing that sells trained models rather than raw data and prices against accuracy targets (chen2019demonstration), privacy preserving valuation with fair payment that integrates secure computation and verifiable settlement (tian2022private), and operational frameworks that define data boundaries and broker revenues by aggregating Shapley values across records and solving revenue maximization problems for model pricing (tian2022data). Recent surveys situate these developments within a broader family of Shapley-based methods and document applications across the digital economy (baghcheband2024shapley).

### 2.2 Data Valuation and Its Link to Data Markets

A data marketplace focuses on downstream utility such as accuracy, coverage, or risk. Data valuation maps model facing utility to contributor facing payouts and underpins prices, rebates, and revenue sharing (fleckenstein2023review). Two complementary families have emerged.

The first family is model centric. Shapley-based approaches adapt the Shapley value from cooperative game theory to data valuation by treating each data source as a cooperating player in training (shapley1953value), and foundational work on private information in digital markets motivates this perspective for data contributions (kleinberg2001value). DS operationalizes the idea with permutation based estimators and shows practical benefits for curation and acquisition (ghorbani2019data). Efficient computation is available in special cases such as nearest neighbor models (jia2019towards; jia2019efficient), while recent work streamlines estimation and broadens objectives through distributional valuation, cardinality weighted variants, learned least squares estimators, and single run attribution at foundation model scale (ghorbani2020distributional; kwon2021beta; panda2024fw; wang2024data). In decentralized settings, Federated Shapley values recover participation sequences with no extra communication beyond standard FL, and vertical-FL approaches offer model-free, privacy-preserving data valuation (wang2020principled; Han2025VFL). Model markets that support machine unlearning motivate sharded Shapley formulations for fast value updates (xia2023equitable). Non-Shapley methods estimate contribution using sensitivity analyses or learned proxies, including influence functions and representation-based approaches such as Datamodels and TRAK (koh2017understanding; ilyas2022datamodels; park2023trak). However, these methods lack the uniqueness and fairness guarantees that motivate Shapley-based rules for translating model utility into payouts.

The second family is market and policy centric. Reviews synthesize market-based, economic, and dimensional approaches to pricing, accounting for, and comparing datasets (fleckenstein2023review). Within this lens, laney2017infonomics treats information as a corporate asset for accounting, governance, and mergers and acquisitions. Business analyses measure the economic value of data and cross border data flows and outline a global data value chain (nguyen2020measuring). Policy analyses highlight nonrivalry and argue that broad access, mediated by privacy and ownership rights, can raise welfare (jones2020nonrivalry). Related proposals explore taxation and dividend style mechanisms for personal data (lucas2021tax; adams2020datasalestax).

## 3 Data Shapley and Its Limitations

This section first reviews the classical DS framework and its axiomatic foundations. We then show that DS undervalues informational originality in duplicated or augmented data because symmetry treats original and synthetic sources as interchangeable, failing to separate novelty from redundancy. Next, we examine sequential training workflows, such as federated or staged pipelines, where contributions depend on arrival time and realized model states, and evaluating value by enumerating counterfactual training histories is both conceptually misaligned with the observed trajectory and practically infeasible.

### 3.1 Preliminary: Data Shapley

DS (ghorbani2019data) provides an axiomatic rule for attributing value to training data sources according to their contributions to predictive performance. Let $D=\{z_{1},\ldots,z_{n}\}$ denote the collection of data sources, where duplicates are allowed. Each source $z$ is a finite collection of labeled instances; we write $\operatorname{\mathrm{Ins}}(z)$ for the collection of instances owned by $z$, allowing repeated instances. We use $\lvert\cdot\rvert$ for the number of sources in a collection, counting duplicates by their frequency, and define the corresponding instance count by $m(\cdot):=\lvert\operatorname{\mathrm{Ins}}(\cdot)\rvert$. For any subset of sources $S\subseteq D$, the induced instance pool is $\operatorname{\mathrm{Ins}}(S):=\bigcup_{z\in S}\operatorname{\mathrm{Ins}}(z)$, where the union aggregates frequencies so that repeated instances across sources are counted by their total frequency; the resulting source and instance counts are $\lvert S\rvert$ and $m(S)$.

Let $\mathcal{A}$ be a learning algorithm that maps any $S\subseteq D$ to a trained model $\mathcal{A}(S)$, and let $v:2^{D}\to\mathbb{R}$ be a utility function that evaluates $\mathcal{A}(S)$. For example, if the utility is accuracy on a fixed holdout set $D_{\text{test}}$, then $v(S)$ is the accuracy of $\mathcal{A}(S)$ evaluated on $D_{\text{test}}$. The goal of data valuation is to assign each data source $z\in D$ a value $\phi(z;\,v,D)\in\mathbb{R}$ that reflects its contribution to the overall model utility. DS defines $\phi$ as the unique allocation that equals each source’s expected marginal contribution over all coalitions (equivalently, over all permutations) and that satisfies four axioms: efficiency, linearity, nullity, and symmetry. We state these axioms next.

###### Axiom 3.1 (Efficiency).

$\sum_{z\in D}\phi(z;\,v,D)=v(D)-v(\emptyset)$.

###### Axiom 3.2 (Linearity).

For scalars $\alpha,\beta$ and utility functions $u,v$, $\phi(z;\alpha u+\beta v,D)=\alpha\,\phi(z;u,D)+\beta\,\phi(z;v,D)$ for all $z\in D$.

###### Axiom 3.3 (Nullity).

If $v(S\cup\{z\})=v(S)$ for all $S\subseteq D\setminus\{z\}$, then $\phi(z;\,v,D)=0$.

###### Axiom 3.4 (Symmetry).

If $v(S\cup\{z\})=v(S\cup\{z^{\prime}\})$ for all $S\subseteq D\setminus\{z,z^{\prime}\}$, then $\phi(z;\,v,D)=\phi(z^{\prime};v,D)$.

###### Definition 3.1 (Single-step marginal contribution).

For $z\in D$ and $S\subseteq D\setminus\{z\}$, the marginal contribution of $z$ to $S$ is

$$
\Delta(z\mid S)\ :=\ v(S\cup\{z\})-v(S),
$$

which is the one-step utility gain from adding $z$ to the model trained on $S$.

The DS value of $z\in D$ is the unique allocation satisfying Axioms 3.1–3.4 (shapley1953value; ghorbani2019data), and it admits the form

$$
\phi(z;\,v,D)=\frac{1}{n}\sum_{S\subseteq D\setminus\{z\}}\binom{n-1}{\lvert S\rvert}^{-1}\,\Delta(z\mid S).
$$

Equivalently, the permutation form is

$$
\phi(z;\,v,D)=\frac{1}{n!}\sum_{\pi(D)\in\Pi(D)}\Bigl[v\bigl(\pi^{<z}(D)\cup\{z\}\bigr)-v\bigl(\pi^{<z}(D)\bigr)\Bigr],
$$

where $\Pi(D)$ is the set of all permutations of $D$. For a permutation $\pi(D)=(z_{o_{1}},\ldots,z_{o_{n}})$ and the unique index $j$ such that $z_{o_{j}}=z$, the predecessor set is $\pi^{<z}(D):=\{z_{o_{1}},\ldots,z_{o_{j-1}}\}$. Eq.(3) averages the one-step utility gain of $z$ over all insertion positions across all permutations, whereas the subset form in Eq.(2) averages over all subsets with the appropriate combinatorial weights.

### 3.2 Synthetic Data and Directional Dependence

Synthetic data, encompassing both traditional augmentations (e.g., image rotations, flips, crops) and outputs from modern generative models (e.g., GANs, diffusion models), has become ubiquitous in contemporary machine learning pipelines. Such data is typically produced by transforming or extrapolating from existing data instances to enhance generalization, mitigate class imbalance, or augment training sets in low-resource contexts (shorten2019survey; feng2021survey). Classical approaches include geometric transformations such as image rotations and crops (simard2003best), algorithmic resampling methods such as SMOTE for class balancing (chawla2002smote), and generative approaches such as GANs for producing realistic synthetic instances (antoniou2017data; frid2018gan).

Synthetic instances are derived from originals, so they often add redundancy and informational overlap; their value should therefore be judged relative to the underlying originals. Under the symmetry axiom, classical DS treats sources as interchangeable and ignores directional dependence, which can assign nearly the same value to originals and to synthetics that add little new signal. Consequently, valuations can diverge from true incremental utility. To illustrate, we next present an extreme duplication case.

###### Example 1 (continued).

Let $D_{1}=\{z_{1,1},\ldots,z_{1,n}\}$ denote the collection of original data sources, where $z_{1,i}$ is contributed by the $i$ th contributor, and let $D_{2}=\{z_{2,1},\ldots,z_{2,n}\}$ be an exact duplicate with $\operatorname{\mathrm{Ins}}(z_{2,i})=\operatorname{\mathrm{Ins}}(z_{1,i})$ for all $i\in[n]:=\{1,2,\ldots,n\}$. Define $D^{\mathrm{dup}}:=D_{1}\cup D_{2}$ so that each source is duplicated once. Let $R_{\mathcal{A}}(\operatorname{\mathrm{Ins}}(S))$ denote the empirical risk of $\mathcal{A}\in\mathcal{H}$ evaluated on the instances $\operatorname{\mathrm{Ins}}(S)$ for any collection of sources $S$, and let $\mathcal{A}_{S}$ be an empirical risk minimizer as in Eq. (4). Because $R_{\mathcal{A}}(\operatorname{\mathrm{Ins}}(S))$ averages loss over instances, duplicating every source multiplies instance counts by a common factor but does not change the set of minimizers; hence $v(D^{\mathrm{dup}})=v(D_{1})$. Under DS on $D^{\mathrm{dup}}$, each original source $z_{1,i}$ has the same value as its duplicate $z_{2,i}$. Consequently, the total value attributed to the originals equals one half of the total on $D^{\mathrm{dup}}$, which coincides with the value obtained when training on $D_{1}$ alone.

###### Lemma 3.1 (Symmetric valuation under redundant duplication).

Fix a loss function $L(\cdot)$ and a hypothesis class $\mathcal{H}$. For any finite collection of data sources $S$, define empirical risk minimization over data instances by

$$
R_{\mathcal{A}}\big(\operatorname{\mathrm{Ins}}(S)\big)\;:=\;\frac{1}{\bigl|\operatorname{\mathrm{Ins}}(S)\bigr|}\sum_{(x,y)\in\operatorname{\mathrm{Ins}}(S)}L\!\big(\mathcal{A}(x),y\big),\qquad\mathcal{A}_{S}\in\arg\min_{\mathcal{A}\in\mathcal{H}}R_{\mathcal{A}}\big(\operatorname{\mathrm{Ins}}(S)\big).
$$

Then, for every $i\in[n]$, $\phi(z_{1,i};\,v,D^{\mathrm{dup}})=\;\phi(z_{2,i};\,v,D^{\mathrm{dup}}),$ and

$$
\sum_{z\in D_{1}}\phi(z;\,v,D^{\mathrm{dup}})\;=\;\sum_{z\in D_{2}}\phi(z;\,v,D^{\mathrm{dup}})\;=\;\tfrac{1}{2}\sum_{z\in D^{\mathrm{dup}}}\phi(z;\,v,D^{\mathrm{dup}})\;=\;\tfrac{1}{2}\bigl(v(D_{1})-v(\varnothing)\bigr).
$$

Uniform attribution under redundancy has important normative and technical consequences. In the marketplace of Figure 1, symmetry assigns equal credit to an original source and to a trivially duplicated source. Together with Lemma 3.1, this implies that a broker who merely duplicates the original corpus can capture one half of the total value, which obscures the dependence of synthetic data on the original sources and misallocates rewards away from contributors. On the technical side, symmetric valuation can encourage the accumulation of redundant synthetic data. Empirical evidence indicates that repeated training on such data induces self-contamination and forms of model collapse, which degrade out-of-sample generalization and produce low-variance predictions (shumailov2024ai; yang2024understanding; gerstgrasser2024model).

Although Example 1.1 and Lemma 3.1 analyze the extreme case of exact duplication, the same concern arises for augmented data that are not literal copies but transformations of the originals. When transformations are small, classical DS often assigns augmented instances nearly the same value as their corresponding originals because it does not distinguish informational originality from redundancy. Section 6.1 returns to both settings, namely augmentation and exact duplication, and shows empirically that our method differentiates novelty from replication and yields allocations that better reflect incremental, model-relevant information.

### 3.3 Sequential Training and Temporal Dependence

Sequential training updates a model over rounds $t\in[T]:=\{1,2,\dots,T\}$, with data arriving in batches or from distributed contributors, as in online learning, incremental learning, federated learning, or multi-stage LLM fine-tuning. Let $I$ be the index set of all contributors across rounds, and in round $t$ let $I_{t}\subseteq I$ denote the active contributors who supply the data sources $D_{t}=\{z_{t,i}\}_{i\in I_{t}}$. Within each round, updates are submitted in random order, whereas the sequence of rounds $[T]$ is fixed. Define the prefix $U_{t-1}:=\bigcup_{j=1}^{t-1}D_{j}$ as the sources incorporated before round $t$. The learning algorithm then produces the realized trajectory

$$
\mathcal{A}_{\mathrm{init}},\quad\mathcal{A}_{1}(D_{1}),\quad\mathcal{A}_{2}(D_{2};\,U_{1}),\ \ldots,\ \mathcal{A}_{t}(D_{t};\,U_{t-1}),\ \ldots,\ \mathcal{A}_{T}(D_{T};\,U_{T-1}),
$$

where $\mathcal{A}_{t}(D_{t};\,U_{t-1})$ denotes the model obtained by updating $\mathcal{A}_{t-1}(D_{t-1};U_{t-2})$ using $D_{t}$. Thus, a contributor’s impact depends on both the content of their source and the round in which it is submitted. Utility should be evaluated against the model state that actually prevails at that round along the realized trajectory, rather than over counterfactual training paths.

Classical DS computes a data source’s value by averaging its marginal contribution over all subsets of remaining sources as in Eq.(2). For instance, in federated learning, this would require permuting contributors across rounds and retraining to construct counterfactual model trajectories. Such a procedure is infeasible under the communication and computation constraints of federated learning and, more importantly, it fails to respect the realized temporal order of updates and the sequence of model states that actually occurred. The following example illustrates this.

###### Example 2 (continued).

Consider a federated learning setting with four contributors over two rounds. In round $1$, two contributors submit $D_{1}=\{z_{1,1},z_{1,2}\}$; in round $2$, two new contributors submit $D_{2}=\{z_{2,1},z_{2,2}\}$. The realized trajectory is $\mathcal{A}_{1}(D_{1})$ followed by $\mathcal{A}_{2}(D_{2};\,D_{1})$. To assess the value of the round-2 source $z_{2,1}$ along this realized trajectory, we condition on the actual model state $\mathcal{A}_{1}(D_{1})$ and average its one-step marginal contributions over subsets within round $2$:

- the utility gain from adding $z_{2,1}$ to $D_{1}$ (the full round-1 data), and
- the utility gain from adding $z_{2,1}$ to $D_{1}\cup\{z_{2,2}\}$ (the full round-1 data plus the other round-2 source).

This state-conditioned evaluation respects the sequential order of training by keeping $D_{1}$ fixed to reflect the realized model state at the end of round 1 and varying only the within-round context for $z_{2,1}$ in round 2.

By contrast, DS should average over all possible subsets of $\{z_{1,1},z_{1,2},z_{2,2}\}$. It therefore assigns value to $z_{2,1}$ at counterfactual positions that precede the full round-1 data (e.g., adding $z_{2,1}$ to $\varnothing$, $\{z_{1,1}\}$, or $\{z_{1,2}\}$). Implementing this requires constructing and retraining along hypothetical trajectories that never occurred, which violates the observed temporal order of the workflow and is computationally prohibitive in federated learning settings due to communication and device constraints.

This example highlights the need for a valuation approach that respects the temporal structure of sequential training pipelines. We therefore formalize the notion of a state-conditioned marginal contribution, which evaluates a source $z$ in its round $t$ by measuring its effect relative to the realized model state $\mathcal{A}_{t-1}$ at the start of that round, holding earlier rounds’ trajectory fixed and varying only its within-round context.

###### Definition 3.2 (One-step state-conditioned marginal contribution).

Fix a round $t\in[T]$ with aggregate dataset $D_{t}$ and realized model state $\mathcal{A}_{t-1}$ at its start. For any data source $z\in D_{t}$ and subset $S_{t}\subseteq D_{t}\setminus\{z\}$, the state conditioned marginal contribution of $z$ given $S_{t}$ and $\mathcal{A}_{t-1}$ is

$$
\Delta_{\mathcal{A}_{t-1}}\bigl(z\mid S_{t}\bigr)\;:=\;v\bigl(S_{t}\cup\{z\};\,\mathcal{A}_{t-1}\bigr)\;-\;v\bigl(S_{t};\,\mathcal{A}_{t-1}\bigr),
$$

where $v\bigl(S_{t};\mathcal{A}_{t-1}\bigr)$ and $v\bigl(S_{t}\cup\{z\};\mathcal{A}_{t-1}\bigr)$ denote the utilities obtained by updating the model from state $\mathcal{A}_{t-1}$ using $S_{t}$ and $S_{t}\cup\{z\}$, respectively.

In machine learning applications, $\Delta_{\mathcal{A}_{t-1}}\bigl(z\mid S_{t}\bigr)$ in Eq.(5) measures the change in performance incrementally updating the model state $\mathcal{A}_{t-1}$ on $S_{t}\cup\{z\}$ instead of $S_{t}$ alone. In contrast, the classical marginal contribution in Eq.(1) is obtained as the special case with the initial model state, identifying $v(S;\mathcal{A}_{\mathrm{init}})\equiv v(S)$, so that $\Delta(z\mid S)=\Delta_{\mathcal{A}_{\mathrm{init}}}(z\mid S).$

To maintain practical feasibility and fidelity to the temporal structure of sequential training, we restrict valuation to the realized model trajectory. For any source $z\in D_{t}$, all utility evaluations are anchored to the realized model state $\mathcal{A}_{t-1}$ from previous rounds, so each contribution is measured against the historical context in which it actually occurred. We evaluate the value of $z\in D_{t}$ as

$$
\overline{\Delta}_{t}\!\bigl(z\mid\mathcal{A}_{t-1}\bigr)\;=\;\frac{1}{|D_{t}|}\sum_{S_{t}\subseteq D_{t}\setminus\{z\}}\binom{|D_{t}|-1}{|S_{t}|}^{-1}\,\Delta_{\mathcal{A}_{t-1}}\!\bigl(z\mid S_{t}\bigr),
$$

which mirrors the combinatorial averaging in Eq. (2) and anchors the valuation of $z$ to the realized state $\mathcal{A}_{t-1}$ along the actual sequential trajectory. Utilizing classical DS would compute marginal contributions by training a new model for each subset starting from the initial model $\mathcal{A}_{\mathrm{init}}$ and rebuilding a counterfactual trajectory of earlier rounds before measuring the marginal contribution (see Eq.(2)). In contrast, Eq.(6) evaluates all marginal contributions at the realized model state $\mathcal{A}_{t-1}$: earlier rounds are held fixed exactly as they occurred, and only the within-round position of $z$ varies. This avoids prohibitive communication costs for simulating counterfactual trajectories and preserves the temporal structure of the sequential training.

###### Remark 3.1 (Aggregating value for contributors active across multiple rounds).

A contributor may submit data sources in multiple rounds. We treat each submission $z_{t,i}$ as a distinct source in its round $t$ and evaluate its state–conditioned value using Eq.(6). Let $\mathcal{T}_{i}:=\{\,t\in[T]:i\in I_{t}\,\}$ be the set of rounds in which contributor $i$ is active. The contributor’s total value should be $\Phi_{i}\;=\;\sum_{t\in\mathcal{T}_{i}}\overline{\Delta}_{t}\!\bigl(z_{t,i}\mid\mathcal{A}_{t-1}\bigr).$

###### Lemma 3.2 (Violation of symmetry along the realized sequential trajectory).

Consider a $T$ -round sequential training process with aggregate datasets $D=\{D_{t}\}_{t=1}^{T}$ and realized model states $\{\mathcal{A}_{t}\}_{t=0}^{T}$ as described above. Let $z^{\star}_{k,i}\in D_{k}$ and $z^{\star}_{\ell,i}\in D_{\ell}$ be two identical sources, that is, $\operatorname{\mathrm{Ins}}(z^{\star}_{k,i})=\operatorname{\mathrm{Ins}}(z^{\star}_{\ell,i})$, with $1\leq k<\ell\leq T$. If the utility $v(\,\cdot\,;\mathcal{A})$ depends on the model state $\mathcal{A}$, then their values along the realized trajectory (as in Eq. (6)) generally satisfy $\overline{\Delta}_{k}\!\bigl(z^{\star}_{k,i}\mid\mathcal{A}_{k-1}\bigr)\;\neq\;\overline{\Delta}_{\ell}\!\bigl(z^{\star}_{\ell,i}\mid\mathcal{A}_{\ell-1}\bigr)$ without further assumptions.

Even when two data sources are identical in content, conditioning on the realized training trajectory typically yields different average contributions across rounds because the model states $\mathcal{A}_{k-1}$ and $\mathcal{A}_{\ell-1}$ differ. This contradicts the classical symmetry axiom, which would assign equal value to identical sources irrespective of at which round they participate. Equality arises only when the utility is state insensitive, so that $\Delta_{\mathcal{A}}(z\mid S)$ does not depend on $\mathcal{A}$, a condition that rarely holds in practice since utility is usually a performance metric of the trained machine learning model. These observations motivate relaxing symmetry in sequential training and valuing each data source relative to the model state actually in place at the time it is incorporated.

## 4 Asymmetric Data Shapley

While classical DS is widely used for data valuation, it relies on the symmetry axiom. This assumption overlooks key features of modern machine learning workflows, including directional dependence among data sources and temporal order in training pipelines, and can yield unintuitive valuations (see Section 3). To address these limitations, we adapt the weighted Shapley value from cooperative game theory (nowak1995axiomatizations) to supervised learning and introduce Asymmetric Data Shapley (ADS). ADS computes a data source’s value as its average marginal contribution taken uniformly over permutations that respect an application-specific ordering of the data into groups. In this way, ADS preserves efficiency and linearity, retains symmetry within each group, and reduces to classical DS when there is a single group. The ordered structure can encode rounds in sequential or federated training, precedence between original and synthetic data, or other application-driven constraints. In what follows, we restate the relevant definitions from nowak1995axiomatizations using machine learning terminology and present the axiomatic foundation of our proposed framework.

###### Definition 4.1 (Ordered data groups).

Let $\sigma=(D_{1},\ldots,D_{T})$ be an ordered collection of nonempty groups of data sources, and $D=\bigcup_{t=1}^{T}D_{t}$ be the full training set. Define the group index map $\kappa:D\to[T]$ by $\kappa(z)=t$ if and only if $z\in D_{t}$. The ordered data groups $\sigma$ induces the relations

$$
z\equiv_{\sigma}z^{\prime}\ \Longleftrightarrow\ \kappa(z)=\kappa(z^{\prime}),\qquad z\prec_{\sigma}z^{\prime}\ \Longleftrightarrow\ \kappa(z)<\kappa(z^{\prime}),\qquad z\preceq_{\sigma}z^{\prime}\ \Longleftrightarrow\ \kappa(z)\leq\kappa(z^{\prime}).
$$

Intuitively, $z\equiv_{\sigma}z^{\prime}$ means the two sources belong to the same group; $z\prec_{\sigma}z^{\prime}$ means the group containing $z$ precedes the group containing $z^{\prime}$ under $\sigma$; and $z\preceq_{\sigma}z^{\prime}$ allows either equality (same group) or precedence. In what follows, we restrict attention to permutations of data sources whose group indices are nondecreasing, thereby respecting this precedence.

###### Definition 4.2 (Ordered permutations).

Let $\sigma=(D_{1},\ldots,D_{T})$ be the ordered data groups from Definition 4.1, and let $D=\bigcup_{t=1}^{T}D_{t}$ be the full training set. Let $\Pi(S)$ denote the set of all permutations of a finite collection of sources $S$, and let $\kappa$ be the group index map from Definition 4.1. A permutation $\pi_{\sigma}(D)=(z_{o_{1}},\ldots,z_{o_{n}})\in\Pi(D)$ *respects* $\sigma$ if the group indices are nondecreasing: $\kappa(z_{o_{1}})\;\leq\;\cdots\;\leq\;\kappa(z_{o_{n}}).$ Equivalently, $\pi_{\sigma}(D)$ is obtained by concatenating, in group order, a within–group permutation of each group:

$$
\pi_{\sigma}(D)=(\pi(D_{1}),\ldots,\pi(D_{T})),\qquad\pi(D_{t})\in\Pi(D_{t})\ \text{for all }t\in[T].
$$

The set of all permutations that respect $\sigma$ is

$$
\Pi_{\sigma}(D)\;:=\;\bigl\{\,(\pi(D_{1}),\ldots,\pi(D_{T}))\;:\;\pi(D_{t})\in\Pi(D_{t})\ \text{for all }t\in[T]\,\bigr\},
$$

with cardinality $\,|\Pi_{\sigma}(D)|=\prod_{t=1}^{T}(|D_{t}|!)\,$.

Given Definitions 4.1 and 4.2, an application’s grouping or ordering structure is represented by a proper choice of $\sigma$ on the full training set $D$. For directional dependence between original and augmented sources (Section 3.2), set $\sigma=(D_{\mathrm{orig}},D_{\mathrm{aug}})$ with originals placed before augmentations. For sequential training (Section 3.3), including federated learning and multi-stage LLM fine tuning, let $\sigma=(D_{1},\ldots,D_{T})$ with $D_{t}$ the aggregate dataset in round $t$. Under ADS, valuation averages only over the ordered permutations that respect $\sigma$, that is, elements of $\Pi_{\sigma}(D)$, which incorporates the workflow’s grouping and precedence constraints into the resulting data values.

Before stating ADS formally, we introduce an axiom (Axiom 4.1) that replaces full symmetry with a weaker requirement: sources are interchangeable within groups, while precedence is enforced across groups. This axiom is a specialization of the $\omega$ -Mutual Dependence axiom in nowak1995axiomatizations, which assigns equal value to mutually dependent sources that belong to the same group. Our motivation follows Section 3: in duplication and augmentation, derivative sources depend on originals and should not be treated as independent substitutes; in sequential training, the same source placed in different rounds interacts with different model states, so its value should not be identical across rounds. The axiom encodes these requirements by granting equal value to mutually dependent sources within a group and assigning zero value to any mutually dependent source that must precede its counterpart in the specified group order. This restriction removes spurious credit from permutations that contradict the application-specific order and focuses averaging marginal contributions over the ordered permutations consistent with $\sigma$.

###### Axiom 4.1 (Intra-Group Uniform Mutual Dependence).

Let $\sigma=(D_{1},\ldots,D_{T})$ be an ordered collection of nonempty groups of data sources from Definition 4.1 and $D=\bigcup_{t=1}^{T}D_{t}$ be the full training set. If two sources $z_{i},z_{j}\in D$ are mutually dependent, that is,

$$
\Delta(z_{i}\mid S)=\Delta(z_{j}\mid S)\quad\text{for all }S\subseteq D\setminus\{z_{i},z_{j}\},
$$

then

$$
\phi^{\sigma}(z_{i};v,D)=\begin{cases}\phi^{\sigma}(z_{j};v,D),&\text{if }z_{i}\equiv_{\sigma}z_{j},\\[2.5pt]
0,&\text{if }z_{i}\prec_{\sigma}z_{j},\end{cases}
$$

where $\equiv_{\sigma}$ and $\prec_{\sigma}$ are induced by $\sigma$ as in Definition 4.1.

Axiom 4.1 is a minimal relaxation of symmetry that resolves the issues identified in Section 3. It preserves equal valuation of indistinguishable sources within a group, enforces precedence across groups, and rules out permutations that is not consistent with the specified group order. Together with Efficiency, Linearity, and Nullity, this axiom uniquely determines our ADS valuation rule (Theorem 4.1).

###### Theorem 4.1 (Asymmetric Data Shapley).

Let $\sigma=(D_{1},\ldots,D_{T})$ be an ordered collection of nonempty groups of data sources from Definition 4.1 and $D=\bigcup_{t=1}^{T}D_{t}$ be the full training set. For $z\in D$, the ADS value $\phi^{\sigma}(z;\,v,D)$ is the unique allocation that satisfies Axioms 3.1, 3.2, 3.3, and 4.1. It admits the permutation form

$$
\phi^{\sigma}(z;\,v,D)=\sum_{\pi(D)\in\Pi(D)}p_{\pi_{\sigma}}\bigl[v\bigl(\pi^{<z}(D)\cup\{z\}\bigr)-v\bigl(\pi^{<z}(D)\bigr)\bigr],
$$

with weights

$$
p_{\pi_{\sigma}}=\begin{cases}\displaystyle\frac{1}{\prod_{t=1}^{T}(|D_{t}|!)},&\pi(D)\in\Pi_{\sigma}(D),\\[6.00006pt]
0,&\text{otherwise},\end{cases}
$$

where $\pi^{<z}$ is defined in Eq. (3).

###### Proposition 4.1 (Subset form of ADS).

Let $U_{t-1}:=\bigcup_{j=1}^{t-1}D_{j}$ be the union of all sources in groups preceding $t$. Then the ADS value of $z\in D_{t}$ admits the equivalent subset form

$$
\phi^{\sigma}(z;\,v,D)=\frac{1}{|D_{t}|}\sum_{S_{t}\subseteq D_{t}\setminus\{z\}}\binom{|D_{t}|-1}{|S_{t}|}^{-1}\Bigl[v\bigl(U_{t-1}\cup S_{t}\cup\{z\}\bigr)-v\bigl(U_{t-1}\cup S_{t}\bigr)\Bigr].
$$

###### Remark 4.1 (ADS in Sequential Training Pipelines).

Eq.(9) gives the subset form of ADS. In sequential training pipelines, the contribution of $U_{t-1}$ is typically absorbed into the realized model state at the end of round $t-1$, denoted $\mathcal{A}_{t-1}$. The impact of $z\in D_{t}$ is therefore evaluated at this fixed state through the state-conditioned marginal contribution $\Delta_{\mathcal{A}_{t-1}}$, which respects the observed round order and avoids constructing counterfactual training trajectories. In this case,

$$
\phi^{\sigma}(z;\,v,D)=\frac{1}{|D_{t}|}\sum_{S_{t}\subseteq D_{t}\setminus\{z\}}\binom{|D_{t}|-1}{|S_{t}|}^{-1}\,\Delta_{\mathcal{A}_{t-1}}\!\bigl(z\mid S_{t}\bigr),
$$

Theorem 4.1 shows that ADS averages one-step marginal contributions only over permutations that respect the group order under $\sigma$. Proposition 4.1 gives an equivalent subset form within the each group $D_{t},t\in[T]$. This representation makes clear that ADS values a source relative to the information already incorporated from preceding groups and depends only on within group combinatorics. Remark 4.1 specializes this subset form to sequential training pipelines by absorbing the prefix $U_{t-1}$ into the model state $\mathcal{A}_{t-1}$ at the start of round $t$. The state-conditioned marginal contribution $\Delta_{\mathcal{A}_{t-1}}(z\mid S_{t})$ then measures the effect of adding $z$ while the realized historical trajectory is kept intact. This avoids constructing counterfactual training paths, respects the observed temporal order, and enables feasible computation for applications like federated learning and multi-stage LLM fine-tuning.

Note that the Federated Shapley proposed by wang2020principled coincides with Remark 4.1 at the implementation level. In each communication round, it keeps the realized history fixed, computes Shapley values for the contributors active in that round based on the current global model, and then aggregates these round level scores over time. Conceptually, however, this construction treats each round as a separate symmetric game and does not provide a unified axiomatic characterization of the global valuation problem across all rounds and their precedence relations. In contrast, ADS models the entire process as a single asymmetric game defined on the $\sigma$, which yields global properties such as group efficiency and makes clear that symmetry is preserved within groups while precedence is allowed across groups. This holistic perspective highlights the interconnected contributions of sources across different groups and supports extensions to broader structure-aware valuation scenarios beyond federated learning, including directional dependence between original and synthetic data and temporal dependence in a variety of sequential training pipelines.

###### Proposition 4.2 (Group Efficiency).

Let $U_{t-1}:=\bigcup_{j=1}^{t-1}D_{j}$ be the union of all sources in groups preceding $t$, with $U_{0}:=\varnothing$. Then, for every $t\in[T]$,

$$
\sum_{z\in D_{t}}\phi^{\sigma}(z;\,v,D)\;=\;v(U_{t})-v(U_{t-1}).
$$

Proposition 4.2 states that the total value assigned to a group equals the incremental utility delivered by that group relative to all preceding groups. For $t=1$, this yields $\sum_{z\in D_{1}}\phi^{\sigma}(z;\,v,D)=v(D_{1})-v(\varnothing)$, which coincides with the efficiency (Axiom 3.1) of DS computed on $D_{1}$. ADS thus preserves efficiency at the group level while allowing ordered structure through an application-specific choice of $\sigma$. We next instantiate $\sigma$ for three common valuation tasks.

###### Proposition 4.3 (Data Shapley as a special case).

If $\sigma=(D)$ consists of a single group, then $\Pi_{\sigma}(D)=\Pi(D)$, $p_{\pi_{\sigma}}=1/n!$, and $\phi^{\sigma}$ reduces to the classical DS value. In this case, the subset and permutation forms coincide with Eqs.(2) and (3).

###### Proposition 4.4 (Data augmentation).

Let $D_{\text{orig}}$ be the collection of original sources and let $D_{\text{aug}}$ be the collection of augmented sources produced by transforming or extrapolating from $D_{\text{orig}}$. Let $D:=D_{\text{orig}}\cup D_{\text{aug}}$ be the full training set. For the ordered data groups $\sigma=(D_{\text{orig}},\,D_{\text{aug}})$,

$$
p_{\pi_{\sigma}}\;=\;\begin{cases}\dfrac{1}{(|D_{\text{orig}}|!)\,(|D_{\text{aug}}|!)},&\pi\in R^{\sigma}(D),\\[6.00006pt]
0,&\text{otherwise.}\end{cases}
$$

Under Proposition 4.4, $\phi^{\sigma}$ assigns to data sources in $D_{\text{orig}}$ their standalone value and to data sources in $D_{\text{aug}}$ their incremental value conditional on $D_{\text{orig}}$. This recognizes informational originality and the directional dependence of duplicated or augmented data on the originals, addressing the concern in Section 3.2.

###### Proposition 4.5 (Sequential training).

Consider a $T$ -round sequential training pipeline. In each round $t\in[T]$, each contributor $i\in I_{t}\subseteq I$ contributes a source $z_{t,i}$, and the round- $t$ dataset is $D_{t}:=\{z_{t,i}:i\in I_{t}\}$. Let $D=\bigcup_{t=1}^{T}D_{t}$ be the full training set. The realized model trajectory at the start of each round is

$$
\mathcal{A}_{\mathrm{init}},\ \mathcal{A}_{1}(D_{1}),\ \ldots,\ \mathcal{A}_{t}\!\Big(D_{t};\,\bigcup_{j=1}^{t-1}D_{j}\Big),\ \ldots,\ \mathcal{A}_{T}\!\Big(D_{T};\,\bigcup_{j=1}^{T-1}D_{j}\Big).
$$

For the ordered data groups $\sigma=(D_{1},\ldots,D_{T})$,

$$
p_{\pi_{\sigma}}\;=\;\begin{cases}\displaystyle\frac{1}{\prod_{t=1}^{T}(|D_{t}|!)},&\pi\in R^{\sigma}(D),\\[5.50003pt]
0,&\text{otherwise.}\end{cases}
$$

Under Proposition 4.5, $\phi^{\sigma}$ assigns to each $z\in D_{t}$ its incremental value on top of the realized model state $\mathcal{A}_{t-1}$ (see Eq.(10)). Concretely, the marginal contribution of $z\in D_{t}$ is evaluated at the fixed state $\mathcal{A}_{t-1}$ through $\Delta_{\mathcal{A}_{t-1}}(z\mid S_{t})$, which respects the temporal order of training and ties valuation to the training trajectory that actually occurred, addressing the issues discussed in Section 3.3.

## 5 Efficient Computation of Asymmetric Data Shapley

While Shapley-based data valuation methods are theoretically well founded, their practical use is limited by the cost of exact computation, which requires evaluating marginal contributions across all $n!$ permutations of a training dataset with $n$ sources. To overcome this barrier, we develop two efficient algorithms for estimating and exact computation of ADS. First, a model-agnostic Monte Carlo estimator that yields an unbiased approximation and applies to general machine learning models. Second, a $K$ -nearest neighbors (KNN) surrogate that enables exact computation when the predictive model is from the KNN family.

### 5.1 Monte Carlo Method

ADS is the expected one-step marginal contribution under the uniform distribution on the set of ordered permutations $\Pi_{\sigma}(D)$ (see Eq.(7)). We can sample $\pi_{\sigma}(D)\sim\mathrm{Unif}\!\bigl(\Pi_{\sigma}(D)\bigr)$ by a two-step construction: (i) for each group $D_{t}$, draw an independent within-group permutation of data sources $\pi(D_{t})\sim\mathrm{Unif}\!\bigl(\Pi(D_{t})\bigr)$; (ii) concatenate in group order to obtain $\pi_{\sigma}(D)=(\pi(D_{1}),\ldots,\pi(D_{T}))\in\Pi_{\sigma}(D)$. This mirrors the sharded sampling idea of xia2023equitable, but here the group order is fixed by $\sigma$. When $\sigma$ has a single group, $\Pi_{\sigma}(D)=\Pi(D)$ and the procedure reduces to the standard Monte Carlo Shapley approach (jia2019efficient; ghorbani2019data). Following wu2023variance, we select the number of sampled permutations to achieve a prescribed accuracy guarantee.

###### Definition 5.1 ((ϵ,δ)(\\epsilon,\\delta)-approximation).

Let $D=\{z_{1},\ldots,z_{n}\}$ be the full training dataset and $\bm{\phi}^{\sigma}(v;D):=\bigl(\phi^{\sigma}(z_{1};\,v,D),\ldots,\phi^{\sigma}(z_{n};\,v,D)\bigr)\in\mathbb{R}^{n}$. An estimator $\widehat{\bm{\phi}}^{\sigma}$ is an $(\epsilon,\delta)$ -approximation if

$$
\mathbb{P}\!\left(\bigl\|\widehat{\bm{\phi}}^{\sigma}-\bm{\phi}^{\sigma}\bigr\|_{\infty}\leq\epsilon\right)\ \geq\ 1-\delta.
$$

For bounded utilities, we adapt the result of wu2023variance to the ADS setting and obtain the following sample complexity guarantee. Let $r$ bound the range of one step marginal contributions (for accuracy based utilities, $r=1$). Then it suffices to use $m_{\star}=\left\lceil\frac{r^{2}}{2\epsilon^{2}}\log\!\left(\frac{2n}{\delta}\right)\right\rceil$ independent samples $\pi_{\sigma}(D)\sim\mathrm{Unif}\!\bigl(\Pi_{\sigma}(D)\bigr)$ to achieve an $(\epsilon,\delta)$ approximation uniformly over all $n$ data sources in the $\ell_{\infty}$ norm. Each sampled ordered permutation yields marginal contributions for all $n$ sources in a single left to right pass, so the total evaluation cost is $O(m_{\star}n)=O\!\left(\frac{n}{\epsilon^{2}}\log\!\left(\frac{n}{\delta}\right)\right)$. The corresponding Monte Carlo procedure is summarized in Algorithm 1.

Algorithm 1 Monte Carlo Asymmetric Data Shapley (MC-ADS)

Ordered data groups $\sigma=(D_{1},\ldots,D_{T})$; $D=\bigcup_{t=1}^{T}D_{t}$; contributor index set $I$ with active contributors $I_{t}\subseteq I$ in group $t\in[T]$ and sources $D_{t}=\{z_{t,i}\}_{i\in I_{t}}$; utility $v:2^{D}\to\mathbb{R}$; the fixed holdout test set $D_{\text{test}}$; tolerances $\epsilon>0$, $\delta\in(0,1)$; range bound $r$ on one–step marginal contribution.

Sample size: $m_{\star}\leftarrow\left\lceil\dfrac{r^{2}}{2\epsilon^{2}}\,\log\!\bigl(\dfrac{2|D|}{\delta}\bigr)\right\rceil$.

Initialize: $\widehat{\phi}^{\sigma}(z)\leftarrow 0$ for all $z\in D$.

for $s=1$ to $m_{\star}$ do

  for $t=1$ to $T$ do

   Sample a within–group permutation of sources $\pi^{s}(D_{t})\sim\mathrm{Unif}\!\bigl(\Pi(D_{t})\bigr)$.

  end for

  Concatenate by group order $\pi^{s}(D)\leftarrow(\pi^{s}(D_{1}),\ldots,\pi^{s}(D_{T}))$. Write $\pi^{s}(D)=(z_{o_{1}},\ldots,z_{o_{n}})$.

   $P\leftarrow\varnothing$; $u_{\mathrm{prev}}\leftarrow v(P)$.

  for $j=1$ to $n$ do

    $P\leftarrow P\cup\{z_{o_{j}}\}$; $u_{\mathrm{cur}}\leftarrow v(P)$.

    $\widehat{\phi}^{\sigma}(z_{o_{j}})\leftarrow\dfrac{s-1}{s}\,\widehat{\phi}^{\sigma}(z_{o_{j}})+\dfrac{1}{s}\,\bigl(u_{\mathrm{cur}}-u_{\mathrm{prev}}\bigr)$.

    $u_{\mathrm{prev}}\leftarrow u_{\mathrm{cur}}$.

  end for

end for

Aggregate to contributors: For each $i\in I$, let $\mathcal{T}_{i}:=\{t\in[T]:i\in I_{t}\}$ and set $\widehat{\Phi}_{i}\;\leftarrow\;\sum_{t\in\mathcal{T}_{i}}\widehat{\phi}^{\sigma}(z_{t,i})$.

Output: MC-ADS estimates for each source $\{\widehat{\phi}^{\sigma}(z):z\in D\}$ and for each contributor $\{\widehat{\Phi}_{i}:i\in I\}$.

### 5.2 KNN Surrogate Method

Evaluating utilities on many subsets along sampled permutations, as in Algorithm 1, typically entails repeated retraining and becomes infeasible for large datasets or complex models. To avoid retraining, we extend the KNN surrogate approach to ADS, which enables closed-form, single-pass computation of data values (jia2019efficient; wang2023threshold).

A practical subtlety arises because our valuation units are data sources, each of which is a finite collection of instances owned by a contributor, whereas KNN is defined in terms of distances and majority votes over instances. We address this mismatch by computing ADS on the instance ground set and then aggregating the resulting values back to sources and contributors. For the group of sources $D_{t}$, the corresponding group of instances is $\operatorname{\mathrm{Ins}}(D_{t}):=\bigcup_{z\in D_{t}}\operatorname{\mathrm{Ins}}(z)$. We then evaluate ADS at the instance level using the ordered instance groups $\bigl(\operatorname{\mathrm{Ins}}(D_{1}),\ldots,\operatorname{\mathrm{Ins}}(D_{T})\bigr)$. The ADS value of a source $z$ is defined as the sum of the ADS values of its instances, and a contributor’s value is the sum over all sources owned by that contributor. We next specify a KNN utility and characterize how instance contributions vary with their positions among nearest neighbors. This neighbor relation yields an iterative and exact computation of ADS without retraining (Theorem 5.1); full details appear in Algorithm 2.

###### Definition 5.2 (Utility for the KNN classifier).

Let $\sigma=\bigl(D_{1},\ldots,D_{T}\bigr)$ be the ordered groups of sources and $D=\bigcup_{t=1}^{T}D_{t}$ be the full training set. Let $D_{\text{test}}=\{q_{\text{test},\ell}=(x_{\text{test},\ell},y_{\text{test},\ell})\}_{\ell=1}^{n_{\text{test}}}$ be a test set with categorical labels, where each $q_{\text{test},\ell}$ is a single labeled instance. For any subset of sources $S\subseteq D$, let $\operatorname{\mathrm{Ins}}(S)$ denote the induced collection of training instances and write $m(S):=\lvert\operatorname{\mathrm{Ins}}(S)\rvert$ for its instance count. Let $d(\cdot,\cdot)$ be a fixed distance metric between two instances.

Fix a test instance $q_{\text{test}}=(x_{\text{test}},y_{\text{test}})$. For $i=1,\ldots,m(C_{t})$, let $I_{{d}}^{i}(x_{\text{test}};\operatorname{\mathrm{Ins}}(S))$ denote the index of the $i$ th nearest neighbor of $x_{\text{test}}$ among the instances in $\operatorname{\mathrm{Ins}}(S)$ under $d$, with ties broken by a fixed deterministic rule. Define $K^{\prime}:=\min\{K,\,m(S)\}$. The KNN utility of $S$ at $q_{\text{test}}$ is

$$
v_{\text{knn}}(S)\;:=\;\frac{1}{K^{\prime}}\sum_{i=1}^{K^{\prime}}\mathbf{1}\!\left[y_{I_{{d}}^{i}(x_{\text{test}};\operatorname{\mathrm{Ins}}(S))}=y_{\text{test}}\right],
$$

that is, the fraction of the $K^{\prime}$ nearest neighbors in $\operatorname{\mathrm{Ins}}(S)$ whose labels equal $y_{\text{test}}$.

Note that Eq.(12) is defined for a single test instance. When reporting an aggregate utility, for example average accuracy over $D_{\text{test}}$, we average Eq.(12) over $\ell=1,\ldots,n_{\text{test}}$. All statements below are presented pointwise in $q_{\text{test}}$ and are aggregated over $D_{\text{test}}$ in Algorithm 2.

For notational convenience, fix a group index $t\in[T]$. Let $P_{t}:=\operatorname{\mathrm{Ins}}(U_{t-1})=\bigcup_{j=1}^{t-1}\operatorname{\mathrm{Ins}}(D_{j})$ be the collection of instances drawn from all groups that precede $t$ (so $P_{1}=\operatorname{\mathrm{Ins}}(U_{0})=\varnothing$), and $C_{t}:=\operatorname{\mathrm{Ins}}(D_{t})$ be the collection of instances in the current group $t$. Their numbers of instances are $m(P_{t})$ and $m(C_{t})$, respectively. Order all instances in $P_{t}\cup C_{t}$ by increasing distance to $x_{\text{test}}$ under the metric $d$. For $i=1,\ldots,m(C_{t})$, define

$$
c_{t,i}\;:=\;\bigl|\bigl\{\,q=(x,y)\in P_{t}:\ d(x,x_{\text{test}})\;<\;d\!\bigl(x_{I_{{d}}^{i}(x_{\text{test}};C_{t})},x_{\text{test}}\bigr)\,\bigr\}\bigr|.
$$

Thus $c_{t,i}$ counts how many instances from preceding groups are closer to $x_{\text{test}}$ than the $i$ th nearest instance taken from $C_{t}$. We use a strict inequality to exclude ties; any remaining ties are broken by a fixed deterministic rule, which does not affect the results that follow.

###### Theorem 5.1 (Iterative characterization of ADS under the KNN surrogate).

Under this setup, fix a test instance $q_{\text{test}}=(x_{\text{test}},y_{\text{test}})$ and a group index $t\in[T]$. For $i=1,\ldots,m(C_{t})-1$, let $q_{I_{{d}}^{i}(x_{\text{test}};C_{t})}$ and $q_{I_{{d}}^{\,i+1}(x_{\text{test}};C_{t})}$ denote the $i$ th and $(i+1)$ th nearest neighbors to $x_{\text{test}}$ among the instances in $C_{t}$ under $d$. Then we have:

1. If $K\leq c_{t,i}$, then $\phi^{\sigma}\!\bigl(q_{I_{{d}}^{i}(x_{\text{test}};C_{t})};v_{knn},\operatorname{\mathrm{Ins}}(D)\bigr)-\phi^{\sigma}\!\bigl(q_{I_{{d}}^{\,i+1}(x_{\text{test}};C_{t})};v_{knn},\operatorname{\mathrm{Ins}}(D)\bigr)=0$.
2. If $K>c_{t,i+1}=c_{t,i}=c_{t}$, then
	$$
	\displaystyle\phi^{\sigma}\!\bigl(q_{I_{{d}}^{i}(x_{\text{test}};C_{t})};v_{knn},\operatorname{\mathrm{Ins}}(D)\bigr)-\phi^{\sigma}\!\bigl(q_{I_{{d}}^{\,i+1}(x_{\text{test}};C_{t})};v_{knn},\operatorname{\mathrm{Ins}}(D)\bigr)
	$$
	 
	$$
	\displaystyle=\frac{\mathbf{1}[y_{I_{{d}}^{i}(x_{\text{test}};C_{t})}=y_{\text{test}}]-\mathbf{1}[y_{I_{{d}}^{\,i+1}(x_{\text{test}};C_{t})}=y_{\text{test}}]}{K}\cdot\frac{\min\{K-c_{t},\,i\}}{i}.
	$$
3. If $K>c_{t,i+1}>c_{t,i}$, then
	$$
	\displaystyle\phi^{\sigma}\!\bigl(q_{I_{{d}}^{i}(x_{\text{test}};C_{t})};v_{knn},\operatorname{\mathrm{Ins}}(D)\bigr)-\phi^{\sigma}\!\bigl(q_{I_{{d}}^{\,i+1}(x_{\text{test}};C_{t})};v_{knn},\operatorname{\mathrm{Ins}}(D)\bigr)
	$$
	 
	$$
	\displaystyle\qquad=\;\frac{\mathbf{1}[y_{I_{{d}}^{i}(x_{\text{test}};C_{t})}=y_{\text{test}}]-\mathbf{1}[y_{I_{{d}}^{\,i+1}(x_{\text{test}};C_{t})}=y_{\text{test}}]}{K}\cdot\frac{\min\{K-c_{t,i+1},\,i\}}{i}
	$$
	 
	$$
	\displaystyle+\frac{1}{m(C_{t})-1}\sum_{s=K-c_{t,i+1}}^{m(C_{t})-2}\sum_{u=K-c_{t,i+1}}^{\min\{K-c_{t,i}-1,\,s\}}\frac{\binom{s}{u}\binom{m(C_{t})-2-s}{i-1-u}}{\binom{m(C_{t})-2}{i-1}}\,\frac{\mathbf{1}[y_{I_{{d}}^{i}(x_{\text{test}};C_{t})}=y_{\text{test}}]-\mathbf{1}[y_{I_{{d}}^{K-u}(x_{\text{test}};P_{t})}=y_{\text{test}}]}{K}.
	$$
4. If $c_{t,i}<K\leq c_{t,i+1}$, then
	$$
	\displaystyle\phi^{\sigma}(q_{I^{d}_{i}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))-\phi^{\sigma}(q_{I^{d}_{i+1}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))
	$$
	 
	$$
	\displaystyle\qquad=\frac{1}{m(C_{t})-1}\sum_{s=0}^{m(C_{t})-2}\sum_{u=0}^{\min\{K-c_{t,i}-1,s\}}\frac{\binom{s}{u}\binom{m(C_{t})-2-s}{i-1-u}}{\binom{m(C_{t})-2}{i-1}}\,\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{\,I^{d}_{\,K-u}(x_{\text{test}};P_{t})}=y_{\text{test}}]}{K},
	$$

To complete the algorithm, we must also specify the base value that initializes the recursion, namely the ADS of the farthest instance from $x_{\text{test}}$ within $C_{t}$, $\phi^{\sigma}\bigl(q_{I_{{d}}^{m(C_{t})}(x_{\text{test}};C_{t})};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D)\bigr)$, whose closed-form expression is derived in Appendix B.6. Theorem 5.1 yields an exact, single-pass recurrence over instances within each ordered group $C_{t}=\operatorname{\mathrm{Ins}}(D_{t})$. As in symmetric KNN Shapley (jia2019efficient), computing a single global ranking of the $n$ training instances by distance to a fixed test point costs $O(n\log n)$. Given this order, KNN-ADS evaluates the value of each instance by a linear scan within each group, so the computation in group $t$ is $O\!\big(m(C_{t})\big)$. Summing over groups gives $\sum_{t}O\!\big(m(C_{t})\big)=O(n)$. As a result, the distance sort is the bottleneck, so the overall time per test instance is $O(n\log n)$ with $n$ training instances.

Algorithm 2 K-Nearest Neighbor Asymmetric Data Shapley (KNN-ADS)

Ordered data groups $\sigma=(D_{1},\ldots,D_{T})$; $D=\bigcup_{t=1}^{T}D_{t}$; contributor index set $I$ with active contributors $I_{t}\subseteq I$ in group $t\in[T]$ and sources $D_{t}=\{z_{t,i}\}_{i\in I_{t}}$; KNN utility $v_{knn}$; test set $D_{\text{test}}=\{(x_{\text{test},\ell},y_{\text{test},\ell})\}_{\ell=1}^{n_{\text{test}}}$; neighbors $K$; distance metric $d$.

Initialize: $\widehat{\phi}^{\sigma}(q)\leftarrow 0$ for all $q\in\operatorname{\mathrm{Ins}}(D)$;    $\widehat{\phi}^{\sigma}(z)\leftarrow 0$ for all $z\in D$;    $\widehat{\Phi}_{i}\leftarrow 0$ for all $i\in I$.

for $\ell=1$ to $n_{\text{test}}$ do

  for $t=1$ to $T$ do

     $P_{t}\leftarrow\bigcup_{j=1}^{t-1}\operatorname{\mathrm{Ins}}(D_{j})$,    $C_{t}\leftarrow\operatorname{\mathrm{Ins}}(D_{t})$.

    Rank $P_{t}\cup C_{t}$ by distance to $x_{\text{test},\ell}$ under $d$, and compute $c_{t,i}$ for $i=1,\ldots,m(C_{t})$.

    Compute the base value $\phi^{\sigma}\bigl(q_{I_{{d}}^{m(C_{t})}(x_{\text{test},l};C_{t})};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D)\bigr)$ using Appendix B.6.

    for $i=m(C_{t})-1$ down to $1$ do

     Update $\phi^{\sigma}\!\bigl(q_{I_{{d}}^{i}(x_{\text{test},l};C_{t})};v_{knn},\operatorname{\mathrm{Ins}}(D)\bigr)$ from $\phi^{\sigma}\!\bigl(q_{I_{{d}}^{\,i+1}(x_{\text{test},l};C_{t})};v_{knn},\operatorname{\mathrm{Ins}}(D)\bigr)$ using Theorem 5.1.

    end for

    Aggregate over the test set: for all $q\in C_{t}$, set $\widehat{\phi}^{\sigma}(q)\leftarrow\widehat{\phi}^{\sigma}(q)+n_{\text{test}}^{-1}\,\phi^{\sigma}(q;v_{knn},\operatorname{\mathrm{Ins}}(D))$.

  end for

end for

Aggregate to sources: for each $z\in D$, set $\widehat{\phi}^{\sigma}(z)\leftarrow\sum_{q\in\operatorname{\mathrm{Ins}}(z)}\widehat{\phi}^{\sigma}(q)$.

Aggregate to contributors: for each $i\in I$, set $\mathcal{T}_{i}:=\{\,t\in[T]:i\in I_{t}\,\}$ and $\widehat{\Phi}_{i}\leftarrow\sum_{i\in\mathcal{T}_{i}}\widehat{\phi}^{\sigma}(z_{t,i})$.

Output: $\{\widehat{\phi}^{\sigma}(q):q\in\operatorname{\mathrm{Ins}}(D)\}$,   $\{\widehat{\phi}^{\sigma}(z):z\in D\}$,   and $\{\widehat{\Phi}_{i}:i\in I\}$.

## 6 Experiments and Applications

This section applies ADS in three representative machine learning and data market workflows: (1) quantifying each synthetic source’s incremental contribution beyond the original training sources, (2) assessing participant contributions in federated learning, and (3) guiding the optimal procurement of datasets for multi-stage LLM fine-tuning. Throughout, we choose the ordered groups $\sigma$ to align with the structure of each task. For synthetic data, we set $\sigma=(D_{\text{orig}},D_{\text{aug}})$ (Proposition 4.4). For sequential training pipelines, including federated learning and multi-stage LLM fine-tuning, we set $\sigma=(D_{1},\ldots,D_{T})$, where each $D_{t}$ is the aggregated dataset in round $t$ (Proposition 4.5). In the first two settings, we consider a large number of contributors and therefore use the Monte Carlo estimator MC-ADS (Algorithm 1) for unbiased approximation together with the KNN surrogate KNN-ADS (Algorithm 2) for exact evaluation in KNN models in order to reduce computational cost. In the LLM application, we work with a small number of contributors and rely on the definition of ADS in Remark 4.1 to compute values exactly along the realized fine-tuning trajectory.

### 6.1 Synthetic Data Valuation

We evaluate ADS on three benchmark datasets that capture complementary augmentation regimes: Adult (misc\_adult\_2), MNIST (lecun1998gradient), and Omniglot (lake2015human). On Adult, we apply Borderline-SMOTE (han2005borderline) to oversample the minority class and reduce imbalance. Using MNIST, we generate variants with slight rotations and translations to boost generalization. On Omniglot, we synthesize additional images with a generative adversarial network (antoniou2017data) to expand training in low-resource settings. In all cases, the final machine learning model is trained on the union of original and augmented data. For simplicity, we treat each data source as a single instance and compute values for every original and augmented instance using the following methods. For Adult and MNIST, we train a 5-nearest neighbor classifier and report valuations from MC-ADS, MC-DS (ghorbani2019data), KNN-ADS, KNN-DS (jia2019efficient), and leave one out (LOO). For Omniglot, we train a logistic regression model and report MC-ADS, MC-DS, and LOO.

#### 6.1.1 Sanity checks via add/remove interventions.

We first test whether ADS distinguishes informative synthetic data instances from highly redundant ones. For each dataset, we run two complementary interventions. In the removal experiment, we first rank all augmented instances using each valuation method. We then remove a fraction of augmented instances $(0\%\ \text{to}\ 30\%)$ from the full training set, either the lowest valued subset or the highest valued subset, and retrain the model before evaluating on the held out test set. In the addition experiment, we start from the original dataset and add a fraction of instances $(0\%\ \text{to}\ 30\%)$ drawn from the augmented pool according to each ranking, again either the lowest ranked subset or the highest ranked subset, and then retrain the model and evaluate on the held out test set. We report relative accuracy, defined as test accuracy normalized by the baseline model trained before any intervention. All results are averaged over 10 random seeds with $95\%$ confidence intervals; see Figure 2.

Across all datasets and both interventions, ADS most closely tracks the true incremental contribution of synthetic instances beyond the original dataset. Removing the lowest-ranked augmented instances from the full training dataset under ADS yields the largest gains in relative accuracy compared with DS, LOO, or random removal (panel (a)), indicating that ADS more effectively filters harmful or redundant augmentations. Conversely, under ADS, removing the highest ranked augmented instances from the full training set yields the largest accuracy drop (panel (b)), indicating that ADS identifies the most informative augmentations whose absence most degrades performance. Additional experiments mirror these patterns: adding the lowest-ranked augmented instances to the original training set under ADS reduces accuracy the most (panel (c)), while adding the highest-ranked ones yields the largest accuracy gains (panel (d)). These effects are strongest on Adult and MNIST for both MC-ADS and KNN-ADS, and on Omniglot for MC-ADS, whereas DS methods (MC-DS, KNN-DS) and LOO exhibit weaker discrimination. Overall, these interventions show that ADS reliably identifies both harmful and helpful synthetic instances and assigns values consistent with their incremental contributions beyond the original dataset.

![Refer to caption](https://arxiv.org/html/2511.12863v1/x1.png)

Figure 2: Relative accuracy is test accuracy normalized by the baseline model before any intervention. (a) and (b) remove low-value and high-value augmented points, respectively. (c) and (d) add low-value and high-value augmented points to the original set, respectively. We compare ADS ( MC-ADS and KNN-ADS where applicable) with symmetric baselines ( MC-DS ), LOO, and random selection. Results are averaged over 10 seeds with 95% confidence intervals. produces the strongest positive and negative shifts in the expected directions, indicating better discrimination between informative and redundant augmentations.

#### 6.1.2 Implications for data marketplaces.

In market settings, synthetic data often arises when multiple contributors contribute original data sources and a third party broker aggregates, refines, or augments these data before selling a trained model to buyers (Figure 1). Under Proposition 4.4, ADS provides a fair allocation rule for splitting value between contributors and the broker that respects the primacy of original sources and rewards augmentation that contributes genuine informational novelty. We illustrate these implications using two simulated data market scenarios on MNIST, with a 5-nearest neighbor classifier as the downstream model and KNN-ADS for exact value computation.

- Replication scenario. The broker duplicates the contributors’ data and includes the replicas in training. We consider three configurations: Original (one copy of each instance), Copied once (two copies), and Copied twice (three copies). Under DS, replication reduces the contributors’ share and reallocates value to the broker as more copies are added. Under ADS, contributors retain the full value of their original data, and the broker is credited only with the incremental change from adding duplicates, which is small but positive for a KNN classifier. Panel (a) of Figure 3 compares total values under DS and ADS: DS exhibits a growing transfer to the broker from Original to Copied once and Copied twice, whereas ADS keeps the contributors’ total essentially unchanged and allocates only small increments to each additional copy.
- Augmentation scenario. The broker generates synthetic images using small rotations and translations. Each synthetic instance is valued by DS or ADS; we retain only those with positive value, since a positive value indicates an expected improvement in performance, whereas a zero or negative value indicates no expected change or a degradation. The final model is trained on the union of the original data and the retained augmented data. Under ADS, contributors retain the same total value as before augmentation, and the broker receives credit only for the additional beneficial information that the retained synthetic data contribute to the model. Under DS, part of the contributors’ value is reassigned to the broker after augmentation. This suggests that some augmentations primarily reexpress information already present in the original data, and DS thus attributes that portion of informational value to the broker, raising copyright concerns. Panel (b) of Figure 3 illustrates this contrast: DS shifts value away from contributors, whereas ADS preserves contributors’ totals and credits the broker only for the incremental gains from informative synthetic data.

![Refer to caption](https://arxiv.org/html/2511.12863v1/x2.png)

Figure 3: Fair allocation on MNIST under two broker strategies, shown as side by side ( DS vs. ADS ) stacked totals within each configuration. (a) Replication: As identical copies are added ( Original → \\rightarrow Copied once Copied twice ), progressively shifts value from contributors to the broker, while keeps the contributors’ total essentially unchanged and assigns only small incremental gains to each copy. (b) Augmentation: After retaining only positively valued augmentations, still reallocates value away from contributors, while preserves the contributors’ total and credits the broker exactly for the incremental gains contributed by informative synthetic data.

These market level outcomes follow directly from the group efficiency property of ADS (Proposition 4.2): the total value assigned to any group equals its incremental contribution beyond all preceding groups. In the marketplace setting, this implies that contributors (the prior group) retain the entire value created by the original dataset, while the broker (a later group) is compensated only for the incremental gains contributed by informative synthetic data. As a result, redundant replication yields little to no payment, whereas informative augmentation is rewarded in proportion to its incremental gain. By encoding this dependency structure, ADS under Proposition 4.4 addresses the concern in Section 3.2: it protects the primacy of the original data while aligning incentives for high-quality, novel synthetic data.

### 6.2 Participant Valuation in Federated Learning

We simulate a federated learning environment on MNIST (lecun1998gradient) with $30$ contributors. We subsample $3{,}600$ images ($360$ per class) from the MNIST training split and randomly allocate $120$ images to each contributor. Training proceeds for five communication rounds, with six contributors sampled in each round. To stress-test robustness to low quality updates, we designate $50\%$ of contributors as noisy and flip each label in their local datasets with probability $0.7$. The global model is a two layer multilayer perceptron with ReLU activations, trained using FedAvg (mcmahan2017communication). For utility, we report classification accuracy on a held out test set of 10,000 images from the MNIST validation split.

We conduct two interventions to evaluate ADS in identifying valuable and noisy contributors. First, at the start of each round, we compute a value for every contributor, select the top three or four to submit updates, and record the test accuracy at the end of the round. We compare MC-ADS with LOO and a random selection baseline. We omit DS methods because, as discussed in Section 3.3, evaluating the required counterfactual training trajectories is infeasible in this federated setting due to communication and computation overhead. Second, we assess noise detection by ranking contributors according to their values, with lower scores indicating poorer contributors. The detection metric is the cumulative share of noisy contributors contained within the lowest ranked portion as the inclusion threshold increases from the bottom of the ranking. All results are averaged over 100 runs, and we report 95% confidence intervals.

Figure 4 summarizes the findings. Panels (a) and (b) show that when selecting the top $3$ or top $4$ contributors per round, MC-ADS achieves higher accuracy and faster improvement than LOO or random selection. Panel (c) shows that MC-ADS attains the highest cumulative detection rate of noisy contributors among the lowest ranked contributors, demonstrating a stronger ability to consign low value or corrupted contributors to the tail of the distribution. These gains arise because ADS evaluates each contributor conditional on the realized global model state in its round (see Section 3.3), which respects the temporal structure of federated learning and more effectively separates informative updates from detrimental noise than LOO or random selection.

![Refer to caption](https://arxiv.org/html/2511.12863v1/x3.png)

Figure 4: Federated learning with noisy contributors. (a) and (b): test accuracy when the top 3 or top 4 contributors per round are selected using different valuation methods. (c): cumulative detection rate of noisy contributors as we sweep through the worst ranked contributors. Results are averaged over 100 runs with 95 % 95\\% confidence intervals. MC-ADS consistently yields faster accuracy gains and superior noise detection relative to LOO and random selection.

### 6.3 Dataset Procurement in Multi-Stage LLM Fine-Tuning

We simulate a multi-stage LLM fine-tuning process over 30 rounds to predict product ratings in a market research setting. In each round, three contributors each provide a dataset of 50 labeled instances, and a budget constraint allows the firm to purchase data from only one contributor. To optimize spending, the firm selects the highest-value contributor each round according to the chosen valuation rule. This setup reflects a realistic workflow in which a company engages multiple contributors to collect consumer preference data for staged LLM fine-tuning. The resulting valuations guide resource allocation by prioritizing high-quality datasets for training and rewarding their contributors, while also providing actionable feedback to lower-performing contributors to improve subsequent submissions.

To generate the simulated dataset, we define each product profile by three categorical attributes—color (red, blue, green, black), size (small, medium, large, extra-large), and material (cotton, wool, polyester, leather)—each comprising four discrete levels. For each attribute $i\in\mathcal{I}$ ($\mathcal{I}=\{1,2,3\}$) and level $j\in\mathcal{J}^{i}$ ($\mathcal{J}^{i}=\{1,2,3,4\}$), the part-worth utility $u_{ij}$ is independently sampled from a uniform distribution. The true rating $y$ for a product profile $x$, defined by the combination of attribute levels $x=(a_{1},a_{2},a_{3})^{\top}$, $a_{i}\in\mathcal{J}^{i}$, is generated as $y=\sum_{i=1}^{|\mathcal{I}|}u_{i,a_{i}}+\epsilon,$ where $u_{i,a_{i}}$ denotes the part-worth utility associated with level $a_{i}$ of attribute $i$, and $\epsilon$ represents Gaussian noise capturing unobserved heterogeneity in consumer preferences. We refer to datasets generated using these true utilities (i.e., $u_{ij}$) as the true dataset. To simulate varying data quality, we construct noisy dataset with the same attribute structure, but ratings generated as $y^{\prime}=\sum_{i=1}^{|\mathcal{I}|}u^{\prime}_{i,a_{i}}+\epsilon^{\prime}$, where $u^{\prime}_{i,a_{i}}$ is sampled from uniform distribution with $u^{\prime}_{i,a_{i}}\neq u_{i,a_{i}}$ and $\epsilon^{\prime}$ is Gaussian noise (see Appendix C.3 for details). At the start of each round, three data contributors submit datasets with varying quality: 30%, 60%, and 90% of their data are from noisy datasets. Our objective is to select the best candidate dataset at each round to fine-tune the LLM. We compare our proposed ADS, against LOO and a random selection baseline.

![Refer to caption](https://arxiv.org/html/2511.12863v1/x4.png)

(a) Llama 3.2-3B

![Refer to caption](https://arxiv.org/html/2511.12863v1/x9.png)

(a) LOO for Llama 3.1-8B

The LLM generates predicted ratings, from which we estimate the part-worth utilities $\hat{u}_{ij}$ using a least-squares estimate. The performance of the fine-tuned LLM in approximating the underlying consumer preference model is evaluated using the average estimation error (AvgErr), defined as $AvgErr=\frac{1}{n}\sum_{i\in\mathcal{I}}\sum_{j\in\mathcal{J}^{i}}|\hat{u}_{ij}-u_{ij}|,$ where $u_{ij}$ and $\hat{u}_{ij}$ denote the true and estimated part-worth utilities, respectively, and $n=\sum_{i\in\mathcal{I}}|\mathcal{J}^{i}|=12$ represents the total number of attribute levels. The AvgErr thus measures the average absolute deviation between the true and estimated utilities across all attribute levels. In each round, we treat each contributor’s dataset as an indivisible unit. For ADS, we compute each dataset’s average marginal contribution to model performance across all possible permutations of dataset inclusion orders (according to Proposition 4.5). Model performance is measured by AvgErr on a held-out validation set of 320 product profiles. The dataset yielding the highest average performance gain is then selected for that round’s fine-tuning. For the LOO method, we evaluate the value of each dataset as the difference in AvgErr on the validation set when the dataset is included versus excluded, without considering the inclusion order. The random-selection baseline simply chooses one contributor’s dataset uniformly at random for each round of training. After each round’s training with the selected dataset, we record the estimated part-worth utilities based on the LLM’s predicted ratings for a held-out test set of 320 product profiles. We conduct experiments on four representative LLMs (i.e., LLaMA 3.2-3B, LLaMA 3.1-8B, Qwen 3-0.6B, and Qwen 3-8B) and report average performance over 15 independent runs in Figure 5. Experimental details and hyperparameter configurations can be found in Appendix C.3.

As shown in Figure 5, ADS consistently achieves lower AvgErr and faster convergence than baseline methods across all four LLM base models, demonstrating its superior capability to identify the most beneficial datasets for multi-stage fine-tuning. Furthermore, as depicted in Figure 6, ADS exhibits a lower probability of selecting datasets with high proportions of noisy data points, indicating its effectiveness in distinguishing high-quality data contributors within the candidate pool. In contrast, the LOO method tends to select noisier datasets, resulting in suboptimal fine-tuning outcomes. Additional data selection comparisons for other models are provided in Figure 8 in Appendix C.3. Collectively, these results highlight the robustness of ADS in identifying the most valuable datasets for multi-stage LLM fine-tuning and its potential to enable fair compensation mechanisms in data marketplaces.

## 7 Concluding Remarks

We introduced ADS, a novel valuation framework that addresses the pressing challenges of real-world data marketplaces and modern ML/AI systems. More specifically, ADS extends classical DS by relaxing the symmetry axiom, making the valuation structure-aware so that temporal and directional dependencies among data sources are explicitly reflected in their values. ADS assigns each source a value equal to its average one-step marginal contribution computed only over permutations that respect an application-specific ordering of data source groups, ensuring credit reflects the context in which the data were actually used. This yields a tractable, context-aware valuation rule where traditional methods fall short.

On the algorithmic side, we provided two complementary procedures: a Monte Carlo estimator (MC-ADS) with probabilistic accuracy guarantees and a KNN surrogate (KNN-ADS) that yields exact values for nearest-neighbor predictors at practical cost per test point. Together, these estimators make ADS tractable in practice for valuation tasks involving complex models and large-scale datasets. We provide extensive empirical evidence that ADS more faithfully captures each source’s contribution in settings where directional or temporal dependence during training is consequential: (i) distinguishing helpful from redundant or harmful synthetic data in augmentation pipelines, (ii) identifying valuable and noisy data contributors in federated learning, and (iii) guiding data acquisition for multi-stage LLM fine-tuning. Across all these tasks, ADS consistently elevates informative sources and assigns near-zero or negative value to highly redundant, mislabeled, or otherwise low-quality data, yielding a more faithful mapping from task utility to contributor-level compensation than baseline methods.

Beyond methodological innovation, ADS offers a principled link between procurement and compensation in platform-mediated model-as-a-service data markets. It recognizes data as a nonrival, combinatorial good and internalizes replication effects by granting duplicated or synthetic data credit only for incremental utility beyond the original sources. This safeguards the value and provenance of human-authored sources, sharpens the identification of both informative and highly redundant synthetic inputs, and enables transparent, utility-linked payouts. In the era of generative AI where human-created data and model-generated outputs are often combined in training, ADS encourages meaningful augmentation and supports a sustainable marketplace. Together, these properties promote unbiased data acquisition and equitable compensation, closing the incentive loop required for sustainable data exchange.

Several directions merit future work. First, learn or adapt the grouping and precedence structure from data, and rigorously assess robustness to misspecification. Second, develop exact or near-exact surrogates beyond KNN and extend ADS to additional learning paradigms, including self-supervised, contrastive, and pretraining. Third, couple ADS with market mechanisms, such as procurement auctions and revenue-sharing contracts, to operationalize fair and transparent data markets. We hope this structure-aware perspective on data valuation advances principled contributor compensation and sustainable data exchange by aligning both with the realities of contemporary machine learning and AI workflows.

## Appendix A Notation and Symbols

| Symbol | Description |
| --- | --- |
| $\sigma=(D_{1},\ldots,D_{T})$ | Ordered groups of data sources that encode the application specific precedence structure. |
| $D=\bigcup_{t=1}^{T}D_{t}$ | Full set of training data sources, formed by the union of all groups in $\sigma$. |
| $z\in D_{t}$ | A data source (a finite collection of instances) in group $t$. |
| $\operatorname{\mathrm{Ins}}(z)$ | Collection of training instances contained in source $z$. |
| $\operatorname{\mathrm{Ins}}(S)$ | Collection of training instances contained in a collection of sources $S\subseteq D$, that is $\operatorname{\mathrm{Ins}}(S)=\bigcup_{z\in S}\operatorname{\mathrm{Ins}}(z)$. |
| $\lvert S\rvert$ | Number of data sources in $S$ (duplicates allowed). |
| $m(S):=\lvert\operatorname{\mathrm{Ins}}(S)\rvert$ | Number of instances in $S$ (duplicates allowed). |
| $\mathcal{A}$ | Learning algorithm that maps a training dataset to a trained model. |
| $v(S)\equiv v(S;\mathcal{A}_{\text{init}})$ | Utility of sources $S$ when the model state is fixed at the initial state $\mathcal{A}_{\text{init}}$. |
| $v(S;\mathcal{A})$ | Utility of sources $S$ when the model state is fixed at an arbitrary state $\mathcal{A}$. |
| $\Delta(z\mid S)\equiv\Delta_{\mathcal{A}_{\text{init}}}(z\mid S)$ | One-step marginal contribution of source $z$ to $S$ at the initial model state. |
| $\Delta_{\mathcal{A}}(z\mid S)$ | One-step marginal contribution of source $z$ to $S$ when the model state is $\mathcal{A}$. |
| $\phi(z;v,D)$ | Classical Data Shapley (DS) value of a source $z\in D$ under utility $v$. |
| $\phi^{\sigma}(z;v,D)$ | Asymmetric Data Shapley (ADS) value of a source $z\in D$ under utility $v$ and ordered groups $\sigma$. |
| $U_{t-1}=\bigcup_{j=1}^{t-1}D_{j}$ | Union of all sources in groups that precede group $t$ (with $U_{0}=\varnothing$). |
| $\Pi(D)$ | Set of all $\lvert D\rvert!$ permutations of the sources in $D$. |
| $\Pi_{\sigma}(D)$ | Set of all $\prod_{t=1}^{T}\bigl(\lvert D_{t}\rvert!\bigr)$ permutations of $D$ that respect the group order in $\sigma$. |
| $\pi=(z_{o_{1}},\ldots,z_{o_{n}})$ | A permutation of the $n=\lvert D\rvert$ data sources in $D$. |
| $\pi^{<z}$ | Set of all predecessors of source $z$ in the permutation $\pi$. |
| $d(\cdot,\cdot)$ | Distance metric on the feature space between two instances. |
| $I_{{i}}^{x_{\text{test}};\operatorname{\mathrm{Ins}}(S)}(x_{\text{test}};$) | Index of the $i$ th nearest neighbor of $x_{\text{test}}$ among the instances in $\operatorname{\mathrm{Ins}}(S)$ under the metric $d$, with deterministic tie breaking. |
| $P_{t}:=\operatorname{\mathrm{Ins}}(U_{t-1})$ | For a fixed group index $t$, collection of instances drawn from all groups that precede $t$ (so $P_{1}=\operatorname{\mathrm{Ins}}(U_{0})=\varnothing$). |
| $C_{t}:=\operatorname{\mathrm{Ins}}(D_{t})$ | For a fixed group index $t$, collection of instances belonging to group $t$. |
| $c_{t,i}$ | For a fixed group index $t$, number of instances in $P_{t}$ that are closer to $x_{\text{test}}$ than the $i$ th nearest neighbor of $x_{\text{test}}$ within $C_{t}$. |
| $K$ | Number of neighbors used in the $K$ nearest neighbor classifier. |
| $\mathbf{1}[\cdot]$ | Indicator function, equal to $1$ if the condition holds and $0$ otherwise. |
| $r,\ \epsilon,\ \delta$ | Range bound $r$ for one-step marginal contributions and accuracy and confidence tolerances $\epsilon,\delta$ in an $(\epsilon,\delta)$ approximation for the MC-ADS algorithm. |
| $m_{\star}$ | Monte Carlo sample size required to achieve an $(\epsilon,\delta)$ approximation uniformly over all sources in $D$. |

Table 1: Notations used in the paper.

## Appendix B Proofs

### B.1 Proof of Lemma 3.1

Fix $i\in[n]$ and consider any $S\subseteq D^{\mathrm{dup}}\setminus\{z_{1,i},z_{2,i}\}$, $\operatorname{\mathrm{Ins}}(z_{2,i})=\operatorname{\mathrm{Ins}}(z_{1,i})$, thus adding either source to $S$ induces the same collection of training instances:

$$
\forall\,S\subseteq D^{\mathrm{dup}}\setminus\{z_{1,i},z_{2,i}\}:\;\operatorname{\mathrm{Ins}}\!\bigl(S\cup\{z_{1,i}\}\bigr)=\operatorname{\mathrm{Ins}}\!\bigl(S\cup\{z_{2,i}\}\bigr)\;\Longrightarrow\;v\!\bigl(S\cup\{z_{1,i}\}\bigr)=v\!\bigl(S\cup\{z_{2,i}\}\bigr).
$$

Equivalently, the one–step marginal contribution match for all such $S$,

$$
\Delta(z_{1,i}\mid S):=v(S\cup\{z_{1,i}\})-v(S)=v(S\cup\{z_{2,i}\})-v(S)=:\Delta(z_{2,i}\mid S),
$$

and the symmetry axiom implies

$$
\phi\!\big(z_{1,i};\,v,D^{\mathrm{dup}}\big)=\phi\!\big(z_{2,i};\,v,D^{\mathrm{dup}}\big).
$$

Next, by efficiency,

$$
\sum_{z\in D^{\mathrm{dup}}}\phi\!\big(z;\,v,D^{\mathrm{dup}}\big)=v(D^{\mathrm{dup}})-v(\varnothing).
$$

Duplicating every source multiplies instance counts in $\operatorname{\mathrm{Ins}}(D_{1})$ by the same positive constant and therefore leaves the ERM minimizer set unchanged:

$$
\arg\min_{\mathcal{A}\in\mathcal{H}}R_{\mathcal{A}}\!\big(\operatorname{\mathrm{Ins}}(D^{\mathrm{dup}})\big)=\arg\min_{\mathcal{A}\in\mathcal{H}}R_{\mathcal{A}}\!\big(\operatorname{\mathrm{Ins}}(D_{1})\big).
$$

Consequently, the utility of the ERM minimizers are identical, i.e., $v(D^{\mathrm{dup}})=v(D_{1})$. Summing over $i=1,\ldots,n$ yields

$$
\sum_{z\in D_{1}}\phi\!\big(z;\,v,D^{\mathrm{dup}}\big)=\sum_{z\in D_{2}}\phi\!\big(z;\,v,D^{\mathrm{dup}}\big)=\tfrac{1}{2}\sum_{z\in D^{\mathrm{dup}}}\phi\!\big(z;\,v,D^{\mathrm{dup}}\big)=\tfrac{1}{2}\bigl(v(D_{1})-v(\varnothing)\bigr),
$$

$\square$

### B.2 Proof of Lemma 3.2 (constructive counterexample)

We construct a realized sequential training trajectory with a fixed number of rounds in which two identical sources receive different state-conditioned values. For simplicity, we treat each source $z$ as a single labeled instance; the argument extends verbatim to sources containing multiple instances.

##### Prediction rule and utility.

Labels take values in $\{-1,+1\}$. For a model state $\mathcal{A}$, let $Y(\mathcal{A})\subseteq\{-1,+1\}$ denote the collection of labels already incorporated into $\mathcal{A}$. For any finite set of sources $S$, let $Y(S)$ be the corresponding collection of their labels. Define the additive vote total

$$
T(\mathcal{A},S)=\sum_{y\in Y(\mathcal{A})}y\;+\;\sum_{y\in Y(S)}y=\bigl(m_{+}(\mathcal{A})-m_{-}(\mathcal{A})\bigr)\;+\;\bigl(m_{+}(S)-m_{-}(S)\bigr),
$$

where

$$
m_{+}(\mathcal{A}):=|\{\,y\in Y(\mathcal{A}):\,y=+1\,\}|,\quad m_{-}(\mathcal{A}):=|\{\,y\in Y(\mathcal{A}):\,y=-1\,\}|,
$$
 
$$
m_{+}(S):=|\{\,y\in Y(S):\,y=+1\,\}|,\quad m_{-}(S):=|\{\,y\in Y(S):\,y=-1\,\}|.
$$

where $|\cdot|$ denotes the cardinality, that is, the number of source (instance).

The learner predicts $\widehat{y}(\mathcal{A},S)=\operatorname{sgn}_{-}\!\bigl(T(\mathcal{A},S)\bigr)$, where

$$
\operatorname{sgn}_{-}(x)\;=\;\begin{cases}+1,&x>0,\\
-1,&x\leq 0,\end{cases}
$$

i.e., ties (including the empty set) default to $-1$. The utility at state $\mathcal{A}$ for a set $S$ is

$$
v(S;\mathcal{A})\;:=\;\mathbf{1}\!\left\{\widehat{y}(\mathcal{A},S)=+1\right\}\;=\;\mathbf{1}\!\left\{T(\mathcal{A},S)>0\right\}\in\{0,1\},
$$

so $v(\cdot;\mathcal{A})$ depends on the state $\mathcal{A}$ through $Y(\mathcal{A})$.

##### Fixed trajectory with identical sources.

Take $T=3$ rounds with one instance in each round $D_{1}=\{z^{\star}\},D_{2}=\{w\},D_{3}=\{z^{\star}\}$ and labels $y_{z^{\star}}=+1,y_{w}=+1$. Let $k=1$ and $\ell=3$. Denote the realized model states

$$
\mathcal{A}_{\text{init}},\qquad\mathcal{A}_{1}=\mathcal{A}(D_{1})=\mathcal{A}(\{z^{\star}\}),\qquad\mathcal{A}_{2}=\mathcal{A}(D_{1}\cup D_{2})=\mathcal{A}(\{z^{\star},w\}).
$$

Because $|D_{t}|=1$ for $t\in\{1,3\}$, the within-round averaging in (6) degenerates to the single subset $S_{t}=\varnothing$; hence for $t\in\{1,3\}$ and $z=z^{\star}$,

$$
\overline{\Delta}_{t}\!\bigl(z\mid\mathcal{A}_{t-1}\bigr)=\Delta_{\mathcal{A}_{t-1}}\!\bigl(z\mid\varnothing\bigr)=v(\{z\};\mathcal{A}_{t-1})-v(\varnothing;\mathcal{A}_{t-1}),
$$

as in (5).

##### Round k=1k=1.

At $\mathcal{A}_{\text{init}}$: $Y(\mathcal{A}_{\text{init}})=\varnothing$, so $T(\mathcal{A}_{\text{init}},\varnothing)=0$ and $v(\varnothing;\mathcal{A}_{\text{init}})=\mathbf{1}\{0>0\}=0$. With $\{z^{\star}\}$: $T(\mathcal{A}_{\text{init}},\{z^{\star}\})=+1$, hence $v(\{z^{\star}\};\mathcal{A}_{\text{init}})=\mathbf{1}\{1>0\}=1$. Therefore

$$
\overline{\Delta}_{1}\!\bigl(z^{\star}\mid\mathcal{A}_{\text{init}}\bigr)=v(\{z^{\star}\};\mathcal{A}_{\text{init}})-v(\varnothing;\mathcal{A}_{\text{init}})=1-0=1.
$$

##### Round ℓ=3\\ell=3.

At $\mathcal{A}_{2}$: $Y(\mathcal{A}_{2})=\{+1,+1\}$, so $T(\mathcal{A}_{2},\varnothing)=+2$ and $v(\varnothing;\mathcal{A}_{2})=\mathbf{1}\{2>0\}=1$. With $\{z^{\star}\}$: $T(\mathcal{A}_{2},\{z^{\star}\})=+3$, hence $v(\{z^{\star}\};\mathcal{A}_{2})=\mathbf{1}\{3>0\}=1$. Therefore

$$
\overline{\Delta}_{3}\!\bigl(z^{\star}\mid\mathcal{A}_{2}\bigr)=v(\{z^{\star}\};\mathcal{A}_{2})-v(\varnothing;\mathcal{A}_{2})=1-1=0.
$$

Hence, along this fixed realized trajectory with identical sources placed in rounds $k=1$ and $\ell=3$, $\overline{\Delta}_{k}\!\bigl(z^{\star}\mid\mathcal{A}_{k-1}\bigr)=1$ while $\overline{\Delta}_{\ell}\!\bigl(z^{\star}\mid\mathcal{A}_{\ell-1}\bigr)=0$, so the two state–conditioned values differ. The only change between $k$ and $\ell$ is the intervening update of the model state from round 2, hence the lemma follows: whenever $v(\cdot;\mathcal{A})$ depends on $\mathcal{A}$, identical sources generally receive different values across rounds on the realized sequential process.

### B.3 Proof of Theorem 4.1

Consider a weight system $\omega=(\Lambda,\sigma)$ in the sense of weighted Shapley values \[nowak1995axiomatizations\], where $\Lambda=(\lambda_{1},\ldots,\lambda_{n})^{\top}$ assigns zero or positive weights to the each source in $D=\{z_{1},\ldots,z_{n}\}$ and $\sigma=(D_{1},\ldots,D_{T})$ with $D=\bigcup_{t=1}^{T}D_{t}$ imposes the group precedence. By the axiomatization of weighted random–order values (see Remark 2.2 of \[nowak1995axiomatizations\]), the axioms of Efficiency, Linearity, Nullity, and $\omega$ –Mutual Dependence uniquely determine a value with the permutation form

$$
\phi_{\omega}(z;v,D)\;=\;\sum_{\pi\in\Pi(D)}p_{\pi}^{\omega}\left[v\!\bigl(\pi^{<z}(D)\cup\{z\}\bigr)-v\!\bigl(\pi^{<z}(D)\bigr)\right],
$$

where, for a permutation $\pi(D)=(z_{o_{1}},\ldots,z_{o_{n}})$ and the unique index $r$ with $z_{o_{j}}=z$, the predecessor set is $\pi^{<z}(D):=\{z_{o_{1}},\ldots,z_{o_{j-1}}\}$; the probability of permutation $\pi$ is

$$
p_{\pi}^{\omega}\;=\;\begin{cases}\displaystyle\prod_{j=1}^{n}\frac{\lambda_{o_{j}}}{\sum\limits_{z_{o_{\ell}}\in\mathrm{Max}^{\sigma}(\{z_{o_{1}},\ldots,z_{o_{j}}\})}\lambda_{o_{\ell}}},&\text{if }\pi\in\Pi_{\sigma}(D),\\[10.00002pt]
0,&\text{otherwise},\end{cases}
$$

and $\mathrm{Max}^{\sigma}(S):=\{z\in S:\;z\succeq_{\sigma}z^{\prime}\text{ for all }z^{\prime}\in S\}$ selects the maximal elements in $S$ under the precedence defined in $\sigma$.

Our Axiom 4.1 is a specialization of $\omega$ -Mutual Dependence that enforces symmetry within each group, that is, equal value of mutually dependent sources inside each group $D_{t}$. Equivalently, there exist group weights $(\lambda^{(1)},\ldots,\lambda^{(T)})$ such that $\lambda_{i}=\lambda^{(t)}$ for all $z_{i}\in D_{t}$. For any $\pi\in\Pi_{\sigma}(D)$, elements are appended group by group. When the $r$ th element from $D_{t}$ is appended, the numerator in (14) is $\lambda^{(t)}$ and the denominator is $r\,\lambda^{(t)}$, so the contribution from $D_{t}$ equals $\prod_{r=1}^{|D_{t}|}\frac{\lambda^{(t)}}{r\,\lambda^{(t)}}=1/|D_{t}|!$. Multiplying over $t=1,\ldots,T$ yields

$$
p_{\pi}^{\omega}=\prod_{t=1}^{T}\frac{1}{|D_{t}|!}\;=\;\frac{1}{\prod_{t=1}^{T}(|D_{t}|!)}\quad\text{for all }\pi\in\Pi_{\sigma}(D),
$$

and $p_{\pi}^{\omega}=0$ otherwise. Substituting these probabilities into (13) gives

$$
p_{\pi}^{\omega}=p_{\pi}^{\sigma}=\begin{cases}\displaystyle\frac{1}{\prod_{t=1}^{T}(|D_{t}|!)},&\pi\in\Pi_{\sigma}(D),\\[6.00006pt]
0,&\text{otherwise},\end{cases}
$$

which is exactly the permutation weights (8). Uniqueness follows from the cited axiomatization given the four axioms in Theorem 4.1, hence establishing Theorem 4.1. $\square$

### B.4 Proof of Proposition 4.1

Fix $t\in[T]$ and $z\in D_{t}$. Let $U_{t-1}:=\bigcup_{j=1}^{t-1}D_{j}$ denote the union of all sources in groups preceding $t$. For any $\pi\in\Pi_{\sigma}(D)$ the predecessor set of $z$ satisfies

$$
U_{t-1}\;\subseteq\;\pi^{<z}(D)\;\subseteq\;U_{t-1}\cup\bigl(D_{t}\setminus\{z\}\bigr),
$$

so there exists a unique $S_{t}\subseteq D_{t}\setminus\{z\}$ such that $\pi^{<z}(D)=U_{t-1}\cup S_{t}$.

For a fixed $S_{t}\subseteq D_{t}\setminus\{z\}$, the number of ordered permutations in $\Pi_{\sigma}(D)$ that produce this predecessor set equals

$$
N(S_{t})=\Bigl(\prod_{j=1}^{t-1}|D_{j}|!\Bigr)\cdot|S_{t}|!\cdot\bigl(|D_{t}|-|S_{t}|-1\bigr)!\cdot\Bigl(\prod_{j=t+1}^{T}|D_{j}|!\Bigr),
$$

corresponding to arbitrary within–group orders for $D_{1},\ldots,D_{t-1}$, then an order of $S_{t}$, then an order of $D_{t}\setminus(S_{t}\cup\{z\})$, followed by arbitrary orders for the remaining groups. Each such permutation has probability $1/\prod_{j=1}^{T}(|D_{j}|!)$ under (8). Therefore,

$$
\displaystyle\phi^{\sigma}(z;\,v,D)
$$
 
$$
\displaystyle=\sum_{\pi\in\Pi_{\sigma}(D)}\frac{1}{\prod_{j=1}^{T}(|D_{j}|!)}\left[v\bigl(\pi^{<z}(D)\cup\{z\}\bigr)-v\bigl(\pi^{<z}(D)\bigr)\right]
$$
 
$$
\displaystyle=\sum_{S_{t}\subseteq D_{t}\setminus\{z\}}\frac{N(S_{t})}{\prod_{j=1}^{T}(|D_{j}|!)}\,\Bigl[v\bigl(U_{t-1}\cup S_{t}\cup\{z\}\bigr)-v\bigl(U_{t-1}\cup S_{t}\bigr)\Bigr]
$$
 
$$
\displaystyle=\frac{1}{|D_{t}|!}\sum_{S_{t}\subseteq D_{t}\setminus\{z\}}|S_{t}|!\,\bigl(|D_{t}|-|S_{t}|-1\bigr)!\,\Bigl[v\bigl(U_{t-1}\cup S_{t}\cup\{z\}\bigr)-v\bigl(U_{t-1}\cup S_{t}\bigr)\Bigr].
$$

Using $\displaystyle\binom{|D_{t}|-1}{|S_{t}|}=\frac{(|D_{t}|-1)!}{|S_{t}|!\,(|D_{t}|-|S_{t}|-1)!}$ gives

$$
\phi^{\sigma}(z;\,v,D)=\frac{1}{|D_{t}|}\sum_{S_{t}\subseteq D_{t}\setminus\{z\}}\binom{|D_{t}|-1}{|S_{t}|}^{-1}\Bigl[v\bigl(U_{t-1}\cup S_{t}\cup\{z\}\bigr)-v\bigl(U_{t-1}\cup S_{t}\bigr)\Bigr],
$$

which is exactly (9). For $t=1$, $U_{t-1}=\emptyset$ and (9) reduces to the subset form of classical DS within $D_{1}$. $\square$

### B.5 Proof of Proposition 4.2

Fix a group index $t\in[T]$ and an ordered permutation $\pi_{\sigma}(D)=(z_{o_{1}},\ldots,z_{o_{n}})\in\Pi_{\sigma}(D)$. For any source $z_{o_{j}}$, define its predecessor set in $\pi_{\sigma}(D)$ by

$$
\pi^{<z_{o_{j}}}(D)\;:=\;\{z_{o_{1}},\ldots,z_{o_{j-1}}\},\qquad j=1,\ldots,n,
$$

with the convention $\pi^{<z_{o_{1}}}(D)=\varnothing$.

Because $\pi_{\sigma}(D)$ respects the group order under $\sigma$, all sources in $U_{t}\;=\;\bigcup_{j=1}^{t}D_{j}$ appear before any source in $D\setminus U_{t}$. Hence the first $\lvert U_{t}\rvert$ positions of $\pi_{\sigma}(D)$ are exactly the sources in $U_{t}$, which we denote by $z_{o_{1}},\ldots,z_{o_{\lvert U_{t}\rvert}}$ in their permutation order. Telescoping along this prefix gives

$$
\displaystyle v(U_{t})-v(\varnothing)
$$
 
$$
\displaystyle=v\bigl(\{z_{o_{1}},\ldots,z_{o_{\lvert U_{t}\rvert}}\}\bigr)-v(\varnothing)
$$
 
$$
\displaystyle=\sum_{j=1}^{\lvert U_{t}\rvert}\Bigl[v\bigl(\{z_{o_{1}},\ldots,z_{o_{j}}\}\bigr)-v\bigl(\{z_{o_{1}},\ldots,z_{o_{j-1}}\}\bigr)\Bigr]
$$
 
$$
\displaystyle=\sum_{j=1}^{\lvert U_{t}\rvert}\Bigl[v\bigl(\pi^{<z_{o_{j}}}(D)\cup\{z_{o_{j}}\}\bigr)-v\bigl(\pi^{<z_{o_{j}}}(D)\bigr)\Bigr]
$$
 
$$
\displaystyle=\sum_{z\in U_{t}}\Bigl[v\bigl(\pi^{<z}(D)\cup\{z\}\bigr)-v\bigl(\pi^{<z}(D)\bigr)\Bigr],
$$

where the last equality simply reindexes the sum over $z\in U_{t}$. This identity holds for every $\pi_{\sigma}(D)\in\Pi_{\sigma}(D)$.

By the permutation form of ADS,

$$
\phi^{\sigma}(z;v,D)=\frac{1}{\lvert\Pi_{\sigma}(D)\rvert}\sum_{\pi_{\sigma}(D)\in\Pi_{\sigma}(D)}\Bigl[v\bigl(\pi^{<z}(D)\cup\{z\}\bigr)-v\bigl(\pi^{<z}(D)\bigr)\Bigr],\qquad z\in D.
$$

Averaging the identity above uniformly over $\pi_{\sigma}(D)\in\Pi_{\sigma}(D)$ and exchanging the order of summation yields

$$
\displaystyle\sum_{z\in U_{t}}\phi^{\sigma}(z;v,D)
$$
 
$$
\displaystyle=\sum_{z\in U_{t}}\frac{1}{\lvert\Pi_{\sigma}(D)\rvert}\sum_{\pi_{\sigma}(D)\in\Pi_{\sigma}(D)}\Bigl[v\bigl(\pi^{<z}(D)\cup\{z\}\bigr)-v\bigl(\pi^{<z}(D)\bigr)\Bigr]
$$
 
$$
\displaystyle=\frac{1}{\lvert\Pi_{\sigma}(D)\rvert}\sum_{\pi_{\sigma}(D)\in\Pi_{\sigma}(D)}\sum_{z\in U_{t}}\Bigl[v\bigl(\pi^{<z}(D)\cup\{z\}\bigr)-v\bigl(\pi^{<z}(D)\bigr)\Bigr]
$$
 
$$
\displaystyle=v(U_{t})-v(\varnothing).
$$

Applying the same argument with $t-1$ in place of $t$ gives

$$
\sum_{z\in U_{t-1}}\phi^{\sigma}(z;v,D)=v(U_{t-1})-v(\varnothing).
$$

Subtracting this from the previous display and using $D_{t}=U_{t}\setminus U_{t-1}$, we obtain

$$
\sum_{z\in D_{t}}\phi^{\sigma}(z;v,D)=v(U_{t})-v(U_{t-1}),
$$

which proves Proposition 4.2. $\square$

### B.6 Proof of Theorem 5.1

To prove Theorem 5.1, we first derive a general difference identity for two data instances in the same group, and then apply it with $v=v_{\text{knn}}$ on the instance ground set.

##### Preliminaries and notations.

Fix an ordered collection $\sigma=(D_{1},\ldots,D_{T})$ of nonempty groups of sources and let $D=\bigcup_{t=1}^{T}D_{t}$ be the full training dataset. For a given group index $t\in[T]$, recall

$$
P_{t}\;:=\;\operatorname{\mathrm{Ins}}(U_{t-1})=\bigcup_{j=1}^{t-1}\operatorname{\mathrm{Ins}}(D_{j}),\qquad C_{t}\;:=\;\operatorname{\mathrm{Ins}}(D_{t}),
$$

and write $m(P_{t}):=|P_{t}|$ and $m(C_{t}):=|C_{t}|$ for their numbers of instances.

Fix a test instance $q_{\text{test}}=(x_{\text{test}},y_{\text{test}})$ and a distance metric $d(\cdot,\cdot)$. For any finite set of instances $T\subseteq\operatorname{\mathrm{Ins}}(D)$ and any $i\geq 1$, let $I_{{d}}^{i}(x_{\text{test}};T)$ denote the index of the $i$ th nearest neighbor of $x_{\text{test}}$ in $T$ under $d$, and write $q_{I_{{d}}^{i}(x_{\text{test}};T)}$ for that instance. For a finite collection of sources $S\subseteq D$, we abbreviate $I_{{d}}^{i}(x_{\text{test}};\operatorname{\mathrm{Ins}}(S))$ by $I_{{d}}^{i}(x_{\text{test}};S)$ when convenient.

The KNN utility at $q_{\text{test}}$ for a finite collection of sources $S\subseteq D$ is, as in Definition 5.2,

$$
v_{\text{knn}}(S)\;:=\;\frac{1}{K^{\prime}}\sum_{i=1}^{K^{\prime}}\mathbf{1}\!\left[y_{I_{{d}}^{i}(x_{\text{test}};\operatorname{\mathrm{Ins}}(S))}=y_{\text{test}}\right],\qquad K^{\prime}:=\min\{K,\;m(S)\},
$$

where $m(S):=|\operatorname{\mathrm{Ins}}(S)|$ is the number of instances induced by $S$.

For a given group index $t\in[T]$ and $i=1,\ldots,m(C_{t})$, we define

$$
c_{t,i}\;:=\;\bigl|\bigl\{\,q=(x,y)\in P_{t}:\ d\bigl(x,x_{\text{test}}\bigr)\;<\;d\!\bigl(x_{I_{{d}}^{i}(x_{\text{test}};C_{t})},x_{\text{test}}\bigr)\,\bigr\}\bigr|,
$$

so $c_{t,i}$ counts how many instances from preceding groups $P_{t}$ are closer to $x_{\text{test}}$ than the $i$ th nearest instance taken from $C_{t}$. We now state a general identity that expresses the difference of ADS values for two instances within $C_{t}$ as a weighted difference of utilities.

###### Lemma B.1 (Intra-group difference identity).

Fix a group index $t\in[T]$ and two distinct instances $q_{i},q_{j}\in C_{t}$. Let $\widetilde{C}_{t}:=C_{t}\setminus\{q_{i},q_{j}\}$. Then, for any utility $v$ defined on subsets of the instance ground set $\operatorname{\mathrm{Ins}}(D)$,

$$
\displaystyle\phi^{\sigma}(q_{i};v,\operatorname{\mathrm{Ins}}(D))-\phi^{\sigma}(q_{j};v,\operatorname{\mathrm{Ins}}(D))
$$
 
$$
\displaystyle=\frac{1}{m(C_{t})-1}\sum_{S\subseteq\widetilde{C}_{t}}\binom{m(C_{t})-2}{m(S)}^{-1}\Bigl[v\bigl(P_{t}\cup S\cup\{q_{i}\}\bigr)-v\bigl(P_{t}\cup S\cup\{q_{j}\}\bigr)\Bigr].
$$

##### Proof of Lemma B.1.

By Proposition 4.1, applied at the instance level for the fixed group $t\in[T]$ and ground set $\operatorname{\mathrm{Ins}}(D)$,

$$
\displaystyle\phi^{\sigma}(q_{i};v,\operatorname{\mathrm{Ins}}(D))
$$
 
$$
\displaystyle=\frac{1}{m(C_{t})}\sum_{S\subseteq C_{t}\setminus\{q_{i}\}}\binom{m(C_{t})-1}{m(S)}^{-1}\Bigl[v\bigl(P_{t}\cup S\cup\{q_{i}\}\bigr)-v\bigl(P_{t}\cup S\bigr)\Bigr],
$$
$$
\displaystyle\phi^{\sigma}(q_{j};v,\operatorname{\mathrm{Ins}}(D))
$$
 
$$
\displaystyle=\frac{1}{m(C_{t})}\sum_{S\subseteq C_{t}\setminus\{q_{j}\}}\binom{m(C_{t})-1}{m(S)}^{-1}\Bigl[v\bigl(P_{t}\cup S\cup\{q_{j}\}\bigr)-v\bigl(P_{t}\cup S\bigr)\Bigr].
$$

Subtracting and splitting each sum into subsets that do or do not contain the other instance gives

$$
\displaystyle\phi^{\sigma}(q_{i};v,\operatorname{\mathrm{Ins}}(D))-\phi^{\sigma}(q_{j};v,\operatorname{\mathrm{Ins}}(D))
$$
 
$$
\displaystyle=\frac{1}{m(C_{t})}\sum_{S\subseteq\widetilde{C}_{t}}\binom{m(C_{t})-1}{m(S)}^{-1}\Bigl[v\bigl(P_{t}\cup S\cup\{q_{i}\}\bigr)-v\bigl(P_{t}\cup S\cup\{q_{j}\}\bigr)\Bigr]
$$
 
$$
\displaystyle\quad+\frac{1}{m(C_{t})}\sum_{\begin{subarray}{c}S\subseteq C_{t}\setminus\{q_{i}\}\\
q_{j}\in S\end{subarray}}\binom{m(C_{t})-1}{m(S)}^{-1}\Bigl[v\bigl(P_{t}\cup S\cup\{q_{i}\}\bigr)-v\bigl(P_{t}\cup S\bigr)\Bigr]
$$
 
$$
\displaystyle\quad-\frac{1}{m(C_{t})}\sum_{\begin{subarray}{c}S\subseteq C_{t}\setminus\{q_{j}\}\\
q_{i}\in S\end{subarray}}\binom{m(C_{t})-1}{m(S)}^{-1}\Bigl[v\bigl(P_{t}\cup S\cup\{q_{j}\}\bigr)-v\bigl(P_{t}\cup S\bigr)\Bigr].
$$

Reindex the last two sums by writing $S=S^{\prime}\cup\{q_{j}\}$ and $S=S^{\prime}\cup\{q_{i}\}$, respectively, with $S^{\prime}\subseteq\widetilde{C}_{t}$. After reindexing, the weight becomes $\binom{m(C_{t})-1}{m(S^{\prime})+1}^{-1}$ and the $-v(\cdot)$ terms cancel, which yields

$$
\displaystyle\phi^{\sigma}(q_{i};v,\operatorname{\mathrm{Ins}}(D))-\phi^{\sigma}(q_{j};v,\operatorname{\mathrm{Ins}}(D))
$$
 
$$
\displaystyle=\frac{1}{m(C_{t})}\sum_{S^{\prime}\subseteq\widetilde{C}_{t}}\biggl\{\binom{m(C_{t})-1}{m(S^{\prime})}^{-1}+\binom{m(C_{t})-1}{m(S^{\prime})+1}^{-1}\biggr\}
$$
 
$$
\displaystyle\hskip 50.00008pt\times\Bigl[v\bigl(P_{t}\cup S^{\prime}\cup\{q_{i}\}\bigr)-v\bigl(P_{t}\cup S^{\prime}\cup\{q_{j}\}\bigr)\Bigr].
$$

Using the binomial identity

$$
\frac{1}{\binom{M}{s}}+\frac{1}{\binom{M}{s+1}}=\frac{M+1}{M}\cdot\frac{1}{\binom{M-1}{s}},\qquad(M=m(C_{t})-1,\ s=m(S^{\prime})),
$$

we obtain

$$
\displaystyle\phi^{\sigma}(q_{i};v,\operatorname{\mathrm{Ins}}(D))-\phi^{\sigma}(q_{j};v,\operatorname{\mathrm{Ins}}(D))
$$
 
$$
\displaystyle=\frac{1}{m(C_{t})-1}\sum_{S^{\prime}\subseteq\widetilde{C}_{t}}\binom{m(C_{t})-2}{m(S^{\prime})}^{-1}\Bigl[v\bigl(P_{t}\cup S^{\prime}\cup\{q_{i}\}\bigr)-v\bigl(P_{t}\cup S^{\prime}\cup\{q_{j}\}\bigr)\Bigr],
$$

which proves Lemma B.1. $\square$

##### Proof of Theorem 5.1.

Fix a group index $t\in[T]$ and a test instance $q_{\text{test}}=(x_{\text{test}},y_{\text{test}})$. Order the instances in $P_{t}\cup C_{t}$ by increasing distance to $x_{\text{test}}$ under $d$. For $i=1,\ldots,m(C_{t})$, write $q_{I^{d}_{i}}:=q_{I^{d}_{i}(x_{\text{test}};C_{t})}\in C_{t}$, so $q_{I^{d}_{i}}$ is the $i$ th nearest neighbor to $x_{\text{test}}$ inside $C_{t}$. Fix $i\in\{1,\ldots,m(C_{t})-1\}$ and consider the consecutive neighbors $q_{I^{d}_{i}},q_{I^{d}_{i+1}}$. For any $S\subseteq C_{t}\setminus\{q_{I^{d}_{i}},q_{I^{d}_{i+1}}\}$, decompose

$$
S=S_{1}\cup S_{2},\qquad S_{1}\subseteq\{q_{I^{d}_{1}},\ldots,q_{I^{d}_{i-1}}\},\quad S_{2}\subseteq\{q_{I^{d}_{i+2}},\ldots,q_{I^{d}_{m(C_{t})}}\}.
$$

Let

$$
\Delta^{+}_{i,P_{t}}(S):=v_{\text{knn}}\bigl(P_{t}\cup S\cup\{q_{I^{d}_{i}}\}\bigr)-v_{\text{knn}}\bigl(P_{t}\cup S\bigr)
$$

be the change in KNN utility when adding $q_{I^{d}_{i}}$ to $P_{t}\cup S$. Because $m(S_{1})$ instances of $S$ are closer to $x_{\text{test}}$ than $q_{I^{d}_{i}}$ and $c_{t,i}$ instances of $P_{t}$ are closer than $q_{I^{d}_{i}}$, the total number of instances in $P_{t}\cup S$ that precede $q_{I^{d}_{i}}$ is $c_{t,i}+m(S_{1})$.

If $c_{t,i}+m(S_{1})\geq K$, then $q_{I^{d}_{i}}$ cannot enter the $K$ nearest neighbors of $x_{\text{test}}$ in $P_{t}\cup S\cup\{q_{I^{d}_{i}}\}$, so $\Delta^{+}_{i,P_{t}}(S)=0$. If $c_{t,i}+m(S_{1})<K$, then $q_{I^{d}_{i}}$ becomes one of the $K$ nearest neighbors after being added and replaces the previous $K$ th neighbor of $x_{\text{test}}$ in $P_{t}\cup S$. Let $q_{I^{d}_{K}(x_{\text{test}};P_{t}\cup S)}$ denote this $K$ th nearest neighbor before insertion. By the definition of $v_{\text{knn}}$, we have

$$
\Delta^{+}_{i,P_{t}}(S)=\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{\,I^{d}_{K}(x_{\text{test}};P_{t}\cup S)}=y_{\text{test}}]}{K}.
$$

An analogous expression holds for

$$
\Delta^{+}_{i+1,P_{t}}(S):=v_{\text{knn}}\bigl(P_{t}\cup S\cup\{q_{I^{d}_{i+1}}\}\bigr)-v_{\text{knn}}\bigl(P_{t}\cup S\bigr).
$$

Define the pointwise KNN utility difference

$$
\Delta_{i,P_{t}}(S):=v_{\text{knn}}\bigl(P_{t}\cup S\cup\{q_{I^{d}_{i}}\}\bigr)-v_{\text{knn}}\bigl(P_{t}\cup S\cup\{q_{I^{d}_{i+1}}\}\bigr)=\Delta^{+}_{i,P_{t}}(S)-\Delta^{+}_{i+1,P_{t}}(S).
$$

Using the expressions above for $\Delta^{+}_{i,P_{t}}(S)$ and $\Delta^{+}_{i+1,P_{t}}(S)$, a case analysis on $c_{t,i}$, $c_{t,i+1}$ and $m(S_{1})$ yields:

- If $m(S_{1})\geq K-c_{t,i}$, then both $c_{t,i}+m(S_{1})\geq K$ and $c_{t,i+1}+m(S_{1})\geq K$, so neither point enters the $K$ nearest neighbors and
	$$
	\Delta_{i,P_{t}}(S)=0.
	$$
- If $m(S_{1})<K-c_{t,i+1}$, then $c_{t,i}+m(S_{1})<K$ and $c_{t,i+1}+m(S_{1})<K$. In this regime, $q_{I^{d}_{i}}$ and $q_{I^{d}_{i+1}}$ both enter the $K$ nearest neighbors and displace the same original $K$ th neighbor in $P_{t}\cup S$. Thus the terms involving $q_{I^{d}_{K}(x_{\text{test}};P_{t}\cup S)}$ cancel and
	$$
	\Delta_{i,P_{t}}(S)=\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{q_{I^{d}_{i+1}}}=y_{\text{test}}]}{K}.
	$$
- If $K-c_{t,i+1}\leq m(S_{1})<K-c_{t,i}$, then $c_{t,i}+m(S_{1})<K$ but $c_{t,i+1}+m(S_{1})\geq K$. Hence $q_{I^{d}_{i+1}}$ never enters the $K$ nearest neighbors, while $q_{I^{d}_{i}}$ does, so
	$$
	\Delta_{i,P_{t}}(S)=\Delta^{+}_{i,P_{t}}(S).
	$$
	In this regime, the $K$ nearest neighbors of $x_{\text{test}}$ in $P_{t}\cup S$ consist of all points in $S_{1}$ together with the $(K-m(S_{1}))$ nearest neighbors in $P_{t}$. Consequently, the original $K$ th neighbor in $P_{t}\cup S$ is exactly $q_{I^{d}_{K-m(S_{1})}(x_{\text{test}};P_{t})}$, and
	$$
	\Delta_{i,P_{t}}(S)=\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{\,I^{d}_{\,K-m(S_{1})}(x_{\text{test}};P_{t})}=y_{\text{test}}]}{K}.
	$$

Summarizing, exactly one of the following three regimes applies:

$$
\displaystyle m(S_{1})\geq K-c_{t,i}\quad\Rightarrow\quad\Delta_{i,P_{t}}(S)=0;
$$
$$
\displaystyle m(S_{1})<K-c_{t,i+1}\quad\Rightarrow\quad\Delta_{i,P_{t}}(S)=\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{q_{I^{d}_{i+1}}}=y_{\text{test}}]}{K};
$$
$$
\displaystyle K-c_{t,i+1}\leq m(S_{1})<K-c_{t,i}\quad\Rightarrow\quad\Delta_{i,P_{t}}(S)=\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{\,I^{d}_{\,K-m(S_{1})}(x_{\text{test}};P_{t})}=y_{\text{test}}]}{K}.
$$

Applying Lemma B.1 with $v=v_{\text{knn}}$ and $(q_{I^{d}_{i}},q_{j})=(q_{I^{d}_{i}},q_{I^{d}_{i+1}})$ gives

$$
\phi^{\sigma}(q_{I^{d}_{i}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))-\phi^{\sigma}(q_{I^{d}_{i+1}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))=\frac{1}{m(C_{t})-1}\sum_{S\subseteq C_{t}\setminus\{q_{I^{d}_{i}},q_{I^{d}_{i+1}}\}}\binom{m(C_{t})-2}{m(S)}^{-1}\,\Delta_{i,P_{t}}(S).
$$

We now evaluate (18) in the four cases of Theorem 5.1.

Case 1: $K\leq c_{t,i}$. Then $K-c_{t,i}\leq 0$, so condition (15) holds for every $S$ and $\Delta_{i,P_{t}}(S)=0$. Hence

$$
\displaystyle\phi^{\sigma}(q_{I^{d}_{i}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))-\phi^{\sigma}(q_{I^{d}_{i+1}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))
$$
 
$$
\displaystyle=\frac{1}{m(C_{t})-1}\sum_{S\subseteq C_{t}\setminus\{q_{I^{d}_{i}},q_{I^{d}_{i+1}}\}}\binom{m(C_{t})-2}{m(S)}^{-1}\,\Delta_{i,P_{t}}(S)
$$
 
$$
\displaystyle=\frac{1}{m(C_{t})-1}\sum_{S\subseteq C_{t}\setminus\{q_{I^{d}_{i}},q_{I^{d}_{i+1}}\}}\binom{m(C_{t})-2}{m(S)}^{-1}\cdot 0
$$
 
$$
\displaystyle=0,
$$

which is exactly Item 1 of Theorem 5.1.

Case 2: $K>c_{t,i+1}=c_{t,i}=:c_{t}$. Here $c_{t,i}$ and $c_{t,i+1}$ coincide, so regime (16) is the only one that can occur when $m(S_{1})<K-c_{t}$, while regime (15) applies and $\Delta_{i,P_{t}}(S)=0$ whenever $m(S_{1})\geq K-c_{t}$. Using the decomposition $S=S_{1}\cup S_{2}$ from above, write $s:=m(S_{1})$ and $r:=m(S_{2})$, so $m(S)=s+r$. For fixed $(s,r)$, the number of sets $S$ with these cardinalities equals $\binom{i-1}{s}\binom{m(C_{t})-i-1}{r}$. Substituting $\Delta_{i,P_{t}}(S)$ from (16) into (18) and summing only over those $S$ with $s<K-c_{t}$ yields

$$
\displaystyle\phi^{\sigma}(q_{I^{d}_{i}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))-\phi^{\sigma}(q_{I^{d}_{i+1}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))
$$
 
$$
\displaystyle=\frac{1}{m(C_{t})-1}\sum_{s=0}^{\min\{K-c_{t}-1,i-1\}}\sum_{r=0}^{m(C_{t})-i-1}\binom{i-1}{s}\binom{m(C_{t})-i-1}{r}\binom{m(C_{t})-2}{s+r}^{-1}
$$
 
$$
\displaystyle\hskip 40.00006pt\times\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{q_{I^{d}_{i+1}}}=y_{\text{test}}]}{K}.
$$

We use the following hypergeometric averaging identity: for integers $A,M\geq 0$ and $0\leq s\leq A$,

$$
\frac{1}{A+M+1}\sum_{r=0}^{M}\binom{A}{s}\binom{M}{r}\binom{A+M}{s+r}^{-1}=\frac{1}{A+1}.
$$

We now prove (19). First, rewrite the reciprocal binomial coefficient using the Beta integral. For any integers $N,k\geq 0$,

$$
\binom{N}{k}^{-1}=\frac{k!(N-k)!}{N!}=(N+1)\int_{0}^{1}x^{k}(1-x)^{N-k}\,dx.
$$

Applying this with $N=A+M$ and $k=s+r$ gives

$$
\binom{A+M}{s+r}^{-1}=(A+M+1)\int_{0}^{1}x^{s+r}(1-x)^{A+M-s-r}\,dx.
$$

Substituting into the left-hand side of (19) yields

$$
\displaystyle=\frac{1}{A+M+1}\sum_{r=0}^{M}\binom{A}{s}\binom{M}{r}\binom{A+M}{s+r}^{-1}
$$
 
$$
\displaystyle=\frac{1}{A+M+1}\sum_{r=0}^{M}\binom{A}{s}\binom{M}{r}(A+M+1)\int_{0}^{1}x^{s+r}(1-x)^{A+M-s-r}\,dx
$$
 
$$
\displaystyle=\binom{A}{s}\sum_{r=0}^{M}\binom{M}{r}\int_{0}^{1}x^{s+r}(1-x)^{A+M-s-r}\,dx
$$
 
$$
\displaystyle=\binom{A}{s}\int_{0}^{1}x^{s}(1-x)^{A-s}\sum_{r=0}^{M}\binom{M}{r}x^{r}(1-x)^{M-r}\,dx
$$
 
$$
\displaystyle=\binom{A}{s}\int_{0}^{1}x^{s}(1-x)^{A-s}\bigl(x+(1-x)\bigr)^{M}\,dx
$$
 
$$
\displaystyle=\binom{A}{s}\int_{0}^{1}x^{s}(1-x)^{A-s}\,dx.
$$

The remaining integral is again a Beta function: $\int_{0}^{1}x^{s}(1-x)^{A-s}\,dx=\frac{1}{(A+1)\binom{A}{s}}$, therefore

$$
\text{LHS}=\binom{A}{s}\cdot\frac{1}{(A+1)\binom{A}{s}}=\frac{1}{A+1}.
$$

This proves (19). In our setting, set $A=i-1$ and $M=m(C_{t})-i-1$. Substituting these values into (19) gives

$$
\frac{1}{m(C_{t})-1}\sum_{r=0}^{m(C_{t})-i-1}\binom{i-1}{s}\binom{m(C_{t})-i-1}{r}\binom{m(C_{t})-2}{s+r}^{-1}=\frac{1}{i}.
$$

Hence

$$
\displaystyle\phi^{\sigma}(q_{I^{d}_{i}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))-\phi^{\sigma}(q_{I^{d}_{i+1}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))
$$
 
$$
\displaystyle=\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{q_{I^{d}_{i+1}}}=y_{\text{test}}]}{K}
$$
 
$$
\displaystyle\quad\times\frac{1}{i}\sum_{s=0}^{\min\{K-c_{t}-1,i-1\}}1.
$$

Since $\sum_{s=0}^{\min\{K-c_{t}-1,i-1\}}1=\min\{K-c_{t},i\}$, we obtain

$$
\phi^{\sigma}(q_{I^{d}_{i}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))-\phi^{\sigma}(q_{I^{d}_{i+1}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))=\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{q_{I^{d}_{i+1}}}=y_{\text{test}}]}{K}\cdot\frac{\min\{K-c_{t},i\}}{i},
$$

which is exactly Item 2 of Theorem 5.1.

Case 3: $K>c_{t,i+1}>c_{t,i}$. Here both regimes (16) and (17) can occur.

Regime (16) corresponds to all $S$ with $m(S_{1})<K-c_{t,i+1}$. Repeating the counting argument in Case 2 with $c_{t}$ replaced by $c_{t,i+1}$ yields

$$
\displaystyle\frac{1}{m(C_{t})-1}\sum_{\begin{subarray}{c}S\subseteq C_{t}\setminus\{q_{I^{d}_{i}},q_{I^{d}_{i+1}}\}\\
m(S_{1})<K-c_{t,i+1}\end{subarray}}\binom{m(C_{t})-2}{m(S)}^{-1}\,\Delta_{i,P_{t}}(S)
$$
 
$$
\displaystyle\qquad=\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{q_{I^{d}_{i+1}}}=y_{\text{test}}]}{K}\cdot\frac{\min\{K-c_{t,i+1},i\}}{i}.
$$

Regime (17) corresponds to all $S$ with $K-c_{t,i+1}\leq m(S_{1})<K-c_{t,i},m(S_{1})\leq i-1$. For such $S$, we have

$$
\Delta_{i,P_{t}}(S)=\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{\,I^{d}_{\,K-m(S_{1})}(x_{\text{test}};P_{t})}=y_{\text{test}}]}{K}.
$$

Using (18) and the decomposition $S=S_{1}\cup S_{2}$, we write the contribution of regime (17) as

$$
\displaystyle\frac{1}{m(C_{t})-1}\sum_{\begin{subarray}{c}S\subseteq C_{t}\setminus\{q_{I^{d}_{i}},q_{I^{d}_{i+1}}\}\\
K-c_{t,i+1}\leq m(S_{1})<K-c_{t,i}\end{subarray}}\binom{m(C_{t})-2}{m(S)}^{-1}\,\Delta_{i,P_{t}}(S)
$$
 
$$
\displaystyle\quad=\frac{1}{m(C_{t})-1}\sum_{k=K-c_{t,i+1}}^{m(C_{t})-2}\binom{m(C_{t})-2}{k}^{-1}\sum_{\begin{subarray}{c}S_{1},S_{2}\\
S_{1}\cup S_{2}\subseteq C_{t}\setminus\{q_{I^{d}_{i}},q_{I^{d}_{i+1}}\}\\
K-c_{t,i+1}\leq m(S_{1})<K-c_{t,i}\\
m(S_{1})+m(S_{2})=k\end{subarray}}\Delta_{i,P_{t}}(S_{1}\cup S_{2}),
$$

where $s:=m(S)$ is the total number of instances in $S$. Let $u:=m(S_{1})$ be the total number of instances in $S_{1}$. For fixed $(s,u)$ with $K-c_{t,i+1}\leq u<K-c_{t,i},u\leq k$, the number of subsets $S=S_{1}\cup S_{2}$ satisfying $m(S)=s$ and $m(S_{1})=u$ equals $\binom{i-1}{u}\binom{m(C_{t})-i-1}{s-u}$, since $S_{1}$ is chosen from the first $i-1$ neighbors in $C_{t}$, and $S_{2}$ from the remaining $m(C_{t})-i-1$ instances. Substituting the expression for $\Delta_{i,P_{t}}(S)$ from (17), we obtain

$$
\displaystyle\frac{1}{m(C_{t})-1}\sum_{\begin{subarray}{c}S\subseteq C_{t}\setminus\{q_{I^{d}_{i}},q_{I^{d}_{i+1}}\}\\
K-c_{t,i+1}\leq m(S_{1})<K-c_{t,i}\end{subarray}}\binom{m(C_{t})-2}{m(S)}^{-1}\,\Delta_{i,P_{t}}(S)
$$
 
$$
\displaystyle\quad=\frac{1}{m(C_{t})-1}\sum_{s=K-c_{t,i+1}}^{m(C_{t})-2}\binom{m(C_{t})-2}{k}^{-1}\sum_{u=K-c_{t,i+1}}^{\min\{K-c_{t,i}-1,k\}}\binom{i-1}{u}\binom{m(C_{t})-i-1}{s-u}
$$
 
$$
\displaystyle\qquad\qquad\times\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{\,I^{d}_{\,K-u}(x_{\text{test}};P_{t})}=y_{\text{test}}]}{K}.
$$

Using the combinatorial identity

$$
\binom{i-1}{u}\binom{m(C_{t})-i-1}{s-u}\binom{m(C_{t})-2}{k}^{-1}=\frac{\binom{s}{u}\binom{m(C_{t})-2-s}{i-1-u}}{\binom{m(C_{t})-2}{i-1}},
$$

we can rewrite the previous expression as

$$
\displaystyle\frac{1}{m(C_{t})-1}\sum_{\begin{subarray}{c}S\subseteq C_{t}\setminus\{q_{I^{d}_{i}},q_{I^{d}_{i+1}}\}\\
K-c_{t,i+1}\leq m(S_{1})<K-c_{t,i}\end{subarray}}\binom{m(C_{t})-2}{m(S)}^{-1}\,\Delta_{i,P_{t}}(S)
$$
 
$$
\displaystyle\quad=\frac{1}{m(C_{t})-1}\sum_{s=K-c_{t,i+1}}^{m(C_{t})-2}\sum_{u=K-c_{t,i+1}}^{\min\{K-c_{t,i}-1,s\}}\frac{\binom{s}{u}\binom{m(C_{t})-2-s}{i-1-u}}{\binom{m(C_{t})-2}{i-1}}
$$
 
$$
\displaystyle\qquad\qquad\times\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{\,I^{d}_{\,K-u}(x_{\text{test}};P_{t})}=y_{\text{test}}]}{K},
$$

which is the second term in Item 3 of Theorem 5.1. Adding the contributions from regimes (16) and (17) gives the full expression in Item 3.

Case 4: $c_{t,i}<K\leq c_{t,i+1}$. In this case only regime (17) contributes. Recall that regime (17) holds whenever $K-c_{t,i+1}\leq|S_{1}|<K-c_{t,i},\qquad|S_{1}|\leq i-1$. Under the Case 4 condition $K\leq c_{t,i+1}$, we have $K-c_{t,i+1}\leq 0$, so the lower bound becomes $0\leq|S_{1}|<K-c_{t,i},|S_{1}|\leq i-1$. Thus, for regime (17) we require

$$
0\leq|S_{1}|\leq\min\{K-c_{t,i}-1,\ i-1\}.
$$

Let $s:=|S|$ and $u:=|S_{1}|$, so $|S_{2}|=s-u$. For fixed $(s,u)$ with $0\leq u\leq\min\{K-c_{t,i}-1,i-1,s\}$, the number of subsets $S$ with $|S|=s$ and $|S_{1}|=u$ is $\binom{i-1}{u}\binom{m(C_{t})-i-1}{s-u}$. Substituting the regime (17) expression for $\Delta_{i,P_{t}}(S)$ into (18) and summing over all feasible $(s,u)$, we obtain

$$
\displaystyle\phi^{\sigma}(q_{I^{d}_{i}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))-\phi^{\sigma}(q_{I^{d}_{i+1}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))
$$
 
$$
\displaystyle\quad=\frac{1}{m(C_{t})-1}\sum_{s=0}^{m(C_{t})-2}\sum_{u=0}^{\min\{K-c_{t,i}-1,s,i-1\}}\binom{i-1}{u}\binom{m(C_{t})-i-1}{s-u}\binom{m(C_{t})-2}{s}^{-1}
$$
 
$$
\displaystyle\hskip 40.00006pt\times\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{\,I^{d}_{\,K-u}(x_{\text{test}};P_{t})}=y_{\text{test}}]}{K}.
$$

The inner upper bound $\min\{K-c_{t,i}-1,s,i-1\}$ can be simplified to $\min\{K-c_{t,i}-1,s\}$, since $u\leq i-1$ is already enforced by the binomial coefficient $\binom{i-1}{u}$. Next, we rewrite the counting factor using a standard hypergeometric-style identity: for all integers $s,u$ with $0\leq u\leq s$,

$$
\binom{i-1}{u}\binom{m(C_{t})-i-1}{s-u}=\binom{m(C_{t})-2}{s}\,\frac{\binom{s}{u}\binom{m(C_{t})-2-s}{i-1-u}}{\binom{m(C_{t})-2}{i-1}}.
$$

Substituting this identity and canceling the factor $\binom{m(C_{t})-2}{s}$ with $\binom{m(C_{t})-2}{s}^{-1}$ yields

$$
\displaystyle\phi^{\sigma}(q_{I^{d}_{i}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))-\phi^{\sigma}(q_{I^{d}_{i+1}};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))
$$
 
$$
\displaystyle\quad=\frac{1}{m(C_{t})-1}\sum_{s=0}^{m(C_{t})-2}\sum_{u=0}^{\min\{K-c_{t,i}-1,s\}}\frac{\binom{s}{u}\binom{m(C_{t})-2-s}{i-1-u}}{\binom{m(C_{t})-2}{i-1}}\,\frac{\mathbf{1}[y_{q_{I^{d}_{i}}}=y_{\text{test}}]-\mathbf{1}[y_{\,I^{d}_{\,K-u}(x_{\text{test}};P_{t})}=y_{\text{test}}]}{K},
$$

which is exactly the expression in Item 4 of Theorem 5.1.

Base case. Fix a group index $t$, let $q_{\max}:=q_{I^{d}_{m(C_{t})}(x_{\text{test}};C_{t})}$ denote the farthest instance wthin $C_{t}$ from $x_{\text{test}}$, and define

$$
c_{\max}:=c_{t,m(C_{t})}=\bigl|\bigl\{\,q=(x,y)\in P_{t}:\ d\bigl(x,x_{\text{test}}\bigr)<d\!\bigl(x_{q_{\max}},x_{\text{test}}\bigr)\,\bigr\}\bigr|.
$$

For any $S\subseteq C_{t}\setminus\{q_{\max}\}$, define

$$
\Delta_{\max}(S):=v_{\text{knn}}\bigl(P_{t}\cup S\cup\{q_{\max}\}\bigr)-v_{\text{knn}}(P_{t}\cup S).
$$

By inspecting the $K$ nearest neighbors of $x_{\text{test}}$ inside $P_{t}\cup S$ before and after adding $q_{\max}$, we obtain three regimes:

$$
\Delta_{\max}(S)=\begin{cases}0,&m(S)\geq K-c_{\max},\\[4.49997pt]
\dfrac{\mathbf{1}[y_{q_{\max}}=y_{\text{test}}]}{K},&m(S)<K-c_{\max}\ \text{and}\ m(P_{t})=c_{\max},\\[8.50006pt]
\dfrac{\mathbf{1}[y_{q_{\max}}=y_{\text{test}}]-\mathbf{1}[y_{\,I^{d}_{\,K-m(S)}(x_{\text{test}};P_{t})}=y_{\text{test}}]}{K},&m(S)<K-c_{\max}\ \text{and}\ m(P_{t})>c_{\max}.\end{cases}
$$

By Proposition 4.1 applied at the instance level for group $t$,

$$
\phi^{\sigma}(q_{\max};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))=\frac{1}{m(C_{t})}\sum_{S\subseteq C_{t}\setminus\{q_{\max}\}}\binom{m(C_{t})-1}{m(S)}^{-1}\,\Delta_{\max}(S).
$$

*Base Case (i): $K\leq c_{\max}$.* Then $K-c_{\max}\leq 0$, so every $S$ satisfies $m(S)\geq K-c_{\max}$ and hence $\Delta_{\max}(S)=0$. From (20) we immediately obtain

$$
\phi^{\sigma}(q_{\max};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))=0.
$$

*Base Case (ii): $K>c_{\max}$ and $m(P_{t})=c_{\max}$.* Here $\Delta_{\max}(S)=0$ whenever $m(S)\geq K-c_{\max}$, and $\Delta_{\max}(S)=\frac{\mathbf{1}[y_{q_{\max}}=y_{\text{test}}]}{K}\quad\text{for all }S\text{ with }m(S)<K-c_{\max}$. Write $s:=m(S)$. Then (20) can be rewritten as

$$
\displaystyle\phi^{\sigma}(q_{\max};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))
$$
 
$$
\displaystyle=\frac{1}{m(C_{t})}\sum_{s=0}^{K-c_{\max}-1}\sum_{\begin{subarray}{c}S\subseteq C_{t}\setminus\{q_{\max}\}\\
m(S)=s\end{subarray}}\binom{m(C_{t})-1}{s}^{-1}\frac{\mathbf{1}[y_{q_{\max}}=y_{\text{test}}]}{K}.
$$

For each fixed $s$, the number of subsets $S\subseteq C_{t}\setminus\{q_{\max}\}$ with $m(S)=s$ is $\binom{m(C_{t})-1}{s}$, therefore $\sum_{\begin{subarray}{c}S\subseteq C_{t}\setminus\{q_{\max}\}\\
m(S)=s\end{subarray}}\binom{m(C_{t})-1}{s}^{-1}=1$. Hence,

$$
\phi^{\sigma}(q_{\max};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))=\frac{\mathbf{1}[y_{q_{\max}}=y_{\text{test}}]}{K}\cdot\frac{1}{m(C_{t})}\sum_{s=0}^{K-c_{\max}-1}1=\frac{\mathbf{1}[y_{q_{\max}}=y_{\text{test}}]}{K}\cdot\frac{K-c_{\max}}{m(C_{t})}.
$$

*Base Case (iii): $K>c_{\max}$ and $m(P_{t})>c_{\max}$.* Now for $m(S)\geq K-c_{\max}$ we still have $\Delta_{\max}(S)=0$, and for $m(S)<K-c_{\max}$,

$$
\Delta_{\max}(S)=\frac{\mathbf{1}[y_{q_{\max}}=y_{\text{test}}]-\mathbf{1}[y_{\,I^{d}_{\,K-m(S)}(x_{\text{test}};P_{t})}=y_{\text{test}}]}{K}.
$$

Again writing $s:=m(S)$ and substituting into (20) gives

$$
\displaystyle\phi^{\sigma}(q_{\max};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))
$$
 
$$
\displaystyle=\frac{1}{m(C_{t})}\sum_{s=0}^{K-c_{\max}-1}\sum_{\begin{subarray}{c}S\subseteq C_{t}\setminus\{q_{\max}\}\\
m(S)=s\end{subarray}}\binom{m(C_{t})-1}{s}^{-1}\frac{\mathbf{1}[y_{q_{\max}}=y_{\text{test}}]-\mathbf{1}[y_{\,I^{d}_{\,K-s}(x_{\text{test}};P_{t})}=y_{\text{test}}]}{K}
$$
 
$$
\displaystyle=\frac{1}{m(C_{t})K}\sum_{s=0}^{K-c_{\max}-1}\Bigl(\mathbf{1}[y_{q_{\max}}=y_{\text{test}}]-\mathbf{1}[y_{\,I^{d}_{\,K-s}(x_{\text{test}};P_{t})}=y_{\text{test}}]\Bigr)
$$
 
$$
\displaystyle=\frac{K-c_{\max}}{K\,m(C_{t})}\,\mathbf{1}[y_{q_{\max}}=y_{\text{test}}]-\frac{1}{K\,m(C_{t})}\sum_{s=0}^{K-c_{\max}-1}\mathbf{1}[y_{\,I^{d}_{\,K-s}(x_{\text{test}};P_{t})}=y_{\text{test}}].
$$

Collecting the three cases, the base case value of $q_{\max}$ is

$$
\displaystyle\phi^{\sigma}(q_{\max};v_{\text{knn}},\operatorname{\mathrm{Ins}}(D))
$$
 
$$
\displaystyle=\begin{cases}0,&K\leq c_{\max},\\[4.49997pt]
\dfrac{K-c_{\max}}{K\,m(C_{t})}\,\mathbf{1}[y_{q_{\max}}=y_{\text{test}}],&K>c_{\max},m(P_{t})=c_{\max},\\[10.00002pt]
\dfrac{K-c_{\max}}{K\,m(C_{t})}\,\mathbf{1}[y_{q_{\max}}=y_{\text{test}}]-\dfrac{1}{K\,m(C_{t})}\displaystyle\sum_{s=0}^{K-c_{\max}-1}\mathbf{1}[y_{\,I^{d}_{\,K-s}(x_{\text{test}};P_{t})}=y_{\text{test}}],&K>c_{\max},m(P_{t})>c_{\max}.\end{cases}
$$

Combining the base case with the pairwise difference formulas in Cases 1–4 yields the full iterative characterization in Theorem 5.1. $\square$

## Appendix C Experiment Details and Additional Results

This appendix details the experimental setup and reports supplementary analyses. Unless otherwise noted, all computations were performed on CPUs. The LLM experiments in Section 6.3 were run on a server with $8\times$ NVIDIA RTX 4090 GPUs.

### C.1 Valuation of Synthetic Data

Our objective is to measure the intrinsic value of each original instance and the incremental value contributed by each augmented instance beyond the original dataset. Following Proposition 4.4, we compute ADS with $\sigma=(D_{\text{orig}},D_{\text{aug}})$, placing all original sources before augmentation sources. Utility is defined as accuracy on a fixed holdout test set. Treating each instance as a source, we estimate its value using both the Monte Carlo estimator (MC-ADS) and the exact $k$ -nearest neighbor surrogate (KNN-ADS).

We validate the values through removal and addition experiments. For removal, we delete a fixed fraction of augmented instances from the full training dataset ($D=D_{\text{orig}}\cup D_{\text{aug}}$) according to rankings based of different valuation methods and retrain; for addition, we add a fixed fraction of augmented instances to the original training set $D_{\text{orig}}$ according to rankings based of different valuation methods and retrain. A good ranking should produce the largest performance gain when removing the lowest valued instances and the largest performance loss when removing the highest valued points. Likewise, it should give the smallest improvement when adding the lowest valued instances and the largest improvement when adding the highest valued instances. Each curve is the average over 10 independent runs with different random seeds; we report means and 90% percent confidence bands.

#### C.1.1 Adult Experimental Setup.

The Adult dataset \[misc\_adult\_2\] has $48{,}842$ observations for binary income classification. After dropping rows with missing values, the training split has $75.09\%$ negatives (income $\leq$ $50K) and the test split has $75.51\%$ negatives. We subsample $800$ negatives and $200$ positives for training ($n=1{,}000$) and $400$ negatives and $100$ positives for testing. To balance the training set we generate minority samples with Borderline–SMOTE \[han2005borderline\] using $3$ neighbors, which yields $1{,}600$ training instances. A $5$ –NN classifier is trained on the augmented training set $D=D_{\text{orig}}\cup D_{\text{aug}}$. Utility is measured by accuracy on the held‐out test set. For each training instance we compute MC-ADS, KNN-ADS, MC-DS <sup>1</sup>, KNN-DS <sup>2</sup>, and Leave–One–Out (LOO) <sup>3</sup> values. Monte Carlo estimates stabilize after $5{,}000$ permutations. We repeat the entire pipeline ten times under different seeds and plot the mean with a 90% percent confidence band (first row of Figure 2 in the main text).

#### C.1.2 MNIST Experimental Setup

MNIST \[lecun1998gradient\] contains 70,000 grayscale images of handwritten digits (28 $\times$ 28 pixels), labeled from 0 to 9, with 60,000 images for training and 10,000 for testing. We uniformly sample $50$ training and $50$ test images per class, giving $500$ training and $500$ test images. Each training image is transformed once using random rotation in $[-45^{\circ},45^{\circ}]$, random horizontal and vertical shifts in $[-0.0625,0.0625]$, and isotropic scaling in $[0.9,1.1]$ \[shorten2019survey\]. The full training dataset therefore has $1{,}000$ images. We train a $5$ –NN classifier and compute MC-ADS, KNN-ADS, MC-DS <sup>1</sup>, KNN-DS <sup>2</sup>, and LOO values. Accuracy on the test set is the utility. Monte Carlo estimates stabilize after $3{,}000$ permutations. Results are averaged over ten seeds and summarized by means and 90% percent confidence bands (second row of Figure 2 in the main text).

#### C.1.3 Omniglot Experimental Setup.

Omniglot \[lake2015human\] contains 1,623 unique characters, each with 20 grayscale images drawn by different individuals. We sample $5$ images per class from classes #1420–1439 to form the original training and test sets. We then synthesize $100$ additional images using DAGAN \[antoniou2017data\] <sup>4</sup>, resulting in $200$ training images. A logistic regression model is trained with solver liblinear and a maximum of $5{,}000$ iterations. We compute MC-ADS, KNN-ADS, MC-DS <sup>1</sup>, KNN-DS <sup>2</sup>, and LOO values. Utility is test accuracy. Monte Carlo estimates stabilize after $2{,}000$ permutations. We report means and ninety percent confidence bands over ten seeds (third row of Figure 2 in the main text).

#### C.1.4 Additional Omniglot Results with 3–NN

We repeat the Omniglot study with a $3$ –NN classifier. Monte Carlo estimates stabilize after $2{,}000$ permutations. Figure 7 summarizes ten independent runs. Panels (a) and (b) show that removing low or high ADS valued augmented points from the augmented training set yields the largest gain or loss in accuracy. Panels (c) and (d) show that adding low or high ADS valued augmented points to the original set yields the smallest or largest improvement. MC-ADS is slightly stronger than KNN-ADS, and both substantially outperform their symmetric counterparts (MC-DS and KNN-DS) at identifying both harmful and highly beneficial augmentations.

![Refer to caption](https://arxiv.org/html/2511.12863v1/x12.png)

Figure 7: The results of data removal and addition experiments on the Omniglot dataset using a 3-NN classifier.

### C.2 Participant Valuation in Federated Learning

We simulate a synchronous federated learning environment on MNIST \[lecun1998gradient\] with $30$ contributors partitioned evenly across $T=5$ communication rounds (six contributors in each round). The aggregated dataset from six contributors in round $t$ is denoted as $D_{t}$. Each contributor holds $120$ labeled images sampled without replacement from the MNIST training split, for a total of $3{,}600$ images (balanced at $360$ per class at sampling time). All images are converted to tensors and flattened to $784$ features.

##### Global model and local training.

The global model is a two–hidden–layer MLP (input $784$; hidden sizes $200$ / $200$ with ReLU; $10$ -way output), trained with cross-entropy loss. At the start of each round $t$, the server broadcasts the current global parameters. Each of the six contributors in $D_{t}$ trains locally for $50$ epochs using full-batch SGD (learning rate $0.01$). The server aggregates by simple parameter averaging (FedAvg \[mcmahan2017communication\] with equal weights) of the post–local-training models.

##### Noisy contributors.

To test ADS’s capability to distinguish good from noisy contributors, we designate a fraction $\rho=0.5$ of the contributors as noisy. These $15$ contributors are chosen uniformly at random. For each noisy contributor and each of his/her image instances, the true label is independently flipped with probability $p=0.7$ to a uniformly chosen incorrect class in $\{0,\dots,9\}\setminus\{y_{\text{true}}\}$; input features are left unchanged.

##### Utility and seeds.

Utility is the validation accuracy on the fixed MNIST test set ($10{,}000$ images). Results are averaged over $100$ independent seeds; each seed re-samples the training subset and the set of noisy contributors. All runs are executed on CPU; within a seed, all methods share the same global initialization and local-training hyperparameters.

##### Valuation methods (sequential training adaptations).

We introduce here the within‐round variant of classical LOO used throughout our sequential training settings. It adapts the standard definition from Appendix C.1 to pipelines with ordered rounds and is the version applied in federated learning (Appendix 6.2) and multi‐stage LLM fine–tuning (Appendix 6.3). Consider a realized trajectory with rounds $t\in[T]$ and model state $\mathcal{A}_{t-1}$ at the start of round $t$. Let $D_{t}$ denote the data sources active in round $t$. The within‐round LOO value for source $z_{i}\in D_{t}$ is the utility loss from omitting $z_{i}$ while holding the past trajectory fixed and updating the model $\mathcal{A}_{t-1}$ using only the remaining round $t$ sources. This construction respects the realized trajectory—earlier rounds are not recomputed and the cross-round order is never altered. Operationally, our within-round LOO coincides with the Federated Leave-One-Out baseline of wang2020principled: in each round $t$, it evaluates the change in utility when a selected participant is removed from that round’s aggregation, and the overall LOO score is the sum of these per-round losses across rounds. For ADS, we analogously enforce the sequential structure by conditioning on model state $\mathcal{A}_{t-1}$ and averaging one–step marginal contributions over permutations *within* $D_{t}$, consistent with Remark 4.1 in the main text. MC-ADS empirically stabilizes with $200$ permutations (out of $6!=720$ per round).

##### Top-kk contributor selection.

This experiment assesses whether ADS can effectively identify contributors whose updates most improve the global model. For a chosen $k$ (we report $k\in\{3,4\}$ in the main-text plots), each method ranks the six contributors in round $t$ by its score—under MC-ADS, within-round LOO, or a random ranking—and selects the top $k$. We then restart training from the initial global model and, at each round $t$, update the global model using only the top- $k$ contributors selected for that round, recording validation accuracy after every round. For clarity, accuracy curves report the mean across 100 seeds with 95% normal-approximation confidence intervals.

##### Noisy contributor detection.

This experiment tests whether ADS can identify noisy contributors whose updates harm the global model. Within each round, we rank the six contributors in ascending order by score (MC-ADS, within-round LOO, or a random ranking), so lower scores indicate lower quality. For rounds containing at least one truly noisy contributor, we compute a cumulative detection curve: moving upward from the bottom of the ranking, we plot the fraction of noisy contributors among those revealed so far. We report the mean curve across six rounds and 100 seeds with 95% normal-approximation confidence intervals.

### C.3 Dataset Procurement in Multi-stage LLM Fine-tuning

For the valuation methods, we use the sequential training adapations (ADS in Remark 4.1 and within-round LOO) explained in Appendix C.2. For the true dataset, the part-worth utility $u_{i,j}$ for each attribute level is sampled from uniform distribution $\mathcal{U}(0,95/3)$. And the Gaussian noise $\epsilon$ is drawn from $\mathcal{N}(0,5)$. Similarly, the part-worth utility for noisy dataset $u^{\prime}_{i,j}$ is sampled from uniform distribution $\mathcal{U}(0,95/3)$ but with a different random seed, resulting in a distinct set of utility values. And the $\epsilon^{\prime}$ is drawn from $\mathcal{N}(0,5)$. The final ratings are clipped to ensure they fell within the range \[0, 100\].

We selected four open-source foundation models for our experiments: Llama 3.1-8B, Llama 3.2-3B, Qwen 3-8B, Qwen3-0.6B. In each experimental run, one of these four models is used as the base for fine-tuning. The models are fine-tuned to act as a consumer providing product ratings. The instruction prompt used for training and inference is:“You are a consumer who rates products based on their attributes. Please provide a rating for the given product information. The score should be an integer between 0 and 100. Do not include any explanations. The rating format must be ‘Rating: X’.” The training procedure consisted of two main phases. The base model is first fine-tuned on a separate, biased dataset of 500 samples. This dataset is used to taught the model an initial, incorrect rating behavior. In each fine-tuning round, candidate datasets are created by mixing data from the true dataset and the noisy dataset with varying proportion.

The sampling temperature for all LLM text generation was fixed at 0.7. Model performance is evaluated after each fine-tuning stage. The fine-tuned model is prompted to generate ratings for all possible product profiles ($4^{3}=64$ profiles). Each profile is rated 5 times, resulting in 320 total ratings. The generated ratings are used to recover the part-worth utilities via an Ordinary Least Squares (OLS) regression. The quality of the fine-tuned model is measured by the average estimation error between these estimated utilities and the ground-truth utilities of the target model.

![Refer to caption](https://arxiv.org/html/2511.12863v1/x13.png)

(a) LOO for Llama 3.2-3B