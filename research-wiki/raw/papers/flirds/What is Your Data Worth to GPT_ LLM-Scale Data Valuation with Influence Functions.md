---
title: "What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions"
source: "https://arxiv.org/html/2405.13954v1"
author:
published:
created: 2026-05-05
description:
tags:
  - "clippings"
---
Sang Keun Choe <sup>1</sup>   Hwijeen Ahn <sup>1†</sup> Juhan Bae <sup>2†</sup> Kewen Zhao <sup>1†</sup>  
Minsoo Kang <sup>3</sup> Youngseog Chung <sup>1</sup> Adithya Pratapa <sup>1</sup> Willie Neiswanger <sup>4</sup>  
Emma Strubell <sup>1</sup> Teruko Mitamura <sup>1</sup> Jeff Schneider <sup>1</sup> Eduard Hovy <sup>1</sup> Roger Grosse <sup>2</sup> Eric Xing <sup>1,5</sup>  
${}^{1}\,$ Carnegie Mellon University ${}^{2}\,$ University of Toronto ${}^{3}\,$ Georgia Tech ${}^{4}\,$ USC ${}^{5}\,$ MBZUAI Lead author: [sangkeuc@andrew.cmu.edu](mailto:sangkeuc@andrew.cmu.edu).$\;\,{}^{\dagger}$ Main contributors.

###### Abstract

Large language models (LLMs) are trained on a vast amount of human-written data, but data providers often remain uncredited. In response to this issue, data valuation (or data attribution <sup>1</sup>), which quantifies the contribution or value of each data to the model output, has been discussed as a potential solution. Nevertheless, applying existing data valuation methods to recent LLMs and their vast training datasets has been largely limited by prohibitive compute and memory costs. In this work, we focus on influence functions, a popular gradient-based data valuation method, and significantly improve its scalability with an efficient gradient projection strategy called LoGra that leverages the gradient structure in backpropagation. We then provide a theoretical motivation of gradient projection approaches to influence functions to promote trust in the data valuation process. Lastly, we lower the barrier to implementing data valuation systems by introducing Logix, a software package that can transform existing training code into data valuation code with minimal effort. In our data valuation experiments, LoGra achieves competitive accuracy against more expensive baselines while showing up to 6,500 $\times$ improvement in throughput and 5 $\times$ reduction in GPU memory usage when applied to Llama3-8B-Instruct and the 1B-token dataset (open source project: [link](https://github.com/logix-project/logix)).

## 1 Introduction

Despite the well-recognized importance of training data in advancing the capabilities of large language models (LLMs) [^7] [^31] [^47], there is no agreed-upon mechanisms for crediting or compensating data providers. As LLMs are increasingly integrated into our society and economy, the absence of such mechanisms has aggravated a tension between data and model providers, exemplified by recent legal challenges involving major tech companies [^29] [^39]. In this atmosphere, data valuation, which quantifies the contribution of each training data to the model output, has been discussed as a potential technical solution for tackling these societal issues [^13] [^15] [^25] [^28] [^56] [^62].

At a high level, most data valuation algorithms interpret the model output as a coalition of its training data, and evaluate the contribution of each example based on its influence on the model output when included or excluded from the training dataset [^15] [^26] [^32] [^35]. If an inclusion of a specific training example consistently improves model performance, high value can be assigned to this example for its contribution. However, applying existing data valuation methods to recent LLMs and their vast training datasets has faced significant scalability challenges to date. For instance, sampling-based methods, such as the Shapley value [^15] [^35] or Datamodels [^26], require retraining the model multiple times with varied combinations of data subsets to directly model the effect of in/excluding each data. Unfortunately, such repeated retraining is hardly affordable even for small models, let alone LLMs. To overcome this issue, gradient-based methods, including influence functions [^32] [^42], approximate the effect of data in/exclusion on the model output using gradient information without costly retraining. Even so, scaling gradient-based methods to LLMs is hindered by prohibitive compute and memory costs originating in the high-dimensional nature of the gradient.

Consequently, the main objective of this work is to bridge the gap in scaling existing data valuation methods to recent LLMs and their vast training datasets. Toward this goal, we focus on influence functions [^32] [^42], a representative gradient-based data valuation method, and significantly improve its scalability with an efficient gradient projection algorithm. We visualize the proposed data valuation system in Figure 1, and detail our technical contributions below:

Figure 1: Data valuation system architecture. (Left Bottom) We first extract the Hessian and gradients for all training data using efficient gradient projection LoGra and store them in a database. (Left Top) At test time, we similarly extract gradients and query the database. (Right) The database returns similarity scores with respect to training examples that can be used for data valuation/attribution.

- Employing gradient structures in backpropagation, we develop a novel low-rank gradient projection algorithm LoGra that improves space & time complexity of gradient projection, a major scalability bottleneck in prior work [^42] [^48], from $O(nk)$ to $O(\sqrt{nk})$ where $n$ and $k$ are model and projection dimensions. Furthermore, LoGra directly computes projected gradients without materializing full gradients, enabling low GPU memory and high GPU utilization for improved efficiency. Lastly, we show that LoGra can be easily implemented with small add-on layers, similarly to LoRA [^24].
- By interpreting a damping term in influence functions as a spectral gradient sparsification mechanism, we (1) offer a theoretical motivation of gradient projection approaches to influence functions and (2) derive a specialized PCA initialization scheme for LoGra.
- We introduce software named Logix that (1) makes it simple to convert existing training code into data valuation code, (2) is compatible with various scalability tools and features in the LLM ecosystem, and (3) is extensible to implement other data valuation or interpretability algorithms.
- In our data valuation experiments, LoGra demonstrates competitive accuracy against more costly baselines, while showing up to 6,500 $\times$ increase in throughput and 5 $\times$ reduction in GPU memory, when applied to Llama3-8B-Instruct [^1] and the 1B-token dataset, compared to EKFAC influence [^17], the state-of-the-art and only runnable baseline at this scale. We also observe that most valuable data identified by LoGra generally share qualitative similarities with the queried LLM output.

## 2 Scalability Bottlenecks in Influence Functions

Most data valuation algorithms (e.g., data Shapley [^15]) evaluate the contribution or value of a specific example $x$ on the utility $v$ (e.g., test loss), that can further be used for crediting data providers, by measuring the overall change in the utility $v$ when in/excluding $x$ as follows:

$$
\displaystyle\textsc{Value}(x;v)=\sum_{S\subseteq D\backslash\{x\}}w\big{(}v(S%
\cup\{x\})-v(S)\big{)}
$$

where $D$ is the training dataset, $S$ is a subset of $D$, and $w$ is an (algorithm-specific) weighting term. Intuitively, the larger the utility gain from an inclusion of $x$ is, the larger the value of $x$ is.

One popular instantiation of Eq. (1) is the leave-one-out error [^32], a semivalue [^10] that is a basis for most data attribution methods and only considers $S$ with $|S|=|D|-1$ (i.e., leaving one example $x$ from the entire dataset $D$). However, naively computing the leave-one-out-error requires retraining the model multiple times for each $x\in D$, which is hardly affordable even in small-scale setups. To overcome this issue, influence functions, a representative gradient-based method, efficiently simulates the effect of model retraining without an example $x_{tr}$ on the utility using gradient information as:

$$
\displaystyle\textsc{Influence}(x_{tr},x_{te})=g_{te}^{\top}H^{-1}g_{tr}
$$

where $g_{tr}$ and $g_{te}$ are train and test gradients respectively, and $H$ is the Hessian matrix. Concretely, influence functions approximate the effect of removing $x_{tr}$ by updating the model parameters with a Newton step in the direction of $H^{-1}g_{tr}$, and uses a first-order Taylor approximation to estimate how this update will affect the test utility. In practice, computing influence functions involves two key steps of (1) solving the inverse Hessian-vector product (iHVP) with $g_{te}$, and (2) taking the dot product of this iHVP with the gradient $g_{tr}$ for each training example.

Despite their comparative efficiency, influence functions remain difficult to scale to recent LLMs due to the high compute and memory costs associated with both steps. First, space and time complexity of naive iHVP are respectively $O(n^{2})$ and $O(n^{3})$, both of which are impractical in recent LLMs with $n>10^{9}$ parameters. To address this issue, various tricks for efficiency, such as iterative methods [^32] or EKFAC approximation [^17], have been proposed. Second, to ensure fair valuation, one must compute influence scores with all training data, which requires access to their gradients. However, computing gradients for all training data approximately amounts to one-epoch training, the cost of which often exceeds $1M in the context of LLM (pre)training. If training gradients were to be recomputed frequently for regular data valuation, the total cost can quickly become astronomical. Thus, while it is technically possible to run a few influence function analyses to interpret interesting LLM outputs using efficient iHVP tricks [^17], doing it in a scalable and sustainable way to build a practical data valuation system remains a significant challenge.

In an attempt to mitigate the aforementioned cost issues, Arnoldi IF [^48] and TRAK [^42] recently explored the strategy of projecting gradients onto a low-dimensional space and computing influence scores on the subspace spanned by the projection matrix as follows:

$$
\displaystyle\textsc{Influence}(x_{tr},x_{te};P)=\big{(}Pg_{te}\big{)}^{\top}%
\big{(}PHP^{\top}\big{)}^{-1}\big{(}Pg_{tr}\big{)}
$$

where $P\in\mathbb{R}^{k\times n}$ is the projection matrix given the model and projection dimensions of $n$ and $k$. Under this strategy, the iHVP operation also occurs in a low-dimensional space, meaning that $n$ in memory and compute complexity of iHVP gets replaced with $k\ll n$. Furthermore, low-rank projection enables writing projected gradients for all training data to disk once and simply reading them as new test data arrives without costly re-computations. This converts an influence function problem into a vector similarity search problem, for which various system optimizations exist [^30].

In essence, this strategy significantly reduces both iHVP and training gradient recomputation costs by introducing an additional process of low-rank gradient projection $Pg$. However, the additional compute/memory costs and accuracy degradation incurred from low-rank gradient projection has not been thoroughly studied to date. First, assuming that the batch size is $b$, the compute cost of naive batched gradient projection is $O(bkn)$. Noting that the compute cost of backpropgation is $O(bn)$ (or $O(btn)$ if we consider the time dimension), the cost of gradient projection is usually larger than that of backpropagation given a reasonably large $k$ for the expressivity. Second, the memory costs for full per-sample gradient and the projection matrix are $O(bn)$ and $O(kn)$. If an 8B model were to be used, each of these costs amounts to 32GB $\times b$ (or $\times k$) GPU memory. While Arnoldi IF and TRAK attempt to address the memory costs of the per-sample gradient and projection matrix respectively with forward-mode Jacobian-vector products and a custom CUDA kernel trick, neither of them are able to solve both issues altogether. This leads Arnoldi IF and TRAK to use very small $k$ and $b$, each of which results in decreased accuracy of influence scores due to limited expressivity and poor efficiency from low GPU utilization. Since accuracy and efficiency are both critical for effective data valuation, we deduce that further advancements in the gradient projection approach are necessary.

## 3 Scaling Data Valuation & Influence Functions

In light of these issues, we first design a memory and compute efficient gradient projection algorithm called LoGra, that leverages the inherent gradient structure in backpropagation (Section 3.1). Then, we provide an intuitive theoretical analysis on why gradient projection approaches work in influence functions (Section 3.2). Finally, we distill our insights obtained from studying (scalable) influence functions into a new open-source software, called Logix, which achieves high compatibility, extensibility, and usability, to facilitate data valuation research (Section 3.3). In this section, we build our arguments at the granularity of each layer (or module) instead of the whole network for clarity.

### 3.1 Algorithm: Memory and Compute Efficient Gradient Projection

Most layers in neural networks, such as linear and convolutional layers, essentially perform matrix multiplication. Given the input $x_{i}\in\mathbb{R}^{n_{i}\times T}$, the output $x_{o}\in\mathbb{R}^{n_{o}\times T}$, the weight $W\in\mathbb{R}^{n_{o}\times n_{i}}$ for the layer, its forward and backward computations can be written as follows:

$$
\displaystyle x_{o}=
$$
 
$$
\displaystyle\,Wx_{i}
$$
 
$$
\displaystyle\text{vec}(\mathcal{D}W)=\sum_{t=1}^{T}x_{i,t}\,\otimes
$$
 
$$
\displaystyle\,\mathcal{D}x_{o,t}\,,\;\;\mathcal{D}x_{i}=W^{\top}\mathcal{D}x_%
{o}
$$

where $T$ denotes for the sequence dimension in language modeling, $\mathcal{D}$ the derivative with respect to the loss, $\otimes$ the Kronecker product, and $\text{vec}(\cdot)$ the vectorization operation. In Eq. (5), we observe that gradient $\text{vec}(\mathcal{D}W)$ obtained during backpropagation is structured as a sum of Kronecker products between forward and backward activations. LoGra leverages this observation to impose an additional Kronecker-product structure on the projection matrix $P$ as follows:

$$
\displaystyle P\text{vec}(\mathcal{D}W)\triangleq(P_{i}\otimes P_{o})\text{vec%
}(\mathcal{D}W)=\sum_{t=1}^{T}(P_{i}\otimes P_{o})(x_{i,t}\otimes\mathcal{D}x_%
{o,t})=\sum_{t=1}^{T}P_{i}x_{i,t}\otimes P_{o}\mathcal{D}x_{o,t}
$$

where $P_{i}\in\mathbb{R}^{k_{i}\times n_{i}}$, $P_{o}\in\mathbb{R}^{k_{o}\times n_{o}}$, and $P=P_{i}\otimes P_{o}$. In Eq. (6), LoGra first projects forward and backward activations onto low-dimensional spaces with $P_{i}$ and $P_{o}$ respectively, and then reconstructs projected gradient directly from these projected activations. This is in contrast to traditional gradient projection [^42], which first computes raw gradient and then projects it onto a low-dimensional space.

Now, we compare memory/compute efficiency of LoGra to that of naive gradient projection, especially under the setting of $n_{i}\approx n_{o}\approx\sqrt{n}$ and $k_{i}\approx k_{o}\approx\sqrt{k}$. First, both memory/compute costs of per-sample gradient computations reduce from $O(bn)$ to $O(bk)$. Second, both memory/compute costs of gradient projection reduce from $O(bnk)$ to $O(b\sqrt{nk})$. To clearly see this benefit, given the model/projection sizes of 8B/4k, we note that projection matrix sizes are about 1GB and 128TB respectively for LoGra and naive projection. As such, while enjoying general efficiency gains from gradient projection we disscussed in Section 2, LoGra further improves the efficiency of per-sample gradient computations significantly at a marginal cost of the additional gradient projection process.

Figure 2: LoGra.

Furthermore, leveraging the fact that projection occurs in the activation space, LoGra can be easily implemented with small add-on layers that are composed of encoder, bottleneck, and decoder, each of which is initialized with $P_{i}$, zero, and $P_{o}$ as shown in Figure 2. If we ignore the bottleneck layer, the overall architecture is identical to the popular LoRA architecture [^24]. While it is intuitive that the roles of encoder and decoder are projecting forward and backward activations respectively, we emphasize two critical roles of the bottleneck layer here. First, its zero initialization ensures that the rest of both forward and backward computations remain unaffected by these add-on layers. Second, per-sample projected gradients can be obtained by simply computing per-sample gradients for the bottleneck layer, using automatic differentiation of an underlying framework without complicated implementation efforts.

### 3.2 Theory: Why Gradient Projection Works in Influence Functions

While LoGra can significantly improve scalability of influence functions, an inherent criticism of any gradient projection approach is that information loss from the projection process may render the resulting influence analysis invalid. Unfortunately, theoretical analyses from prior work [^42] [^48] only discuss the indirect effect of gradient projection on proxy concepts like gradient flow or iHVP variance, which are loosely related to influence functions. To promote trust in the data valuation process, we provide here a mathematical motivation of gradient projection approaches to influence functions. Toward this goal, we interpret a damping term in influence functions that is typically added to ensure the invertibility of the Hessian $H$ as a spectral gradient sparsification mechanism. A formal argument and our derivation are respectively provided in Lemma 1 and in Appendix D.

###### Lemma 1

Let $\{e_{1},\cdots,e_{n}\}$ and $\{\lambda_{1},\cdots,\lambda_{n}\}$ be eigenvectors and eigenvalues of the Hessian $H$. Expressing $g_{tr/te}=\sum_{i}c_{tr/te,i}\cdot(\sqrt{\lambda_{i}}e_{i})$, the following holds under Assumption 1:

$$
\displaystyle\textsc{Influence}(x_{tr},x_{te})=g_{te}^{\top}(H+\lambda I)^{-1}%
g_{tr}=\sum_{i=1}^{n}\frac{\lambda_{i}}{\lambda_{i}+\lambda}c_{tr,i}c_{te,i}\;%
\;\text{and}\;\;\mathbb{E}[c_{\cdot,i}^{2}]\approx 1.
$$

Lemma 1 shows that a damping term softly limits the number of components in influence computations by penalizing contributions from small components. Given the prevalence and practical importance of a damping term in influence functions [^5], we can motivate gradient projection as an alternative way of (hard-)limiting influence computations to components in the projection matrix. To make LoGra similarly penalize small components, we develop an initialization scheme that exploits the Kronecker-Factored Approximate Curvature (KFAC) algorithm [^37]. As a quick overview, KFAC approximates the block-wise Hessian with the Kronecker product of uncentered forward and backward covariances of each layer, respectively denoted with $C_{F}$ and $C_{B}$, as $H\approx H_{KFAC}=C_{F}\otimes C_{B}$. Expressing $C_{F}$ and $C_{B}$ as $Q_{F}\Lambda_{F}Q_{F}^{\top}$ and $Q_{B}\Lambda_{B}Q_{B}^{\top}$ with eigendecomposition, it is easy to show that eigenvectors and eigenvalues of $H_{KFAC}$ are $Q_{F}\otimes Q_{B}$ and $\Lambda_{F}\otimes\Lambda_{B}$. Consequently, we can approximately discard the smaller components of $H$ by initializing $P_{i}$ and $P_{o}$ with $Q_{F}^{1:k_{i}}$ and $Q_{B}^{1:k_{o}}$, where $Q_{\cdot}^{1:k}$ is a collection of top- $k$ eigenvectors (similar to performing PCA on forward and backward activations). In Section 4, we experiment with both PCA and random initialization schemes.

### 3.3 Software: Compatibility, Extensibility, and Usability

Besides algorithmic efficiency, another major bottleneck in the practical adoption of data valuation systems is often the challenge of implementation. In particular, we observe that gradient computation in LLMs, which is a building block for influence functions, typically requires support from other scalability tools like DeepSpeed [^46] or relies on high-level frameworks like HF Transformers [^55]. However, most existing software that can be used for data valuation (e.g., Captum [^33] and TRAK [^42]) is largely incompatible with these tools due to the (too) high level of abstraction in their APIs.

[⬇](data:text/plain;base64,aW1wb3J0IGxvZ2l4CgojIHNldHVwCnJ1biA9IGxvZ2l4LmluaXQocHJvamVjdCwgY29uZmlnKQpydW4uc2V0dXAoInN0YXQiOiAia2ZhYyIsICJzYXZlIjogImdyYWQiKQpydW4ud2F0Y2gobW9kZWwpCgojIHRyYWluIGxvZyAmIHN0YXRpc3RpYwpmb3IgYmF0Y2ggaW4gdHJhaW5fbG9hZGVyOgogIHdpdGggcnVuKGRhdGFfaWQ9YmF0Y2hbImlucHV0X2lkcyJdKToKICAgIGxvc3MgPSBtb2RlbChiYXRjaCkKICAgIGxvc3MuYmFja3dhcmQoKQpydW4uZmluYWxpemUoKQoKIyB0ZXN0IHRpbWUgaW5mbHVlbmNlIGFuYWx5c2lzCndpdGggcnVuKGRhdGFfaWQ9dHN0X2JhdGNoWyJpbnB1dF9pZHMiXSk6CiAgbG9zcyA9IG1vZGVsKHRzdF9iYXRjaCkKICBsb3NzLmJhY2t3YXJkKCkKcnVuLmNvbXB1dGVfaW5mbHVlbmNlX2FsbCgp)

import logix

\# setup

run = logix.init(project, config)

run.setup("stat": "kfac", "save": "grad")

run.watch(model)

\# train log & statistic

for batch in train\_loader:

with run(data\_id=batch\["input\_ids"\]):

loss = model(batch)

loss.backward()

run.finalize()

\# test time influence analysis

with run(data\_id=tst\_batch\["input\_ids"\]):

loss = model(tst\_batch)

loss.backward()

run.compute\_influence\_all()

Figure 3: Code Example of Logix.

Subsequently, we develop a new software package, Logix, design of which enables an easy conversion of users’ existing training code into data valuation code, by promoting compatibility with other tools in the LLM ecosystem. To this end, we first notice that most influence function algorithms simply require collecting train logs (e.g., gradient, activation) and their statistics (e.g., covariance). As a result, given arbitrary users’ training code, data valuation software only need to intercept these logs, and provide basic primitives to compute various statistics with them. Leveraging this observation, Logix implements log interceptions and compute primitives using PyTorch hooks. Notably, the use of hooks makes Logix compatible with diverse other tools as hooks can be seamlessly integrated with most PyTorch features (e.g., FSDP, autocast, compile). In addition, Logix is extensible, as users can easily define and add custom primitives inside hooks. Finally, Logix is easy-to-use as its context manager automatically handles adding appropriate hooks and primitives to relevant modules with minimal code changes. In Appendix E, we provide a more detailed comparison between Logix and other relevant (interpretability) software, and describe notable optimization techniques (e.g., efficient data IO) implemented in it. Code examples can be found in Figure 3, Appendix B, and our project [page](https://github.com/logix-project/logix).

## 4 Experiments

In this section, we evaluate the effectiveness of LoGra in terms of accuracy and efficiency, both of which are important in practical data valuation systems. Specifically, we first perform two types of counterfactual evaluations to quantitatively study data valuation accuracy of LoGra on small-scale setups (Section 4.1). Then, we scale LoGra to LLMs and their massive training data, where we investigate qualitative accuracy (i.e., how similar most valuable training data are to the model output) and memory/compute efficiency (Section 4.2). Finally, our appendix includes more qualitative results of data valuation (Appendix A), pseudo-code for LLM experiments (Appendix B), and experimental details such as hyperperameters and compute resources (Appendix C).

### 4.1 Quantitative Accuracy with Counterfactual Evaluation

(a) Brittleness test

(b) Linear datamodeling score (LDS)

Figure 4: Quantitative accuracy evaluation of data valuation algorithms. We excluded TRAK in the WikiText experiments due to lack of a public implementation for language modeling tasks.

To quantitatively assess accuracy of data valuation algorithms, we adopt two counterfactual evaluation methods: brittleness test [^26] and linear datamodeling score (LDS) [^42]. First, the brittleness test focuses on accuracy in successfully identifying top valuable data. To this end, it first removes the top- $k$ valuable data identified by each algorithm, retrains the model without them multiple times with different random seeds, and measures the overall change in the model output. The larger the output change is, the more accurate the algorithm is in identifying top valuable data. Second, LDS measures general valuation accuracy of all training data under the additivity assumption. Specifically, given multiple data subsets $\{S_{i}\}$ of the fixed size (e.g., $|S_{i}|=|D|/2$), LDS estimates the test performance of the model trained on $S_{i}$ by summing the values of all examples in $S_{i}$ returned by each algorithm, and compares it against the gold performance obtained by actually training the model on $S_{i}$ using the Spearman correlation. Noting that linear datamodels have a connection to the game-theoretic data value (e.g., Shapley value) [^26], LDS can serve as a principled way to study data valuation accuracy.

We perform these counterfactual evaluations on three benchmarks where many rounds (up to 1800) of retraining is feasible: (1) MLP with FMNIST, (2) ResNet-9 [^23] with CIFAR-10, and (3) GPT2 [^44] with WikiText. On these benchmarks, we compare accuracy of LoGra against four popular data valuation baselines, including gradient dot product [^43], TRAK [^42], EKFAC influence [^17], and representation similarity [^22]. With the aim of bearing relevance to a large-scale setting with LLMs and their vast training data, we have only considered baseline methods that satisfy the following two conditions. First, the method cannot retrain the model multiple times for identifying top- $k$ valuable data.<sup>2</sup> Second, the method only has access to the final model checkpoint, which is the case for most LLMs. Given the above setup, we present our experiment results in Figure 4.

We observe that LoGra slightly underperforms EKFAC influence, which is a few orders of magnitude slower in our large-scale experiments (Section 4.2), while noticeably outperforming other baselines. We attribute competitive accuracy of LoGra to two factors. First, unlike TRAK of which projection dimension is limited by the huge projection matrix, LoGra can efficiently afford a higher projection dimension thanks to its sublinear memory/compute costs for gradient projection, and thus achieve the higher expressivity. Second, gradient projection enables LoGra to compute raw projected Fisher information matrix (or Hessian) without an approximation as in EKFAC influence. We expect that a more accurate computation of the Hessian generally leads to more accurate data valuation results.

Comparing the initialization schemes for LoGra (PCA vs. random), we observe that LoGra-PCA outperforms LoGra-random on the FMNIST and CIFAR benchmarks. Hence, we hypothesize that it is generally more accurate to compute influence functions with larger components, similar to the spectral gradient sparsification effect of a damping term we discussed in Section 3.2. To understand a relatively poor performance of LoGra-PCA on WikiText+GPT2, we point out that the Transformer architecture [^52] used in this benchmark lacks the specialized KFAC Hessian approximation, unlike naive MLP [^37] or convolutional [^18] architectures in other benchmarks. Subsequently, our ad-hoc implementation of the PCA initialization based on the naive MLP architecture (i.e., no weight sharing) may not successfully keep larger components of the GPT2 Hessian, failing to deliver its benefit. As a result, we decide to use LoGra-random for our LLM experiments in the next subsection.

### 4.2 Scaling to Billion-Scale Models & Datasets

Given competitive accuracy of LoGra, we now evaluate its practical utility in valuing billion-scale training data for billion-scale models. Specifically, we adopt GPT2-XL (1.5B) [^44], Pythia-1.4B [^6], and Llama3-8B-Instruct [^1] as our models, and conduct data valuation on a random 1B-token subset of the OpenWebText (OWT) dataset [^16]. The major motivations behind choosing OWT as our data valuation dataset are twofold. First, we observe that OWT consists of relatively higher-quality data compared to other LLM training datasets like C4 [^45] or Dolma [^51] while maintaining the diversity unlike other high-quality datasets like WikiText [^38]. Second, we anticipate that OWT largely overlaps with training datasets of all our models. In detail, GPT2-XL is trained on the WebText dataset that shares the same data curation process with OWT, Pythia-1.4B is trained on the Pile dataset [^14] that includes an extension of OWT (i.e., OpenWebText2), and we suppose a majority of OWT would be a part of Llama3’s massive 15T-token pretraining dataset. We also note that our OWT subset size (i.e., 1B tokens) was mainly limited by the available storage, not by compute (see Table 1). If we had access to a storage size of 1PB, performing data valuation with a dataset size of 100B+ tokens would be readily feasible using the same compute resource.

Efficiency. To begin with, we compare memory and compute efficiency of LoGra against EKFAC influence [^17], the state-of-the-art and only algorithm that can run on billion-scale models without CUDA out-of-memory (OOM) errors. Indeed, we confirm that running TRAK or Arnoldi IF with billion-scale models results in CUDA OOM errors even on A100 GPUs with 80GB VRAM due to their gigantic projection matrix sizes. We report GPU memory usage and throughput of both logging (one-time) and influence computation (recurring) phases for the Llama3-8B-Instruct experiment with one A100 GPU and half-precision in Table 1.

<table><tbody><tr><td></td><td colspan="4">Logging (Compute & save Hessian | grad)</td><td colspan="4">Compute Influence (Dot product between test & train grads)</td></tr><tr><td></td><td> Batch</td><td> Throughput</td><td> Memory</td><td> Storage</td><td> Train Batch</td><td> Test Batch</td><td> Throughput</td><td> Memory</td></tr><tr><td>EKFAC</td><td>1</td><td>1740 / 419 <sup>∗</sup></td><td>71 / 80 <sup>∗</sup> GB</td><td>89 GB</td><td>4</td><td>4</td><td>12.2</td><td>75 GB</td></tr><tr><td>LoGra</td><td>1</td><td>3430</td><td>23 GB</td><td>3.5 TB</td><td>256</td><td>4</td><td>1599.6</td><td>14 GB</td></tr><tr><td>LoGra</td><td>16</td><td>4696</td><td>79 GB</td><td>3.5 TB</td><td>256</td><td>256</td><td>79003.9</td><td>15 GB</td></tr></tbody></table>

Table 1: Memory & compute efficiency analyses for LoGra and EKFAC. Throughput is measured as tokens/s for logging and (train, test) pairs/s for influence computations. <sup>∗</sup> EKFAC logging consists of two subphases of KFAC fitting (left of /) and corrected eigenvalue fitting (right of /).

Due to the huge size of raw gradients (e.g., 16GB in fp16 for an 8B model), EKFAC cannot afford storing raw gradients for all training data to disk. As a result, EKFAC needs to recompute all training gradients for each test batch, and thus requires allocating extra GPU memory on model weights and intermediate activations. This largely limits both train/test batch sizes and throughput (12.2 pairs/s), and performing data valuation with EKFAC for 256 test data and 1B-token training data would take 11,300 A100 GPU hours, rendering it hardly usable in most practical setups.

In contrast, with its (efficient) gradient projection, LoGra not only significantly improves compute and memory efficiency, but also avoids training gradient recomputations at the costs of disk space for storing projected training gradients and latency from data IO. Since the storage cost is typically much cheaper than the compute cost <sup>3</sup>, we believe our trade-off offers considerable practical benefits. Furthermore, we can largely hide the data IO cost by overlapping gradient reading/writing processes with other computations. For instance, given the fixed train gradient batch size of 256 (i.e., fixed data loading time), we are able to successfully overlap the process of loading training gradients from disk with influence computations against up to 256 test gradients, and thereby achieve almost 6,500 $\times$ improvement in throughput from EKFAC influence. Noting that our GPU memory usage is far from saturated even with the train/test batch size of 256, we believe that more throughput improvements can be achieved simply by further increasing train/test batch sizes.

Qualitative Accuracy. Next, we analyze qualitative similarities between queried LLM outputs and most valuable data identified by LoGra that can be critical for promoting trust in the data valuation system [^56]. Importantly, we observe that naive influence functions frequently return outlier data with high gradient norms as most valuable data, as also noted in [^4] [^17]. To mitigate this issue, we instead use $l$ -RelatIF, a variant of influence functions that normalizes the original influence score with the self-influence score of each training data to penalize such outlier effects [^4]. Our experimental results are provided in Figure 5 (concise) and in Appendix A (extensive).

(a) Llama3-8B-Instruct

(b) GPT2-XL (1.5B)

(c) Pythia-1.4B

Figure 5: Qualitative accuracy of data valuations with LoGra. Important keywords in each example are manually highlighted for the improved readability. More examples can be found in Appendix A.

We observe that most valuable data identified by LoGra, especially for Llama3-8B-Instruct and GPT2-XL, share qualitative similarities (e.g., semantics, style, token overlaps) with the queried LLM outputs. For instance, given Llama3’s response on the dream manipulation product, LoGra identifies a scientific article that studies manually inducing the lucid dream as most valuable data in Figure 5(a). In Figure 5(b), both the GPT2-XL output and the corresponding most valuable data discuss the need for reducing emissions in the coal industry and its connection to the specific administration. In Figure 5(c), the concept of “lifting barbell or dumbells” appear in the model output and the most valuable data.

However, we also notice several failure cases where the identified most valuable data seemingly do not share qualitative similarities with the LLM output, especially with Pythia-1.4B (Appendix A.3). We here provide three potential explanations on these failing examples based on our experiments. First, attributed data may lack qualitative similarities when the queried LLM output itself is incoherent that its gradient does not encode meaningful information. This aligns with our observation that the failure case occurs more frequently with lower-tier models like Pythia-1.4B whose outputs generally are of lower quality. Second, since we only used a 1B-token subset for data valuation, it is possible that our valuation dataset may lack similar data to some queries. As noted above, our experiment was largely limited by the storage of our cluster (not by the compute), so exploring data valuation on an industry-scale cluster would be interesting future work. Third, we posit that train/test gradients in influence functions may encode diverse information including features that are hardly perceptible to humans [^27]. Therefore, it is possible that attributed data are indeed valuable for increasing the likelihood of the queried output by contributing to these other aspects while sharing little qualitative similarities. A more extensive argument on this final point can be found in Appendix A.3.

## 5 Related Work

Data Valuation. Measuring the value (or contribution) of training data on the model outputs has gained lots of attention recently. Exemplified by Data Shapley [^15], a flurry of prior work [^28] [^35] [^53] proposed exploiting the Shapley value or concepts from cooperative game theory to address the data valuation problem. However, most existing approaches in this line require repeated retraining of the model, a cost of which is hardly affordable even with small models. In addition to game-theoretic approaches, data valuation has also been tackled using reinforcement learning [^61], meta learning [^9], and training-free methods [^41] [^57]. Nevertheless, these works either suffer from high complexity from the need to train other models [^9] [^61] or high computational costs [^41]. We direct readers to Sim et al. [^50] for a more extensive survey on diverse data valuation approaches.

Influence Functions. Influence functions, a classic concept from robust statistics [^20], estimate the infinitesimal effect of removing or adding a training data point without model retraining. They have various applications in machine learning, such as interpreting the model’s behavior [^21] [^42] [^17] and curating training datasets [^36] [^11]. However, when applied to large neural networks, the computation of the iHVP and its dot product with all training examples introduce scalability challenges. Besides gradient projection, past works have explored computing influence functions only on the last (few) layers [^32] [^48] to mitigate these challenges. However, subsequent works [^12] [^17] have shown that the influence on only a subset of layers is insufficient to capture the overall influence of a training data point. To avoid computing the gradient of all training examples, various filtering strategies, such as using the similarity in the model’s representation space [^19] or TF-IDF [^60] [^17], have also been proposed. While it is possible to adopt these filtering strategies for LoGra, they may introduce bias in the selection of the most influential sequences. For example, filtering candidate training sequences with TF-IDF might miss interesting influential sequences that do not share many tokens but are semantically related. Recently, similarly to LoGra, DataInf [^34] and LESS [^59] proposed using LoRA to efficiently compute influence functions. However, these approaches are only applicable in finetuning settings, whereas LoGra also supports influence analyses for pretraining.

## 6 Conclusion

In this work, we explored scaling data valuations with influence functions to billion-scale models and datasets as a potential technical solution to properly credit or compensate data providers for training LLMs. Toward this goal, we developed a novel gradient projection algorithm that can significantly improve the scalability of influence functions, and designed a simple and interoperable software. Our experiments showed that LoGra achieves competitive accuracy to other more expensive baselines on counterfactual evaluations, while efficiently scaling to billion-scale models and datasets, thereby demonstrating the initial potential of the practical data valuation system. Last but not least, we discuss broader impacts and limitations of our work in Appendix F.

## Acknowledgement

We thank Daphne Ippolito, Shaily Bhatt, and Yongchan Kwon for providing insightful feedback in preparing the early version of the manuscript, and Jared Fernandez for the useful discussion on the LLM data valuation experiment. We acknowledge the CMU Babel cluster and its administrators for reliably providing necessary compute resources for this work.

## References

## Appendix A Qualitative Analysis

In this section, we provide more qualitative analyses on most valuable data identified by LoGra. In particular, we show top-2 $\sim$ 4 valuable data for each query here, given the possibility that the model utilizes information from multiple training examples for generating its output. We also include several failure cases where identified most valuable data do not share qualitative similarities with the queried LLM output.

### A.1 Llama3-8B-Instruct

#### A.1.1 Example 1

Figure 6: Llama3-8B-Instruct data valuation result.

#### A.1.2 Example 2

Figure 7: Llama3-8B-Instruct data valuation result.

#### A.1.3 Example 3

Figure 8: Llama3-8B-Instruct data valuation result.

#### A.1.4 Example 4

Figure 9: Llama3-8B-Instruct data valuation result.

#### A.1.5 Example 5

Figure 10: Llama3-8B-Instruct data valuation result. LoGra identifies novel literature as most valuable data.

#### A.1.6 Example 6 (Failure)

Figure 11: Llama3-8B-Instruct data valuation result.

### A.2 GPT2-XL

#### A.2.1 Example 1

Figure 12: GPT2-XL data valuation result.

#### A.2.2 Example 2

Figure 13: GPT2-XL data valuation result.

#### A.2.3 Example 3

Figure 14: GPT2-XL data valuation result.

#### A.2.4 Example 4

Figure 15: GPT2-XL data valuation result.

#### A.2.5 Example 5 (Failure)

Figure 16: GPT2-XL data valuation result.

### A.3 Pythia-1.4B (with many failure cases)

While a majority of experiments with Llama3-8B-Instruct and GPT2-XL returned semantically or stylistically similar texts as most valuable data, we observed that the quality of most valuable data from Pythia-1.4B experiments are generally much poorer. Here, we provide one hypothesis behind this observation. Influence functions tend to give a high score for the example that contributes most to decreasing (test) loss at the current weight [^3]. At the same time, it is also hypothesized that different layers learn different concepts at different stages of training [^8]. Combining these two facts, when interpreting influence analysis results, we need to think about which features the model is most likely learning at the current weight. Here, we specifically discuss two factors: training data quality and training steps. First, if the training data quality is low, then there would be a lot of features (e.g., random email address) that are frequent enough in the training dataset to be considered as learnable patterns. In other words, even though these features look redundant to humans, they may still be useful for decreasing loss from the model perspective. Second, many LLMs are only pretrained for a single epoch, or under-trained to their pretraining dataset. That being said, redundant features from the first point would likely still remain as learning-worthy features at the end of training and are captured by influence functions. In sum, we hypothesize that as the model is well-trained on a high-quality dataset, influence functions would capture more similar data to the query LLM output. This hypothesis may also explain the observation from Grosse et al. [^17] that most valuable data identified by influence functions on larger models tend to share more semantic similarity with results on smaller models, noting that larger models tend to converge faster to the point where they can only further decrease loss by learning high-level features. With this, we present our experiment results with Pythia-1.4B below. (some of them are not totally bad, but mostly lack specificity to be considered as “most” valuable data to humans)

#### A.3.1 Example 1

Figure 17: Pythia-1.4B data valuation result. LoGra captures the broad topic of soccer but lacks the specificity (except for the third most valuable data, which states that Christiano Ronaldo is the best soccer play who won the Ballon d’Or award).

#### A.3.2 Example 2

Figure 18: Pythia-1.4B data valuation result. We suspect that the random url in the model output dominates the query gradient and affects the data valuation result.

#### A.3.3 Example 3

Figure 19: Pythia-1.4B data valuation result.

#### A.3.4 Example 4

Figure 20: Pythia-1.4B data valuation result.

## Appendix B Code Examples

We provide a simplified code for our language modeling experiment from Section 4.2 to demonstrate usability of Logix. Logix is open-sourced under Apache 2.0 license [here](https://github.com/logix-project/logix).

### B.1 Log Extraction

[⬇](data:text/plain;base64,aW1wb3J0IGxvZ2l4CmZyb20gbG9naXguc3RhdGlzdGljIGltcG9ydCBDb3ZhcmlhbmNlCgptb2RlbCwgdG9rZW5pemVyLCB0cmFpbl9sb2FkZXIgPSBzZXR1cCgpCgojIEluaXRpYWxpemUgTG9nSVgKcnVuID0gbG9naXguaW5pdChwcm9qZWN0PSJsbG0iLCBjb25maWc9ImNvbmZpZy55YW1sIikKCiMgUmVnaXN0ZXIgdGhlIG1vZGVsCnJ1bi53YXRjaChtb2RlbCwgdHlwZV9maWx0ZXI9W25uLkxpbmVhcl0sIG5hbWVfZmlsdGVyPVsibWxwIl0pCgojIEFkZCBMb0dyYQpydW4uYWRkX2xvcmEoKQoKIyBTZXR1cCBsb2dnaW5nCnJ1bi5zZXR1cCgibG9nIjogImdyYWQiLCAic2F2ZSI6ICJncmFkIiwgInN0YXRpc3RpYyI6IHsiZ3JhZCI6IENvdmFyaWFuY2V9KQoKIyBTdGFydCBsb2dnaW5nCmZvciBiYXRjaCBpbiB0cmFpbl9sb2FkZXI6CiAgICBkYXRhX2lkID0gdG9rZW5pemVyLmJhdGNoX2RlY29kZShiYXRjaFsiaW5wdXRfaWRzIl0pCiAgICB0YXJnZXRzID0gYmF0Y2gucG9wKCJsYWJlbHMiKQogICAgd2l0aCBydW4oZGF0YV9pZD1kYXRhX2lkLCBtYXNrPWJhdGNoWyJhdHRlbnRpb25fbWFzayJdKToKICAgICAgICAjIFVzZXIncyBleGlzdGluZyB0cmFpbmluZyBjb2RlCiAgICAgICAgbW9kZWwuemVyb19ncmFkKCkKICAgICAgICBsbV9sb2dpdHMgPSBtb2RlbCgqKmJhdGNoKQogICAgICAgIHNoaWZ0X2xvZ2l0cyA9IGxtX2xvZ2l0c1suLi4sIDotMSwgOl0uY29udGlndW91cygpCiAgICAgICAgc2hpZnRfbGFiZWxzID0gdGFyZ2V0c1suLi4sIDE6XS5jb250aWd1b3VzKCkKICAgICAgICBsb3NzID0gRi5jcm9zc19lbnRyb3B5KHNoaWZ0X2xvZ2l0cy52aWV3KC0xLCBzaGlmdF9sb2dpdHMuc2l6ZSgtMSkpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgc2hpZnRfbGFiZWxzLnZpZXcoLTEpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVkdWN0aW9uPSJzdW0iLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaWdub3JlX2luZGV4PS0xMDApCiAgICAgICAgbG9zcy5iYWNrd2FyZCgpCgojIEZpbmFsaXplIGxvZ2dpbmcKbG9naXguZmluYWxpemUoKQ==)

import logix

from logix.statistic import Covariance

model, tokenizer, train\_loader = setup()

\# Initialize LogIX

run = logix.init(project="llm", config="config.yaml")

run.watch(model, type\_filter=\[nn.Linear\], name\_filter=\["mlp"\])

\# Add LoGra

run.add\_lora()

\# Setup logging

run.setup("log": "grad", "save": "grad", "statistic": {"grad": Covariance})

\# Start logging

for batch in train\_loader:

data\_id = tokenizer.batch\_decode(batch\["input\_ids"\])

targets = batch.pop("labels")

with run(data\_id=data\_id, mask=batch\["attention\_mask"\]):

\# User’s existing training code

model.zero\_grad()

lm\_logits = model(\*\*batch)

shift\_logits = lm\_logits\[...,:-1,:\].contiguous()

shift\_labels = targets\[..., 1:\].contiguous()

loss = F.cross\_entropy(shift\_logits.view(-1, shift\_logits.size(-1)),

shift\_labels.view(-1),

reduction="sum",

ignore\_index=-100)

loss.backward()

\# Finalize logging

logix.finalize()

### B.2 Influence Computation

[⬇](data:text/plain;base64,aW1wb3J0IGxvZ2l4Cgptb2RlbCwgdG9rZW5pemVyLCB0ZXN0X2xvYWRlciA9IHNldHVwKCkKCnJ1biA9IGxvZ2l4LmluaXQocHJvamVjdD0ibGxtIiwgY29uZmlnPSJjb25maWcueWFtbCIpCnJ1bi53YXRjaChtb2RlbCwgdHlwZV9maWx0ZXI9W25uLkxpbmVhcl0sIG5hbWVfZmlsdGVyPVsibWxwIl0pCgojIExvYWQgc2F2ZWQgbG9ncyAoZS5nLiB0cmFpbiBncmFkaWVudCAmIEhlc3NpYW4pCmxvZ2l4LmluaXRpYWxpemVfZnJvbV9sb2coKQpsb2dfbG9hZGVyID0gbG9naXguYnVpbGRfbG9nX2RhdGFsb2FkZXIoYmF0Y2hfc2l6ZT02NCkKCmxvZ2l4LnNldHVwKHsibG9nIjogImdyYWQifSkKZm9yIGJhdGNoIGluIHRlc3RfbG9hZGVyOgogICAgZGF0YV9pZCA9IHRva2VuaXplci5iYXRjaF9kZWNvZGUoYmF0Y2hbImlucHV0X2lkcyJdKQogICAgdGFyZ2V0cyA9IGJhdGNoLnBvcCgibGFiZWxzIikKICAgIHdpdGggcnVuKGRhdGFfaWQ9ZGF0YV9pZCwgbWFzaz1iYXRjaFsiYXR0ZW50aW9uX21hc2siXSk6CiAgICAgICAgbW9kZWwuemVyb19ncmFkKCkKICAgICAgICBsbV9sb2dpdHMgPSBtb2RlbCgqKmJhdGNoKQogICAgICAgIHNoaWZ0X2xvZ2l0cyA9IGxtX2xvZ2l0c1suLi4sIDotMSwgOl0uY29udGlndW91cygpCiAgICAgICAgc2hpZnRfbGFiZWxzID0gdGFyZ2V0c1suLi4sIDE6XS5jb250aWd1b3VzKCkKICAgICAgICBsb3NzID0gRi5jcm9zc19lbnRyb3B5KHNoaWZ0X2xvZ2l0cy52aWV3KC0xLCBzaGlmdF9sb2dpdHMuc2l6ZSgtMSkpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgc2hpZnRfbGFiZWxzLnZpZXcoLTEpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVkdWN0aW9uPSJzdW0iLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaWdub3JlX2luZGV4PS0xMDApCiAgICAgICAgbG9zcy5iYWNrd2FyZCgpCgogICAgIyBHZXQgdGhlIChncmFkaWVudCkgbG9nIGZvciB0aGUgY3VycmVudCB0ZXN0IGJhdGNoCiAgICB0ZXN0X2xvZyA9IHJ1bi5nZXRfbG9nKCkKCiAgICAjIENvbXB1dGUgaW5mbHVlbmNlIHNjb3JlcyAod2l0aCBsLVJlYWx0SUYpCiAgICBpbmZsdWVuY2Vfc2NvcmVzID0gcnVuLmNvbXB1dGVfaW5mbHVlbmNlX2FsbCh0ZXN0X2xvZywgbG9nX2xvYWRlciwgbW9kZT0iY29zaW5lIikK)

import logix

model, tokenizer, test\_loader = setup()

run = logix.init(project="llm", config="config.yaml")

run.watch(model, type\_filter=\[nn.Linear\], name\_filter=\["mlp"\])

\# Load saved logs (e.g. train gradient & Hessian)

logix.initialize\_from\_log()

log\_loader = logix.build\_log\_dataloader(batch\_size=64)

logix.setup({"log": "grad"})

for batch in test\_loader:

data\_id = tokenizer.batch\_decode(batch\["input\_ids"\])

targets = batch.pop("labels")

with run(data\_id=data\_id, mask=batch\["attention\_mask"\]):

model.zero\_grad()

lm\_logits = model(\*\*batch)

shift\_logits = lm\_logits\[...,:-1,:\].contiguous()

shift\_labels = targets\[..., 1:\].contiguous()

loss = F.cross\_entropy(shift\_logits.view(-1, shift\_logits.size(-1)),

shift\_labels.view(-1),

reduction="sum",

ignore\_index=-100)

loss.backward()

\# Get the (gradient) log for the current test batch

test\_log = run.get\_log()

\# Compute influence scores (with l-RealtIF)

influence\_scores = run.compute\_influence\_all(test\_log, log\_loader, mode="cosine")

## Appendix C Experiment Details

For EKFAC influence [^17] and LoGra, we set the damping term in influence functions as $0.1\times$ mean(eigenvalues) for all layers following the practice in Grosse et al. [^17].

### C.1 Quantitative Counterfactual Experiments

For all our quantitative counterfactual experiments, we project gradients onto a low-dimensional space using LoGra with $k_{i}=k_{o}=128$. We used the same experimental setup, including the configurations for the baseline data valuation techniques, from Park et al. [^42] and Bae et al. [^2]. We used one A100 GPU with 80GB VRAM for all our counterfactual evaluation experiments. For model training, we used hyperparameters in Table 2 for each experiment.

|  | FMNIST | CIFAR-10 | WikiText |
| --- | --- | --- | --- |
| Model | 3-layer MLP | ResNet-9 | GPT2 |
| Optimizer | SGD-M | SGD-M | AdamW |
| LR Scheduler | None | Cyclic | None |
| Learning Rate | 3e-2 | 4e-1 | 3e-5 |
| Weight Decay | 1e-3 | 1e-3 | 1e-2 |
| Batch Size | 64 | 512 | 8 |
| Sequence Length | N/A | N/A | 512 |
| Epochs | 20 | 25 | 3 |

Table 2: Hyperparameter used in experiments in Section 4

##### Brittleness Test.

For classification tasks, we first selected 100 correctly classified test examples when the model is trained on the full dataset (across all 5 random seeds). Then, for each test example $x_{te}$, we identified the top- $k$ influential data points using the data valuation algorithm, removed these training data points, retrained the model, and examined if this removal causes misclassification of $x_{te}$ on average (across 3 random seeds). In Figure 4, we reported the fraction of test examples (out of 100) that get misclassified after removing at most $k$ training data points. For the language modeling task, we selected the 50 test sequences, obtained the top influential training sequences using the data valuation method, and reported the mean test perplexity after removing the top- $k$ influential sequences and retraining the model.

##### Linear Datamodeling Score (LDS).

We measured LDS by generating 100 data subsets of size $|S_{i}|=|D|/2$. For each data subset, we retrained the model 10 times for FashionMNIST, 20 times for CIFAR-10, and $5$ times for WikiText to construct the ground truth. The LDS results in Figure 4 show the mean and standard deviation of LDS obtained from 5 distinctly trained models. A more detailed description of the LDS evaluation can be found in Park et al. [^42].

### C.2 Scaling to Billion-Scale Models and Datasets

We used up to 4 A100 GPUs with 80GB VRAM for these experiments. To save the storage cost, we used $k_{i}=k_{o}=64$ for gradient projection in this experiment. Unlike counterfactual evaluations, as our LLM experiments do not require any retraining, there are no other noticeable hyperparameters to report. We used tf32 precision in all our LLM experiments to prevent gradient quality degradation.

## Appendix D Derivation of Lemma 1

###### Assumption 1

In this work, we make the following two assumptions on train & test gradient distributions and the Hessian $H$:

1\. Given that language modeling falls under the maximum likelihood framework, we replace the Hessian $H$ with the Fisher Information Matrix (FIM), and further approximate the FIM with the empirical FIM, i.e.,

$$
\displaystyle H
$$
 
$$
\displaystyle=\mathbb{E}_{p_{\theta}(y|x)}\big{[}\nabla\log p_{\theta}(y|x)%
\nabla\log p_{\theta}(y|x)^{\top}\big{]}
$$
 
$$
\displaystyle\approx\frac{1}{N}\sum_{(x_{n},y_{n})\in D_{tr}}\big{[}\nabla\log
p%
_{\theta}(y_{n}|x_{n})\nabla\log p_{\theta}(y_{n}|x_{n})^{\top}\big{]}
$$

2\. Given that test data are directly sampled from the model given the prompts, we assume test gradients $g_{te}$ and train gradients $g_{tr}$ approximately follow the same distribution.

Lemma 1 Let $\{e_{1},\cdots,e_{n}\}$ and $\{\lambda_{1},\cdots,\lambda_{n}\}$ be eigenvectors and eigenvalues of the Hessian $H$. With Assumption 1 and $g_{tr/te}=\sum_{i}c_{tr/te,i}\cdot(\sqrt{\lambda_{i}}e_{i})$, the following holds:

$$
\textsc{IF}(x_{tr},x_{te})=g_{te}^{\top}(H+\lambda I)^{-1}g_{tr}=\sum_{i=1}^{n%
}\frac{\lambda_{i}}{\lambda_{i}+\lambda}c_{tr,i}c_{te,i}\;\;\text{and}\;\;%
\mathbb{E}[c_{\cdot,i}^{2}]\approx 1.
$$

Proof.

Let $Q=[e_{1},\cdots,e_{n}]$ and $\Lambda=diag(\lambda_{1},\cdots,\lambda_{n})$.

$$
\displaystyle\textsc{IF}(x_{tr},x_{te})
$$
 
$$
\displaystyle=g_{te}^{\top}(H+\lambda I)^{-1}g_{tr}
$$
 
$$
\displaystyle=g_{te}^{\top}(Q\Lambda Q^{\top}+\lambda I)^{-1}g_{tr}
$$
 
$$
\displaystyle=g_{te}^{\top}\big{(}Q(\Lambda+\lambda I)Q^{\top}\big{)}^{-1}g_{tr}
$$
 
$$
\displaystyle=g_{te}^{\top}Q(\Lambda+\lambda I)^{-1}Q^{\top}g_{tr}
$$
 
$$
\displaystyle=\Bigg{(}\sum_{i}c_{te,i}\cdot(\sqrt{\lambda_{i}}e_{i})\Bigg{)}^{%
\top}Q(\Lambda+\lambda I)^{-1}Q^{\top}\Bigg{(}\sum_{i}c_{tr,i}\cdot(\sqrt{%
\lambda_{i}}e_{i})\Bigg{)}
$$
 
$$
\displaystyle=\big{[}c_{te,1}\sqrt{\lambda_{1}};\cdots;c_{te,n}\sqrt{\lambda_{%
n}}\big{]}^{\top}(\Lambda+\lambda I)^{-1}\big{[}c_{tr,1}\sqrt{\lambda_{1}};%
\cdots;c_{tr,n}\sqrt{\lambda_{n}}\big{]}
$$
 
$$
\displaystyle=\sum_{i=1}^{n}\frac{\lambda_{i}}{\lambda_{i}+\lambda}c_{tr,i}c_{%
te,i}
$$
 
$$
\displaystyle\square
$$

Since we assume $g_{te}$ and $g_{tr}$ follow the same distribution, we need to show $\mathbb{E}[c_{tr,i}^{2}]\approx 1$ for all $i$.

$$
\displaystyle\Lambda
$$
 
$$
\displaystyle=Q^{\top}Q\Lambda Q^{\top}Q
$$
 
$$
\displaystyle=Q^{\top}HQ
$$
 
$$
\displaystyle\approx\frac{1}{N}\sum_{(x_{i},y_{i})\in D_{tr}}Q^{\top}\big{[}%
\nabla\log p_{\theta}(y_{n}|x_{n})\nabla\log p_{\theta}(y_{n}|x_{n})^{\top}%
\big{]}Q\quad(\text{Assumption 1})
$$
 
$$
\displaystyle=\mathbb{E}\big{[}Q^{\top}g_{tr}g_{tr}^{\top}Q\big{]}
$$
 
$$
\displaystyle=\mathbb{E}\Bigg{[}Q^{\top}\Bigg{(}\sum_{i}c_{tr,i}\cdot(\sqrt{%
\lambda_{i}}e_{i})\Bigg{)}\Bigg{(}\sum_{i}c_{tr,i}\cdot(\sqrt{\lambda_{i}}e_{i%
})\Bigg{)}^{\top}Q\Bigg{]}
$$
 
$$
\displaystyle=\mathbb{E}\Big{[}\big{[}c_{tr,1}\sqrt{\lambda_{1}};\cdots;c_{tr,%
n}\sqrt{\lambda_{n}}\big{]}\big{[}c_{tr,1}\sqrt{\lambda_{1}};\cdots;c_{tr,n}%
\sqrt{\lambda_{n}}\big{]}^{\top}\Big{]}
$$

Inspecting diagonal terms, we get $\lambda_{i}\approx\mathbb{E}[c_{tr,i}^{2}\lambda_{i}]=\mathbb{E}[c_{tr,i}^{2}]%
\lambda_{i}$.  
Therefore, $\mathbb{E}[c_{tr,i}^{2}]\approx 1$. $\square$

## Appendix E Logix Details

In this section, we discuss several key differences between Logix and other interpretability tools, and optimizations we implemented in Logix.

### E.1 Differences with Other Tools

Influence functions have been extensively studied as an interpretable AI method. Accordingly, there have been several tools originating in the AI interpretability field that implement influence functions, with most notable examples including Captum [^33], TRAK [^42], and Kronfluence [^17]. Overall, the software design of these tools aim at easing the from-scratch implementation of influence functions by introducing a lot of abstraction, following the philosophy of high-level frameworks. In fact, such software designs were well-received in the pre-LLM era. Nonetheless, as scaling has become a key aspect of AI research, the (LLM) development ecosystem has become complicated and being able to compatibly work with other tools in the ecosystem has become a core aspect in the ML software design. Hence, unlike existing software, the design of Logix aims at enabling the easy conversion of users’ (already efficient) training codes into data valuation codes. This design is also motivated by the observation that gradient is simply a by-product of the training procedure so that we can reuse most of the training code for data valuation without needing to write the gradient computation code from scratch as in other tools.

Recently, there have been active developments in (mechanistic) interpretability software, represented by TransformerLens [^40] and pyvene [^58]. Interestingly, these software also extensively use PyTorch hooks, similarly to Logix, probably due to its high compatibility with other features such as autocast, distributed data parallelism, fully-sharded data parallelism, and gradient checkpointing. Nevertheless, we point out two major differences between these (mechanistic) interpretability software and Logix. First, support for dataset-level statistics computations in Logix is largely missing in these tools. In data valuation, we often need to compute several dataset-level statistics such as the Hessian (or Fisher information matrix) for accurate influence computations, and therby supporting these computations seamlessly was an important design principle behind Logix. However, analyses in (mechanistic) interpretability research typically focuses on each instance and computing dataset-level statistics is typically not supported. Second, support for efficient data IO in Logix is not a priority in other tools. As we propose to convert the data valuation problem into a vector similarity search problem with gradient projection, we put efforts into improving efficiency of data IO (see the next subsection for details), whereas this issue is rarely considered in other interpretability tools. We hope to explore the possibility of supporting both data valuation and other interpretability research in a unified way with Logix as our future work.

### E.2 Optimizations

Efficient Data IO With LoGra, we propose to save projected gradients for all training data to disk, and frequently load them as a new test batch arrives. As a result, reducing latency from data IO renders to be critical in realizing efficient data valuation. In particular, as the total size of all training gradients is usually far beyond the limit of CPU memory, we should optimize data transfer between disk and CPU (or GPU). To address this issue, we adopted the memory-mapped files that bypasses the need for intermediate copying between kernel space and user space, reducing the overhead associated with data IO operations. The use of the memory-mapped files is also motivated by the observation that, given each query batch, data valuation often requires computing influence scores with all training data. Therefore, we can access training gradients in a predefined or sequential order instead of in a random order, which can be done efficiently with memory-mapped files (sequential access is faster than random access).

Moreover, we overlap memory-mapped-file-based data IO with computations to further enhance data valuation efficiency. In the logging phase, we overlap the process of saving gradients extracted from the current training batch to disk with computations for the next training batch using Python multiprocessing. In the influence computation phase, we overlap the process of loading saved training gradients from disk with computing a dot product with the query batch using the pre-fetching feature of PyTorch DataLoader.

We also note that more efficient data IO can be achieved by the use of more advanced techniques like GPU-accelerated vector database, especially in the production setting. While we considered supporting this feature, we decided to focus on the memory-mapped-file-based data IO in our initial version of Logix, as it offers more flexibility to explore different algorithms in the research setting.

Memory Optimization When dealing with LLMs, GPU memory is often a major scaling bottleneck. To alleviate this issue, we support CPU offloading of dataset-level statistics by utilizing the sequential nature of backpropgation. When this feature is enabled, we by default keep all dataset-level statistics (e.g., gradient covariance) on CPU, move it to GPU when the corresponding module is called during forward/backward passes, and then move it back to CPU asynchronously as soon as updating statistics for the module is done. Depending on the CPU-GPU communication bandwidth, this feature may slow down the logging process.

Communication Optimization If training data are split across multiple processes with distributed training, we need to aggregate dataset-level statistics across processes for consistency. To minimize the communication cost, we delay the synchronization process until the training loop (one epoch) is over, and perform synchronization only once at the end. Following the similar logic, users can maximize the efficiency of the logging phase by disabling gradient synchronization (e.g., torch.no\_sync).

## Appendix F Broader Impacts & Limitations

### F.1 Broader Impacts

The data valuation problem can be a socially sensitive topic. As of now, we do not have the agreed-upon social norm for data valuation, and thus we refrained from discussing how exact data values should be determined based on our method. Rather, our work is an initial attempt to tackle the technical challenges in enabling LLM-scale data valuation. For equitable data valuation, we believe future research for improving both accuracy and efficiency of data valuation systems along with extensive social discussions are necessary.

### F.2 Limitations & Future Work

We generally observed that influence function approaches are susceptible to outlier data with large gradient norms. This outlier issue is particularly severe for language modeling tasks due to the fact that the gradient of each sequence is the sum of gradients for all tokens in that sequence. If a few tokens in the sequence have large gradient norms, their gradients may dominate the total gradient for the sequence and hurt data valuation accuracy. While our work tried to reduce the outlier effect with (self-influence) normalization, exploring other filtering heuristics (e.g., $L_{2}/L_{1}$ norm ratio [^17]) may be an interesting research direction.

We attempted to lay the software foundation for data valuation with Logix, but did not implement extensive system support, such as high-performance vector database (e.g., Faiss [^30]). We expect further system optimizations would enable significantly more efficient data valuation. To reduce the cost of influence functions, our work mostly explored low-rank gradient projection, which compresses the gradient in a spectral domain in essence. Noting that gradient compression has been extensively studied in the efficient distributed training literature, it is worth exploring (or combining) different gradient compression strategies, e.g., top- $k$ compression [^49] or low-bit compression [^54], to further reduce the compute/memory/storage costs for influence functions.

[^1]: AI@Meta. Llama 3 model card, 2024.

[^2]: Juhan Bae, Wu Lin, Jonathan Lorraine, and Roger Grosse. Training data attribution via approximate unrolled differentiation, 2024.

[^3]: Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi, and Roger B Grosse. If influence functions are the answer, then what is the question? Advances in Neural Information Processing Systems, 35:17953–17967, 2022.

[^4]: Elnaz Barshan, Marc-Etienne Brunet, and Gintare Karolina Dziugaite. Relatif: Identifying explanatory training samples via relative influence. In International Conference on Artificial Intelligence and Statistics, pages 1899–1909. PMLR, 2020.

[^5]: Samyadeep Basu, Philip Pope, and Soheil Feizi. Influence functions in deep learning are fragile. arXiv preprint arXiv:2006.14651, 2020.

[^6]: Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pages 2397–2430. PMLR, 2023.

[^7]: Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

[^8]: Yixiong Chen, Alan Yuille, and Zongwei Zhou. Which layer is learning faster? a systematic exploration of layer-wise convergence rate for deep neural networks. In The Eleventh International Conference on Learning Representations, 2023.

[^9]: Sang Choe, Sanket Vaibhav Mehta, Hwijeen Ahn, Willie Neiswanger, Pengtao Xie, Emma Strubell, and Eric Xing. Making scalable meta learning practical. Advances in neural information processing systems, 36, 2024.

[^10]: Pradeep Dubey, Abraham Neyman, and Robert James Weber. Value theory without efficiency. Mathematics of Operations Research, 6(1):122–128, 1981.

[^11]: Logan Engstrom, Axel Feldmann, and Aleksander Madry. Dsdm: Model-aware dataset selection with datamodels. arXiv preprint arXiv:2401.12926, 2024.

[^12]: Vitaly Feldman and Chiyuan Zhang. What neural networks memorize and why: Discovering the long tail via influence estimation. Advances in Neural Information Processing Systems, 33:2881–2891, 2020.

[^13]: Raul Castro Fernandez. Data-sharing markets: Model, protocol, and algorithms to incentivize the formation of data-sharing consortia. In Proceedings ACMSIGMOD International Conference on Management of Data, 2023.

[^14]: Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

[^15]: Amirata Ghorbani and James Zou. Data shapley: Equitable valuation of data for machine learning. In International conference on machine learning, pages 2242–2251. PMLR, 2019.

[^16]: Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. Openwebtext corpus, 2019.

[^17]: Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, Evan Hubinger, Kamilė Lukošiūtė, Karina Nguyen, Nicholas Joseph, Sam McCandlish, Jared Kaplan, and Samuel R. Bowman. Studying large language model generalization with influence functions, 2023.

[^18]: Roger Grosse and James Martens. A kronecker-factored approximate fisher matrix for convolution layers. In International Conference on Machine Learning, pages 573–582. PMLR, 2016.

[^19]: Han Guo, Nazneen Fatema Rajani, Peter Hase, Mohit Bansal, and Caiming Xiong. Fastif: Scalable influence functions for efficient model interpretation and debugging. arXiv preprint arXiv:2012.15781, 2020.

[^20]: Frank R Hampel. The influence curve and its role in robust estimation. Journal of the american statistical association, 69(346):383–393, 1974.

[^21]: Xiaochuang Han, Byron C Wallace, and Yulia Tsvetkov. Explaining black box predictions and unveiling data artifacts through influence functions. arXiv preprint arXiv:2005.06676, 2020.

[^22]: Kazuaki Hanawa, Sho Yokoi, Satoshi Hara, and Kentaro Inui. Evaluation of similarity-based explanations. arXiv preprint arXiv:2006.04528, 2020.

[^23]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[^24]: Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2021.

[^25]: Jie Huang and Kevin Chen-Chuan Chang. Citation: A key to building responsible and accountable large language models. arXiv preprint arXiv:2307.02185, 2023.

[^26]: Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. Datamodels: Predicting predictions from training data, 2022.

[^27]: Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan Engstrom, Brandon Tran, and Aleksander Madry. Adversarial examples are not bugs, they are features. Advances in neural information processing systems, 32, 2019.

[^28]: Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nick Hynes, Nezihe Merve Gürel, Bo Li, Ce Zhang, Dawn Song, and Costas J Spanos. Towards efficient data valuation based on the shapley value. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 1167–1176. PMLR, 2019.

[^29]: J.L. et al. v. Alphabet Inc. Case 3:23-cv-03416, N.D. Cal., 2023.

[^30]: Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 7(3):535–547, 2019.

[^31]: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020.

[^32]: Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International conference on machine learning, pages 1885–1894. PMLR, 2017.

[^33]: Narine Kokhlikyan, Vivek Miglani, Miguel Martin, Edward Wang, Bilal Alsallakh, Jonathan Reynolds, Alexander Melnikov, Natalia Kliushkina, Carlos Araya, Siqi Yan, et al. Captum: A unified and generic model interpretability library for pytorch. arXiv preprint arXiv:2009.07896, 2020.

[^34]: Yongchan Kwon, Eric Wu, Kevin Wu, and James Zou. Datainf: Efficiently estimating data influence in loRA-tuned LLMs and diffusion models. In The Twelfth International Conference on Learning Representations, 2024.

[^35]: Yongchan Kwon and James Zou. Beta shapley: a unified and noise-reduced data valuation framework for machine learning. arXiv preprint arXiv:2110.14049, 2021.

[^36]: Zhuoming Liu, Hao Ding, Huaping Zhong, Weijia Li, Jifeng Dai, and Conghui He. Influence selection for active learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9274–9283, 2021.

[^37]: James Martens and Roger Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In International conference on machine learning, pages 2408–2417. PMLR, 2015.

[^38]: Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.

[^39]: Cade Metz. Lawsuit takes aim at the way A.I. is built. New York Times, 2022.

[^40]: Neel Nanda and Joseph Bloom. Transformerlens. [https://github.com/TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), 2022.

[^41]: Ki Nohyun, Hoyong Choi, and Hye Won Chung. Data valuation without training of a model. In The Eleventh International Conference on Learning Representations, 2023.

[^42]: Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander Madry. TRAK: Attributing model behavior at scale. In International Conference on Machine Learning, pages 27074–27113. PMLR, 2023.

[^43]: Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. Advances in Neural Information Processing Systems, 33:19920–19930, 2020.

[^44]: Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

[^45]: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

[^46]: Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 3505–3506, 2020.

[^47]: Yasaman Razeghi, IV RobertL.Logan, Matt Gardner, and Sameer Singh. Impact of pretraining term frequencies on few-shot numerical reasoning. In Conference on Empirical Methods in Natural Language Processing, 2022.

[^48]: Andrea Schioppa, Polina Zablotskaia, David Vilar, and Artem Sokolov. Scaling up influence functions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 8179–8186, 2022.

[^49]: Shaohuai Shi, Xiaowen Chu, Ka Chun Cheung, and Simon See. Understanding top-k sparsification in distributed deep learning. arXiv preprint arXiv:1911.08772, 2019.

[^50]: Rachael Hwee Ling Sim, Xinyi Xu, and Bryan Kian Hsiang Low. Data valuation in machine learning:" ingredients", strategies, and open challenges. In IJCAI, pages 5607–5614, 2022.

[^51]: Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, et al. Dolma: An open corpus of three trillion tokens for language model pretraining research. arXiv preprint arXiv:2402.00159, 2024.

[^52]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[^53]: Jiachen T Wang and Ruoxi Jia. Data banzhaf: A robust data valuation framework for machine learning. In International Conference on Artificial Intelligence and Statistics, pages 6388–6421. PMLR, 2023.

[^54]: Wei Wen, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Helen Li. Terngrad: Ternary gradients to reduce communication in distributed deep learning. In Neural Information Processing Systems, 2017.

[^55]: Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, October 2020. Association for Computational Linguistics.

[^56]: Theodora Worledge, Judy Hanwen Shen, Nicole Meister, Caleb Winston, and Carlos Guestrin. Unifying corroborative and contributive attributions in large language models. arXiv preprint arXiv:2311.12233, 2023.

[^57]: Zhaoxuan Wu, Yao Shu, and Bryan Kian Hsiang Low. Davinz: Data valuation using deep neural networks at initialization. In International Conference on Machine Learning, pages 24150–24176. PMLR, 2022.

[^58]: Zhengxuan Wu, Atticus Geiger, Aryaman Arora, Jing Huang, Zheng Wang, Noah D Goodman, Christopher D Manning, and Christopher Potts. pyvene: A library for understanding and improving pytorch models via interventions. arXiv preprint arXiv:2403.07809, 2024.

[^59]: Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. Less: Selecting influential data for targeted instruction tuning. arXiv preprint arXiv:2402.04333, 2024.

[^60]: Chih-Kuan Yeh, Ankur Taly, Mukund Sundararajan, Frederick Liu, and Pradeep Ravikumar. First is better than last for language data influence. Advances in Neural Information Processing Systems, 35:32285–32298, 2022.

[^61]: Jinsung Yoon, Sercan Arik, and Tomas Pfister. Data valuation using reinforcement learning. In International Conference on Machine Learning, pages 10842–10851. PMLR, 2020.

[^62]: Boxin Zhao, Boxiang Lyu, Raul Castro Fernandez, and Mladen Kolar. Addressing budget allocation and revenue allocation in data market environments using an adaptive sampling algorithm. In International Conference on Machine Learning, pages 42081–42097. PMLR, 2023.