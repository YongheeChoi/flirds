# Figure 1 concept prompt — Flirds (current; 2026-07-27)

---

Use case: scientific-educational

Asset type: Figure 1 for a machine-learning conference paper; wide landscape academic infographic intended to span both columns of a two-column layout (approximately 2:1 aspect ratio).

Create a clean, publication-quality two-panel conceptual diagram explaining Flirds. Use a white background, crisp flat vector-like styling, a restrained navy/teal/muted-orange/gray palette, thin consistent strokes, generous whitespace, and readable sans-serif typography. All text must be legible when the figure is reduced to a two-column width.

## Panel (a) — "Observed round game (Eq. 5)"

Show one federated round, left to right:

- The current global model \(w^r\) on the left.
- A cohort \(P_r\) selected from \(N\) clients: draw a few participating clients in color and one or two non-participating clients greyed out, labeled "not selected this round." This partial participation must be visible.
- Each participating client runs several local steps and submits only the cumulative displacement \(\delta_k^r\). Draw the local steps as a small dashed path inside the client box with a note: "server observes only the accumulated displacement, not the individual steps."
- Server aggregation weights \(p_k^r = n_k/\sum_{j\in P_r} n_j\) attached to each client's arrow.
- An example coalition \(S \subseteq P_r\) whose move is \(\sum_{k\in S} p_k^r \delta_k^r\), leading to a candidate model and a validation-loss decrease \(u_r(S)\). Annotate: "the weights the server actually used — no renormalization within a coalition." Show that a client left out of \(S\) contributes the zero vector while the other clients' weights stay unchanged.
- The grand-coalition path \(S = P_r \to w^{r+1}\), highlighted as "the actual server update," with a small strip below indicating that summing these across rounds telescopes to \(\ell_{\mathrm{val}}(w^0) - \ell_{\mathrm{val}}(w^R)\).

## Panel (b) — "What is computed exactly, and what is approximated?"

Three cards:

- **"Retraining-based Shapley"** — retrain every client subset from scratch; asks the counterfactual "what if only \(S\) had participated"; its Shapley value is computed exactly over all \(2^N\) subsets. Small note: the federated server cannot run this itself (no access to raw client data).
- **"In-run Shapley"** — fix the one observed trajectory; enumerate every coalition of each round game; sum the per-round Shapley values over rounds.
- **"Flirds"** — same log, same players, same weights, same expansion point; the exact Shapley value of the second-order Taylor surrogate \(\hat u_r\); closed form (Eq. 6) from the validation gradient and one Hessian-vector product per round.

Relations:

- Connect "Retraining-based Shapley" and "In-run Shapley" with a grey dashed two-way connector labeled "different games, different questions" and "relation is empirical (§5.2)."
- Draw one bold teal arrow from "In-run Shapley" to "Flirds" labeled "second-order Taylor surrogate," with the badge "only approximation: truncation."
- Do **not** draw any approximation arrow from "Retraining-based Shapley" to "Flirds."
- Bracket "In-run Shapley" and "Flirds" together as targeting the same round game (Eq. 5).

## Bottom cost strip

- Retraining-based Shapley: \(2^N\) retrainings
- In-run Shapley: \(\sum_r 2^{|P_r|}\) validation evaluations
- Flirds: one JVP per round (validation gradient and HVP obtained together) plus \(|P_r|\) inner products

## Scientific constraints

- "Retraining-based Shapley" and "In-run Shapley" are exact values of **different games**; neither is a universal notion of data value. Do not label either as ground truth, oracle, or reference.
- Flirds approximates In-run Shapley through a quadratic Taylor surrogate — never Retraining-based Shapley.
- Flirds **is** the exact Shapley value **of** that surrogate; do not write that Flirds is a surrogate.
- Do not imply coalition-dependent weight renormalization anywhere.
- Do not use "exact" as a name modifier ("exact in-run Shapley" is wrong); use it only predicatively if needed ("computed exactly").
- Do not use "fixed-weight" as a compound label; express weight handling descriptively as above.
- Do not add extra technical claims, decorative elements, gradients, 3D effects, or watermarks.
