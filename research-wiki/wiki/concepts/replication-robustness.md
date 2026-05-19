---
type: concept
title: Replication-robustness
created: 2026-05-05
updated: 2026-05-05
sources: [asymmetric-data-shapley, ipfl-model-market]
tags: [data-market, axiom, fairness]
---

# Replication-robustness

A property of a data-valuation rule (or, more broadly, market mechanism): **a seller cannot inflate revenue by submitting duplicates of their data**. Originally formalized by Agarwal et al. for data marketplaces.

Formally: the total payment to a seller and any replicas they submit must not exceed the payment that would be assigned to the seller's original data alone.

## Why classical Shapley fails this

The symmetry axiom of [[concepts/shapley-value]] treats interchangeable players identically. If a seller submits two copies of their data as two separate "players," they are by symmetry assigned equal value — collectively earning *twice* what a single submission would.

This pathology motivates the [[concepts/asymmetric-data-shapley|asymmetric variant]] (which introduces directional dependence between original and derivative sources) and is a recurring concern in [[concepts/data-market|data market]] design.

## See also

- [[concepts/shapley-value]]
- [[concepts/asymmetric-data-shapley]]
- [[concepts/data-market]]
- [[threads/symmetry-and-asymmetry-axioms]]
