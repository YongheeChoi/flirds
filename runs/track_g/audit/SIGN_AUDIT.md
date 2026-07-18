# Track G Stage 0 -- phi sign audit (2026-07-19)

Contribution orientation (= -stored phi; helpful client -> POSITIVE).
Rows: 73288 over 309 run-dirs; detectors excluded.

## P1  clean cells: cumulative contribution sign (gate false-exclusion check)

canon variant = the canonical configs; probe variants (lr/steps/rank/...) are listed separately -- do NOT mix them into the do-no-harm claim.

| stage (regime/scale) | variant | method | clients>0 / total | min contribution |
|---|---|---|---|---|
| anchor5-N10k10/1B | anchor10 | (b)oracle | 10/10 | +0.007450 |
| anchor5-N10k10/1B | anchor10 | Fed-LOO | 10/10 | +0.007376 |
| anchor5-N10k10/1B | anchor10 | Flirds | 10/10 | +0.007462 |
| anchor5-N10k10/1B | anchor10 | Flirds1st | 10/10 | +0.007527 |
| anchor5-N10k10/1B | anchor10 | loss-heur | 10/10 | +0.007518 |
| anchor5-N5k5/1B | adamw | (a)oracle | 0/15 | -0.210607 |
| anchor5-N5k5/1B | adamw | (b)oracle | 1/15 | -0.406485 |
| anchor5-N5k5/1B | adamw | Banzhaf | 1/15 | -0.407317 |
| anchor5-N5k5/1B | adamw | ComFedSV | 7/15 | -1.579182 |
| anchor5-N5k5/1B | adamw | Fed-LOO | 15/15 | +0.093864 |
| anchor5-N5k5/1B | adamw | FedIF | 15/15 | +0.130616 |
| anchor5-N5k5/1B | adamw | FedSV | 7/15 | -0.974120 |
| anchor5-N5k5/1B | adamw | Flirds | 0/15 | -2.917955 |
| anchor5-N5k5/1B | adamw | Flirds1st | 0/15 | -0.425567 |
| anchor5-N5k5/1B | adamw | GTG | 5/15 | -0.526699 |
| anchor5-N5k5/1B | adamw | ShapleyFL | 15/15 | +0.010339 |
| anchor5-N5k5/1B | adamw | loss-heur | 0/15 | -0.893706 |
| anchor5-N5k5/1B | canon | (a)oracle | 15/15 | +0.013401 |
| anchor5-N5k5/1B | canon | (b)oracle | 30/30 | +0.013491 |
| anchor5-N5k5/1B | canon | Banzhaf | 15/15 | +0.013493 |
| anchor5-N5k5/1B | canon | ComFedSV | 14/15 | -0.001211 |
| anchor5-N5k5/1B | canon | Fed-LOO | 15/15 | +0.013961 |
| anchor5-N5k5/1B | canon | FedIF | 15/15 | +0.001137 |
| anchor5-N5k5/1B | canon | FedSV | 15/15 | +0.011995 |
| anchor5-N5k5/1B | canon | Flirds | 30/30 | +0.013509 |
| anchor5-N5k5/1B | canon | Flirds1st | 30/30 | +0.013594 |
| anchor5-N5k5/1B | canon | GTG | 15/15 | +0.011395 |
| anchor5-N5k5/1B | canon | ShapleyFL | 14/15 | +0.000000 |
| anchor5-N5k5/1B | canon | loss-heur | 30/30 | +0.013572 |
| anchor5-N5k5/1B | lr1e-3_st20 | (b)oracle | 5/5 | +0.018710 |
| anchor5-N5k5/1B | lr1e-3_st20 | Banzhaf | 5/5 | +0.018719 |
| anchor5-N5k5/1B | lr1e-3_st20 | ComFedSV | 4/5 | -0.001380 |
| anchor5-N5k5/1B | lr1e-3_st20 | Fed-LOO | 5/5 | +0.018372 |
| anchor5-N5k5/1B | lr1e-3_st20 | FedIF | 5/5 | +0.000028 |
| anchor5-N5k5/1B | lr1e-3_st20 | FedSV | 5/5 | +0.014762 |
| anchor5-N5k5/1B | lr1e-3_st20 | Flirds | 5/5 | +0.018790 |
| anchor5-N5k5/1B | lr1e-3_st20 | Flirds1st | 5/5 | +0.019090 |
| anchor5-N5k5/1B | lr1e-3_st20 | GTG | 5/5 | +0.017582 |
| anchor5-N5k5/1B | lr1e-3_st20 | ShapleyFL | 5/5 | +0.000000 |
| anchor5-N5k5/1B | lr1e-3_st20 | loss-heur | 5/5 | +0.019015 |
| anchor5-N5k5/1B | lr1e-3_st30 | (b)oracle | 5/5 | +0.020347 |
| anchor5-N5k5/1B | lr1e-3_st30 | Banzhaf | 5/5 | +0.020367 |
| anchor5-N5k5/1B | lr1e-3_st30 | ComFedSV | 4/5 | -0.000718 |
| anchor5-N5k5/1B | lr1e-3_st30 | Fed-LOO | 5/5 | +0.019887 |
| anchor5-N5k5/1B | lr1e-3_st30 | FedIF | 5/5 | +0.000011 |
| anchor5-N5k5/1B | lr1e-3_st30 | FedSV | 5/5 | +0.015224 |
| anchor5-N5k5/1B | lr1e-3_st30 | Flirds | 5/5 | +0.020513 |
| anchor5-N5k5/1B | lr1e-3_st30 | Flirds1st | 5/5 | +0.020822 |
| anchor5-N5k5/1B | lr1e-3_st30 | GTG | 5/5 | +0.019691 |
| anchor5-N5k5/1B | lr1e-3_st30 | ShapleyFL | 5/5 | +0.000000 |
| anchor5-N5k5/1B | lr1e-3_st30 | loss-heur | 5/5 | +0.020729 |
| anchor5-N5k5/1B | lr2e-3_st10 | (b)oracle | 5/5 | +0.018021 |
| anchor5-N5k5/1B | lr2e-3_st10 | Banzhaf | 5/5 | +0.018028 |
| anchor5-N5k5/1B | lr2e-3_st10 | ComFedSV | 4/5 | -0.002096 |
| anchor5-N5k5/1B | lr2e-3_st10 | Fed-LOO | 5/5 | +0.017749 |
| anchor5-N5k5/1B | lr2e-3_st10 | FedIF | 5/5 | +0.000191 |
| anchor5-N5k5/1B | lr2e-3_st10 | FedSV | 5/5 | +0.014826 |
| anchor5-N5k5/1B | lr2e-3_st10 | Flirds | 5/5 | +0.018096 |
| anchor5-N5k5/1B | lr2e-3_st10 | Flirds1st | 5/5 | +0.018336 |
| anchor5-N5k5/1B | lr2e-3_st10 | GTG | 5/5 | +0.016115 |
| anchor5-N5k5/1B | lr2e-3_st10 | ShapleyFL | 5/5 | +0.000466 |
| anchor5-N5k5/1B | lr2e-3_st10 | loss-heur | 5/5 | +0.018261 |
| anchor5-N5k5/1B | lr2e-3_st20 | (b)oracle | 5/5 | +0.020368 |
| anchor5-N5k5/1B | lr2e-3_st20 | Banzhaf | 5/5 | +0.020399 |
| anchor5-N5k5/1B | lr2e-3_st20 | ComFedSV | 4/5 | -0.001949 |
| anchor5-N5k5/1B | lr2e-3_st20 | Fed-LOO | 5/5 | +0.019725 |
| anchor5-N5k5/1B | lr2e-3_st20 | FedIF | 5/5 | +0.000387 |
| anchor5-N5k5/1B | lr2e-3_st20 | FedSV | 5/5 | +0.012876 |
| anchor5-N5k5/1B | lr2e-3_st20 | Flirds | 5/5 | +0.020633 |
| anchor5-N5k5/1B | lr2e-3_st20 | Flirds1st | 5/5 | +0.021017 |
| anchor5-N5k5/1B | lr2e-3_st20 | GTG | 5/5 | +0.018784 |
| anchor5-N5k5/1B | lr2e-3_st20 | ShapleyFL | 5/5 | +0.000000 |
| anchor5-N5k5/1B | lr2e-3_st20 | loss-heur | 5/5 | +0.020883 |
| anchor5-N5k5/1B | lr2e-3_st30 | (b)oracle | 5/5 | +0.021369 |
| anchor5-N5k5/1B | lr2e-3_st30 | Banzhaf | 5/5 | +0.021441 |
| anchor5-N5k5/1B | lr2e-3_st30 | ComFedSV | 4/5 | -0.000611 |
| anchor5-N5k5/1B | lr2e-3_st30 | Fed-LOO | 5/5 | +0.020416 |
| anchor5-N5k5/1B | lr2e-3_st30 | FedIF | 5/5 | +0.000005 |
| anchor5-N5k5/1B | lr2e-3_st30 | FedSV | 5/5 | +0.014127 |
| anchor5-N5k5/1B | lr2e-3_st30 | Flirds | 5/5 | +0.021967 |
| anchor5-N5k5/1B | lr2e-3_st30 | Flirds1st | 5/5 | +0.022228 |
| anchor5-N5k5/1B | lr2e-3_st30 | GTG | 5/5 | +0.020003 |
| anchor5-N5k5/1B | lr2e-3_st30 | ShapleyFL | 5/5 | +0.000000 |
| anchor5-N5k5/1B | lr2e-3_st30 | loss-heur | 5/5 | +0.022035 |
| anchor5-N5k5/1B | lr3e-3_st10 | (b)oracle | 5/5 | +0.019186 |
| anchor5-N5k5/1B | lr3e-3_st10 | Banzhaf | 5/5 | +0.019205 |
| anchor5-N5k5/1B | lr3e-3_st10 | ComFedSV | 4/5 | -0.002277 |
| anchor5-N5k5/1B | lr3e-3_st10 | Fed-LOO | 5/5 | +0.018760 |
| anchor5-N5k5/1B | lr3e-3_st10 | FedIF | 5/5 | +0.000036 |
| anchor5-N5k5/1B | lr3e-3_st10 | FedSV | 5/5 | +0.013652 |
| anchor5-N5k5/1B | lr3e-3_st10 | Flirds | 5/5 | +0.019359 |
| anchor5-N5k5/1B | lr3e-3_st10 | Flirds1st | 5/5 | +0.019682 |
| anchor5-N5k5/1B | lr3e-3_st10 | GTG | 5/5 | +0.017085 |
| anchor5-N5k5/1B | lr3e-3_st10 | ShapleyFL | 5/5 | +0.000000 |
| anchor5-N5k5/1B | lr3e-3_st10 | loss-heur | 5/5 | +0.019535 |
| anchor5-N5k5/1B | lr3e-3_st20 | (b)oracle | 5/5 | +0.020580 |
| anchor5-N5k5/1B | lr3e-3_st20 | Banzhaf | 5/5 | +0.020651 |
| anchor5-N5k5/1B | lr3e-3_st20 | ComFedSV | 4/5 | -0.002053 |
| anchor5-N5k5/1B | lr3e-3_st20 | Fed-LOO | 5/5 | +0.019665 |
| anchor5-N5k5/1B | lr3e-3_st20 | FedIF | 5/5 | +0.000132 |
| anchor5-N5k5/1B | lr3e-3_st20 | FedSV | 5/5 | +0.010797 |
| anchor5-N5k5/1B | lr3e-3_st20 | Flirds | 5/5 | +0.021165 |
| anchor5-N5k5/1B | lr3e-3_st20 | Flirds1st | 5/5 | +0.021450 |
| anchor5-N5k5/1B | lr3e-3_st20 | GTG | 5/5 | +0.018408 |
| anchor5-N5k5/1B | lr3e-3_st20 | ShapleyFL | 5/5 | +0.000000 |
| anchor5-N5k5/1B | lr3e-3_st20 | loss-heur | 5/5 | +0.021213 |
| anchor5-N5k5/1B | lr3e-3_st30 | (b)oracle | 5/5 | +0.021601 |
| anchor5-N5k5/1B | lr3e-3_st30 | Banzhaf | 5/5 | +0.021741 |
| anchor5-N5k5/1B | lr3e-3_st30 | ComFedSV | 4/5 | -0.000197 |
| anchor5-N5k5/1B | lr3e-3_st30 | Fed-LOO | 5/5 | +0.020307 |
| anchor5-N5k5/1B | lr3e-3_st30 | FedIF | 5/5 | +0.000002 |
| anchor5-N5k5/1B | lr3e-3_st30 | FedSV | 5/5 | +0.013972 |
| anchor5-N5k5/1B | lr3e-3_st30 | Flirds | 5/5 | +0.022738 |
| anchor5-N5k5/1B | lr3e-3_st30 | Flirds1st | 5/5 | +0.022612 |
| anchor5-N5k5/1B | lr3e-3_st30 | GTG | 5/5 | +0.019734 |
| anchor5-N5k5/1B | lr3e-3_st30 | ShapleyFL | 5/5 | +0.000000 |
| anchor5-N5k5/1B | lr3e-3_st30 | loss-heur | 5/5 | +0.022334 |
| anchor5-N5k5/1B | r32 | (b)oracle | 5/5 | +0.016196 |
| anchor5-N5k5/1B | r32 | Banzhaf | 5/5 | +0.016198 |
| anchor5-N5k5/1B | r32 | ComFedSV | 4/5 | -0.000913 |
| anchor5-N5k5/1B | r32 | FedIF | 5/5 | +0.000143 |
| anchor5-N5k5/1B | r32 | FedSV | 5/5 | +0.014023 |
| anchor5-N5k5/1B | r32 | Flirds | 5/5 | +0.016220 |
| anchor5-N5k5/1B | r32 | Flirds1st | 5/5 | +0.016519 |
| anchor5-N5k5/1B | r32 | GTG | 5/5 | +0.015368 |
| anchor5-N5k5/1B | r32 | ShapleyFL | 5/5 | +0.000000 |
| anchor5-N5k5/1B | r32 | loss-heur | 5/5 | +0.016449 |
| anchor5-N5k5/1B | r64 | (b)oracle | 5/5 | +0.017471 |
| anchor5-N5k5/1B | r64 | Banzhaf | 5/5 | +0.017473 |
| anchor5-N5k5/1B | r64 | ComFedSV | 4/5 | -0.000752 |
| anchor5-N5k5/1B | r64 | Fed-LOO | 5/5 | +0.017047 |
| anchor5-N5k5/1B | r64 | FedIF | 5/5 | +0.000090 |
| anchor5-N5k5/1B | r64 | FedSV | 5/5 | +0.014148 |
| anchor5-N5k5/1B | r64 | Flirds | 5/5 | +0.017498 |
| anchor5-N5k5/1B | r64 | Flirds1st | 5/5 | +0.018002 |
| anchor5-N5k5/1B | r64 | GTG | 5/5 | +0.016866 |
| anchor5-N5k5/1B | r64 | ShapleyFL | 5/5 | +0.000000 |
| anchor5-N5k5/1B | r64 | loss-heur | 5/5 | +0.017888 |
| anchor5-N5k5/1B | removal | (a)oracle | 15/15 | +0.013774 |
| anchor5-N5k5/1B | removal | (b)oracle | 15/15 | +0.013795 |
| anchor5-N5k5/1B | removal | Banzhaf | 15/15 | +0.013797 |
| anchor5-N5k5/1B | removal | ComFedSV | 14/15 | -0.001129 |
| anchor5-N5k5/1B | removal | Fed-LOO | 15/15 | +0.013703 |
| anchor5-N5k5/1B | removal | FedIF | 15/15 | +0.000691 |
| anchor5-N5k5/1B | removal | FedSV | 15/15 | +0.012247 |
| anchor5-N5k5/1B | removal | Flirds | 15/15 | +0.013815 |
| anchor5-N5k5/1B | removal | Flirds1st | 15/15 | +0.013901 |
| anchor5-N5k5/1B | removal | GTG | 15/15 | +0.011756 |
| anchor5-N5k5/1B | removal | ShapleyFL | 14/15 | +0.000000 |
| anchor5-N5k5/1B | removal | loss-heur | 15/15 | +0.013880 |
| anchor5-N5k5/3B | canon | (b)oracle | 15/15 | +0.013028 |
| anchor5-N5k5/3B | canon | Banzhaf | 15/15 | +0.013029 |
| anchor5-N5k5/3B | canon | ComFedSV | 15/15 | +0.000329 |
| anchor5-N5k5/3B | canon | FedIF | 13/15 | +0.000000 |
| anchor5-N5k5/3B | canon | FedSV | 15/15 | +0.010023 |
| anchor5-N5k5/3B | canon | Flirds | 15/15 | +0.013043 |
| anchor5-N5k5/3B | canon | Flirds1st | 15/15 | +0.013038 |
| anchor5-N5k5/3B | canon | GTG | 15/15 | +0.010554 |
| anchor5-N5k5/3B | canon | ShapleyFL | 14/15 | +0.000000 |
| anchor5-N5k5/3B | canon | loss-heur | 15/15 | +0.013038 |
| anchor5-N5k5/7B | canon | (b)oracle | 15/15 | +0.004662 |
| anchor5-N5k5/7B | canon | Banzhaf | 15/15 | +0.004662 |
| anchor5-N5k5/7B | canon | ComFedSV | 14/15 | -0.001516 |
| anchor5-N5k5/7B | canon | FedIF | 13/15 | +0.000000 |
| anchor5-N5k5/7B | canon | FedSV | 15/15 | +0.003378 |
| anchor5-N5k5/7B | canon | Flirds | 15/15 | +0.004661 |
| anchor5-N5k5/7B | canon | Flirds1st | 15/15 | +0.004611 |
| anchor5-N5k5/7B | canon | GTG | 15/15 | +0.003298 |
| anchor5-N5k5/7B | canon | ShapleyFL | 13/15 | +0.000000 |
| anchor5-N5k5/7B | canon | loss-heur | 15/15 | +0.004620 |
| device100/1B | canon | ComFedSV | 48/98 | -0.000000 |
| device100/1B | canon | Fed-LOO | 98/98 | +0.000079 |
| device100/1B | canon | FedIF | 92/98 | +0.000000 |
| device100/1B | canon | Flirds | 98/98 | +0.000079 |
| device100/1B | canon | Flirds1st | 98/98 | +0.000078 |
| device100/1B | canon | loss-heur | 98/98 | +0.000078 |
| iid5/1B | canon | (b)oracle | 15/15 | +0.005986 |
| iid5/1B | canon | Banzhaf | 15/15 | +0.005987 |
| iid5/1B | canon | FedIF | 13/15 | +0.000000 |
| iid5/1B | canon | FedSV | 15/15 | +0.005479 |
| iid5/1B | canon | Flirds | 15/15 | +0.005994 |
| iid5/1B | canon | Flirds1st | 15/15 | +0.005993 |
| iid5/1B | canon | GTG | 15/15 | +0.005457 |
| iid5/1B | canon | ShapleyFL | 13/15 | +0.000000 |
| iid5/1B | canon | loss-heur | 15/15 | +0.005991 |
| silo5/1B | canon | (b)oracle | 15/15 | +0.002091 |
| silo5/1B | canon | Banzhaf | 15/15 | +0.002091 |
| silo5/1B | canon | FedIF | 12/15 | +0.000000 |
| silo5/1B | canon | FedSV | 15/15 | +0.000660 |
| silo5/1B | canon | Flirds | 15/15 | +0.002093 |
| silo5/1B | canon | Flirds1st | 15/15 | +0.002082 |
| silo5/1B | canon | GTG | 15/15 | +0.000443 |
| silo5/1B | canon | ShapleyFL | 13/15 | +0.000000 |
| silo5/1B | canon | loss-heur | 15/15 | +0.002079 |
| std20-N20k2/1B | canon | (b)oracle | 120/120 | +0.000536 |
| std20-N20k2/1B | canon | ComFedSV | 35/60 | -0.000000 |
| std20-N20k2/1B | canon | Fed-LOO | 60/60 | +0.000760 |
| std20-N20k2/1B | canon | FedIF | 59/60 | +0.000000 |
| std20-N20k2/1B | canon | FedSV | 59/60 | -0.000435 |
| std20-N20k2/1B | canon | Flirds | 120/120 | +0.000547 |
| std20-N20k2/1B | canon | Flirds1st | 120/120 | +0.000756 |
| std20-N20k2/1B | canon | GTG | 60/60 | +0.000726 |
| std20-N20k2/1B | canon | ShapleyFL | 59/60 | +0.000000 |
| std20-N20k2/1B | canon | loss-heur | 120/120 | +0.000534 |
| std20-N20k2/3B | canon | (b)oracle | 60/60 | +0.002158 |
| std20-N20k2/3B | canon | ComFedSV | 34/60 | -0.000000 |
| std20-N20k2/3B | canon | FedIF | 58/60 | +0.000000 |
| std20-N20k2/3B | canon | FedSV | 60/60 | +0.001691 |
| std20-N20k2/3B | canon | Flirds | 60/60 | +0.002159 |
| std20-N20k2/3B | canon | Flirds1st | 60/60 | +0.002223 |
| std20-N20k2/3B | canon | GTG | 60/60 | +0.001944 |
| std20-N20k2/3B | canon | ShapleyFL | 60/60 | +0.000000 |
| std20-N20k2/3B | canon | loss-heur | 60/60 | +0.002160 |
| std20-N20k2/7B | canon | (b)oracle | 60/60 | +0.000046 |
| std20-N20k2/7B | canon | ComFedSV | 34/60 | -0.000000 |
| std20-N20k2/7B | canon | FedIF | 60/60 | +0.000158 |
| std20-N20k2/7B | canon | FedSV | 59/60 | -0.000777 |
| std20-N20k2/7B | canon | Flirds | 60/60 | +0.000051 |
| std20-N20k2/7B | canon | Flirds1st | 60/60 | +0.000096 |
| std20-N20k2/7B | canon | GTG | 60/60 | +0.000655 |
| std20-N20k2/7B | canon | ShapleyFL | 59/60 | +0.000000 |
| std20-N20k2/7B | canon | loss-heur | 60/60 | +0.000049 |
| std20-N50k5/1B | r16 | (b)oracle | 50/50 | +0.000582 |
| std20-N50k5/1B | r16 | ComFedSV | 30/50 | -0.000000 |
| std20-N50k5/1B | r16 | Fed-LOO | 50/50 | +0.000584 |
| std20-N50k5/1B | r16 | FedIF | 50/50 | +0.000121 |
| std20-N50k5/1B | r16 | FedSV | 50/50 | +0.000308 |
| std20-N50k5/1B | r16 | Flirds | 50/50 | +0.000585 |
| std20-N50k5/1B | r16 | Flirds1st | 50/50 | +0.000604 |
| std20-N50k5/1B | r16 | GTG | 50/50 | +0.000606 |
| std20-N50k5/1B | r16 | ShapleyFL | 50/50 | +0.000000 |
| std20-N50k5/1B | r16 | loss-heur | 50/50 | +0.000582 |
| std20-N50k5/1B | r32 | (b)oracle | 50/50 | +0.000572 |
| std20-N50k5/1B | r32 | ComFedSV | 30/50 | -0.000000 |
| std20-N50k5/1B | r32 | Fed-LOO | 50/50 | +0.000572 |
| std20-N50k5/1B | r32 | FedIF | 50/50 | +0.000085 |
| std20-N50k5/1B | r32 | FedSV | 50/50 | +0.000160 |
| std20-N50k5/1B | r32 | Flirds | 50/50 | +0.000574 |
| std20-N50k5/1B | r32 | Flirds1st | 50/50 | +0.000590 |
| std20-N50k5/1B | r32 | GTG | 50/50 | +0.000598 |
| std20-N50k5/1B | r32 | ShapleyFL | 50/50 | +0.000000 |
| std20-N50k5/1B | r32 | loss-heur | 50/50 | +0.000572 |
| std20-N50k5/1B | r64 | (b)oracle | 50/50 | +0.000583 |
| std20-N50k5/1B | r64 | ComFedSV | 29/50 | -0.000000 |
| std20-N50k5/1B | r64 | Fed-LOO | 50/50 | +0.000583 |
| std20-N50k5/1B | r64 | FedIF | 50/50 | +0.000030 |
| std20-N50k5/1B | r64 | FedSV | 50/50 | +0.000202 |
| std20-N50k5/1B | r64 | Flirds | 50/50 | +0.000585 |
| std20-N50k5/1B | r64 | Flirds1st | 50/50 | +0.000598 |
| std20-N50k5/1B | r64 | GTG | 50/50 | +0.000515 |
| std20-N50k5/1B | r64 | ShapleyFL | 50/50 | +0.000000 |
| std20-N50k5/1B | r64 | loss-heur | 50/50 | +0.000583 |

CNN iid (canonical) uncorrupt clients:

| dataset | method | clients>0 / total | min |
|---|---|---|---|
| cifar10-iid | (b)oracle | 70/70 | +0.1024 |
| cifar10-iid | Banzhaf | 70/70 | +0.1170 |
| cifar10-iid | ComFedSV | 49/70 | -0.5741 |
| cifar10-iid | Fed-LOO | 40/40 | +0.0446 |
| cifar10-iid | FedIF | 70/70 | +0.2243 |
| cifar10-iid | FedSV | 60/70 | -0.0558 |
| cifar10-iid | Flirds | 70/70 | +0.1175 |
| cifar10-iid | Flirds1st | 70/70 | +0.0984 |
| cifar10-iid | GTG | 70/70 | +0.0290 |
| cifar10-iid | Ripple | 18/40 | -252.6251 |
| cifar10-iid | ShapleyFL | 70/70 | +0.1438 |
| cifar10-iid | loss-heur | 70/70 | +0.0788 |
| mnist-iid | (b)oracle | 70/70 | +0.2192 |
| mnist-iid | Banzhaf | 70/70 | +0.2701 |
| mnist-iid | ComFedSV | 70/70 | +0.0159 |
| mnist-iid | Fed-LOO | 40/40 | +0.1041 |
| mnist-iid | FedIF | 70/70 | +0.1417 |
| mnist-iid | FedSV | 70/70 | +0.0487 |
| mnist-iid | Flirds | 70/70 | +0.1238 |
| mnist-iid | Flirds1st | 70/70 | +0.0577 |
| mnist-iid | GTG | 70/70 | +0.1790 |
| mnist-iid | Ripple | 40/40 | +10.4758 |
| mnist-iid | ShapleyFL | 70/70 | +0.1000 |
| mnist-iid | loss-heur | 70/70 | +0.0689 |

## P2  free-rider corrupt-client contribution (strict >0 rule check)

| stage | threat | method | n | bit-exact 0 | n>0 | n<0 | mean | max abs |
|---|---|---|---|---|---|---|---|---|
| device100/1B | freerider_random | (b)oracle | 15 | 0/15 | 8 | 7 | +1.103e-08 | 1.988e-07 |
| device100/1B | freerider_random | ComFedSV | 135 | 0/135 | 87 | 48 | +3.132e-129 | 3.978e-127 |
| device100/1B | freerider_random | FedIF | 135 | 73/135 | 62 | 0 | +8.299e-05 | 6.358e-04 |
| device100/1B | freerider_random | FedSV | 15 | 0/15 | 0 | 15 | -5.035e-04 | 1.110e-03 |
| device100/1B | freerider_random | Flirds | 135 | 0/135 | 65 | 70 | -8.302e-10 | 3.884e-07 |
| device100/1B | freerider_random | Flirds1st | 135 | 0/135 | 66 | 69 | -8.328e-10 | 3.878e-07 |
| device100/1B | freerider_random | GTG | 15 | 0/15 | 0 | 15 | -3.970e-04 | 7.579e-04 |
| device100/1B | freerider_random | ShapleyFL | 15 | 7/15 | 8 | 0 | +1.303e-04 | 7.775e-04 |
| device100/1B | freerider_random | loss-heur | 135 | 34/135 | 47 | 54 | -4.945e-08 | 1.192e-06 |
| device100/1B | freerider_zero | (b)oracle | 15 | 15/15 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| device100/1B | freerider_zero | ComFedSV | 135 | 0/135 | 88 | 47 | +1.203e-124 | 8.552e-123 |
| device100/1B | freerider_zero | FedIF | 135 | 135/135 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| device100/1B | freerider_zero | FedSV | 15 | 0/15 | 0 | 15 | -5.883e-04 | 1.324e-03 |
| device100/1B | freerider_zero | Flirds | 135 | 135/135 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| device100/1B | freerider_zero | Flirds1st | 135 | 135/135 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| device100/1B | freerider_zero | GTG | 15 | 0/15 | 0 | 15 | -4.705e-04 | 9.081e-04 |
| device100/1B | freerider_zero | ShapleyFL | 15 | 11/15 | 4 | 0 | +6.989e-18 | 3.450e-17 |
| device100/1B | freerider_zero | loss-heur | 135 | 135/135 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| iid5/1B | freerider_random | (b)oracle | 3 | 0/3 | 2 | 1 | +7.139e-07 | 1.291e-06 |
| iid5/1B | freerider_random | Banzhaf | 3 | 0/3 | 2 | 1 | +7.053e-07 | 1.304e-06 |
| iid5/1B | freerider_random | FedIF | 3 | 3/3 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| iid5/1B | freerider_random | FedSV | 3 | 0/3 | 0 | 3 | -9.678e-03 | 1.038e-02 |
| iid5/1B | freerider_random | Flirds | 3 | 0/3 | 3 | 0 | +7.425e-07 | 1.238e-06 |
| iid5/1B | freerider_random | Flirds1st | 3 | 0/3 | 3 | 0 | +7.350e-07 | 1.241e-06 |
| iid5/1B | freerider_random | GTG | 3 | 0/3 | 0 | 3 | -7.506e-03 | 8.046e-03 |
| iid5/1B | freerider_random | ShapleyFL | 3 | 3/3 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| iid5/1B | freerider_random | loss-heur | 3 | 0/3 | 2 | 1 | +4.371e-07 | 1.431e-06 |
| iid5/1B | freerider_zero | (b)oracle | 3 | 3/3 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| iid5/1B | freerider_zero | Banzhaf | 3 | 3/3 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| iid5/1B | freerider_zero | FedIF | 3 | 3/3 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| iid5/1B | freerider_zero | FedSV | 3 | 0/3 | 0 | 3 | -8.488e-03 | 9.082e-03 |
| iid5/1B | freerider_zero | Flirds | 3 | 3/3 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| iid5/1B | freerider_zero | Flirds1st | 3 | 3/3 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| iid5/1B | freerider_zero | GTG | 3 | 0/3 | 0 | 3 | -6.585e-03 | 7.068e-03 |
| iid5/1B | freerider_zero | ShapleyFL | 3 | 3/3 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| iid5/1B | freerider_zero | loss-heur | 3 | 3/3 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/1B | freerider_delta | (b)oracle | 3 | 0/3 | 3 | 0 | +2.491e-03 | 2.697e-03 |
| silo5/1B | freerider_delta | Banzhaf | 3 | 0/3 | 3 | 0 | +2.491e-03 | 2.697e-03 |
| silo5/1B | freerider_delta | ComFedSV | 3 | 0/3 | 2 | 1 | +1.377e-03 | 3.062e-03 |
| silo5/1B | freerider_delta | Fed-LOO | 3 | 0/3 | 3 | 0 | +2.492e-03 | 2.700e-03 |
| silo5/1B | freerider_delta | FedIF | 3 | 0/3 | 3 | 0 | +9.596e-01 | 9.596e-01 |
| silo5/1B | freerider_delta | FedSV | 3 | 0/3 | 3 | 0 | +2.089e-03 | 2.284e-03 |
| silo5/1B | freerider_delta | Flirds | 3 | 0/3 | 3 | 0 | +2.492e-03 | 2.699e-03 |
| silo5/1B | freerider_delta | Flirds1st | 3 | 0/3 | 3 | 0 | +2.491e-03 | 2.695e-03 |
| silo5/1B | freerider_delta | GTG | 3 | 0/3 | 3 | 0 | +2.103e-03 | 2.209e-03 |
| silo5/1B | freerider_delta | ShapleyFL | 3 | 0/3 | 3 | 0 | +3.050e-01 | 3.104e-01 |
| silo5/1B | freerider_delta | loss-heur | 3 | 0/3 | 3 | 0 | +2.490e-03 | 2.695e-03 |
| silo5/1B | freerider_random | (b)oracle | 24 | 0/24 | 8 | 16 | -2.989e-07 | 4.558e-06 |
| silo5/1B | freerider_random | Banzhaf | 24 | 1/24 | 8 | 15 | -2.521e-07 | 4.783e-06 |
| silo5/1B | freerider_random | ComFedSV | 18 | 0/18 | 0 | 18 | -4.133e-03 | 5.015e-03 |
| silo5/1B | freerider_random | Fed-LOO | 18 | 2/18 | 5 | 11 | -2.782e-07 | 4.768e-06 |
| silo5/1B | freerider_random | FedIF | 24 | 24/24 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/1B | freerider_random | FedSV | 24 | 0/24 | 0 | 24 | -4.310e-03 | 5.340e-03 |
| silo5/1B | freerider_random | Flirds | 24 | 0/24 | 6 | 18 | -3.109e-07 | 4.523e-06 |
| silo5/1B | freerider_random | Flirds1st | 24 | 0/24 | 6 | 18 | -3.128e-07 | 4.550e-06 |
| silo5/1B | freerider_random | GTG | 24 | 0/24 | 0 | 24 | -3.420e-03 | 4.145e-03 |
| silo5/1B | freerider_random | ShapleyFL | 24 | 24/24 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/1B | freerider_random | loss-heur | 24 | 2/24 | 6 | 16 | -3.278e-07 | 3.815e-06 |
| silo5/1B | freerider_zero | (b)oracle | 9 | 9/9 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/1B | freerider_zero | Banzhaf | 9 | 9/9 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/1B | freerider_zero | ComFedSV | 3 | 0/3 | 0 | 3 | -4.277e-03 | 5.470e-03 |
| silo5/1B | freerider_zero | Fed-LOO | 3 | 3/3 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/1B | freerider_zero | FedIF | 9 | 9/9 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/1B | freerider_zero | FedSV | 9 | 0/9 | 0 | 9 | -4.321e-03 | 5.290e-03 |
| silo5/1B | freerider_zero | Flirds | 9 | 9/9 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/1B | freerider_zero | Flirds1st | 9 | 9/9 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/1B | freerider_zero | GTG | 9 | 0/9 | 0 | 9 | -3.433e-03 | 4.096e-03 |
| silo5/1B | freerider_zero | ShapleyFL | 9 | 9/9 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/1B | freerider_zero | loss-heur | 9 | 9/9 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/3B | freerider_random | (b)oracle | 2 | 0/2 | 1 | 1 | +6.159e-08 | 1.431e-07 |
| silo5/3B | freerider_random | FedIF | 2 | 2/2 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/3B | freerider_random | Flirds | 2 | 0/2 | 2 | 0 | +1.246e-07 | 2.158e-07 |
| silo5/3B | freerider_random | Flirds1st | 2 | 0/2 | 2 | 0 | +1.256e-07 | 2.138e-07 |
| silo5/3B | freerider_random | loss-heur | 2 | 1/2 | 0 | 1 | -4.768e-07 | 9.537e-07 |
| silo5/3B | freerider_zero | (b)oracle | 2 | 2/2 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/3B | freerider_zero | FedIF | 2 | 2/2 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/3B | freerider_zero | Flirds | 2 | 2/2 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/3B | freerider_zero | Flirds1st | 2 | 2/2 | 0 | 0 | +0.000e+00 | 0.000e+00 |
| silo5/3B | freerider_zero | loss-heur | 2 | 2/2 | 0 | 0 | +0.000e+00 | 0.000e+00 |

## P3  noisy corrupt-client contribution vs noisy_rate (LLM dose-cell basis)

| regime/scale | method | nr=0 | nr=0.1 | nr=0.25 | nr=0.5 | nr=0.75 | nr=1 | 0-crossing nr |
|---|---|---|---|---|---|---|---|---|
| device100/1B | (b)oracle | nan | nan | nan | nan | nan | +0.000259 | single dose |
| device100/1B | ComFedSV | nan | nan | nan | nan | nan | +0.000000 | single dose |
| device100/1B | FedIF | nan | nan | nan | nan | nan | +0.115678 | single dose |
| device100/1B | FedSV | nan | nan | nan | nan | nan | +0.000143 | single dose |
| device100/1B | Flirds | nan | nan | nan | nan | nan | +0.000221 | single dose |
| device100/1B | Flirds1st | nan | nan | nan | nan | nan | +0.000220 | single dose |
| device100/1B | GTG | nan | nan | nan | nan | nan | +0.000122 | single dose |
| device100/1B | ShapleyFL | nan | nan | nan | nan | nan | +0.150324 | single dose |
| device100/1B | loss-heur | nan | nan | nan | nan | nan | +0.000220 | single dose |
| iid5/1B | (b)oracle | nan | nan | nan | nan | nan | +0.004931 | single dose |
| iid5/1B | Banzhaf | nan | nan | nan | nan | nan | +0.004932 | single dose |
| iid5/1B | FedIF | nan | nan | nan | nan | nan | +0.000000 | single dose |
| iid5/1B | FedSV | nan | nan | nan | nan | nan | +0.002000 | single dose |
| iid5/1B | Flirds | nan | nan | nan | nan | nan | +0.004937 | single dose |
| iid5/1B | Flirds1st | nan | nan | nan | nan | nan | +0.004949 | single dose |
| iid5/1B | GTG | nan | nan | nan | nan | nan | +0.002592 | single dose |
| iid5/1B | ShapleyFL | nan | nan | nan | nan | nan | +0.000000 | single dose |
| iid5/1B | loss-heur | nan | nan | nan | nan | nan | +0.004942 | single dose |
| silo5/1B | (b)oracle | +0.002420 | +0.002456 | +0.002407 | +0.002281 | +0.001991 | +0.001806 | ~3.44 = UNREACHABLE (nr caps at 1.0) |
| silo5/1B | Banzhaf | +0.002420 | +0.002456 | +0.002407 | +0.002281 | +0.001991 | +0.001806 | ~3.44 = UNREACHABLE (nr caps at 1.0) |
| silo5/1B | ComFedSV | -0.000533 | -0.000562 | -0.000818 | -0.001153 | -0.001434 | -0.001945 | none (positive across ladder) |
| silo5/1B | Fed-LOO | +0.002437 | +0.002472 | +0.002424 | +0.002296 | +0.002006 | +0.001871 | ~4.45 = UNREACHABLE (nr caps at 1.0) |
| silo5/1B | FedIF | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | none (positive across ladder) |
| silo5/1B | FedSV | +0.000855 | +0.000844 | +0.000609 | +0.000236 | -0.000151 | -0.000677 | ~0.65 |
| silo5/1B | Flirds | +0.002422 | +0.002458 | +0.002410 | +0.002283 | +0.001993 | +0.001808 | ~3.44 = UNREACHABLE (nr caps at 1.0) |
| silo5/1B | Flirds1st | +0.002407 | +0.002443 | +0.002394 | +0.002269 | +0.001980 | +0.001795 | ~3.42 = UNREACHABLE (nr caps at 1.0) |
| silo5/1B | GTG | +0.001048 | +0.001047 | +0.000798 | +0.000429 | +0.000019 | -0.000486 | ~0.76 |
| silo5/1B | ShapleyFL | +0.035020 | +0.013451 | +0.000008 | +0.000000 | +0.000000 | +0.000000 | ~0.50 |
| silo5/1B | loss-heur | +0.002404 | +0.002440 | +0.002390 | +0.002264 | +0.001977 | +0.001792 | ~3.41 = UNREACHABLE (nr caps at 1.0) |
| silo5/3B | (b)oracle | nan | nan | nan | nan | nan | +0.000728 | single dose |
| silo5/3B | FedIF | nan | nan | nan | nan | nan | +0.000000 | single dose |
| silo5/3B | Flirds | nan | nan | nan | nan | nan | +0.000729 | single dose |
| silo5/3B | Flirds1st | nan | nan | nan | nan | nan | +0.000715 | single dose |
| silo5/3B | loss-heur | nan | nan | nan | nan | nan | +0.000714 | single dose |

## P4  CNN corrupt-client contribution vs per-client rate (C2 dose-ladder basis)

| dataset/scenario | method | corr(contribution, rate) | crossing rate | note |
|---|---|---|---|---|
| cifar10-feature_noise | (b)oracle | -0.925 | ~0.974 (extrapolated) | monotone down |
| cifar10-feature_noise | Banzhaf | -0.928 | ~0.977 (extrapolated) | monotone down |
| cifar10-feature_noise | ComFedSV | -0.366 | ~0.050 | non-monotone |
| cifar10-feature_noise | Fed-LOO | -0.905 | >1 (unreachable on rate axis) | monotone down |
| cifar10-feature_noise | FedIF | -0.916 | none | monotone down |
| cifar10-feature_noise | FedSV | -0.644 | none | non-monotone |
| cifar10-feature_noise | Flirds | -0.922 | ~0.967 (extrapolated) | monotone down |
| cifar10-feature_noise | Flirds1st | -0.919 | ~0.813 (extrapolated) | monotone down |
| cifar10-feature_noise | GTG | -0.858 | none | monotone down |
| cifar10-feature_noise | Ripple | +0.245 | ~0.096 | non-monotone |
| cifar10-feature_noise | ShapleyFL | +0.168 | none | non-monotone |
| cifar10-feature_noise | loss-heur | -0.935 | ~0.709 (extrapolated) | monotone down |
| cifar10-label_flip | (b)oracle | -0.991 | ~0.332 (extrapolated) | monotone down |
| cifar10-label_flip | Banzhaf | -0.992 | ~0.344 (extrapolated) | monotone down |
| cifar10-label_flip | ComFedSV | -0.688 | ~0.216 (extrapolated) | non-monotone |
| cifar10-label_flip | Fed-LOO | -0.994 | ~0.286 (extrapolated) | monotone down |
| cifar10-label_flip | FedIF | -0.967 | ~0.367 (extrapolated) | monotone down |
| cifar10-label_flip | FedSV | -0.783 | ~0.314 (extrapolated) | non-monotone |
| cifar10-label_flip | Flirds | -0.991 | ~0.346 (extrapolated) | monotone down |
| cifar10-label_flip | Flirds1st | -0.972 | ~0.378 (extrapolated) | monotone down |
| cifar10-label_flip | GTG | -0.818 | ~0.272 (extrapolated) | monotone down |
| cifar10-label_flip | Ripple | -0.440 | ~0.138 | non-monotone |
| cifar10-label_flip | ShapleyFL | -0.357 | none | non-monotone |
| cifar10-label_flip | loss-heur | -0.982 | ~0.329 (extrapolated) | monotone down |
| mnist-feature_noise | (b)oracle | -0.276 | >1 (unreachable on rate axis) | non-monotone |
| mnist-feature_noise | Banzhaf | -0.378 | >1 (unreachable on rate axis) | non-monotone |
| mnist-feature_noise | ComFedSV | -0.590 | ~0.366 (extrapolated) | non-monotone |
| mnist-feature_noise | Fed-LOO | -0.039 | >1 (unreachable on rate axis) | non-monotone |
| mnist-feature_noise | FedIF | +0.732 | ~0.621 (extrapolated) | non-monotone |
| mnist-feature_noise | FedSV | -0.828 | ~0.615 (extrapolated) | monotone down |
| mnist-feature_noise | Flirds | -0.265 | >1 (unreachable on rate axis) | non-monotone |
| mnist-feature_noise | Flirds1st | +0.356 | >1 (unreachable on rate axis) | non-monotone |
| mnist-feature_noise | GTG | -0.315 | >1 (unreachable on rate axis) | non-monotone |
| mnist-feature_noise | Ripple | +0.938 | >1 (unreachable on rate axis) | monotone up |
| mnist-feature_noise | ShapleyFL | +0.877 | none | monotone up |
| mnist-feature_noise | loss-heur | +0.130 | >1 (unreachable on rate axis) | non-monotone |
| mnist-label_flip | (b)oracle | -0.959 | ~0.271 (extrapolated) | monotone down |
| mnist-label_flip | Banzhaf | -0.960 | ~0.286 (extrapolated) | monotone down |
| mnist-label_flip | ComFedSV | -0.998 | ~0.136 | monotone down |
| mnist-label_flip | Fed-LOO | -0.960 | ~0.268 (extrapolated) | monotone down |
| mnist-label_flip | FedIF | -0.927 | ~0.217 (extrapolated) | monotone down |
| mnist-label_flip | FedSV | -0.992 | ~0.132 | monotone down |
| mnist-label_flip | Flirds | -0.948 | ~0.245 (extrapolated) | monotone down |
| mnist-label_flip | Flirds1st | -0.939 | ~0.190 | monotone down |
| mnist-label_flip | GTG | -0.999 | ~0.182 | monotone down |
| mnist-label_flip | Ripple | -0.923 | ~0.546 (extrapolated) | monotone down |
| mnist-label_flip | ShapleyFL | -0.976 | ~0.212 (extrapolated) | monotone down |
| mnist-label_flip | loss-heur | -0.953 | ~0.187 | monotone down |

## P5  corrupt-client sign by method (value-level decision differences; canonical dose only)

| stage | threat | method | mean corrupt contribution | sign |
|---|---|---|---|---|
| device100/1B | freerider_random | (b)oracle | +1.103e-08 | + |
| device100/1B | freerider_random | ComFedSV | +3.132e-129 | + |
| device100/1B | freerider_random | FedIF | +8.299e-05 | + |
| device100/1B | freerider_random | FedSV | -5.035e-04 | - |
| device100/1B | freerider_random | Flirds | -8.302e-10 | - |
| device100/1B | freerider_random | Flirds1st | -8.328e-10 | - |
| device100/1B | freerider_random | GTG | -3.970e-04 | - |
| device100/1B | freerider_random | ShapleyFL | +1.303e-04 | + |
| device100/1B | freerider_random | loss-heur | -4.945e-08 | - |
| device100/1B | freerider_zero | (b)oracle | +0.000e+00 | 0 |
| device100/1B | freerider_zero | ComFedSV | +1.203e-124 | + |
| device100/1B | freerider_zero | FedIF | +0.000e+00 | 0 |
| device100/1B | freerider_zero | FedSV | -5.883e-04 | - |
| device100/1B | freerider_zero | Flirds | +0.000e+00 | 0 |
| device100/1B | freerider_zero | Flirds1st | +0.000e+00 | 0 |
| device100/1B | freerider_zero | GTG | -4.705e-04 | - |
| device100/1B | freerider_zero | ShapleyFL | +6.989e-18 | + |
| device100/1B | freerider_zero | loss-heur | +0.000e+00 | 0 |
| device100/1B | noisy | (b)oracle | +2.590e-04 | + |
| device100/1B | noisy | ComFedSV | +4.778e-124 | + |
| device100/1B | noisy | FedIF | +1.157e-01 | + |
| device100/1B | noisy | FedSV | +1.433e-04 | + |
| device100/1B | noisy | Flirds | +2.208e-04 | + |
| device100/1B | noisy | Flirds1st | +2.204e-04 | + |
| device100/1B | noisy | GTG | +1.223e-04 | + |
| device100/1B | noisy | ShapleyFL | +1.503e-01 | + |
| device100/1B | noisy | loss-heur | +2.202e-04 | + |
| iid5/1B | freerider_random | (b)oracle | +7.139e-07 | + |
| iid5/1B | freerider_random | Banzhaf | +7.053e-07 | + |
| iid5/1B | freerider_random | FedIF | +0.000e+00 | 0 |
| iid5/1B | freerider_random | FedSV | -9.678e-03 | - |
| iid5/1B | freerider_random | Flirds | +7.425e-07 | + |
| iid5/1B | freerider_random | Flirds1st | +7.350e-07 | + |
| iid5/1B | freerider_random | GTG | -7.506e-03 | - |
| iid5/1B | freerider_random | ShapleyFL | +0.000e+00 | 0 |
| iid5/1B | freerider_random | loss-heur | +4.371e-07 | + |
| iid5/1B | freerider_zero | (b)oracle | +0.000e+00 | 0 |
| iid5/1B | freerider_zero | Banzhaf | +0.000e+00 | 0 |
| iid5/1B | freerider_zero | FedIF | +0.000e+00 | 0 |
| iid5/1B | freerider_zero | FedSV | -8.488e-03 | - |
| iid5/1B | freerider_zero | Flirds | +0.000e+00 | 0 |
| iid5/1B | freerider_zero | Flirds1st | +0.000e+00 | 0 |
| iid5/1B | freerider_zero | GTG | -6.585e-03 | - |
| iid5/1B | freerider_zero | ShapleyFL | +0.000e+00 | 0 |
| iid5/1B | freerider_zero | loss-heur | +0.000e+00 | 0 |
| iid5/1B | noisy | (b)oracle | +4.931e-03 | + |
| iid5/1B | noisy | Banzhaf | +4.932e-03 | + |
| iid5/1B | noisy | FedIF | +0.000e+00 | 0 |
| iid5/1B | noisy | FedSV | +2.000e-03 | + |
| iid5/1B | noisy | Flirds | +4.937e-03 | + |
| iid5/1B | noisy | Flirds1st | +4.949e-03 | + |
| iid5/1B | noisy | GTG | +2.592e-03 | + |
| iid5/1B | noisy | ShapleyFL | +0.000e+00 | 0 |
| iid5/1B | noisy | loss-heur | +4.942e-03 | + |
| silo5/1B | freerider_delta | (b)oracle | +2.491e-03 | + |
| silo5/1B | freerider_delta | Banzhaf | +2.491e-03 | + |
| silo5/1B | freerider_delta | ComFedSV | +1.377e-03 | + |
| silo5/1B | freerider_delta | Fed-LOO | +2.492e-03 | + |
| silo5/1B | freerider_delta | FedIF | +9.596e-01 | + |
| silo5/1B | freerider_delta | FedSV | +2.089e-03 | + |
| silo5/1B | freerider_delta | Flirds | +2.492e-03 | + |
| silo5/1B | freerider_delta | Flirds1st | +2.491e-03 | + |
| silo5/1B | freerider_delta | GTG | +2.103e-03 | + |
| silo5/1B | freerider_delta | ShapleyFL | +3.050e-01 | + |
| silo5/1B | freerider_delta | loss-heur | +2.490e-03 | + |
| silo5/1B | freerider_random | (b)oracle | -5.709e-07 | - |
| silo5/1B | freerider_random | Banzhaf | -5.215e-07 | - |
| silo5/1B | freerider_random | ComFedSV | -3.925e-03 | - |
| silo5/1B | freerider_random | Fed-LOO | -4.768e-07 | - |
| silo5/1B | freerider_random | FedIF | +0.000e+00 | 0 |
| silo5/1B | freerider_random | FedSV | -4.421e-03 | - |
| silo5/1B | freerider_random | Flirds | -5.805e-07 | - |
| silo5/1B | freerider_random | Flirds1st | -5.858e-07 | - |
| silo5/1B | freerider_random | GTG | -3.509e-03 | - |
| silo5/1B | freerider_random | ShapleyFL | +0.000e+00 | 0 |
| silo5/1B | freerider_random | loss-heur | -5.298e-07 | - |
| silo5/1B | freerider_zero | (b)oracle | +0.000e+00 | 0 |
| silo5/1B | freerider_zero | Banzhaf | +0.000e+00 | 0 |
| silo5/1B | freerider_zero | ComFedSV | -4.277e-03 | - |
| silo5/1B | freerider_zero | Fed-LOO | +0.000e+00 | 0 |
| silo5/1B | freerider_zero | FedIF | +0.000e+00 | 0 |
| silo5/1B | freerider_zero | FedSV | -4.321e-03 | - |
| silo5/1B | freerider_zero | Flirds | +0.000e+00 | 0 |
| silo5/1B | freerider_zero | Flirds1st | +0.000e+00 | 0 |
| silo5/1B | freerider_zero | GTG | -3.433e-03 | - |
| silo5/1B | freerider_zero | ShapleyFL | +0.000e+00 | 0 |
| silo5/1B | freerider_zero | loss-heur | +0.000e+00 | 0 |
| silo5/1B | noisy | (b)oracle | +1.790e-03 | + |
| silo5/1B | noisy | Banzhaf | +1.790e-03 | + |
| silo5/1B | noisy | ComFedSV | -1.953e-03 | - |
| silo5/1B | noisy | Fed-LOO | +1.872e-03 | + |
| silo5/1B | noisy | FedIF | +0.000e+00 | 0 |
| silo5/1B | noisy | FedSV | -6.756e-04 | - |
| silo5/1B | noisy | Flirds | +1.792e-03 | + |
| silo5/1B | noisy | Flirds1st | +1.778e-03 | + |
| silo5/1B | noisy | GTG | -4.860e-04 | - |
| silo5/1B | noisy | ShapleyFL | +0.000e+00 | 0 |
| silo5/1B | noisy | loss-heur | +1.775e-03 | + |
| silo5/3B | freerider_random | (b)oracle | +6.159e-08 | + |
| silo5/3B | freerider_random | FedIF | +0.000e+00 | 0 |
| silo5/3B | freerider_random | Flirds | +1.246e-07 | + |
| silo5/3B | freerider_random | Flirds1st | +1.256e-07 | + |
| silo5/3B | freerider_random | loss-heur | -4.768e-07 | - |
| silo5/3B | freerider_zero | (b)oracle | +0.000e+00 | 0 |
| silo5/3B | freerider_zero | FedIF | +0.000e+00 | 0 |
| silo5/3B | freerider_zero | Flirds | +0.000e+00 | 0 |
| silo5/3B | freerider_zero | Flirds1st | +0.000e+00 | 0 |
| silo5/3B | freerider_zero | loss-heur | +0.000e+00 | 0 |
| silo5/3B | noisy | (b)oracle | +7.283e-04 | + |
| silo5/3B | noisy | FedIF | +0.000e+00 | 0 |
| silo5/3B | noisy | Flirds | +7.294e-04 | + |
| silo5/3B | noisy | Flirds1st | +7.152e-04 | + |
| silo5/3B | noisy | loss-heur | +7.136e-04 | + |

## Recommendations (auto-computed Stage 0 outputs)

1. **LLM noisy sign-gate operating region: NONE on nr in (0,1]** -- Flirds corrupt-client cumulative contribution stays positive across the whole nr ladder (linear-extrapolated crossing Flirds ~3.44, (b)oracle ~3.44, loss-heur ~3.41); nr caps at 1.0, so the tau=0 sign gate CANNOT fire on noisy at any dose.  The §2.1 noisy@canon parity prediction is confirmed at value level; noisy recovery must come from the z-gate or V2w down-weighting, not dose escalation.  LLM dose cells for Track G: canonical nr=1.0 suffices for the parity check; at most ONE extra cell (nr=0.75, steepest decline) if a dose-trend datapoint is wanted -- a crossing hunt is pointless.

2. **GTG/FedSV sign-gates WOULD fire on noisy@canon** (crossings GTG ~0.76 / FedSV ~0.65 on the nr ladder) -- but that is their coalition-renorm value error relative to the (b) truth (+, net-helpful), not a calibrated decision; report as the value-level-fidelity -> decision-difference story, and note the in-run-(b)-game-0 vs retrain-(a)-game-0 distinction (removal canon shows removing the noisy client DOES help after retraining).

3a. **frrand @ silo5**: Flirds corrupt contribution n>0=3 / n<0=8 of 11 (mean -4.52e-07) -- the frrand cumulative sign is a NEAR-ZERO COIN FLIP, not reliably negative: the strict->0 rule catches frzero exactly, but frrand exclusion will be seed-dependent (min_obs/burn-in + per-round screen matter).  Register this as a §2.1 amendment BEFORE running.

3b. **frrand @ iid5**: Flirds corrupt contribution n>0=3 / n<0=0 of 3 (mean +7.42e-07) -- the frrand cumulative sign is a NEAR-ZERO COIN FLIP, not reliably negative: the strict->0 rule catches frzero exactly, but frrand exclusion will be seed-dependent (min_obs/burn-in + per-round screen matter).  Register this as a §2.1 amendment BEFORE running.

4. **CNN label-flip dose ladder (3 points)**: per-client-rate crossings across val methods span ~0.13-0.55 (C1 N=10 ladder, extrapolated).  Pick **rates {0.15, 0.35, 0.70}** for the C2 gate grid -- one below every crossing, one inside the span, one safely past it (C2 needs the new C2_FLIP_RATE fixed-rate knob; the legacy U(0.5,1) per-client rate sits entirely ABOVE the crossing span, which is why C2 mult already won there).


## Coverage (every phi file found under runs/)

| run-dir | family | rows |
|---|---|---|
| measured_2026-07/e3_cost_smoke/cifar10_iid_seed0 | cnn_wide | 10 |
| measured_2026-07/e3_cost_smoke/mnist_iid_seed0 | cnn_wide | 10 |
| measured_2026-07/taylor/llama1b_r10_seed0 | cnn_wide | 5 |
| measured_2026-07/taylor/llama1b_r10_seed1 | cnn_wide | 5 |
| measured_2026-07/taylor/llama1b_r10_seed2 | cnn_wide | 5 |
| measured_2026-07/tf32_ab/cifar10_iid_tf32off_seed0 | cnn_wide | 10 |
| measured_2026-07/tf32_ab/cifar10_iid_tf32on_seed0 | cnn_wide | 10 |
| measured_2026-07/tf32_ab/cifar10_label-flip_tf32off_seed0 | cnn_wide | 10 |
| measured_2026-07/tf32_ab/cifar10_label-flip_tf32on_seed0 | cnn_wide | 10 |
| measured_2026-07/timing_device100/1B_device100-a0.5_clean | phase2 | 588 |
| phase1/rundirs/1B_silo5_full-lr1e-3_seed0 | cnn_wide | 5 |
| phase1/rundirs/1B_silo5_full-lr1e-3_seed1 | cnn_wide | 5 |
| phase1/rundirs/1B_silo5_full-lr1e-3_seed2 | cnn_wide | 5 |
| phase1/rundirs/1B_silo5_full-lr3e-3_seed0 | cnn_wide | 5 |
| phase1/rundirs/1B_silo5_full-lr3e-3_seed1 | cnn_wide | 5 |
| phase1/rundirs/1B_silo5_full-lr3e-3_seed2 | cnn_wide | 5 |
| phase1/rundirs/1B_silo5_mini_seed0 | cnn_wide | 5 |
| phase1/rundirs/1B_silo5_smoke_seed0 | cnn_wide | 5 |
| phase1/rundirs/1B_silo5_sweep-lr1e-3_seed0 | cnn_wide | 5 |
| phase1/rundirs/1B_silo5_sweep-lr1e-4_seed0 | cnn_wide | 5 |
| phase1/rundirs/1B_silo5_sweep-lr3e-3_seed0 | cnn_wide | 5 |
| phase1/rundirs/1B_silo5_sweep-lr3e-4_seed0 | cnn_wide | 5 |
| phase2_matrix/rundirs/1B_device100-a0.01_frrand | phase2 | 1450 |
| phase2_matrix/rundirs/1B_device100-a0.01_frzero | phase2 | 1450 |
| phase2_matrix/rundirs/1B_device100-a0.01_noisy | phase2 | 1450 |
| phase2_matrix/rundirs/1B_device100-a0.0_frrand | phase2 | 1450 |
| phase2_matrix/rundirs/1B_device100-a0.0_frzero | phase2 | 1450 |
| phase2_matrix/rundirs/1B_device100-a0.0_noisy | phase2 | 1450 |
| phase2_matrix/rundirs/1B_device100-a0.0_poison | phase2 | 1500 |
| phase2_matrix/rundirs/1B_device100-a0.1_frrand | phase2 | 1450 |
| phase2_matrix/rundirs/1B_device100-a0.1_frzero | phase2 | 1450 |
| phase2_matrix/rundirs/1B_device100-a0.1_noisy | phase2 | 1450 |
| phase2_matrix/rundirs/1B_device100-a0.5_frrand_anchor | phase2 | 2610 |
| phase2_matrix/rundirs/1B_device100-a0.5_frzero_anchor | phase2 | 2610 |
| phase2_matrix/rundirs/1B_device100-a0.5_noisy_anchor | phase2 | 2610 |
| phase2_matrix/rundirs/1B_device100-a0.5_poison | phase2 | 1500 |
| phase2_matrix/rundirs/1B_device100-a5.0_frrand | phase2 | 1450 |
| phase2_matrix/rundirs/1B_device100-a5.0_frzero | phase2 | 1450 |
| phase2_matrix/rundirs/1B_device100-a5.0_noisy | phase2 | 1450 |
| phase2_matrix/rundirs/1B_iid5_clean | phase2 | 135 |
| phase2_matrix/rundirs/1B_iid5_frrand | phase2 | 135 |
| phase2_matrix/rundirs/1B_iid5_frzero | phase2 | 135 |
| phase2_matrix/rundirs/1B_iid5_noisy | phase2 | 135 |
| phase2_matrix/rundirs/1B_iid5_poison | phase2 | 135 |
| phase2_matrix/rundirs/1B_silo5_clean | phase2 | 135 |
| phase2_matrix/rundirs/1B_silo5_frrand | phase2 | 135 |
| phase2_matrix/rundirs/1B_silo5_frzero | phase2 | 135 |
| phase2_matrix/rundirs/1B_silo5_noisy | phase2 | 135 |
| phase2_matrix/rundirs/1B_silo5_poison | phase2 | 135 |
| phase2_matrix/rundirs/3B_silo5_frrand | phase2 | 25 |
| phase2_matrix/rundirs/3B_silo5_frzero | phase2 | 25 |
| phase2_matrix/rundirs/3B_silo5_noisy | phase2 | 25 |
| phase2_matrix/rundirs/3B_silo5_poison | phase2 | 25 |
| phase2_matrix/rundirs_2026-07/1B_device100-a0.01_frrand | phase2 | 1450 |
| phase2_matrix/rundirs_2026-07/1B_device100-a0.01_frzero | phase2 | 1450 |
| phase2_matrix/rundirs_2026-07/1B_device100-a0.01_noisy | phase2 | 1450 |
| phase2_matrix/rundirs_2026-07/1B_device100-a0.0_frrand | phase2 | 1450 |
| phase2_matrix/rundirs_2026-07/1B_device100-a0.0_frzero | phase2 | 1450 |
| phase2_matrix/rundirs_2026-07/1B_device100-a0.0_noisy | phase2 | 1450 |
| phase2_matrix/rundirs_2026-07/1B_device100-a0.0_poison | phase2 | 1500 |
| phase2_matrix/rundirs_2026-07/1B_device100-a0.1_frrand | phase2 | 1450 |
| phase2_matrix/rundirs_2026-07/1B_device100-a0.1_frzero | phase2 | 1450 |
| phase2_matrix/rundirs_2026-07/1B_device100-a0.1_noisy | phase2 | 1450 |
| phase2_matrix/rundirs_2026-07/1B_device100-a0.5_poison | phase2 | 1500 |
| phase2_matrix/rundirs_2026-07/1B_device100-a5.0_frrand | phase2 | 1450 |
| phase2_matrix/rundirs_2026-07/1B_device100-a5.0_frzero | phase2 | 1450 |
| phase2_matrix/rundirs_2026-07/1B_device100-a5.0_noisy | phase2 | 1450 |
| phase2_matrix/rundirs_2026-07/1B_silo5_frdelta | phase2 | 165 |
| phase2_matrix/rundirs_2026-07/1B_silo5_frrand | phase2 | 135 |
| phase2_matrix/rundirs_2026-07/1B_silo5_frzero | phase2 | 135 |
| phase2_matrix/rundirs_2026-07/1B_silo5_noisy | phase2 | 135 |
| phase2_matrix/rundirs_2026-07/1B_silo5_poison | phase2 | 135 |
| phase2_matrix/rundirs_2026-07/3B_silo5_frrand | phase2 | 25 |
| phase2_matrix/rundirs_2026-07/3B_silo5_frzero | phase2 | 25 |
| phase2_matrix/rundirs_2026-07/3B_silo5_noisy | phase2 | 25 |
| phase2_matrix/rundirs_2026-07/3B_silo5_poison | phase2 | 25 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w0.5_k0.2_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w0.5_k0.2_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w0.5_k0.2_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w0.5_k0.5_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w0.5_k0.5_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w0.5_k0.5_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w0.5_k1.0_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w0.5_k1.0_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w0.5_k1.0_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w1_k0.2_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w1_k0.2_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w1_k0.2_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w1_k0.5_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w1_k0.5_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w1_k0.5_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w2_k0.2_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w2_k0.2_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w2_k0.2_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w2_k0.5_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w2_k0.5_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w2_k0.5_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w2_k1.0_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w2_k1.0_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w2_k1.0_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w4_k0.2_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w4_k0.2_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w4_k0.2_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w4_k0.5_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w4_k0.5_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w4_k0.5_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w4_k1.0_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w4_k1.0_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_iid_w4_k1.0_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w0.5_k0.2_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w0.5_k0.2_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w0.5_k0.2_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w0.5_k0.5_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w0.5_k0.5_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w0.5_k0.5_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w0.5_k1.0_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w0.5_k1.0_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w0.5_k1.0_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w1_k0.2_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w1_k0.2_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w1_k0.2_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w1_k0.5_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w1_k0.5_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w1_k0.5_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w2_k0.2_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w2_k0.2_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w2_k0.2_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w2_k0.5_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w2_k0.5_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w2_k0.5_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w2_k1.0_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w2_k1.0_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w2_k1.0_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w4_k0.2_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w4_k0.2_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w4_k0.2_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w4_k0.5_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w4_k0.5_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w4_k0.5_seed2 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w4_k1.0_seed0 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w4_k1.0_seed1 | cnn_wide | 10 |
| probe_signal/cnn_c1/pc1_cifar10_label-flip_w4_k1.0_seed2 | cnn_wide | 10 |
| probe_signal/rundirs/1B_anchor5_lr1e-3_st20_seed0 | track_d | 55 |
| probe_signal/rundirs/1B_anchor5_lr1e-3_st30_seed0 | track_d | 55 |
| probe_signal/rundirs/1B_anchor5_lr2e-3_st10_seed0 | track_d | 55 |
| probe_signal/rundirs/1B_anchor5_lr2e-3_st20_seed0 | track_d | 55 |
| probe_signal/rundirs/1B_anchor5_lr2e-3_st30_seed0 | track_d | 55 |
| probe_signal/rundirs/1B_anchor5_lr3e-3_st10_seed0 | track_d | 55 |
| probe_signal/rundirs/1B_anchor5_lr3e-3_st20_seed0 | track_d | 55 |
| probe_signal/rundirs/1B_anchor5_lr3e-3_st30_seed0 | track_d | 55 |
| probe_signal/rundirs/1B_anchor5_r32_seed0 | track_d | 50 |
| probe_signal/rundirs/1B_anchor5_r64_seed0 | track_d | 55 |
| probe_signal/rundirs/1B_std50k5_r16_seed0 | track_d | 500 |
| probe_signal/rundirs/1B_std50k5_r32_seed0 | track_d | 500 |
| probe_signal/rundirs/1B_std50k5_r64_seed0 | track_d | 500 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm0.25_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm0.25_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm0.25_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm0.5_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm0.5_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm0.5_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm1.0_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm1.0_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm1.0_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm2.0_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm2.0_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm2.0_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm4.0_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm4.0_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_dose_dm4.0_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_removal_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_removal_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frrand_removal_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frzero_removal_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frzero_removal_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_frzero_removal_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0.1_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0.1_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0.1_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0.25_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0.25_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0.25_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0.5_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0.5_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0.5_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0.75_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0.75_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0.75_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr0_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr1.0_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr1.0_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_dose_nr1.0_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_removal_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_removal_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_noisy_removal_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.1_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.1_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.1_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.2_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.2_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.2_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.3_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.3_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.3_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.4_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.4_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.4_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.5_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.5_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.5_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.6_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.6_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.6_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.7_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.7_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.7_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.8_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.8_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.8_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.9_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.9_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf0.9_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf1.0_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf1.0_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_dose_pf1.0_seed2 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_removal_seed0 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_removal_seed1 | phase2 | 55 |
| removal_dose/rundirs/1B_silo5_poison_removal_seed2 | phase2 | 55 |
| removal_dose/rundirs_cnn/cifar10_feature-noise_seed0 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/cifar10_feature-noise_seed1 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/cifar10_feature-noise_seed2 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/cifar10_iid_seed0 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/cifar10_iid_seed1 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/cifar10_iid_seed2 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/cifar10_label-flip_seed0 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/cifar10_label-flip_seed1 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/cifar10_label-flip_seed2 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/mnist_feature-noise_seed0 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/mnist_feature-noise_seed1 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/mnist_feature-noise_seed2 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/mnist_iid_seed0 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/mnist_iid_seed1 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/mnist_iid_seed2 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/mnist_label-flip_seed0 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/mnist_label-flip_seed1 | cnn_wide | 10 |
| removal_dose/rundirs_cnn/mnist_label-flip_seed2 | cnn_wide | 10 |
| removal_dose/rundirs_trackd/1B_anchor5_adamw_seed0 | track_d | 60 |
| removal_dose/rundirs_trackd/1B_anchor5_adamw_seed1 | track_d | 60 |
| removal_dose/rundirs_trackd/1B_anchor5_adamw_seed2 | track_d | 60 |
| removal_dose/rundirs_trackd/1B_anchor5_removal_seed0 | track_d | 60 |
| removal_dose/rundirs_trackd/1B_anchor5_removal_seed1 | track_d | 60 |
| removal_dose/rundirs_trackd/1B_anchor5_removal_seed2 | track_d | 60 |
| track_c/c1/cifar10_feature-noise_seed0 | cnn_wide | 10 |
| track_c/c1/cifar10_feature-noise_seed1 | cnn_wide | 10 |
| track_c/c1/cifar10_feature-noise_seed2 | cnn_wide | 10 |
| track_c/c1/cifar10_iid_seed0 | cnn_wide | 10 |
| track_c/c1/cifar10_iid_seed1 | cnn_wide | 10 |
| track_c/c1/cifar10_iid_seed2 | cnn_wide | 10 |
| track_c/c1/cifar10_label-flip_seed0 | cnn_wide | 10 |
| track_c/c1/cifar10_label-flip_seed1 | cnn_wide | 10 |
| track_c/c1/cifar10_label-flip_seed2 | cnn_wide | 10 |
| track_c/c1/cifar10_label-skew_seed0 | cnn_wide | 10 |
| track_c/c1/cifar10_label-skew_seed1 | cnn_wide | 10 |
| track_c/c1/cifar10_label-skew_seed2 | cnn_wide | 10 |
| track_c/c1/cifar10_quantity-skew_seed0 | cnn_wide | 10 |
| track_c/c1/cifar10_quantity-skew_seed1 | cnn_wide | 10 |
| track_c/c1/cifar10_quantity-skew_seed2 | cnn_wide | 10 |
| track_c/c1/mnist_feature-noise_seed0 | cnn_wide | 10 |
| track_c/c1/mnist_feature-noise_seed1 | cnn_wide | 10 |
| track_c/c1/mnist_feature-noise_seed2 | cnn_wide | 10 |
| track_c/c1/mnist_iid_seed0 | cnn_wide | 10 |
| track_c/c1/mnist_iid_seed1 | cnn_wide | 10 |
| track_c/c1/mnist_iid_seed2 | cnn_wide | 10 |
| track_c/c1/mnist_label-flip_seed0 | cnn_wide | 10 |
| track_c/c1/mnist_label-flip_seed1 | cnn_wide | 10 |
| track_c/c1/mnist_label-flip_seed2 | cnn_wide | 10 |
| track_c/c1/mnist_label-skew_seed0 | cnn_wide | 10 |
| track_c/c1/mnist_label-skew_seed1 | cnn_wide | 10 |
| track_c/c1/mnist_label-skew_seed2 | cnn_wide | 10 |
| track_c/c1/mnist_quantity-skew_seed0 | cnn_wide | 10 |
| track_c/c1/mnist_quantity-skew_seed1 | cnn_wide | 10 |
| track_c/c1/mnist_quantity-skew_seed2 | cnn_wide | 10 |
| track_d/rundirs/1B_anchor5_seed0 | track_d | 55 |
| track_d/rundirs/1B_anchor5_seed1 | track_d | 55 |
| track_d/rundirs/1B_anchor5_seed2 | track_d | 55 |
| track_d/rundirs/1B_std20_seed0 | track_d | 180 |
| track_d/rundirs/1B_std20_seed1 | track_d | 180 |
| track_d/rundirs/1B_std20_seed2 | track_d | 180 |
| track_d/rundirs/3B_anchor5_seed0 | track_d | 50 |
| track_d/rundirs/3B_anchor5_seed1 | track_d | 50 |
| track_d/rundirs/3B_anchor5_seed2 | track_d | 50 |
| track_d/rundirs/3B_std20_seed0 | track_d | 180 |
| track_d/rundirs/3B_std20_seed1 | track_d | 180 |
| track_d/rundirs/3B_std20_seed2 | track_d | 180 |
| track_d/rundirs/7B_anchor5_seed0 | track_d | 50 |
| track_d/rundirs/7B_anchor5_seed1 | track_d | 50 |
| track_d/rundirs/7B_anchor5_seed2 | track_d | 50 |
| track_d/rundirs/7B_std20_seed0 | track_d | 180 |
| track_d/rundirs/7B_std20_seed1 | track_d | 180 |
| track_d/rundirs/7B_std20_seed2 | track_d | 180 |
| track_d/rundirs_e4_fedloo/1B_anchor5_seed0 | track_d | 25 |
| track_d/rundirs_e4_fedloo/1B_anchor5_seed1 | track_d | 25 |
| track_d/rundirs_e4_fedloo/1B_anchor5_seed2 | track_d | 25 |
| track_d/rundirs_e4_fedloo/1B_std20_seed0 | track_d | 100 |
| track_d/rundirs_e4_fedloo/1B_std20_seed1 | track_d | 100 |
| track_d/rundirs_e4_fedloo/1B_std20_seed2 | track_d | 100 |
| track_d/rundirs_e5_n10/1B_anchor10_seed0 | track_d | 50 |

Run-dirs with metrics but NO per-client phi (not auditable for sign): track_c/c2 + probe_signal/cnn_c2 arm cells (OnlineScorer state was never persisted -- the Track G per-round phi_rounds.parquet logging closes exactly this gap going forward).
