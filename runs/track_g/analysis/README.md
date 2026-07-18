# Track G analysis (auto-generated from rundirs -- rerun make_analysis.py)

## [1] performance delta + recovery  (delta = vanilla_loss - arm_loss, +=better; recovery = delta / (vanilla - oracle_excl))

| regime | threat | nr | seed | arm | final_val_loss | delta | recovery | mmlu | rouge_l | prediction | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| silo5 | clean | 1 | 0 | flirds_gate_v1 | 2.2936 | +0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 0 | flirds_gate_v2 | 2.2936 | -0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 0 | flirds_w | 2.2887 | +0.0049 |  |  |  |  |  |
| silo5 | clean | 1 | 0 | flirds_zgate_v2 | 2.2936 | -0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 0 | lossheur_gate_v2 | 2.2936 | -0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 0 | oracleb_gate_v2 | 2.2936 | -0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 0 | v3_random | 2.2936 | +0.0000 |  |  |  |  |  |
| silo5 | clean | 1 | 0 | v3_sign | 2.2936 | +0.0000 |  |  |  | parity |  |
| silo5 | clean | 1 | 0 | v3_z | 2.2936 | +0.0000 |  |  |  | parity |  |
| silo5 | clean | 1 | 0 | vanilla | 2.2936 | +0.0000 |  |  |  |  |  |
| silo5 | clean | 1 | 1 | flirds_gate_v1 | 2.3851 | +0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 1 | flirds_gate_v2 | 2.3851 | -0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 1 | flirds_w | 2.3806 | +0.0045 |  |  |  |  |  |
| silo5 | clean | 1 | 1 | flirds_zgate_v2 | 2.3851 | -0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 1 | lossheur_gate_v2 | 2.3851 | -0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 1 | oracleb_gate_v2 | 2.3851 | -0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 1 | v3_random | 2.3851 | +0.0000 |  |  |  |  |  |
| silo5 | clean | 1 | 1 | v3_sign | 2.3851 | +0.0000 |  |  |  | parity |  |
| silo5 | clean | 1 | 1 | v3_z | 2.3851 | +0.0000 |  |  |  | parity |  |
| silo5 | clean | 1 | 1 | vanilla | 2.3851 | +0.0000 |  |  |  |  |  |
| silo5 | noisy | 1 | 0 | flirds_gate_v1 | 2.2962 | +0.0000 | +0.0000 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 1 | 0 | oracle_excl | 2.2946 | +0.0016 | +1.0000 |  |  |  |  |
| silo5 | noisy | 1 | 0 | random_excl | 2.2963 | -0.0000 | -0.0237 |  |  |  |  |
| silo5 | noisy | 1 | 0 | vanilla | 2.2962 | +0.0000 | +0.0000 |  |  |  |  |

## [2] convergence (rounds-to-target = first round entering-loss <= the cell's vanilla final loss)

| regime | threat | nr | seed | arm | rounds_to_target |
|---|---|---|---|---|---|
| silo5 | clean | 1 | 0 | flirds_gate_v1 | 10.0000 |
| silo5 | clean | 1 | 0 | flirds_gate_v2 |  |
| silo5 | clean | 1 | 0 | flirds_w | 8.0000 |
| silo5 | clean | 1 | 0 | flirds_zgate_v2 |  |
| silo5 | clean | 1 | 0 | lossheur_gate_v2 |  |
| silo5 | clean | 1 | 0 | oracleb_gate_v2 |  |
| silo5 | clean | 1 | 0 | v3_random |  |
| silo5 | clean | 1 | 0 | v3_sign |  |
| silo5 | clean | 1 | 0 | v3_z |  |
| silo5 | clean | 1 | 0 | vanilla | 10.0000 |
| silo5 | clean | 1 | 1 | flirds_gate_v1 | 10.0000 |
| silo5 | clean | 1 | 1 | flirds_gate_v2 |  |
| silo5 | clean | 1 | 1 | flirds_w | 8.0000 |
| silo5 | clean | 1 | 1 | flirds_zgate_v2 |  |
| silo5 | clean | 1 | 1 | lossheur_gate_v2 |  |
| silo5 | clean | 1 | 1 | oracleb_gate_v2 |  |
| silo5 | clean | 1 | 1 | v3_random |  |
| silo5 | clean | 1 | 1 | v3_sign |  |
| silo5 | clean | 1 | 1 | v3_z |  |
| silo5 | clean | 1 | 1 | vanilla | 10.0000 |
| silo5 | noisy | 1 | 0 | flirds_gate_v1 | 10.0000 |
| silo5 | noisy | 1 | 0 | oracle_excl | 10.0000 |
| silo5 | noisy | 1 | 0 | random_excl |  |
| silo5 | noisy | 1 | 0 | vanilla | 10.0000 |

## [3] gate accuracy (per-round excluded set vs corrupt; micro P/R) + vanilla-observer per-round false-fire

| regime | threat | nr | seed | arm | precision | recall | n_excluded_pairs | false_excl_pairs | n_fallback_rounds |
|---|---|---|---|---|---|---|---|---|---|
| silo5 | clean | 1 | 0 | flirds_gate_v1 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 0 | flirds_gate_v2 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 0 | flirds_zgate_v2 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 0 | lossheur_gate_v2 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 0 | oracleb_gate_v2 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 0 | vanilla |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 1 | flirds_gate_v1 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 1 | flirds_gate_v2 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 1 | flirds_zgate_v2 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 1 | lossheur_gate_v2 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 1 | oracleb_gate_v2 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 1 | vanilla |  |  | 0 | 0 | 0 |
| silo5 | noisy | 1 | 0 | flirds_gate_v1 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 0 | vanilla |  | 0.0000 | 0 | 0 | 0 |

vanilla observer (per-round raw, the project's first per-round phi record):

| regime | threat | nr | seed | clean_raw_false_fire_rate | all_clean_cum_pos_from_round |
|---|---|---|---|---|---|
| silo5 | clean | 1 | 0 | 0.0000 | 0.0000 |
| silo5 | clean | 1 | 1 | 0.0000 | 0.0000 |
| silo5 | noisy | 1 | 0 | 0.0000 | 0.0000 |

## CNN (track_c2 gate cells)

| dataset | partition | threat | strength | flip_rate | seed | arm | final_acc | delta_acc | recovery | auroc |
|---|---|---|---|---|---|---|---|---|---|---|
| cifar10 | dir1 | clean | main |  | 0 | flirds_gate_v1 | 0.6448 | +0.0059 |  |  |
| cifar10 | dir1 | clean | main |  | 0 | flirds_gate_v2 | 0.6246 | -0.0142 |  |  |
| cifar10 | dir1 | clean | main |  | 0 | flirds_gatew_v1 | 0.6419 | +0.0030 |  |  |
| cifar10 | dir1 | clean | main |  | 0 | flirds_gatew_v2 | 0.6189 | -0.0200 |  |  |
| cifar10 | dir1 | clean | main |  | 0 | flirds_mult | 0.6434 | +0.0045 |  |  |
| cifar10 | dir1 | clean | main |  | 0 | flirds_zgate_v2 | 0.6345 | -0.0044 |  |  |
| cifar10 | dir1 | clean | main |  | 0 | vanilla | 0.6389 | +0.0000 |  |  |
| cifar10 | dir1 | clean | main |  | 1 | flirds_gate_v1 | 0.6384 | +0.0048 |  |  |
| cifar10 | dir1 | clean | main |  | 1 | flirds_gate_v2 | 0.6394 | +0.0058 |  |  |
| cifar10 | dir1 | clean | main |  | 1 | flirds_gatew_v1 | 0.6321 | -0.0015 |  |  |
| cifar10 | dir1 | clean | main |  | 1 | flirds_gatew_v2 | 0.6236 | -0.0100 |  |  |
| cifar10 | dir1 | clean | main |  | 1 | flirds_mult | 0.6394 | +0.0058 |  |  |
| cifar10 | dir1 | clean | main |  | 1 | flirds_zgate_v2 | 0.6424 | +0.0088 |  |  |
| cifar10 | dir1 | clean | main |  | 1 | vanilla | 0.6336 | +0.0000 |  |  |
| cifar10 | dir1 | clean | main |  | 2 | flirds_gate_v1 | 0.6239 | -0.0202 |  |  |
| cifar10 | dir1 | clean | main |  | 2 | flirds_gate_v2 | 0.6305 | -0.0136 |  |  |
| cifar10 | dir1 | clean | main |  | 2 | flirds_gatew_v1 | 0.5854 | -0.0587 |  |  |
| cifar10 | dir1 | clean | main |  | 2 | flirds_gatew_v2 | 0.6139 | -0.0302 |  |  |
| cifar10 | dir1 | clean | main |  | 2 | flirds_mult | 0.6449 | +0.0008 |  |  |
| cifar10 | dir1 | clean | main |  | 2 | flirds_zgate_v2 | 0.6254 | -0.0187 |  |  |
| cifar10 | dir1 | clean | main |  | 2 | vanilla | 0.6441 | +0.0000 |  |  |
| cifar10 | dir1 | free_rider | main |  | 0 | flirds_gate_v1 | 0.6174 | +0.0306 | +0.9108 | 0.9833 |
| cifar10 | dir1 | free_rider | main |  | 0 | flirds_gate_v2 | 0.6146 | +0.0279 | +0.8290 | 0.7833 |
| cifar10 | dir1 | free_rider | main |  | 0 | flirds_gatew_v1 | 0.6244 | +0.0376 | +1.1190 | 1.0000 |
| cifar10 | dir1 | free_rider | main |  | 0 | flirds_gatew_v2 | 0.6114 | +0.0246 | +0.7323 | 0.7000 |
| cifar10 | dir1 | free_rider | main |  | 0 | flirds_mult | 0.5980 | +0.0112 | +0.3346 | 0.3571 |
| cifar10 | dir1 | free_rider | main |  | 0 | flirds_zgate_v2 | 0.5817 | -0.0050 | -0.1487 | 0.5167 |
| cifar10 | dir1 | free_rider | main |  | 0 | oracle_excl | 0.6204 | +0.0336 | +1.0000 |  |
| cifar10 | dir1 | free_rider | main |  | 0 | random_excl | 0.5930 | +0.0062 | +0.1859 |  |
| cifar10 | dir1 | free_rider | main |  | 0 | vanilla | 0.5867 | +0.0000 | +0.0000 |  |
| cifar10 | dir1 | free_rider | main |  | 1 | flirds_gate_v1 | 0.6131 | +0.0219 | +0.8333 | 0.9667 |
| cifar10 | dir1 | free_rider | main |  | 1 | flirds_gate_v2 | 0.6150 | +0.0237 | +0.9048 | 0.6667 |
| cifar10 | dir1 | free_rider | main |  | 1 | flirds_gatew_v1 | 0.6145 | +0.0232 | +0.8857 | 0.9667 |
| cifar10 | dir1 | free_rider | main |  | 1 | flirds_gatew_v2 | 0.6146 | +0.0234 | +0.8905 | 0.6833 |
| cifar10 | dir1 | free_rider | main |  | 1 | flirds_mult | 0.5994 | +0.0081 | +0.3095 | 0.4313 |
| cifar10 | dir1 | free_rider | main |  | 1 | flirds_zgate_v2 | 0.5827 | -0.0085 | -0.3238 | 0.6167 |
| cifar10 | dir1 | free_rider | main |  | 1 | oracle_excl | 0.6175 | +0.0262 | +1.0000 |  |
| cifar10 | dir1 | free_rider | main |  | 1 | random_excl | 0.5979 | +0.0066 | +0.2524 |  |
| cifar10 | dir1 | free_rider | main |  | 1 | vanilla | 0.5913 | +0.0000 | +0.0000 |  |
| cifar10 | dir1 | free_rider | main |  | 2 | flirds_gate_v1 | 0.6012 | +0.0155 | +0.4147 | 1.0000 |
| cifar10 | dir1 | free_rider | main |  | 2 | flirds_gate_v2 | 0.6149 | +0.0291 | +0.7793 | 0.6667 |
| cifar10 | dir1 | free_rider | main |  | 2 | flirds_gatew_v1 | 0.5950 | +0.0092 | +0.2475 | 1.0000 |
| cifar10 | dir1 | free_rider | main |  | 2 | flirds_gatew_v2 | 0.5909 | +0.0051 | +0.1371 | 0.7167 |
| cifar10 | dir1 | free_rider | main |  | 2 | flirds_mult | 0.5962 | +0.0105 | +0.2809 | 0.4025 |
| cifar10 | dir1 | free_rider | main |  | 2 | flirds_zgate_v2 | 0.5875 | +0.0018 | +0.0468 | 0.5167 |
| cifar10 | dir1 | free_rider | main |  | 2 | oracle_excl | 0.6231 | +0.0374 | +1.0000 |  |
| cifar10 | dir1 | free_rider | main |  | 2 | random_excl | 0.5606 | -0.0251 | -0.6722 |  |
| cifar10 | dir1 | free_rider | main |  | 2 | vanilla | 0.5857 | +0.0000 | +0.0000 |  |
| cifar10 | dir1 | grad_noise | main |  | 0 | flirds_gate_v1 | 0.5724 | +0.3066 | +0.8646 | 0.9987 |
| cifar10 | dir1 | grad_noise | main |  | 0 | flirds_gate_v2 | 0.5370 | +0.2713 | +0.7649 | 0.9892 |
| cifar10 | dir1 | grad_noise | main |  | 0 | flirds_gatew_v1 | 0.5449 | +0.2791 | +0.7871 | 0.9996 |
| cifar10 | dir1 | grad_noise | main |  | 0 | flirds_gatew_v2 | 0.5783 | +0.3125 | +0.8812 | 0.9788 |
| cifar10 | dir1 | grad_noise | main |  | 0 | flirds_mult | 0.4629 | +0.1971 | +0.5559 | 0.9925 |
| cifar10 | dir1 | grad_noise | main |  | 0 | flirds_zgate_v2 | 0.3789 | +0.1131 | +0.3190 | 0.9967 |
| cifar10 | dir1 | grad_noise | main |  | 0 | oracle_excl | 0.6204 | +0.3546 | +1.0000 |  |
| cifar10 | dir1 | grad_noise | main |  | 0 | random_excl | 0.2411 | -0.0246 | -0.0694 |  |
| cifar10 | dir1 | grad_noise | main |  | 0 | vanilla | 0.2657 | +0.0000 | +0.0000 |  |
| cifar10 | dir1 | grad_noise | main |  | 1 | flirds_gate_v1 | 0.5129 | +0.2915 | +0.7359 | 0.9996 |
| cifar10 | dir1 | grad_noise | main |  | 1 | flirds_gate_v2 | 0.5853 | +0.3639 | +0.9186 | 0.9950 |
| cifar10 | dir1 | grad_noise | main |  | 1 | flirds_gatew_v1 | 0.5058 | +0.2844 | +0.7179 | 0.9983 |
| cifar10 | dir1 | grad_noise | main |  | 1 | flirds_gatew_v2 | 0.5944 | +0.3730 | +0.9416 | 0.9954 |
| cifar10 | dir1 | grad_noise | main |  | 1 | flirds_mult | 0.4457 | +0.2244 | +0.5664 | 0.9975 |
| cifar10 | dir1 | grad_noise | main |  | 1 | flirds_zgate_v2 | 0.3287 | +0.1074 | +0.2711 | 0.9946 |
| cifar10 | dir1 | grad_noise | main |  | 1 | oracle_excl | 0.6175 | +0.3961 | +1.0000 |  |
| cifar10 | dir1 | grad_noise | main |  | 1 | random_excl | 0.2819 | +0.0605 | +0.1527 |  |
| cifar10 | dir1 | grad_noise | main |  | 1 | vanilla | 0.2214 | +0.0000 | +0.0000 |  |
| cifar10 | dir1 | grad_noise | main |  | 2 | flirds_gate_v1 | 0.5533 | +0.3095 | +0.8158 | 1.0000 |
| cifar10 | dir1 | grad_noise | main |  | 2 | flirds_gate_v2 | 0.5781 | +0.3344 | +0.8814 | 0.9912 |
| cifar10 | dir1 | grad_noise | main |  | 2 | flirds_gatew_v1 | 0.5034 | +0.2596 | +0.6843 | 1.0000 |
| cifar10 | dir1 | grad_noise | main |  | 2 | flirds_gatew_v2 | 0.5896 | +0.3459 | +0.9117 | 0.9946 |
| cifar10 | dir1 | grad_noise | main |  | 2 | flirds_mult | 0.4005 | +0.1568 | +0.4132 | 0.9908 |
| cifar10 | dir1 | grad_noise | main |  | 2 | flirds_zgate_v2 | 0.3180 | +0.0743 | +0.1957 | 1.0000 |
| cifar10 | dir1 | grad_noise | main |  | 2 | oracle_excl | 0.6231 | +0.3794 | +1.0000 |  |
| cifar10 | dir1 | grad_noise | main |  | 2 | random_excl | 0.2541 | +0.0104 | +0.0273 |  |
| cifar10 | dir1 | grad_noise | main |  | 2 | vanilla | 0.2437 | +0.0000 | +0.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_gate_v1 | 0.6235 | +0.0015 | +0.2927 | 0.5393 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_gate_v1 | 0.6061 | +0.0074 | +0.2599 | 0.7306 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_gate_v1 | 0.6065 | +0.0505 | +0.7100 | 0.9924 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_gate_v2 | 0.6159 | -0.0061 | -1.1951 | 0.6049 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_gate_v2 | 0.5936 | -0.0051 | -0.1806 | 0.6604 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_gate_v2 | 0.5641 | +0.0081 | +0.1142 | 0.8083 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_gatew_v1 | 0.6224 | +0.0004 | +0.0732 | 0.5721 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_gatew_v1 | 0.6012 | +0.0025 | +0.0881 | 0.6789 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_gatew_v1 | 0.5360 | -0.0200 | -0.2812 | 0.9462 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_gatew_v2 | 0.5978 | -0.0242 | -4.7317 | 0.4426 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_gatew_v2 | 0.5833 | -0.0155 | -0.5463 | 0.5532 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_gatew_v2 | 0.5938 | +0.0377 | +0.5308 | 0.6679 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_mult | 0.6246 | +0.0026 | +0.5122 | 0.4435 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_mult | 0.6124 | +0.0136 | +0.4802 | 0.6646 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_mult | 0.5995 | +0.0435 | +0.6116 | 0.8676 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_zgate_v2 | 0.6148 | -0.0072 | -1.4146 | 0.5351 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_zgate_v2 | 0.6088 | +0.0100 | +0.3524 | 0.9092 |
| cifar10 | dir1 | label_flip | main |  | 0 | flirds_zgate_v2 | 0.5889 | +0.0329 | +0.4622 | 0.9899 |
| cifar10 | dir1 | label_flip | main |  | 0 | oracle_excl | 0.6271 | +0.0051 | +1.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 0 | oracle_excl | 0.6271 | +0.0284 | +1.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 0 | oracle_excl | 0.6271 | +0.0711 | +1.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 0 | random_excl | 0.6066 | -0.0154 | -3.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 0 | random_excl | 0.5817 | -0.0170 | -0.5991 |  |
| cifar10 | dir1 | label_flip | main |  | 0 | random_excl | 0.5599 | +0.0039 | +0.0545 |  |
| cifar10 | dir1 | label_flip | main |  | 0 | vanilla | 0.6220 | +0.0000 | +0.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 0 | vanilla | 0.5988 | +0.0000 | +0.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 0 | vanilla | 0.5560 | +0.0000 | +0.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_gate_v1 | 0.6122 | -0.0019 | -0.2586 | 0.5349 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_gate_v1 | 0.5797 | +0.0051 | +0.1096 | 0.7845 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_gate_v1 | 0.5525 | +0.0534 | +0.4366 | 0.9732 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_gate_v2 | 0.5951 | -0.0190 | -2.6207 | 0.4796 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_gate_v2 | 0.5745 | -0.0001 | -0.0027 | 0.6963 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_gate_v2 | 0.5690 | +0.0699 | +0.5716 | 0.8293 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_gatew_v1 | 0.6124 | -0.0018 | -0.2414 | 0.5240 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_gatew_v1 | 0.5914 | +0.0167 | +0.3583 | 0.7969 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_gatew_v1 | 0.5410 | +0.0419 | +0.3425 | 0.9567 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_gatew_v2 | 0.5881 | -0.0260 | -3.5862 | 0.4319 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_gatew_v2 | 0.5811 | +0.0065 | +0.1390 | 0.6635 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_gatew_v2 | 0.5784 | +0.0792 | +0.6483 | 0.7825 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_mult | 0.6165 | +0.0024 | +0.3276 | 0.5557 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_mult | 0.5881 | +0.0135 | +0.2888 | 0.8550 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_mult | 0.5755 | +0.0764 | +0.6247 | 0.9339 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_zgate_v2 | 0.6200 | +0.0059 | +0.8103 | 0.5877 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_zgate_v2 | 0.5819 | +0.0072 | +0.1551 | 0.9111 |
| cifar10 | dir1 | label_flip | main |  | 1 | flirds_zgate_v2 | 0.4964 | -0.0027 | -0.0225 | 0.9844 |
| cifar10 | dir1 | label_flip | main |  | 1 | oracle_excl | 0.6214 | +0.0072 | +1.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 1 | oracle_excl | 0.6214 | +0.0467 | +1.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 1 | oracle_excl | 0.6214 | +0.1223 | +1.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 1 | random_excl | 0.5969 | -0.0172 | -2.3793 |  |
| cifar10 | dir1 | label_flip | main |  | 1 | random_excl | 0.5687 | -0.0059 | -0.1257 |  |
| cifar10 | dir1 | label_flip | main |  | 1 | random_excl | 0.5071 | +0.0080 | +0.0654 |  |
| cifar10 | dir1 | label_flip | main |  | 1 | vanilla | 0.6141 | +0.0000 | +0.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 1 | vanilla | 0.5746 | +0.0000 | +0.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 1 | vanilla | 0.4991 | +0.0000 | +0.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_gate_v1 | 0.6122 | -0.0034 | -0.5000 | 0.5548 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_gate_v1 | 0.6048 | +0.0235 | +0.5714 | 0.8539 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_gate_v1 | 0.5870 | +0.0681 | +0.6582 | 0.9976 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_gate_v2 | 0.5863 | -0.0294 | -4.3519 | 0.5705 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_gate_v2 | 0.5486 | -0.0326 | -0.7933 | 0.6636 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_gate_v2 | 0.5805 | +0.0616 | +0.5954 | 0.7539 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_gatew_v1 | 0.5733 | -0.0424 | -6.2778 | 0.5640 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_gatew_v1 | 0.5824 | +0.0011 | +0.0274 | 0.7656 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_gatew_v1 | 0.5669 | +0.0480 | +0.4638 | 0.9735 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_gatew_v2 | 0.5939 | -0.0217 | -3.2222 | 0.5151 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_gatew_v2 | 0.5819 | +0.0006 | +0.0152 | 0.6471 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_gatew_v2 | 0.5709 | +0.0520 | +0.5024 | 0.7146 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_mult | 0.6104 | -0.0052 | -0.7778 | 0.5299 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_mult | 0.6002 | +0.0190 | +0.4620 | 0.8109 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_mult | 0.5859 | +0.0670 | +0.6473 | 0.9707 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_zgate_v2 | 0.6099 | -0.0058 | -0.8519 | 0.5921 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_zgate_v2 | 0.5735 | -0.0078 | -0.1884 | 0.9342 |
| cifar10 | dir1 | label_flip | main |  | 2 | flirds_zgate_v2 | 0.5165 | -0.0024 | -0.0229 | 1.0000 |
| cifar10 | dir1 | label_flip | main |  | 2 | oracle_excl | 0.6224 | +0.0068 | +1.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 2 | oracle_excl | 0.6224 | +0.0411 | +1.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 2 | oracle_excl | 0.6224 | +0.1035 | +1.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 2 | random_excl | 0.5736 | -0.0420 | -6.2222 |  |
| cifar10 | dir1 | label_flip | main |  | 2 | random_excl | 0.5290 | -0.0523 | -1.2705 |  |
| cifar10 | dir1 | label_flip | main |  | 2 | random_excl | 0.4385 | -0.0804 | -0.7766 |  |
| cifar10 | dir1 | label_flip | main |  | 2 | vanilla | 0.6156 | +0.0000 | +0.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 2 | vanilla | 0.5813 | +0.0000 | +0.0000 |  |
| cifar10 | dir1 | label_flip | main |  | 2 | vanilla | 0.5189 | +0.0000 | +0.0000 |  |
| cifar10 | iid | clean | main |  | 0 | flirds_gate_v1 | 0.6438 | -0.0054 |  |  |
| cifar10 | iid | clean | main |  | 0 | flirds_gate_v2 | 0.6454 | -0.0037 |  |  |
| cifar10 | iid | clean | main |  | 0 | flirds_gatew_v1 | 0.6455 | -0.0036 |  |  |
| cifar10 | iid | clean | main |  | 0 | flirds_gatew_v2 | 0.6369 | -0.0122 |  |  |
| cifar10 | iid | clean | main |  | 0 | flirds_mult | 0.6469 | -0.0022 |  |  |
| cifar10 | iid | clean | main |  | 0 | flirds_zgate_v2 | 0.6455 | -0.0036 |  |  |
| cifar10 | iid | clean | main |  | 0 | vanilla | 0.6491 | +0.0000 |  |  |
| cifar10 | iid | clean | main |  | 1 | flirds_gate_v1 | 0.6395 | -0.0088 |  |  |
| cifar10 | iid | clean | main |  | 1 | flirds_gate_v2 | 0.6488 | +0.0005 |  |  |
| cifar10 | iid | clean | main |  | 1 | flirds_gatew_v1 | 0.6149 | -0.0334 |  |  |
| cifar10 | iid | clean | main |  | 1 | flirds_gatew_v2 | 0.6466 | -0.0016 |  |  |
| cifar10 | iid | clean | main |  | 1 | flirds_mult | 0.6481 | -0.0001 |  |  |
| cifar10 | iid | clean | main |  | 1 | flirds_zgate_v2 | 0.6501 | +0.0019 |  |  |
| cifar10 | iid | clean | main |  | 1 | vanilla | 0.6482 | +0.0000 |  |  |
| cifar10 | iid | clean | main |  | 2 | flirds_gate_v1 | 0.6411 | -0.0078 |  |  |
| cifar10 | iid | clean | main |  | 2 | flirds_gate_v2 | 0.6342 | -0.0146 |  |  |
| cifar10 | iid | clean | main |  | 2 | flirds_gatew_v1 | 0.6414 | -0.0075 |  |  |
| cifar10 | iid | clean | main |  | 2 | flirds_gatew_v2 | 0.6402 | -0.0086 |  |  |
| cifar10 | iid | clean | main |  | 2 | flirds_mult | 0.6450 | -0.0039 |  |  |
| cifar10 | iid | clean | main |  | 2 | flirds_zgate_v2 | 0.6486 | -0.0002 |  |  |
| cifar10 | iid | clean | main |  | 2 | vanilla | 0.6489 | +0.0000 |  |  |
| cifar10 | iid | free_rider | main |  | 0 | flirds_gate_v1 | 0.6048 | -0.0051 | -0.2071 | 1.0000 |
| cifar10 | iid | free_rider | main |  | 0 | flirds_gate_v2 | 0.6256 | +0.0158 | +0.6364 | 0.9333 |
| cifar10 | iid | free_rider | main |  | 0 | flirds_gatew_v1 | 0.6178 | +0.0079 | +0.3182 | 1.0000 |
| cifar10 | iid | free_rider | main |  | 0 | flirds_gatew_v2 | 0.6272 | +0.0174 | +0.7020 | 0.9167 |
| cifar10 | iid | free_rider | main |  | 0 | flirds_mult | 0.6279 | +0.0180 | +0.7273 | 0.4646 |
| cifar10 | iid | free_rider | main |  | 0 | flirds_zgate_v2 | 0.6055 | -0.0044 | -0.1768 | 0.9000 |
| cifar10 | iid | free_rider | main |  | 0 | oracle_excl | 0.6346 | +0.0248 | +1.0000 |  |
| cifar10 | iid | free_rider | main |  | 0 | random_excl | 0.6035 | -0.0064 | -0.2576 |  |
| cifar10 | iid | free_rider | main |  | 0 | vanilla | 0.6099 | +0.0000 | +0.0000 |  |
| cifar10 | iid | free_rider | main |  | 1 | flirds_gate_v1 | 0.6312 | +0.0281 | +0.8333 | 1.0000 |
| cifar10 | iid | free_rider | main |  | 1 | flirds_gate_v2 | 0.6355 | +0.0324 | +0.9593 | 0.8500 |
| cifar10 | iid | free_rider | main |  | 1 | flirds_gatew_v1 | 0.6304 | +0.0272 | +0.8074 | 1.0000 |
| cifar10 | iid | free_rider | main |  | 1 | flirds_gatew_v2 | 0.6352 | +0.0321 | +0.9519 | 0.9167 |
| cifar10 | iid | free_rider | main |  | 1 | flirds_mult | 0.6202 | +0.0171 | +0.5074 | 0.3871 |
| cifar10 | iid | free_rider | main |  | 1 | flirds_zgate_v2 | 0.6058 | +0.0026 | +0.0778 | 0.9167 |
| cifar10 | iid | free_rider | main |  | 1 | oracle_excl | 0.6369 | +0.0337 | +1.0000 |  |
| cifar10 | iid | free_rider | main |  | 1 | random_excl | 0.6036 | +0.0005 | +0.0148 |  |
| cifar10 | iid | free_rider | main |  | 1 | vanilla | 0.6031 | +0.0000 | +0.0000 |  |
| cifar10 | iid | free_rider | main |  | 2 | flirds_gate_v1 | 0.6268 | +0.0149 | +0.6330 | 1.0000 |
| cifar10 | iid | free_rider | main |  | 2 | flirds_gate_v2 | 0.6314 | +0.0195 | +0.8298 | 0.8667 |
| cifar10 | iid | free_rider | main |  | 2 | flirds_gatew_v1 | 0.6265 | +0.0146 | +0.6223 | 1.0000 |
| cifar10 | iid | free_rider | main |  | 2 | flirds_gatew_v2 | 0.6325 | +0.0206 | +0.8777 | 0.8667 |
| cifar10 | iid | free_rider | main |  | 2 | flirds_mult | 0.6295 | +0.0176 | +0.7500 | 0.3579 |
| cifar10 | iid | free_rider | main |  | 2 | flirds_zgate_v2 | 0.6138 | +0.0019 | +0.0798 | 0.8333 |
| cifar10 | iid | free_rider | main |  | 2 | oracle_excl | 0.6354 | +0.0235 | +1.0000 |  |
| cifar10 | iid | free_rider | main |  | 2 | random_excl | 0.5886 | -0.0232 | -0.9894 |  |
| cifar10 | iid | free_rider | main |  | 2 | vanilla | 0.6119 | +0.0000 | +0.0000 |  |
| cifar10 | iid | grad_noise | main |  | 0 | flirds_gate_v1 | 0.5836 | +0.3249 | +0.8643 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 0 | flirds_gate_v2 | 0.6285 | +0.3697 | +0.9837 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 0 | flirds_gatew_v1 | 0.5836 | +0.3249 | +0.8643 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 0 | flirds_gatew_v2 | 0.6136 | +0.3549 | +0.9441 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 0 | flirds_mult | 0.5519 | +0.2931 | +0.7798 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 0 | flirds_zgate_v2 | 0.2970 | +0.0383 | +0.1018 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 0 | oracle_excl | 0.6346 | +0.3759 | +1.0000 |  |
| cifar10 | iid | grad_noise | main |  | 0 | random_excl | 0.2655 | +0.0068 | +0.0180 |  |
| cifar10 | iid | grad_noise | main |  | 0 | vanilla | 0.2587 | +0.0000 | +0.0000 |  |
| cifar10 | iid | grad_noise | main |  | 1 | flirds_gate_v1 | 0.5747 | +0.3230 | +0.8387 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 1 | flirds_gate_v2 | 0.5901 | +0.3384 | +0.8786 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 1 | flirds_gatew_v1 | 0.5775 | +0.3258 | +0.8458 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 1 | flirds_gatew_v2 | 0.6152 | +0.3635 | +0.9438 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 1 | flirds_mult | 0.5198 | +0.2680 | +0.6959 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 1 | flirds_zgate_v2 | 0.3362 | +0.0845 | +0.2194 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 1 | oracle_excl | 0.6369 | +0.3851 | +1.0000 |  |
| cifar10 | iid | grad_noise | main |  | 1 | random_excl | 0.2744 | +0.0226 | +0.0587 |  |
| cifar10 | iid | grad_noise | main |  | 1 | vanilla | 0.2517 | +0.0000 | +0.0000 |  |
| cifar10 | iid | grad_noise | main |  | 2 | flirds_gate_v1 | 0.5804 | +0.3217 | +0.8540 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 2 | flirds_gate_v2 | 0.6244 | +0.3658 | +0.9708 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 2 | flirds_gatew_v1 | 0.5704 | +0.3117 | +0.8275 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 2 | flirds_gatew_v2 | 0.6265 | +0.3679 | +0.9764 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 2 | flirds_mult | 0.5271 | +0.2685 | +0.7127 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 2 | flirds_zgate_v2 | 0.3241 | +0.0655 | +0.1739 | 1.0000 |
| cifar10 | iid | grad_noise | main |  | 2 | oracle_excl | 0.6354 | +0.3768 | +1.0000 |  |
| cifar10 | iid | grad_noise | main |  | 2 | random_excl | 0.2535 | -0.0051 | -0.0136 |  |
| cifar10 | iid | grad_noise | main |  | 2 | vanilla | 0.2586 | +0.0000 | +0.0000 |  |
| cifar10 | iid | label_flip | main |  | 0 | flirds_gate_v1 | 0.6076 | -0.0230 | +16.7273 | 0.6931 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_gate_v1 | 0.6248 | +0.0299 | +0.8691 | 0.9639 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_gate_v1 | 0.5713 | +0.0333 | +0.3644 | 1.0000 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_gate_v2 | 0.6245 | -0.0061 | +4.4545 | 0.5750 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_gate_v2 | 0.6105 | +0.0156 | +0.4545 | 0.7650 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_gate_v2 | 0.6032 | +0.0652 | +0.7151 | 0.8394 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_gatew_v1 | 0.6192 | -0.0114 | +8.2727 | 0.6301 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_gatew_v1 | 0.6196 | +0.0247 | +0.7200 | 0.9840 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_gatew_v1 | 0.6054 | +0.0674 | +0.7384 | 1.0000 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_gatew_v2 | 0.6181 | -0.0125 | +9.0909 | 0.3762 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_gatew_v2 | 0.6079 | +0.0130 | +0.3782 | 0.6053 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_gatew_v2 | 0.6091 | +0.0711 | +0.7795 | 0.8541 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_mult | 0.6256 | -0.0050 | +3.6364 | 0.3859 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_mult | 0.6191 | +0.0242 | +0.7055 | 0.7516 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_mult | 0.6142 | +0.0762 | +0.8356 | 0.9756 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_zgate_v2 | 0.6216 | -0.0090 | +6.5455 | 0.6478 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_zgate_v2 | 0.6049 | +0.0100 | +0.2909 | 0.9954 |
| cifar10 | iid | label_flip | main |  | 0 | flirds_zgate_v2 | 0.5911 | +0.0531 | +0.5822 | 1.0000 |
| cifar10 | iid | label_flip | main |  | 0 | oracle_excl | 0.6292 | -0.0014 | +1.0000 |  |
| cifar10 | iid | label_flip | main |  | 0 | oracle_excl | 0.6292 | +0.0344 | +1.0000 |  |
| cifar10 | iid | label_flip | main |  | 0 | oracle_excl | 0.6292 | +0.0912 | +1.0000 |  |
| cifar10 | iid | label_flip | main |  | 0 | random_excl | 0.6126 | -0.0180 | +13.0909 |  |
| cifar10 | iid | label_flip | main |  | 0 | random_excl | 0.5962 | +0.0014 | +0.0400 |  |
| cifar10 | iid | label_flip | main |  | 0 | random_excl | 0.5610 | +0.0230 | +0.2521 |  |
| cifar10 | iid | label_flip | main |  | 0 | vanilla | 0.6306 | +0.0000 | -0.0000 |  |
| cifar10 | iid | label_flip | main |  | 0 | vanilla | 0.5949 | +0.0000 | +0.0000 |  |
| cifar10 | iid | label_flip | main |  | 0 | vanilla | 0.5380 | +0.0000 | +0.0000 |  |
| cifar10 | iid | label_flip | main |  | 1 | flirds_gate_v1 | 0.6284 | -0.0020 | -0.4000 | 0.6274 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_gate_v1 | 0.6071 | +0.0155 | +0.3543 | 0.9984 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_gate_v1 | 0.5773 | +0.0739 | +0.5597 | 1.0000 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_gate_v2 | 0.6225 | -0.0079 | -1.5750 | 0.6466 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_gate_v2 | 0.6099 | +0.0182 | +0.4171 | 0.8205 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_gate_v2 | 0.5986 | +0.0952 | +0.7216 | 0.8986 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_gatew_v1 | 0.6270 | -0.0034 | -0.6750 | 0.6647 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_gatew_v1 | 0.6034 | +0.0118 | +0.2686 | 0.9972 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_gatew_v1 | 0.5950 | +0.0916 | +0.6941 | 1.0000 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_gatew_v2 | 0.6120 | -0.0184 | -3.6750 | 0.5008 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_gatew_v2 | 0.6079 | +0.0163 | +0.3714 | 0.7107 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_gatew_v2 | 0.5965 | +0.0931 | +0.7055 | 0.8774 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_mult | 0.6302 | -0.0001 | -0.0250 | 0.4002 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_mult | 0.6162 | +0.0246 | +0.5629 | 0.8393 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_mult | 0.6020 | +0.0986 | +0.7472 | 0.9872 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_zgate_v2 | 0.6366 | +0.0062 | +1.2500 | 0.7304 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_zgate_v2 | 0.5925 | +0.0009 | +0.0200 | 1.0000 |
| cifar10 | iid | label_flip | main |  | 1 | flirds_zgate_v2 | 0.5209 | +0.0175 | +0.1326 | 1.0000 |
| cifar10 | iid | label_flip | main |  | 1 | oracle_excl | 0.6354 | +0.0050 | +1.0000 |  |
| cifar10 | iid | label_flip | main |  | 1 | oracle_excl | 0.6354 | +0.0438 | +1.0000 |  |
| cifar10 | iid | label_flip | main |  | 1 | oracle_excl | 0.6354 | +0.1320 | +1.0000 |  |
| cifar10 | iid | label_flip | main |  | 1 | random_excl | 0.6032 | -0.0271 | -5.4250 |  |
| cifar10 | iid | label_flip | main |  | 1 | random_excl | 0.5699 | -0.0217 | -0.4971 |  |
| cifar10 | iid | label_flip | main |  | 1 | random_excl | 0.4898 | -0.0136 | -0.1032 |  |
| cifar10 | iid | label_flip | main |  | 1 | vanilla | 0.6304 | +0.0000 | +0.0000 |  |
| cifar10 | iid | label_flip | main |  | 1 | vanilla | 0.5916 | +0.0000 | +0.0000 |  |
| cifar10 | iid | label_flip | main |  | 1 | vanilla | 0.5034 | +0.0000 | +0.0000 |  |
| cifar10 | iid | label_flip | main |  | 2 | flirds_gate_v1 | 0.6126 | -0.0094 | -1.5000 | 0.7134 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_gate_v1 | 0.6174 | +0.0271 | +0.7138 | 0.9940 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_gate_v1 | 0.6159 | +0.1059 | +0.8953 | 1.0000 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_gate_v2 | 0.6214 | -0.0006 | -0.1000 | 0.6499 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_gate_v2 | 0.5870 | -0.0033 | -0.0855 | 0.8475 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_gate_v2 | 0.5883 | +0.0783 | +0.6617 | 0.8900 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_gatew_v1 | 0.6141 | -0.0079 | -1.2600 | 0.7102 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_gatew_v1 | 0.6105 | +0.0202 | +0.5329 | 0.9972 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_gatew_v1 | 0.6185 | +0.1085 | +0.9175 | 1.0000 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_gatew_v2 | 0.6109 | -0.0111 | -1.7800 | 0.4898 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_gatew_v2 | 0.5817 | -0.0085 | -0.2237 | 0.6877 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_gatew_v2 | 0.6039 | +0.0939 | +0.7939 | 0.8703 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_mult | 0.6170 | -0.0050 | -0.8000 | 0.3693 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_mult | 0.6130 | +0.0227 | +0.5987 | 0.9077 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_mult | 0.6066 | +0.0966 | +0.8171 | 0.9916 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_zgate_v2 | 0.6175 | -0.0045 | -0.7200 | 0.6989 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_zgate_v2 | 0.5959 | +0.0056 | +0.1480 | 1.0000 |
| cifar10 | iid | label_flip | main |  | 2 | flirds_zgate_v2 | 0.5126 | +0.0026 | +0.0222 | 1.0000 |
| cifar10 | iid | label_flip | main |  | 2 | oracle_excl | 0.6282 | +0.0062 | +1.0000 |  |
| cifar10 | iid | label_flip | main |  | 2 | oracle_excl | 0.6282 | +0.0380 | +1.0000 |  |
| cifar10 | iid | label_flip | main |  | 2 | oracle_excl | 0.6282 | +0.1182 | +1.0000 |  |
| cifar10 | iid | label_flip | main |  | 2 | random_excl | 0.5894 | -0.0326 | -5.2200 |  |
| cifar10 | iid | label_flip | main |  | 2 | random_excl | 0.5507 | -0.0395 | -1.0395 |  |
| cifar10 | iid | label_flip | main |  | 2 | random_excl | 0.4561 | -0.0539 | -0.4556 |  |
| cifar10 | iid | label_flip | main |  | 2 | vanilla | 0.6220 | +0.0000 | +0.0000 |  |
| cifar10 | iid | label_flip | main |  | 2 | vanilla | 0.5903 | +0.0000 | +0.0000 |  |
| cifar10 | iid | label_flip | main |  | 2 | vanilla | 0.5100 | +0.0000 | +0.0000 |  |

## V2w promotion gate (spec §5-2): **DO NOT PROMOTE (report CNN-only -- an honest finding)**

  cifar10/dir1/free_rider(str=main): V2w-V2 mean dAcc=-0.0092 FAIL
  cifar10/dir1/grad_noise(str=main): V2w-V2 mean dAcc=+0.0206 OK
  cifar10/dir1/label_flip(str=main): V2w-V2 mean dAcc=+0.0046 OK
  cifar10/iid/free_rider(str=main): V2w-V2 mean dAcc=+0.0008 OK
  cifar10/iid/grad_noise(str=main): V2w-V2 mean dAcc=+0.0041 OK
  cifar10/iid/label_flip(str=main): V2w-V2 mean dAcc=-0.0020 FAIL
  clean cifar10_dir1_clean_g_seed0: V2w dAcc=-0.0200 FAIL(parity broken)
  clean cifar10_dir1_clean_g_seed1: V2w dAcc=-0.0100 FAIL(parity broken)
  clean cifar10_dir1_clean_g_seed2: V2w dAcc=-0.0302 FAIL(parity broken)
  clean cifar10_iid_clean_g_seed0: V2w dAcc=-0.0122 FAIL(parity broken)
  clean cifar10_iid_clean_g_seed1: V2w dAcc=-0.0016 OK
  clean cifar10_iid_clean_g_seed2: V2w dAcc=-0.0086 FAIL(parity broken)

## CNN V3 (track_c1 C1_V3 cells)

- cifar10_feature_noise_v3_seed0 sign_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1702529191970825 acc=0.6105 (full: 1.1076332330703735/0.63175)
- cifar10_feature_noise_v3_seed0 z_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 9] val_loss=1.1152021884918213 acc=0.624125 (full: 1.1076332330703735/0.63175)
- cifar10_feature_noise_v3_seed0 sign_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1702529191970825 acc=0.6105 (full: 1.1076332330703735/0.63175)
- cifar10_feature_noise_v3_seed0 z_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 9] val_loss=1.1152021884918213 acc=0.624125 (full: 1.1076332330703735/0.63175)
- cifar10_feature_noise_v3_seed0 random: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1702529191970825 acc=0.6105 (full: 1.1076332330703735/0.63175)
- cifar10_feature_noise_v3_seed1 sign_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1717321872711182 acc=0.609 (full: 1.1263816356658936/0.621875)
- cifar10_feature_noise_v3_seed1 z_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1717321872711182 acc=0.609 (full: 1.1263816356658936/0.621875)
- cifar10_feature_noise_v3_seed1 sign_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1717321872711182 acc=0.609 (full: 1.1263816356658936/0.621875)
- cifar10_feature_noise_v3_seed1 z_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1717321872711182 acc=0.609 (full: 1.1263816356658936/0.621875)
- cifar10_feature_noise_v3_seed1 random: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1717321872711182 acc=0.609 (full: 1.1263816356658936/0.621875)
- cifar10_feature_noise_v3_seed2 sign_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.127548336982727 acc=0.623625 (full: 1.1358436346054077/0.61825)
- cifar10_feature_noise_v3_seed2 z_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 8, 9] val_loss=1.1307119131088257 acc=0.622625 (full: 1.1358436346054077/0.61825)
- cifar10_feature_noise_v3_seed2 sign_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.127548336982727 acc=0.623625 (full: 1.1358436346054077/0.61825)
- cifar10_feature_noise_v3_seed2 z_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 8, 9] val_loss=1.1307119131088257 acc=0.622625 (full: 1.1358436346054077/0.61825)
- cifar10_feature_noise_v3_seed2 random: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.127548336982727 acc=0.623625 (full: 1.1358436346054077/0.61825)
- cifar10_label_flip_v3_seed0 sign_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1528667211532593 acc=0.61725 (full: 1.1218472719192505/0.627875)
- cifar10_label_flip_v3_seed0 z_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 9] val_loss=1.1229904890060425 acc=0.628125 (full: 1.1218472719192505/0.627875)
- cifar10_label_flip_v3_seed0 sign_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1528667211532593 acc=0.61725 (full: 1.1218472719192505/0.627875)
- cifar10_label_flip_v3_seed0 z_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 9] val_loss=1.1229904890060425 acc=0.628125 (full: 1.1218472719192505/0.627875)
- cifar10_label_flip_v3_seed0 random: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1528667211532593 acc=0.61725 (full: 1.1218472719192505/0.627875)
- cifar10_label_flip_v3_seed1 sign_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.144560694694519 acc=0.622875 (full: 1.1404670476913452/0.624875)
- cifar10_label_flip_v3_seed1 z_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8] val_loss=1.1624077558517456 acc=0.6255 (full: 1.1404670476913452/0.624875)
- cifar10_label_flip_v3_seed1 sign_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.144560694694519 acc=0.622875 (full: 1.1404670476913452/0.624875)
- cifar10_label_flip_v3_seed1 z_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8] val_loss=1.1624077558517456 acc=0.6255 (full: 1.1404670476913452/0.624875)
- cifar10_label_flip_v3_seed1 random: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.144560694694519 acc=0.622875 (full: 1.1404670476913452/0.624875)
- cifar10_label_flip_v3_seed2 sign_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1416267156600952 acc=0.62775 (full: 1.1367138624191284/0.6315)
- cifar10_label_flip_v3_seed2 z_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1416267156600952 acc=0.62775 (full: 1.1367138624191284/0.6315)
- cifar10_label_flip_v3_seed2 sign_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1416267156600952 acc=0.62775 (full: 1.1367138624191284/0.6315)
- cifar10_label_flip_v3_seed2 z_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1416267156600952 acc=0.62775 (full: 1.1367138624191284/0.6315)
- cifar10_label_flip_v3_seed2 random: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=1.1416267156600952 acc=0.62775 (full: 1.1367138624191284/0.6315)
- mnist_feature_noise_v3_seed0 sign_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.05761987343430519 acc=0.980625 (full: 0.05717025697231293/0.981)
- mnist_feature_noise_v3_seed0 z_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.05761987343430519 acc=0.980625 (full: 0.05717025697231293/0.981)
- mnist_feature_noise_v3_seed0 sign_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.05761987343430519 acc=0.980625 (full: 0.05717025697231293/0.981)
- mnist_feature_noise_v3_seed0 z_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 9] val_loss=0.05794680863618851 acc=0.981 (full: 0.05717025697231293/0.981)
- mnist_feature_noise_v3_seed0 random: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.05761987343430519 acc=0.980625 (full: 0.05717025697231293/0.981)
- mnist_feature_noise_v3_seed1 sign_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.06491689383983612 acc=0.981125 (full: 0.0640549287199974/0.98)
- mnist_feature_noise_v3_seed1 z_Flirds: kept=[1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.06376102566719055 acc=0.98025 (full: 0.0640549287199974/0.98)
- mnist_feature_noise_v3_seed1 sign_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.06491689383983612 acc=0.981125 (full: 0.0640549287199974/0.98)
- mnist_feature_noise_v3_seed1 z_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 9] val_loss=0.06437192112207413 acc=0.98025 (full: 0.0640549287199974/0.98)
- mnist_feature_noise_v3_seed1 random: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.06491689383983612 acc=0.981125 (full: 0.0640549287199974/0.98)
- mnist_feature_noise_v3_seed2 sign_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.06979236006736755 acc=0.977375 (full: 0.06876019388437271/0.978625)
- mnist_feature_noise_v3_seed2 z_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 8] val_loss=0.06675303727388382 acc=0.97825 (full: 0.06876019388437271/0.978625)
- mnist_feature_noise_v3_seed2 sign_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.06979236006736755 acc=0.977375 (full: 0.06876019388437271/0.978625)
- mnist_feature_noise_v3_seed2 z_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8] val_loss=0.06892785429954529 acc=0.97825 (full: 0.06876019388437271/0.978625)
- mnist_feature_noise_v3_seed2 random: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.06979236006736755 acc=0.977375 (full: 0.06876019388437271/0.978625)
- mnist_label_flip_v3_seed0 sign_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.1325567066669464 acc=0.9805 (full: 0.13372592628002167/0.97925)
- mnist_label_flip_v3_seed0 z_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.1325567066669464 acc=0.9805 (full: 0.13372592628002167/0.97925)
- mnist_label_flip_v3_seed0 sign_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.1325567066669464 acc=0.9805 (full: 0.13372592628002167/0.97925)
- mnist_label_flip_v3_seed0 z_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.1325567066669464 acc=0.9805 (full: 0.13372592628002167/0.97925)
- mnist_label_flip_v3_seed0 random: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.1325567066669464 acc=0.9805 (full: 0.13372592628002167/0.97925)
- mnist_label_flip_v3_seed1 sign_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.13461598753929138 acc=0.977875 (full: 0.13766686618328094/0.979375)
- mnist_label_flip_v3_seed1 z_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.13461598753929138 acc=0.977875 (full: 0.13766686618328094/0.979375)
- mnist_label_flip_v3_seed1 sign_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.13461598753929138 acc=0.977875 (full: 0.13766686618328094/0.979375)
- mnist_label_flip_v3_seed1 z_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.13461598753929138 acc=0.977875 (full: 0.13766686618328094/0.979375)
- mnist_label_flip_v3_seed1 random: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.13461598753929138 acc=0.977875 (full: 0.13766686618328094/0.979375)
- mnist_label_flip_v3_seed2 sign_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.1467311829328537 acc=0.977625 (full: 0.14914661645889282/0.978125)
- mnist_label_flip_v3_seed2 z_Flirds: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.1467311829328537 acc=0.977625 (full: 0.14914661645889282/0.978125)
- mnist_label_flip_v3_seed2 sign_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.1467311829328537 acc=0.977625 (full: 0.14914661645889282/0.978125)
- mnist_label_flip_v3_seed2 z_(b)oracle: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.1467311829328537 acc=0.977625 (full: 0.14914661645889282/0.978125)
- mnist_label_flip_v3_seed2 random: kept=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] val_loss=0.1467311829328537 acc=0.977625 (full: 0.14914661645889282/0.978125)