# Track G analysis (auto-generated from rundirs -- rerun make_analysis.py)

## [1] performance delta + recovery  (delta = vanilla_loss - arm_loss, +=better; recovery = delta / (vanilla - oracle_excl))

| regime | threat | nr | seed | arm | final_val_loss | delta | recovery | mmlu | rouge_l | prediction | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iid5 | clean | 1 | 0 | flirds_gate_v1 | 1.3789 | +0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| iid5 | clean | 1 | 0 | flirds_gate_v2 | 1.3789 | +0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| iid5 | clean | 1 | 0 | flirds_w | 1.3783 | +0.0006 |  |  |  |  |  |
| iid5 | clean | 1 | 0 | flirds_zgate_v2 | 1.3788 | +0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | MISS |
| iid5 | clean | 1 | 0 | lossheur_gate_v2 | 1.3789 | +0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| iid5 | clean | 1 | 0 | v3_random | 1.3789 | +0.0000 |  |  |  |  |  |
| iid5 | clean | 1 | 0 | v3_sign | 1.3789 | +0.0000 |  |  |  | parity |  |
| iid5 | clean | 1 | 0 | v3_z | 1.3789 | +0.0000 |  |  |  | parity |  |
| iid5 | clean | 1 | 0 | vanilla | 1.3789 | +0.0000 |  |  |  |  |  |
| iid5 | clean | 1 | 1 | flirds_gate_v1 | 1.2808 | +0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| iid5 | clean | 1 | 1 | flirds_gate_v2 | 1.2808 | +0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| iid5 | clean | 1 | 1 | flirds_w | 1.2802 | +0.0006 |  |  |  |  |  |
| iid5 | clean | 1 | 1 | flirds_zgate_v2 | 1.2804 | +0.0004 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | MISS |
| iid5 | clean | 1 | 1 | lossheur_gate_v2 | 1.2808 | +0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| iid5 | clean | 1 | 1 | v3_random | 1.2808 | +0.0000 |  |  |  |  |  |
| iid5 | clean | 1 | 1 | v3_sign | 1.2808 | +0.0000 |  |  |  | parity |  |
| iid5 | clean | 1 | 1 | v3_z | 1.2804 | +0.0004 |  |  |  | parity |  |
| iid5 | clean | 1 | 1 | vanilla | 1.2808 | +0.0000 |  |  |  |  |  |
| iid5 | clean | 1 | 2 | flirds_gate_v1 | 1.2844 | +0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| iid5 | clean | 1 | 2 | flirds_gate_v2 | 1.2844 | +0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| iid5 | clean | 1 | 2 | flirds_w | 1.2839 | +0.0006 |  |  |  |  |  |
| iid5 | clean | 1 | 2 | flirds_zgate_v2 | 1.2839 | +0.0006 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | MISS |
| iid5 | clean | 1 | 2 | lossheur_gate_v2 | 1.2844 | +0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| iid5 | clean | 1 | 2 | v3_random | 1.2844 | +0.0000 |  |  |  |  |  |
| iid5 | clean | 1 | 2 | v3_sign | 1.2844 | +0.0000 |  |  |  | parity |  |
| iid5 | clean | 1 | 2 | v3_z | 1.2840 | +0.0004 |  |  |  | parity |  |
| iid5 | clean | 1 | 2 | vanilla | 1.2844 | +0.0000 |  |  |  |  |  |
| iid5 | frzero | 1 | 0 | flirds_gate_v1 | 1.3773 | +0.0069 | +0.9004 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 0 | flirds_gate_v2 | 1.3765 | +0.0077 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 0 | flirds_w | 1.3772 | +0.0069 | +0.9019 |  |  |  |  |
| iid5 | frzero | 1 | 0 | flirds_zgate_v2 | 1.3765 | +0.0077 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 0 | lossheur_gate_v2 | 1.3765 | +0.0077 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 0 | oracle_excl | 1.3765 | +0.0077 | +1.0000 |  |  |  |  |
| iid5 | frzero | 1 | 0 | random_excl | 1.3765 | +0.0077 | +1.0000 |  |  |  |  |
| iid5 | frzero | 1 | 0 | v3_random | 1.3860 | -0.0019 | -0.2412 |  |  |  |  |
| iid5 | frzero | 1 | 0 | v3_sign | 1.3765 | +0.0077 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 0 | v3_z | 1.3765 | +0.0077 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 0 | vanilla | 1.3842 | +0.0000 | +0.0000 |  |  |  |  |
| iid5 | frzero | 1 | 1 | flirds_gate_v1 | 1.2790 | +0.0069 | +0.9005 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 1 | flirds_gate_v2 | 1.2782 | +0.0077 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 1 | flirds_w | 1.2790 | +0.0069 | +0.9020 |  |  |  |  |
| iid5 | frzero | 1 | 1 | flirds_zgate_v2 | 1.2781 | +0.0078 | +1.0159 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 1 | lossheur_gate_v2 | 1.2782 | +0.0077 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 1 | oracle_excl | 1.2782 | +0.0077 | +1.0000 |  |  |  |  |
| iid5 | frzero | 1 | 1 | random_excl | 1.2880 | -0.0021 | -0.2697 |  |  |  |  |
| iid5 | frzero | 1 | 1 | v3_random | 1.2881 | -0.0021 | -0.2771 |  |  |  |  |
| iid5 | frzero | 1 | 1 | v3_sign | 1.2782 | +0.0077 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 1 | v3_z | 1.2782 | +0.0077 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 1 | vanilla | 1.2859 | +0.0000 | +0.0000 |  |  |  |  |
| iid5 | frzero | 1 | 2 | flirds_gate_v1 | 1.2838 | +0.0057 | +0.8992 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 2 | flirds_gate_v2 | 1.2832 | +0.0063 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 2 | flirds_w | 1.2838 | +0.0057 | +0.9048 |  |  |  |  |
| iid5 | frzero | 1 | 2 | flirds_zgate_v2 | 1.2830 | +0.0066 | +1.0372 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 2 | lossheur_gate_v2 | 1.2832 | +0.0063 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 2 | oracle_excl | 1.2832 | +0.0063 | +1.0000 |  |  |  |  |
| iid5 | frzero | 1 | 2 | random_excl | 1.2907 | -0.0011 | -0.1747 |  |  |  |  |
| iid5 | frzero | 1 | 2 | v3_random | 1.2832 | +0.0063 | +1.0000 |  |  |  |  |
| iid5 | frzero | 1 | 2 | v3_sign | 1.2832 | +0.0063 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 2 | v3_z | 1.2832 | +0.0063 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| iid5 | frzero | 1 | 2 | vanilla | 1.2895 | +0.0000 | +0.0000 |  |  |  |  |
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
| silo5 | clean | 1 | 2 | flirds_gate_v1 | 2.3179 | +0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 2 | flirds_gate_v2 | 2.3179 | -0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 2 | flirds_w | 2.3126 | +0.0053 |  |  |  |  |  |
| silo5 | clean | 1 | 2 | flirds_zgate_v2 | 2.3179 | -0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 2 | lossheur_gate_v2 | 2.3179 | -0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 2 | oracleb_gate_v2 | 2.3179 | -0.0000 |  |  |  | vanilla parity; 0 false-exclusions (cum all positive) | HIT |
| silo5 | clean | 1 | 2 | v3_random | 2.3179 | +0.0000 |  |  |  |  |  |
| silo5 | clean | 1 | 2 | v3_sign | 2.3179 | +0.0000 |  |  |  | parity |  |
| silo5 | clean | 1 | 2 | v3_z | 2.3179 | +0.0000 |  |  |  | parity |  |
| silo5 | clean | 1 | 2 | vanilla | 2.3179 | +0.0000 |  |  |  |  |  |
| silo5 | frrand | 1 | 0 | flirds_gate_v1 | 2.2961 | +0.0027 | +0.6955 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 0 | flirds_gate_v2 | 2.2961 | +0.0027 | +0.6953 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 0 | flirds_w | 2.2936 | +0.0052 | +1.3567 |  |  |  |  |
| silo5 | frrand | 1 | 0 | flirds_zgate_v2 | 2.2949 | +0.0038 | +1.0000 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 0 | lossheur_gate_v2 | 2.2961 | +0.0027 | +0.6955 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 0 | oracle_excl | 2.2949 | +0.0038 | +1.0000 |  |  |  |  |
| silo5 | frrand | 1 | 0 | oracleb_gate_v2 | 2.2961 | +0.0027 | +0.6953 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 0 | random_excl | 2.2949 | +0.0038 | +1.0000 |  |  |  |  |
| silo5 | frrand | 1 | 0 | v3_random | 2.2988 | -0.0000 | -0.0019 |  |  |  |  |
| silo5 | frrand | 1 | 0 | v3_sign | 2.2949 | +0.0038 | +0.9999 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 0 | v3_z | 2.2949 | +0.0038 | +0.9999 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 0 | vanilla | 2.2988 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | frrand | 1 | 1 | flirds_gate_v1 | 2.3864 | +0.0011 | +0.2971 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 1 | flirds_gate_v2 | 2.3864 | +0.0011 | +0.2972 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 1 | flirds_w | 2.3826 | +0.0048 | +1.3301 |  |  |  |  |
| silo5 | frrand | 1 | 1 | flirds_zgate_v2 | 2.3838 | +0.0036 | +1.0001 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 1 | lossheur_gate_v2 | 2.3860 | +0.0014 | +0.3966 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 1 | oracle_excl | 2.3838 | +0.0036 | +1.0000 |  |  |  |  |
| silo5 | frrand | 1 | 1 | oracleb_gate_v2 | 2.3864 | +0.0011 | +0.2968 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 1 | random_excl | 2.3874 | +0.0001 | +0.0158 |  |  |  |  |
| silo5 | frrand | 1 | 1 | v3_random | 2.3875 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | frrand | 1 | 1 | v3_sign | 2.3875 | +0.0000 | +0.0000 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 1 | v3_z | 2.3838 | +0.0036 | +1.0000 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 1 | vanilla | 2.3875 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | frrand | 1 | 2 | flirds_gate_v1 | 2.3217 | +0.0013 | +0.3949 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 2 | flirds_gate_v2 | 2.3217 | +0.0013 | +0.3950 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 2 | flirds_w | 2.3179 | +0.0050 | +1.5678 |  |  |  |  |
| silo5 | frrand | 1 | 2 | flirds_zgate_v2 | 2.3229 | -0.0000 | -0.0001 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 2 | lossheur_gate_v2 | 2.3201 | +0.0028 | +0.8983 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 2 | oracle_excl | 2.3197 | +0.0032 | +1.0000 |  |  |  |  |
| silo5 | frrand | 1 | 2 | oracleb_gate_v2 | 2.3220 | +0.0009 | +0.2956 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 2 | random_excl | 2.3225 | +0.0004 | +0.1379 |  |  |  |  |
| silo5 | frrand | 1 | 2 | v3_random | 2.3229 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | frrand | 1 | 2 | v3_sign | 2.3229 | +0.0000 | +0.0000 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 2 | v3_z | 2.3229 | +0.0000 | +0.0000 |  |  | gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip -> exclusion seed-dependent |  |
| silo5 | frrand | 1 | 2 | vanilla | 2.3229 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | frzero | 1 | 0 | flirds_gate_v1 | 2.2956 | +0.0034 | +0.8976 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 0 | flirds_gate_v2 | 2.2953 | +0.0038 | +1.0001 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 0 | flirds_w | 2.2939 | +0.0051 | +1.3628 |  |  |  |  |
| silo5 | frzero | 1 | 0 | flirds_zgate_v2 | 2.2953 | +0.0038 | +1.0001 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 0 | lossheur_gate_v2 | 2.2953 | +0.0038 | +1.0001 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 0 | oracle_excl | 2.2953 | +0.0038 | +1.0000 |  |  |  |  |
| silo5 | frzero | 1 | 0 | oracleb_gate_v2 | 2.2953 | +0.0038 | +1.0001 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 0 | random_excl | 2.2953 | +0.0038 | +1.0000 |  |  |  |  |
| silo5 | frzero | 1 | 0 | v3_random | 2.2989 | +0.0001 | +0.0197 |  |  |  |  |
| silo5 | frzero | 1 | 0 | v3_sign | 2.2953 | +0.0038 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 0 | v3_z | 2.2953 | +0.0038 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 0 | vanilla | 2.2990 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | frzero | 1 | 1 | flirds_gate_v1 | 2.3833 | +0.0035 | +0.8981 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 1 | flirds_gate_v2 | 2.3830 | +0.0039 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 1 | flirds_w | 2.3817 | +0.0051 | +1.3209 |  |  |  |  |
| silo5 | frzero | 1 | 1 | flirds_zgate_v2 | 2.3830 | +0.0039 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 1 | lossheur_gate_v2 | 2.3830 | +0.0039 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 1 | oracle_excl | 2.3830 | +0.0039 | +1.0000 |  |  |  |  |
| silo5 | frzero | 1 | 1 | oracleb_gate_v2 | 2.3830 | +0.0039 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 1 | random_excl | 2.3868 | +0.0001 | +0.0164 |  |  |  |  |
| silo5 | frzero | 1 | 1 | v3_random | 2.3906 | -0.0037 | -0.9644 |  |  |  |  |
| silo5 | frzero | 1 | 1 | v3_sign | 2.3830 | +0.0039 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 1 | v3_z | 2.3830 | +0.0039 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 1 | vanilla | 2.3868 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | frzero | 1 | 2 | flirds_gate_v1 | 2.3197 | +0.0029 | +0.8977 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 2 | flirds_gate_v2 | 2.3194 | +0.0033 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 2 | flirds_w | 2.3175 | +0.0051 | +1.5782 |  |  |  |  |
| silo5 | frzero | 1 | 2 | flirds_zgate_v2 | 2.3226 | +0.0000 | +0.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | MISS |
| silo5 | frzero | 1 | 2 | lossheur_gate_v2 | 2.3194 | +0.0033 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 2 | oracle_excl | 2.3194 | +0.0033 | +1.0000 |  |  |  |  |
| silo5 | frzero | 1 | 2 | oracleb_gate_v2 | 2.3194 | +0.0033 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 2 | random_excl | 2.3222 | +0.0005 | +0.1525 |  |  |  |  |
| silo5 | frzero | 1 | 2 | v3_random | 2.3194 | +0.0033 | +1.0000 |  |  |  |  |
| silo5 | frzero | 1 | 2 | v3_sign | 2.3194 | +0.0033 | +1.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | HIT |
| silo5 | frzero | 1 | 2 | v3_z | 2.3226 | +0.0000 | +0.0000 |  |  | gain (~+0.007-class val-loss; exact-0 rule) | MISS |
| silo5 | frzero | 1 | 2 | vanilla | 2.3226 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | noisy | 0.75 | 0 | flirds_gate_v1 | 2.2960 | +0.0000 | +0.0000 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 0.75 | 0 | flirds_gate_v2 | 2.2960 | +0.0000 | +0.0002 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 0.75 | 0 | flirds_w | 2.2914 | +0.0047 | +3.1793 |  |  |  |  |
| silo5 | noisy | 0.75 | 0 | flirds_zgate_v2 | 2.2960 | +0.0000 | +0.0002 |  |  | recovery candidate (cohort-relative gate) |  |
| silo5 | noisy | 0.75 | 0 | lossheur_gate_v2 | 2.2960 | +0.0000 | +0.0002 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 0.75 | 0 | oracle_excl | 2.2946 | +0.0015 | +1.0000 |  |  |  |  |
| silo5 | noisy | 0.75 | 0 | oracleb_gate_v2 | 2.2960 | +0.0000 | +0.0002 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 0.75 | 0 | random_excl | 2.2960 | +0.0000 | +0.0227 |  |  |  |  |
| silo5 | noisy | 0.75 | 0 | v3_random | 2.2960 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | noisy | 0.75 | 0 | v3_sign | 2.2960 | +0.0000 | +0.0000 |  |  | parity |  |
| silo5 | noisy | 0.75 | 0 | v3_z | 2.2960 | +0.0000 | +0.0000 |  |  | parity |  |
| silo5 | noisy | 0.75 | 0 | vanilla | 2.2960 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | noisy | 1 | 0 | flirds_gate_v1 | 2.2962 | +0.0000 | +0.0000 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 1 | 0 | flirds_gate_v2 | 2.2962 | +0.0000 | +0.0000 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 1 | 0 | flirds_w | 2.2915 | +0.0047 | +2.8727 |  |  |  |  |
| silo5 | noisy | 1 | 0 | flirds_zgate_v2 | 2.2962 | +0.0000 | +0.0000 |  |  | recovery candidate (cohort-relative gate) |  |
| silo5 | noisy | 1 | 0 | lossheur_gate_v2 | 2.2962 | +0.0000 | +0.0000 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 1 | 0 | oracle_excl | 2.2946 | +0.0016 | +1.0000 |  |  |  |  |
| silo5 | noisy | 1 | 0 | oracleb_gate_v2 | 2.2962 | +0.0000 | +0.0000 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 1 | 0 | random_excl | 2.2963 | -0.0000 | -0.0237 |  |  |  |  |
| silo5 | noisy | 1 | 0 | v3_random | 2.2962 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | noisy | 1 | 0 | v3_sign | 2.2962 | +0.0000 | +0.0000 |  |  | parity |  |
| silo5 | noisy | 1 | 0 | v3_z | 2.2962 | +0.0000 | +0.0000 |  |  | parity |  |
| silo5 | noisy | 1 | 0 | vanilla | 2.2962 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | noisy | 1 | 1 | flirds_gate_v1 | 2.3867 | +0.0000 | +0.0000 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 1 | 1 | flirds_gate_v2 | 2.3867 | +0.0000 | +0.0000 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 1 | 1 | flirds_w | 2.3821 | +0.0046 | +3.0448 |  |  |  |  |
| silo5 | noisy | 1 | 1 | flirds_zgate_v2 | 2.3867 | +0.0000 | +0.0000 |  |  | recovery candidate (cohort-relative gate) |  |
| silo5 | noisy | 1 | 1 | lossheur_gate_v2 | 2.3867 | +0.0000 | +0.0000 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 1 | 1 | oracle_excl | 2.3852 | +0.0015 | +1.0000 |  |  |  |  |
| silo5 | noisy | 1 | 1 | oracleb_gate_v2 | 2.3867 | +0.0000 | +0.0000 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 1 | 1 | random_excl | 2.3852 | +0.0015 | +1.0000 |  |  |  |  |
| silo5 | noisy | 1 | 1 | v3_random | 2.3867 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | noisy | 1 | 1 | v3_sign | 2.3867 | +0.0000 | +0.0000 |  |  | parity |  |
| silo5 | noisy | 1 | 1 | v3_z | 2.3867 | +0.0000 | +0.0000 |  |  | parity |  |
| silo5 | noisy | 1 | 1 | vanilla | 2.3867 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | noisy | 1 | 2 | flirds_gate_v1 | 2.3190 | +0.0000 | +0.0000 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 1 | 2 | flirds_gate_v2 | 2.3190 | +0.0000 | +0.0001 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 1 | 2 | flirds_w | 2.3145 | +0.0045 | +2.2056 |  |  |  |  |
| silo5 | noisy | 1 | 2 | flirds_zgate_v2 | 2.3190 | +0.0000 | +0.0001 |  |  | recovery candidate (cohort-relative gate) |  |
| silo5 | noisy | 1 | 2 | lossheur_gate_v2 | 2.3190 | +0.0000 | +0.0001 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 1 | 2 | oracle_excl | 2.3170 | +0.0020 | +1.0000 |  |  |  |  |
| silo5 | noisy | 1 | 2 | oracleb_gate_v2 | 2.3190 | +0.0000 | +0.0001 |  |  | PARITY -- gate silent (no 0-crossing on nr<=1, audit P3) | HIT |
| silo5 | noisy | 1 | 2 | random_excl | 2.3179 | +0.0011 | +0.5579 |  |  |  |  |
| silo5 | noisy | 1 | 2 | v3_random | 2.3190 | +0.0000 | +0.0000 |  |  |  |  |
| silo5 | noisy | 1 | 2 | v3_sign | 2.3190 | +0.0000 | +0.0000 |  |  | parity |  |
| silo5 | noisy | 1 | 2 | v3_z | 2.3190 | +0.0000 | +0.0000 |  |  | parity |  |
| silo5 | noisy | 1 | 2 | vanilla | 2.3190 | +0.0000 | +0.0000 |  |  |  |  |
| std50k5 | mixed | 1 | 0 | flirds_gate_v2 | 1.2860 | +0.0027 | +1.2026 | 0.4699 | 0.2794 | approaches oracle_excl (FR share recovered) | HIT |
| std50k5 | mixed | 1 | 0 | oracle_excl | 1.2864 | +0.0023 | +1.0000 | 0.4718 | 0.2771 |  |  |
| std50k5 | mixed | 1 | 0 | random_excl | 1.2884 | +0.0003 | +0.1134 | 0.4724 | 0.2749 |  |  |
| std50k5 | mixed | 1 | 0 | shapleyfl_gate_v2 | 1.2863 | +0.0024 | +1.0452 | 0.4721 | 0.2775 | <= random_excl (fidelity-collapse stage) |  |
| std50k5 | mixed | 1 | 0 | vanilla | 1.2887 | +0.0000 | +0.0000 | 0.4702 | 0.2743 |  |  |
| std50k5 | mixed | 1 | 1 | flirds_gate_v2 | 1.2357 |  |  | 0.4739 | 0.2865 | approaches oracle_excl (FR share recovered) |  |
| std50k5 | mixed | 1 | 1 | shapleyfl_gate_v2 | 1.2350 |  |  | 0.4749 | 0.2902 | <= random_excl (fidelity-collapse stage) |  |
| std50k5 | mixed | 1 | 2 | shapleyfl_gate_v2 | 1.2709 |  |  | 0.4737 | 0.2819 | <= random_excl (fidelity-collapse stage) |  |

## [2] convergence (rounds-to-target = first round entering-loss <= the cell's vanilla final loss)

| regime | threat | nr | seed | arm | rounds_to_target |
|---|---|---|---|---|---|
| iid5 | clean | 1 | 0 | flirds_gate_v1 | 10.0000 |
| iid5 | clean | 1 | 0 | flirds_gate_v2 | 10.0000 |
| iid5 | clean | 1 | 0 | flirds_w | 10.0000 |
| iid5 | clean | 1 | 0 | flirds_zgate_v2 | 10.0000 |
| iid5 | clean | 1 | 0 | lossheur_gate_v2 | 10.0000 |
| iid5 | clean | 1 | 0 | v3_random |  |
| iid5 | clean | 1 | 0 | v3_sign |  |
| iid5 | clean | 1 | 0 | v3_z |  |
| iid5 | clean | 1 | 0 | vanilla | 10.0000 |
| iid5 | clean | 1 | 1 | flirds_gate_v1 | 10.0000 |
| iid5 | clean | 1 | 1 | flirds_gate_v2 | 10.0000 |
| iid5 | clean | 1 | 1 | flirds_w | 10.0000 |
| iid5 | clean | 1 | 1 | flirds_zgate_v2 | 10.0000 |
| iid5 | clean | 1 | 1 | lossheur_gate_v2 | 10.0000 |
| iid5 | clean | 1 | 1 | v3_random |  |
| iid5 | clean | 1 | 1 | v3_sign |  |
| iid5 | clean | 1 | 1 | v3_z |  |
| iid5 | clean | 1 | 1 | vanilla | 10.0000 |
| iid5 | clean | 1 | 2 | flirds_gate_v1 | 10.0000 |
| iid5 | clean | 1 | 2 | flirds_gate_v2 | 10.0000 |
| iid5 | clean | 1 | 2 | flirds_w | 10.0000 |
| iid5 | clean | 1 | 2 | flirds_zgate_v2 | 10.0000 |
| iid5 | clean | 1 | 2 | lossheur_gate_v2 | 10.0000 |
| iid5 | clean | 1 | 2 | v3_random |  |
| iid5 | clean | 1 | 2 | v3_sign |  |
| iid5 | clean | 1 | 2 | v3_z |  |
| iid5 | clean | 1 | 2 | vanilla | 10.0000 |
| iid5 | frzero | 1 | 0 | flirds_gate_v1 | 9.0000 |
| iid5 | frzero | 1 | 0 | flirds_gate_v2 | 9.0000 |
| iid5 | frzero | 1 | 0 | flirds_w | 9.0000 |
| iid5 | frzero | 1 | 0 | flirds_zgate_v2 | 9.0000 |
| iid5 | frzero | 1 | 0 | lossheur_gate_v2 | 9.0000 |
| iid5 | frzero | 1 | 0 | oracle_excl | 9.0000 |
| iid5 | frzero | 1 | 0 | random_excl | 9.0000 |
| iid5 | frzero | 1 | 0 | v3_random |  |
| iid5 | frzero | 1 | 0 | v3_sign |  |
| iid5 | frzero | 1 | 0 | v3_z |  |
| iid5 | frzero | 1 | 0 | vanilla | 10.0000 |
| iid5 | frzero | 1 | 1 | flirds_gate_v1 | 9.0000 |
| iid5 | frzero | 1 | 1 | flirds_gate_v2 | 9.0000 |
| iid5 | frzero | 1 | 1 | flirds_w | 9.0000 |
| iid5 | frzero | 1 | 1 | flirds_zgate_v2 | 8.0000 |
| iid5 | frzero | 1 | 1 | lossheur_gate_v2 | 9.0000 |
| iid5 | frzero | 1 | 1 | oracle_excl | 9.0000 |
| iid5 | frzero | 1 | 1 | random_excl |  |
| iid5 | frzero | 1 | 1 | v3_random |  |
| iid5 | frzero | 1 | 1 | v3_sign |  |
| iid5 | frzero | 1 | 1 | v3_z |  |
| iid5 | frzero | 1 | 1 | vanilla | 10.0000 |
| iid5 | frzero | 1 | 2 | flirds_gate_v1 | 9.0000 |
| iid5 | frzero | 1 | 2 | flirds_gate_v2 | 9.0000 |
| iid5 | frzero | 1 | 2 | flirds_w | 9.0000 |
| iid5 | frzero | 1 | 2 | flirds_zgate_v2 | 8.0000 |
| iid5 | frzero | 1 | 2 | lossheur_gate_v2 | 9.0000 |
| iid5 | frzero | 1 | 2 | oracle_excl | 9.0000 |
| iid5 | frzero | 1 | 2 | random_excl |  |
| iid5 | frzero | 1 | 2 | v3_random |  |
| iid5 | frzero | 1 | 2 | v3_sign |  |
| iid5 | frzero | 1 | 2 | v3_z |  |
| iid5 | frzero | 1 | 2 | vanilla | 10.0000 |
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
| silo5 | clean | 1 | 2 | flirds_gate_v1 | 10.0000 |
| silo5 | clean | 1 | 2 | flirds_gate_v2 |  |
| silo5 | clean | 1 | 2 | flirds_w | 8.0000 |
| silo5 | clean | 1 | 2 | flirds_zgate_v2 |  |
| silo5 | clean | 1 | 2 | lossheur_gate_v2 |  |
| silo5 | clean | 1 | 2 | oracleb_gate_v2 |  |
| silo5 | clean | 1 | 2 | v3_random |  |
| silo5 | clean | 1 | 2 | v3_sign |  |
| silo5 | clean | 1 | 2 | v3_z |  |
| silo5 | clean | 1 | 2 | vanilla | 10.0000 |
| silo5 | frrand | 1 | 0 | flirds_gate_v1 | 9.0000 |
| silo5 | frrand | 1 | 0 | flirds_gate_v2 | 9.0000 |
| silo5 | frrand | 1 | 0 | flirds_w | 8.0000 |
| silo5 | frrand | 1 | 0 | flirds_zgate_v2 | 9.0000 |
| silo5 | frrand | 1 | 0 | lossheur_gate_v2 | 9.0000 |
| silo5 | frrand | 1 | 0 | oracle_excl | 9.0000 |
| silo5 | frrand | 1 | 0 | oracleb_gate_v2 | 9.0000 |
| silo5 | frrand | 1 | 0 | random_excl | 9.0000 |
| silo5 | frrand | 1 | 0 | v3_random |  |
| silo5 | frrand | 1 | 0 | v3_sign |  |
| silo5 | frrand | 1 | 0 | v3_z |  |
| silo5 | frrand | 1 | 0 | vanilla | 10.0000 |
| silo5 | frrand | 1 | 1 | flirds_gate_v1 | 10.0000 |
| silo5 | frrand | 1 | 1 | flirds_gate_v2 | 10.0000 |
| silo5 | frrand | 1 | 1 | flirds_w | 8.0000 |
| silo5 | frrand | 1 | 1 | flirds_zgate_v2 | 9.0000 |
| silo5 | frrand | 1 | 1 | lossheur_gate_v2 | 10.0000 |
| silo5 | frrand | 1 | 1 | oracle_excl | 9.0000 |
| silo5 | frrand | 1 | 1 | oracleb_gate_v2 | 10.0000 |
| silo5 | frrand | 1 | 1 | random_excl | 10.0000 |
| silo5 | frrand | 1 | 1 | v3_random |  |
| silo5 | frrand | 1 | 1 | v3_sign |  |
| silo5 | frrand | 1 | 1 | v3_z |  |
| silo5 | frrand | 1 | 1 | vanilla | 10.0000 |
| silo5 | frrand | 1 | 2 | flirds_gate_v1 | 10.0000 |
| silo5 | frrand | 1 | 2 | flirds_gate_v2 | 10.0000 |
| silo5 | frrand | 1 | 2 | flirds_w | 7.0000 |
| silo5 | frrand | 1 | 2 | flirds_zgate_v2 |  |
| silo5 | frrand | 1 | 2 | lossheur_gate_v2 | 9.0000 |
| silo5 | frrand | 1 | 2 | oracle_excl | 9.0000 |
| silo5 | frrand | 1 | 2 | oracleb_gate_v2 | 10.0000 |
| silo5 | frrand | 1 | 2 | random_excl | 10.0000 |
| silo5 | frrand | 1 | 2 | v3_random |  |
| silo5 | frrand | 1 | 2 | v3_sign |  |
| silo5 | frrand | 1 | 2 | v3_z |  |
| silo5 | frrand | 1 | 2 | vanilla | 10.0000 |
| silo5 | frzero | 1 | 0 | flirds_gate_v1 | 9.0000 |
| silo5 | frzero | 1 | 0 | flirds_gate_v2 | 9.0000 |
| silo5 | frzero | 1 | 0 | flirds_w | 8.0000 |
| silo5 | frzero | 1 | 0 | flirds_zgate_v2 | 9.0000 |
| silo5 | frzero | 1 | 0 | lossheur_gate_v2 | 9.0000 |
| silo5 | frzero | 1 | 0 | oracle_excl | 9.0000 |
| silo5 | frzero | 1 | 0 | oracleb_gate_v2 | 9.0000 |
| silo5 | frzero | 1 | 0 | random_excl | 9.0000 |
| silo5 | frzero | 1 | 0 | v3_random |  |
| silo5 | frzero | 1 | 0 | v3_sign |  |
| silo5 | frzero | 1 | 0 | v3_z |  |
| silo5 | frzero | 1 | 0 | vanilla | 10.0000 |
| silo5 | frzero | 1 | 1 | flirds_gate_v1 | 9.0000 |
| silo5 | frzero | 1 | 1 | flirds_gate_v2 | 9.0000 |
| silo5 | frzero | 1 | 1 | flirds_w | 8.0000 |
| silo5 | frzero | 1 | 1 | flirds_zgate_v2 | 9.0000 |
| silo5 | frzero | 1 | 1 | lossheur_gate_v2 | 9.0000 |
| silo5 | frzero | 1 | 1 | oracle_excl | 9.0000 |
| silo5 | frzero | 1 | 1 | oracleb_gate_v2 | 9.0000 |
| silo5 | frzero | 1 | 1 | random_excl | 10.0000 |
| silo5 | frzero | 1 | 1 | v3_random |  |
| silo5 | frzero | 1 | 1 | v3_sign |  |
| silo5 | frzero | 1 | 1 | v3_z |  |
| silo5 | frzero | 1 | 1 | vanilla | 10.0000 |
| silo5 | frzero | 1 | 2 | flirds_gate_v1 | 9.0000 |
| silo5 | frzero | 1 | 2 | flirds_gate_v2 | 9.0000 |
| silo5 | frzero | 1 | 2 | flirds_w | 7.0000 |
| silo5 | frzero | 1 | 2 | flirds_zgate_v2 | 10.0000 |
| silo5 | frzero | 1 | 2 | lossheur_gate_v2 | 9.0000 |
| silo5 | frzero | 1 | 2 | oracle_excl | 9.0000 |
| silo5 | frzero | 1 | 2 | oracleb_gate_v2 | 9.0000 |
| silo5 | frzero | 1 | 2 | random_excl | 10.0000 |
| silo5 | frzero | 1 | 2 | v3_random |  |
| silo5 | frzero | 1 | 2 | v3_sign |  |
| silo5 | frzero | 1 | 2 | v3_z |  |
| silo5 | frzero | 1 | 2 | vanilla | 10.0000 |
| silo5 | noisy | 0.75 | 0 | flirds_gate_v1 | 10.0000 |
| silo5 | noisy | 0.75 | 0 | flirds_gate_v2 | 10.0000 |
| silo5 | noisy | 0.75 | 0 | flirds_w | 8.0000 |
| silo5 | noisy | 0.75 | 0 | flirds_zgate_v2 | 10.0000 |
| silo5 | noisy | 0.75 | 0 | lossheur_gate_v2 | 10.0000 |
| silo5 | noisy | 0.75 | 0 | oracle_excl | 10.0000 |
| silo5 | noisy | 0.75 | 0 | oracleb_gate_v2 | 10.0000 |
| silo5 | noisy | 0.75 | 0 | random_excl | 10.0000 |
| silo5 | noisy | 0.75 | 0 | v3_random |  |
| silo5 | noisy | 0.75 | 0 | v3_sign |  |
| silo5 | noisy | 0.75 | 0 | v3_z |  |
| silo5 | noisy | 0.75 | 0 | vanilla | 10.0000 |
| silo5 | noisy | 1 | 0 | flirds_gate_v1 | 10.0000 |
| silo5 | noisy | 1 | 0 | flirds_gate_v2 | 10.0000 |
| silo5 | noisy | 1 | 0 | flirds_w | 8.0000 |
| silo5 | noisy | 1 | 0 | flirds_zgate_v2 | 10.0000 |
| silo5 | noisy | 1 | 0 | lossheur_gate_v2 | 10.0000 |
| silo5 | noisy | 1 | 0 | oracle_excl | 10.0000 |
| silo5 | noisy | 1 | 0 | oracleb_gate_v2 | 10.0000 |
| silo5 | noisy | 1 | 0 | random_excl |  |
| silo5 | noisy | 1 | 0 | v3_random |  |
| silo5 | noisy | 1 | 0 | v3_sign |  |
| silo5 | noisy | 1 | 0 | v3_z |  |
| silo5 | noisy | 1 | 0 | vanilla | 10.0000 |
| silo5 | noisy | 1 | 1 | flirds_gate_v1 | 10.0000 |
| silo5 | noisy | 1 | 1 | flirds_gate_v2 | 10.0000 |
| silo5 | noisy | 1 | 1 | flirds_w | 8.0000 |
| silo5 | noisy | 1 | 1 | flirds_zgate_v2 | 10.0000 |
| silo5 | noisy | 1 | 1 | lossheur_gate_v2 | 10.0000 |
| silo5 | noisy | 1 | 1 | oracle_excl | 10.0000 |
| silo5 | noisy | 1 | 1 | oracleb_gate_v2 | 10.0000 |
| silo5 | noisy | 1 | 1 | random_excl | 10.0000 |
| silo5 | noisy | 1 | 1 | v3_random |  |
| silo5 | noisy | 1 | 1 | v3_sign |  |
| silo5 | noisy | 1 | 1 | v3_z |  |
| silo5 | noisy | 1 | 1 | vanilla | 10.0000 |
| silo5 | noisy | 1 | 2 | flirds_gate_v1 | 10.0000 |
| silo5 | noisy | 1 | 2 | flirds_gate_v2 | 10.0000 |
| silo5 | noisy | 1 | 2 | flirds_w | 8.0000 |
| silo5 | noisy | 1 | 2 | flirds_zgate_v2 | 10.0000 |
| silo5 | noisy | 1 | 2 | lossheur_gate_v2 | 10.0000 |
| silo5 | noisy | 1 | 2 | oracle_excl | 9.0000 |
| silo5 | noisy | 1 | 2 | oracleb_gate_v2 | 10.0000 |
| silo5 | noisy | 1 | 2 | random_excl | 10.0000 |
| silo5 | noisy | 1 | 2 | v3_random |  |
| silo5 | noisy | 1 | 2 | v3_sign |  |
| silo5 | noisy | 1 | 2 | v3_z |  |
| silo5 | noisy | 1 | 2 | vanilla | 10.0000 |
| std50k5 | mixed | 1 | 0 | flirds_gate_v2 | 151.0000 |
| std50k5 | mixed | 1 | 0 | oracle_excl | 156.0000 |
| std50k5 | mixed | 1 | 0 | random_excl | 193.0000 |
| std50k5 | mixed | 1 | 0 | shapleyfl_gate_v2 | 154.0000 |
| std50k5 | mixed | 1 | 0 | vanilla | 198.0000 |
| std50k5 | mixed | 1 | 1 | flirds_gate_v2 |  |
| std50k5 | mixed | 1 | 1 | shapleyfl_gate_v2 |  |
| std50k5 | mixed | 1 | 2 | shapleyfl_gate_v2 |  |

## [3] gate accuracy (per-round excluded set vs corrupt; micro P/R) + vanilla-observer per-round false-fire

| regime | threat | nr | seed | arm | precision | recall | n_excluded_pairs | false_excl_pairs | n_fallback_rounds |
|---|---|---|---|---|---|---|---|---|---|
| iid5 | clean | 1 | 0 | flirds_gate_v1 |  |  | 0 | 0 | 0 |
| iid5 | clean | 1 | 0 | flirds_gate_v2 |  |  | 0 | 0 | 0 |
| iid5 | clean | 1 | 0 | flirds_zgate_v2 | 0.0000 |  | 2 | 2 | 0 |
| iid5 | clean | 1 | 0 | lossheur_gate_v2 |  |  | 0 | 0 | 0 |
| iid5 | clean | 1 | 0 | vanilla |  |  | 0 | 0 | 0 |
| iid5 | clean | 1 | 1 | flirds_gate_v1 |  |  | 0 | 0 | 0 |
| iid5 | clean | 1 | 1 | flirds_gate_v2 |  |  | 0 | 0 | 0 |
| iid5 | clean | 1 | 1 | flirds_zgate_v2 | 0.0000 |  | 6 | 6 | 0 |
| iid5 | clean | 1 | 1 | lossheur_gate_v2 |  |  | 0 | 0 | 0 |
| iid5 | clean | 1 | 1 | vanilla |  |  | 0 | 0 | 0 |
| iid5 | clean | 1 | 2 | flirds_gate_v1 |  |  | 0 | 0 | 0 |
| iid5 | clean | 1 | 2 | flirds_gate_v2 |  |  | 0 | 0 | 0 |
| iid5 | clean | 1 | 2 | flirds_zgate_v2 | 0.0000 |  | 12 | 12 | 0 |
| iid5 | clean | 1 | 2 | lossheur_gate_v2 |  |  | 0 | 0 | 0 |
| iid5 | clean | 1 | 2 | vanilla |  |  | 0 | 0 | 0 |
| iid5 | frzero | 1 | 0 | flirds_gate_v1 | 1.0000 | 1.0000 | 10 | 0 | 0 |
| iid5 | frzero | 1 | 0 | flirds_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| iid5 | frzero | 1 | 0 | flirds_zgate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| iid5 | frzero | 1 | 0 | lossheur_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| iid5 | frzero | 1 | 0 | vanilla |  | 0.0000 | 0 | 0 | 0 |
| iid5 | frzero | 1 | 1 | flirds_gate_v1 | 1.0000 | 1.0000 | 10 | 0 | 0 |
| iid5 | frzero | 1 | 1 | flirds_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| iid5 | frzero | 1 | 1 | flirds_zgate_v2 | 0.5833 | 1.0000 | 12 | 5 | 0 |
| iid5 | frzero | 1 | 1 | lossheur_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| iid5 | frzero | 1 | 1 | vanilla |  | 0.0000 | 0 | 0 | 0 |
| iid5 | frzero | 1 | 2 | flirds_gate_v1 | 1.0000 | 1.0000 | 10 | 0 | 0 |
| iid5 | frzero | 1 | 2 | flirds_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| iid5 | frzero | 1 | 2 | flirds_zgate_v2 | 0.5833 | 1.0000 | 12 | 5 | 0 |
| iid5 | frzero | 1 | 2 | lossheur_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| iid5 | frzero | 1 | 2 | vanilla |  | 0.0000 | 0 | 0 | 0 |
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
| silo5 | clean | 1 | 2 | flirds_gate_v1 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 2 | flirds_gate_v2 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 2 | flirds_zgate_v2 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 2 | lossheur_gate_v2 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 2 | oracleb_gate_v2 |  |  | 0 | 0 | 0 |
| silo5 | clean | 1 | 2 | vanilla |  |  | 0 | 0 | 0 |
| silo5 | frrand | 1 | 0 | flirds_gate_v1 | 1.0000 | 0.8000 | 8 | 0 | 0 |
| silo5 | frrand | 1 | 0 | flirds_gate_v2 | 1.0000 | 0.8571 | 6 | 0 | 0 |
| silo5 | frrand | 1 | 0 | flirds_zgate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| silo5 | frrand | 1 | 0 | lossheur_gate_v2 | 1.0000 | 0.8571 | 6 | 0 | 0 |
| silo5 | frrand | 1 | 0 | oracleb_gate_v2 | 1.0000 | 0.7143 | 5 | 0 | 0 |
| silo5 | frrand | 1 | 0 | vanilla |  | 0.0000 | 0 | 0 | 0 |
| silo5 | frrand | 1 | 1 | flirds_gate_v1 | 1.0000 | 0.3000 | 3 | 0 | 0 |
| silo5 | frrand | 1 | 1 | flirds_gate_v2 | 1.0000 | 0.4286 | 3 | 0 | 0 |
| silo5 | frrand | 1 | 1 | flirds_zgate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| silo5 | frrand | 1 | 1 | lossheur_gate_v2 | 1.0000 | 0.4286 | 3 | 0 | 0 |
| silo5 | frrand | 1 | 1 | oracleb_gate_v2 | 1.0000 | 0.2857 | 2 | 0 | 0 |
| silo5 | frrand | 1 | 1 | vanilla |  | 0.0000 | 0 | 0 | 0 |
| silo5 | frrand | 1 | 2 | flirds_gate_v1 | 1.0000 | 0.4000 | 4 | 0 | 0 |
| silo5 | frrand | 1 | 2 | flirds_gate_v2 | 1.0000 | 0.2857 | 2 | 0 | 0 |
| silo5 | frrand | 1 | 2 | flirds_zgate_v2 | 1.0000 | 0.1429 | 1 | 0 | 0 |
| silo5 | frrand | 1 | 2 | lossheur_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| silo5 | frrand | 1 | 2 | oracleb_gate_v2 | 1.0000 | 0.1429 | 1 | 0 | 0 |
| silo5 | frrand | 1 | 2 | vanilla |  | 0.0000 | 0 | 0 | 0 |
| silo5 | frzero | 1 | 0 | flirds_gate_v1 | 1.0000 | 1.0000 | 10 | 0 | 0 |
| silo5 | frzero | 1 | 0 | flirds_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| silo5 | frzero | 1 | 0 | flirds_zgate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| silo5 | frzero | 1 | 0 | lossheur_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| silo5 | frzero | 1 | 0 | oracleb_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| silo5 | frzero | 1 | 0 | vanilla |  | 0.0000 | 0 | 0 | 0 |
| silo5 | frzero | 1 | 1 | flirds_gate_v1 | 1.0000 | 1.0000 | 10 | 0 | 0 |
| silo5 | frzero | 1 | 1 | flirds_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| silo5 | frzero | 1 | 1 | flirds_zgate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| silo5 | frzero | 1 | 1 | lossheur_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| silo5 | frzero | 1 | 1 | oracleb_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| silo5 | frzero | 1 | 1 | vanilla |  | 0.0000 | 0 | 0 | 0 |
| silo5 | frzero | 1 | 2 | flirds_gate_v1 | 1.0000 | 1.0000 | 10 | 0 | 0 |
| silo5 | frzero | 1 | 2 | flirds_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| silo5 | frzero | 1 | 2 | flirds_zgate_v2 | 1.0000 | 0.1429 | 1 | 0 | 0 |
| silo5 | frzero | 1 | 2 | lossheur_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| silo5 | frzero | 1 | 2 | oracleb_gate_v2 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| silo5 | frzero | 1 | 2 | vanilla |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 0.75 | 0 | flirds_gate_v1 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 0.75 | 0 | flirds_gate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 0.75 | 0 | flirds_zgate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 0.75 | 0 | lossheur_gate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 0.75 | 0 | oracleb_gate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 0.75 | 0 | vanilla |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 0 | flirds_gate_v1 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 0 | flirds_gate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 0 | flirds_zgate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 0 | lossheur_gate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 0 | oracleb_gate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 0 | vanilla |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 1 | flirds_gate_v1 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 1 | flirds_gate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 1 | flirds_zgate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 1 | lossheur_gate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 1 | oracleb_gate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 1 | vanilla |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 2 | flirds_gate_v1 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 2 | flirds_gate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 2 | flirds_zgate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 2 | lossheur_gate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 2 | oracleb_gate_v2 |  | 0.0000 | 0 | 0 | 0 |
| silo5 | noisy | 1 | 2 | vanilla |  | 0.0000 | 0 | 0 | 0 |
| std50k5 | mixed | 1 | 0 | flirds_gate_v2 | 0.9281 | 0.5705 | 1168 | 84 | 0 |
| std50k5 | mixed | 1 | 0 | shapleyfl_gate_v2 | 0.9011 | 0.8584 | 1810 | 179 | 0 |
| std50k5 | mixed | 1 | 0 | vanilla |  | 0.0000 | 0 | 0 | 0 |
| std50k5 | mixed | 1 | 1 | flirds_gate_v2 | 0.8856 | 0.6479 | 1390 | 159 | 4 |
| std50k5 | mixed | 1 | 1 | shapleyfl_gate_v2 | 0.8366 | 0.8568 | 1946 | 318 | 0 |
| std50k5 | mixed | 1 | 2 | shapleyfl_gate_v2 | 0.8726 | 0.7321 | 1594 | 203 | 0 |

vanilla observer (per-round raw, the project's first per-round phi record):

| regime | threat | nr | seed | clean_raw_false_fire_rate | all_clean_cum_pos_from_round |
|---|---|---|---|---|---|
| iid5 | clean | 1 | 0 | 0.0000 | 0.0000 |
| iid5 | clean | 1 | 1 | 0.0000 | 0.0000 |
| iid5 | clean | 1 | 2 | 0.0000 | 0.0000 |
| iid5 | frzero | 1 | 0 | 0.0000 | 0.0000 |
| iid5 | frzero | 1 | 1 | 0.0000 | 0.0000 |
| iid5 | frzero | 1 | 2 | 0.0000 | 0.0000 |
| silo5 | clean | 1 | 0 | 0.0000 | 0.0000 |
| silo5 | clean | 1 | 1 | 0.0000 | 0.0000 |
| silo5 | clean | 1 | 2 | 0.0000 | 0.0000 |
| silo5 | frrand | 1 | 0 | 0.0000 | 0.0000 |
| silo5 | frrand | 1 | 1 | 0.0000 | 0.0000 |
| silo5 | frrand | 1 | 2 | 0.0000 | 0.0000 |
| silo5 | frzero | 1 | 0 | 0.0000 | 0.0000 |
| silo5 | frzero | 1 | 1 | 0.0000 | 0.0000 |
| silo5 | frzero | 1 | 2 | 0.0000 | 0.0000 |
| silo5 | noisy | 0.75 | 0 | 0.0000 | 0.0000 |
| silo5 | noisy | 1 | 0 | 0.0000 | 0.0000 |
| silo5 | noisy | 1 | 1 | 0.0000 | 0.0000 |
| silo5 | noisy | 1 | 2 | 0.0000 | 0.0000 |
| std50k5 | mixed | 1 | 0 | 0.0025 | 54.0000 |

## CNN (track_c2 gate cells)

| dataset | partition | threat | strength | flip_rate | seed | arm | final_acc | delta_acc | gap | recovery | auroc | false_excl_pairs | excl_precision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar10 | dir1 | clean | main |  | 0 | flirds_gate_v1 | 0.6448 | +0.0059 |  |  |  | 428.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 0 | flirds_gate_v2 | 0.6246 | -0.0142 |  |  |  | 4110.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 0 | flirds_gatew_v1 | 0.6419 | +0.0030 |  |  |  | 324.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 0 | flirds_gatew_v2 | 0.6189 | -0.0200 |  |  |  | 3403.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 0 | flirds_mult | 0.6434 | +0.0045 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 0 | flirds_zgate_v2 | 0.6345 | -0.0044 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 0 | vanilla | 0.6389 | +0.0000 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 1 | flirds_gate_v1 | 0.6384 | +0.0048 |  |  |  | 437.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 1 | flirds_gate_v2 | 0.6394 | +0.0058 |  |  |  | 3152.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 1 | flirds_gatew_v1 | 0.6321 | -0.0015 |  |  |  | 325.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 1 | flirds_gatew_v2 | 0.6236 | -0.0100 |  |  |  | 2860.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 1 | flirds_mult | 0.6394 | +0.0058 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 1 | flirds_zgate_v2 | 0.6424 | +0.0088 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 1 | vanilla | 0.6336 | +0.0000 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 2 | flirds_gate_v1 | 0.6239 | -0.0202 |  |  |  | 454.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 2 | flirds_gate_v2 | 0.6305 | -0.0136 |  |  |  | 4162.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 2 | flirds_gatew_v1 | 0.5854 | -0.0587 |  |  |  | 323.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 2 | flirds_gatew_v2 | 0.6139 | -0.0302 |  |  |  | 3393.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 2 | flirds_mult | 0.6449 | +0.0008 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 2 | flirds_zgate_v2 | 0.6254 | -0.0187 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 2 | vanilla | 0.6441 | +0.0000 |  |  |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 0 | flirds_gate_v1 | 0.6174 | +0.0306 | 0.0336 | +0.9108 | 0.9833 | 157.0000 | 0.7535 |
| cifar10 | dir1 | free_rider | main |  | 0 | flirds_gate_v2 | 0.6146 | +0.0279 | 0.0336 | +0.8290 | 0.7833 | 1220.0000 | 0.7716 |
| cifar10 | dir1 | free_rider | main |  | 0 | flirds_gatew_v1 | 0.6244 | +0.0376 | 0.0336 | +1.1190 | 1.0000 | 100.0000 | 0.8276 |
| cifar10 | dir1 | free_rider | main |  | 0 | flirds_gatew_v2 | 0.6114 | +0.0246 | 0.0336 | +0.7323 | 0.7000 | 1426.0000 | 0.7436 |
| cifar10 | dir1 | free_rider | main |  | 0 | flirds_mult | 0.5980 | +0.0112 | 0.0336 | +0.3346 | 0.3571 |  |  |
| cifar10 | dir1 | free_rider | main |  | 0 | flirds_zgate_v2 | 0.5817 | -0.0050 | 0.0336 | -0.1487 | 0.5167 |  |  |
| cifar10 | dir1 | free_rider | main |  | 0 | oracle_excl | 0.6204 | +0.0336 | 0.0336 | +1.0000 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 0 | random_excl | 0.5930 | +0.0062 | 0.0336 | +0.1859 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 0 | vanilla | 0.5867 | +0.0000 | 0.0336 | +0.0000 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 1 | flirds_gate_v1 | 0.6131 | +0.0219 | 0.0262 | +0.8333 | 0.9667 | 159.0000 | 0.7508 |
| cifar10 | dir1 | free_rider | main |  | 1 | flirds_gate_v2 | 0.6150 | +0.0237 | 0.0262 | +0.9048 | 0.6667 | 1619.0000 | 0.7139 |
| cifar10 | dir1 | free_rider | main |  | 1 | flirds_gatew_v1 | 0.6145 | +0.0232 | 0.0262 | +0.8857 | 0.9667 | 117.0000 | 0.8037 |
| cifar10 | dir1 | free_rider | main |  | 1 | flirds_gatew_v2 | 0.6146 | +0.0234 | 0.0262 | +0.8905 | 0.6833 | 1264.0000 | 0.7605 |
| cifar10 | dir1 | free_rider | main |  | 1 | flirds_mult | 0.5994 | +0.0081 | 0.0262 | +0.3095 | 0.4313 |  |  |
| cifar10 | dir1 | free_rider | main |  | 1 | flirds_zgate_v2 | 0.5827 | -0.0085 | 0.0262 | -0.3238 | 0.6167 |  |  |
| cifar10 | dir1 | free_rider | main |  | 1 | oracle_excl | 0.6175 | +0.0262 | 0.0262 | +1.0000 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 1 | random_excl | 0.5979 | +0.0066 | 0.0262 | +0.2524 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 1 | vanilla | 0.5913 | +0.0000 | 0.0262 | +0.0000 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 2 | flirds_gate_v1 | 0.6012 | +0.0155 | 0.0374 | +0.4147 | 1.0000 | 154.0000 | 0.7609 |
| cifar10 | dir1 | free_rider | main |  | 2 | flirds_gate_v2 | 0.6149 | +0.0291 | 0.0374 | +0.7793 | 0.6667 | 1803.0000 | 0.6878 |
| cifar10 | dir1 | free_rider | main |  | 2 | flirds_gatew_v1 | 0.5950 | +0.0092 | 0.0374 | +0.2475 | 1.0000 | 104.0000 | 0.8249 |
| cifar10 | dir1 | free_rider | main |  | 2 | flirds_gatew_v2 | 0.5909 | +0.0051 | 0.0374 | +0.1371 | 0.7167 | 1310.0000 | 0.7511 |
| cifar10 | dir1 | free_rider | main |  | 2 | flirds_mult | 0.5962 | +0.0105 | 0.0374 | +0.2809 | 0.4025 |  |  |
| cifar10 | dir1 | free_rider | main |  | 2 | flirds_zgate_v2 | 0.5875 | +0.0018 | 0.0374 | +0.0468 | 0.5167 |  |  |
| cifar10 | dir1 | free_rider | main |  | 2 | oracle_excl | 0.6231 | +0.0374 | 0.0374 | +1.0000 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 2 | random_excl | 0.5606 | -0.0251 | 0.0374 | -0.6722 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 2 | vanilla | 0.5857 | +0.0000 | 0.0374 | +0.0000 |  |  |  |
| cifar10 | dir1 | frrand | main |  | 0 | flirds_gate_v1 | 0.5999 | +0.0134 | 0.0336 | +0.3978 | 0.8996 | 267.0000 | 0.4785 |
| cifar10 | dir1 | frrand | main |  | 0 | flirds_gate_v2 | 0.5806 | -0.0059 | 0.0336 | -0.1747 | 0.5096 | 2514.0000 | 0.5195 |
| cifar10 | dir1 | frrand | main |  | 0 | flirds_gatew_v1 | 0.6254 | +0.0389 | 0.0336 | +1.1561 | 1.0000 | 99.0000 | 0.7188 |
| cifar10 | dir1 | frrand | main |  | 0 | flirds_gatew_v2 | 0.6109 | +0.0244 | 0.0336 | +0.7249 | 0.8500 | 886.0000 | 0.7582 |
| cifar10 | dir1 | frrand | main |  | 0 | flirds_mult | 0.5971 | +0.0106 | 0.0336 | +0.3160 | 0.3567 |  |  |
| cifar10 | dir1 | frrand | main |  | 0 | flirds_zgate_v2 | 0.5909 | +0.0044 | 0.0336 | +0.1301 | 0.5858 |  |  |
| cifar10 | dir1 | frrand | main |  | 0 | oracle_excl | 0.6201 | +0.0336 | 0.0336 | +1.0000 |  |  |  |
| cifar10 | dir1 | frrand | main |  | 0 | random_excl | 0.5934 | +0.0069 | 0.0336 | +0.2045 |  |  |  |
| cifar10 | dir1 | frrand | main |  | 0 | vanilla | 0.5865 | +0.0000 | 0.0336 | +0.0000 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 0 | flirds_gate_v1 | 0.5724 | +0.3066 | 0.3546 | +0.8646 | 0.9987 | 203.0000 | 0.6915 |
| cifar10 | dir1 | grad_noise | main |  | 0 | flirds_gate_v2 | 0.5370 | +0.2713 | 0.3546 | +0.7649 | 0.9892 | 1243.0000 | 0.7649 |
| cifar10 | dir1 | grad_noise | main |  | 0 | flirds_gatew_v1 | 0.5449 | +0.2791 | 0.3546 | +0.7871 | 0.9996 | 172.0000 | 0.7287 |
| cifar10 | dir1 | grad_noise | main |  | 0 | flirds_gatew_v2 | 0.5783 | +0.3125 | 0.3546 | +0.8812 | 0.9788 | 1651.0000 | 0.7141 |
| cifar10 | dir1 | grad_noise | main |  | 0 | flirds_mult | 0.4629 | +0.1971 | 0.3546 | +0.5559 | 0.9925 |  |  |
| cifar10 | dir1 | grad_noise | main |  | 0 | flirds_zgate_v2 | 0.3789 | +0.1131 | 0.3546 | +0.3190 | 0.9967 |  |  |
| cifar10 | dir1 | grad_noise | main |  | 0 | oracle_excl | 0.6204 | +0.3546 | 0.3546 | +1.0000 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 0 | random_excl | 0.2411 | -0.0246 | 0.3546 | -0.0694 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 0 | vanilla | 0.2657 | +0.0000 | 0.3546 | +0.0000 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 1 | flirds_gate_v1 | 0.5129 | +0.2915 | 0.3961 | +0.7359 | 0.9996 | 194.0000 | 0.7025 |
| cifar10 | dir1 | grad_noise | main |  | 1 | flirds_gate_v2 | 0.5853 | +0.3639 | 0.3961 | +0.9186 | 0.9950 | 1607.0000 | 0.7158 |
| cifar10 | dir1 | grad_noise | main |  | 1 | flirds_gatew_v1 | 0.5058 | +0.2844 | 0.3961 | +0.7179 | 0.9983 | 209.0000 | 0.6819 |
| cifar10 | dir1 | grad_noise | main |  | 1 | flirds_gatew_v2 | 0.5944 | +0.3730 | 0.3961 | +0.9416 | 0.9954 | 1766.0000 | 0.6925 |
| cifar10 | dir1 | grad_noise | main |  | 1 | flirds_mult | 0.4457 | +0.2244 | 0.3961 | +0.5664 | 0.9975 |  |  |
| cifar10 | dir1 | grad_noise | main |  | 1 | flirds_zgate_v2 | 0.3287 | +0.1074 | 0.3961 | +0.2711 | 0.9946 |  |  |
| cifar10 | dir1 | grad_noise | main |  | 1 | oracle_excl | 0.6175 | +0.3961 | 0.3961 | +1.0000 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 1 | random_excl | 0.2819 | +0.0605 | 0.3961 | +0.1527 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 1 | vanilla | 0.2214 | +0.0000 | 0.3961 | +0.0000 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 2 | flirds_gate_v1 | 0.5533 | +0.3095 | 0.3794 | +0.8158 | 1.0000 | 215.0000 | 0.6884 |
| cifar10 | dir1 | grad_noise | main |  | 2 | flirds_gate_v2 | 0.5781 | +0.3344 | 0.3794 | +0.8814 | 0.9912 | 1231.0000 | 0.7664 |
| cifar10 | dir1 | grad_noise | main |  | 2 | flirds_gatew_v1 | 0.5034 | +0.2596 | 0.3794 | +0.6843 | 1.0000 | 166.0000 | 0.7390 |
| cifar10 | dir1 | grad_noise | main |  | 2 | flirds_gatew_v2 | 0.5896 | +0.3459 | 0.3794 | +0.9117 | 0.9946 | 1458.0000 | 0.7334 |
| cifar10 | dir1 | grad_noise | main |  | 2 | flirds_mult | 0.4005 | +0.1568 | 0.3794 | +0.4132 | 0.9908 |  |  |
| cifar10 | dir1 | grad_noise | main |  | 2 | flirds_zgate_v2 | 0.3180 | +0.0743 | 0.3794 | +0.1957 | 1.0000 |  |  |
| cifar10 | dir1 | grad_noise | main |  | 2 | oracle_excl | 0.6231 | +0.3794 | 0.3794 | +1.0000 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 2 | random_excl | 0.2541 | +0.0104 | 0.3794 | +0.0273 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 2 | vanilla | 0.2437 | +0.0000 | 0.3794 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | flirds_gate_v1 | 0.6235 | +0.0015 | 0.0051 |  | 0.5393 | 244.0000 | 0.4163 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | flirds_gate_v1 | 0.6061 | +0.0074 | 0.0284 | +0.2599 | 0.7306 | 194.0000 | 0.5571 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | flirds_gate_v1 | 0.6065 | +0.0505 | 0.0711 | +0.7100 | 0.9924 | 158.0000 | 0.6715 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | flirds_gate_v2 | 0.6159 | -0.0061 | 0.0051 |  | 0.6049 | 1025.0000 | 0.6047 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | flirds_gate_v2 | 0.5936 | -0.0051 | 0.0284 | -0.1806 | 0.6604 | 1524.0000 | 0.5818 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | flirds_gate_v2 | 0.5641 | +0.0081 | 0.0711 | +0.1142 | 0.8083 | 1246.0000 | 0.7063 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | flirds_gatew_v1 | 0.6224 | +0.0004 | 0.0051 |  | 0.5721 | 185.0000 | 0.4108 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | flirds_gatew_v1 | 0.6012 | +0.0025 | 0.0284 | +0.0881 | 0.6789 | 155.0000 | 0.5646 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | flirds_gatew_v1 | 0.5360 | -0.0200 | 0.0711 | -0.2812 | 0.9462 | 127.0000 | 0.6760 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | flirds_gatew_v2 | 0.5978 | -0.0242 | 0.0051 |  | 0.4426 | 1746.0000 | 0.4149 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | flirds_gatew_v2 | 0.5833 | -0.0155 | 0.0284 | -0.5463 | 0.5532 | 1071.0000 | 0.5906 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | flirds_gatew_v2 | 0.5938 | +0.0377 | 0.0711 | +0.5308 | 0.6679 | 1292.0000 | 0.6704 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | flirds_mult | 0.6246 | +0.0026 | 0.0051 |  | 0.4435 |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | flirds_mult | 0.6124 | +0.0136 | 0.0284 | +0.4802 | 0.6646 |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | flirds_mult | 0.5995 | +0.0435 | 0.0711 | +0.6116 | 0.8676 |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | flirds_zgate_v2 | 0.6148 | -0.0072 | 0.0051 |  | 0.5351 |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | flirds_zgate_v2 | 0.6088 | +0.0100 | 0.0284 | +0.3524 | 0.9092 |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | flirds_zgate_v2 | 0.5889 | +0.0329 | 0.0711 | +0.4622 | 0.9899 |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | oracle_excl | 0.6271 | +0.0051 | 0.0051 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | oracle_excl | 0.6271 | +0.0284 | 0.0284 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | oracle_excl | 0.6271 | +0.0711 | 0.0711 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | random_excl | 0.6066 | -0.0154 | 0.0051 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | random_excl | 0.5817 | -0.0170 | 0.0284 | -0.5991 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | random_excl | 0.5599 | +0.0039 | 0.0711 | +0.0545 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | vanilla | 0.6220 | +0.0000 | 0.0051 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | vanilla | 0.5988 | +0.0000 | 0.0284 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | vanilla | 0.5560 | +0.0000 | 0.0711 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | flirds_gate_v1 | 0.6122 | -0.0019 | 0.0072 |  | 0.5349 | 219.0000 | 0.4798 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | flirds_gate_v1 | 0.5797 | +0.0051 | 0.0467 | +0.1096 | 0.7845 | 159.0000 | 0.6521 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | flirds_gate_v1 | 0.5525 | +0.0534 | 0.1223 | +0.4366 | 0.9732 | 126.0000 | 0.7595 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | flirds_gate_v2 | 0.5951 | -0.0190 | 0.0072 |  | 0.4796 | 1547.0000 | 0.4774 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | flirds_gate_v2 | 0.5745 | -0.0001 | 0.0467 | -0.0027 | 0.6963 | 1046.0000 | 0.7269 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | flirds_gate_v2 | 0.5690 | +0.0699 | 0.1223 | +0.5716 | 0.8293 | 865.0000 | 0.8329 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | flirds_gatew_v1 | 0.6124 | -0.0018 | 0.0072 |  | 0.5240 | 184.0000 | 0.4620 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | flirds_gatew_v1 | 0.5914 | +0.0167 | 0.0467 | +0.3583 | 0.7969 | 128.0000 | 0.6649 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | flirds_gatew_v1 | 0.5410 | +0.0419 | 0.1223 | +0.3425 | 0.9567 | 104.0000 | 0.7684 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | flirds_gatew_v2 | 0.5881 | -0.0260 | 0.0072 |  | 0.4319 | 1270.0000 | 0.4998 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | flirds_gatew_v2 | 0.5811 | +0.0065 | 0.0467 | +0.1390 | 0.6635 | 1028.0000 | 0.7100 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | flirds_gatew_v2 | 0.5784 | +0.0792 | 0.1223 | +0.6483 | 0.7825 | 1539.0000 | 0.7128 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | flirds_mult | 0.6165 | +0.0024 | 0.0072 |  | 0.5557 |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | flirds_mult | 0.5881 | +0.0135 | 0.0467 | +0.2888 | 0.8550 |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | flirds_mult | 0.5755 | +0.0764 | 0.1223 | +0.6247 | 0.9339 |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | flirds_zgate_v2 | 0.6200 | +0.0059 | 0.0072 |  | 0.5877 |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | flirds_zgate_v2 | 0.5819 | +0.0072 | 0.0467 | +0.1551 | 0.9111 |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | flirds_zgate_v2 | 0.4964 | -0.0027 | 0.1223 | -0.0225 | 0.9844 |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | oracle_excl | 0.6214 | +0.0072 | 0.0072 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | oracle_excl | 0.6214 | +0.0467 | 0.0467 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | oracle_excl | 0.6214 | +0.1223 | 0.1223 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | random_excl | 0.5969 | -0.0172 | 0.0072 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | random_excl | 0.5687 | -0.0059 | 0.0467 | -0.1257 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | random_excl | 0.5071 | +0.0080 | 0.1223 | +0.0654 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | vanilla | 0.6141 | +0.0000 | 0.0072 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | vanilla | 0.5746 | +0.0000 | 0.0467 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | vanilla | 0.4991 | +0.0000 | 0.1223 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | flirds_gate_v1 | 0.6122 | -0.0034 | 0.0068 |  | 0.5548 | 221.0000 | 0.4977 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | flirds_gate_v1 | 0.6048 | +0.0235 | 0.0411 | +0.5714 | 0.8539 | 148.0000 | 0.6711 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | flirds_gate_v1 | 0.5870 | +0.0681 | 0.1035 | +0.6582 | 0.9976 | 128.0000 | 0.7571 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | flirds_gate_v2 | 0.5863 | -0.0294 | 0.0068 |  | 0.5705 | 1751.0000 | 0.5103 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | flirds_gate_v2 | 0.5486 | -0.0326 | 0.0411 | -0.7933 | 0.6636 | 2047.0000 | 0.6247 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | flirds_gate_v2 | 0.5805 | +0.0616 | 0.1035 | +0.5954 | 0.7539 | 1895.0000 | 0.6881 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | flirds_gatew_v1 | 0.5733 | -0.0424 | 0.0068 |  | 0.5640 | 163.0000 | 0.5262 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | flirds_gatew_v1 | 0.5824 | +0.0011 | 0.0411 | +0.0274 | 0.7656 | 115.0000 | 0.6909 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | flirds_gatew_v1 | 0.5669 | +0.0480 | 0.1035 | +0.4638 | 0.9735 | 107.0000 | 0.7723 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | flirds_gatew_v2 | 0.5939 | -0.0217 | 0.0068 |  | 0.5151 | 1343.0000 | 0.5388 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | flirds_gatew_v2 | 0.5819 | +0.0006 | 0.0411 | +0.0152 | 0.6471 | 951.0000 | 0.6729 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | flirds_gatew_v2 | 0.5709 | +0.0520 | 0.1035 | +0.5024 | 0.7146 | 1557.0000 | 0.7067 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | flirds_mult | 0.6104 | -0.0052 | 0.0068 |  | 0.5299 |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | flirds_mult | 0.6002 | +0.0190 | 0.0411 | +0.4620 | 0.8109 |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | flirds_mult | 0.5859 | +0.0670 | 0.1035 | +0.6473 | 0.9707 |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | flirds_zgate_v2 | 0.6099 | -0.0058 | 0.0068 |  | 0.5921 |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | flirds_zgate_v2 | 0.5735 | -0.0078 | 0.0411 | -0.1884 | 0.9342 |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | flirds_zgate_v2 | 0.5165 | -0.0024 | 0.1035 | -0.0229 | 1.0000 |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | oracle_excl | 0.6224 | +0.0068 | 0.0068 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | oracle_excl | 0.6224 | +0.0411 | 0.0411 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | oracle_excl | 0.6224 | +0.1035 | 0.1035 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | random_excl | 0.5736 | -0.0420 | 0.0068 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | random_excl | 0.5290 | -0.0523 | 0.0411 | -1.2705 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | random_excl | 0.4385 | -0.0804 | 0.1035 | -0.7766 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | vanilla | 0.6156 | +0.0000 | 0.0068 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | vanilla | 0.5813 | +0.0000 | 0.0411 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | vanilla | 0.5189 | +0.0000 | 0.1035 | +0.0000 |  |  |  |
| cifar10 | iid | clean | main |  | 0 | flirds_gate_v1 | 0.6438 | -0.0054 |  |  |  | 383.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 0 | flirds_gate_v2 | 0.6454 | -0.0037 |  |  |  | 450.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 0 | flirds_gatew_v1 | 0.6455 | -0.0036 |  |  |  | 313.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 0 | flirds_gatew_v2 | 0.6369 | -0.0122 |  |  |  | 643.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 0 | flirds_mult | 0.6469 | -0.0022 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 0 | flirds_zgate_v2 | 0.6455 | -0.0036 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 0 | vanilla | 0.6491 | +0.0000 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 1 | flirds_gate_v1 | 0.6395 | -0.0088 |  |  |  | 363.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 1 | flirds_gate_v2 | 0.6488 | +0.0005 |  |  |  | 651.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 1 | flirds_gatew_v1 | 0.6149 | -0.0334 |  |  |  | 318.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 1 | flirds_gatew_v2 | 0.6466 | -0.0016 |  |  |  | 746.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 1 | flirds_mult | 0.6481 | -0.0001 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 1 | flirds_zgate_v2 | 0.6501 | +0.0019 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 1 | vanilla | 0.6482 | +0.0000 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 2 | flirds_gate_v1 | 0.6411 | -0.0078 |  |  |  | 381.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 2 | flirds_gate_v2 | 0.6342 | -0.0146 |  |  |  | 583.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 2 | flirds_gatew_v1 | 0.6414 | -0.0075 |  |  |  | 311.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 2 | flirds_gatew_v2 | 0.6402 | -0.0086 |  |  |  | 547.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 2 | flirds_mult | 0.6450 | -0.0039 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 2 | flirds_zgate_v2 | 0.6486 | -0.0002 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 2 | vanilla | 0.6489 | +0.0000 |  |  |  |  |  |
| cifar10 | iid | free_rider | main |  | 0 | flirds_gate_v1 | 0.6048 | -0.0051 | 0.0248 | -0.2071 | 1.0000 | 123.0000 | 0.7960 |
| cifar10 | iid | free_rider | main |  | 0 | flirds_gate_v2 | 0.6256 | +0.0158 | 0.0248 | +0.6364 | 0.9333 | 176.0000 | 0.9587 |
| cifar10 | iid | free_rider | main |  | 0 | flirds_gatew_v1 | 0.6178 | +0.0079 | 0.0248 | +0.3182 | 1.0000 | 94.0000 | 0.8362 |
| cifar10 | iid | free_rider | main |  | 0 | flirds_gatew_v2 | 0.6272 | +0.0174 | 0.0248 | +0.7020 | 0.9167 | 215.0000 | 0.9500 |
| cifar10 | iid | free_rider | main |  | 0 | flirds_mult | 0.6279 | +0.0180 | 0.0248 | +0.7273 | 0.4646 |  |  |
| cifar10 | iid | free_rider | main |  | 0 | flirds_zgate_v2 | 0.6055 | -0.0044 | 0.0248 | -0.1768 | 0.9000 |  |  |
| cifar10 | iid | free_rider | main |  | 0 | oracle_excl | 0.6346 | +0.0248 | 0.0248 | +1.0000 |  |  |  |
| cifar10 | iid | free_rider | main |  | 0 | random_excl | 0.6035 | -0.0064 | 0.0248 | -0.2576 |  |  |  |
| cifar10 | iid | free_rider | main |  | 0 | vanilla | 0.6099 | +0.0000 | 0.0248 | +0.0000 |  |  |  |
| cifar10 | iid | free_rider | main |  | 1 | flirds_gate_v1 | 0.6312 | +0.0281 | 0.0337 | +0.8333 | 1.0000 | 116.0000 | 0.8050 |
| cifar10 | iid | free_rider | main |  | 1 | flirds_gate_v2 | 0.6355 | +0.0324 | 0.0337 | +0.9593 | 0.8500 | 401.0000 | 0.9085 |
| cifar10 | iid | free_rider | main |  | 1 | flirds_gatew_v1 | 0.6304 | +0.0272 | 0.0337 | +0.8074 | 1.0000 | 94.0000 | 0.8360 |
| cifar10 | iid | free_rider | main |  | 1 | flirds_gatew_v2 | 0.6352 | +0.0321 | 0.0337 | +0.9519 | 0.9167 | 168.0000 | 0.9591 |
| cifar10 | iid | free_rider | main |  | 1 | flirds_mult | 0.6202 | +0.0171 | 0.0337 | +0.5074 | 0.3871 |  |  |
| cifar10 | iid | free_rider | main |  | 1 | flirds_zgate_v2 | 0.6058 | +0.0026 | 0.0337 | +0.0778 | 0.9167 |  |  |
| cifar10 | iid | free_rider | main |  | 1 | oracle_excl | 0.6369 | +0.0337 | 0.0337 | +1.0000 |  |  |  |
| cifar10 | iid | free_rider | main |  | 1 | random_excl | 0.6036 | +0.0005 | 0.0337 | +0.0148 |  |  |  |
| cifar10 | iid | free_rider | main |  | 1 | vanilla | 0.6031 | +0.0000 | 0.0337 | +0.0000 |  |  |  |
| cifar10 | iid | free_rider | main |  | 2 | flirds_gate_v1 | 0.6268 | +0.0149 | 0.0235 | +0.6330 | 1.0000 | 126.0000 | 0.7955 |
| cifar10 | iid | free_rider | main |  | 2 | flirds_gate_v2 | 0.6314 | +0.0195 | 0.0235 | +0.8298 | 0.8667 | 352.0000 | 0.9194 |
| cifar10 | iid | free_rider | main |  | 2 | flirds_gatew_v1 | 0.6265 | +0.0146 | 0.0235 | +0.6223 | 1.0000 | 93.0000 | 0.8405 |
| cifar10 | iid | free_rider | main |  | 2 | flirds_gatew_v2 | 0.6325 | +0.0206 | 0.0235 | +0.8777 | 0.8667 | 315.0000 | 0.9271 |
| cifar10 | iid | free_rider | main |  | 2 | flirds_mult | 0.6295 | +0.0176 | 0.0235 | +0.7500 | 0.3579 |  |  |
| cifar10 | iid | free_rider | main |  | 2 | flirds_zgate_v2 | 0.6138 | +0.0019 | 0.0235 | +0.0798 | 0.8333 |  |  |
| cifar10 | iid | free_rider | main |  | 2 | oracle_excl | 0.6354 | +0.0235 | 0.0235 | +1.0000 |  |  |  |
| cifar10 | iid | free_rider | main |  | 2 | random_excl | 0.5886 | -0.0232 | 0.0235 | -0.9894 |  |  |  |
| cifar10 | iid | free_rider | main |  | 2 | vanilla | 0.6119 | +0.0000 | 0.0235 | +0.0000 |  |  |  |
| cifar10 | iid | frrand | main |  | 0 | flirds_gate_v1 | 0.6195 | +0.0086 | 0.0239 | +0.3613 | 0.9833 | 172.0000 | 0.5795 |
| cifar10 | iid | frrand | main |  | 0 | flirds_gate_v2 | 0.6105 | -0.0004 | 0.0239 | -0.0157 | 0.8683 | 326.0000 | 0.8910 |
| cifar10 | iid | frrand | main |  | 0 | flirds_gatew_v1 | 0.6319 | +0.0210 | 0.0239 | +0.8796 | 1.0000 | 95.0000 | 0.6984 |
| cifar10 | iid | frrand | main |  | 0 | flirds_gatew_v2 | 0.6295 | +0.0186 | 0.0239 | +0.7801 | 0.9667 | 18.0000 | 0.9941 |
| cifar10 | iid | frrand | main |  | 0 | flirds_mult | 0.6278 | +0.0169 | 0.0239 | +0.7068 | 0.4629 |  |  |
| cifar10 | iid | frrand | main |  | 0 | flirds_zgate_v2 | 0.6062 | -0.0046 | 0.0239 | -0.1937 | 0.8762 |  |  |
| cifar10 | iid | frrand | main |  | 0 | oracle_excl | 0.6348 | +0.0239 | 0.0239 | +1.0000 |  |  |  |
| cifar10 | iid | frrand | main |  | 0 | random_excl | 0.6038 | -0.0071 | 0.0239 | -0.2984 |  |  |  |
| cifar10 | iid | frrand | main |  | 0 | vanilla | 0.6109 | +0.0000 | 0.0239 | +0.0000 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 0 | flirds_gate_v1 | 0.5836 | +0.3249 | 0.3759 | +0.8643 | 1.0000 | 207.0000 | 0.6929 |
| cifar10 | iid | grad_noise | main |  | 0 | flirds_gate_v2 | 0.6285 | +0.3697 | 0.3759 | +0.9837 | 1.0000 | 821.0000 | 0.8339 |
| cifar10 | iid | grad_noise | main |  | 0 | flirds_gatew_v1 | 0.5836 | +0.3249 | 0.3759 | +0.8643 | 1.0000 | 162.0000 | 0.7437 |
| cifar10 | iid | grad_noise | main |  | 0 | flirds_gatew_v2 | 0.6136 | +0.3549 | 0.3759 | +0.9441 | 1.0000 | 407.0000 | 0.9090 |
| cifar10 | iid | grad_noise | main |  | 0 | flirds_mult | 0.5519 | +0.2931 | 0.3759 | +0.7798 | 1.0000 |  |  |
| cifar10 | iid | grad_noise | main |  | 0 | flirds_zgate_v2 | 0.2970 | +0.0383 | 0.3759 | +0.1018 | 1.0000 |  |  |
| cifar10 | iid | grad_noise | main |  | 0 | oracle_excl | 0.6346 | +0.3759 | 0.3759 | +1.0000 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 0 | random_excl | 0.2655 | +0.0068 | 0.3759 | +0.0180 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 0 | vanilla | 0.2587 | +0.0000 | 0.3759 | +0.0000 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 1 | flirds_gate_v1 | 0.5747 | +0.3230 | 0.3851 | +0.8387 | 1.0000 | 198.0000 | 0.7036 |
| cifar10 | iid | grad_noise | main |  | 1 | flirds_gate_v2 | 0.5901 | +0.3384 | 0.3851 | +0.8786 | 1.0000 | 1161.0000 | 0.7747 |
| cifar10 | iid | grad_noise | main |  | 1 | flirds_gatew_v1 | 0.5775 | +0.3258 | 0.3851 | +0.8458 | 1.0000 | 191.0000 | 0.7110 |
| cifar10 | iid | grad_noise | main |  | 1 | flirds_gatew_v2 | 0.6152 | +0.3635 | 0.3851 | +0.9438 | 1.0000 | 712.0000 | 0.8482 |
| cifar10 | iid | grad_noise | main |  | 1 | flirds_mult | 0.5198 | +0.2680 | 0.3851 | +0.6959 | 1.0000 |  |  |
| cifar10 | iid | grad_noise | main |  | 1 | flirds_zgate_v2 | 0.3362 | +0.0845 | 0.3851 | +0.2194 | 1.0000 |  |  |
| cifar10 | iid | grad_noise | main |  | 1 | oracle_excl | 0.6369 | +0.3851 | 0.3851 | +1.0000 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 1 | random_excl | 0.2744 | +0.0226 | 0.3851 | +0.0587 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 1 | vanilla | 0.2517 | +0.0000 | 0.3851 | +0.0000 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 2 | flirds_gate_v1 | 0.5804 | +0.3217 | 0.3768 | +0.8540 | 1.0000 | 222.0000 | 0.6838 |
| cifar10 | iid | grad_noise | main |  | 2 | flirds_gate_v2 | 0.6244 | +0.3658 | 0.3768 | +0.9708 | 1.0000 | 833.0000 | 0.8288 |
| cifar10 | iid | grad_noise | main |  | 2 | flirds_gatew_v1 | 0.5704 | +0.3117 | 0.3768 | +0.8275 | 1.0000 | 176.0000 | 0.7337 |
| cifar10 | iid | grad_noise | main |  | 2 | flirds_gatew_v2 | 0.6265 | +0.3679 | 0.3768 | +0.9764 | 1.0000 | 542.0000 | 0.8800 |
| cifar10 | iid | grad_noise | main |  | 2 | flirds_mult | 0.5271 | +0.2685 | 0.3768 | +0.7127 | 1.0000 |  |  |
| cifar10 | iid | grad_noise | main |  | 2 | flirds_zgate_v2 | 0.3241 | +0.0655 | 0.3768 | +0.1739 | 1.0000 |  |  |
| cifar10 | iid | grad_noise | main |  | 2 | oracle_excl | 0.6354 | +0.3768 | 0.3768 | +1.0000 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 2 | random_excl | 0.2535 | -0.0051 | 0.3768 | -0.0136 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 2 | vanilla | 0.2586 | +0.0000 | 0.3768 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | flirds_gate_v1 | 0.6076 | -0.0230 | -0.0014 |  | 0.6931 | 187.0000 | 0.4806 |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | flirds_gate_v1 | 0.6248 | +0.0299 | 0.0344 | +0.8691 | 0.9639 | 126.0000 | 0.7175 |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | flirds_gate_v1 | 0.5713 | +0.0333 | 0.0912 | +0.3644 | 1.0000 | 96.0000 | 0.8033 |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | flirds_gate_v2 | 0.6245 | -0.0061 | -0.0014 |  | 0.5750 | 301.0000 | 0.6707 |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | flirds_gate_v2 | 0.6105 | +0.0156 | 0.0344 | +0.4545 | 0.7650 | 187.0000 | 0.9408 |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | flirds_gate_v2 | 0.6032 | +0.0652 | 0.0912 | +0.7151 | 0.8394 | 210.0000 | 0.9438 |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | flirds_gatew_v1 | 0.6192 | -0.0114 | -0.0014 |  | 0.6301 | 165.0000 | 0.4572 |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | flirds_gatew_v1 | 0.6196 | +0.0247 | 0.0344 | +0.7200 | 0.9840 | 100.0000 | 0.7549 |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | flirds_gatew_v1 | 0.6054 | +0.0674 | 0.0912 | +0.7384 | 1.0000 | 81.0000 | 0.8298 |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | flirds_gatew_v2 | 0.6181 | -0.0125 | -0.0014 |  | 0.3762 | 227.0000 | 0.6133 |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | flirds_gatew_v2 | 0.6079 | +0.0130 | 0.0344 | +0.3782 | 0.6053 | 278.0000 | 0.8864 |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | flirds_gatew_v2 | 0.6091 | +0.0711 | 0.0912 | +0.7795 | 0.8541 | 420.0000 | 0.8951 |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | flirds_mult | 0.6256 | -0.0050 | -0.0014 |  | 0.3859 |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | flirds_mult | 0.6191 | +0.0242 | 0.0344 | +0.7055 | 0.7516 |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | flirds_mult | 0.6142 | +0.0762 | 0.0912 | +0.8356 | 0.9756 |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | flirds_zgate_v2 | 0.6216 | -0.0090 | -0.0014 |  | 0.6478 |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | flirds_zgate_v2 | 0.6049 | +0.0100 | 0.0344 | +0.2909 | 0.9954 |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | flirds_zgate_v2 | 0.5911 | +0.0531 | 0.0912 | +0.5822 | 1.0000 |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | oracle_excl | 0.6292 | -0.0014 | -0.0014 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | oracle_excl | 0.6292 | +0.0344 | 0.0344 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | oracle_excl | 0.6292 | +0.0912 | 0.0912 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | random_excl | 0.6126 | -0.0180 | -0.0014 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | random_excl | 0.5962 | +0.0014 | 0.0344 | +0.0400 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | random_excl | 0.5610 | +0.0230 | 0.0912 | +0.2521 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | vanilla | 0.6306 | +0.0000 | -0.0014 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | vanilla | 0.5949 | +0.0000 | 0.0344 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | vanilla | 0.5380 | +0.0000 | 0.0912 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | flirds_gate_v1 | 0.6284 | -0.0020 | 0.0050 |  | 0.6274 | 163.0000 | 0.5688 |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | flirds_gate_v1 | 0.6071 | +0.0155 | 0.0438 | +0.3543 | 0.9984 | 92.0000 | 0.8083 |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | flirds_gate_v1 | 0.5773 | +0.0739 | 0.1320 | +0.5597 | 1.0000 | 87.0000 | 0.8466 |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | flirds_gate_v2 | 0.6225 | -0.0079 | 0.0050 |  | 0.6466 | 144.0000 | 0.9146 |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | flirds_gate_v2 | 0.6099 | +0.0182 | 0.0438 | +0.4171 | 0.8205 | 404.0000 | 0.9044 |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | flirds_gate_v2 | 0.5986 | +0.0952 | 0.1320 | +0.7216 | 0.8986 | 576.0000 | 0.8916 |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | flirds_gatew_v1 | 0.6270 | -0.0034 | 0.0050 |  | 0.6647 | 128.0000 | 0.5897 |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | flirds_gatew_v1 | 0.6034 | +0.0118 | 0.0438 | +0.2686 | 0.9972 | 75.0000 | 0.8315 |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | flirds_gatew_v1 | 0.5950 | +0.0916 | 0.1320 | +0.6941 | 1.0000 | 79.0000 | 0.8579 |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | flirds_gatew_v2 | 0.6120 | -0.0184 | 0.0050 |  | 0.5008 | 223.0000 | 0.8079 |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | flirds_gatew_v2 | 0.6079 | +0.0163 | 0.0438 | +0.3714 | 0.7107 | 326.0000 | 0.9136 |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | flirds_gatew_v2 | 0.5965 | +0.0931 | 0.1320 | +0.7055 | 0.8774 | 427.0000 | 0.9139 |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | flirds_mult | 0.6302 | -0.0001 | 0.0050 |  | 0.4002 |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | flirds_mult | 0.6162 | +0.0246 | 0.0438 | +0.5629 | 0.8393 |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | flirds_mult | 0.6020 | +0.0986 | 0.1320 | +0.7472 | 0.9872 |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | flirds_zgate_v2 | 0.6366 | +0.0062 | 0.0050 |  | 0.7304 |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | flirds_zgate_v2 | 0.5925 | +0.0009 | 0.0438 | +0.0200 | 1.0000 |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | flirds_zgate_v2 | 0.5209 | +0.0175 | 0.1320 | +0.1326 | 1.0000 |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | oracle_excl | 0.6354 | +0.0050 | 0.0050 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | oracle_excl | 0.6354 | +0.0438 | 0.0438 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | oracle_excl | 0.6354 | +0.1320 | 0.1320 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | random_excl | 0.6032 | -0.0271 | 0.0050 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | random_excl | 0.5699 | -0.0217 | 0.0438 | -0.4971 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | random_excl | 0.4898 | -0.0136 | 0.1320 | -0.1032 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | vanilla | 0.6304 | +0.0000 | 0.0050 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | vanilla | 0.5916 | +0.0000 | 0.0438 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | vanilla | 0.5034 | +0.0000 | 0.1320 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | flirds_gate_v1 | 0.6126 | -0.0094 | 0.0062 |  | 0.7134 | 160.0000 | 0.5897 |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | flirds_gate_v1 | 0.6174 | +0.0271 | 0.0380 | +0.7138 | 0.9940 | 96.0000 | 0.7975 |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | flirds_gate_v1 | 0.6159 | +0.1059 | 0.1182 | +0.8953 | 1.0000 | 88.0000 | 0.8442 |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | flirds_gate_v2 | 0.6214 | -0.0006 | 0.0062 |  | 0.6499 | 282.0000 | 0.8239 |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | flirds_gate_v2 | 0.5870 | -0.0033 | 0.0380 | -0.0855 | 0.8475 | 374.0000 | 0.9122 |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | flirds_gate_v2 | 0.5883 | +0.0783 | 0.1182 | +0.6617 | 0.8900 | 550.0000 | 0.8884 |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | flirds_gatew_v1 | 0.6141 | -0.0079 | 0.0062 |  | 0.7102 | 123.0000 | 0.6083 |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | flirds_gatew_v1 | 0.6105 | +0.0202 | 0.0380 | +0.5329 | 0.9972 | 77.0000 | 0.8274 |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | flirds_gatew_v1 | 0.6185 | +0.1085 | 0.1182 | +0.9175 | 1.0000 | 69.0000 | 0.8752 |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | flirds_gatew_v2 | 0.6109 | -0.0111 | 0.0062 |  | 0.4898 | 258.0000 | 0.7970 |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | flirds_gatew_v2 | 0.5817 | -0.0085 | 0.0380 | -0.2237 | 0.6877 | 503.0000 | 0.8777 |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | flirds_gatew_v2 | 0.6039 | +0.0939 | 0.1182 | +0.7939 | 0.8703 | 405.0000 | 0.9160 |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | flirds_mult | 0.6170 | -0.0050 | 0.0062 |  | 0.3693 |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | flirds_mult | 0.6130 | +0.0227 | 0.0380 | +0.5987 | 0.9077 |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | flirds_mult | 0.6066 | +0.0966 | 0.1182 | +0.8171 | 0.9916 |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | flirds_zgate_v2 | 0.6175 | -0.0045 | 0.0062 |  | 0.6989 |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | flirds_zgate_v2 | 0.5959 | +0.0056 | 0.0380 | +0.1480 | 1.0000 |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | flirds_zgate_v2 | 0.5126 | +0.0026 | 0.1182 | +0.0222 | 1.0000 |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | oracle_excl | 0.6282 | +0.0062 | 0.0062 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | oracle_excl | 0.6282 | +0.0380 | 0.0380 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | oracle_excl | 0.6282 | +0.1182 | 0.1182 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | random_excl | 0.5894 | -0.0326 | 0.0062 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | random_excl | 0.5507 | -0.0395 | 0.0380 | -1.0395 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | random_excl | 0.4561 | -0.0539 | 0.1182 | -0.4556 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | vanilla | 0.6220 | +0.0000 | 0.0062 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | vanilla | 0.5903 | +0.0000 | 0.0380 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | vanilla | 0.5100 | +0.0000 | 0.1182 | +0.0000 |  |  |  |
| cifar10 | qskew | clean | main |  | 0 | flirds_gate_v1 | 0.6614 | -0.0076 |  |  |  | 358.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 0 | flirds_gate_v2 | 0.6591 | -0.0099 |  |  |  | 1703.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 0 | flirds_gatew_v1 | 0.6570 | -0.0120 |  |  |  | 276.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 0 | flirds_gatew_v2 | 0.6561 | -0.0129 |  |  |  | 1254.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 0 | flirds_mult | 0.6637 | -0.0053 |  |  |  |  |  |
| cifar10 | qskew | clean | main |  | 0 | flirds_zgate_v2 | 0.6666 | -0.0024 |  |  |  |  |  |
| cifar10 | qskew | clean | main |  | 0 | vanilla | 0.6690 | +0.0000 |  |  |  |  |  |
| cifar10 | qskew | free_rider | main |  | 0 | flirds_gate_v1 | 0.6472 | +0.0241 | 0.0201 | +1.1988 | 0.9667 | 125.0000 | 0.7934 |
| cifar10 | qskew | free_rider | main |  | 0 | flirds_gate_v2 | 0.6316 | +0.0085 | 0.0201 | +0.4224 | 0.6833 | 1306.0000 | 0.7590 |
| cifar10 | qskew | free_rider | main |  | 0 | flirds_gatew_v1 | 0.6406 | +0.0175 | 0.0201 | +0.8696 | 0.9833 | 95.0000 | 0.8348 |
| cifar10 | qskew | free_rider | main |  | 0 | flirds_gatew_v2 | 0.6312 | +0.0081 | 0.0201 | +0.4037 | 0.6500 | 1123.0000 | 0.7855 |
| cifar10 | qskew | free_rider | main |  | 0 | flirds_mult | 0.6366 | +0.0135 | 0.0201 | +0.6708 | 0.4325 |  |  |
| cifar10 | qskew | free_rider | main |  | 0 | flirds_zgate_v2 | 0.6202 | -0.0029 | 0.0201 | -0.1429 | 0.8333 |  |  |
| cifar10 | qskew | free_rider | main |  | 0 | oracle_excl | 0.6432 | +0.0201 | 0.0201 | +1.0000 |  |  |  |
| cifar10 | qskew | free_rider | main |  | 0 | random_excl | 0.6128 | -0.0104 | 0.0201 | -0.5155 |  |  |  |
| cifar10 | qskew | free_rider | main |  | 0 | vanilla | 0.6231 | +0.0000 | 0.0201 | +0.0000 |  |  |  |
| cifar10 | qskew | frrand | main |  | 0 | flirds_gate_v1 | 0.6284 | +0.0045 | 0.0194 |  | 0.8667 | 205.0000 | 0.5341 |
| cifar10 | qskew | frrand | main |  | 0 | flirds_gate_v2 | 0.6240 | +0.0001 | 0.0194 |  | 0.8071 | 828.0000 | 0.7943 |
| cifar10 | qskew | frrand | main |  | 0 | flirds_gatew_v1 | 0.6309 | +0.0070 | 0.0194 |  | 0.9667 | 105.0000 | 0.6866 |
| cifar10 | qskew | frrand | main |  | 0 | flirds_gatew_v2 | 0.6341 | +0.0103 | 0.0194 |  | 0.7812 | 622.0000 | 0.8248 |
| cifar10 | qskew | frrand | main |  | 0 | flirds_mult | 0.6358 | +0.0119 | 0.0194 |  | 0.4383 |  |  |
| cifar10 | qskew | frrand | main |  | 0 | flirds_zgate_v2 | 0.6215 | -0.0024 | 0.0194 |  | 0.8358 |  |  |
| cifar10 | qskew | frrand | main |  | 0 | oracle_excl | 0.6432 | +0.0194 | 0.0194 |  |  |  |  |
| cifar10 | qskew | frrand | main |  | 0 | random_excl | 0.6118 | -0.0121 | 0.0194 |  |  |  |  |
| cifar10 | qskew | frrand | main |  | 0 | vanilla | 0.6239 | +0.0000 | 0.0194 |  |  |  |  |
| cifar10 | qskew | grad_noise | main |  | 0 | flirds_gate_v1 | 0.5441 | +0.2916 | 0.3907 | +0.7463 | 0.9958 | 197.0000 | 0.6965 |
| cifar10 | qskew | grad_noise | main |  | 0 | flirds_gate_v2 | 0.6366 | +0.3841 | 0.3907 | +0.9830 | 0.9879 | 1273.0000 | 0.7608 |
| cifar10 | qskew | grad_noise | main |  | 0 | flirds_gatew_v1 | 0.2039 | -0.0486 | 0.3907 | -0.1244 | 0.9067 | 278.0000 | 0.6166 |
| cifar10 | qskew | grad_noise | main |  | 0 | flirds_gatew_v2 | 0.6304 | +0.3779 | 0.3907 | +0.9671 | 0.9871 | 888.0000 | 0.8190 |
| cifar10 | qskew | grad_noise | main |  | 0 | flirds_mult | 0.4871 | +0.2346 | 0.3907 | +0.6004 | 0.9446 |  |  |
| cifar10 | qskew | grad_noise | main |  | 0 | flirds_zgate_v2 | 0.4339 | +0.1814 | 0.3907 | +0.4642 | 0.9833 |  |  |
| cifar10 | qskew | grad_noise | main |  | 0 | oracle_excl | 0.6432 | +0.3907 | 0.3907 | +1.0000 |  |  |  |
| cifar10 | qskew | grad_noise | main |  | 0 | random_excl | 0.2556 | +0.0031 | 0.3907 | +0.0080 |  |  |  |
| cifar10 | qskew | grad_noise | main |  | 0 | vanilla | 0.2525 | +0.0000 | 0.3907 | +0.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | flirds_gate_v1 | 0.6325 | -0.0081 | 0.0082 |  | 0.5523 | 205.0000 | 0.4225 |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | flirds_gate_v1 | 0.6176 | +0.0106 | 0.0419 | +0.2537 | 0.8108 | 152.0000 | 0.6415 |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | flirds_gate_v1 | 0.5827 | +0.0241 | 0.0902 | +0.2673 | 0.9706 | 126.0000 | 0.7353 |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | flirds_gate_v2 | 0.6298 | -0.0109 | 0.0082 |  | 0.4687 | 941.0000 | 0.5300 |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | flirds_gate_v2 | 0.6101 | +0.0031 | 0.0419 | +0.0746 | 0.7175 | 715.0000 | 0.7950 |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | flirds_gate_v2 | 0.5969 | +0.0383 | 0.0902 | +0.4238 | 0.7945 | 1150.0000 | 0.7431 |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | flirds_gatew_v1 | 0.6404 | -0.0002 | 0.0082 |  | 0.5322 | 157.0000 | 0.3938 |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | flirds_gatew_v1 | 0.6196 | +0.0126 | 0.0419 | +0.3015 | 0.8087 | 115.0000 | 0.6637 |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | flirds_gatew_v1 | 0.5374 | -0.0212 | 0.0902 | -0.2355 | 0.9739 | 111.0000 | 0.7528 |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | flirds_gatew_v2 | 0.6391 | -0.0015 | 0.0082 |  | 0.3775 | 580.0000 | 0.4821 |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | flirds_gatew_v2 | 0.6171 | +0.0101 | 0.0419 | +0.2418 | 0.5221 | 977.0000 | 0.6503 |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | flirds_gatew_v2 | 0.5823 | +0.0236 | 0.0902 | +0.2618 | 0.6675 | 940.0000 | 0.7573 |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | flirds_mult | 0.6374 | -0.0032 | 0.0082 |  | 0.2459 |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | flirds_mult | 0.6248 | +0.0178 | 0.0419 | +0.4239 | 0.5628 |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | flirds_mult | 0.6252 | +0.0666 | 0.0902 | +0.7382 | 0.7907 |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | flirds_zgate_v2 | 0.6430 | +0.0024 | 0.0082 |  | 0.4212 |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | flirds_zgate_v2 | 0.6251 | +0.0181 | 0.0419 | +0.4328 | 0.9252 |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | flirds_zgate_v2 | 0.5995 | +0.0409 | 0.0902 | +0.4529 | 0.9878 |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | oracle_excl | 0.6489 | +0.0082 | 0.0082 |  |  |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | oracle_excl | 0.6489 | +0.0419 | 0.0419 | +1.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | oracle_excl | 0.6489 | +0.0902 | 0.0902 | +1.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | random_excl | 0.6378 | -0.0029 | 0.0082 |  |  |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | random_excl | 0.6191 | +0.0121 | 0.0419 | +0.2896 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | random_excl | 0.5891 | +0.0305 | 0.0902 | +0.3380 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | vanilla | 0.6406 | +0.0000 | 0.0082 |  |  |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | vanilla | 0.6070 | +0.0000 | 0.0419 | +0.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | vanilla | 0.5586 | +0.0000 | 0.0902 | +0.0000 |  |  |  |
| cifar10 | shard | clean | main |  | 0 | flirds_gate_v1 | 0.4556 | -0.0572 |  |  |  | 455.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 0 | flirds_gate_v2 | 0.4491 | -0.0637 |  |  |  | 3747.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 0 | flirds_gatew_v1 | 0.3905 | -0.1224 |  |  |  | 368.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 0 | flirds_gatew_v2 | 0.4009 | -0.1120 |  |  |  | 3207.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 0 | flirds_mult | 0.5271 | +0.0142 |  |  |  |  |  |
| cifar10 | shard | clean | main |  | 0 | flirds_zgate_v2 | 0.5042 | -0.0086 |  |  |  |  |  |
| cifar10 | shard | clean | main |  | 0 | vanilla | 0.5129 | +0.0000 |  |  |  |  |  |
| cifar10 | shard | clean | main |  | 1 | flirds_gate_v1 | 0.4629 | +0.0260 |  |  |  | 442.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 1 | flirds_gate_v2 | 0.4674 | +0.0305 |  |  |  | 2984.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 1 | flirds_gatew_v1 | 0.1930 | -0.2439 |  |  |  | 393.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 1 | flirds_gatew_v2 | 0.4959 | +0.0590 |  |  |  | 3678.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 1 | flirds_mult | 0.4547 | +0.0179 |  |  |  |  |  |
| cifar10 | shard | clean | main |  | 1 | flirds_zgate_v2 | 0.4180 | -0.0189 |  |  |  |  |  |
| cifar10 | shard | clean | main |  | 1 | vanilla | 0.4369 | +0.0000 |  |  |  |  |  |
| cifar10 | shard | free_rider | main |  | 0 | flirds_gate_v1 | 0.2018 | -0.1751 | 0.0805 | -2.1755 | 0.9833 | 208.0000 | 0.6977 |
| cifar10 | shard | free_rider | main |  | 0 | flirds_gate_v2 | 0.3644 | -0.0125 | 0.0805 | -0.1553 | 0.6667 | 1870.0000 | 0.6896 |
| cifar10 | shard | free_rider | main |  | 0 | flirds_gatew_v1 | 0.2054 | -0.1715 | 0.0805 | -2.1304 | 1.0000 | 179.0000 | 0.7284 |
| cifar10 | shard | free_rider | main |  | 0 | flirds_gatew_v2 | 0.4675 | +0.0906 | 0.0805 | +1.1258 | 0.6333 | 1821.0000 | 0.6937 |
| cifar10 | shard | free_rider | main |  | 0 | flirds_mult | 0.4120 | +0.0351 | 0.0805 | +0.4363 | 0.3150 |  |  |
| cifar10 | shard | free_rider | main |  | 0 | flirds_zgate_v2 | 0.4126 | +0.0358 | 0.0805 | +0.4441 | 0.4167 |  |  |
| cifar10 | shard | free_rider | main |  | 0 | oracle_excl | 0.4574 | +0.0805 | 0.0805 | +1.0000 |  |  |  |
| cifar10 | shard | free_rider | main |  | 0 | random_excl | 0.4059 | +0.0290 | 0.0805 | +0.3602 |  |  |  |
| cifar10 | shard | free_rider | main |  | 0 | vanilla | 0.3769 | +0.0000 | 0.0805 | +0.0000 |  |  |  |
| cifar10 | shard | frrand | main |  | 0 | flirds_gate_v1 | 0.4706 | +0.0939 | 0.0806 | +1.1643 | 0.8187 | 308.0000 | 0.4390 |
| cifar10 | shard | frrand | main |  | 0 | flirds_gate_v2 | 0.4057 | +0.0290 | 0.0806 | +0.3597 | 0.6833 | 1698.0000 | 0.6622 |
| cifar10 | shard | frrand | main |  | 0 | flirds_gatew_v1 | 0.2076 | -0.1691 | 0.0806 | -2.0977 | 1.0000 | 180.0000 | 0.5714 |
| cifar10 | shard | frrand | main |  | 0 | flirds_gatew_v2 | 0.4461 | +0.0694 | 0.0806 | +0.8605 | 0.8833 | 751.0000 | 0.7662 |
| cifar10 | shard | frrand | main |  | 0 | flirds_mult | 0.4118 | +0.0350 | 0.0806 | +0.4341 | 0.3150 |  |  |
| cifar10 | shard | frrand | main |  | 0 | flirds_zgate_v2 | 0.4674 | +0.0906 | 0.0806 | +1.1240 | 0.4667 |  |  |
| cifar10 | shard | frrand | main |  | 0 | oracle_excl | 0.4574 | +0.0806 | 0.0806 | +1.0000 |  |  |  |
| cifar10 | shard | frrand | main |  | 0 | random_excl | 0.4084 | +0.0316 | 0.0806 | +0.3922 |  |  |  |
| cifar10 | shard | frrand | main |  | 0 | vanilla | 0.3767 | +0.0000 | 0.0806 | +0.0000 |  |  |  |
| cifar10 | shard | grad_noise | main |  | 0 | flirds_gate_v1 | 0.1174 | -0.0535 | 0.2865 | -0.1867 | 0.8383 | 277.0000 | 0.5554 |
| cifar10 | shard | grad_noise | main |  | 0 | flirds_gate_v2 | 0.4079 | +0.2370 | 0.2865 | +0.8272 | 0.8408 | 3582.0000 | 0.5289 |
| cifar10 | shard | grad_noise | main |  | 0 | flirds_gatew_v1 | 0.1430 | -0.0279 | 0.2865 | -0.0973 | 0.8108 | 229.0000 | 0.5655 |
| cifar10 | shard | grad_noise | main |  | 0 | flirds_gatew_v2 | 0.4062 | +0.2354 | 0.2865 | +0.8216 | 0.9004 | 2152.0000 | 0.6468 |
| cifar10 | shard | grad_noise | main |  | 0 | flirds_mult | 0.2829 | +0.1120 | 0.2865 | +0.3909 | 0.9908 |  |  |
| cifar10 | shard | grad_noise | main |  | 0 | flirds_zgate_v2 | 0.1659 | -0.0050 | 0.2865 | -0.0175 | 1.0000 |  |  |
| cifar10 | shard | grad_noise | main |  | 0 | oracle_excl | 0.4574 | +0.2865 | 0.2865 | +1.0000 |  |  |  |
| cifar10 | shard | grad_noise | main |  | 0 | random_excl | 0.1809 | +0.0100 | 0.2865 | +0.0349 |  |  |  |
| cifar10 | shard | grad_noise | main |  | 0 | vanilla | 0.1709 | +0.0000 | 0.2865 | +0.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | flirds_gate_v1 | 0.4517 | -0.0271 | -0.0217 | +1.2471 | 0.3178 | 285.0000 | 0.3357 |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | flirds_gate_v1 | 0.3029 | -0.1344 | 0.0199 |  | 0.3443 | 293.0000 | 0.3122 |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | flirds_gate_v1 | 0.2416 | -0.1247 | 0.0907 | -1.3747 | 0.3728 | 294.0000 | 0.2864 |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | flirds_gate_v2 | 0.2299 | -0.2490 | -0.0217 | +11.4483 | 0.3573 | 2417.0000 | 0.2160 |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | flirds_gate_v2 | 0.1732 | -0.2640 | 0.0199 |  | 0.2102 | 3187.0000 | 0.1809 |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | flirds_gate_v2 | 0.3540 | -0.0124 | 0.0907 | -0.1364 | 0.2699 | 3318.0000 | 0.0758 |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | flirds_gatew_v1 | 0.3101 | -0.1688 | -0.0217 | +7.7586 | 0.3279 | 244.0000 | 0.2989 |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | flirds_gatew_v1 | 0.2451 | -0.1921 | 0.0199 |  | 0.3804 | 246.0000 | 0.2545 |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | flirds_gatew_v1 | 0.2160 | -0.1504 | 0.0907 | -1.6570 | 0.3918 | 245.0000 | 0.2462 |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | flirds_gatew_v2 | 0.4494 | -0.0295 | -0.0217 | +1.3563 | 0.3274 | 2748.0000 | 0.3031 |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | flirds_gatew_v2 | 0.3871 | -0.0501 | 0.0199 |  | 0.2930 | 2610.0000 | 0.0419 |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | flirds_gatew_v2 | 0.1745 | -0.1919 | 0.0907 | -2.1143 | 0.2135 | 2988.0000 | 0.0565 |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | flirds_mult | 0.5064 | +0.0275 | -0.0217 | -1.2644 | 0.4590 |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | flirds_mult | 0.4575 | +0.0202 | 0.0199 |  | 0.5158 |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | flirds_mult | 0.3729 | +0.0065 | 0.0907 | +0.0716 | 0.5439 |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | flirds_zgate_v2 | 0.4699 | -0.0090 | -0.0217 | +0.4138 | 0.3127 |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | flirds_zgate_v2 | 0.4487 | +0.0115 | 0.0199 |  | 0.3607 |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | flirds_zgate_v2 | 0.2789 | -0.0875 | 0.0907 | -0.9642 | 0.3888 |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | oracle_excl | 0.4571 | -0.0217 | -0.0217 | +1.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | oracle_excl | 0.4571 | +0.0199 | 0.0199 |  |  |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | oracle_excl | 0.4571 | +0.0907 | 0.0907 | +1.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | random_excl | 0.3762 | -0.1026 | -0.0217 | +4.7184 |  |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | random_excl | 0.3469 | -0.0904 | 0.0199 |  |  |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | random_excl | 0.3086 | -0.0578 | 0.0907 | -0.6364 |  |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | vanilla | 0.4789 | +0.0000 | -0.0217 | -0.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | vanilla | 0.4373 | +0.0000 | 0.0199 |  |  |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | vanilla | 0.3664 | +0.0000 | 0.0907 | +0.0000 |  |  |  |
| fmnist | dir1 | clean | main |  | 0 | flirds_gate_v1 | 0.8459 | +0.0519 |  |  |  | 456.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 0 | flirds_gate_v2 | 0.8275 | +0.0335 |  |  |  | 2746.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 0 | flirds_gatew_v1 | 0.8475 | +0.0535 |  |  |  | 323.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 0 | flirds_gatew_v2 | 0.8435 | +0.0495 |  |  |  | 1860.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 0 | flirds_mult | 0.8154 | +0.0214 |  |  |  |  |  |
| fmnist | dir1 | clean | main |  | 0 | flirds_zgate_v2 | 0.8455 | +0.0515 |  |  |  |  |  |
| fmnist | dir1 | clean | main |  | 0 | vanilla | 0.7940 | +0.0000 |  |  |  |  |  |
| fmnist | dir1 | free_rider | main |  | 0 | flirds_gate_v1 | 0.7819 | -0.0119 | 0.0440 | -0.2699 | 0.9333 | 196.0000 | 0.7101 |
| fmnist | dir1 | free_rider | main |  | 0 | flirds_gate_v2 | 0.8371 | +0.0434 | 0.0440 | +0.9858 | 0.7000 | 1544.0000 | 0.7298 |
| fmnist | dir1 | free_rider | main |  | 0 | flirds_gatew_v1 | 0.7943 | +0.0005 | 0.0440 | +0.0114 | 0.9833 | 142.0000 | 0.7717 |
| fmnist | dir1 | free_rider | main |  | 0 | flirds_gatew_v2 | 0.8204 | +0.0266 | 0.0440 | +0.6051 | 0.7333 | 1244.0000 | 0.7673 |
| fmnist | dir1 | free_rider | main |  | 0 | flirds_mult | 0.8085 | +0.0148 | 0.0440 | +0.3352 | 0.3946 |  |  |
| fmnist | dir1 | free_rider | main |  | 0 | flirds_zgate_v2 | 0.8021 | +0.0084 | 0.0440 | +0.1903 | 0.7000 |  |  |
| fmnist | dir1 | free_rider | main |  | 0 | oracle_excl | 0.8377 | +0.0440 | 0.0440 | +1.0000 |  |  |  |
| fmnist | dir1 | free_rider | main |  | 0 | random_excl | 0.8095 | +0.0158 | 0.0440 | +0.3580 |  |  |  |
| fmnist | dir1 | free_rider | main |  | 0 | vanilla | 0.7937 | +0.0000 | 0.0440 | +0.0000 |  |  |  |
| fmnist | dir1 | frrand | main |  | 0 | flirds_gate_v1 | 0.7934 | -0.0027 | 0.0416 | -0.0661 | 0.8350 | 295.0000 | 0.4517 |
| fmnist | dir1 | frrand | main |  | 0 | flirds_gate_v2 | 0.7989 | +0.0028 | 0.0416 | +0.0661 | 0.6033 | 2039.0000 | 0.5764 |
| fmnist | dir1 | frrand | main |  | 0 | flirds_gatew_v1 | 0.7991 | +0.0030 | 0.0416 | +0.0721 | 0.9833 | 150.0000 | 0.6042 |
| fmnist | dir1 | frrand | main |  | 0 | flirds_gatew_v2 | 0.8381 | +0.0420 | 0.0416 | +1.0090 | 0.7558 | 956.0000 | 0.7296 |
| fmnist | dir1 | frrand | main |  | 0 | flirds_mult | 0.8071 | +0.0110 | 0.0416 | +0.2643 | 0.3904 |  |  |
| fmnist | dir1 | frrand | main |  | 0 | flirds_zgate_v2 | 0.8009 | +0.0048 | 0.0416 | +0.1141 | 0.7000 |  |  |
| fmnist | dir1 | frrand | main |  | 0 | oracle_excl | 0.8377 | +0.0416 | 0.0416 | +1.0000 |  |  |  |
| fmnist | dir1 | frrand | main |  | 0 | random_excl | 0.8087 | +0.0126 | 0.0416 | +0.3033 |  |  |  |
| fmnist | dir1 | frrand | main |  | 0 | vanilla | 0.7961 | +0.0000 | 0.0416 | +0.0000 |  |  |  |
| fmnist | dir1 | grad_noise | main |  | 0 | flirds_gate_v1 | 0.7969 | +0.0486 | 0.0895 | +0.5433 | 0.9871 | 226.0000 | 0.6367 |
| fmnist | dir1 | grad_noise | main |  | 0 | flirds_gate_v2 | 0.8592 | +0.1110 | 0.0895 | +1.2402 | 0.9512 | 1325.0000 | 0.7496 |
| fmnist | dir1 | grad_noise | main |  | 0 | flirds_gatew_v1 | 0.7691 | +0.0209 | 0.0895 | +0.2332 | 0.9888 | 151.0000 | 0.7140 |
| fmnist | dir1 | grad_noise | main |  | 0 | flirds_gatew_v2 | 0.8578 | +0.1095 | 0.0895 | +1.2235 | 0.9808 | 859.0000 | 0.8091 |
| fmnist | dir1 | grad_noise | main |  | 0 | flirds_mult | 0.8066 | +0.0584 | 0.0895 | +0.6522 | 0.9896 |  |  |
| fmnist | dir1 | grad_noise | main |  | 0 | flirds_zgate_v2 | 0.8079 | +0.0596 | 0.0895 | +0.6662 | 0.9996 |  |  |
| fmnist | dir1 | grad_noise | main |  | 0 | oracle_excl | 0.8377 | +0.0895 | 0.0895 | +1.0000 |  |  |  |
| fmnist | dir1 | grad_noise | main |  | 0 | random_excl | 0.7705 | +0.0222 | 0.0895 | +0.2486 |  |  |  |
| fmnist | dir1 | grad_noise | main |  | 0 | vanilla | 0.7482 | +0.0000 | 0.0895 | +0.0000 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | flirds_gate_v1 | 0.6817 | -0.1401 | -0.0015 |  | 0.9197 | 185.0000 | 0.6358 |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | flirds_gate_v1 | 0.8014 | -0.0108 | 0.0082 |  | 0.9954 | 174.0000 | 0.6871 |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | flirds_gate_v1 | 0.7571 | -0.0389 | 0.0244 | -1.5949 | 1.0000 | 195.0000 | 0.6761 |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | flirds_gate_v2 | 0.8423 | +0.0204 | -0.0015 |  | 0.8567 | 892.0000 | 0.7163 |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | flirds_gate_v2 | 0.8155 | +0.0034 | 0.0082 |  | 0.8878 | 1215.0000 | 0.7357 |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | flirds_gate_v2 | 0.8335 | +0.0375 | 0.0244 | +1.5385 | 0.9630 | 1247.0000 | 0.7507 |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | flirds_gatew_v1 | 0.8346 | +0.0127 | -0.0015 |  | 0.9096 | 142.0000 | 0.6459 |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | flirds_gatew_v1 | 0.8156 | +0.0035 | 0.0082 |  | 0.9933 | 143.0000 | 0.7111 |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | flirds_gatew_v1 | 0.7901 | -0.0059 | 0.0244 | -0.2410 | 1.0000 | 195.0000 | 0.6814 |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | flirds_gatew_v2 | 0.8360 | +0.0141 | -0.0015 |  | 0.8029 | 1332.0000 | 0.6282 |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | flirds_gatew_v2 | 0.8316 | +0.0195 | 0.0082 |  | 0.8680 | 1576.0000 | 0.6749 |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | flirds_gatew_v2 | 0.8383 | +0.0423 | 0.0244 | +1.7333 | 0.9668 | 1307.0000 | 0.7435 |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | flirds_mult | 0.8239 | +0.0020 | -0.0015 |  | 0.8684 |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | flirds_mult | 0.8043 | -0.0079 | 0.0082 |  | 0.9777 |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | flirds_mult | 0.7770 | -0.0190 | 0.0244 | -0.7795 | 0.9996 |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | flirds_zgate_v2 | 0.8364 | +0.0145 | -0.0015 |  | 0.9252 |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | flirds_zgate_v2 | 0.8207 | +0.0086 | 0.0082 |  | 1.0000 |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | flirds_zgate_v2 | 0.8197 | +0.0237 | 0.0244 | +0.9744 | 1.0000 |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | oracle_excl | 0.8204 | -0.0015 | -0.0015 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | oracle_excl | 0.8204 | +0.0082 | 0.0082 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | oracle_excl | 0.8204 | +0.0244 | 0.0244 | +1.0000 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | random_excl | 0.7943 | -0.0276 | -0.0015 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | random_excl | 0.7849 | -0.0272 | 0.0082 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | random_excl | 0.7695 | -0.0265 | 0.0244 | -1.0872 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | vanilla | 0.8219 | +0.0000 | -0.0015 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | vanilla | 0.8121 | +0.0000 | 0.0082 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | vanilla | 0.7960 | +0.0000 | 0.0244 | +0.0000 |  |  |  |
| fmnist | iid | clean | main |  | 0 | flirds_gate_v1 | 0.8518 | +0.0001 |  |  |  | 397.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 0 | flirds_gate_v2 | 0.8520 | +0.0004 |  |  |  | 828.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 0 | flirds_gatew_v1 | 0.8492 | -0.0024 |  |  |  | 298.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 0 | flirds_gatew_v2 | 0.8508 | -0.0009 |  |  |  | 648.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 0 | flirds_mult | 0.8529 | +0.0013 |  |  |  |  |  |
| fmnist | iid | clean | main |  | 0 | flirds_zgate_v2 | 0.8535 | +0.0019 |  |  |  |  |  |
| fmnist | iid | clean | main |  | 0 | vanilla | 0.8516 | +0.0000 |  |  |  |  |  |
| fmnist | iid | free_rider | main |  | 0 | flirds_gate_v1 | 0.8501 | +0.0288 | 0.0290 | +0.9914 | 1.0000 | 140.0000 | 0.7742 |
| fmnist | iid | free_rider | main |  | 0 | flirds_gate_v2 | 0.8506 | +0.0292 | 0.0290 | +1.0086 | 0.9000 | 515.0000 | 0.8884 |
| fmnist | iid | free_rider | main |  | 0 | flirds_gatew_v1 | 0.8474 | +0.0260 | 0.0290 | +0.8966 | 0.9833 | 114.0000 | 0.8081 |
| fmnist | iid | free_rider | main |  | 0 | flirds_gatew_v2 | 0.8504 | +0.0290 | 0.0290 | +1.0000 | 0.8833 | 463.0000 | 0.8994 |
| fmnist | iid | free_rider | main |  | 0 | flirds_mult | 0.8375 | +0.0161 | 0.0290 | +0.5560 | 0.5092 |  |  |
| fmnist | iid | free_rider | main |  | 0 | flirds_zgate_v2 | 0.8165 | -0.0049 | 0.0290 | -0.1681 | 0.9833 |  |  |
| fmnist | iid | free_rider | main |  | 0 | oracle_excl | 0.8504 | +0.0290 | 0.0290 | +1.0000 |  |  |  |
| fmnist | iid | free_rider | main |  | 0 | random_excl | 0.8289 | +0.0075 | 0.0290 | +0.2586 |  |  |  |
| fmnist | iid | free_rider | main |  | 0 | vanilla | 0.8214 | +0.0000 | 0.0290 | +0.0000 |  |  |  |
| fmnist | iid | frrand | main |  | 0 | flirds_gate_v1 | 0.8269 | +0.0060 | 0.0295 | +0.2034 | 0.9487 | 198.0000 | 0.5733 |
| fmnist | iid | frrand | main |  | 0 | flirds_gate_v2 | 0.8407 | +0.0199 | 0.0295 | +0.6737 | 0.8733 | 746.0000 | 0.7882 |
| fmnist | iid | frrand | main |  | 0 | flirds_gatew_v1 | 0.8464 | +0.0255 | 0.0295 | +0.8644 | 0.9992 | 125.0000 | 0.6556 |
| fmnist | iid | frrand | main |  | 0 | flirds_gatew_v2 | 0.8491 | +0.0282 | 0.0295 | +0.9576 | 0.9671 | 44.0000 | 0.9846 |
| fmnist | iid | frrand | main |  | 0 | flirds_mult | 0.8355 | +0.0146 | 0.0295 | +0.4958 | 0.4858 |  |  |
| fmnist | iid | frrand | main |  | 0 | flirds_zgate_v2 | 0.8179 | -0.0030 | 0.0295 | -0.1017 | 0.9754 |  |  |
| fmnist | iid | frrand | main |  | 0 | oracle_excl | 0.8504 | +0.0295 | 0.0295 | +1.0000 |  |  |  |
| fmnist | iid | frrand | main |  | 0 | random_excl | 0.8290 | +0.0081 | 0.0295 | +0.2754 |  |  |  |
| fmnist | iid | frrand | main |  | 0 | vanilla | 0.8209 | +0.0000 | 0.0295 | +0.0000 |  |  |  |
| fmnist | iid | grad_noise | main |  | 0 | flirds_gate_v1 | 0.8306 | +0.0421 | 0.0619 | +0.6808 | 1.0000 | 162.0000 | 0.7259 |
| fmnist | iid | grad_noise | main |  | 0 | flirds_gate_v2 | 0.8684 | +0.0799 | 0.0619 | +1.2909 | 0.9950 | 1144.0000 | 0.7760 |
| fmnist | iid | grad_noise | main |  | 0 | flirds_gatew_v1 | 0.8321 | +0.0436 | 0.0619 | +0.7051 | 0.8054 | 173.0000 | 0.7112 |
| fmnist | iid | grad_noise | main |  | 0 | flirds_gatew_v2 | 0.8698 | +0.0813 | 0.0619 | +1.3131 | 0.9987 | 210.0000 | 0.9457 |
| fmnist | iid | grad_noise | main |  | 0 | flirds_mult | 0.8472 | +0.0587 | 0.0619 | +0.9495 | 1.0000 |  |  |
| fmnist | iid | grad_noise | main |  | 0 | flirds_zgate_v2 | 0.7844 | -0.0041 | 0.0619 | -0.0667 | 1.0000 |  |  |
| fmnist | iid | grad_noise | main |  | 0 | oracle_excl | 0.8504 | +0.0619 | 0.0619 | +1.0000 |  |  |  |
| fmnist | iid | grad_noise | main |  | 0 | random_excl | 0.8003 | +0.0118 | 0.0619 | +0.1899 |  |  |  |
| fmnist | iid | grad_noise | main |  | 0 | vanilla | 0.7885 | +0.0000 | 0.0619 | +0.0000 |  |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 0 | flirds_gate_v1 | 0.8502 | +0.0039 | 0.0091 |  | 0.9861 | 153.0000 | 0.7156 |
| fmnist | iid | label_flip | main | 0.3500 | 0 | flirds_gate_v1 | 0.8391 | +0.0024 | 0.0188 |  | 1.0000 | 171.0000 | 0.7116 |
| fmnist | iid | label_flip | main | 0.7000 | 0 | flirds_gate_v1 | 0.8296 | +0.0076 | 0.0335 | +0.2276 | 1.0000 | 186.0000 | 0.7005 |
| fmnist | iid | label_flip | main | 0.1500 | 0 | flirds_gate_v2 | 0.8454 | -0.0010 | 0.0091 |  | 0.9369 | 331.0000 | 0.8985 |
| fmnist | iid | label_flip | main | 0.3500 | 0 | flirds_gate_v2 | 0.8516 | +0.0149 | 0.0188 |  | 0.9874 | 750.0000 | 0.8325 |
| fmnist | iid | label_flip | main | 0.7000 | 0 | flirds_gate_v2 | 0.8490 | +0.0270 | 0.0335 | +0.8060 | 1.0000 | 509.0000 | 0.8839 |
| fmnist | iid | label_flip | main | 0.1500 | 0 | flirds_gatew_v1 | 0.8484 | +0.0020 | 0.0091 |  | 0.9807 | 136.0000 | 0.7275 |
| fmnist | iid | label_flip | main | 0.3500 | 0 | flirds_gatew_v1 | 0.8417 | +0.0050 | 0.0188 |  | 1.0000 | 156.0000 | 0.7277 |
| fmnist | iid | label_flip | main | 0.7000 | 0 | flirds_gatew_v1 | 0.8354 | +0.0134 | 0.0335 | +0.3993 | 1.0000 | 179.0000 | 0.7122 |
| fmnist | iid | label_flip | main | 0.1500 | 0 | flirds_gatew_v2 | 0.8510 | +0.0046 | 0.0091 |  | 0.9546 | 171.0000 | 0.9420 |
| fmnist | iid | label_flip | main | 0.3500 | 0 | flirds_gatew_v2 | 0.8500 | +0.0132 | 0.0188 |  | 0.9983 | 95.0000 | 0.9750 |
| fmnist | iid | label_flip | main | 0.7000 | 0 | flirds_gatew_v2 | 0.8535 | +0.0315 | 0.0335 | +0.9403 | 0.9983 | 203.0000 | 0.9499 |
| fmnist | iid | label_flip | main | 0.1500 | 0 | flirds_mult | 0.8500 | +0.0036 | 0.0091 |  | 0.9933 |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 0 | flirds_mult | 0.8485 | +0.0118 | 0.0188 |  | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 0 | flirds_mult | 0.8486 | +0.0266 | 0.0335 | +0.7948 | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 0 | flirds_zgate_v2 | 0.8486 | +0.0022 | 0.0091 |  | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 0 | flirds_zgate_v2 | 0.8410 | +0.0042 | 0.0188 |  | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 0 | flirds_zgate_v2 | 0.8354 | +0.0134 | 0.0335 | +0.3993 | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 0 | oracle_excl | 0.8555 | +0.0091 | 0.0091 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 0 | oracle_excl | 0.8555 | +0.0188 | 0.0188 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 0 | oracle_excl | 0.8555 | +0.0335 | 0.0335 | +1.0000 |  |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 0 | random_excl | 0.8474 | +0.0010 | 0.0091 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 0 | random_excl | 0.8404 | +0.0036 | 0.0188 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 0 | random_excl | 0.8283 | +0.0063 | 0.0335 | +0.1866 |  |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 0 | vanilla | 0.8464 | +0.0000 | 0.0091 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 0 | vanilla | 0.8367 | +0.0000 | 0.0188 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 0 | vanilla | 0.8220 | +0.0000 | 0.0335 | +0.0000 |  |  |  |

## V2w promotion gate (spec §5-2): **DO NOT PROMOTE (report CNN-only -- an honest finding)**

  cifar10/dir1/free_rider(str=main): V2w-V2 mean dAcc=-0.0092 FAIL
  cifar10/dir1/frrand(str=main): V2w-V2 mean dAcc=+0.0302 OK
  cifar10/dir1/grad_noise(str=main): V2w-V2 mean dAcc=+0.0206 OK
  cifar10/dir1/label_flip(str=main): V2w-V2 mean dAcc=+0.0046 OK
  cifar10/iid/free_rider(str=main): V2w-V2 mean dAcc=+0.0008 OK
  cifar10/iid/frrand(str=main): V2w-V2 mean dAcc=+0.0190 OK
  cifar10/iid/grad_noise(str=main): V2w-V2 mean dAcc=+0.0041 OK
  cifar10/iid/label_flip(str=main): V2w-V2 mean dAcc=-0.0020 FAIL
  cifar10/qskew/free_rider(str=main): V2w-V2 mean dAcc=-0.0004 FAIL
  cifar10/qskew/frrand(str=main): V2w-V2 mean dAcc=+0.0101 OK
  cifar10/qskew/grad_noise(str=main): V2w-V2 mean dAcc=-0.0062 FAIL
  cifar10/qskew/label_flip(str=main): V2w-V2 mean dAcc=+0.0006 OK
  cifar10/shard/free_rider(str=main): V2w-V2 mean dAcc=+0.1031 OK
  cifar10/shard/frrand(str=main): V2w-V2 mean dAcc=+0.0404 OK
  cifar10/shard/grad_noise(str=main): V2w-V2 mean dAcc=-0.0016 FAIL
  cifar10/shard/label_flip(str=main): V2w-V2 mean dAcc=+0.0846 OK
  fmnist/dir1/free_rider(str=main): V2w-V2 mean dAcc=-0.0168 FAIL
  fmnist/dir1/frrand(str=main): V2w-V2 mean dAcc=+0.0393 OK
  fmnist/dir1/grad_noise(str=main): V2w-V2 mean dAcc=-0.0015 FAIL
  fmnist/dir1/label_flip(str=main): V2w-V2 mean dAcc=+0.0049 OK
  fmnist/iid/free_rider(str=main): V2w-V2 mean dAcc=-0.0002 FAIL
  fmnist/iid/frrand(str=main): V2w-V2 mean dAcc=+0.0084 OK
  fmnist/iid/grad_noise(str=main): V2w-V2 mean dAcc=+0.0014 OK
  fmnist/iid/label_flip(str=main): V2w-V2 mean dAcc=+0.0028 OK
  clean cifar10_dir1_clean_g_seed0: V2w dAcc=-0.0200 FAIL(parity broken)
  clean cifar10_dir1_clean_g_seed1: V2w dAcc=-0.0100 FAIL(parity broken)
  clean cifar10_dir1_clean_g_seed2: V2w dAcc=-0.0302 FAIL(parity broken)
  clean cifar10_iid_clean_g_seed0: V2w dAcc=-0.0122 FAIL(parity broken)
  clean cifar10_iid_clean_g_seed1: V2w dAcc=-0.0016 OK
  clean cifar10_iid_clean_g_seed2: V2w dAcc=-0.0086 FAIL(parity broken)
  clean cifar10_qskew_clean_g_seed0: V2w dAcc=-0.0129 FAIL(parity broken)
  clean cifar10_shard_clean_g_seed0: V2w dAcc=-0.1120 FAIL(parity broken)
  clean cifar10_shard_clean_g_seed1: V2w dAcc=+0.0590 FAIL(parity broken)
  clean fmnist_dir1_clean_g_seed0: V2w dAcc=+0.0495 FAIL(parity broken)
  clean fmnist_iid_clean_g_seed0: V2w dAcc=-0.0009 OK

## CNN skew 분해 (2×2: iid=skew없음 / shard=label만 / qskew=size만 / dir1=둘다) — 3-seed 평균

> ⚠️ 가법 분해 아님: shard의 label-skew(1.95 클래스/클라)는 dir1(9.87)보다, qskew의 size-skew(24×)는 dir1(6.2×)보다 세다. 축 귀속만 읽는다.

**cifar10 / clean** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.6415 | 0.4592 | 0.6614 | 0.6357 |
| flirds_gate_v2 | 0.6428 | 0.4582 | 0.6591 | 0.6315 |
| flirds_gatew_v1 | 0.6339 | 0.2918 | 0.6570 | 0.6198 |
| flirds_gatew_v2 | 0.6412 | 0.4484 | 0.6561 | 0.6188 |
| flirds_mult | 0.6467 | 0.4909 | 0.6637 | 0.6425 |
| flirds_zgate_v2 | 0.6481 | 0.4611 | 0.6666 | 0.6341 |
| vanilla | 0.6488 | 0.4749 | 0.6690 | 0.6389 |

**cifar10 / free_rider** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.6209 (+0.42) | 0.2018 (-2.18) | 0.6472 (+1.20) | 0.6106 (+0.72) |
| flirds_gate_v2 | 0.6308 (+0.81) | 0.3644 (-0.16) | 0.6316 (+0.42) | 0.6148 (+0.84) |
| flirds_gatew_v1 | 0.6249 (+0.58) | 0.2054 (-2.13) | 0.6406 (+0.87) | 0.6113 (+0.75) |
| flirds_gatew_v2 | 0.6317 (+0.84) | 0.4675 (+1.13) | 0.6312 (+0.40) | 0.6056 (+0.59) |
| flirds_mult | 0.6259 (+0.66) | 0.4120 (+0.44) | 0.6366 (+0.67) | 0.5979 (+0.31) |
| flirds_zgate_v2 | 0.6083 (-0.01) | 0.4126 (+0.44) | 0.6202 (-0.14) | 0.5840 (-0.14) |
| oracle_excl | 0.6356 (+1.00) | 0.4574 (+1.00) | 0.6432 (+1.00) | 0.6203 (+1.00) |
| random_excl | 0.5986 (-0.41) | 0.4059 (+0.36) | 0.6128 (-0.52) | 0.5838 (-0.08) |
| vanilla | 0.6083 (+0.00) | 0.3769 (+0.00) | 0.6231 (+0.00) | 0.5879 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0273, shard=0.0805, qskew=0.0201, dir1=0.0324

**cifar10 / frrand** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.6195 (+0.36) | 0.4706 (+1.16) | 0.6284 | 0.5999 (+0.40) |
| flirds_gate_v2 | 0.6105 (-0.02) | 0.4057 (+0.36) | 0.6240 | 0.5806 (-0.17) |
| flirds_gatew_v1 | 0.6319 (+0.88) | 0.2076 (-2.10) | 0.6309 | 0.6254 (+1.16) |
| flirds_gatew_v2 | 0.6295 (+0.78) | 0.4461 (+0.86) | 0.6341 | 0.6109 (+0.72) |
| flirds_mult | 0.6278 (+0.71) | 0.4118 (+0.43) | 0.6358 | 0.5971 (+0.32) |
| flirds_zgate_v2 | 0.6062 (-0.19) | 0.4674 (+1.12) | 0.6215 | 0.5909 (+0.13) |
| oracle_excl | 0.6348 (+1.00) | 0.4574 (+1.00) | 0.6432 | 0.6201 (+1.00) |
| random_excl | 0.6038 (-0.30) | 0.4084 (+0.39) | 0.6118 | 0.5934 (+0.20) |
| vanilla | 0.6109 (+0.00) | 0.3767 (+0.00) | 0.6239 | 0.5865 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0239, shard=0.0806, qskew=0.0194, dir1=0.0336

**cifar10 / grad_noise** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.5796 (+0.85) | 0.1174 (-0.19) | 0.5441 (+0.75) | 0.5462 (+0.81) |
| flirds_gate_v2 | 0.6143 (+0.94) | 0.4079 (+0.83) | 0.6366 (+0.98) | 0.5668 (+0.85) |
| flirds_gatew_v1 | 0.5772 (+0.85) | 0.1430 (-0.10) | 0.2039 (-0.12) | 0.5180 (+0.73) |
| flirds_gatew_v2 | 0.6185 (+0.95) | 0.4062 (+0.82) | 0.6304 (+0.97) | 0.5874 (+0.91) |
| flirds_mult | 0.5329 (+0.73) | 0.2829 (+0.39) | 0.4871 (+0.60) | 0.4364 (+0.51) |
| flirds_zgate_v2 | 0.3191 (+0.17) | 0.1659 (-0.02) | 0.4339 (+0.46) | 0.3419 (+0.26) |
| oracle_excl | 0.6356 (+1.00) | 0.4574 (+1.00) | 0.6432 (+1.00) | 0.6203 (+1.00) |
| random_excl | 0.2645 (+0.02) | 0.1809 (+0.03) | 0.2556 (+0.01) | 0.2590 (+0.04) |
| vanilla | 0.2564 (+0.00) | 0.1709 (+0.00) | 0.2525 (+0.00) | 0.2436 (+0.00) |

gap(oracle_excl−vanilla): iid=0.3793, shard=0.2865, qskew=0.3907, dir1=0.3767

**cifar10 / label_flip@0.15** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.6162 | 0.4517 (+1.25) | 0.6325 | 0.6160 |
| flirds_gate_v2 | 0.6228 | 0.2299 (+11.45) | 0.6298 | 0.5991 |
| flirds_gatew_v1 | 0.6201 | 0.3101 (+7.76) | 0.6404 | 0.6027 |
| flirds_gatew_v2 | 0.6137 | 0.4494 (+1.36) | 0.6391 | 0.5932 |
| flirds_mult | 0.6243 | 0.5064 (-1.26) | 0.6374 | 0.6172 |
| flirds_zgate_v2 | 0.6252 | 0.4699 (+0.41) | 0.6430 | 0.6149 |
| oracle_excl | 0.6310 | 0.4571 (+1.00) | 0.6489 | 0.6236 |
| random_excl | 0.6018 | 0.3762 (+4.72) | 0.6378 | 0.5924 |
| vanilla | 0.6277 | 0.4789 (+0.00) | 0.6406 | 0.6172 |

gap(oracle_excl−vanilla): iid=0.0033, shard=-0.0217, qskew=0.0082, dir1=0.0064

**cifar10 / label_flip@0.35** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.6164 (+0.65) | 0.3029 | 0.6176 (+0.25) | 0.5969 (+0.31) |
| flirds_gate_v2 | 0.6025 (+0.26) | 0.1732 | 0.6101 (+0.07) | 0.5723 (-0.33) |
| flirds_gatew_v1 | 0.6112 (+0.51) | 0.2451 | 0.6196 (+0.30) | 0.5917 (+0.16) |
| flirds_gatew_v2 | 0.5992 (+0.18) | 0.3871 | 0.6171 (+0.24) | 0.5821 (-0.13) |
| flirds_mult | 0.6161 (+0.62) | 0.4575 | 0.6248 (+0.42) | 0.6002 (+0.41) |
| flirds_zgate_v2 | 0.5978 (+0.15) | 0.4487 | 0.6251 (+0.43) | 0.5880 (+0.11) |
| oracle_excl | 0.6310 (+1.00) | 0.4571 | 0.6489 (+1.00) | 0.6236 (+1.00) |
| random_excl | 0.5723 (-0.50) | 0.3469 | 0.6191 (+0.29) | 0.5598 (-0.67) |
| vanilla | 0.5923 (+0.00) | 0.4373 | 0.6070 (+0.00) | 0.5849 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0387, shard=0.0199, qskew=0.0419, dir1=0.0387

**cifar10 / label_flip@0.7** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.5881 (+0.61) | 0.2416 (-1.37) | 0.5827 (+0.27) | 0.5820 (+0.60) |
| flirds_gate_v2 | 0.5967 (+0.70) | 0.3540 (-0.14) | 0.5969 (+0.42) | 0.5712 (+0.43) |
| flirds_gatew_v1 | 0.6063 (+0.78) | 0.2160 (-1.66) | 0.5374 (-0.24) | 0.5480 (+0.18) |
| flirds_gatew_v2 | 0.6032 (+0.76) | 0.1745 (-2.11) | 0.5823 (+0.26) | 0.5810 (+0.56) |
| flirds_mult | 0.6076 (+0.80) | 0.3729 (+0.07) | 0.6252 (+0.74) | 0.5870 (+0.63) |
| flirds_zgate_v2 | 0.5415 (+0.25) | 0.2789 (-0.96) | 0.5995 (+0.45) | 0.5339 (+0.14) |
| oracle_excl | 0.6310 (+1.00) | 0.4571 (+1.00) | 0.6489 (+1.00) | 0.6236 (+1.00) |
| random_excl | 0.5023 (-0.10) | 0.3086 (-0.64) | 0.5891 (+0.34) | 0.5018 (-0.22) |
| vanilla | 0.5171 (+0.00) | 0.3664 (+0.00) | 0.5586 (+0.00) | 0.5247 (+0.00) |

gap(oracle_excl−vanilla): iid=0.1138, shard=0.0907, qskew=0.0902, dir1=0.0990

**fmnist / clean** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8518 | 0.8459 |
| flirds_gate_v2 | 0.8520 | 0.8275 |
| flirds_gatew_v1 | 0.8492 | 0.8475 |
| flirds_gatew_v2 | 0.8508 | 0.8435 |
| flirds_mult | 0.8529 | 0.8154 |
| flirds_zgate_v2 | 0.8535 | 0.8455 |
| vanilla | 0.8516 | 0.7940 |

**fmnist / free_rider** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8501 (+0.99) | 0.7819 (-0.27) |
| flirds_gate_v2 | 0.8506 (+1.01) | 0.8371 (+0.99) |
| flirds_gatew_v1 | 0.8474 (+0.90) | 0.7943 (+0.01) |
| flirds_gatew_v2 | 0.8504 (+1.00) | 0.8204 (+0.61) |
| flirds_mult | 0.8375 (+0.56) | 0.8085 (+0.34) |
| flirds_zgate_v2 | 0.8165 (-0.17) | 0.8021 (+0.19) |
| oracle_excl | 0.8504 (+1.00) | 0.8377 (+1.00) |
| random_excl | 0.8289 (+0.26) | 0.8095 (+0.36) |
| vanilla | 0.8214 (+0.00) | 0.7937 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0290, dir1=0.0440

**fmnist / frrand** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8269 (+0.20) | 0.7934 (-0.07) |
| flirds_gate_v2 | 0.8407 (+0.67) | 0.7989 (+0.07) |
| flirds_gatew_v1 | 0.8464 (+0.86) | 0.7991 (+0.07) |
| flirds_gatew_v2 | 0.8491 (+0.96) | 0.8381 (+1.01) |
| flirds_mult | 0.8355 (+0.50) | 0.8071 (+0.26) |
| flirds_zgate_v2 | 0.8179 (-0.10) | 0.8009 (+0.11) |
| oracle_excl | 0.8504 (+1.00) | 0.8377 (+1.00) |
| random_excl | 0.8290 (+0.28) | 0.8087 (+0.30) |
| vanilla | 0.8209 (+0.00) | 0.7961 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0295, dir1=0.0416

**fmnist / grad_noise** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8306 (+0.68) | 0.7969 (+0.54) |
| flirds_gate_v2 | 0.8684 (+1.29) | 0.8592 (+1.24) |
| flirds_gatew_v1 | 0.8321 (+0.71) | 0.7691 (+0.23) |
| flirds_gatew_v2 | 0.8698 (+1.31) | 0.8578 (+1.22) |
| flirds_mult | 0.8472 (+0.95) | 0.8066 (+0.65) |
| flirds_zgate_v2 | 0.7844 (-0.07) | 0.8079 (+0.67) |
| oracle_excl | 0.8504 (+1.00) | 0.8377 (+1.00) |
| random_excl | 0.8003 (+0.19) | 0.7705 (+0.25) |
| vanilla | 0.7885 (+0.00) | 0.7482 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0619, dir1=0.0895

**fmnist / label_flip@0.15** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8502 | 0.6817 |
| flirds_gate_v2 | 0.8454 | 0.8423 |
| flirds_gatew_v1 | 0.8484 | 0.8346 |
| flirds_gatew_v2 | 0.8510 | 0.8360 |
| flirds_mult | 0.8500 | 0.8239 |
| flirds_zgate_v2 | 0.8486 | 0.8364 |
| oracle_excl | 0.8555 | 0.8204 |
| random_excl | 0.8474 | 0.7943 |
| vanilla | 0.8464 | 0.8219 |

gap(oracle_excl−vanilla): iid=0.0091, dir1=-0.0015

**fmnist / label_flip@0.35** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8391 | 0.8014 |
| flirds_gate_v2 | 0.8516 | 0.8155 |
| flirds_gatew_v1 | 0.8417 | 0.8156 |
| flirds_gatew_v2 | 0.8500 | 0.8316 |
| flirds_mult | 0.8485 | 0.8043 |
| flirds_zgate_v2 | 0.8410 | 0.8207 |
| oracle_excl | 0.8555 | 0.8204 |
| random_excl | 0.8404 | 0.7849 |
| vanilla | 0.8367 | 0.8121 |

gap(oracle_excl−vanilla): iid=0.0188, dir1=0.0082

**fmnist / label_flip@0.7** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8296 (+0.23) | 0.7571 (-1.59) |
| flirds_gate_v2 | 0.8490 (+0.81) | 0.8335 (+1.54) |
| flirds_gatew_v1 | 0.8354 (+0.40) | 0.7901 (-0.24) |
| flirds_gatew_v2 | 0.8535 (+0.94) | 0.8383 (+1.73) |
| flirds_mult | 0.8486 (+0.79) | 0.7770 (-0.78) |
| flirds_zgate_v2 | 0.8354 (+0.40) | 0.8197 (+0.97) |
| oracle_excl | 0.8555 (+1.00) | 0.8204 (+1.00) |
| random_excl | 0.8283 (+0.19) | 0.7695 (-1.09) |
| vanilla | 0.8220 (+0.00) | 0.7960 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0335, dir1=0.0244

## 사전등록 예측 대조 (README 확장 ②; MISS 그대로 보고)

- **H-K1** cifar10 free_rider V2 recovery iid=+0.81, shard=-0.16, qskew=+0.42, dir1=+0.84 -> **MISS**
- **H-K2** cifar10 iid frrand V2 recovery=-0.02 (frzero=+0.81; ratio=-0.02 — <=0.6이면 LLM 감사의 코인플립과 일치) -> **MISS**
- **H-K3** cifar10 clean 오발화 pairs iid=561, shard=3366, qskew=1703, dir1=3808 | V2 dAcc iid=-0.0060, shard=-0.0166, qskew=-0.0099, dir1=-0.0074 -> **MISS**
- **H-K4** cifar10 free_rider recovery seed-sd iid=0.162 -> pending
- **H-K4** cifar10 grad_noise recovery seed-sd iid=0.057 -> pending
- **H-K5** cifar10 lf@0.15 gap iid=0.0033, shard=-0.0217, qskew=0.0082, dir1=0.0064 -> **MISS**
- **H-K1** fmnist free_rider V2 recovery iid=+1.01, dir1=+0.99 -> pending
- **H-K2** fmnist iid frrand V2 recovery=+0.67 (frzero=+1.01; ratio=+0.67 — <=0.6이면 LLM 감사의 코인플립과 일치) -> **MISS**
- **H-K3** fmnist clean 오발화 pairs iid=828, dir1=2746 | V2 dAcc iid=+0.0004, dir1=+0.0335 -> pending
- **H-K4** fmnist free_rider recovery seed-sd  -> pending
- **H-K4** fmnist grad_noise recovery seed-sd  -> pending
- **H-K5** fmnist lf@0.15 gap iid=0.0091, dir1=-0.0015 -> **HIT**
- **H-K6** fmnist↔cifar10 recovery diff iid/free_rider=0.20, iid/frrand=0.69, iid/grad_noise=0.35, iid/label_flip=0.11, dir1/free_rider=0.15, dir1/frrand=0.24, dir1/grad_noise=0.39, dir1/label_flip=1.11 -> **MISS**

## C2 소프트-arm 같은-셀 대조 (runs/track_c/c2, read-only)

| dataset | partition | threat | C2 vanilla | G vanilla | C2 flirds_mult | G flirds_gate_v2 | 비고 |
|---|---|---|---|---|---|---|---|
| cifar10 | dir1 | clean | 0.6380 | 0.6389 | 0.6417 | 0.6315 | same cell |
| cifar10 | dir1 | free_rider | 0.5871 | 0.5879 | 0.5967 | 0.6148 | same cell |
| cifar10 | dir1 | grad_noise | 0.2447 | 0.2436 | 0.4333 | 0.5668 | same cell |
| cifar10 | iid | clean | 0.6479 | 0.6488 | 0.6460 | 0.6428 | same cell |
| cifar10 | iid | free_rider | 0.6084 | 0.6083 | 0.6264 | 0.6308 | same cell |
| cifar10 | iid | grad_noise | 0.2627 | 0.2564 | 0.5401 | 0.6143 | same cell |
| cifar10 | shard | clean | 0.4751 | 0.4749 | 0.4977 | 0.4582 | same cell |
| cifar10 | shard | free_rider | 0.3982 | 0.3769 | 0.4165 | 0.3644 | same cell |
| cifar10 | shard | grad_noise | 0.1667 | 0.1709 | 0.2843 | 0.4079 | same cell |
| fmnist | dir1 | clean | 0.8117 | 0.7940 | 0.8293 | 0.8275 | same cell |
| fmnist | dir1 | free_rider | 0.8081 | 0.7937 | 0.8205 | 0.8371 | same cell |
| fmnist | dir1 | grad_noise | 0.7400 | 0.7482 | 0.7948 | 0.8592 | same cell |
| fmnist | iid | clean | 0.8559 | 0.8516 | 0.8555 | 0.8520 | same cell |
| fmnist | iid | free_rider | 0.8282 | 0.8214 | 0.8405 | 0.8506 | same cell |
| fmnist | iid | grad_noise | 0.7828 | 0.7885 | 0.8305 | 0.8684 | same cell |

⚠️ qskew·frrand는 C2 대응 셀 없음. label_flip은 C2가 strmain(rate~U(0.5,1))이라 Track G의 고정 dose와 같은 셀이 아니어서 제외.

## 스택 재현성 (동일 config·seed, 두 스택) — 감사 M1

(restack 셀 없음 — 전 CNN 표는 단일 원본 스택)

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