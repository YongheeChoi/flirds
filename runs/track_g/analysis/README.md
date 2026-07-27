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

| dataset | partition | threat | strength | flip_rate | seed | n_corrupt | arm | final_acc | delta_acc | gap | recovery | auroc | false_excl_pairs | excl_precision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar10 | dir1 | clean | main |  | 0 | 0 | flirds_gate_v1 | 0.6448 | +0.0059 |  |  |  | 428.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 0 | 0 | flirds_gate_v2 | 0.6246 | -0.0142 |  |  |  | 4110.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 0 | 0 | flirds_gatew_v1 | 0.6419 | +0.0030 |  |  |  | 324.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 0 | 0 | flirds_gatew_v2 | 0.6189 | -0.0200 |  |  |  | 3403.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 0 | 0 | flirds_mult | 0.6434 | +0.0045 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 0 | 0 | flirds_zgate_v2 | 0.6345 | -0.0044 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 0 | 0 | vanilla | 0.6389 | +0.0000 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 1 | 0 | flirds_gate_v1 | 0.6384 | +0.0048 |  |  |  | 437.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 1 | 0 | flirds_gate_v2 | 0.6394 | +0.0058 |  |  |  | 3152.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 1 | 0 | flirds_gatew_v1 | 0.6321 | -0.0015 |  |  |  | 325.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 1 | 0 | flirds_gatew_v2 | 0.6236 | -0.0100 |  |  |  | 2860.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 1 | 0 | flirds_mult | 0.6394 | +0.0058 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 1 | 0 | flirds_zgate_v2 | 0.6424 | +0.0088 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 1 | 0 | vanilla | 0.6336 | +0.0000 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 2 | 0 | flirds_gate_v1 | 0.6239 | -0.0202 |  |  |  | 454.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 2 | 0 | flirds_gate_v2 | 0.6305 | -0.0136 |  |  |  | 4162.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 2 | 0 | flirds_gatew_v1 | 0.5854 | -0.0587 |  |  |  | 323.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 2 | 0 | flirds_gatew_v2 | 0.6139 | -0.0302 |  |  |  | 3393.0000 | 0.0000 |
| cifar10 | dir1 | clean | main |  | 2 | 0 | flirds_mult | 0.6449 | +0.0008 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 2 | 0 | flirds_zgate_v2 | 0.6254 | -0.0187 |  |  |  |  |  |
| cifar10 | dir1 | clean | main |  | 2 | 0 | vanilla | 0.6441 | +0.0000 |  |  |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 0 | 40 | flirds_gate_v1 | 0.6174 | +0.0306 | 0.0336 | +0.9108 | 0.9833 | 157.0000 | 0.7535 |
| cifar10 | dir1 | free_rider | main |  | 0 | 40 | flirds_gate_v2 | 0.6146 | +0.0279 | 0.0336 | +0.8290 | 0.7833 | 1220.0000 | 0.7716 |
| cifar10 | dir1 | free_rider | main |  | 0 | 40 | flirds_gatew_v1 | 0.6244 | +0.0376 | 0.0336 | +1.1190 | 1.0000 | 100.0000 | 0.8276 |
| cifar10 | dir1 | free_rider | main |  | 0 | 40 | flirds_gatew_v2 | 0.6114 | +0.0246 | 0.0336 | +0.7323 | 0.7000 | 1426.0000 | 0.7436 |
| cifar10 | dir1 | free_rider | main |  | 0 | 40 | flirds_mult | 0.5980 | +0.0112 | 0.0336 | +0.3346 | 0.3571 |  |  |
| cifar10 | dir1 | free_rider | main |  | 0 | 40 | flirds_zgate_v2 | 0.5817 | -0.0050 | 0.0336 | -0.1487 | 0.5167 |  |  |
| cifar10 | dir1 | free_rider | main |  | 0 | 40 | oracle_excl | 0.6204 | +0.0336 | 0.0336 | +1.0000 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 0 | 40 | random_excl | 0.5930 | +0.0062 | 0.0336 | +0.1859 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 0 | 40 | vanilla | 0.5867 | +0.0000 | 0.0336 | +0.0000 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 1 | 40 | flirds_gate_v1 | 0.6131 | +0.0219 | 0.0262 | +0.8333 | 0.9667 | 159.0000 | 0.7508 |
| cifar10 | dir1 | free_rider | main |  | 1 | 40 | flirds_gate_v2 | 0.6150 | +0.0237 | 0.0262 | +0.9048 | 0.6667 | 1619.0000 | 0.7139 |
| cifar10 | dir1 | free_rider | main |  | 1 | 40 | flirds_gatew_v1 | 0.6145 | +0.0232 | 0.0262 | +0.8857 | 0.9667 | 117.0000 | 0.8037 |
| cifar10 | dir1 | free_rider | main |  | 1 | 40 | flirds_gatew_v2 | 0.6146 | +0.0234 | 0.0262 | +0.8905 | 0.6833 | 1264.0000 | 0.7605 |
| cifar10 | dir1 | free_rider | main |  | 1 | 40 | flirds_mult | 0.5994 | +0.0081 | 0.0262 | +0.3095 | 0.4313 |  |  |
| cifar10 | dir1 | free_rider | main |  | 1 | 40 | flirds_zgate_v2 | 0.5827 | -0.0085 | 0.0262 | -0.3238 | 0.6167 |  |  |
| cifar10 | dir1 | free_rider | main |  | 1 | 40 | oracle_excl | 0.6175 | +0.0262 | 0.0262 | +1.0000 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 1 | 40 | random_excl | 0.5979 | +0.0066 | 0.0262 | +0.2524 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 1 | 40 | vanilla | 0.5913 | +0.0000 | 0.0262 | +0.0000 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 2 | 40 | flirds_gate_v1 | 0.6012 | +0.0155 | 0.0374 | +0.4147 | 1.0000 | 154.0000 | 0.7609 |
| cifar10 | dir1 | free_rider | main |  | 2 | 40 | flirds_gate_v2 | 0.6149 | +0.0291 | 0.0374 | +0.7793 | 0.6667 | 1803.0000 | 0.6878 |
| cifar10 | dir1 | free_rider | main |  | 2 | 40 | flirds_gatew_v1 | 0.5950 | +0.0092 | 0.0374 | +0.2475 | 1.0000 | 104.0000 | 0.8249 |
| cifar10 | dir1 | free_rider | main |  | 2 | 40 | flirds_gatew_v2 | 0.5909 | +0.0051 | 0.0374 | +0.1371 | 0.7167 | 1310.0000 | 0.7511 |
| cifar10 | dir1 | free_rider | main |  | 2 | 40 | flirds_mult | 0.5962 | +0.0105 | 0.0374 | +0.2809 | 0.4025 |  |  |
| cifar10 | dir1 | free_rider | main |  | 2 | 40 | flirds_zgate_v2 | 0.5875 | +0.0018 | 0.0374 | +0.0468 | 0.5167 |  |  |
| cifar10 | dir1 | free_rider | main |  | 2 | 40 | oracle_excl | 0.6231 | +0.0374 | 0.0374 | +1.0000 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 2 | 40 | random_excl | 0.5606 | -0.0251 | 0.0374 | -0.6722 |  |  |  |
| cifar10 | dir1 | free_rider | main |  | 2 | 40 | vanilla | 0.5857 | +0.0000 | 0.0374 | +0.0000 |  |  |  |
| cifar10 | dir1 | frrand | main |  | 0 | 40 | flirds_gate_v1 | 0.5999 | +0.0134 | 0.0336 | +0.3978 | 0.8996 | 267.0000 | 0.4785 |
| cifar10 | dir1 | frrand | main |  | 0 | 40 | flirds_gate_v2 | 0.5806 | -0.0059 | 0.0336 | -0.1747 | 0.5096 | 2514.0000 | 0.5195 |
| cifar10 | dir1 | frrand | main |  | 0 | 40 | flirds_gatew_v1 | 0.6254 | +0.0389 | 0.0336 | +1.1561 | 1.0000 | 99.0000 | 0.7188 |
| cifar10 | dir1 | frrand | main |  | 0 | 40 | flirds_gatew_v2 | 0.6109 | +0.0244 | 0.0336 | +0.7249 | 0.8500 | 886.0000 | 0.7582 |
| cifar10 | dir1 | frrand | main |  | 0 | 40 | flirds_mult | 0.5971 | +0.0106 | 0.0336 | +0.3160 | 0.3567 |  |  |
| cifar10 | dir1 | frrand | main |  | 0 | 40 | flirds_zgate_v2 | 0.5909 | +0.0044 | 0.0336 | +0.1301 | 0.5858 |  |  |
| cifar10 | dir1 | frrand | main |  | 0 | 40 | oracle_excl | 0.6201 | +0.0336 | 0.0336 | +1.0000 |  |  |  |
| cifar10 | dir1 | frrand | main |  | 0 | 40 | random_excl | 0.5934 | +0.0069 | 0.0336 | +0.2045 |  |  |  |
| cifar10 | dir1 | frrand | main |  | 0 | 40 | vanilla | 0.5865 | +0.0000 | 0.0336 | +0.0000 |  |  |  |
| cifar10 | dir1 | frrand | main |  | 1 | 40 | flirds_gate_v1 | 0.5949 | +0.0026 | 0.0241 | +0.1088 | 0.7979 | 279.0000 | 0.4795 |
| cifar10 | dir1 | frrand | main |  | 1 | 40 | flirds_gate_v2 | 0.5962 | +0.0040 | 0.0241 | +0.1658 | 0.6046 | 2081.0000 | 0.5705 |
| cifar10 | dir1 | frrand | main |  | 1 | 40 | flirds_gatew_v1 | 0.6191 | +0.0269 | 0.0241 | +1.1140 | 0.9833 | 120.0000 | 0.6639 |
| cifar10 | dir1 | frrand | main |  | 1 | 40 | flirds_gatew_v2 | 0.6136 | +0.0214 | 0.0241 | +0.8860 | 0.7833 | 1216.0000 | 0.7129 |
| cifar10 | dir1 | frrand | main |  | 1 | 40 | flirds_mult | 0.5972 | +0.0050 | 0.0241 | +0.2073 | 0.4196 |  |  |
| cifar10 | dir1 | frrand | main |  | 1 | 40 | flirds_zgate_v2 | 0.5809 | -0.0114 | 0.0241 | -0.4715 | 0.6167 |  |  |
| cifar10 | dir1 | frrand | main |  | 1 | 40 | oracle_excl | 0.6164 | +0.0241 | 0.0241 | +1.0000 |  |  |  |
| cifar10 | dir1 | frrand | main |  | 1 | 40 | random_excl | 0.5968 | +0.0045 | 0.0241 | +0.1865 |  |  |  |
| cifar10 | dir1 | frrand | main |  | 1 | 40 | vanilla | 0.5923 | +0.0000 | 0.0241 | +0.0000 |  |  |  |
| cifar10 | dir1 | frrand | main |  | 2 | 40 | flirds_gate_v1 | 0.5913 | +0.0073 | 0.0379 | +0.1914 | 0.8333 | 280.0000 | 0.4717 |
| cifar10 | dir1 | frrand | main |  | 2 | 40 | flirds_gate_v2 | 0.5915 | +0.0075 | 0.0379 | +0.1980 | 0.5321 | 2519.0000 | 0.5489 |
| cifar10 | dir1 | frrand | main |  | 2 | 40 | flirds_gatew_v1 | 0.5917 | +0.0078 | 0.0379 | +0.2046 | 1.0000 | 105.0000 | 0.7075 |
| cifar10 | dir1 | frrand | main |  | 2 | 40 | flirds_gatew_v2 | 0.5875 | +0.0035 | 0.0379 | +0.0924 | 0.7346 | 1245.0000 | 0.7050 |
| cifar10 | dir1 | frrand | main |  | 2 | 40 | flirds_mult | 0.5954 | +0.0114 | 0.0379 | +0.3003 | 0.4029 |  |  |
| cifar10 | dir1 | frrand | main |  | 2 | 40 | flirds_zgate_v2 | 0.5855 | +0.0015 | 0.0379 | +0.0396 | 0.5333 |  |  |
| cifar10 | dir1 | frrand | main |  | 2 | 40 | oracle_excl | 0.6219 | +0.0379 | 0.0379 | +1.0000 |  |  |  |
| cifar10 | dir1 | frrand | main |  | 2 | 40 | random_excl | 0.5616 | -0.0224 | 0.0379 | -0.5908 |  |  |  |
| cifar10 | dir1 | frrand | main |  | 2 | 40 | vanilla | 0.5840 | +0.0000 | 0.0379 | +0.0000 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 0 | 40 | flirds_gate_v1 | 0.5724 | +0.3066 | 0.3546 | +0.8646 | 0.9987 | 203.0000 | 0.6915 |
| cifar10 | dir1 | grad_noise | main |  | 0 | 40 | flirds_gate_v2 | 0.5370 | +0.2713 | 0.3546 | +0.7649 | 0.9892 | 1243.0000 | 0.7649 |
| cifar10 | dir1 | grad_noise | main |  | 0 | 40 | flirds_gatew_v1 | 0.5449 | +0.2791 | 0.3546 | +0.7871 | 0.9996 | 172.0000 | 0.7287 |
| cifar10 | dir1 | grad_noise | main |  | 0 | 40 | flirds_gatew_v2 | 0.5783 | +0.3125 | 0.3546 | +0.8812 | 0.9788 | 1651.0000 | 0.7141 |
| cifar10 | dir1 | grad_noise | main |  | 0 | 40 | flirds_mult | 0.4629 | +0.1971 | 0.3546 | +0.5559 | 0.9925 |  |  |
| cifar10 | dir1 | grad_noise | main |  | 0 | 40 | flirds_zgate_v2 | 0.3789 | +0.1131 | 0.3546 | +0.3190 | 0.9967 |  |  |
| cifar10 | dir1 | grad_noise | main |  | 0 | 40 | oracle_excl | 0.6204 | +0.3546 | 0.3546 | +1.0000 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 0 | 40 | random_excl | 0.2411 | -0.0246 | 0.3546 | -0.0694 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 0 | 40 | vanilla | 0.2657 | +0.0000 | 0.3546 | +0.0000 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 1 | 40 | flirds_gate_v1 | 0.5129 | +0.2915 | 0.3961 | +0.7359 | 0.9996 | 194.0000 | 0.7025 |
| cifar10 | dir1 | grad_noise | main |  | 1 | 40 | flirds_gate_v2 | 0.5853 | +0.3639 | 0.3961 | +0.9186 | 0.9950 | 1607.0000 | 0.7158 |
| cifar10 | dir1 | grad_noise | main |  | 1 | 40 | flirds_gatew_v1 | 0.5058 | +0.2844 | 0.3961 | +0.7179 | 0.9983 | 209.0000 | 0.6819 |
| cifar10 | dir1 | grad_noise | main |  | 1 | 40 | flirds_gatew_v2 | 0.5944 | +0.3730 | 0.3961 | +0.9416 | 0.9954 | 1766.0000 | 0.6925 |
| cifar10 | dir1 | grad_noise | main |  | 1 | 40 | flirds_mult | 0.4457 | +0.2244 | 0.3961 | +0.5664 | 0.9975 |  |  |
| cifar10 | dir1 | grad_noise | main |  | 1 | 40 | flirds_zgate_v2 | 0.3287 | +0.1074 | 0.3961 | +0.2711 | 0.9946 |  |  |
| cifar10 | dir1 | grad_noise | main |  | 1 | 40 | oracle_excl | 0.6175 | +0.3961 | 0.3961 | +1.0000 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 1 | 40 | random_excl | 0.2819 | +0.0605 | 0.3961 | +0.1527 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 1 | 40 | vanilla | 0.2214 | +0.0000 | 0.3961 | +0.0000 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 2 | 40 | flirds_gate_v1 | 0.5533 | +0.3095 | 0.3794 | +0.8158 | 1.0000 | 215.0000 | 0.6884 |
| cifar10 | dir1 | grad_noise | main |  | 2 | 40 | flirds_gate_v2 | 0.5781 | +0.3344 | 0.3794 | +0.8814 | 0.9912 | 1231.0000 | 0.7664 |
| cifar10 | dir1 | grad_noise | main |  | 2 | 40 | flirds_gatew_v1 | 0.5034 | +0.2596 | 0.3794 | +0.6843 | 1.0000 | 166.0000 | 0.7390 |
| cifar10 | dir1 | grad_noise | main |  | 2 | 40 | flirds_gatew_v2 | 0.5896 | +0.3459 | 0.3794 | +0.9117 | 0.9946 | 1458.0000 | 0.7334 |
| cifar10 | dir1 | grad_noise | main |  | 2 | 40 | flirds_mult | 0.4005 | +0.1568 | 0.3794 | +0.4132 | 0.9908 |  |  |
| cifar10 | dir1 | grad_noise | main |  | 2 | 40 | flirds_zgate_v2 | 0.3180 | +0.0743 | 0.3794 | +0.1957 | 1.0000 |  |  |
| cifar10 | dir1 | grad_noise | main |  | 2 | 40 | oracle_excl | 0.6231 | +0.3794 | 0.3794 | +1.0000 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 2 | 40 | random_excl | 0.2541 | +0.0104 | 0.3794 | +0.0273 |  |  |  |
| cifar10 | dir1 | grad_noise | main |  | 2 | 40 | vanilla | 0.2437 | +0.0000 | 0.3794 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | 39 | flirds_gate_v1 | 0.6235 | +0.0015 | 0.0051 |  | 0.5393 | 244.0000 | 0.4163 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | 39 | flirds_gate_v1 | 0.6061 | +0.0074 | 0.0284 | +0.2599 | 0.7306 | 194.0000 | 0.5571 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | 39 | flirds_gate_v1 | 0.6065 | +0.0505 | 0.0711 | +0.7100 | 0.9924 | 158.0000 | 0.6715 |
| cifar10 | dir1 | label_flip | main |  | 0 | 39 | flirds_gate_v1 | 0.5906 | +0.0317 | 0.0673 | +0.4721 | 0.9908 | 161.0000 | 0.6734 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | 39 | flirds_gate_v2 | 0.6159 | -0.0061 | 0.0051 |  | 0.6049 | 1025.0000 | 0.6047 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | 39 | flirds_gate_v2 | 0.5936 | -0.0051 | 0.0284 | -0.1806 | 0.6604 | 1524.0000 | 0.5818 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | 39 | flirds_gate_v2 | 0.5641 | +0.0081 | 0.0711 | +0.1142 | 0.8083 | 1246.0000 | 0.7063 |
| cifar10 | dir1 | label_flip | main |  | 0 | 39 | flirds_gate_v2 | 0.5650 | +0.0061 | 0.0673 | +0.0911 | 0.8142 | 1411.0000 | 0.6776 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | 39 | flirds_gatew_v1 | 0.6224 | +0.0004 | 0.0051 |  | 0.5721 | 185.0000 | 0.4108 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | 39 | flirds_gatew_v1 | 0.6012 | +0.0025 | 0.0284 | +0.0881 | 0.6789 | 155.0000 | 0.5646 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | 39 | flirds_gatew_v1 | 0.5360 | -0.0200 | 0.0711 | -0.2812 | 0.9462 | 127.0000 | 0.6760 |
| cifar10 | dir1 | label_flip | main |  | 0 | 39 | flirds_gatew_v1 | 0.5594 | +0.0005 | 0.0673 | +0.0074 | 0.9554 | 123.0000 | 0.6925 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | 39 | flirds_gatew_v2 | 0.5978 | -0.0242 | 0.0051 |  | 0.4426 | 1746.0000 | 0.4149 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | 39 | flirds_gatew_v2 | 0.5833 | -0.0155 | 0.0284 | -0.5463 | 0.5532 | 1071.0000 | 0.5906 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | 39 | flirds_gatew_v2 | 0.5938 | +0.0377 | 0.0711 | +0.5308 | 0.6679 | 1292.0000 | 0.6704 |
| cifar10 | dir1 | label_flip | main |  | 0 | 39 | flirds_gatew_v2 | 0.5677 | +0.0089 | 0.0673 | +0.1320 | 0.6936 | 1272.0000 | 0.6948 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | 39 | flirds_mult | 0.6246 | +0.0026 | 0.0051 |  | 0.4435 |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | 39 | flirds_mult | 0.6124 | +0.0136 | 0.0284 | +0.4802 | 0.6646 |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | 39 | flirds_mult | 0.5995 | +0.0435 | 0.0711 | +0.6116 | 0.8676 |  |  |
| cifar10 | dir1 | label_flip | main |  | 0 | 39 | flirds_mult | 0.5999 | +0.0410 | 0.0673 | +0.6097 | 0.8743 |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | 39 | flirds_zgate_v2 | 0.6148 | -0.0072 | 0.0051 |  | 0.5351 |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | 39 | flirds_zgate_v2 | 0.6088 | +0.0100 | 0.0284 | +0.3524 | 0.9092 |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | 39 | flirds_zgate_v2 | 0.5889 | +0.0329 | 0.0711 | +0.4622 | 0.9899 |  |  |
| cifar10 | dir1 | label_flip | main |  | 0 | 39 | flirds_zgate_v2 | 0.5845 | +0.0256 | 0.0673 | +0.3810 | 1.0000 |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | 39 | oracle_excl | 0.6271 | +0.0051 | 0.0051 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | 39 | oracle_excl | 0.6271 | +0.0284 | 0.0284 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | 39 | oracle_excl | 0.6271 | +0.0711 | 0.0711 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main |  | 0 | 39 | oracle_excl | 0.6261 | +0.0673 | 0.0673 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | 39 | random_excl | 0.6066 | -0.0154 | 0.0051 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | 39 | random_excl | 0.5817 | -0.0170 | 0.0284 | -0.5991 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | 39 | random_excl | 0.5599 | +0.0039 | 0.0711 | +0.0545 |  |  |  |
| cifar10 | dir1 | label_flip | main |  | 0 | 39 | random_excl | 0.5573 | -0.0016 | 0.0673 | -0.0242 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 0 | 39 | vanilla | 0.6220 | +0.0000 | 0.0051 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 0 | 39 | vanilla | 0.5988 | +0.0000 | 0.0284 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 0 | 39 | vanilla | 0.5560 | +0.0000 | 0.0711 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main |  | 0 | 39 | vanilla | 0.5589 | +0.0000 | 0.0673 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | 48 | flirds_gate_v1 | 0.6122 | -0.0019 | 0.0072 |  | 0.5349 | 219.0000 | 0.4798 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | 48 | flirds_gate_v1 | 0.5797 | +0.0051 | 0.0467 | +0.1096 | 0.7845 | 159.0000 | 0.6521 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | 48 | flirds_gate_v1 | 0.5525 | +0.0534 | 0.1223 | +0.4366 | 0.9732 | 126.0000 | 0.7595 |
| cifar10 | dir1 | label_flip | main |  | 1 | 48 | flirds_gate_v1 | 0.5923 | +0.1154 | 0.1427 | +0.8082 | 0.9255 | 124.0000 | 0.7678 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | 48 | flirds_gate_v2 | 0.5951 | -0.0190 | 0.0072 |  | 0.4796 | 1547.0000 | 0.4774 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | 48 | flirds_gate_v2 | 0.5745 | -0.0001 | 0.0467 | -0.0027 | 0.6963 | 1046.0000 | 0.7269 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | 48 | flirds_gate_v2 | 0.5690 | +0.0699 | 0.1223 | +0.5716 | 0.8293 | 865.0000 | 0.8329 |
| cifar10 | dir1 | label_flip | main |  | 1 | 48 | flirds_gate_v2 | 0.5389 | +0.0620 | 0.1427 | +0.4343 | 0.7929 | 1038.0000 | 0.7987 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | 48 | flirds_gatew_v1 | 0.6124 | -0.0018 | 0.0072 |  | 0.5240 | 184.0000 | 0.4620 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | 48 | flirds_gatew_v1 | 0.5914 | +0.0167 | 0.0467 | +0.3583 | 0.7969 | 128.0000 | 0.6649 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | 48 | flirds_gatew_v1 | 0.5410 | +0.0419 | 0.1223 | +0.3425 | 0.9567 | 104.0000 | 0.7684 |
| cifar10 | dir1 | label_flip | main |  | 1 | 48 | flirds_gatew_v1 | 0.5125 | +0.0356 | 0.1427 | +0.2496 | 0.9507 | 101.0000 | 0.7765 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | 48 | flirds_gatew_v2 | 0.5881 | -0.0260 | 0.0072 |  | 0.4319 | 1270.0000 | 0.4998 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | 48 | flirds_gatew_v2 | 0.5811 | +0.0065 | 0.0467 | +0.1390 | 0.6635 | 1028.0000 | 0.7100 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | 48 | flirds_gatew_v2 | 0.5784 | +0.0792 | 0.1223 | +0.6483 | 0.7825 | 1539.0000 | 0.7128 |
| cifar10 | dir1 | label_flip | main |  | 1 | 48 | flirds_gatew_v2 | 0.5884 | +0.1115 | 0.1427 | +0.7811 | 0.7704 | 1033.0000 | 0.7741 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | 48 | flirds_mult | 0.6165 | +0.0024 | 0.0072 |  | 0.5557 |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | 48 | flirds_mult | 0.5881 | +0.0135 | 0.0467 | +0.2888 | 0.8550 |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | 48 | flirds_mult | 0.5755 | +0.0764 | 0.1223 | +0.6247 | 0.9339 |  |  |
| cifar10 | dir1 | label_flip | main |  | 1 | 48 | flirds_mult | 0.5703 | +0.0934 | 0.1427 | +0.6541 | 0.9463 |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | 48 | flirds_zgate_v2 | 0.6200 | +0.0059 | 0.0072 |  | 0.5877 |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | 48 | flirds_zgate_v2 | 0.5819 | +0.0072 | 0.0467 | +0.1551 | 0.9111 |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | 48 | flirds_zgate_v2 | 0.4964 | -0.0027 | 0.1223 | -0.0225 | 0.9844 |  |  |
| cifar10 | dir1 | label_flip | main |  | 1 | 48 | flirds_zgate_v2 | 0.5513 | +0.0744 | 0.1427 | +0.5210 | 0.9940 |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | 48 | oracle_excl | 0.6214 | +0.0072 | 0.0072 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | 48 | oracle_excl | 0.6214 | +0.0467 | 0.0467 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | 48 | oracle_excl | 0.6214 | +0.1223 | 0.1223 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main |  | 1 | 48 | oracle_excl | 0.6196 | +0.1427 | 0.1427 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | 48 | random_excl | 0.5969 | -0.0172 | 0.0072 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | 48 | random_excl | 0.5687 | -0.0059 | 0.0467 | -0.1257 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | 48 | random_excl | 0.5071 | +0.0080 | 0.1223 | +0.0654 |  |  |  |
| cifar10 | dir1 | label_flip | main |  | 1 | 48 | random_excl | 0.4999 | +0.0230 | 0.1427 | +0.1611 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 1 | 48 | vanilla | 0.6141 | +0.0000 | 0.0072 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 1 | 48 | vanilla | 0.5746 | +0.0000 | 0.0467 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 1 | 48 | vanilla | 0.4991 | +0.0000 | 0.1223 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main |  | 1 | 48 | vanilla | 0.4769 | +0.0000 | 0.1427 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | 47 | flirds_gate_v1 | 0.6122 | -0.0034 | 0.0068 |  | 0.5548 | 221.0000 | 0.4977 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | 47 | flirds_gate_v1 | 0.6048 | +0.0235 | 0.0411 | +0.5714 | 0.8539 | 148.0000 | 0.6711 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | 47 | flirds_gate_v1 | 0.5870 | +0.0681 | 0.1035 | +0.6582 | 0.9976 | 128.0000 | 0.7571 |
| cifar10 | dir1 | label_flip | main |  | 2 | 47 | flirds_gate_v1 | 0.5853 | +0.0764 | 0.1131 | +0.6751 | 0.9864 | 126.0000 | 0.7591 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | 47 | flirds_gate_v2 | 0.5863 | -0.0294 | 0.0068 |  | 0.5705 | 1751.0000 | 0.5103 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | 47 | flirds_gate_v2 | 0.5486 | -0.0326 | 0.0411 | -0.7933 | 0.6636 | 2047.0000 | 0.6247 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | 47 | flirds_gate_v2 | 0.5805 | +0.0616 | 0.1035 | +0.5954 | 0.7539 | 1895.0000 | 0.6881 |
| cifar10 | dir1 | label_flip | main |  | 2 | 47 | flirds_gate_v2 | 0.5563 | +0.0474 | 0.1131 | +0.4188 | 0.7820 | 1898.0000 | 0.6762 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | 47 | flirds_gatew_v1 | 0.5733 | -0.0424 | 0.0068 |  | 0.5640 | 163.0000 | 0.5262 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | 47 | flirds_gatew_v1 | 0.5824 | +0.0011 | 0.0411 | +0.0274 | 0.7656 | 115.0000 | 0.6909 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | 47 | flirds_gatew_v1 | 0.5669 | +0.0480 | 0.1035 | +0.4638 | 0.9735 | 107.0000 | 0.7723 |
| cifar10 | dir1 | label_flip | main |  | 2 | 47 | flirds_gatew_v1 | 0.5711 | +0.0623 | 0.1131 | +0.5503 | 0.9506 | 109.0000 | 0.7720 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | 47 | flirds_gatew_v2 | 0.5939 | -0.0217 | 0.0068 |  | 0.5151 | 1343.0000 | 0.5388 |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | 47 | flirds_gatew_v2 | 0.5819 | +0.0006 | 0.0411 | +0.0152 | 0.6471 | 951.0000 | 0.6729 |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | 47 | flirds_gatew_v2 | 0.5709 | +0.0520 | 0.1035 | +0.5024 | 0.7146 | 1557.0000 | 0.7067 |
| cifar10 | dir1 | label_flip | main |  | 2 | 47 | flirds_gatew_v2 | 0.5576 | +0.0488 | 0.1131 | +0.4309 | 0.7130 | 1421.0000 | 0.7046 |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | 47 | flirds_mult | 0.6104 | -0.0052 | 0.0068 |  | 0.5299 |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | 47 | flirds_mult | 0.6002 | +0.0190 | 0.0411 | +0.4620 | 0.8109 |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | 47 | flirds_mult | 0.5859 | +0.0670 | 0.1035 | +0.6473 | 0.9707 |  |  |
| cifar10 | dir1 | label_flip | main |  | 2 | 47 | flirds_mult | 0.5846 | +0.0757 | 0.1131 | +0.6696 | 0.9582 |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | 47 | flirds_zgate_v2 | 0.6099 | -0.0058 | 0.0068 |  | 0.5921 |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | 47 | flirds_zgate_v2 | 0.5735 | -0.0078 | 0.0411 | -0.1884 | 0.9342 |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | 47 | flirds_zgate_v2 | 0.5165 | -0.0024 | 0.1035 | -0.0229 | 1.0000 |  |  |
| cifar10 | dir1 | label_flip | main |  | 2 | 47 | flirds_zgate_v2 | 0.5235 | +0.0146 | 0.1131 | +0.1293 | 1.0000 |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | 47 | oracle_excl | 0.6224 | +0.0068 | 0.0068 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | 47 | oracle_excl | 0.6224 | +0.0411 | 0.0411 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | 47 | oracle_excl | 0.6224 | +0.1035 | 0.1035 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main |  | 2 | 47 | oracle_excl | 0.6220 | +0.1131 | 0.1131 | +1.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | 47 | random_excl | 0.5736 | -0.0420 | 0.0068 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | 47 | random_excl | 0.5290 | -0.0523 | 0.0411 | -1.2705 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | 47 | random_excl | 0.4385 | -0.0804 | 0.1035 | -0.7766 |  |  |  |
| cifar10 | dir1 | label_flip | main |  | 2 | 47 | random_excl | 0.4214 | -0.0875 | 0.1131 | -0.7735 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.1500 | 2 | 47 | vanilla | 0.6156 | +0.0000 | 0.0068 |  |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.3500 | 2 | 47 | vanilla | 0.5813 | +0.0000 | 0.0411 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main | 0.7000 | 2 | 47 | vanilla | 0.5189 | +0.0000 | 0.1035 | +0.0000 |  |  |  |
| cifar10 | dir1 | label_flip | main |  | 2 | 47 | vanilla | 0.5089 | +0.0000 | 0.1131 | +0.0000 |  |  |  |
| cifar10 | iid | clean | main |  | 0 | 0 | flirds_gate_v1 | 0.6438 | -0.0054 |  |  |  | 383.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 0 | 0 | flirds_gate_v2 | 0.6454 | -0.0037 |  |  |  | 450.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 0 | 0 | flirds_gatew_v1 | 0.6455 | -0.0036 |  |  |  | 313.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 0 | 0 | flirds_gatew_v2 | 0.6369 | -0.0122 |  |  |  | 643.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 0 | 0 | flirds_mult | 0.6469 | -0.0022 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 0 | 0 | flirds_zgate_v2 | 0.6455 | -0.0036 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 0 | 0 | vanilla | 0.6491 | +0.0000 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 1 | 0 | flirds_gate_v1 | 0.6395 | -0.0088 |  |  |  | 363.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 1 | 0 | flirds_gate_v2 | 0.6488 | +0.0005 |  |  |  | 651.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 1 | 0 | flirds_gatew_v1 | 0.6149 | -0.0334 |  |  |  | 318.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 1 | 0 | flirds_gatew_v2 | 0.6466 | -0.0016 |  |  |  | 746.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 1 | 0 | flirds_mult | 0.6481 | -0.0001 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 1 | 0 | flirds_zgate_v2 | 0.6501 | +0.0019 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 1 | 0 | vanilla | 0.6482 | +0.0000 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 2 | 0 | flirds_gate_v1 | 0.6411 | -0.0078 |  |  |  | 381.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 2 | 0 | flirds_gate_v2 | 0.6342 | -0.0146 |  |  |  | 583.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 2 | 0 | flirds_gatew_v1 | 0.6414 | -0.0075 |  |  |  | 311.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 2 | 0 | flirds_gatew_v2 | 0.6402 | -0.0086 |  |  |  | 547.0000 | 0.0000 |
| cifar10 | iid | clean | main |  | 2 | 0 | flirds_mult | 0.6450 | -0.0039 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 2 | 0 | flirds_zgate_v2 | 0.6486 | -0.0002 |  |  |  |  |  |
| cifar10 | iid | clean | main |  | 2 | 0 | vanilla | 0.6489 | +0.0000 |  |  |  |  |  |
| cifar10 | iid | free_rider | main |  | 0 | 40 | flirds_gate_v1 | 0.6048 | -0.0051 | 0.0248 | -0.2071 | 1.0000 | 123.0000 | 0.7960 |
| cifar10 | iid | free_rider | main |  | 0 | 40 | flirds_gate_v2 | 0.6256 | +0.0158 | 0.0248 | +0.6364 | 0.9333 | 176.0000 | 0.9587 |
| cifar10 | iid | free_rider | main |  | 0 | 40 | flirds_gatew_v1 | 0.6178 | +0.0079 | 0.0248 | +0.3182 | 1.0000 | 94.0000 | 0.8362 |
| cifar10 | iid | free_rider | main |  | 0 | 40 | flirds_gatew_v2 | 0.6272 | +0.0174 | 0.0248 | +0.7020 | 0.9167 | 215.0000 | 0.9500 |
| cifar10 | iid | free_rider | main |  | 0 | 40 | flirds_mult | 0.6279 | +0.0180 | 0.0248 | +0.7273 | 0.4646 |  |  |
| cifar10 | iid | free_rider | main |  | 0 | 40 | flirds_zgate_v2 | 0.6055 | -0.0044 | 0.0248 | -0.1768 | 0.9000 |  |  |
| cifar10 | iid | free_rider | main |  | 0 | 40 | oracle_excl | 0.6346 | +0.0248 | 0.0248 | +1.0000 |  |  |  |
| cifar10 | iid | free_rider | main |  | 0 | 40 | random_excl | 0.6035 | -0.0064 | 0.0248 | -0.2576 |  |  |  |
| cifar10 | iid | free_rider | main |  | 0 | 40 | vanilla | 0.6099 | +0.0000 | 0.0248 | +0.0000 |  |  |  |
| cifar10 | iid | free_rider | main |  | 1 | 40 | flirds_gate_v1 | 0.6312 | +0.0281 | 0.0337 | +0.8333 | 1.0000 | 116.0000 | 0.8050 |
| cifar10 | iid | free_rider | main |  | 1 | 40 | flirds_gate_v2 | 0.6355 | +0.0324 | 0.0337 | +0.9593 | 0.8500 | 401.0000 | 0.9085 |
| cifar10 | iid | free_rider | main |  | 1 | 40 | flirds_gatew_v1 | 0.6304 | +0.0272 | 0.0337 | +0.8074 | 1.0000 | 94.0000 | 0.8360 |
| cifar10 | iid | free_rider | main |  | 1 | 40 | flirds_gatew_v2 | 0.6352 | +0.0321 | 0.0337 | +0.9519 | 0.9167 | 168.0000 | 0.9591 |
| cifar10 | iid | free_rider | main |  | 1 | 40 | flirds_mult | 0.6202 | +0.0171 | 0.0337 | +0.5074 | 0.3871 |  |  |
| cifar10 | iid | free_rider | main |  | 1 | 40 | flirds_zgate_v2 | 0.6058 | +0.0026 | 0.0337 | +0.0778 | 0.9167 |  |  |
| cifar10 | iid | free_rider | main |  | 1 | 40 | oracle_excl | 0.6369 | +0.0337 | 0.0337 | +1.0000 |  |  |  |
| cifar10 | iid | free_rider | main |  | 1 | 40 | random_excl | 0.6036 | +0.0005 | 0.0337 | +0.0148 |  |  |  |
| cifar10 | iid | free_rider | main |  | 1 | 40 | vanilla | 0.6031 | +0.0000 | 0.0337 | +0.0000 |  |  |  |
| cifar10 | iid | free_rider | main |  | 2 | 40 | flirds_gate_v1 | 0.6268 | +0.0149 | 0.0235 | +0.6330 | 1.0000 | 126.0000 | 0.7955 |
| cifar10 | iid | free_rider | main |  | 2 | 40 | flirds_gate_v2 | 0.6314 | +0.0195 | 0.0235 | +0.8298 | 0.8667 | 352.0000 | 0.9194 |
| cifar10 | iid | free_rider | main |  | 2 | 40 | flirds_gatew_v1 | 0.6265 | +0.0146 | 0.0235 | +0.6223 | 1.0000 | 93.0000 | 0.8405 |
| cifar10 | iid | free_rider | main |  | 2 | 40 | flirds_gatew_v2 | 0.6325 | +0.0206 | 0.0235 | +0.8777 | 0.8667 | 315.0000 | 0.9271 |
| cifar10 | iid | free_rider | main |  | 2 | 40 | flirds_mult | 0.6295 | +0.0176 | 0.0235 | +0.7500 | 0.3579 |  |  |
| cifar10 | iid | free_rider | main |  | 2 | 40 | flirds_zgate_v2 | 0.6138 | +0.0019 | 0.0235 | +0.0798 | 0.8333 |  |  |
| cifar10 | iid | free_rider | main |  | 2 | 40 | oracle_excl | 0.6354 | +0.0235 | 0.0235 | +1.0000 |  |  |  |
| cifar10 | iid | free_rider | main |  | 2 | 40 | random_excl | 0.5886 | -0.0232 | 0.0235 | -0.9894 |  |  |  |
| cifar10 | iid | free_rider | main |  | 2 | 40 | vanilla | 0.6119 | +0.0000 | 0.0235 | +0.0000 |  |  |  |
| cifar10 | iid | frrand | main |  | 0 | 40 | flirds_gate_v1 | 0.6195 | +0.0086 | 0.0239 | +0.3613 | 0.9833 | 172.0000 | 0.5795 |
| cifar10 | iid | frrand | main |  | 0 | 40 | flirds_gate_v2 | 0.6105 | -0.0004 | 0.0239 | -0.0157 | 0.8683 | 326.0000 | 0.8910 |
| cifar10 | iid | frrand | main |  | 0 | 40 | flirds_gatew_v1 | 0.6319 | +0.0210 | 0.0239 | +0.8796 | 1.0000 | 95.0000 | 0.6984 |
| cifar10 | iid | frrand | main |  | 0 | 40 | flirds_gatew_v2 | 0.6295 | +0.0186 | 0.0239 | +0.7801 | 0.9667 | 18.0000 | 0.9941 |
| cifar10 | iid | frrand | main |  | 0 | 40 | flirds_mult | 0.6278 | +0.0169 | 0.0239 | +0.7068 | 0.4629 |  |  |
| cifar10 | iid | frrand | main |  | 0 | 40 | flirds_zgate_v2 | 0.6062 | -0.0046 | 0.0239 | -0.1937 | 0.8762 |  |  |
| cifar10 | iid | frrand | main |  | 0 | 40 | oracle_excl | 0.6348 | +0.0239 | 0.0239 | +1.0000 |  |  |  |
| cifar10 | iid | frrand | main |  | 0 | 40 | random_excl | 0.6038 | -0.0071 | 0.0239 | -0.2984 |  |  |  |
| cifar10 | iid | frrand | main |  | 0 | 40 | vanilla | 0.6109 | +0.0000 | 0.0239 | +0.0000 |  |  |  |
| cifar10 | iid | frrand | main |  | 1 | 40 | flirds_gate_v1 | 0.6125 | +0.0110 | 0.0350 | +0.3143 | 0.9667 | 179.0000 | 0.5885 |
| cifar10 | iid | frrand | main |  | 1 | 40 | flirds_gate_v2 | 0.6318 | +0.0302 | 0.0350 | +0.8643 | 0.8337 | 395.0000 | 0.8911 |
| cifar10 | iid | frrand | main |  | 1 | 40 | flirds_gatew_v1 | 0.6334 | +0.0319 | 0.0350 | +0.9107 | 1.0000 | 104.0000 | 0.7135 |
| cifar10 | iid | frrand | main |  | 1 | 40 | flirds_gatew_v2 | 0.6324 | +0.0309 | 0.0350 | +0.8821 | 0.9558 | 96.0000 | 0.9707 |
| cifar10 | iid | frrand | main |  | 1 | 40 | flirds_mult | 0.6224 | +0.0209 | 0.0350 | +0.5964 | 0.3933 |  |  |
| cifar10 | iid | frrand | main |  | 1 | 40 | flirds_zgate_v2 | 0.6068 | +0.0052 | 0.0350 | +0.1500 | 0.9167 |  |  |
| cifar10 | iid | frrand | main |  | 1 | 40 | oracle_excl | 0.6365 | +0.0350 | 0.0350 | +1.0000 |  |  |  |
| cifar10 | iid | frrand | main |  | 1 | 40 | random_excl | 0.6051 | +0.0036 | 0.0350 | +0.1036 |  |  |  |
| cifar10 | iid | frrand | main |  | 1 | 40 | vanilla | 0.6015 | +0.0000 | 0.0350 | +0.0000 |  |  |  |
| cifar10 | iid | frrand | main |  | 2 | 40 | flirds_gate_v1 | 0.6179 | +0.0041 | 0.0207 | +0.1988 | 1.0000 | 187.0000 | 0.5750 |
| cifar10 | iid | frrand | main |  | 2 | 40 | flirds_gate_v2 | 0.6276 | +0.0139 | 0.0207 | +0.6687 | 0.9375 | 205.0000 | 0.9386 |
| cifar10 | iid | frrand | main |  | 2 | 40 | flirds_gatew_v1 | 0.6264 | +0.0126 | 0.0207 | +0.6084 | 1.0000 | 99.0000 | 0.7250 |
| cifar10 | iid | frrand | main |  | 2 | 40 | flirds_gatew_v2 | 0.6306 | +0.0169 | 0.0207 | +0.8133 | 0.9333 | 212.0000 | 0.9307 |
| cifar10 | iid | frrand | main |  | 2 | 40 | flirds_mult | 0.6276 | +0.0139 | 0.0207 | +0.6687 | 0.3937 |  |  |
| cifar10 | iid | frrand | main |  | 2 | 40 | flirds_zgate_v2 | 0.6120 | -0.0018 | 0.0207 | -0.0843 | 0.8329 |  |  |
| cifar10 | iid | frrand | main |  | 2 | 40 | oracle_excl | 0.6345 | +0.0207 | 0.0207 | +1.0000 |  |  |  |
| cifar10 | iid | frrand | main |  | 2 | 40 | random_excl | 0.5900 | -0.0238 | 0.0207 | -1.1446 |  |  |  |
| cifar10 | iid | frrand | main |  | 2 | 40 | vanilla | 0.6138 | +0.0000 | 0.0207 | +0.0000 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 0 | 40 | flirds_gate_v1 | 0.5836 | +0.3249 | 0.3759 | +0.8643 | 1.0000 | 207.0000 | 0.6929 |
| cifar10 | iid | grad_noise | main |  | 0 | 40 | flirds_gate_v2 | 0.6285 | +0.3697 | 0.3759 | +0.9837 | 1.0000 | 821.0000 | 0.8339 |
| cifar10 | iid | grad_noise | main |  | 0 | 40 | flirds_gatew_v1 | 0.5836 | +0.3249 | 0.3759 | +0.8643 | 1.0000 | 162.0000 | 0.7437 |
| cifar10 | iid | grad_noise | main |  | 0 | 40 | flirds_gatew_v2 | 0.6136 | +0.3549 | 0.3759 | +0.9441 | 1.0000 | 407.0000 | 0.9090 |
| cifar10 | iid | grad_noise | main |  | 0 | 40 | flirds_mult | 0.5519 | +0.2931 | 0.3759 | +0.7798 | 1.0000 |  |  |
| cifar10 | iid | grad_noise | main |  | 0 | 40 | flirds_zgate_v2 | 0.2970 | +0.0383 | 0.3759 | +0.1018 | 1.0000 |  |  |
| cifar10 | iid | grad_noise | main |  | 0 | 40 | oracle_excl | 0.6346 | +0.3759 | 0.3759 | +1.0000 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 0 | 40 | random_excl | 0.2655 | +0.0068 | 0.3759 | +0.0180 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 0 | 40 | vanilla | 0.2587 | +0.0000 | 0.3759 | +0.0000 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 1 | 40 | flirds_gate_v1 | 0.5747 | +0.3230 | 0.3851 | +0.8387 | 1.0000 | 198.0000 | 0.7036 |
| cifar10 | iid | grad_noise | main |  | 1 | 40 | flirds_gate_v2 | 0.5901 | +0.3384 | 0.3851 | +0.8786 | 1.0000 | 1161.0000 | 0.7747 |
| cifar10 | iid | grad_noise | main |  | 1 | 40 | flirds_gatew_v1 | 0.5775 | +0.3258 | 0.3851 | +0.8458 | 1.0000 | 191.0000 | 0.7110 |
| cifar10 | iid | grad_noise | main |  | 1 | 40 | flirds_gatew_v2 | 0.6152 | +0.3635 | 0.3851 | +0.9438 | 1.0000 | 712.0000 | 0.8482 |
| cifar10 | iid | grad_noise | main |  | 1 | 40 | flirds_mult | 0.5198 | +0.2680 | 0.3851 | +0.6959 | 1.0000 |  |  |
| cifar10 | iid | grad_noise | main |  | 1 | 40 | flirds_zgate_v2 | 0.3362 | +0.0845 | 0.3851 | +0.2194 | 1.0000 |  |  |
| cifar10 | iid | grad_noise | main |  | 1 | 40 | oracle_excl | 0.6369 | +0.3851 | 0.3851 | +1.0000 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 1 | 40 | random_excl | 0.2744 | +0.0226 | 0.3851 | +0.0587 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 1 | 40 | vanilla | 0.2517 | +0.0000 | 0.3851 | +0.0000 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 2 | 40 | flirds_gate_v1 | 0.5804 | +0.3217 | 0.3768 | +0.8540 | 1.0000 | 222.0000 | 0.6838 |
| cifar10 | iid | grad_noise | main |  | 2 | 40 | flirds_gate_v2 | 0.6244 | +0.3658 | 0.3768 | +0.9708 | 1.0000 | 833.0000 | 0.8288 |
| cifar10 | iid | grad_noise | main |  | 2 | 40 | flirds_gatew_v1 | 0.5704 | +0.3117 | 0.3768 | +0.8275 | 1.0000 | 176.0000 | 0.7337 |
| cifar10 | iid | grad_noise | main |  | 2 | 40 | flirds_gatew_v2 | 0.6265 | +0.3679 | 0.3768 | +0.9764 | 1.0000 | 542.0000 | 0.8800 |
| cifar10 | iid | grad_noise | main |  | 2 | 40 | flirds_mult | 0.5271 | +0.2685 | 0.3768 | +0.7127 | 1.0000 |  |  |
| cifar10 | iid | grad_noise | main |  | 2 | 40 | flirds_zgate_v2 | 0.3241 | +0.0655 | 0.3768 | +0.1739 | 1.0000 |  |  |
| cifar10 | iid | grad_noise | main |  | 2 | 40 | oracle_excl | 0.6354 | +0.3768 | 0.3768 | +1.0000 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 2 | 40 | random_excl | 0.2535 | -0.0051 | 0.3768 | -0.0136 |  |  |  |
| cifar10 | iid | grad_noise | main |  | 2 | 40 | vanilla | 0.2586 | +0.0000 | 0.3768 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | 39 | flirds_gate_v1 | 0.6076 | -0.0230 | -0.0014 |  | 0.6931 | 187.0000 | 0.4806 |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | 39 | flirds_gate_v1 | 0.6248 | +0.0299 | 0.0344 | +0.8691 | 0.9639 | 126.0000 | 0.7175 |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | 39 | flirds_gate_v1 | 0.5713 | +0.0333 | 0.0912 | +0.3644 | 1.0000 | 96.0000 | 0.8033 |
| cifar10 | iid | label_flip | main |  | 0 | 39 | flirds_gate_v1 | 0.5863 | +0.0460 | 0.0881 | +0.5220 | 1.0000 | 104.0000 | 0.7928 |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | 39 | flirds_gate_v2 | 0.6245 | -0.0061 | -0.0014 |  | 0.5750 | 301.0000 | 0.6707 |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | 39 | flirds_gate_v2 | 0.6105 | +0.0156 | 0.0344 | +0.4545 | 0.7650 | 187.0000 | 0.9408 |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | 39 | flirds_gate_v2 | 0.6032 | +0.0652 | 0.0912 | +0.7151 | 0.8394 | 210.0000 | 0.9438 |
| cifar10 | iid | label_flip | main |  | 0 | 39 | flirds_gate_v2 | 0.6126 | +0.0724 | 0.0881 | +0.8213 | 0.8394 | 291.0000 | 0.9228 |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | 39 | flirds_gatew_v1 | 0.6192 | -0.0114 | -0.0014 |  | 0.6301 | 165.0000 | 0.4572 |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | 39 | flirds_gatew_v1 | 0.6196 | +0.0247 | 0.0344 | +0.7200 | 0.9840 | 100.0000 | 0.7549 |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | 39 | flirds_gatew_v1 | 0.6054 | +0.0674 | 0.0912 | +0.7384 | 1.0000 | 81.0000 | 0.8298 |
| cifar10 | iid | label_flip | main |  | 0 | 39 | flirds_gatew_v1 | 0.6035 | +0.0633 | 0.0881 | +0.7177 | 1.0000 | 88.0000 | 0.8189 |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | 39 | flirds_gatew_v2 | 0.6181 | -0.0125 | -0.0014 |  | 0.3762 | 227.0000 | 0.6133 |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | 39 | flirds_gatew_v2 | 0.6079 | +0.0130 | 0.0344 | +0.3782 | 0.6053 | 278.0000 | 0.8864 |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | 39 | flirds_gatew_v2 | 0.6091 | +0.0711 | 0.0912 | +0.7795 | 0.8541 | 420.0000 | 0.8951 |
| cifar10 | iid | label_flip | main |  | 0 | 39 | flirds_gatew_v2 | 0.5731 | +0.0329 | 0.0881 | +0.3730 | 0.8310 | 383.0000 | 0.9004 |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | 39 | flirds_mult | 0.6256 | -0.0050 | -0.0014 |  | 0.3859 |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | 39 | flirds_mult | 0.6191 | +0.0242 | 0.0344 | +0.7055 | 0.7516 |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | 39 | flirds_mult | 0.6142 | +0.0762 | 0.0912 | +0.8356 | 0.9756 |  |  |
| cifar10 | iid | label_flip | main |  | 0 | 39 | flirds_mult | 0.6101 | +0.0699 | 0.0881 | +0.7929 | 0.9718 |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | 39 | flirds_zgate_v2 | 0.6216 | -0.0090 | -0.0014 |  | 0.6478 |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | 39 | flirds_zgate_v2 | 0.6049 | +0.0100 | 0.0344 | +0.2909 | 0.9954 |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | 39 | flirds_zgate_v2 | 0.5911 | +0.0531 | 0.0912 | +0.5822 | 1.0000 |  |  |
| cifar10 | iid | label_flip | main |  | 0 | 39 | flirds_zgate_v2 | 0.5707 | +0.0305 | 0.0881 | +0.3461 | 1.0000 |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | 39 | oracle_excl | 0.6292 | -0.0014 | -0.0014 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | 39 | oracle_excl | 0.6292 | +0.0344 | 0.0344 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | 39 | oracle_excl | 0.6292 | +0.0912 | 0.0912 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main |  | 0 | 39 | oracle_excl | 0.6284 | +0.0881 | 0.0881 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | 39 | random_excl | 0.6126 | -0.0180 | -0.0014 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | 39 | random_excl | 0.5962 | +0.0014 | 0.0344 | +0.0400 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | 39 | random_excl | 0.5610 | +0.0230 | 0.0912 | +0.2521 |  |  |  |
| cifar10 | iid | label_flip | main |  | 0 | 39 | random_excl | 0.5630 | +0.0227 | 0.0881 | +0.2582 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 0 | 39 | vanilla | 0.6306 | +0.0000 | -0.0014 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 0 | 39 | vanilla | 0.5949 | +0.0000 | 0.0344 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 0 | 39 | vanilla | 0.5380 | +0.0000 | 0.0912 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main |  | 0 | 39 | vanilla | 0.5403 | +0.0000 | 0.0881 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | 48 | flirds_gate_v1 | 0.6284 | -0.0020 | 0.0050 |  | 0.6274 | 163.0000 | 0.5688 |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | 48 | flirds_gate_v1 | 0.6071 | +0.0155 | 0.0438 | +0.3543 | 0.9984 | 92.0000 | 0.8083 |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | 48 | flirds_gate_v1 | 0.5773 | +0.0739 | 0.1320 | +0.5597 | 1.0000 | 87.0000 | 0.8466 |
| cifar10 | iid | label_flip | main |  | 1 | 48 | flirds_gate_v1 | 0.5645 | +0.0810 | 0.1503 | +0.5391 | 1.0000 | 83.0000 | 0.8536 |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | 48 | flirds_gate_v2 | 0.6225 | -0.0079 | 0.0050 |  | 0.6466 | 144.0000 | 0.9146 |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | 48 | flirds_gate_v2 | 0.6099 | +0.0182 | 0.0438 | +0.4171 | 0.8205 | 404.0000 | 0.9044 |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | 48 | flirds_gate_v2 | 0.5986 | +0.0952 | 0.1320 | +0.7216 | 0.8986 | 576.0000 | 0.8916 |
| cifar10 | iid | label_flip | main |  | 1 | 48 | flirds_gate_v2 | 0.6140 | +0.1305 | 0.1503 | +0.8686 | 0.9263 | 282.0000 | 0.9416 |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | 48 | flirds_gatew_v1 | 0.6270 | -0.0034 | 0.0050 |  | 0.6647 | 128.0000 | 0.5897 |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | 48 | flirds_gatew_v1 | 0.6034 | +0.0118 | 0.0438 | +0.2686 | 0.9972 | 75.0000 | 0.8315 |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | 48 | flirds_gatew_v1 | 0.5950 | +0.0916 | 0.1320 | +0.6941 | 1.0000 | 79.0000 | 0.8579 |
| cifar10 | iid | label_flip | main |  | 1 | 48 | flirds_gatew_v1 | 0.6054 | +0.1219 | 0.1503 | +0.8111 | 1.0000 | 76.0000 | 0.8640 |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | 48 | flirds_gatew_v2 | 0.6120 | -0.0184 | 0.0050 |  | 0.5008 | 223.0000 | 0.8079 |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | 48 | flirds_gatew_v2 | 0.6079 | +0.0163 | 0.0438 | +0.3714 | 0.7107 | 326.0000 | 0.9136 |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | 48 | flirds_gatew_v2 | 0.5965 | +0.0931 | 0.1320 | +0.7055 | 0.8774 | 427.0000 | 0.9139 |
| cifar10 | iid | label_flip | main |  | 1 | 48 | flirds_gatew_v2 | 0.5944 | +0.1109 | 0.1503 | +0.7379 | 0.8826 | 427.0000 | 0.9136 |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | 48 | flirds_mult | 0.6302 | -0.0001 | 0.0050 |  | 0.4002 |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | 48 | flirds_mult | 0.6162 | +0.0246 | 0.0438 | +0.5629 | 0.8393 |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | 48 | flirds_mult | 0.6020 | +0.0986 | 0.1320 | +0.7472 | 0.9872 |  |  |
| cifar10 | iid | label_flip | main |  | 1 | 48 | flirds_mult | 0.5970 | +0.1135 | 0.1503 | +0.7554 | 0.9876 |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | 48 | flirds_zgate_v2 | 0.6366 | +0.0062 | 0.0050 |  | 0.7304 |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | 48 | flirds_zgate_v2 | 0.5925 | +0.0009 | 0.0438 | +0.0200 | 1.0000 |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | 48 | flirds_zgate_v2 | 0.5209 | +0.0175 | 0.1320 | +0.1326 | 1.0000 |  |  |
| cifar10 | iid | label_flip | main |  | 1 | 48 | flirds_zgate_v2 | 0.5651 | +0.0816 | 0.1503 | +0.5433 | 1.0000 |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | 48 | oracle_excl | 0.6354 | +0.0050 | 0.0050 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | 48 | oracle_excl | 0.6354 | +0.0438 | 0.0438 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | 48 | oracle_excl | 0.6354 | +0.1320 | 0.1320 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main |  | 1 | 48 | oracle_excl | 0.6338 | +0.1503 | 0.1503 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | 48 | random_excl | 0.6032 | -0.0271 | 0.0050 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | 48 | random_excl | 0.5699 | -0.0217 | 0.0438 | -0.4971 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | 48 | random_excl | 0.4898 | -0.0136 | 0.1320 | -0.1032 |  |  |  |
| cifar10 | iid | label_flip | main |  | 1 | 48 | random_excl | 0.4830 | -0.0005 | 0.1503 | -0.0033 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 1 | 48 | vanilla | 0.6304 | +0.0000 | 0.0050 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 1 | 48 | vanilla | 0.5916 | +0.0000 | 0.0438 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 1 | 48 | vanilla | 0.5034 | +0.0000 | 0.1320 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main |  | 1 | 48 | vanilla | 0.4835 | +0.0000 | 0.1503 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | 47 | flirds_gate_v1 | 0.6126 | -0.0094 | 0.0062 |  | 0.7134 | 160.0000 | 0.5897 |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | 47 | flirds_gate_v1 | 0.6174 | +0.0271 | 0.0380 | +0.7138 | 0.9940 | 96.0000 | 0.7975 |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | 47 | flirds_gate_v1 | 0.6159 | +0.1059 | 0.1182 | +0.8953 | 1.0000 | 88.0000 | 0.8442 |
| cifar10 | iid | label_flip | main |  | 2 | 47 | flirds_gate_v1 | 0.6205 | +0.1145 | 0.1235 | +0.9271 | 1.0000 | 90.0000 | 0.8432 |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | 47 | flirds_gate_v2 | 0.6214 | -0.0006 | 0.0062 |  | 0.6499 | 282.0000 | 0.8239 |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | 47 | flirds_gate_v2 | 0.5870 | -0.0033 | 0.0380 | -0.0855 | 0.8475 | 374.0000 | 0.9122 |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | 47 | flirds_gate_v2 | 0.5883 | +0.0783 | 0.1182 | +0.6617 | 0.8900 | 550.0000 | 0.8884 |
| cifar10 | iid | label_flip | main |  | 2 | 47 | flirds_gate_v2 | 0.6021 | +0.0961 | 0.1235 | +0.7783 | 0.8876 | 331.0000 | 0.9315 |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | 47 | flirds_gatew_v1 | 0.6141 | -0.0079 | 0.0062 |  | 0.7102 | 123.0000 | 0.6083 |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | 47 | flirds_gatew_v1 | 0.6105 | +0.0202 | 0.0380 | +0.5329 | 0.9972 | 77.0000 | 0.8274 |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | 47 | flirds_gatew_v1 | 0.6185 | +0.1085 | 0.1182 | +0.9175 | 1.0000 | 69.0000 | 0.8752 |
| cifar10 | iid | label_flip | main |  | 2 | 47 | flirds_gatew_v1 | 0.6179 | +0.1119 | 0.1235 | +0.9059 | 1.0000 | 76.0000 | 0.8640 |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | 47 | flirds_gatew_v2 | 0.6109 | -0.0111 | 0.0062 |  | 0.4898 | 258.0000 | 0.7970 |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | 47 | flirds_gatew_v2 | 0.5817 | -0.0085 | 0.0380 | -0.2237 | 0.6877 | 503.0000 | 0.8777 |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | 47 | flirds_gatew_v2 | 0.6039 | +0.0939 | 0.1182 | +0.7939 | 0.8703 | 405.0000 | 0.9160 |
| cifar10 | iid | label_flip | main |  | 2 | 47 | flirds_gatew_v2 | 0.5815 | +0.0755 | 0.1235 | +0.6113 | 0.8627 | 311.0000 | 0.9348 |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | 47 | flirds_mult | 0.6170 | -0.0050 | 0.0062 |  | 0.3693 |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | 47 | flirds_mult | 0.6130 | +0.0227 | 0.0380 | +0.5987 | 0.9077 |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | 47 | flirds_mult | 0.6066 | +0.0966 | 0.1182 | +0.8171 | 0.9916 |  |  |
| cifar10 | iid | label_flip | main |  | 2 | 47 | flirds_mult | 0.5975 | +0.0915 | 0.1235 | +0.7409 | 0.9835 |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | 47 | flirds_zgate_v2 | 0.6175 | -0.0045 | 0.0062 |  | 0.6989 |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | 47 | flirds_zgate_v2 | 0.5959 | +0.0056 | 0.0380 | +0.1480 | 1.0000 |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | 47 | flirds_zgate_v2 | 0.5126 | +0.0026 | 0.1182 | +0.0222 | 1.0000 |  |  |
| cifar10 | iid | label_flip | main |  | 2 | 47 | flirds_zgate_v2 | 0.5434 | +0.0374 | 0.1235 | +0.3026 | 1.0000 |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | 47 | oracle_excl | 0.6282 | +0.0062 | 0.0062 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | 47 | oracle_excl | 0.6282 | +0.0380 | 0.0380 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | 47 | oracle_excl | 0.6282 | +0.1182 | 0.1182 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main |  | 2 | 47 | oracle_excl | 0.6295 | +0.1235 | 0.1235 | +1.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | 47 | random_excl | 0.5894 | -0.0326 | 0.0062 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | 47 | random_excl | 0.5507 | -0.0395 | 0.0380 | -1.0395 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | 47 | random_excl | 0.4561 | -0.0539 | 0.1182 | -0.4556 |  |  |  |
| cifar10 | iid | label_flip | main |  | 2 | 47 | random_excl | 0.4385 | -0.0675 | 0.1235 | -0.5466 |  |  |  |
| cifar10 | iid | label_flip | main | 0.1500 | 2 | 47 | vanilla | 0.6220 | +0.0000 | 0.0062 |  |  |  |  |
| cifar10 | iid | label_flip | main | 0.3500 | 2 | 47 | vanilla | 0.5903 | +0.0000 | 0.0380 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main | 0.7000 | 2 | 47 | vanilla | 0.5100 | +0.0000 | 0.1182 | +0.0000 |  |  |  |
| cifar10 | iid | label_flip | main |  | 2 | 47 | vanilla | 0.5060 | +0.0000 | 0.1235 | +0.0000 |  |  |  |
| cifar10 | qskew | clean | main |  | 0 | 0 | flirds_gate_v1 | 0.6614 | -0.0076 |  |  |  | 358.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 0 | 0 | flirds_gate_v2 | 0.6591 | -0.0099 |  |  |  | 1703.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 0 | 0 | flirds_gatew_v1 | 0.6570 | -0.0120 |  |  |  | 276.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 0 | 0 | flirds_gatew_v2 | 0.6561 | -0.0129 |  |  |  | 1254.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 0 | 0 | flirds_mult | 0.6637 | -0.0053 |  |  |  |  |  |
| cifar10 | qskew | clean | main |  | 0 | 0 | flirds_zgate_v2 | 0.6666 | -0.0024 |  |  |  |  |  |
| cifar10 | qskew | clean | main |  | 0 | 0 | vanilla | 0.6690 | +0.0000 |  |  |  |  |  |
| cifar10 | qskew | clean | main |  | 1 | 0 | flirds_gate_v1 | 0.6585 | -0.0099 |  |  |  | 368.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 1 | 0 | flirds_gate_v2 | 0.6408 | -0.0276 |  |  |  | 1772.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 1 | 0 | flirds_gatew_v1 | 0.6623 | -0.0061 |  |  |  | 304.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 1 | 0 | flirds_gatew_v2 | 0.6647 | -0.0036 |  |  |  | 1003.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 1 | 0 | flirds_mult | 0.6680 | -0.0004 |  |  |  |  |  |
| cifar10 | qskew | clean | main |  | 1 | 0 | flirds_zgate_v2 | 0.6636 | -0.0048 |  |  |  |  |  |
| cifar10 | qskew | clean | main |  | 1 | 0 | vanilla | 0.6684 | +0.0000 |  |  |  |  |  |
| cifar10 | qskew | clean | main |  | 2 | 0 | flirds_gate_v1 | 0.6538 | -0.0065 |  |  |  | 416.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 2 | 0 | flirds_gate_v2 | 0.6348 | -0.0255 |  |  |  | 2437.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 2 | 0 | flirds_gatew_v1 | 0.6486 | -0.0116 |  |  |  | 323.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 2 | 0 | flirds_gatew_v2 | 0.6462 | -0.0140 |  |  |  | 1623.0000 | 0.0000 |
| cifar10 | qskew | clean | main |  | 2 | 0 | flirds_mult | 0.6591 | -0.0011 |  |  |  |  |  |
| cifar10 | qskew | clean | main |  | 2 | 0 | flirds_zgate_v2 | 0.6600 | -0.0002 |  |  |  |  |  |
| cifar10 | qskew | clean | main |  | 2 | 0 | vanilla | 0.6603 | +0.0000 |  |  |  |  |  |
| cifar10 | qskew | free_rider | main |  | 0 | 40 | flirds_gate_v1 | 0.6472 | +0.0241 | 0.0201 | +1.1988 | 0.9667 | 125.0000 | 0.7934 |
| cifar10 | qskew | free_rider | main |  | 0 | 40 | flirds_gate_v2 | 0.6316 | +0.0085 | 0.0201 | +0.4224 | 0.6833 | 1306.0000 | 0.7590 |
| cifar10 | qskew | free_rider | main |  | 0 | 40 | flirds_gatew_v1 | 0.6406 | +0.0175 | 0.0201 | +0.8696 | 0.9833 | 95.0000 | 0.8348 |
| cifar10 | qskew | free_rider | main |  | 0 | 40 | flirds_gatew_v2 | 0.6312 | +0.0081 | 0.0201 | +0.4037 | 0.6500 | 1123.0000 | 0.7855 |
| cifar10 | qskew | free_rider | main |  | 0 | 40 | flirds_mult | 0.6366 | +0.0135 | 0.0201 | +0.6708 | 0.4325 |  |  |
| cifar10 | qskew | free_rider | main |  | 0 | 40 | flirds_zgate_v2 | 0.6202 | -0.0029 | 0.0201 | -0.1429 | 0.8333 |  |  |
| cifar10 | qskew | free_rider | main |  | 0 | 40 | oracle_excl | 0.6432 | +0.0201 | 0.0201 | +1.0000 |  |  |  |
| cifar10 | qskew | free_rider | main |  | 0 | 40 | random_excl | 0.6128 | -0.0104 | 0.0201 | -0.5155 |  |  |  |
| cifar10 | qskew | free_rider | main |  | 0 | 40 | vanilla | 0.6231 | +0.0000 | 0.0201 | +0.0000 |  |  |  |
| cifar10 | qskew | free_rider | main |  | 1 | 40 | flirds_gate_v1 | 0.6489 | +0.0080 | 0.0141 |  | 0.9667 | 143.0000 | 0.7701 |
| cifar10 | qskew | free_rider | main |  | 1 | 40 | flirds_gate_v2 | 0.6465 | +0.0056 | 0.0141 |  | 0.7333 | 878.0000 | 0.8187 |
| cifar10 | qskew | free_rider | main |  | 1 | 40 | flirds_gatew_v1 | 0.6522 | +0.0114 | 0.0141 |  | 0.9833 | 107.0000 | 0.8174 |
| cifar10 | qskew | free_rider | main |  | 1 | 40 | flirds_gatew_v2 | 0.6445 | +0.0036 | 0.0141 |  | 0.6167 | 834.0000 | 0.8258 |
| cifar10 | qskew | free_rider | main |  | 1 | 40 | flirds_mult | 0.6434 | +0.0025 | 0.0141 |  | 0.3479 |  |  |
| cifar10 | qskew | free_rider | main |  | 1 | 40 | flirds_zgate_v2 | 0.6331 | -0.0077 | 0.0141 |  | 0.7333 |  |  |
| cifar10 | qskew | free_rider | main |  | 1 | 40 | oracle_excl | 0.6550 | +0.0141 | 0.0141 |  |  |  |  |
| cifar10 | qskew | free_rider | main |  | 1 | 40 | random_excl | 0.6306 | -0.0102 | 0.0141 |  |  |  |  |
| cifar10 | qskew | free_rider | main |  | 1 | 40 | vanilla | 0.6409 | +0.0000 | 0.0141 |  |  |  |  |
| cifar10 | qskew | free_rider | main |  | 2 | 40 | flirds_gate_v1 | 0.6365 | +0.0120 | 0.0152 |  | 0.9333 | 139.0000 | 0.7790 |
| cifar10 | qskew | free_rider | main |  | 2 | 40 | flirds_gate_v2 | 0.6238 | -0.0008 | 0.0152 |  | 0.7667 | 787.0000 | 0.8354 |
| cifar10 | qskew | free_rider | main |  | 2 | 40 | flirds_gatew_v1 | 0.6411 | +0.0166 | 0.0152 |  | 0.9500 | 113.0000 | 0.8126 |
| cifar10 | qskew | free_rider | main |  | 2 | 40 | flirds_gatew_v2 | 0.6236 | -0.0009 | 0.0152 |  | 0.5167 | 1371.0000 | 0.7419 |
| cifar10 | qskew | free_rider | main |  | 2 | 40 | flirds_mult | 0.6349 | +0.0104 | 0.0152 |  | 0.3704 |  |  |
| cifar10 | qskew | free_rider | main |  | 2 | 40 | flirds_zgate_v2 | 0.6204 | -0.0041 | 0.0152 |  | 0.6833 |  |  |
| cifar10 | qskew | free_rider | main |  | 2 | 40 | oracle_excl | 0.6398 | +0.0152 | 0.0152 |  |  |  |  |
| cifar10 | qskew | free_rider | main |  | 2 | 40 | random_excl | 0.6035 | -0.0210 | 0.0152 |  |  |  |  |
| cifar10 | qskew | free_rider | main |  | 2 | 40 | vanilla | 0.6245 | +0.0000 | 0.0152 |  |  |  |  |
| cifar10 | qskew | frrand | main |  | 0 | 40 | flirds_gate_v1 | 0.6284 | +0.0045 | 0.0194 |  | 0.8667 | 205.0000 | 0.5341 |
| cifar10 | qskew | frrand | main |  | 0 | 40 | flirds_gate_v2 | 0.6240 | +0.0001 | 0.0194 |  | 0.8071 | 828.0000 | 0.7943 |
| cifar10 | qskew | frrand | main |  | 0 | 40 | flirds_gatew_v1 | 0.6309 | +0.0070 | 0.0194 |  | 0.9667 | 105.0000 | 0.6866 |
| cifar10 | qskew | frrand | main |  | 0 | 40 | flirds_gatew_v2 | 0.6341 | +0.0103 | 0.0194 |  | 0.7812 | 622.0000 | 0.8248 |
| cifar10 | qskew | frrand | main |  | 0 | 40 | flirds_mult | 0.6358 | +0.0119 | 0.0194 |  | 0.4383 |  |  |
| cifar10 | qskew | frrand | main |  | 0 | 40 | flirds_zgate_v2 | 0.6215 | -0.0024 | 0.0194 |  | 0.8358 |  |  |
| cifar10 | qskew | frrand | main |  | 0 | 40 | oracle_excl | 0.6432 | +0.0194 | 0.0194 |  |  |  |  |
| cifar10 | qskew | frrand | main |  | 0 | 40 | random_excl | 0.6118 | -0.0121 | 0.0194 |  |  |  |  |
| cifar10 | qskew | frrand | main |  | 0 | 40 | vanilla | 0.6239 | +0.0000 | 0.0194 |  |  |  |  |
| cifar10 | qskew | frrand | main |  | 1 | 40 | flirds_gate_v1 | 0.6386 | -0.0025 | 0.0139 |  | 0.9167 | 210.0000 | 0.5588 |
| cifar10 | qskew | frrand | main |  | 1 | 40 | flirds_gate_v2 | 0.6434 | +0.0023 | 0.0139 |  | 0.7367 | 1000.0000 | 0.7431 |
| cifar10 | qskew | frrand | main |  | 1 | 40 | flirds_gatew_v1 | 0.6514 | +0.0103 | 0.0139 |  | 0.9667 | 106.0000 | 0.7135 |
| cifar10 | qskew | frrand | main |  | 1 | 40 | flirds_gatew_v2 | 0.6438 | +0.0026 | 0.0139 |  | 0.7542 | 579.0000 | 0.8506 |
| cifar10 | qskew | frrand | main |  | 1 | 40 | flirds_mult | 0.6442 | +0.0031 | 0.0139 |  | 0.3538 |  |  |
| cifar10 | qskew | frrand | main |  | 1 | 40 | flirds_zgate_v2 | 0.6310 | -0.0101 | 0.0139 |  | 0.7442 |  |  |
| cifar10 | qskew | frrand | main |  | 1 | 40 | oracle_excl | 0.6550 | +0.0139 | 0.0139 |  |  |  |  |
| cifar10 | qskew | frrand | main |  | 1 | 40 | random_excl | 0.6300 | -0.0111 | 0.0139 |  |  |  |  |
| cifar10 | qskew | frrand | main |  | 1 | 40 | vanilla | 0.6411 | +0.0000 | 0.0139 |  |  |  |  |
| cifar10 | qskew | frrand | main |  | 2 | 40 | flirds_gate_v1 | 0.6272 | +0.0012 | 0.0138 |  | 0.8662 | 227.0000 | 0.5191 |
| cifar10 | qskew | frrand | main |  | 2 | 40 | flirds_gate_v2 | 0.6314 | +0.0054 | 0.0138 |  | 0.7017 | 1316.0000 | 0.7016 |
| cifar10 | qskew | frrand | main |  | 2 | 40 | flirds_gatew_v1 | 0.6495 | +0.0235 | 0.0138 |  | 0.9667 | 105.0000 | 0.7099 |
| cifar10 | qskew | frrand | main |  | 2 | 40 | flirds_gatew_v2 | 0.6262 | +0.0002 | 0.0138 |  | 0.6850 | 792.0000 | 0.8003 |
| cifar10 | qskew | frrand | main |  | 2 | 40 | flirds_mult | 0.6346 | +0.0086 | 0.0138 |  | 0.3533 |  |  |
| cifar10 | qskew | frrand | main |  | 2 | 40 | flirds_zgate_v2 | 0.6211 | -0.0049 | 0.0138 |  | 0.7492 |  |  |
| cifar10 | qskew | frrand | main |  | 2 | 40 | oracle_excl | 0.6398 | +0.0138 | 0.0138 |  |  |  |  |
| cifar10 | qskew | frrand | main |  | 2 | 40 | random_excl | 0.6026 | -0.0234 | 0.0138 |  |  |  |  |
| cifar10 | qskew | frrand | main |  | 2 | 40 | vanilla | 0.6260 | +0.0000 | 0.0138 |  |  |  |  |
| cifar10 | qskew | grad_noise | main |  | 0 | 40 | flirds_gate_v1 | 0.5441 | +0.2916 | 0.3907 | +0.7463 | 0.9958 | 197.0000 | 0.6965 |
| cifar10 | qskew | grad_noise | main |  | 0 | 40 | flirds_gate_v2 | 0.6366 | +0.3841 | 0.3907 | +0.9830 | 0.9879 | 1273.0000 | 0.7608 |
| cifar10 | qskew | grad_noise | main |  | 0 | 40 | flirds_gatew_v1 | 0.2039 | -0.0486 | 0.3907 | -0.1244 | 0.9067 | 278.0000 | 0.6166 |
| cifar10 | qskew | grad_noise | main |  | 0 | 40 | flirds_gatew_v2 | 0.6304 | +0.3779 | 0.3907 | +0.9671 | 0.9871 | 888.0000 | 0.8190 |
| cifar10 | qskew | grad_noise | main |  | 0 | 40 | flirds_mult | 0.4871 | +0.2346 | 0.3907 | +0.6004 | 0.9446 |  |  |
| cifar10 | qskew | grad_noise | main |  | 0 | 40 | flirds_zgate_v2 | 0.4339 | +0.1814 | 0.3907 | +0.4642 | 0.9833 |  |  |
| cifar10 | qskew | grad_noise | main |  | 0 | 40 | oracle_excl | 0.6432 | +0.3907 | 0.3907 | +1.0000 |  |  |  |
| cifar10 | qskew | grad_noise | main |  | 0 | 40 | random_excl | 0.2556 | +0.0031 | 0.3907 | +0.0080 |  |  |  |
| cifar10 | qskew | grad_noise | main |  | 0 | 40 | vanilla | 0.2525 | +0.0000 | 0.3907 | +0.0000 |  |  |  |
| cifar10 | qskew | grad_noise | main |  | 1 | 40 | flirds_gate_v1 | 0.5920 | +0.3082 | 0.3713 | +0.8303 | 0.9946 | 223.0000 | 0.6667 |
| cifar10 | qskew | grad_noise | main |  | 1 | 40 | flirds_gate_v2 | 0.6296 | +0.3459 | 0.3713 | +0.9316 | 0.9842 | 1293.0000 | 0.7600 |
| cifar10 | qskew | grad_noise | main |  | 1 | 40 | flirds_gatew_v1 | 0.2627 | -0.0210 | 0.3713 | -0.0566 | 0.8275 | 299.0000 | 0.5943 |
| cifar10 | qskew | grad_noise | main |  | 1 | 40 | flirds_gatew_v2 | 0.6041 | +0.3204 | 0.3713 | +0.8630 | 0.9983 | 472.0000 | 0.8926 |
| cifar10 | qskew | grad_noise | main |  | 1 | 40 | flirds_mult | 0.5079 | +0.2241 | 0.3713 | +0.6037 | 0.9471 |  |  |
| cifar10 | qskew | grad_noise | main |  | 1 | 40 | flirds_zgate_v2 | 0.4959 | +0.2121 | 0.3713 | +0.5714 | 0.9862 |  |  |
| cifar10 | qskew | grad_noise | main |  | 1 | 40 | oracle_excl | 0.6550 | +0.3713 | 0.3713 | +1.0000 |  |  |  |
| cifar10 | qskew | grad_noise | main |  | 1 | 40 | random_excl | 0.3361 | +0.0524 | 0.3713 | +0.1411 |  |  |  |
| cifar10 | qskew | grad_noise | main |  | 1 | 40 | vanilla | 0.2838 | +0.0000 | 0.3713 | +0.0000 |  |  |  |
| cifar10 | qskew | grad_noise | main |  | 2 | 40 | flirds_gate_v1 | 0.3212 | +0.0551 | 0.3736 | +0.1475 | 0.9304 | 263.0000 | 0.6301 |
| cifar10 | qskew | grad_noise | main |  | 2 | 40 | flirds_gate_v2 | 0.6236 | +0.3575 | 0.3736 | +0.9568 | 0.9863 | 1219.0000 | 0.7629 |
| cifar10 | qskew | grad_noise | main |  | 2 | 40 | flirds_gatew_v1 | 0.2724 | +0.0062 | 0.3736 | +0.0167 | 0.7287 | 323.0000 | 0.5864 |
| cifar10 | qskew | grad_noise | main |  | 2 | 40 | flirds_gatew_v2 | 0.5629 | +0.2968 | 0.3736 | +0.7942 | 0.8783 | 3510.0000 | 0.5346 |
| cifar10 | qskew | grad_noise | main |  | 2 | 40 | flirds_mult | 0.4951 | +0.2290 | 0.3736 | +0.6129 | 0.9471 |  |  |
| cifar10 | qskew | grad_noise | main |  | 2 | 40 | flirds_zgate_v2 | 0.4789 | +0.2127 | 0.3736 | +0.5694 | 0.9817 |  |  |
| cifar10 | qskew | grad_noise | main |  | 2 | 40 | oracle_excl | 0.6398 | +0.3736 | 0.3736 | +1.0000 |  |  |  |
| cifar10 | qskew | grad_noise | main |  | 2 | 40 | random_excl | 0.2445 | -0.0216 | 0.3736 | -0.0579 |  |  |  |
| cifar10 | qskew | grad_noise | main |  | 2 | 40 | vanilla | 0.2661 | +0.0000 | 0.3736 | +0.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | 39 | flirds_gate_v1 | 0.6325 | -0.0081 | 0.0082 |  | 0.5523 | 205.0000 | 0.4225 |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | 39 | flirds_gate_v1 | 0.6176 | +0.0106 | 0.0419 | +0.2537 | 0.8108 | 152.0000 | 0.6415 |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | 39 | flirds_gate_v1 | 0.5827 | +0.0241 | 0.0902 | +0.2673 | 0.9706 | 126.0000 | 0.7353 |
| cifar10 | qskew | label_flip | main |  | 0 | 39 | flirds_gate_v1 | 0.6170 | +0.0501 | 0.0820 | +0.6113 | 0.9723 | 127.0000 | 0.7354 |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | 39 | flirds_gate_v2 | 0.6298 | -0.0109 | 0.0082 |  | 0.4687 | 941.0000 | 0.5300 |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | 39 | flirds_gate_v2 | 0.6101 | +0.0031 | 0.0419 | +0.0746 | 0.7175 | 715.0000 | 0.7950 |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | 39 | flirds_gate_v2 | 0.5969 | +0.0383 | 0.0902 | +0.4238 | 0.7945 | 1150.0000 | 0.7431 |
| cifar10 | qskew | label_flip | main |  | 0 | 39 | flirds_gate_v2 | 0.6076 | +0.0407 | 0.0820 | +0.4970 | 0.7936 | 856.0000 | 0.7915 |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | 39 | flirds_gatew_v1 | 0.6404 | -0.0002 | 0.0082 |  | 0.5322 | 157.0000 | 0.3938 |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | 39 | flirds_gatew_v1 | 0.6196 | +0.0126 | 0.0419 | +0.3015 | 0.8087 | 115.0000 | 0.6637 |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | 39 | flirds_gatew_v1 | 0.5374 | -0.0212 | 0.0902 | -0.2355 | 0.9739 | 111.0000 | 0.7528 |
| cifar10 | qskew | label_flip | main |  | 0 | 39 | flirds_gatew_v1 | 0.5008 | -0.0661 | 0.0820 | -0.8064 | 0.9765 | 99.0000 | 0.7708 |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | 39 | flirds_gatew_v2 | 0.6391 | -0.0015 | 0.0082 |  | 0.3775 | 580.0000 | 0.4821 |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | 39 | flirds_gatew_v2 | 0.6171 | +0.0101 | 0.0419 | +0.2418 | 0.5221 | 977.0000 | 0.6503 |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | 39 | flirds_gatew_v2 | 0.5823 | +0.0236 | 0.0902 | +0.2618 | 0.6675 | 940.0000 | 0.7573 |
| cifar10 | qskew | label_flip | main |  | 0 | 39 | flirds_gatew_v2 | 0.6184 | +0.0515 | 0.0820 | +0.6280 | 0.7213 | 862.0000 | 0.7772 |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | 39 | flirds_mult | 0.6374 | -0.0032 | 0.0082 |  | 0.2459 |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | 39 | flirds_mult | 0.6248 | +0.0178 | 0.0419 | +0.4239 | 0.5628 |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | 39 | flirds_mult | 0.6252 | +0.0666 | 0.0902 | +0.7382 | 0.7907 |  |  |
| cifar10 | qskew | label_flip | main |  | 0 | 39 | flirds_mult | 0.6241 | +0.0573 | 0.0820 | +0.6982 | 0.7978 |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | 39 | flirds_zgate_v2 | 0.6430 | +0.0024 | 0.0082 |  | 0.4212 |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | 39 | flirds_zgate_v2 | 0.6251 | +0.0181 | 0.0419 | +0.4328 | 0.9252 |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | 39 | flirds_zgate_v2 | 0.5995 | +0.0409 | 0.0902 | +0.4529 | 0.9878 |  |  |
| cifar10 | qskew | label_flip | main |  | 0 | 39 | flirds_zgate_v2 | 0.6111 | +0.0443 | 0.0820 | +0.5396 | 0.9811 |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | 39 | oracle_excl | 0.6489 | +0.0082 | 0.0082 |  |  |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | 39 | oracle_excl | 0.6489 | +0.0419 | 0.0419 | +1.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | 39 | oracle_excl | 0.6489 | +0.0902 | 0.0902 | +1.0000 |  |  |  |
| cifar10 | qskew | label_flip | main |  | 0 | 39 | oracle_excl | 0.6489 | +0.0820 | 0.0820 | +1.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | 39 | random_excl | 0.6378 | -0.0029 | 0.0082 |  |  |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | 39 | random_excl | 0.6191 | +0.0121 | 0.0419 | +0.2896 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | 39 | random_excl | 0.5891 | +0.0305 | 0.0902 | +0.3380 |  |  |  |
| cifar10 | qskew | label_flip | main |  | 0 | 39 | random_excl | 0.5875 | +0.0206 | 0.0820 | +0.2515 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 0 | 39 | vanilla | 0.6406 | +0.0000 | 0.0082 |  |  |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 0 | 39 | vanilla | 0.6070 | +0.0000 | 0.0419 | +0.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 0 | 39 | vanilla | 0.5586 | +0.0000 | 0.0902 | +0.0000 |  |  |  |
| cifar10 | qskew | label_flip | main |  | 0 | 39 | vanilla | 0.5669 | +0.0000 | 0.0820 | +0.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 1 | 48 | flirds_gate_v1 | 0.6336 | -0.0092 | 0.0059 |  | 0.5565 | 195.0000 | 0.4460 |
| cifar10 | qskew | label_flip | main | 0.3500 | 1 | 48 | flirds_gate_v1 | 0.5994 | -0.0016 | 0.0478 | -0.0340 | 0.8061 | 149.0000 | 0.6621 |
| cifar10 | qskew | label_flip | main | 0.7000 | 1 | 48 | flirds_gate_v1 | 0.5891 | +0.0815 | 0.1411 | +0.5775 | 0.9808 | 125.0000 | 0.7646 |
| cifar10 | qskew | label_flip | main |  | 1 | 48 | flirds_gate_v1 | 0.5215 | +0.0612 | 0.1885 | +0.3249 | 0.9724 | 127.0000 | 0.7695 |
| cifar10 | qskew | label_flip | main | 0.1500 | 1 | 48 | flirds_gate_v2 | 0.6430 | +0.0001 | 0.0059 |  | 0.5397 | 731.0000 | 0.6341 |
| cifar10 | qskew | label_flip | main | 0.3500 | 1 | 48 | flirds_gate_v2 | 0.6312 | +0.0302 | 0.0478 | +0.6335 | 0.6751 | 663.0000 | 0.8335 |
| cifar10 | qskew | label_flip | main | 0.7000 | 1 | 48 | flirds_gate_v2 | 0.6239 | +0.1162 | 0.1411 | +0.8237 | 0.8165 | 775.0000 | 0.8424 |
| cifar10 | qskew | label_flip | main |  | 1 | 48 | flirds_gate_v2 | 0.4743 | +0.0140 | 0.1885 | +0.0743 | 0.8229 | 926.0000 | 0.8222 |
| cifar10 | qskew | label_flip | main | 0.1500 | 1 | 48 | flirds_gatew_v1 | 0.6294 | -0.0135 | 0.0059 |  | 0.5805 | 161.0000 | 0.4371 |
| cifar10 | qskew | label_flip | main | 0.3500 | 1 | 48 | flirds_gatew_v1 | 0.6121 | +0.0111 | 0.0478 | +0.2330 | 0.8626 | 123.0000 | 0.6694 |
| cifar10 | qskew | label_flip | main | 0.7000 | 1 | 48 | flirds_gatew_v1 | 0.6016 | +0.0940 | 0.1411 | +0.6661 | 0.9651 | 119.0000 | 0.7644 |
| cifar10 | qskew | label_flip | main |  | 1 | 48 | flirds_gatew_v1 | 0.6151 | +0.1549 | 0.1885 | +0.8216 | 0.9864 | 110.0000 | 0.7826 |
| cifar10 | qskew | label_flip | main | 0.1500 | 1 | 48 | flirds_gatew_v2 | 0.6251 | -0.0177 | 0.0059 |  | 0.3618 | 722.0000 | 0.5378 |
| cifar10 | qskew | label_flip | main | 0.3500 | 1 | 48 | flirds_gatew_v2 | 0.6170 | +0.0160 | 0.0478 | +0.3351 | 0.5725 | 627.0000 | 0.8180 |
| cifar10 | qskew | label_flip | main | 0.7000 | 1 | 48 | flirds_gatew_v2 | 0.6042 | +0.0966 | 0.1411 | +0.6847 | 0.6442 | 1143.0000 | 0.7727 |
| cifar10 | qskew | label_flip | main |  | 1 | 48 | flirds_gatew_v2 | 0.5966 | +0.1364 | 0.1885 | +0.7235 | 0.6571 | 1589.0000 | 0.7178 |
| cifar10 | qskew | label_flip | main | 0.1500 | 1 | 48 | flirds_mult | 0.6421 | -0.0008 | 0.0059 |  | 0.2432 |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 1 | 48 | flirds_mult | 0.6182 | +0.0172 | 0.0478 | +0.3613 | 0.4692 |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 1 | 48 | flirds_mult | 0.5769 | +0.0693 | 0.1411 | +0.4907 | 0.7528 |  |  |
| cifar10 | qskew | label_flip | main |  | 1 | 48 | flirds_mult | 0.5734 | +0.1131 | 0.1885 | +0.6001 | 0.7556 |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 1 | 48 | flirds_zgate_v2 | 0.6390 | -0.0039 | 0.0059 |  | 0.5040 |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 1 | 48 | flirds_zgate_v2 | 0.6246 | +0.0236 | 0.0478 | +0.4948 | 0.9431 |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 1 | 48 | flirds_zgate_v2 | 0.5972 | +0.0896 | 0.1411 | +0.6351 | 0.9868 |  |  |
| cifar10 | qskew | label_flip | main |  | 1 | 48 | flirds_zgate_v2 | 0.5938 | +0.1335 | 0.1885 | +0.7082 | 0.9900 |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 1 | 48 | oracle_excl | 0.6488 | +0.0059 | 0.0059 |  |  |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 1 | 48 | oracle_excl | 0.6488 | +0.0478 | 0.0478 | +1.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 1 | 48 | oracle_excl | 0.6488 | +0.1411 | 0.1411 | +1.0000 |  |  |  |
| cifar10 | qskew | label_flip | main |  | 1 | 48 | oracle_excl | 0.6488 | +0.1885 | 0.1885 | +1.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 1 | 48 | random_excl | 0.6138 | -0.0291 | 0.0059 |  |  |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 1 | 48 | random_excl | 0.5824 | -0.0186 | 0.0478 | -0.3901 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 1 | 48 | random_excl | 0.5162 | +0.0086 | 0.1411 | +0.0611 |  |  |  |
| cifar10 | qskew | label_flip | main |  | 1 | 48 | random_excl | 0.5114 | +0.0511 | 0.1885 | +0.2712 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 1 | 48 | vanilla | 0.6429 | +0.0000 | 0.0059 |  |  |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 1 | 48 | vanilla | 0.6010 | +0.0000 | 0.0478 | +0.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 1 | 48 | vanilla | 0.5076 | +0.0000 | 0.1411 | +0.0000 |  |  |  |
| cifar10 | qskew | label_flip | main |  | 1 | 48 | vanilla | 0.4602 | +0.0000 | 0.1885 | +0.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 2 | 47 | flirds_gate_v1 | 0.6070 | -0.0220 | 0.0015 |  | 0.6238 | 217.0000 | 0.4694 |
| cifar10 | qskew | label_flip | main | 0.3500 | 2 | 47 | flirds_gate_v1 | 0.6139 | +0.0300 | 0.0466 | +0.6434 | 0.8784 | 151.0000 | 0.6760 |
| cifar10 | qskew | label_flip | main | 0.7000 | 2 | 47 | flirds_gate_v1 | 0.6112 | +0.1305 | 0.1497 | +0.8715 | 0.9595 | 115.0000 | 0.7866 |
| cifar10 | qskew | label_flip | main |  | 2 | 47 | flirds_gate_v1 | 0.6080 | +0.1509 | 0.1734 | +0.8702 | 0.9691 | 114.0000 | 0.7841 |
| cifar10 | qskew | label_flip | main | 0.1500 | 2 | 47 | flirds_gate_v2 | 0.6104 | -0.0186 | 0.0015 |  | 0.6090 | 927.0000 | 0.6494 |
| cifar10 | qskew | label_flip | main | 0.3500 | 2 | 47 | flirds_gate_v2 | 0.5383 | -0.0456 | 0.0466 | -0.9786 | 0.7025 | 836.0000 | 0.7963 |
| cifar10 | qskew | label_flip | main | 0.7000 | 2 | 47 | flirds_gate_v2 | 0.5876 | +0.1069 | 0.1497 | +0.7137 | 0.7997 | 1032.0000 | 0.7968 |
| cifar10 | qskew | label_flip | main |  | 2 | 47 | flirds_gate_v2 | 0.5620 | +0.1049 | 0.1734 | +0.6049 | 0.7704 | 944.0000 | 0.8132 |
| cifar10 | qskew | label_flip | main | 0.1500 | 2 | 47 | flirds_gatew_v1 | 0.6210 | -0.0080 | 0.0015 |  | 0.5817 | 167.0000 | 0.4452 |
| cifar10 | qskew | label_flip | main | 0.3500 | 2 | 47 | flirds_gatew_v1 | 0.5811 | -0.0028 | 0.0466 | -0.0590 | 0.8346 | 122.0000 | 0.6648 |
| cifar10 | qskew | label_flip | main | 0.7000 | 2 | 47 | flirds_gatew_v1 | 0.5995 | +0.1188 | 0.1497 | +0.7930 | 0.9763 | 96.0000 | 0.8000 |
| cifar10 | qskew | label_flip | main |  | 2 | 47 | flirds_gatew_v1 | 0.5952 | +0.1381 | 0.1734 | +0.7967 | 0.9675 | 100.0000 | 0.7951 |
| cifar10 | qskew | label_flip | main | 0.1500 | 2 | 47 | flirds_gatew_v2 | 0.6074 | -0.0216 | 0.0015 |  | 0.3404 | 878.0000 | 0.4478 |
| cifar10 | qskew | label_flip | main | 0.3500 | 2 | 47 | flirds_gatew_v2 | 0.5859 | +0.0020 | 0.0466 | +0.0429 | 0.4922 | 952.0000 | 0.7182 |
| cifar10 | qskew | label_flip | main | 0.7000 | 2 | 47 | flirds_gatew_v2 | 0.5696 | +0.0889 | 0.1497 | +0.5935 | 0.6632 | 1599.0000 | 0.7066 |
| cifar10 | qskew | label_flip | main |  | 2 | 47 | flirds_gatew_v2 | 0.5726 | +0.1155 | 0.1734 | +0.6662 | 0.6728 | 1645.0000 | 0.6868 |
| cifar10 | qskew | label_flip | main | 0.1500 | 2 | 47 | flirds_mult | 0.6279 | -0.0011 | 0.0015 |  | 0.3918 |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 2 | 47 | flirds_mult | 0.6098 | +0.0259 | 0.0466 | +0.5550 | 0.6403 |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 2 | 47 | flirds_mult | 0.5891 | +0.1084 | 0.1497 | +0.7237 | 0.8888 |  |  |
| cifar10 | qskew | label_flip | main |  | 2 | 47 | flirds_mult | 0.5849 | +0.1278 | 0.1734 | +0.7368 | 0.8920 |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 2 | 47 | flirds_zgate_v2 | 0.6310 | +0.0020 | 0.0015 |  | 0.4922 |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 2 | 47 | flirds_zgate_v2 | 0.5971 | +0.0132 | 0.0466 | +0.2842 | 0.9257 |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 2 | 47 | flirds_zgate_v2 | 0.5931 | +0.1124 | 0.1497 | +0.7504 | 0.9755 |  |  |
| cifar10 | qskew | label_flip | main |  | 2 | 47 | flirds_zgate_v2 | 0.5870 | +0.1299 | 0.1734 | +0.7491 | 0.9587 |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 2 | 47 | oracle_excl | 0.6305 | +0.0015 | 0.0015 |  |  |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 2 | 47 | oracle_excl | 0.6305 | +0.0466 | 0.0466 | +1.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 2 | 47 | oracle_excl | 0.6305 | +0.1497 | 0.1497 | +1.0000 |  |  |  |
| cifar10 | qskew | label_flip | main |  | 2 | 47 | oracle_excl | 0.6305 | +0.1734 | 0.1734 | +1.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 2 | 47 | random_excl | 0.6018 | -0.0272 | 0.0015 |  |  |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 2 | 47 | random_excl | 0.5573 | -0.0266 | 0.0466 | -0.5710 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 2 | 47 | random_excl | 0.4819 | +0.0011 | 0.1497 | +0.0075 |  |  |  |
| cifar10 | qskew | label_flip | main |  | 2 | 47 | random_excl | 0.4636 | +0.0065 | 0.1734 | +0.0375 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.1500 | 2 | 47 | vanilla | 0.6290 | +0.0000 | 0.0015 |  |  |  |  |
| cifar10 | qskew | label_flip | main | 0.3500 | 2 | 47 | vanilla | 0.5839 | +0.0000 | 0.0466 | +0.0000 |  |  |  |
| cifar10 | qskew | label_flip | main | 0.7000 | 2 | 47 | vanilla | 0.4808 | +0.0000 | 0.1497 | +0.0000 |  |  |  |
| cifar10 | qskew | label_flip | main |  | 2 | 47 | vanilla | 0.4571 | +0.0000 | 0.1734 | +0.0000 |  |  |  |
| cifar10 | shard | clean | main |  | 0 | 0 | flirds_gate_v1 | 0.4556 | -0.0572 |  |  |  | 455.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 0 | 0 | flirds_gate_v2 | 0.4491 | -0.0637 |  |  |  | 3747.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 0 | 0 | flirds_gatew_v1 | 0.3905 | -0.1224 |  |  |  | 368.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 0 | 0 | flirds_gatew_v2 | 0.4009 | -0.1120 |  |  |  | 3207.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 0 | 0 | flirds_mult | 0.5271 | +0.0142 |  |  |  |  |  |
| cifar10 | shard | clean | main |  | 0 | 0 | flirds_zgate_v2 | 0.5042 | -0.0086 |  |  |  |  |  |
| cifar10 | shard | clean | main |  | 0 | 0 | vanilla | 0.5129 | +0.0000 |  |  |  |  |  |
| cifar10 | shard | clean | main |  | 1 | 0 | flirds_gate_v1 | 0.4629 | +0.0260 |  |  |  | 442.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 1 | 0 | flirds_gate_v2 | 0.4674 | +0.0305 |  |  |  | 2984.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 1 | 0 | flirds_gatew_v1 | 0.1930 | -0.2439 |  |  |  | 393.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 1 | 0 | flirds_gatew_v2 | 0.4959 | +0.0590 |  |  |  | 3678.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 1 | 0 | flirds_mult | 0.4547 | +0.0179 |  |  |  |  |  |
| cifar10 | shard | clean | main |  | 1 | 0 | flirds_zgate_v2 | 0.4180 | -0.0189 |  |  |  |  |  |
| cifar10 | shard | clean | main |  | 1 | 0 | vanilla | 0.4369 | +0.0000 |  |  |  |  |  |
| cifar10 | shard | clean | main |  | 2 | 0 | flirds_gate_v1 | 0.4474 | -0.0282 |  |  |  | 460.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 2 | 0 | flirds_gate_v2 | 0.4435 | -0.0321 |  |  |  | 3419.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 2 | 0 | flirds_gatew_v1 | 0.4356 | -0.0400 |  |  |  | 402.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 2 | 0 | flirds_gatew_v2 | 0.3812 | -0.0944 |  |  |  | 4118.0000 | 0.0000 |
| cifar10 | shard | clean | main |  | 2 | 0 | flirds_mult | 0.5111 | +0.0355 |  |  |  |  |  |
| cifar10 | shard | clean | main |  | 2 | 0 | flirds_zgate_v2 | 0.5221 | +0.0465 |  |  |  |  |  |
| cifar10 | shard | clean | main |  | 2 | 0 | vanilla | 0.4756 | +0.0000 |  |  |  |  |  |
| cifar10 | shard | free_rider | main |  | 0 | 40 | flirds_gate_v1 | 0.2018 | -0.1751 | 0.0805 | -2.1755 | 0.9833 | 208.0000 | 0.6977 |
| cifar10 | shard | free_rider | main |  | 0 | 40 | flirds_gate_v2 | 0.3644 | -0.0125 | 0.0805 | -0.1553 | 0.6667 | 1870.0000 | 0.6896 |
| cifar10 | shard | free_rider | main |  | 0 | 40 | flirds_gatew_v1 | 0.2054 | -0.1715 | 0.0805 | -2.1304 | 1.0000 | 179.0000 | 0.7284 |
| cifar10 | shard | free_rider | main |  | 0 | 40 | flirds_gatew_v2 | 0.4675 | +0.0906 | 0.0805 | +1.1258 | 0.6333 | 1821.0000 | 0.6937 |
| cifar10 | shard | free_rider | main |  | 0 | 40 | flirds_mult | 0.4120 | +0.0351 | 0.0805 | +0.4363 | 0.3150 |  |  |
| cifar10 | shard | free_rider | main |  | 0 | 40 | flirds_zgate_v2 | 0.4126 | +0.0358 | 0.0805 | +0.4441 | 0.4167 |  |  |
| cifar10 | shard | free_rider | main |  | 0 | 40 | oracle_excl | 0.4574 | +0.0805 | 0.0805 | +1.0000 |  |  |  |
| cifar10 | shard | free_rider | main |  | 0 | 40 | random_excl | 0.4059 | +0.0290 | 0.0805 | +0.3602 |  |  |  |
| cifar10 | shard | free_rider | main |  | 0 | 40 | vanilla | 0.3769 | +0.0000 | 0.0805 | +0.0000 |  |  |  |
| cifar10 | shard | free_rider | main |  | 1 | 40 | flirds_gate_v1 | 0.3946 | -0.0304 | 0.0109 |  | 0.9667 | 185.0000 | 0.7214 |
| cifar10 | shard | free_rider | main |  | 1 | 40 | flirds_gate_v2 | 0.3936 | -0.0314 | 0.0109 |  | 0.6333 | 2108.0000 | 0.6591 |
| cifar10 | shard | free_rider | main |  | 1 | 40 | flirds_gatew_v1 | 0.4103 | -0.0147 | 0.0109 |  | 1.0000 | 159.0000 | 0.7508 |
| cifar10 | shard | free_rider | main |  | 1 | 40 | flirds_gatew_v2 | 0.4790 | +0.0540 | 0.0109 |  | 0.6500 | 1689.0000 | 0.7042 |
| cifar10 | shard | free_rider | main |  | 1 | 40 | flirds_mult | 0.4303 | +0.0053 | 0.0109 |  | 0.5408 |  |  |
| cifar10 | shard | free_rider | main |  | 1 | 40 | flirds_zgate_v2 | 0.4751 | +0.0501 | 0.0109 |  | 0.4333 |  |  |
| cifar10 | shard | free_rider | main |  | 1 | 40 | oracle_excl | 0.4359 | +0.0109 | 0.0109 |  |  |  |  |
| cifar10 | shard | free_rider | main |  | 1 | 40 | random_excl | 0.3997 | -0.0252 | 0.0109 |  |  |  |  |
| cifar10 | shard | free_rider | main |  | 1 | 40 | vanilla | 0.4250 | +0.0000 | 0.0109 |  |  |  |  |
| cifar10 | shard | free_rider | main |  | 2 | 40 | flirds_gate_v1 | 0.2716 | -0.1211 | 0.1207 | -1.0031 | 0.9833 | 209.0000 | 0.7010 |
| cifar10 | shard | free_rider | main |  | 2 | 40 | flirds_gate_v2 | 0.4348 | +0.0420 | 0.1207 | +0.3478 | 0.7167 | 1655.0000 | 0.7081 |
| cifar10 | shard | free_rider | main |  | 2 | 40 | flirds_gatew_v1 | 0.2839 | -0.1089 | 0.1207 | -0.9017 | 0.9833 | 184.0000 | 0.7270 |
| cifar10 | shard | free_rider | main |  | 2 | 40 | flirds_gatew_v2 | 0.4617 | +0.0690 | 0.1207 | +0.5714 | 0.6333 | 1556.0000 | 0.7202 |
| cifar10 | shard | free_rider | main |  | 2 | 40 | flirds_mult | 0.4071 | +0.0144 | 0.1207 | +0.1190 | 0.4667 |  |  |
| cifar10 | shard | free_rider | main |  | 2 | 40 | flirds_zgate_v2 | 0.4066 | +0.0139 | 0.1207 | +0.1149 | 0.4000 |  |  |
| cifar10 | shard | free_rider | main |  | 2 | 40 | oracle_excl | 0.5135 | +0.1207 | 0.1207 | +1.0000 |  |  |  |
| cifar10 | shard | free_rider | main |  | 2 | 40 | random_excl | 0.2969 | -0.0959 | 0.1207 | -0.7940 |  |  |  |
| cifar10 | shard | free_rider | main |  | 2 | 40 | vanilla | 0.3927 | +0.0000 | 0.1207 | +0.0000 |  |  |  |
| cifar10 | shard | frrand | main |  | 0 | 40 | flirds_gate_v1 | 0.4706 | +0.0939 | 0.0806 | +1.1643 | 0.8187 | 308.0000 | 0.4390 |
| cifar10 | shard | frrand | main |  | 0 | 40 | flirds_gate_v2 | 0.4057 | +0.0290 | 0.0806 | +0.3597 | 0.6833 | 1698.0000 | 0.6622 |
| cifar10 | shard | frrand | main |  | 0 | 40 | flirds_gatew_v1 | 0.2076 | -0.1691 | 0.0806 | -2.0977 | 1.0000 | 180.0000 | 0.5714 |
| cifar10 | shard | frrand | main |  | 0 | 40 | flirds_gatew_v2 | 0.4461 | +0.0694 | 0.0806 | +0.8605 | 0.8833 | 751.0000 | 0.7662 |
| cifar10 | shard | frrand | main |  | 0 | 40 | flirds_mult | 0.4118 | +0.0350 | 0.0806 | +0.4341 | 0.3150 |  |  |
| cifar10 | shard | frrand | main |  | 0 | 40 | flirds_zgate_v2 | 0.4674 | +0.0906 | 0.0806 | +1.1240 | 0.4667 |  |  |
| cifar10 | shard | frrand | main |  | 0 | 40 | oracle_excl | 0.4574 | +0.0806 | 0.0806 | +1.0000 |  |  |  |
| cifar10 | shard | frrand | main |  | 0 | 40 | random_excl | 0.4084 | +0.0316 | 0.0806 | +0.3922 |  |  |  |
| cifar10 | shard | frrand | main |  | 0 | 40 | vanilla | 0.3767 | +0.0000 | 0.0806 | +0.0000 |  |  |  |
| cifar10 | shard | frrand | main |  | 1 | 40 | flirds_gate_v1 | 0.4047 | -0.0189 | 0.0123 |  | 0.8500 | 280.0000 | 0.4824 |
| cifar10 | shard | frrand | main |  | 1 | 40 | flirds_gate_v2 | 0.4138 | -0.0099 | 0.0123 |  | 0.4333 | 3038.0000 | 0.4792 |
| cifar10 | shard | frrand | main |  | 1 | 40 | flirds_gatew_v1 | 0.4086 | -0.0150 | 0.0123 |  | 1.0000 | 159.0000 | 0.6005 |
| cifar10 | shard | frrand | main |  | 1 | 40 | flirds_gatew_v2 | 0.3705 | -0.0531 | 0.0123 |  | 0.8500 | 983.0000 | 0.7176 |
| cifar10 | shard | frrand | main |  | 1 | 40 | flirds_mult | 0.4288 | +0.0051 | 0.0123 |  | 0.5396 |  |  |
| cifar10 | shard | frrand | main |  | 1 | 40 | flirds_zgate_v2 | 0.4305 | +0.0069 | 0.0123 |  | 0.3833 |  |  |
| cifar10 | shard | frrand | main |  | 1 | 40 | oracle_excl | 0.4359 | +0.0123 | 0.0123 |  |  |  |  |
| cifar10 | shard | frrand | main |  | 1 | 40 | random_excl | 0.3985 | -0.0251 | 0.0123 |  |  |  |  |
| cifar10 | shard | frrand | main |  | 1 | 40 | vanilla | 0.4236 | +0.0000 | 0.0123 |  |  |  |  |
| cifar10 | shard | frrand | main |  | 2 | 40 | flirds_gate_v1 | 0.4301 | +0.0385 | 0.1219 | +0.3159 | 0.8333 | 290.0000 | 0.4487 |
| cifar10 | shard | frrand | main |  | 2 | 40 | flirds_gate_v2 | 0.4605 | +0.0689 | 0.1219 | +0.5651 | 0.5500 | 2723.0000 | 0.5313 |
| cifar10 | shard | frrand | main |  | 2 | 40 | flirds_gatew_v1 | 0.2850 | -0.1066 | 0.1219 | -0.8749 | 0.9833 | 183.0000 | 0.5734 |
| cifar10 | shard | frrand | main |  | 2 | 40 | flirds_gatew_v2 | 0.4218 | +0.0301 | 0.1219 | +0.2472 | 0.7500 | 1462.0000 | 0.6652 |
| cifar10 | shard | frrand | main |  | 2 | 40 | flirds_mult | 0.4062 | +0.0146 | 0.1219 | +0.1200 | 0.4679 |  |  |
| cifar10 | shard | frrand | main |  | 2 | 40 | flirds_zgate_v2 | 0.4368 | +0.0451 | 0.1219 | +0.3703 | 0.4833 |  |  |
| cifar10 | shard | frrand | main |  | 2 | 40 | oracle_excl | 0.5135 | +0.1219 | 0.1219 | +1.0000 |  |  |  |
| cifar10 | shard | frrand | main |  | 2 | 40 | random_excl | 0.2968 | -0.0949 | 0.1219 | -0.7785 |  |  |  |
| cifar10 | shard | frrand | main |  | 2 | 40 | vanilla | 0.3916 | +0.0000 | 0.1219 | +0.0000 |  |  |  |
| cifar10 | shard | grad_noise | main |  | 0 | 40 | flirds_gate_v1 | 0.1174 | -0.0535 | 0.2865 | -0.1867 | 0.8383 | 277.0000 | 0.5554 |
| cifar10 | shard | grad_noise | main |  | 0 | 40 | flirds_gate_v2 | 0.4079 | +0.2370 | 0.2865 | +0.8272 | 0.8408 | 3582.0000 | 0.5289 |
| cifar10 | shard | grad_noise | main |  | 0 | 40 | flirds_gatew_v1 | 0.1430 | -0.0279 | 0.2865 | -0.0973 | 0.8108 | 229.0000 | 0.5655 |
| cifar10 | shard | grad_noise | main |  | 0 | 40 | flirds_gatew_v2 | 0.4062 | +0.2354 | 0.2865 | +0.8216 | 0.9004 | 2152.0000 | 0.6468 |
| cifar10 | shard | grad_noise | main |  | 0 | 40 | flirds_mult | 0.2829 | +0.1120 | 0.2865 | +0.3909 | 0.9908 |  |  |
| cifar10 | shard | grad_noise | main |  | 0 | 40 | flirds_zgate_v2 | 0.1659 | -0.0050 | 0.2865 | -0.0175 | 1.0000 |  |  |
| cifar10 | shard | grad_noise | main |  | 0 | 40 | oracle_excl | 0.4574 | +0.2865 | 0.2865 | +1.0000 |  |  |  |
| cifar10 | shard | grad_noise | main |  | 0 | 40 | random_excl | 0.1809 | +0.0100 | 0.2865 | +0.0349 |  |  |  |
| cifar10 | shard | grad_noise | main |  | 0 | 40 | vanilla | 0.1709 | +0.0000 | 0.2865 | +0.0000 |  |  |  |
| cifar10 | shard | grad_noise | main |  | 1 | 40 | flirds_gate_v1 | 0.1645 | -0.0036 | 0.2678 | -0.0135 | 0.9758 | 230.0000 | 0.6082 |
| cifar10 | shard | grad_noise | main |  | 1 | 40 | flirds_gate_v2 | 0.3425 | +0.1744 | 0.2678 | +0.6513 | 0.9192 | 1925.0000 | 0.6601 |
| cifar10 | shard | grad_noise | main |  | 1 | 40 | flirds_gatew_v1 | 0.1946 | +0.0265 | 0.2678 | +0.0990 | 0.8696 | 190.0000 | 0.6267 |
| cifar10 | shard | grad_noise | main |  | 1 | 40 | flirds_gatew_v2 | 0.3931 | +0.2250 | 0.2678 | +0.8403 | 0.9600 | 2057.0000 | 0.6435 |
| cifar10 | shard | grad_noise | main |  | 1 | 40 | flirds_mult | 0.2789 | +0.1107 | 0.2678 | +0.4136 | 0.9879 |  |  |
| cifar10 | shard | grad_noise | main |  | 1 | 40 | flirds_zgate_v2 | 0.1857 | +0.0176 | 0.2678 | +0.0658 | 0.9987 |  |  |
| cifar10 | shard | grad_noise | main |  | 1 | 40 | oracle_excl | 0.4359 | +0.2678 | 0.2678 | +1.0000 |  |  |  |
| cifar10 | shard | grad_noise | main |  | 1 | 40 | random_excl | 0.1839 | +0.0158 | 0.2678 | +0.0588 |  |  |  |
| cifar10 | shard | grad_noise | main |  | 1 | 40 | vanilla | 0.1681 | +0.0000 | 0.2678 | +0.0000 |  |  |  |
| cifar10 | shard | grad_noise | main |  | 2 | 40 | flirds_gate_v1 | 0.1430 | -0.0181 | 0.3524 | -0.0514 | 0.7721 | 266.0000 | 0.5675 |
| cifar10 | shard | grad_noise | main |  | 2 | 40 | flirds_gate_v2 | 0.3286 | +0.1675 | 0.3524 | +0.4753 | 0.7742 | 3495.0000 | 0.5285 |
| cifar10 | shard | grad_noise | main |  | 2 | 40 | flirds_gatew_v1 | 0.1196 | -0.0415 | 0.3524 | -0.1178 | 0.8621 | 220.0000 | 0.5978 |
| cifar10 | shard | grad_noise | main |  | 2 | 40 | flirds_gatew_v2 | 0.2414 | +0.0803 | 0.3524 | +0.2277 | 0.8671 | 1904.0000 | 0.6472 |
| cifar10 | shard | grad_noise | main |  | 2 | 40 | flirds_mult | 0.2911 | +0.1300 | 0.3524 | +0.3689 | 1.0000 |  |  |
| cifar10 | shard | grad_noise | main |  | 2 | 40 | flirds_zgate_v2 | 0.1860 | +0.0249 | 0.3524 | +0.0706 | 1.0000 |  |  |
| cifar10 | shard | grad_noise | main |  | 2 | 40 | oracle_excl | 0.5135 | +0.3524 | 0.3524 | +1.0000 |  |  |  |
| cifar10 | shard | grad_noise | main |  | 2 | 40 | random_excl | 0.1445 | -0.0166 | 0.3524 | -0.0472 |  |  |  |
| cifar10 | shard | grad_noise | main |  | 2 | 40 | vanilla | 0.1611 | +0.0000 | 0.3524 | +0.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | 39 | flirds_gate_v1 | 0.4517 | -0.0271 | -0.0217 | +1.2471 | 0.3178 | 285.0000 | 0.3357 |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | 39 | flirds_gate_v1 | 0.3029 | -0.1344 | 0.0199 |  | 0.3443 | 293.0000 | 0.3122 |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | 39 | flirds_gate_v1 | 0.2416 | -0.1247 | 0.0907 | -1.3747 | 0.3728 | 294.0000 | 0.2864 |
| cifar10 | shard | label_flip | main |  | 0 | 39 | flirds_gate_v1 | 0.2191 | -0.1449 | 0.0931 | -1.5557 | 0.3825 | 292.0000 | 0.2861 |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | 39 | flirds_gate_v2 | 0.2299 | -0.2490 | -0.0217 | +11.4483 | 0.3573 | 2417.0000 | 0.2160 |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | 39 | flirds_gate_v2 | 0.1732 | -0.2640 | 0.0199 |  | 0.2102 | 3187.0000 | 0.1809 |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | 39 | flirds_gate_v2 | 0.3540 | -0.0124 | 0.0907 | -0.1364 | 0.2699 | 3318.0000 | 0.0758 |
| cifar10 | shard | label_flip | main |  | 0 | 39 | flirds_gate_v2 | 0.3435 | -0.0205 | 0.0931 | -0.2201 | 0.2631 | 3534.0000 | 0.1125 |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | 39 | flirds_gatew_v1 | 0.3101 | -0.1688 | -0.0217 | +7.7586 | 0.3279 | 244.0000 | 0.2989 |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | 39 | flirds_gatew_v1 | 0.2451 | -0.1921 | 0.0199 |  | 0.3804 | 246.0000 | 0.2545 |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | 39 | flirds_gatew_v1 | 0.2160 | -0.1504 | 0.0907 | -1.6570 | 0.3918 | 245.0000 | 0.2462 |
| cifar10 | shard | label_flip | main |  | 0 | 39 | flirds_gatew_v1 | 0.2106 | -0.1534 | 0.0931 | -1.6470 | 0.4048 | 256.0000 | 0.2099 |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | 39 | flirds_gatew_v2 | 0.4494 | -0.0295 | -0.0217 | +1.3563 | 0.3274 | 2748.0000 | 0.3031 |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | 39 | flirds_gatew_v2 | 0.3871 | -0.0501 | 0.0199 |  | 0.2930 | 2610.0000 | 0.0419 |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | 39 | flirds_gatew_v2 | 0.1745 | -0.1919 | 0.0907 | -2.1143 | 0.2135 | 2988.0000 | 0.0565 |
| cifar10 | shard | label_flip | main |  | 0 | 39 | flirds_gatew_v2 | 0.3342 | -0.0297 | 0.0931 | -0.3195 | 0.4191 | 2219.0000 | 0.1745 |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | 39 | flirds_mult | 0.5064 | +0.0275 | -0.0217 | -1.2644 | 0.4590 |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | 39 | flirds_mult | 0.4575 | +0.0202 | 0.0199 |  | 0.5158 |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | 39 | flirds_mult | 0.3729 | +0.0065 | 0.0907 | +0.0716 | 0.5439 |  |  |
| cifar10 | shard | label_flip | main |  | 0 | 39 | flirds_mult | 0.3710 | +0.0070 | 0.0931 | +0.0752 | 0.5570 |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | 39 | flirds_zgate_v2 | 0.4699 | -0.0090 | -0.0217 | +0.4138 | 0.3127 |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | 39 | flirds_zgate_v2 | 0.4487 | +0.0115 | 0.0199 |  | 0.3607 |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | 39 | flirds_zgate_v2 | 0.2789 | -0.0875 | 0.0907 | -0.9642 | 0.3888 |  |  |
| cifar10 | shard | label_flip | main |  | 0 | 39 | flirds_zgate_v2 | 0.3558 | -0.0082 | 0.0931 | -0.0886 | 0.3388 |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | 39 | oracle_excl | 0.4571 | -0.0217 | -0.0217 | +1.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | 39 | oracle_excl | 0.4571 | +0.0199 | 0.0199 |  |  |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | 39 | oracle_excl | 0.4571 | +0.0907 | 0.0907 | +1.0000 |  |  |  |
| cifar10 | shard | label_flip | main |  | 0 | 39 | oracle_excl | 0.4571 | +0.0931 | 0.0931 | +1.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | 39 | random_excl | 0.3762 | -0.1026 | -0.0217 | +4.7184 |  |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | 39 | random_excl | 0.3469 | -0.0904 | 0.0199 |  |  |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | 39 | random_excl | 0.3086 | -0.0578 | 0.0907 | -0.6364 |  |  |  |
| cifar10 | shard | label_flip | main |  | 0 | 39 | random_excl | 0.3167 | -0.0473 | 0.0931 | -0.5074 |  |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 0 | 39 | vanilla | 0.4789 | +0.0000 | -0.0217 | -0.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 0 | 39 | vanilla | 0.4373 | +0.0000 | 0.0199 |  |  |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 0 | 39 | vanilla | 0.3664 | +0.0000 | 0.0907 | +0.0000 |  |  |  |
| cifar10 | shard | label_flip | main |  | 0 | 39 | vanilla | 0.3640 | +0.0000 | 0.0931 | +0.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 1 | 48 | flirds_gate_v1 | 0.4370 | +0.0307 | -0.0136 |  | 0.3542 | 260.0000 | 0.3868 |
| cifar10 | shard | label_flip | main | 0.3500 | 1 | 48 | flirds_gate_v1 | 0.3330 | -0.0311 | 0.0285 | -1.0921 | 0.3834 | 256.0000 | 0.3756 |
| cifar10 | shard | label_flip | main | 0.7000 | 1 | 48 | flirds_gate_v1 | 0.3540 | +0.0170 | 0.0556 | +0.3056 | 0.4643 | 245.0000 | 0.4096 |
| cifar10 | shard | label_flip | main |  | 1 | 48 | flirds_gate_v1 | 0.3701 | +0.0311 | 0.0536 | +0.5804 | 0.4415 | 234.0000 | 0.4279 |
| cifar10 | shard | label_flip | main | 0.1500 | 1 | 48 | flirds_gate_v2 | 0.4291 | +0.0229 | -0.0136 |  | 0.2196 | 2827.0000 | 0.2171 |
| cifar10 | shard | label_flip | main | 0.3500 | 1 | 48 | flirds_gate_v2 | 0.3015 | -0.0626 | 0.0285 | -2.1974 | 0.2632 | 2856.0000 | 0.1585 |
| cifar10 | shard | label_flip | main | 0.7000 | 1 | 48 | flirds_gate_v2 | 0.1625 | -0.1745 | 0.0556 | -3.1371 | 0.2067 | 3050.0000 | 0.1121 |
| cifar10 | shard | label_flip | main |  | 1 | 48 | flirds_gate_v2 | 0.1624 | -0.1766 | 0.0536 | -3.2937 | 0.2192 | 3107.0000 | 0.2225 |
| cifar10 | shard | label_flip | main | 0.1500 | 1 | 48 | flirds_gatew_v1 | 0.4148 | +0.0085 | -0.0136 |  | 0.3578 | 228.0000 | 0.3429 |
| cifar10 | shard | label_flip | main | 0.3500 | 1 | 48 | flirds_gatew_v1 | 0.3954 | +0.0312 | 0.0285 | +1.0965 | 0.4211 | 221.0000 | 0.3115 |
| cifar10 | shard | label_flip | main | 0.7000 | 1 | 48 | flirds_gatew_v1 | 0.4196 | +0.0826 | 0.0556 | +1.4854 | 0.3922 | 210.0000 | 0.3656 |
| cifar10 | shard | label_flip | main |  | 1 | 48 | flirds_gatew_v1 | 0.3961 | +0.0571 | 0.0536 | +1.0653 | 0.4908 | 198.0000 | 0.3774 |
| cifar10 | shard | label_flip | main | 0.1500 | 1 | 48 | flirds_gatew_v2 | 0.4218 | +0.0155 | -0.0136 |  | 0.2248 | 3494.0000 | 0.2356 |
| cifar10 | shard | label_flip | main | 0.3500 | 1 | 48 | flirds_gatew_v2 | 0.2530 | -0.1111 | 0.0285 | -3.8991 | 0.2224 | 3294.0000 | 0.1260 |
| cifar10 | shard | label_flip | main | 0.7000 | 1 | 48 | flirds_gatew_v2 | 0.2023 | -0.1348 | 0.0556 | -2.4225 | 0.1651 | 3276.0000 | 0.1208 |
| cifar10 | shard | label_flip | main |  | 1 | 48 | flirds_gatew_v2 | 0.2464 | -0.0926 | 0.0536 | -1.7273 | 0.1983 | 3238.0000 | 0.2518 |
| cifar10 | shard | label_flip | main | 0.1500 | 1 | 48 | flirds_mult | 0.4049 | -0.0014 | -0.0136 |  | 0.4724 |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 1 | 48 | flirds_mult | 0.3468 | -0.0174 | 0.0285 | -0.6096 | 0.5909 |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 1 | 48 | flirds_mult | 0.2938 | -0.0433 | 0.0556 | -0.7775 | 0.6446 |  |  |
| cifar10 | shard | label_flip | main |  | 1 | 48 | flirds_mult | 0.3063 | -0.0328 | 0.0536 | -0.6107 | 0.6599 |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 1 | 48 | flirds_zgate_v2 | 0.4200 | +0.0137 | -0.0136 |  | 0.3025 |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 1 | 48 | flirds_zgate_v2 | 0.4046 | +0.0405 | 0.0285 | +1.4211 | 0.2997 |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 1 | 48 | flirds_zgate_v2 | 0.2629 | -0.0741 | 0.0556 | -1.3326 | 0.2400 |  |  |
| cifar10 | shard | label_flip | main |  | 1 | 48 | flirds_zgate_v2 | 0.2581 | -0.0809 | 0.0536 | -1.5082 | 0.4579 |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 1 | 48 | oracle_excl | 0.3926 | -0.0136 | -0.0136 |  |  |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 1 | 48 | oracle_excl | 0.3926 | +0.0285 | 0.0285 | +1.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 1 | 48 | oracle_excl | 0.3926 | +0.0556 | 0.0556 | +1.0000 |  |  |  |
| cifar10 | shard | label_flip | main |  | 1 | 48 | oracle_excl | 0.3926 | +0.0536 | 0.0536 | +1.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 1 | 48 | random_excl | 0.4145 | +0.0082 | -0.0136 |  |  |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 1 | 48 | random_excl | 0.3704 | +0.0063 | 0.0285 | +0.2193 |  |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 1 | 48 | random_excl | 0.3026 | -0.0344 | 0.0556 | -0.6180 |  |  |  |
| cifar10 | shard | label_flip | main |  | 1 | 48 | random_excl | 0.2866 | -0.0524 | 0.0536 | -0.9767 |  |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 1 | 48 | vanilla | 0.4062 | +0.0000 | -0.0136 |  |  |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 1 | 48 | vanilla | 0.3641 | +0.0000 | 0.0285 | +0.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 1 | 48 | vanilla | 0.3370 | +0.0000 | 0.0556 | +0.0000 |  |  |  |
| cifar10 | shard | label_flip | main |  | 1 | 48 | vanilla | 0.3390 | +0.0000 | 0.0536 | +0.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 2 | 47 | flirds_gate_v1 | 0.4070 | +0.0079 | 0.0306 | +0.2571 | 0.3087 | 277.0000 | 0.3885 |
| cifar10 | shard | label_flip | main | 0.3500 | 2 | 47 | flirds_gate_v1 | 0.4049 | +0.0777 | 0.1026 | +0.7576 | 0.3256 | 260.0000 | 0.4023 |
| cifar10 | shard | label_flip | main | 0.7000 | 2 | 47 | flirds_gate_v1 | 0.2705 | +0.0389 | 0.1981 | +0.1962 | 0.3248 | 265.0000 | 0.3645 |
| cifar10 | shard | label_flip | main |  | 2 | 47 | flirds_gate_v1 | 0.2567 | +0.0061 | 0.1791 | +0.0342 | 0.3151 | 272.0000 | 0.3600 |
| cifar10 | shard | label_flip | main | 0.1500 | 2 | 47 | flirds_gate_v2 | 0.3972 | -0.0019 | 0.0306 | -0.0612 | 0.2971 | 2887.0000 | 0.2492 |
| cifar10 | shard | label_flip | main | 0.3500 | 2 | 47 | flirds_gate_v2 | 0.2729 | -0.0543 | 0.1026 | -0.5286 | 0.2597 | 3204.0000 | 0.2089 |
| cifar10 | shard | label_flip | main | 0.7000 | 2 | 47 | flirds_gate_v2 | 0.2390 | +0.0074 | 0.1981 | +0.0372 | 0.2067 | 3581.0000 | 0.1502 |
| cifar10 | shard | label_flip | main |  | 2 | 47 | flirds_gate_v2 | 0.1618 | -0.0889 | 0.1791 | -0.4962 | 0.0638 | 4432.0000 | 0.1750 |
| cifar10 | shard | label_flip | main | 0.1500 | 2 | 47 | flirds_gatew_v1 | 0.4106 | +0.0115 | 0.0306 | +0.3755 | 0.3705 | 220.0000 | 0.3939 |
| cifar10 | shard | label_flip | main | 0.3500 | 2 | 47 | flirds_gatew_v1 | 0.3830 | +0.0559 | 0.1026 | +0.5445 | 0.3629 | 220.0000 | 0.3333 |
| cifar10 | shard | label_flip | main | 0.7000 | 2 | 47 | flirds_gatew_v1 | 0.3569 | +0.1253 | 0.1981 | +0.6322 | 0.3364 | 234.0000 | 0.2778 |
| cifar10 | shard | label_flip | main |  | 2 | 47 | flirds_gatew_v1 | 0.3713 | +0.1206 | 0.1791 | +0.6734 | 0.3802 | 231.0000 | 0.3636 |
| cifar10 | shard | label_flip | main | 0.1500 | 2 | 47 | flirds_gatew_v2 | 0.4296 | +0.0305 | 0.0306 | +0.9959 | 0.2798 | 2285.0000 | 0.1394 |
| cifar10 | shard | label_flip | main | 0.3500 | 2 | 47 | flirds_gatew_v2 | 0.2925 | -0.0346 | 0.1026 | -0.3374 | 0.2714 | 2321.0000 | 0.0449 |
| cifar10 | shard | label_flip | main | 0.7000 | 2 | 47 | flirds_gatew_v2 | 0.2605 | +0.0289 | 0.1981 | +0.1457 | 0.2886 | 2290.0000 | 0.2632 |
| cifar10 | shard | label_flip | main |  | 2 | 47 | flirds_gatew_v2 | 0.2343 | -0.0164 | 0.1791 | -0.0914 | 0.2007 | 3298.0000 | 0.1048 |
| cifar10 | shard | label_flip | main | 0.1500 | 2 | 47 | flirds_mult | 0.4542 | +0.0551 | 0.0306 | +1.8000 | 0.3846 |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 2 | 47 | flirds_mult | 0.3840 | +0.0569 | 0.1026 | +0.5542 | 0.3970 |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 2 | 47 | flirds_mult | 0.2775 | +0.0459 | 0.1981 | +0.2315 | 0.3918 |  |  |
| cifar10 | shard | label_flip | main |  | 2 | 47 | flirds_mult | 0.2679 | +0.0172 | 0.1791 | +0.0963 | 0.4063 |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 2 | 47 | flirds_zgate_v2 | 0.4391 | +0.0400 | 0.0306 | +1.3061 | 0.2898 |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 2 | 47 | flirds_zgate_v2 | 0.3690 | +0.0419 | 0.1026 | +0.4080 | 0.2931 |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 2 | 47 | flirds_zgate_v2 | 0.2732 | +0.0416 | 0.1981 | +0.2101 | 0.2931 |  |  |
| cifar10 | shard | label_flip | main |  | 2 | 47 | flirds_zgate_v2 | 0.3331 | +0.0825 | 0.1791 | +0.4606 | 0.4396 |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 2 | 47 | oracle_excl | 0.4298 | +0.0306 | 0.0306 | +1.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 2 | 47 | oracle_excl | 0.4298 | +0.1026 | 0.1026 | +1.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 2 | 47 | oracle_excl | 0.4298 | +0.1981 | 0.1981 | +1.0000 |  |  |  |
| cifar10 | shard | label_flip | main |  | 2 | 47 | oracle_excl | 0.4298 | +0.1791 | 0.1791 | +1.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 2 | 47 | random_excl | 0.3370 | -0.0621 | 0.0306 | -2.0286 |  |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 2 | 47 | random_excl | 0.2611 | -0.0660 | 0.1026 | -0.6431 |  |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 2 | 47 | random_excl | 0.2236 | -0.0080 | 0.1981 | -0.0404 |  |  |  |
| cifar10 | shard | label_flip | main |  | 2 | 47 | random_excl | 0.2119 | -0.0387 | 0.1791 | -0.2163 |  |  |  |
| cifar10 | shard | label_flip | main | 0.1500 | 2 | 47 | vanilla | 0.3991 | +0.0000 | 0.0306 | +0.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.3500 | 2 | 47 | vanilla | 0.3271 | +0.0000 | 0.1026 | +0.0000 |  |  |  |
| cifar10 | shard | label_flip | main | 0.7000 | 2 | 47 | vanilla | 0.2316 | +0.0000 | 0.1981 | +0.0000 |  |  |  |
| cifar10 | shard | label_flip | main |  | 2 | 47 | vanilla | 0.2506 | +0.0000 | 0.1791 | +0.0000 |  |  |  |
| fmnist | dir1 | clean | main |  | 0 | 0 | flirds_gate_v1 | 0.8459 | +0.0519 |  |  |  | 456.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 0 | 0 | flirds_gate_v2 | 0.8275 | +0.0335 |  |  |  | 2746.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 0 | 0 | flirds_gatew_v1 | 0.8475 | +0.0535 |  |  |  | 323.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 0 | 0 | flirds_gatew_v2 | 0.8435 | +0.0495 |  |  |  | 1860.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 0 | 0 | flirds_mult | 0.8154 | +0.0214 |  |  |  |  |  |
| fmnist | dir1 | clean | main |  | 0 | 0 | flirds_zgate_v2 | 0.8455 | +0.0515 |  |  |  |  |  |
| fmnist | dir1 | clean | main |  | 0 | 0 | vanilla | 0.7940 | +0.0000 |  |  |  |  |  |
| fmnist | dir1 | clean | main |  | 1 | 0 | flirds_gate_v1 | 0.8536 | -0.0030 |  |  |  | 457.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 1 | 0 | flirds_gate_v2 | 0.8411 | -0.0155 |  |  |  | 2262.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 1 | 0 | flirds_gatew_v1 | 0.8485 | -0.0081 |  |  |  | 353.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 1 | 0 | flirds_gatew_v2 | 0.8466 | -0.0100 |  |  |  | 2277.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 1 | 0 | flirds_mult | 0.8526 | -0.0040 |  |  |  |  |  |
| fmnist | dir1 | clean | main |  | 1 | 0 | flirds_zgate_v2 | 0.8591 | +0.0025 |  |  |  |  |  |
| fmnist | dir1 | clean | main |  | 1 | 0 | vanilla | 0.8566 | +0.0000 |  |  |  |  |  |
| fmnist | dir1 | clean | main |  | 2 | 0 | flirds_gate_v1 | 0.8170 | +0.0324 |  |  |  | 412.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 2 | 0 | flirds_gate_v2 | 0.8276 | +0.0430 |  |  |  | 2065.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 2 | 0 | flirds_gatew_v1 | 0.8286 | +0.0440 |  |  |  | 304.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 2 | 0 | flirds_gatew_v2 | 0.8466 | +0.0620 |  |  |  | 2640.0000 | 0.0000 |
| fmnist | dir1 | clean | main |  | 2 | 0 | flirds_mult | 0.8197 | +0.0351 |  |  |  |  |  |
| fmnist | dir1 | clean | main |  | 2 | 0 | flirds_zgate_v2 | 0.8399 | +0.0553 |  |  |  |  |  |
| fmnist | dir1 | clean | main |  | 2 | 0 | vanilla | 0.7846 | +0.0000 |  |  |  |  |  |
| fmnist | dir1 | free_rider | main |  | 0 | 40 | flirds_gate_v1 | 0.7819 | -0.0119 | 0.0440 | -0.2699 | 0.9333 | 196.0000 | 0.7101 |
| fmnist | dir1 | free_rider | main |  | 0 | 40 | flirds_gate_v2 | 0.8371 | +0.0434 | 0.0440 | +0.9858 | 0.7000 | 1544.0000 | 0.7298 |
| fmnist | dir1 | free_rider | main |  | 0 | 40 | flirds_gatew_v1 | 0.7943 | +0.0005 | 0.0440 | +0.0114 | 0.9833 | 142.0000 | 0.7717 |
| fmnist | dir1 | free_rider | main |  | 0 | 40 | flirds_gatew_v2 | 0.8204 | +0.0266 | 0.0440 | +0.6051 | 0.7333 | 1244.0000 | 0.7673 |
| fmnist | dir1 | free_rider | main |  | 0 | 40 | flirds_mult | 0.8085 | +0.0148 | 0.0440 | +0.3352 | 0.3946 |  |  |
| fmnist | dir1 | free_rider | main |  | 0 | 40 | flirds_zgate_v2 | 0.8021 | +0.0084 | 0.0440 | +0.1903 | 0.7000 |  |  |
| fmnist | dir1 | free_rider | main |  | 0 | 40 | oracle_excl | 0.8377 | +0.0440 | 0.0440 | +1.0000 |  |  |  |
| fmnist | dir1 | free_rider | main |  | 0 | 40 | random_excl | 0.8095 | +0.0158 | 0.0440 | +0.3580 |  |  |  |
| fmnist | dir1 | free_rider | main |  | 0 | 40 | vanilla | 0.7937 | +0.0000 | 0.0440 | +0.0000 |  |  |  |
| fmnist | dir1 | free_rider | main |  | 1 | 40 | flirds_gate_v1 | 0.8494 | +0.0257 | 0.0234 | +1.1016 | 0.9333 | 172.0000 | 0.7358 |
| fmnist | dir1 | free_rider | main |  | 1 | 40 | flirds_gate_v2 | 0.8455 | +0.0219 | 0.0234 | +0.9358 | 0.7333 | 1285.0000 | 0.7590 |
| fmnist | dir1 | free_rider | main |  | 1 | 40 | flirds_gatew_v1 | 0.8365 | +0.0129 | 0.0234 | +0.5508 | 0.9833 | 124.0000 | 0.7944 |
| fmnist | dir1 | free_rider | main |  | 1 | 40 | flirds_gatew_v2 | 0.8606 | +0.0370 | 0.0234 | +1.5829 | 0.6833 | 1356.0000 | 0.7459 |
| fmnist | dir1 | free_rider | main |  | 1 | 40 | flirds_mult | 0.8351 | +0.0115 | 0.0234 | +0.4920 | 0.4187 |  |  |
| fmnist | dir1 | free_rider | main |  | 1 | 40 | flirds_zgate_v2 | 0.8207 | -0.0029 | 0.0234 | -0.1230 | 0.7333 |  |  |
| fmnist | dir1 | free_rider | main |  | 1 | 40 | oracle_excl | 0.8470 | +0.0234 | 0.0234 | +1.0000 |  |  |  |
| fmnist | dir1 | free_rider | main |  | 1 | 40 | random_excl | 0.8290 | +0.0054 | 0.0234 | +0.2299 |  |  |  |
| fmnist | dir1 | free_rider | main |  | 1 | 40 | vanilla | 0.8236 | +0.0000 | 0.0234 | +0.0000 |  |  |  |
| fmnist | dir1 | free_rider | main |  | 2 | 40 | flirds_gate_v1 | 0.7960 | -0.0110 | 0.0337 | -0.3259 | 0.9333 | 176.0000 | 0.7357 |
| fmnist | dir1 | free_rider | main |  | 2 | 40 | flirds_gate_v2 | 0.8341 | +0.0271 | 0.0337 | +0.8037 | 0.6833 | 1503.0000 | 0.7247 |
| fmnist | dir1 | free_rider | main |  | 2 | 40 | flirds_gatew_v1 | 0.7984 | -0.0086 | 0.0337 | -0.2556 | 0.9667 | 137.0000 | 0.7815 |
| fmnist | dir1 | free_rider | main |  | 2 | 40 | flirds_gatew_v2 | 0.8452 | +0.0382 | 0.0337 | +1.1333 | 0.8500 | 605.0000 | 0.8701 |
| fmnist | dir1 | free_rider | main |  | 2 | 40 | flirds_mult | 0.8177 | +0.0107 | 0.0337 | +0.3185 | 0.4963 |  |  |
| fmnist | dir1 | free_rider | main |  | 2 | 40 | flirds_zgate_v2 | 0.8047 | -0.0023 | 0.0337 | -0.0667 | 0.7667 |  |  |
| fmnist | dir1 | free_rider | main |  | 2 | 40 | oracle_excl | 0.8407 | +0.0337 | 0.0337 | +1.0000 |  |  |  |
| fmnist | dir1 | free_rider | main |  | 2 | 40 | random_excl | 0.7989 | -0.0081 | 0.0337 | -0.2407 |  |  |  |
| fmnist | dir1 | free_rider | main |  | 2 | 40 | vanilla | 0.8070 | +0.0000 | 0.0337 | +0.0000 |  |  |  |
| fmnist | dir1 | frrand | main |  | 0 | 40 | flirds_gate_v1 | 0.7934 | -0.0027 | 0.0416 | -0.0661 | 0.8350 | 295.0000 | 0.4517 |
| fmnist | dir1 | frrand | main |  | 0 | 40 | flirds_gate_v2 | 0.7989 | +0.0028 | 0.0416 | +0.0661 | 0.6033 | 2039.0000 | 0.5764 |
| fmnist | dir1 | frrand | main |  | 0 | 40 | flirds_gatew_v1 | 0.7991 | +0.0030 | 0.0416 | +0.0721 | 0.9833 | 150.0000 | 0.6042 |
| fmnist | dir1 | frrand | main |  | 0 | 40 | flirds_gatew_v2 | 0.8381 | +0.0420 | 0.0416 | +1.0090 | 0.7558 | 956.0000 | 0.7296 |
| fmnist | dir1 | frrand | main |  | 0 | 40 | flirds_mult | 0.8071 | +0.0110 | 0.0416 | +0.2643 | 0.3904 |  |  |
| fmnist | dir1 | frrand | main |  | 0 | 40 | flirds_zgate_v2 | 0.8009 | +0.0048 | 0.0416 | +0.1141 | 0.7000 |  |  |
| fmnist | dir1 | frrand | main |  | 0 | 40 | oracle_excl | 0.8377 | +0.0416 | 0.0416 | +1.0000 |  |  |  |
| fmnist | dir1 | frrand | main |  | 0 | 40 | random_excl | 0.8087 | +0.0126 | 0.0416 | +0.3033 |  |  |  |
| fmnist | dir1 | frrand | main |  | 0 | 40 | vanilla | 0.7961 | +0.0000 | 0.0416 | +0.0000 |  |  |  |
| fmnist | dir1 | frrand | main |  | 1 | 40 | flirds_gate_v1 | 0.8286 | +0.0055 | 0.0239 | +0.2304 | 0.8333 | 252.0000 | 0.4825 |
| fmnist | dir1 | frrand | main |  | 1 | 40 | flirds_gate_v2 | 0.8416 | +0.0185 | 0.0239 | +0.7749 | 0.6983 | 1648.0000 | 0.5933 |
| fmnist | dir1 | frrand | main |  | 1 | 40 | flirds_gatew_v1 | 0.8404 | +0.0172 | 0.0239 | +0.7225 | 0.9833 | 129.0000 | 0.6532 |
| fmnist | dir1 | frrand | main |  | 1 | 40 | flirds_gatew_v2 | 0.8500 | +0.0269 | 0.0239 | +1.1257 | 0.8742 | 797.0000 | 0.7529 |
| fmnist | dir1 | frrand | main |  | 1 | 40 | flirds_mult | 0.8345 | +0.0114 | 0.0239 | +0.4764 | 0.4167 |  |  |
| fmnist | dir1 | frrand | main |  | 1 | 40 | flirds_zgate_v2 | 0.8189 | -0.0042 | 0.0239 | -0.1780 | 0.6667 |  |  |
| fmnist | dir1 | frrand | main |  | 1 | 40 | oracle_excl | 0.8470 | +0.0239 | 0.0239 | +1.0000 |  |  |  |
| fmnist | dir1 | frrand | main |  | 1 | 40 | random_excl | 0.8303 | +0.0071 | 0.0239 | +0.2984 |  |  |  |
| fmnist | dir1 | frrand | main |  | 1 | 40 | vanilla | 0.8231 | +0.0000 | 0.0239 | +0.0000 |  |  |  |
| fmnist | dir1 | frrand | main |  | 2 | 40 | flirds_gate_v1 | 0.7890 | -0.0185 | 0.0333 | -0.5564 | 0.8654 | 253.0000 | 0.4794 |
| fmnist | dir1 | frrand | main |  | 2 | 40 | flirds_gate_v2 | 0.8244 | +0.0169 | 0.0333 | +0.5075 | 0.6683 | 1624.0000 | 0.6194 |
| fmnist | dir1 | frrand | main |  | 2 | 40 | flirds_gatew_v1 | 0.8046 | -0.0029 | 0.0333 | -0.0865 | 0.9629 | 136.0000 | 0.6344 |
| fmnist | dir1 | frrand | main |  | 2 | 40 | flirds_gatew_v2 | 0.8367 | +0.0292 | 0.0333 | +0.8797 | 0.9171 | 340.0000 | 0.8756 |
| fmnist | dir1 | frrand | main |  | 2 | 40 | flirds_mult | 0.8177 | +0.0102 | 0.0333 | +0.3083 | 0.4858 |  |  |
| fmnist | dir1 | frrand | main |  | 2 | 40 | flirds_zgate_v2 | 0.8037 | -0.0038 | 0.0333 | -0.1128 | 0.7417 |  |  |
| fmnist | dir1 | frrand | main |  | 2 | 40 | oracle_excl | 0.8407 | +0.0333 | 0.0333 | +1.0000 |  |  |  |
| fmnist | dir1 | frrand | main |  | 2 | 40 | random_excl | 0.7999 | -0.0076 | 0.0333 | -0.2293 |  |  |  |
| fmnist | dir1 | frrand | main |  | 2 | 40 | vanilla | 0.8075 | +0.0000 | 0.0333 | +0.0000 |  |  |  |
| fmnist | dir1 | grad_noise | main |  | 0 | 40 | flirds_gate_v1 | 0.7969 | +0.0486 | 0.0895 | +0.5433 | 0.9871 | 226.0000 | 0.6367 |
| fmnist | dir1 | grad_noise | main |  | 0 | 40 | flirds_gate_v2 | 0.8592 | +0.1110 | 0.0895 | +1.2402 | 0.9512 | 1325.0000 | 0.7496 |
| fmnist | dir1 | grad_noise | main |  | 0 | 40 | flirds_gatew_v1 | 0.7691 | +0.0209 | 0.0895 | +0.2332 | 0.9888 | 151.0000 | 0.7140 |
| fmnist | dir1 | grad_noise | main |  | 0 | 40 | flirds_gatew_v2 | 0.8578 | +0.1095 | 0.0895 | +1.2235 | 0.9808 | 859.0000 | 0.8091 |
| fmnist | dir1 | grad_noise | main |  | 0 | 40 | flirds_mult | 0.8066 | +0.0584 | 0.0895 | +0.6522 | 0.9896 |  |  |
| fmnist | dir1 | grad_noise | main |  | 0 | 40 | flirds_zgate_v2 | 0.8079 | +0.0596 | 0.0895 | +0.6662 | 0.9996 |  |  |
| fmnist | dir1 | grad_noise | main |  | 0 | 40 | oracle_excl | 0.8377 | +0.0895 | 0.0895 | +1.0000 |  |  |  |
| fmnist | dir1 | grad_noise | main |  | 0 | 40 | random_excl | 0.7705 | +0.0222 | 0.0895 | +0.2486 |  |  |  |
| fmnist | dir1 | grad_noise | main |  | 0 | 40 | vanilla | 0.7482 | +0.0000 | 0.0895 | +0.0000 |  |  |  |
| fmnist | dir1 | grad_noise | main |  | 1 | 40 | flirds_gate_v1 | 0.8004 | +0.0331 | 0.0797 | +0.4154 | 0.9804 | 200.0000 | 0.6616 |
| fmnist | dir1 | grad_noise | main |  | 1 | 40 | flirds_gate_v2 | 0.8625 | +0.0953 | 0.0797 | +1.1944 | 0.9933 | 849.0000 | 0.8182 |
| fmnist | dir1 | grad_noise | main |  | 1 | 40 | flirds_gatew_v1 | 0.7775 | +0.0102 | 0.0797 | +0.1285 | 0.9883 | 145.0000 | 0.7206 |
| fmnist | dir1 | grad_noise | main |  | 1 | 40 | flirds_gatew_v2 | 0.8554 | +0.0881 | 0.0797 | +1.1050 | 0.9763 | 846.0000 | 0.8146 |
| fmnist | dir1 | grad_noise | main |  | 1 | 40 | flirds_mult | 0.7926 | +0.0254 | 0.0797 | +0.3182 | 0.9904 |  |  |
| fmnist | dir1 | grad_noise | main |  | 1 | 40 | flirds_zgate_v2 | 0.7926 | +0.0254 | 0.0797 | +0.3182 | 0.9987 |  |  |
| fmnist | dir1 | grad_noise | main |  | 1 | 40 | oracle_excl | 0.8470 | +0.0797 | 0.0797 | +1.0000 |  |  |  |
| fmnist | dir1 | grad_noise | main |  | 1 | 40 | random_excl | 0.7431 | -0.0241 | 0.0797 | -0.3025 |  |  |  |
| fmnist | dir1 | grad_noise | main |  | 1 | 40 | vanilla | 0.7672 | +0.0000 | 0.0797 | +0.0000 |  |  |  |
| fmnist | dir1 | grad_noise | main |  | 2 | 40 | flirds_gate_v1 | 0.7471 | +0.0428 | 0.1364 | +0.3135 | 0.9746 | 177.0000 | 0.6932 |
| fmnist | dir1 | grad_noise | main |  | 2 | 40 | flirds_gate_v2 | 0.8466 | +0.1422 | 0.1364 | +1.0431 | 0.8267 | 1835.0000 | 0.6844 |
| fmnist | dir1 | grad_noise | main |  | 2 | 40 | flirds_gatew_v1 | 0.7580 | +0.0536 | 0.1364 | +0.3932 | 0.9317 | 158.0000 | 0.7168 |
| fmnist | dir1 | grad_noise | main |  | 2 | 40 | flirds_gatew_v2 | 0.8464 | +0.1420 | 0.1364 | +1.0412 | 0.9867 | 563.0000 | 0.8682 |
| fmnist | dir1 | grad_noise | main |  | 2 | 40 | flirds_mult | 0.7850 | +0.0806 | 0.1364 | +0.5912 | 0.9812 |  |  |
| fmnist | dir1 | grad_noise | main |  | 2 | 40 | flirds_zgate_v2 | 0.7816 | +0.0773 | 0.1364 | +0.5665 | 0.9983 |  |  |
| fmnist | dir1 | grad_noise | main |  | 2 | 40 | oracle_excl | 0.8407 | +0.1364 | 0.1364 | +1.0000 |  |  |  |
| fmnist | dir1 | grad_noise | main |  | 2 | 40 | random_excl | 0.7007 | -0.0036 | 0.1364 | -0.0266 |  |  |  |
| fmnist | dir1 | grad_noise | main |  | 2 | 40 | vanilla | 0.7044 | +0.0000 | 0.1364 | +0.0000 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | 39 | flirds_gate_v1 | 0.6817 | -0.1401 | -0.0015 |  | 0.9197 | 185.0000 | 0.6358 |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | 39 | flirds_gate_v1 | 0.8014 | -0.0108 | 0.0082 |  | 0.9954 | 174.0000 | 0.6871 |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | 39 | flirds_gate_v1 | 0.7571 | -0.0389 | 0.0244 | -1.5949 | 1.0000 | 195.0000 | 0.6761 |
| fmnist | dir1 | label_flip | main |  | 0 | 39 | flirds_gate_v1 | 0.7991 | -0.0016 | 0.0196 |  | 1.0000 | 199.0000 | 0.6801 |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | 39 | flirds_gate_v2 | 0.8423 | +0.0204 | -0.0015 |  | 0.8567 | 892.0000 | 0.7163 |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | 39 | flirds_gate_v2 | 0.8155 | +0.0034 | 0.0082 |  | 0.8878 | 1215.0000 | 0.7357 |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | 39 | flirds_gate_v2 | 0.8335 | +0.0375 | 0.0244 | +1.5385 | 0.9630 | 1247.0000 | 0.7507 |
| fmnist | dir1 | label_flip | main |  | 0 | 39 | flirds_gate_v2 | 0.8283 | +0.0275 | 0.0196 |  | 0.9521 | 1410.0000 | 0.7254 |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | 39 | flirds_gatew_v1 | 0.8346 | +0.0127 | -0.0015 |  | 0.9096 | 142.0000 | 0.6459 |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | 39 | flirds_gatew_v1 | 0.8156 | +0.0035 | 0.0082 |  | 0.9933 | 143.0000 | 0.7111 |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | 39 | flirds_gatew_v1 | 0.7901 | -0.0059 | 0.0244 | -0.2410 | 1.0000 | 195.0000 | 0.6814 |
| fmnist | dir1 | label_flip | main |  | 0 | 39 | flirds_gatew_v1 | 0.7955 | -0.0052 | 0.0196 |  | 0.9996 | 171.0000 | 0.7067 |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | 39 | flirds_gatew_v2 | 0.8360 | +0.0141 | -0.0015 |  | 0.8029 | 1332.0000 | 0.6282 |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | 39 | flirds_gatew_v2 | 0.8316 | +0.0195 | 0.0082 |  | 0.8680 | 1576.0000 | 0.6749 |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | 39 | flirds_gatew_v2 | 0.8383 | +0.0423 | 0.0244 | +1.7333 | 0.9668 | 1307.0000 | 0.7435 |
| fmnist | dir1 | label_flip | main |  | 0 | 39 | flirds_gatew_v2 | 0.8319 | +0.0311 | 0.0196 |  | 0.9596 | 1518.0000 | 0.7040 |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | 39 | flirds_mult | 0.8239 | +0.0020 | -0.0015 |  | 0.8684 |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | 39 | flirds_mult | 0.8043 | -0.0079 | 0.0082 |  | 0.9777 |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | 39 | flirds_mult | 0.7770 | -0.0190 | 0.0244 | -0.7795 | 0.9996 |  |  |
| fmnist | dir1 | label_flip | main |  | 0 | 39 | flirds_mult | 0.7706 | -0.0301 | 0.0196 |  | 1.0000 |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | 39 | flirds_zgate_v2 | 0.8364 | +0.0145 | -0.0015 |  | 0.9252 |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | 39 | flirds_zgate_v2 | 0.8207 | +0.0086 | 0.0082 |  | 1.0000 |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | 39 | flirds_zgate_v2 | 0.8197 | +0.0237 | 0.0244 | +0.9744 | 1.0000 |  |  |
| fmnist | dir1 | label_flip | main |  | 0 | 39 | flirds_zgate_v2 | 0.8080 | +0.0073 | 0.0196 |  | 1.0000 |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | 39 | oracle_excl | 0.8204 | -0.0015 | -0.0015 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | 39 | oracle_excl | 0.8204 | +0.0082 | 0.0082 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | 39 | oracle_excl | 0.8204 | +0.0244 | 0.0244 | +1.0000 |  |  |  |
| fmnist | dir1 | label_flip | main |  | 0 | 39 | oracle_excl | 0.8204 | +0.0196 | 0.0196 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | 39 | random_excl | 0.7943 | -0.0276 | -0.0015 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | 39 | random_excl | 0.7849 | -0.0272 | 0.0082 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | 39 | random_excl | 0.7695 | -0.0265 | 0.0244 | -1.0872 |  |  |  |
| fmnist | dir1 | label_flip | main |  | 0 | 39 | random_excl | 0.7664 | -0.0344 | 0.0196 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 0 | 39 | vanilla | 0.8219 | +0.0000 | -0.0015 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 0 | 39 | vanilla | 0.8121 | +0.0000 | 0.0082 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 0 | 39 | vanilla | 0.7960 | +0.0000 | 0.0244 | +0.0000 |  |  |  |
| fmnist | dir1 | label_flip | main |  | 0 | 39 | vanilla | 0.8007 | +0.0000 | 0.0196 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 1 | 48 | flirds_gate_v1 | 0.8481 | -0.0004 | 0.0087 |  | 0.9107 | 154.0000 | 0.7205 |
| fmnist | dir1 | label_flip | main | 0.3500 | 1 | 48 | flirds_gate_v1 | 0.8462 | +0.0025 | 0.0135 |  | 0.9952 | 172.0000 | 0.7358 |
| fmnist | dir1 | label_flip | main | 0.7000 | 1 | 48 | flirds_gate_v1 | 0.8435 | +0.0231 | 0.0369 | +0.6271 | 1.0000 | 217.0000 | 0.7052 |
| fmnist | dir1 | label_flip | main |  | 1 | 48 | flirds_gate_v1 | 0.8360 | +0.0111 | 0.0324 | +0.3436 | 1.0000 | 208.0000 | 0.7135 |
| fmnist | dir1 | label_flip | main | 0.1500 | 1 | 48 | flirds_gate_v2 | 0.8407 | -0.0078 | 0.0087 |  | 0.8938 | 493.0000 | 0.8817 |
| fmnist | dir1 | label_flip | main | 0.3500 | 1 | 48 | flirds_gate_v2 | 0.8480 | +0.0042 | 0.0135 |  | 0.9042 | 1469.0000 | 0.7555 |
| fmnist | dir1 | label_flip | main | 0.7000 | 1 | 48 | flirds_gate_v2 | 0.8529 | +0.0325 | 0.0369 | +0.8814 | 0.9607 | 1076.0000 | 0.8142 |
| fmnist | dir1 | label_flip | main |  | 1 | 48 | flirds_gate_v2 | 0.8539 | +0.0290 | 0.0324 | +0.8958 | 0.9740 | 1301.0000 | 0.7880 |
| fmnist | dir1 | label_flip | main | 0.1500 | 1 | 48 | flirds_gatew_v1 | 0.8458 | -0.0028 | 0.0087 |  | 0.9299 | 134.0000 | 0.7254 |
| fmnist | dir1 | label_flip | main | 0.3500 | 1 | 48 | flirds_gatew_v1 | 0.8420 | -0.0018 | 0.0135 |  | 0.9936 | 167.0000 | 0.7407 |
| fmnist | dir1 | label_flip | main | 0.7000 | 1 | 48 | flirds_gatew_v1 | 0.8439 | +0.0235 | 0.0369 | +0.6373 | 1.0000 | 225.0000 | 0.6996 |
| fmnist | dir1 | label_flip | main |  | 1 | 48 | flirds_gatew_v1 | 0.8317 | +0.0069 | 0.0324 | +0.2124 | 1.0000 | 231.0000 | 0.6932 |
| fmnist | dir1 | label_flip | main | 0.1500 | 1 | 48 | flirds_gatew_v2 | 0.8488 | +0.0002 | 0.0087 |  | 0.8357 | 978.0000 | 0.7744 |
| fmnist | dir1 | label_flip | main | 0.3500 | 1 | 48 | flirds_gatew_v2 | 0.8520 | +0.0082 | 0.0135 |  | 0.9299 | 1055.0000 | 0.8027 |
| fmnist | dir1 | label_flip | main | 0.7000 | 1 | 48 | flirds_gatew_v2 | 0.8524 | +0.0320 | 0.0369 | +0.8678 | 0.9700 | 2188.0000 | 0.6925 |
| fmnist | dir1 | label_flip | main |  | 1 | 48 | flirds_gatew_v2 | 0.8474 | +0.0225 | 0.0324 | +0.6950 | 0.9535 | 1800.0000 | 0.7262 |
| fmnist | dir1 | label_flip | main | 0.1500 | 1 | 48 | flirds_mult | 0.8530 | +0.0045 | 0.0087 |  | 0.9095 |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 1 | 48 | flirds_mult | 0.8482 | +0.0045 | 0.0135 |  | 0.9788 |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 1 | 48 | flirds_mult | 0.8396 | +0.0192 | 0.0369 | +0.5220 | 0.9996 |  |  |
| fmnist | dir1 | label_flip | main |  | 1 | 48 | flirds_mult | 0.8415 | +0.0166 | 0.0324 | +0.5135 | 0.9992 |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 1 | 48 | flirds_zgate_v2 | 0.8500 | +0.0015 | 0.0087 |  | 0.9796 |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 1 | 48 | flirds_zgate_v2 | 0.8440 | +0.0002 | 0.0135 |  | 1.0000 |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 1 | 48 | flirds_zgate_v2 | 0.8077 | -0.0126 | 0.0369 | -0.3424 | 1.0000 |  |  |
| fmnist | dir1 | label_flip | main |  | 1 | 48 | flirds_zgate_v2 | 0.8283 | +0.0034 | 0.0324 | +0.1042 | 1.0000 |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 1 | 48 | oracle_excl | 0.8572 | +0.0087 | 0.0087 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 1 | 48 | oracle_excl | 0.8572 | +0.0135 | 0.0135 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 1 | 48 | oracle_excl | 0.8572 | +0.0369 | 0.0369 | +1.0000 |  |  |  |
| fmnist | dir1 | label_flip | main |  | 1 | 48 | oracle_excl | 0.8572 | +0.0324 | 0.0324 | +1.0000 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 1 | 48 | random_excl | 0.8490 | +0.0005 | 0.0087 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 1 | 48 | random_excl | 0.8369 | -0.0069 | 0.0135 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 1 | 48 | random_excl | 0.8140 | -0.0064 | 0.0369 | -0.1729 |  |  |  |
| fmnist | dir1 | label_flip | main |  | 1 | 48 | random_excl | 0.8090 | -0.0159 | 0.0324 | -0.4903 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 1 | 48 | vanilla | 0.8485 | +0.0000 | 0.0087 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 1 | 48 | vanilla | 0.8438 | +0.0000 | 0.0135 |  |  |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 1 | 48 | vanilla | 0.8204 | +0.0000 | 0.0369 | +0.0000 |  |  |  |
| fmnist | dir1 | label_flip | main |  | 1 | 48 | vanilla | 0.8249 | +0.0000 | 0.0324 | +0.0000 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 2 | 47 | flirds_gate_v1 | 0.8311 | +0.0637 | 0.0712 | +0.8947 | 0.9281 | 143.0000 | 0.7223 |
| fmnist | dir1 | label_flip | main | 0.3500 | 2 | 47 | flirds_gate_v1 | 0.8103 | +0.0200 | 0.0484 | +0.4134 | 0.9964 | 156.0000 | 0.7451 |
| fmnist | dir1 | label_flip | main | 0.7000 | 2 | 47 | flirds_gate_v1 | 0.7792 | +0.0169 | 0.0762 | +0.2213 | 1.0000 | 178.0000 | 0.7420 |
| fmnist | dir1 | label_flip | main |  | 2 | 47 | flirds_gate_v1 | 0.7780 | -0.0232 | 0.0374 | -0.6221 | 0.9996 | 180.0000 | 0.7421 |
| fmnist | dir1 | label_flip | main | 0.1500 | 2 | 47 | flirds_gate_v2 | 0.8386 | +0.0712 | 0.0712 | +1.0000 | 0.8442 | 730.0000 | 0.8048 |
| fmnist | dir1 | label_flip | main | 0.3500 | 2 | 47 | flirds_gate_v2 | 0.8276 | +0.0374 | 0.0484 | +0.7726 | 0.8699 | 1287.0000 | 0.7552 |
| fmnist | dir1 | label_flip | main | 0.7000 | 2 | 47 | flirds_gate_v2 | 0.8389 | +0.0765 | 0.0762 | +1.0033 | 0.9366 | 1173.0000 | 0.8014 |
| fmnist | dir1 | label_flip | main |  | 2 | 47 | flirds_gate_v2 | 0.8420 | +0.0407 | 0.0374 | +1.0903 | 0.9595 | 1406.0000 | 0.7731 |
| fmnist | dir1 | label_flip | main | 0.1500 | 2 | 47 | flirds_gatew_v1 | 0.8271 | +0.0597 | 0.0712 | +0.8386 | 0.9189 | 116.0000 | 0.7136 |
| fmnist | dir1 | label_flip | main | 0.3500 | 2 | 47 | flirds_gatew_v1 | 0.8205 | +0.0302 | 0.0484 | +0.6253 | 0.9912 | 119.0000 | 0.7772 |
| fmnist | dir1 | label_flip | main | 0.7000 | 2 | 47 | flirds_gatew_v1 | 0.7836 | +0.0212 | 0.0762 | +0.2787 | 1.0000 | 150.0000 | 0.7678 |
| fmnist | dir1 | label_flip | main |  | 2 | 47 | flirds_gatew_v1 | 0.8176 | +0.0164 | 0.0374 | +0.4381 | 1.0000 | 153.0000 | 0.7639 |
| fmnist | dir1 | label_flip | main | 0.1500 | 2 | 47 | flirds_gatew_v2 | 0.8434 | +0.0760 | 0.0712 | +1.0667 | 0.8326 | 788.0000 | 0.7938 |
| fmnist | dir1 | label_flip | main | 0.3500 | 2 | 47 | flirds_gatew_v2 | 0.8427 | +0.0525 | 0.0484 | +1.0853 | 0.8780 | 827.0000 | 0.8250 |
| fmnist | dir1 | label_flip | main | 0.7000 | 2 | 47 | flirds_gatew_v2 | 0.8355 | +0.0731 | 0.0762 | +0.9590 | 0.9558 | 1132.0000 | 0.8036 |
| fmnist | dir1 | label_flip | main |  | 2 | 47 | flirds_gatew_v2 | 0.8435 | +0.0423 | 0.0374 | +1.1304 | 0.9181 | 1158.0000 | 0.8001 |
| fmnist | dir1 | label_flip | main | 0.1500 | 2 | 47 | flirds_mult | 0.8156 | +0.0483 | 0.0712 | +0.6772 | 0.8752 |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 2 | 47 | flirds_mult | 0.8191 | +0.0289 | 0.0484 | +0.5969 | 0.9727 |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 2 | 47 | flirds_mult | 0.8076 | +0.0453 | 0.0762 | +0.5934 | 0.9960 |  |  |
| fmnist | dir1 | label_flip | main |  | 2 | 47 | flirds_mult | 0.8059 | +0.0046 | 0.0374 | +0.1237 | 0.9948 |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 2 | 47 | flirds_zgate_v2 | 0.8399 | +0.0725 | 0.0712 | +1.0175 | 0.9711 |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 2 | 47 | flirds_zgate_v2 | 0.8239 | +0.0336 | 0.0484 | +0.6951 | 0.9984 |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 2 | 47 | flirds_zgate_v2 | 0.8146 | +0.0523 | 0.0762 | +0.6852 | 1.0000 |  |  |
| fmnist | dir1 | label_flip | main |  | 2 | 47 | flirds_zgate_v2 | 0.8313 | +0.0300 | 0.0374 | +0.8027 | 1.0000 |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 2 | 47 | oracle_excl | 0.8386 | +0.0712 | 0.0712 | +1.0000 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 2 | 47 | oracle_excl | 0.8386 | +0.0484 | 0.0484 | +1.0000 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 2 | 47 | oracle_excl | 0.8386 | +0.0762 | 0.0762 | +1.0000 |  |  |  |
| fmnist | dir1 | label_flip | main |  | 2 | 47 | oracle_excl | 0.8386 | +0.0374 | 0.0374 | +1.0000 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 2 | 47 | random_excl | 0.8357 | +0.0684 | 0.0712 | +0.9596 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 2 | 47 | random_excl | 0.8315 | +0.0413 | 0.0484 | +0.8527 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 2 | 47 | random_excl | 0.8091 | +0.0467 | 0.0762 | +0.6131 |  |  |  |
| fmnist | dir1 | label_flip | main |  | 2 | 47 | random_excl | 0.8016 | +0.0004 | 0.0374 | +0.0100 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.1500 | 2 | 47 | vanilla | 0.7674 | +0.0000 | 0.0712 | +0.0000 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.3500 | 2 | 47 | vanilla | 0.7903 | +0.0000 | 0.0484 | +0.0000 |  |  |  |
| fmnist | dir1 | label_flip | main | 0.7000 | 2 | 47 | vanilla | 0.7624 | +0.0000 | 0.0762 | +0.0000 |  |  |  |
| fmnist | dir1 | label_flip | main |  | 2 | 47 | vanilla | 0.8013 | +0.0000 | 0.0374 | +0.0000 |  |  |  |
| fmnist | iid | clean | main |  | 0 | 0 | flirds_gate_v1 | 0.8518 | +0.0001 |  |  |  | 397.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 0 | 0 | flirds_gate_v2 | 0.8520 | +0.0004 |  |  |  | 828.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 0 | 0 | flirds_gatew_v1 | 0.8492 | -0.0024 |  |  |  | 298.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 0 | 0 | flirds_gatew_v2 | 0.8508 | -0.0009 |  |  |  | 648.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 0 | 0 | flirds_mult | 0.8529 | +0.0013 |  |  |  |  |  |
| fmnist | iid | clean | main |  | 0 | 0 | flirds_zgate_v2 | 0.8535 | +0.0019 |  |  |  |  |  |
| fmnist | iid | clean | main |  | 0 | 0 | vanilla | 0.8516 | +0.0000 |  |  |  |  |  |
| fmnist | iid | clean | main |  | 1 | 0 | flirds_gate_v1 | 0.8589 | -0.0020 |  |  |  | 381.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 1 | 0 | flirds_gate_v2 | 0.8621 | +0.0013 |  |  |  | 1663.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 1 | 0 | flirds_gatew_v1 | 0.8610 | +0.0001 |  |  |  | 304.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 1 | 0 | flirds_gatew_v2 | 0.8615 | +0.0006 |  |  |  | 420.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 1 | 0 | flirds_mult | 0.8600 | -0.0009 |  |  |  |  |  |
| fmnist | iid | clean | main |  | 1 | 0 | flirds_zgate_v2 | 0.8622 | +0.0014 |  |  |  |  |  |
| fmnist | iid | clean | main |  | 1 | 0 | vanilla | 0.8609 | +0.0000 |  |  |  |  |  |
| fmnist | iid | clean | main |  | 2 | 0 | flirds_gate_v1 | 0.8545 | -0.0007 |  |  |  | 444.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 2 | 0 | flirds_gate_v2 | 0.8519 | -0.0034 |  |  |  | 1135.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 2 | 0 | flirds_gatew_v1 | 0.8498 | -0.0055 |  |  |  | 348.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 2 | 0 | flirds_gatew_v2 | 0.8506 | -0.0046 |  |  |  | 220.0000 | 0.0000 |
| fmnist | iid | clean | main |  | 2 | 0 | flirds_mult | 0.8536 | -0.0016 |  |  |  |  |  |
| fmnist | iid | clean | main |  | 2 | 0 | flirds_zgate_v2 | 0.8521 | -0.0031 |  |  |  |  |  |
| fmnist | iid | clean | main |  | 2 | 0 | vanilla | 0.8552 | +0.0000 |  |  |  |  |  |
| fmnist | iid | free_rider | main |  | 0 | 40 | flirds_gate_v1 | 0.8501 | +0.0288 | 0.0290 | +0.9914 | 1.0000 | 140.0000 | 0.7742 |
| fmnist | iid | free_rider | main |  | 0 | 40 | flirds_gate_v2 | 0.8506 | +0.0292 | 0.0290 | +1.0086 | 0.9000 | 515.0000 | 0.8884 |
| fmnist | iid | free_rider | main |  | 0 | 40 | flirds_gatew_v1 | 0.8474 | +0.0260 | 0.0290 | +0.8966 | 0.9833 | 114.0000 | 0.8081 |
| fmnist | iid | free_rider | main |  | 0 | 40 | flirds_gatew_v2 | 0.8504 | +0.0290 | 0.0290 | +1.0000 | 0.8833 | 463.0000 | 0.8994 |
| fmnist | iid | free_rider | main |  | 0 | 40 | flirds_mult | 0.8375 | +0.0161 | 0.0290 | +0.5560 | 0.5092 |  |  |
| fmnist | iid | free_rider | main |  | 0 | 40 | flirds_zgate_v2 | 0.8165 | -0.0049 | 0.0290 | -0.1681 | 0.9833 |  |  |
| fmnist | iid | free_rider | main |  | 0 | 40 | oracle_excl | 0.8504 | +0.0290 | 0.0290 | +1.0000 |  |  |  |
| fmnist | iid | free_rider | main |  | 0 | 40 | random_excl | 0.8289 | +0.0075 | 0.0290 | +0.2586 |  |  |  |
| fmnist | iid | free_rider | main |  | 0 | 40 | vanilla | 0.8214 | +0.0000 | 0.0290 | +0.0000 |  |  |  |
| fmnist | iid | free_rider | main |  | 1 | 40 | flirds_gate_v1 | 0.8588 | +0.0218 | 0.0251 | +0.8657 | 0.9667 | 151.0000 | 0.7603 |
| fmnist | iid | free_rider | main |  | 1 | 40 | flirds_gate_v2 | 0.8619 | +0.0249 | 0.0251 | +0.9900 | 0.9167 | 430.0000 | 0.9022 |
| fmnist | iid | free_rider | main |  | 1 | 40 | flirds_gatew_v1 | 0.8599 | +0.0229 | 0.0251 | +0.9104 | 0.9833 | 128.0000 | 0.7891 |
| fmnist | iid | free_rider | main |  | 1 | 40 | flirds_gatew_v2 | 0.8541 | +0.0171 | 0.0251 | +0.6816 | 0.9667 | 198.0000 | 0.9523 |
| fmnist | iid | free_rider | main |  | 1 | 40 | flirds_mult | 0.8495 | +0.0125 | 0.0251 | +0.4975 | 0.5529 |  |  |
| fmnist | iid | free_rider | main |  | 1 | 40 | flirds_zgate_v2 | 0.8357 | -0.0012 | 0.0251 | -0.0498 | 0.9500 |  |  |
| fmnist | iid | free_rider | main |  | 1 | 40 | oracle_excl | 0.8621 | +0.0251 | 0.0251 | +1.0000 |  |  |  |
| fmnist | iid | free_rider | main |  | 1 | 40 | random_excl | 0.8436 | +0.0066 | 0.0251 | +0.2637 |  |  |  |
| fmnist | iid | free_rider | main |  | 1 | 40 | vanilla | 0.8370 | +0.0000 | 0.0251 | +0.0000 |  |  |  |
| fmnist | iid | free_rider | main |  | 2 | 40 | flirds_gate_v1 | 0.8465 | +0.0202 | 0.0289 | +0.7013 | 1.0000 | 139.0000 | 0.7790 |
| fmnist | iid | free_rider | main |  | 2 | 40 | flirds_gate_v2 | 0.8413 | +0.0150 | 0.0289 | +0.5195 | 0.8333 | 718.0000 | 0.8452 |
| fmnist | iid | free_rider | main |  | 2 | 40 | flirds_gatew_v1 | 0.8366 | +0.0104 | 0.0289 | +0.3593 | 1.0000 | 115.0000 | 0.8099 |
| fmnist | iid | free_rider | main |  | 2 | 40 | flirds_gatew_v2 | 0.8504 | +0.0241 | 0.0289 | +0.8355 | 0.8667 | 593.0000 | 0.8693 |
| fmnist | iid | free_rider | main |  | 2 | 40 | flirds_mult | 0.8344 | +0.0081 | 0.0289 | +0.2814 | 0.3904 |  |  |
| fmnist | iid | free_rider | main |  | 2 | 40 | flirds_zgate_v2 | 0.8246 | -0.0016 | 0.0289 | -0.0563 | 0.8833 |  |  |
| fmnist | iid | free_rider | main |  | 2 | 40 | oracle_excl | 0.8551 | +0.0289 | 0.0289 | +1.0000 |  |  |  |
| fmnist | iid | free_rider | main |  | 2 | 40 | random_excl | 0.8155 | -0.0108 | 0.0289 | -0.3723 |  |  |  |
| fmnist | iid | free_rider | main |  | 2 | 40 | vanilla | 0.8263 | +0.0000 | 0.0289 | +0.0000 |  |  |  |
| fmnist | iid | frrand | main |  | 0 | 40 | flirds_gate_v1 | 0.8269 | +0.0060 | 0.0295 | +0.2034 | 0.9487 | 198.0000 | 0.5733 |
| fmnist | iid | frrand | main |  | 0 | 40 | flirds_gate_v2 | 0.8407 | +0.0199 | 0.0295 | +0.6737 | 0.8733 | 746.0000 | 0.7882 |
| fmnist | iid | frrand | main |  | 0 | 40 | flirds_gatew_v1 | 0.8464 | +0.0255 | 0.0295 | +0.8644 | 0.9992 | 125.0000 | 0.6556 |
| fmnist | iid | frrand | main |  | 0 | 40 | flirds_gatew_v2 | 0.8491 | +0.0282 | 0.0295 | +0.9576 | 0.9671 | 44.0000 | 0.9846 |
| fmnist | iid | frrand | main |  | 0 | 40 | flirds_mult | 0.8355 | +0.0146 | 0.0295 | +0.4958 | 0.4858 |  |  |
| fmnist | iid | frrand | main |  | 0 | 40 | flirds_zgate_v2 | 0.8179 | -0.0030 | 0.0295 | -0.1017 | 0.9754 |  |  |
| fmnist | iid | frrand | main |  | 0 | 40 | oracle_excl | 0.8504 | +0.0295 | 0.0295 | +1.0000 |  |  |  |
| fmnist | iid | frrand | main |  | 0 | 40 | random_excl | 0.8290 | +0.0081 | 0.0295 | +0.2754 |  |  |  |
| fmnist | iid | frrand | main |  | 0 | 40 | vanilla | 0.8209 | +0.0000 | 0.0295 | +0.0000 |  |  |  |
| fmnist | iid | frrand | main |  | 1 | 40 | flirds_gate_v1 | 0.8440 | +0.0062 | 0.0244 | +0.2564 | 0.9487 | 205.0000 | 0.5255 |
| fmnist | iid | frrand | main |  | 1 | 40 | flirds_gate_v2 | 0.8562 | +0.0185 | 0.0244 | +0.7590 | 0.8729 | 485.0000 | 0.8681 |
| fmnist | iid | frrand | main |  | 1 | 40 | flirds_gatew_v1 | 0.8602 | +0.0225 | 0.0244 | +0.9231 | 0.9829 | 125.0000 | 0.6603 |
| fmnist | iid | frrand | main |  | 1 | 40 | flirds_gatew_v2 | 0.8615 | +0.0238 | 0.0244 | +0.9744 | 0.9504 | 204.0000 | 0.9280 |
| fmnist | iid | frrand | main |  | 1 | 40 | flirds_mult | 0.8495 | +0.0118 | 0.0244 | +0.4821 | 0.6025 |  |  |
| fmnist | iid | frrand | main |  | 1 | 40 | flirds_zgate_v2 | 0.8379 | +0.0001 | 0.0244 | +0.0051 | 0.9500 |  |  |
| fmnist | iid | frrand | main |  | 1 | 40 | oracle_excl | 0.8621 | +0.0244 | 0.0244 | +1.0000 |  |  |  |
| fmnist | iid | frrand | main |  | 1 | 40 | random_excl | 0.8426 | +0.0049 | 0.0244 | +0.2000 |  |  |  |
| fmnist | iid | frrand | main |  | 1 | 40 | vanilla | 0.8377 | +0.0000 | 0.0244 | +0.0000 |  |  |  |
| fmnist | iid | frrand | main |  | 2 | 40 | flirds_gate_v1 | 0.8301 | +0.0048 | 0.0298 | +0.1597 | 0.9154 | 228.0000 | 0.5512 |
| fmnist | iid | frrand | main |  | 2 | 40 | flirds_gate_v2 | 0.8459 | +0.0205 | 0.0298 | +0.6891 | 0.8037 | 689.0000 | 0.8227 |
| fmnist | iid | frrand | main |  | 2 | 40 | flirds_gatew_v1 | 0.8399 | +0.0145 | 0.0298 | +0.4874 | 1.0000 | 119.0000 | 0.6995 |
| fmnist | iid | frrand | main |  | 2 | 40 | flirds_gatew_v2 | 0.8530 | +0.0276 | 0.0298 | +0.9286 | 0.9833 | 51.0000 | 0.9839 |
| fmnist | iid | frrand | main |  | 2 | 40 | flirds_mult | 0.8337 | +0.0084 | 0.0298 | +0.2815 | 0.3750 |  |  |
| fmnist | iid | frrand | main |  | 2 | 40 | flirds_zgate_v2 | 0.8244 | -0.0010 | 0.0298 | -0.0336 | 0.8992 |  |  |
| fmnist | iid | frrand | main |  | 2 | 40 | oracle_excl | 0.8551 | +0.0298 | 0.0298 | +1.0000 |  |  |  |
| fmnist | iid | frrand | main |  | 2 | 40 | random_excl | 0.8156 | -0.0097 | 0.0298 | -0.3277 |  |  |  |
| fmnist | iid | frrand | main |  | 2 | 40 | vanilla | 0.8254 | +0.0000 | 0.0298 | +0.0000 |  |  |  |
| fmnist | iid | grad_noise | main |  | 0 | 40 | flirds_gate_v1 | 0.8306 | +0.0421 | 0.0619 | +0.6808 | 1.0000 | 162.0000 | 0.7259 |
| fmnist | iid | grad_noise | main |  | 0 | 40 | flirds_gate_v2 | 0.8684 | +0.0799 | 0.0619 | +1.2909 | 0.9950 | 1144.0000 | 0.7760 |
| fmnist | iid | grad_noise | main |  | 0 | 40 | flirds_gatew_v1 | 0.8321 | +0.0436 | 0.0619 | +0.7051 | 0.8054 | 173.0000 | 0.7112 |
| fmnist | iid | grad_noise | main |  | 0 | 40 | flirds_gatew_v2 | 0.8698 | +0.0813 | 0.0619 | +1.3131 | 0.9987 | 210.0000 | 0.9457 |
| fmnist | iid | grad_noise | main |  | 0 | 40 | flirds_mult | 0.8472 | +0.0587 | 0.0619 | +0.9495 | 1.0000 |  |  |
| fmnist | iid | grad_noise | main |  | 0 | 40 | flirds_zgate_v2 | 0.7844 | -0.0041 | 0.0619 | -0.0667 | 1.0000 |  |  |
| fmnist | iid | grad_noise | main |  | 0 | 40 | oracle_excl | 0.8504 | +0.0619 | 0.0619 | +1.0000 |  |  |  |
| fmnist | iid | grad_noise | main |  | 0 | 40 | random_excl | 0.8003 | +0.0118 | 0.0619 | +0.1899 |  |  |  |
| fmnist | iid | grad_noise | main |  | 0 | 40 | vanilla | 0.7885 | +0.0000 | 0.0619 | +0.0000 |  |  |  |
| fmnist | iid | grad_noise | main |  | 1 | 40 | flirds_gate_v1 | 0.8326 | +0.0657 | 0.0953 | +0.6903 | 1.0000 | 210.0000 | 0.6703 |
| fmnist | iid | grad_noise | main |  | 1 | 40 | flirds_gate_v2 | 0.8632 | +0.0964 | 0.0953 | +1.0118 | 0.9979 | 340.0000 | 0.9196 |
| fmnist | iid | grad_noise | main |  | 1 | 40 | flirds_gatew_v1 | 0.8219 | +0.0550 | 0.0953 | +0.5774 | 1.0000 | 166.0000 | 0.7186 |
| fmnist | iid | grad_noise | main |  | 1 | 40 | flirds_gatew_v2 | 0.8585 | +0.0916 | 0.0953 | +0.9619 | 1.0000 | 84.0000 | 0.9779 |
| fmnist | iid | grad_noise | main |  | 1 | 40 | flirds_mult | 0.8360 | +0.0691 | 0.0953 | +0.7257 | 1.0000 |  |  |
| fmnist | iid | grad_noise | main |  | 1 | 40 | flirds_zgate_v2 | 0.8119 | +0.0450 | 0.0953 | +0.4724 | 1.0000 |  |  |
| fmnist | iid | grad_noise | main |  | 1 | 40 | oracle_excl | 0.8621 | +0.0953 | 0.0953 | +1.0000 |  |  |  |
| fmnist | iid | grad_noise | main |  | 1 | 40 | random_excl | 0.7772 | +0.0104 | 0.0953 | +0.1089 |  |  |  |
| fmnist | iid | grad_noise | main |  | 1 | 40 | vanilla | 0.7669 | +0.0000 | 0.0953 | +0.0000 |  |  |  |
| fmnist | iid | grad_noise | main |  | 2 | 40 | flirds_gate_v1 | 0.8370 | +0.0441 | 0.0623 | +0.7088 | 0.8704 | 203.0000 | 0.6905 |
| fmnist | iid | grad_noise | main |  | 2 | 40 | flirds_gate_v2 | 0.8659 | +0.0730 | 0.0623 | +1.1727 | 0.9875 | 1160.0000 | 0.7714 |
| fmnist | iid | grad_noise | main |  | 2 | 40 | flirds_gatew_v1 | 0.8420 | +0.0491 | 0.0623 | +0.7892 | 1.0000 | 186.0000 | 0.7019 |
| fmnist | iid | grad_noise | main |  | 2 | 40 | flirds_gatew_v2 | 0.8678 | +0.0749 | 0.0623 | +1.2028 | 0.9983 | 955.0000 | 0.8019 |
| fmnist | iid | grad_noise | main |  | 2 | 40 | flirds_mult | 0.8083 | +0.0154 | 0.0623 | +0.2470 | 1.0000 |  |  |
| fmnist | iid | grad_noise | main |  | 2 | 40 | flirds_zgate_v2 | 0.8294 | +0.0365 | 0.0623 | +0.5863 | 1.0000 |  |  |
| fmnist | iid | grad_noise | main |  | 2 | 40 | oracle_excl | 0.8551 | +0.0623 | 0.0623 | +1.0000 |  |  |  |
| fmnist | iid | grad_noise | main |  | 2 | 40 | random_excl | 0.7041 | -0.0887 | 0.0623 | -1.4257 |  |  |  |
| fmnist | iid | grad_noise | main |  | 2 | 40 | vanilla | 0.7929 | +0.0000 | 0.0623 | +0.0000 |  |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 0 | 39 | flirds_gate_v1 | 0.8502 | +0.0039 | 0.0091 |  | 0.9861 | 153.0000 | 0.7156 |
| fmnist | iid | label_flip | main | 0.3500 | 0 | 39 | flirds_gate_v1 | 0.8391 | +0.0024 | 0.0188 |  | 1.0000 | 171.0000 | 0.7116 |
| fmnist | iid | label_flip | main | 0.7000 | 0 | 39 | flirds_gate_v1 | 0.8296 | +0.0076 | 0.0335 | +0.2276 | 1.0000 | 186.0000 | 0.7005 |
| fmnist | iid | label_flip | main |  | 0 | 39 | flirds_gate_v1 | 0.8324 | +0.0115 | 0.0346 | +0.3321 | 1.0000 | 180.0000 | 0.7078 |
| fmnist | iid | label_flip | main | 0.1500 | 0 | 39 | flirds_gate_v2 | 0.8454 | -0.0010 | 0.0091 |  | 0.9369 | 331.0000 | 0.8985 |
| fmnist | iid | label_flip | main | 0.3500 | 0 | 39 | flirds_gate_v2 | 0.8516 | +0.0149 | 0.0188 |  | 0.9874 | 750.0000 | 0.8325 |
| fmnist | iid | label_flip | main | 0.7000 | 0 | 39 | flirds_gate_v2 | 0.8490 | +0.0270 | 0.0335 | +0.8060 | 1.0000 | 509.0000 | 0.8839 |
| fmnist | iid | label_flip | main |  | 0 | 39 | flirds_gate_v2 | 0.8538 | +0.0329 | 0.0346 | +0.9495 | 0.9992 | 886.0000 | 0.8127 |
| fmnist | iid | label_flip | main | 0.1500 | 0 | 39 | flirds_gatew_v1 | 0.8484 | +0.0020 | 0.0091 |  | 0.9807 | 136.0000 | 0.7275 |
| fmnist | iid | label_flip | main | 0.3500 | 0 | 39 | flirds_gatew_v1 | 0.8417 | +0.0050 | 0.0188 |  | 1.0000 | 156.0000 | 0.7277 |
| fmnist | iid | label_flip | main | 0.7000 | 0 | 39 | flirds_gatew_v1 | 0.8354 | +0.0134 | 0.0335 | +0.3993 | 1.0000 | 179.0000 | 0.7122 |
| fmnist | iid | label_flip | main |  | 0 | 39 | flirds_gatew_v1 | 0.8407 | +0.0199 | 0.0346 | +0.5740 | 1.0000 | 180.0000 | 0.7115 |
| fmnist | iid | label_flip | main | 0.1500 | 0 | 39 | flirds_gatew_v2 | 0.8510 | +0.0046 | 0.0091 |  | 0.9546 | 171.0000 | 0.9420 |
| fmnist | iid | label_flip | main | 0.3500 | 0 | 39 | flirds_gatew_v2 | 0.8500 | +0.0132 | 0.0188 |  | 0.9983 | 95.0000 | 0.9750 |
| fmnist | iid | label_flip | main | 0.7000 | 0 | 39 | flirds_gatew_v2 | 0.8535 | +0.0315 | 0.0335 | +0.9403 | 0.9983 | 203.0000 | 0.9499 |
| fmnist | iid | label_flip | main |  | 0 | 39 | flirds_gatew_v2 | 0.8549 | +0.0340 | 0.0346 | +0.9819 | 0.9996 | 379.0000 | 0.9099 |
| fmnist | iid | label_flip | main | 0.1500 | 0 | 39 | flirds_mult | 0.8500 | +0.0036 | 0.0091 |  | 0.9933 |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 0 | 39 | flirds_mult | 0.8485 | +0.0118 | 0.0188 |  | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 0 | 39 | flirds_mult | 0.8486 | +0.0266 | 0.0335 | +0.7948 | 1.0000 |  |  |
| fmnist | iid | label_flip | main |  | 0 | 39 | flirds_mult | 0.8448 | +0.0239 | 0.0346 | +0.6895 | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 0 | 39 | flirds_zgate_v2 | 0.8486 | +0.0022 | 0.0091 |  | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 0 | 39 | flirds_zgate_v2 | 0.8410 | +0.0042 | 0.0188 |  | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 0 | 39 | flirds_zgate_v2 | 0.8354 | +0.0134 | 0.0335 | +0.3993 | 1.0000 |  |  |
| fmnist | iid | label_flip | main |  | 0 | 39 | flirds_zgate_v2 | 0.8351 | +0.0142 | 0.0346 | +0.4116 | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 0 | 39 | oracle_excl | 0.8555 | +0.0091 | 0.0091 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 0 | 39 | oracle_excl | 0.8555 | +0.0188 | 0.0188 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 0 | 39 | oracle_excl | 0.8555 | +0.0335 | 0.0335 | +1.0000 |  |  |  |
| fmnist | iid | label_flip | main |  | 0 | 39 | oracle_excl | 0.8555 | +0.0346 | 0.0346 | +1.0000 |  |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 0 | 39 | random_excl | 0.8474 | +0.0010 | 0.0091 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 0 | 39 | random_excl | 0.8404 | +0.0036 | 0.0188 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 0 | 39 | random_excl | 0.8283 | +0.0063 | 0.0335 | +0.1866 |  |  |  |
| fmnist | iid | label_flip | main |  | 0 | 39 | random_excl | 0.8284 | +0.0075 | 0.0346 | +0.2166 |  |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 0 | 39 | vanilla | 0.8464 | +0.0000 | 0.0091 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 0 | 39 | vanilla | 0.8367 | +0.0000 | 0.0188 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 0 | 39 | vanilla | 0.8220 | +0.0000 | 0.0335 | +0.0000 |  |  |  |
| fmnist | iid | label_flip | main |  | 0 | 39 | vanilla | 0.8209 | +0.0000 | 0.0346 | +0.0000 |  |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 1 | 48 | flirds_gate_v1 | 0.8569 | +0.0014 | 0.0051 |  | 0.9876 | 130.0000 | 0.7900 |
| fmnist | iid | label_flip | main | 0.3500 | 1 | 48 | flirds_gate_v1 | 0.8562 | +0.0060 | 0.0104 |  | 1.0000 | 168.0000 | 0.7551 |
| fmnist | iid | label_flip | main | 0.7000 | 1 | 48 | flirds_gate_v1 | 0.8556 | +0.0383 | 0.0433 | +0.8844 | 1.0000 | 188.0000 | 0.7333 |
| fmnist | iid | label_flip | main |  | 1 | 48 | flirds_gate_v1 | 0.8552 | +0.0396 | 0.0450 | +0.8806 | 1.0000 | 191.0000 | 0.7355 |
| fmnist | iid | label_flip | main | 0.1500 | 1 | 48 | flirds_gate_v2 | 0.8628 | +0.0072 | 0.0051 |  | 0.9635 | 273.0000 | 0.9320 |
| fmnist | iid | label_flip | main | 0.3500 | 1 | 48 | flirds_gate_v2 | 0.8614 | +0.0111 | 0.0104 |  | 0.9980 | 267.0000 | 0.9447 |
| fmnist | iid | label_flip | main | 0.7000 | 1 | 48 | flirds_gate_v2 | 0.8595 | +0.0421 | 0.0433 | +0.9740 | 0.9992 | 501.0000 | 0.9043 |
| fmnist | iid | label_flip | main |  | 1 | 48 | flirds_gate_v2 | 0.8612 | +0.0456 | 0.0450 | +1.0139 | 0.9944 | 875.0000 | 0.8453 |
| fmnist | iid | label_flip | main | 0.1500 | 1 | 48 | flirds_gatew_v1 | 0.8548 | -0.0008 | 0.0051 |  | 0.9856 | 120.0000 | 0.7973 |
| fmnist | iid | label_flip | main | 0.3500 | 1 | 48 | flirds_gatew_v1 | 0.8569 | +0.0066 | 0.0104 |  | 1.0000 | 150.0000 | 0.7758 |
| fmnist | iid | label_flip | main | 0.7000 | 1 | 48 | flirds_gatew_v1 | 0.8575 | +0.0401 | 0.0433 | +0.9277 | 1.0000 | 181.0000 | 0.7461 |
| fmnist | iid | label_flip | main |  | 1 | 48 | flirds_gatew_v1 | 0.8601 | +0.0445 | 0.0450 | +0.9889 | 1.0000 | 180.0000 | 0.7483 |
| fmnist | iid | label_flip | main | 0.1500 | 1 | 48 | flirds_gatew_v2 | 0.8602 | +0.0047 | 0.0051 |  | 0.9671 | 77.0000 | 0.9804 |
| fmnist | iid | label_flip | main | 0.3500 | 1 | 48 | flirds_gatew_v2 | 0.8601 | +0.0099 | 0.0104 |  | 0.9964 | 210.0000 | 0.9574 |
| fmnist | iid | label_flip | main | 0.7000 | 1 | 48 | flirds_gatew_v2 | 0.8616 | +0.0443 | 0.0433 | +1.0231 | 1.0000 | 237.0000 | 0.9524 |
| fmnist | iid | label_flip | main |  | 1 | 48 | flirds_gatew_v2 | 0.8584 | +0.0427 | 0.0450 | +0.9500 | 0.9972 | 668.0000 | 0.8784 |
| fmnist | iid | label_flip | main | 0.1500 | 1 | 48 | flirds_mult | 0.8601 | +0.0046 | 0.0051 |  | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 1 | 48 | flirds_mult | 0.8585 | +0.0083 | 0.0104 |  | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 1 | 48 | flirds_mult | 0.8564 | +0.0390 | 0.0433 | +0.9017 | 1.0000 |  |  |
| fmnist | iid | label_flip | main |  | 1 | 48 | flirds_mult | 0.8501 | +0.0345 | 0.0450 | +0.7667 | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 1 | 48 | flirds_zgate_v2 | 0.8556 | +0.0001 | 0.0051 |  | 0.9968 |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 1 | 48 | flirds_zgate_v2 | 0.8480 | -0.0022 | 0.0104 |  | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 1 | 48 | flirds_zgate_v2 | 0.8220 | +0.0046 | 0.0433 | +0.1069 | 1.0000 |  |  |
| fmnist | iid | label_flip | main |  | 1 | 48 | flirds_zgate_v2 | 0.8316 | +0.0160 | 0.0450 | +0.3556 | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 1 | 48 | oracle_excl | 0.8606 | +0.0051 | 0.0051 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 1 | 48 | oracle_excl | 0.8606 | +0.0104 | 0.0104 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 1 | 48 | oracle_excl | 0.8606 | +0.0433 | 0.0433 | +1.0000 |  |  |  |
| fmnist | iid | label_flip | main |  | 1 | 48 | oracle_excl | 0.8606 | +0.0450 | 0.0450 | +1.0000 |  |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 1 | 48 | random_excl | 0.8535 | -0.0020 | 0.0051 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 1 | 48 | random_excl | 0.8448 | -0.0055 | 0.0104 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 1 | 48 | random_excl | 0.8217 | +0.0044 | 0.0433 | +0.1012 |  |  |  |
| fmnist | iid | label_flip | main |  | 1 | 48 | random_excl | 0.8196 | +0.0040 | 0.0450 | +0.0889 |  |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 1 | 48 | vanilla | 0.8555 | +0.0000 | 0.0051 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 1 | 48 | vanilla | 0.8502 | +0.0000 | 0.0104 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 1 | 48 | vanilla | 0.8174 | +0.0000 | 0.0433 | +0.0000 |  |  |  |
| fmnist | iid | label_flip | main |  | 1 | 48 | vanilla | 0.8156 | +0.0000 | 0.0450 | +0.0000 |  |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 2 | 47 | flirds_gate_v1 | 0.8435 | -0.0029 | 0.0085 |  | 0.9699 | 119.0000 | 0.7860 |
| fmnist | iid | label_flip | main | 0.3500 | 2 | 47 | flirds_gate_v1 | 0.8439 | +0.0069 | 0.0179 |  | 1.0000 | 149.0000 | 0.7690 |
| fmnist | iid | label_flip | main | 0.7000 | 2 | 47 | flirds_gate_v1 | 0.8425 | +0.0333 | 0.0456 | +0.7288 | 1.0000 | 161.0000 | 0.7608 |
| fmnist | iid | label_flip | main |  | 2 | 47 | flirds_gate_v1 | 0.8449 | +0.0294 | 0.0394 | +0.7460 | 1.0000 | 165.0000 | 0.7581 |
| fmnist | iid | label_flip | main | 0.1500 | 2 | 47 | flirds_gate_v2 | 0.8518 | +0.0054 | 0.0085 |  | 0.9189 | 330.0000 | 0.9177 |
| fmnist | iid | label_flip | main | 0.3500 | 2 | 47 | flirds_gate_v2 | 0.8482 | +0.0112 | 0.0179 |  | 0.9976 | 547.0000 | 0.8910 |
| fmnist | iid | label_flip | main | 0.7000 | 2 | 47 | flirds_gate_v2 | 0.8489 | +0.0396 | 0.0456 | +0.8685 | 0.9980 | 736.0000 | 0.8622 |
| fmnist | iid | label_flip | main |  | 2 | 47 | flirds_gate_v2 | 0.8465 | +0.0310 | 0.0394 | +0.7873 | 0.9956 | 404.0000 | 0.9200 |
| fmnist | iid | label_flip | main | 0.1500 | 2 | 47 | flirds_gatew_v1 | 0.8458 | -0.0006 | 0.0085 |  | 0.9639 | 121.0000 | 0.7776 |
| fmnist | iid | label_flip | main | 0.3500 | 2 | 47 | flirds_gatew_v1 | 0.8449 | +0.0079 | 0.0179 |  | 1.0000 | 146.0000 | 0.7701 |
| fmnist | iid | label_flip | main | 0.7000 | 2 | 47 | flirds_gatew_v1 | 0.8474 | +0.0381 | 0.0456 | +0.8356 | 1.0000 | 166.0000 | 0.7573 |
| fmnist | iid | label_flip | main |  | 2 | 47 | flirds_gatew_v1 | 0.8471 | +0.0316 | 0.0394 | +0.8032 | 1.0000 | 168.0000 | 0.7576 |
| fmnist | iid | label_flip | main | 0.1500 | 2 | 47 | flirds_gatew_v2 | 0.8489 | +0.0025 | 0.0085 |  | 0.9570 | 179.0000 | 0.9466 |
| fmnist | iid | label_flip | main | 0.3500 | 2 | 47 | flirds_gatew_v2 | 0.8479 | +0.0109 | 0.0179 |  | 0.9928 | 0.0000 | 1.0000 |
| fmnist | iid | label_flip | main | 0.7000 | 2 | 47 | flirds_gatew_v2 | 0.8456 | +0.0364 | 0.0456 | +0.7973 | 0.9992 | 400.0000 | 0.9216 |
| fmnist | iid | label_flip | main |  | 2 | 47 | flirds_gatew_v2 | 0.8461 | +0.0306 | 0.0394 | +0.7778 | 0.9976 | 294.0000 | 0.9414 |
| fmnist | iid | label_flip | main | 0.1500 | 2 | 47 | flirds_mult | 0.8460 | -0.0004 | 0.0085 |  | 0.9996 |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 2 | 47 | flirds_mult | 0.8425 | +0.0055 | 0.0179 |  | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 2 | 47 | flirds_mult | 0.8410 | +0.0317 | 0.0456 | +0.6959 | 1.0000 |  |  |
| fmnist | iid | label_flip | main |  | 2 | 47 | flirds_mult | 0.8401 | +0.0246 | 0.0394 | +0.6254 | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 2 | 47 | flirds_zgate_v2 | 0.8465 | +0.0001 | 0.0085 |  | 0.9984 |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 2 | 47 | flirds_zgate_v2 | 0.8366 | -0.0004 | 0.0179 |  | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 2 | 47 | flirds_zgate_v2 | 0.8136 | +0.0044 | 0.0456 | +0.0959 | 1.0000 |  |  |
| fmnist | iid | label_flip | main |  | 2 | 47 | flirds_zgate_v2 | 0.8310 | +0.0155 | 0.0394 | +0.3937 | 1.0000 |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 2 | 47 | oracle_excl | 0.8549 | +0.0085 | 0.0085 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 2 | 47 | oracle_excl | 0.8549 | +0.0179 | 0.0179 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 2 | 47 | oracle_excl | 0.8549 | +0.0456 | 0.0456 | +1.0000 |  |  |  |
| fmnist | iid | label_flip | main |  | 2 | 47 | oracle_excl | 0.8549 | +0.0394 | 0.0394 | +1.0000 |  |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 2 | 47 | random_excl | 0.8528 | +0.0064 | 0.0085 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 2 | 47 | random_excl | 0.8347 | -0.0022 | 0.0179 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 2 | 47 | random_excl | 0.8004 | -0.0089 | 0.0456 | -0.1945 |  |  |  |
| fmnist | iid | label_flip | main |  | 2 | 47 | random_excl | 0.7996 | -0.0159 | 0.0394 | -0.4032 |  |  |  |
| fmnist | iid | label_flip | main | 0.1500 | 2 | 47 | vanilla | 0.8464 | +0.0000 | 0.0085 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.3500 | 2 | 47 | vanilla | 0.8370 | +0.0000 | 0.0179 |  |  |  |  |
| fmnist | iid | label_flip | main | 0.7000 | 2 | 47 | vanilla | 0.8093 | +0.0000 | 0.0456 | +0.0000 |  |  |  |
| fmnist | iid | label_flip | main |  | 2 | 47 | vanilla | 0.8155 | +0.0000 | 0.0394 | +0.0000 |  |  |  |
| mnist | dir1 | free_rider | main |  | 0 | 40 | oracle_excl | 0.9799 | +0.0069 | 0.0069 |  |  |  |  |
| mnist | dir1 | free_rider | main |  | 0 | 40 | random_excl | 0.9716 | -0.0014 | 0.0069 |  |  |  |  |
| mnist | dir1 | free_rider | main |  | 0 | 40 | vanilla | 0.9730 | +0.0000 | 0.0069 |  |  |  |  |
| mnist | dir1 | free_rider | main |  | 1 | 40 | oracle_excl | 0.9772 | +0.0054 | 0.0054 |  |  |  |  |
| mnist | dir1 | free_rider | main |  | 1 | 40 | random_excl | 0.9729 | +0.0010 | 0.0054 |  |  |  |  |
| mnist | dir1 | free_rider | main |  | 1 | 40 | vanilla | 0.9719 | +0.0000 | 0.0054 |  |  |  |  |
| mnist | dir1 | free_rider | main |  | 2 | 40 | oracle_excl | 0.9758 | +0.0068 | 0.0068 |  |  |  |  |
| mnist | dir1 | free_rider | main |  | 2 | 40 | random_excl | 0.9688 | -0.0002 | 0.0068 |  |  |  |  |
| mnist | dir1 | free_rider | main |  | 2 | 40 | vanilla | 0.9690 | +0.0000 | 0.0068 |  |  |  |  |
| mnist | dir1 | grad_noise | main |  | 0 | 40 | oracle_excl | 0.9799 | +0.0750 | 0.0750 | +1.0000 |  |  |  |
| mnist | dir1 | grad_noise | main |  | 0 | 40 | random_excl | 0.8902 | -0.0146 | 0.0750 | -0.1950 |  |  |  |
| mnist | dir1 | grad_noise | main |  | 0 | 40 | vanilla | 0.9049 | +0.0000 | 0.0750 | +0.0000 |  |  |  |
| mnist | dir1 | grad_noise | main |  | 1 | 40 | oracle_excl | 0.9772 | +0.0735 | 0.0735 | +1.0000 |  |  |  |
| mnist | dir1 | grad_noise | main |  | 1 | 40 | random_excl | 0.9199 | +0.0161 | 0.0735 | +0.2194 |  |  |  |
| mnist | dir1 | grad_noise | main |  | 1 | 40 | vanilla | 0.9038 | +0.0000 | 0.0735 | +0.0000 |  |  |  |
| mnist | dir1 | grad_noise | main |  | 2 | 40 | oracle_excl | 0.9758 | +0.0951 | 0.0951 | +1.0000 |  |  |  |
| mnist | dir1 | grad_noise | main |  | 2 | 40 | random_excl | 0.9024 | +0.0218 | 0.0951 | +0.2286 |  |  |  |
| mnist | dir1 | grad_noise | main |  | 2 | 40 | vanilla | 0.8806 | +0.0000 | 0.0951 | +0.0000 |  |  |  |
| mnist | dir1 | label_flip | main | 0.7000 | 0 | 39 | oracle_excl | 0.9812 | +0.0126 | 0.0126 |  |  |  |  |
| mnist | dir1 | label_flip | main | 0.7000 | 0 | 39 | random_excl | 0.9769 | +0.0083 | 0.0126 |  |  |  |  |
| mnist | dir1 | label_flip | main | 0.7000 | 0 | 39 | vanilla | 0.9686 | +0.0000 | 0.0126 |  |  |  |  |
| mnist | dir1 | label_flip | main | 0.7000 | 1 | 48 | oracle_excl | 0.9794 | +0.0162 | 0.0162 |  |  |  |  |
| mnist | dir1 | label_flip | main | 0.7000 | 1 | 48 | random_excl | 0.9606 | -0.0025 | 0.0162 |  |  |  |  |
| mnist | dir1 | label_flip | main | 0.7000 | 1 | 48 | vanilla | 0.9631 | +0.0000 | 0.0162 |  |  |  |  |
| mnist | dir1 | label_flip | main | 0.7000 | 2 | 47 | oracle_excl | 0.9754 | +0.0198 | 0.0198 |  |  |  |  |
| mnist | dir1 | label_flip | main | 0.7000 | 2 | 47 | random_excl | 0.9587 | +0.0031 | 0.0198 |  |  |  |  |
| mnist | dir1 | label_flip | main | 0.7000 | 2 | 47 | vanilla | 0.9556 | +0.0000 | 0.0198 |  |  |  |  |
| mnist | iid | free_rider | main |  | 0 | 40 | oracle_excl | 0.9815 | +0.0061 | 0.0061 |  |  |  |  |
| mnist | iid | free_rider | main |  | 0 | 40 | random_excl | 0.9778 | +0.0024 | 0.0061 |  |  |  |  |
| mnist | iid | free_rider | main |  | 0 | 40 | vanilla | 0.9754 | +0.0000 | 0.0061 |  |  |  |  |
| mnist | iid | free_rider | main |  | 1 | 40 | oracle_excl | 0.9809 | +0.0064 | 0.0064 |  |  |  |  |
| mnist | iid | free_rider | main |  | 1 | 40 | random_excl | 0.9764 | +0.0019 | 0.0064 |  |  |  |  |
| mnist | iid | free_rider | main |  | 1 | 40 | vanilla | 0.9745 | +0.0000 | 0.0064 |  |  |  |  |
| mnist | iid | free_rider | main |  | 2 | 40 | oracle_excl | 0.9798 | +0.0074 | 0.0074 |  |  |  |  |
| mnist | iid | free_rider | main |  | 2 | 40 | random_excl | 0.9694 | -0.0030 | 0.0074 |  |  |  |  |
| mnist | iid | free_rider | main |  | 2 | 40 | vanilla | 0.9724 | +0.0000 | 0.0074 |  |  |  |  |
| mnist | iid | grad_noise | main |  | 0 | 40 | oracle_excl | 0.9815 | +0.0533 | 0.0533 | +1.0000 |  |  |  |
| mnist | iid | grad_noise | main |  | 0 | 40 | random_excl | 0.9286 | +0.0004 | 0.0533 | +0.0070 |  |  |  |
| mnist | iid | grad_noise | main |  | 0 | 40 | vanilla | 0.9283 | +0.0000 | 0.0533 | +0.0000 |  |  |  |
| mnist | iid | grad_noise | main |  | 1 | 40 | oracle_excl | 0.9809 | +0.0558 | 0.0558 | +1.0000 |  |  |  |
| mnist | iid | grad_noise | main |  | 1 | 40 | random_excl | 0.9199 | -0.0052 | 0.0558 | -0.0942 |  |  |  |
| mnist | iid | grad_noise | main |  | 1 | 40 | vanilla | 0.9251 | +0.0000 | 0.0558 | +0.0000 |  |  |  |
| mnist | iid | grad_noise | main |  | 2 | 40 | oracle_excl | 0.9798 | +0.0667 | 0.0667 | +1.0000 |  |  |  |
| mnist | iid | grad_noise | main |  | 2 | 40 | random_excl | 0.8985 | -0.0145 | 0.0667 | -0.2172 |  |  |  |
| mnist | iid | grad_noise | main |  | 2 | 40 | vanilla | 0.9130 | +0.0000 | 0.0667 | +0.0000 |  |  |  |
| mnist | iid | label_flip | main | 0.7000 | 0 | 39 | oracle_excl | 0.9821 | +0.0172 | 0.0172 |  |  |  |  |
| mnist | iid | label_flip | main | 0.7000 | 0 | 39 | random_excl | 0.9769 | +0.0120 | 0.0172 |  |  |  |  |
| mnist | iid | label_flip | main | 0.7000 | 0 | 39 | vanilla | 0.9649 | +0.0000 | 0.0172 |  |  |  |  |
| mnist | iid | label_flip | main | 0.7000 | 1 | 48 | oracle_excl | 0.9816 | +0.0187 | 0.0187 |  |  |  |  |
| mnist | iid | label_flip | main | 0.7000 | 1 | 48 | random_excl | 0.9661 | +0.0032 | 0.0187 |  |  |  |  |
| mnist | iid | label_flip | main | 0.7000 | 1 | 48 | vanilla | 0.9629 | +0.0000 | 0.0187 |  |  |  |  |
| mnist | iid | label_flip | main | 0.7000 | 2 | 47 | oracle_excl | 0.9789 | +0.0173 | 0.0173 |  |  |  |  |
| mnist | iid | label_flip | main | 0.7000 | 2 | 47 | random_excl | 0.9606 | -0.0010 | 0.0173 |  |  |  |  |
| mnist | iid | label_flip | main | 0.7000 | 2 | 47 | vanilla | 0.9616 | +0.0000 | 0.0173 |  |  |  |  |

## V2w promotion gate (spec §5-2): **DO NOT PROMOTE (report CNN-only -- an honest finding)**

  cifar10/dir1/free_rider(str=main): V2w-V2 mean dAcc=-0.0092 FAIL
  cifar10/dir1/frrand(str=main): V2w-V2 mean dAcc=+0.0145 OK
  cifar10/dir1/grad_noise(str=main): V2w-V2 mean dAcc=+0.0206 OK
  cifar10/dir1/label_flip(str=main): V2w-V2 mean dAcc=+0.0079 OK
  cifar10/iid/free_rider(str=main): V2w-V2 mean dAcc=+0.0008 OK
  cifar10/iid/frrand(str=main): V2w-V2 mean dAcc=+0.0075 OK
  cifar10/iid/grad_noise(str=main): V2w-V2 mean dAcc=+0.0041 OK
  cifar10/iid/label_flip(str=main): V2w-V2 mean dAcc=-0.0081 FAIL
  cifar10/qskew/free_rider(str=main): V2w-V2 mean dAcc=-0.0008 FAIL
  cifar10/qskew/frrand(str=main): V2w-V2 mean dAcc=+0.0018 OK
  cifar10/qskew/grad_noise(str=main): V2w-V2 mean dAcc=-0.0308 FAIL
  cifar10/qskew/label_flip(str=main): V2w-V2 mean dAcc=+0.0100 OK
  cifar10/shard/free_rider(str=main): V2w-V2 mean dAcc=+0.0718 OK
  cifar10/shard/frrand(str=main): V2w-V2 mean dAcc=-0.0139 FAIL
  cifar10/shard/grad_noise(str=main): V2w-V2 mean dAcc=-0.0128 FAIL
  cifar10/shard/label_flip(str=main): V2w-V2 mean dAcc=+0.0382 OK
  fmnist/dir1/free_rider(str=main): V2w-V2 mean dAcc=+0.0032 OK
  fmnist/dir1/frrand(str=main): V2w-V2 mean dAcc=+0.0200 OK
  fmnist/dir1/grad_noise(str=main): V2w-V2 mean dAcc=-0.0030 FAIL
  fmnist/dir1/label_flip(str=main): V2w-V2 mean dAcc=+0.0034 OK
  fmnist/iid/free_rider(str=main): V2w-V2 mean dAcc=+0.0004 OK
  fmnist/iid/frrand(str=main): V2w-V2 mean dAcc=+0.0069 OK
  fmnist/iid/grad_noise(str=main): V2w-V2 mean dAcc=-0.0005 FAIL
  fmnist/iid/label_flip(str=main): V2w-V2 mean dAcc=-0.0001 FAIL
  clean cifar10_dir1_clean_g_seed0: V2w dAcc=-0.0200 FAIL(parity broken)
  clean cifar10_dir1_clean_g_seed1: V2w dAcc=-0.0100 FAIL(parity broken)
  clean cifar10_dir1_clean_g_seed2: V2w dAcc=-0.0302 FAIL(parity broken)
  clean cifar10_iid_clean_g_seed0: V2w dAcc=-0.0122 FAIL(parity broken)
  clean cifar10_iid_clean_g_seed1: V2w dAcc=-0.0016 OK
  clean cifar10_iid_clean_g_seed2: V2w dAcc=-0.0086 FAIL(parity broken)
  clean cifar10_qskew_clean_g_seed0: V2w dAcc=-0.0129 FAIL(parity broken)
  clean cifar10_qskew_clean_g_seed1: V2w dAcc=-0.0036 OK
  clean cifar10_qskew_clean_g_seed2: V2w dAcc=-0.0140 FAIL(parity broken)
  clean cifar10_shard_clean_g_seed0: V2w dAcc=-0.1120 FAIL(parity broken)
  clean cifar10_shard_clean_g_seed1: V2w dAcc=+0.0590 FAIL(parity broken)
  clean cifar10_shard_clean_g_seed2: V2w dAcc=-0.0944 FAIL(parity broken)
  clean fmnist_dir1_clean_g_seed0: V2w dAcc=+0.0495 FAIL(parity broken)
  clean fmnist_dir1_clean_g_seed1: V2w dAcc=-0.0100 FAIL(parity broken)
  clean fmnist_dir1_clean_g_seed2: V2w dAcc=+0.0620 FAIL(parity broken)
  clean fmnist_iid_clean_g_seed0: V2w dAcc=-0.0009 OK
  clean fmnist_iid_clean_g_seed1: V2w dAcc=+0.0006 OK
  clean fmnist_iid_clean_g_seed2: V2w dAcc=-0.0046 OK

## CNN skew 분해 (2×2: iid=skew없음 / shard=label만 / qskew=size만 / dir1=둘다) — 3-seed 평균

> ⚠️ 가법 분해 아님: shard의 label-skew(1.95 클래스/클라)는 dir1(9.87)보다, qskew의 size-skew(24×)는 dir1(6.2×)보다 세다. 축 귀속만 읽는다.

**cifar10 / clean** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.6459 | 0.4553 | 0.6579 | 0.6278 |
| flirds_gate_v2 | 0.6401 | 0.4533 | 0.6449 | 0.5905 |
| flirds_gatew_v1 | 0.6456 | 0.3397 | 0.6560 | 0.6184 |
| flirds_gatew_v2 | 0.6430 | 0.4260 | 0.6557 | 0.6205 |
| flirds_mult | 0.6460 | 0.4977 | 0.6636 | 0.6417 |
| flirds_zgate_v2 | 0.6482 | 0.4815 | 0.6634 | 0.6385 |
| vanilla | 0.6479 | 0.4751 | 0.6659 | 0.6380 |

**cifar10 / free_rider** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.6294 (+0.76) | 0.2893 (-1.59) | 0.6442 (+1.20) | 0.6146 (+0.88) |
| flirds_gate_v2 | 0.6314 (+0.85) | 0.3976 (+0.10) | 0.6340 (+0.42) | 0.5986 (+0.40) |
| flirds_gatew_v1 | 0.6294 (+0.75) | 0.2998 (-1.52) | 0.6447 (+0.87) | 0.6130 (+0.83) |
| flirds_gatew_v2 | 0.6340 (+0.97) | 0.4694 (+0.85) | 0.6331 (+0.40) | 0.6009 (+0.46) |
| flirds_mult | 0.6264 (+0.69) | 0.4165 (+0.28) | 0.6383 (+0.67) | 0.5967 (+0.29) |
| flirds_zgate_v2 | 0.6077 (-0.03) | 0.4315 (+0.28) | 0.6246 (-0.14) | 0.5844 (-0.10) |
| oracle_excl | 0.6352 (+1.00) | 0.4689 (+1.00) | 0.6460 (+1.00) | 0.6195 (+1.00) |
| random_excl | 0.5984 (-0.46) | 0.3675 (-0.22) | 0.6156 (-0.52) | 0.5840 (-0.05) |
| vanilla | 0.6084 (+0.00) | 0.3982 (+0.00) | 0.6295 (+0.00) | 0.5871 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0269, shard=0.0707, qskew=0.0165, dir1=0.0324

**cifar10 / frrand** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.6166 (+0.29) | 0.4352 (+0.74) | 0.6314 | 0.5953 (+0.23) |
| flirds_gate_v2 | 0.6233 (+0.51) | 0.4267 (+0.46) | 0.6329 | 0.5895 (+0.06) |
| flirds_gatew_v1 | 0.6305 (+0.80) | 0.3004 (-1.49) | 0.6439 | 0.6121 (+0.82) |
| flirds_gatew_v2 | 0.6308 (+0.83) | 0.4128 (+0.55) | 0.6347 | 0.6040 (+0.57) |
| flirds_mult | 0.6259 (+0.66) | 0.4156 (+0.28) | 0.6382 | 0.5966 (+0.27) |
| flirds_zgate_v2 | 0.6083 (-0.04) | 0.4449 (+0.75) | 0.6245 | 0.5857 (-0.10) |
| oracle_excl | 0.6352 (+1.00) | 0.4689 (+1.00) | 0.6460 | 0.6195 (+1.00) |
| random_excl | 0.5996 (-0.45) | 0.3679 (-0.19) | 0.6148 | 0.5839 (-0.07) |
| vanilla | 0.6087 (+0.00) | 0.3973 (+0.00) | 0.6303 | 0.5876 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0265, shard=0.0716, qskew=0.0157, dir1=0.0319

**cifar10 / grad_noise** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.5649 (+0.82) | 0.1416 (-0.08) | 0.4858 (+0.57) | 0.4597 (+0.58) |
| flirds_gate_v2 | 0.6155 (+0.95) | 0.3597 (+0.65) | 0.6300 (+0.96) | 0.5721 (+0.87) |
| flirds_gatew_v1 | 0.5492 (+0.77) | 0.1524 (-0.04) | 0.2463 (-0.05) | 0.4132 (+0.45) |
| flirds_gatew_v2 | 0.6181 (+0.95) | 0.3469 (+0.63) | 0.5991 (+0.87) | 0.5835 (+0.90) |
| flirds_mult | 0.5401 (+0.75) | 0.2843 (+0.39) | 0.4967 (+0.61) | 0.4333 (+0.50) |
| flirds_zgate_v2 | 0.3276 (+0.17) | 0.1792 (+0.04) | 0.4695 (+0.53) | 0.3516 (+0.29) |
| oracle_excl | 0.6352 (+1.00) | 0.4689 (+1.00) | 0.6460 (+1.00) | 0.6195 (+1.00) |
| random_excl | 0.2560 (-0.02) | 0.1697 (+0.02) | 0.2787 (+0.03) | 0.2481 (+0.01) |
| vanilla | 0.2627 (+0.00) | 0.1667 (+0.00) | 0.2675 (+0.00) | 0.2447 (+0.00) |

gap(oracle_excl−vanilla): iid=0.3725, shard=0.3022, qskew=0.3785, dir1=0.3748

**cifar10 / label_flip@0.15** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.6213 | 0.4319 (+0.75) | 0.6244 | 0.6161 |
| flirds_gate_v2 | 0.6130 | 0.3521 (+5.69) | 0.6277 | 0.5974 |
| flirds_gatew_v1 | 0.6197 | 0.3785 (+4.07) | 0.6303 | 0.6050 |
| flirds_gatew_v2 | 0.6200 | 0.4336 (+1.18) | 0.6239 | 0.5984 |
| flirds_mult | 0.6252 | 0.4552 (+0.27) | 0.6358 | 0.6165 |
| flirds_zgate_v2 | 0.6262 | 0.4430 (+0.86) | 0.6377 | 0.6114 |
| oracle_excl | 0.6305 | 0.4265 (+1.00) | 0.6427 | 0.6226 |
| random_excl | 0.6016 | 0.3759 (+1.34) | 0.6178 | 0.5932 |
| vanilla | 0.6286 | 0.4281 (+0.00) | 0.6375 | 0.6183 |

gap(oracle_excl−vanilla): iid=0.0019, shard=-0.0016, qskew=0.0052, dir1=0.0042

**cifar10 / label_flip@0.35** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.6159 (+0.65) | 0.3469 (-0.17) | 0.6103 (+0.29) | 0.6006 (+0.41) |
| flirds_gate_v2 | 0.6088 (+0.43) | 0.2492 (-1.36) | 0.5932 (-0.09) | 0.5777 (-0.26) |
| flirds_gatew_v1 | 0.6137 (+0.58) | 0.3412 (+0.82) | 0.6043 (+0.16) | 0.5902 (+0.11) |
| flirds_gatew_v2 | 0.6068 (+0.39) | 0.3109 (-2.12) | 0.6067 (+0.21) | 0.5875 (-0.01) |
| flirds_mult | 0.6155 (+0.62) | 0.3961 (-0.03) | 0.6176 (+0.45) | 0.6014 (+0.41) |
| flirds_zgate_v2 | 0.5953 (+0.11) | 0.4075 (+0.91) | 0.6156 (+0.40) | 0.5772 (-0.23) |
| oracle_excl | 0.6305 (+1.00) | 0.4265 (+1.00) | 0.6427 (+1.00) | 0.6226 (+1.00) |
| random_excl | 0.5702 (-0.53) | 0.3261 (-0.21) | 0.5863 (-0.22) | 0.5601 (-0.69) |
| vanilla | 0.5917 (+0.00) | 0.3762 (+0.00) | 0.5973 (+0.00) | 0.5861 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0388, shard=0.0503, qskew=0.0454, dir1=0.0365

**cifar10 / label_flip@0.7** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.5688 (+0.39) | 0.2887 (-0.29) | 0.5944 (+0.57) | 0.5904 (+0.66) |
| flirds_gate_v2 | 0.5795 (+0.57) | 0.2518 (-1.08) | 0.6028 (+0.65) | 0.5767 (+0.51) |
| flirds_gatew_v1 | 0.6034 (+0.74) | 0.3308 (+0.15) | 0.5795 (+0.41) | 0.5715 (+0.51) |
| flirds_gatew_v2 | 0.5855 (+0.58) | 0.2124 (-1.46) | 0.5854 (+0.51) | 0.5840 (+0.59) |
| flirds_mult | 0.6091 (+0.81) | 0.3147 (-0.16) | 0.5971 (+0.65) | 0.5882 (+0.64) |
| flirds_zgate_v2 | 0.5497 (+0.29) | 0.2717 (-0.70) | 0.5966 (+0.61) | 0.5400 (+0.18) |
| oracle_excl | 0.6305 (+1.00) | 0.4265 (+1.00) | 0.6427 (+1.00) | 0.6226 (+1.00) |
| random_excl | 0.5049 (-0.09) | 0.2783 (-0.43) | 0.5291 (+0.14) | 0.5036 (-0.21) |
| vanilla | 0.5192 (+0.00) | 0.3117 (+0.00) | 0.5157 (+0.00) | 0.5253 (+0.00) |

gap(oracle_excl−vanilla): iid=0.1113, shard=0.1148, qskew=0.1270, dir1=0.0973

**cifar10 / label_flip@strmain** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | shard | qskew | dir1 |
|---|---|---|---|---|
| flirds_gate_v1 | 0.5904 (+0.66) | 0.2820 (-0.31) | 0.5822 (+0.60) | 0.5894 (+0.65) |
| flirds_gate_v2 | 0.6096 (+0.82) | 0.2225 (-1.34) | 0.5480 (+0.39) | 0.5534 (+0.31) |
| flirds_gatew_v1 | 0.6089 (+0.81) | 0.3260 (+0.03) | 0.5704 (+0.27) | 0.5477 (+0.27) |
| flirds_gatew_v2 | 0.5830 (+0.57) | 0.2716 (-0.71) | 0.5959 (+0.67) | 0.5713 (+0.45) |
| flirds_mult | 0.6015 (+0.76) | 0.3150 (-0.15) | 0.5941 (+0.68) | 0.5849 (+0.64) |
| flirds_zgate_v2 | 0.5598 (+0.40) | 0.3157 (-0.38) | 0.5973 (+0.67) | 0.5531 (+0.34) |
| oracle_excl | 0.6305 (+1.00) | 0.4265 (+1.00) | 0.6427 (+1.00) | 0.6226 (+1.00) |
| random_excl | 0.4948 (-0.10) | 0.2717 (-0.57) | 0.5208 (+0.19) | 0.4928 (-0.21) |
| vanilla | 0.5099 (+0.00) | 0.3179 (+0.00) | 0.4948 (+0.00) | 0.5149 (+0.00) |

gap(oracle_excl−vanilla): iid=0.1206, shard=0.1086, qskew=0.1480, dir1=0.1077

**fmnist / clean** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8550 | 0.8388 |
| flirds_gate_v2 | 0.8553 | 0.8321 |
| flirds_gatew_v1 | 0.8533 | 0.8415 |
| flirds_gatew_v2 | 0.8543 | 0.8456 |
| flirds_mult | 0.8555 | 0.8293 |
| flirds_zgate_v2 | 0.8560 | 0.8482 |
| vanilla | 0.8559 | 0.8117 |

**fmnist / free_rider** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8518 (+0.85) | 0.8091 (+0.17) |
| flirds_gate_v2 | 0.8512 (+0.84) | 0.8389 (+0.91) |
| flirds_gatew_v1 | 0.8480 (+0.72) | 0.8097 (+0.10) |
| flirds_gatew_v2 | 0.8516 (+0.84) | 0.8421 (+1.11) |
| flirds_mult | 0.8405 (+0.44) | 0.8205 (+0.38) |
| flirds_zgate_v2 | 0.8256 (-0.09) | 0.8092 (+0.00) |
| oracle_excl | 0.8559 (+1.00) | 0.8418 (+1.00) |
| random_excl | 0.8293 (+0.05) | 0.8125 (+0.12) |
| vanilla | 0.8282 (+0.00) | 0.8081 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0277, dir1=0.0337

**fmnist / frrand** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8337 (+0.21) | 0.8037 (-0.13) |
| flirds_gate_v2 | 0.8476 (+0.71) | 0.8216 (+0.45) |
| flirds_gatew_v1 | 0.8488 (+0.76) | 0.8147 (+0.24) |
| flirds_gatew_v2 | 0.8545 (+0.95) | 0.8416 (+1.00) |
| flirds_mult | 0.8396 (+0.42) | 0.8198 (+0.35) |
| flirds_zgate_v2 | 0.8267 (-0.04) | 0.8078 (-0.06) |
| oracle_excl | 0.8559 (+1.00) | 0.8418 (+1.00) |
| random_excl | 0.8291 (+0.05) | 0.8130 (+0.12) |
| vanilla | 0.8280 (+0.00) | 0.8089 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0279, dir1=0.0329

**fmnist / grad_noise** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8334 (+0.69) | 0.7815 (+0.42) |
| flirds_gate_v2 | 0.8658 (+1.16) | 0.8561 (+1.16) |
| flirds_gatew_v1 | 0.8320 (+0.69) | 0.7682 (+0.25) |
| flirds_gatew_v2 | 0.8653 (+1.16) | 0.8532 (+1.12) |
| flirds_mult | 0.8305 (+0.64) | 0.7948 (+0.52) |
| flirds_zgate_v2 | 0.8085 (+0.33) | 0.7940 (+0.52) |
| oracle_excl | 0.8559 (+1.00) | 0.8418 (+1.00) |
| random_excl | 0.7605 (-0.38) | 0.7381 (-0.03) |
| vanilla | 0.7828 (+0.00) | 0.7400 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0731, dir1=0.1019

**fmnist / label_flip@0.15** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8502 | 0.7870 (+0.89) |
| flirds_gate_v2 | 0.8533 | 0.8405 (+1.00) |
| flirds_gatew_v1 | 0.8496 | 0.8358 (+0.84) |
| flirds_gatew_v2 | 0.8534 | 0.8427 (+1.07) |
| flirds_mult | 0.8520 | 0.8308 (+0.68) |
| flirds_zgate_v2 | 0.8502 | 0.8421 (+1.02) |
| oracle_excl | 0.8570 | 0.8387 (+1.00) |
| random_excl | 0.8512 | 0.8263 (+0.96) |
| vanilla | 0.8494 | 0.8126 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0076, dir1=0.0262

**fmnist / label_flip@0.35** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8464 | 0.8193 (+0.41) |
| flirds_gate_v2 | 0.8537 | 0.8304 (+0.77) |
| flirds_gatew_v1 | 0.8478 | 0.8260 (+0.63) |
| flirds_gatew_v2 | 0.8527 | 0.8421 (+1.09) |
| flirds_mult | 0.8498 | 0.8239 (+0.60) |
| flirds_zgate_v2 | 0.8419 | 0.8295 (+0.70) |
| oracle_excl | 0.8570 | 0.8387 (+1.00) |
| random_excl | 0.8400 | 0.8178 (+0.85) |
| vanilla | 0.8413 | 0.8154 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0157, dir1=0.0234

**fmnist / label_flip@0.7** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8426 (+0.61) | 0.7933 (-0.25) |
| flirds_gate_v2 | 0.8525 (+0.88) | 0.8418 (+1.14) |
| flirds_gatew_v1 | 0.8468 (+0.72) | 0.8059 (+0.22) |
| flirds_gatew_v2 | 0.8536 (+0.92) | 0.8420 (+1.19) |
| flirds_mult | 0.8487 (+0.80) | 0.8081 (+0.11) |
| flirds_zgate_v2 | 0.8237 (+0.20) | 0.8140 (+0.44) |
| oracle_excl | 0.8570 (+1.00) | 0.8387 (+1.00) |
| random_excl | 0.8168 (+0.03) | 0.7975 (-0.22) |
| vanilla | 0.8162 (+0.00) | 0.7929 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0408, dir1=0.0458

**fmnist / label_flip@strmain** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| flirds_gate_v1 | 0.8442 (+0.65) | 0.8044 (-0.14) |
| flirds_gate_v2 | 0.8538 (+0.92) | 0.8414 (+0.99) |
| flirds_gatew_v1 | 0.8493 (+0.79) | 0.8150 (+0.33) |
| flirds_gatew_v2 | 0.8531 (+0.90) | 0.8409 (+0.91) |
| flirds_mult | 0.8450 (+0.69) | 0.8060 (+0.32) |
| flirds_zgate_v2 | 0.8326 (+0.39) | 0.8225 (+0.45) |
| oracle_excl | 0.8570 (+1.00) | 0.8387 (+1.00) |
| random_excl | 0.8159 (-0.03) | 0.7923 (-0.24) |
| vanilla | 0.8173 (+0.00) | 0.8090 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0397, dir1=0.0298

**mnist / free_rider** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| oracle_excl | 0.9807 | 0.9776 |
| random_excl | 0.9745 | 0.9711 |
| vanilla | 0.9741 | 0.9713 |

gap(oracle_excl−vanilla): iid=0.0066, dir1=0.0063

**mnist / grad_noise** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| oracle_excl | 0.9807 (+1.00) | 0.9776 (+1.00) |
| random_excl | 0.9157 (-0.10) | 0.9042 (+0.08) |
| vanilla | 0.9221 (+0.00) | 0.8964 (+0.00) |

gap(oracle_excl−vanilla): iid=0.0586, dir1=0.0812

**mnist / label_flip@0.7** — 절대 acc (recovery; 분모<0.02 → 공란)

| arm | iid | dir1 |
|---|---|---|
| oracle_excl | 0.9809 | 0.9787 |
| random_excl | 0.9679 | 0.9654 |
| vanilla | 0.9631 | 0.9625 |

gap(oracle_excl−vanilla): iid=0.0178, dir1=0.0162

## 사전등록 예측 대조 (README 확장 ②; MISS 그대로 보고)

- **H-K1** cifar10 free_rider V2 recovery iid=+0.85, shard=+0.10, qskew=+0.42, dir1=+0.40 -> **MISS**
- **H-K2** cifar10 iid frrand V2 recovery=+0.51 (frzero=+0.85; ratio=+0.59 — <=0.6이면 LLM 감사의 코인플립과 일치) -> **MISS**
- **H-K3** cifar10 clean 오발화 pairs iid=655, shard=3383, qskew=1971, dir1=4025 | V2 dAcc iid=-0.0078, shard=-0.0218, qskew=-0.0210, dir1=-0.0475 -> **MISS**
- **H-K4** cifar10 free_rider recovery seed-sd iid=0.020 -> pending
- **H-K4** cifar10 grad_noise recovery seed-sd iid=0.046, qskew=0.026 -> **MISS**
- **H-K5** cifar10 lf@0.15 gap iid=0.0019, shard=-0.0016, qskew=0.0052, dir1=0.0042 -> **HIT**
- **H-K1** fmnist free_rider V2 recovery iid=+0.84, dir1=+0.91 -> pending
- **H-K2** fmnist iid frrand V2 recovery=+0.71 (frzero=+0.84; ratio=+0.84 — <=0.6이면 LLM 감사의 코인플립과 일치) -> **HIT**
- **H-K3** fmnist clean 오발화 pairs iid=1209, dir1=2358 | V2 dAcc iid=-0.0006, dir1=+0.0203 -> pending
- **H-K4** fmnist free_rider recovery seed-sd iid=0.277 -> pending
- **H-K4** fmnist grad_noise recovery seed-sd iid=0.140 -> pending
- **H-K5** fmnist lf@0.15 gap iid=0.0076, dir1=0.0262 -> **MISS**
- **H-K1** mnist free_rider V2 recovery  -> pending
- **H-K2** mnist iid frrand V2 recovery= -> pending
- **H-K3** mnist clean 오발화 pairs  | V2 dAcc  -> pending
- **H-K4** mnist free_rider recovery seed-sd  -> pending
- **H-K4** mnist grad_noise recovery seed-sd  -> pending
- **H-K5** mnist lf@0.15 gap  -> pending
- **H-K6** fmnist↔cifar10 recovery diff iid/free_rider=0.01, iid/frrand=0.20, iid/grad_noise=0.21, iid/label_flip=0.32, dir1/free_rider=0.50, dir1/frrand=0.39, dir1/grad_noise=0.28, dir1/label_flip=1.04, dir1/label_flip=0.63 -> **MISS**

## C2 소프트-arm 같은-셀 대조 (runs/track_c/c2, read-only)

| dataset | partition | threat | C2 vanilla | G vanilla | C2 flirds_mult | G flirds_gate_v2 | 비고 |
|---|---|---|---|---|---|---|---|
| cifar10 | dir1 | clean | 0.6380 | 0.6380 | 0.6417 | 0.5905 | same cell |
| cifar10 | dir1 | free_rider | 0.5871 | 0.5871 | 0.5967 | 0.5986 | same cell |
| cifar10 | dir1 | grad_noise | 0.2447 | 0.2447 | 0.4333 | 0.5721 | same cell |
| cifar10 | iid | clean | 0.6479 | 0.6479 | 0.6460 | 0.6401 | same cell |
| cifar10 | iid | free_rider | 0.6084 | 0.6084 | 0.6264 | 0.6314 | same cell |
| cifar10 | iid | grad_noise | 0.2627 | 0.2627 | 0.5401 | 0.6155 | same cell |
| cifar10 | shard | clean | 0.4751 | 0.4751 | 0.4977 | 0.4533 | same cell |
| cifar10 | shard | free_rider | 0.3982 | 0.3982 | 0.4165 | 0.3976 | same cell |
| cifar10 | shard | grad_noise | 0.1667 | 0.1667 | 0.2843 | 0.3597 | same cell |
| fmnist | dir1 | clean | 0.8117 | 0.8117 | 0.8293 | 0.8321 | same cell |
| fmnist | dir1 | free_rider | 0.8081 | 0.8081 | 0.8205 | 0.8389 | same cell |
| fmnist | dir1 | grad_noise | 0.7400 | 0.7400 | 0.7948 | 0.8561 | same cell |
| fmnist | iid | clean | 0.8559 | 0.8559 | 0.8555 | 0.8553 | same cell |
| fmnist | iid | free_rider | 0.8282 | 0.8282 | 0.8405 | 0.8512 | same cell |
| fmnist | iid | grad_noise | 0.7828 | 0.7828 | 0.8305 | 0.8658 | same cell |

⚠️ qskew·frrand는 C2 대응 셀 없음. label_flip은 C2가 strmain(rate~U(0.5,1))이라 Track G의 고정 dose와 같은 셀이 아니어서 제외.

## 스택 재현성 (동일 config·seed, 두 스택) — 감사 M1

| 레짐 | 셀×arm 수 | mean drift | max |drift| |
|---|---|---|---|
| clean | 42 | -0.0023 | 0.0952 |
| free-rider | 54 | -0.0001 | 0.0479 |
| grad-noise | 54 | -0.0124 | 0.2331 |
| label-flip | 162 | +0.0005 | 0.0701 |
| **전체** | 312 | -0.0022 | 0.2331 |

drift = restack(torch 2.11/RTX3090) − orig(torch 2.12/B200), 동일 config·seed. 표의 iid·dir1 행은 restack 값을 쓴다(단일 스택).

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