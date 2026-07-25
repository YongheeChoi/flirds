# W-B — Flirds P1 vs P1w, 확장 CNN 무대 (rundir-only; make_p1w_cnn_table.py)

> P1w ≡ 기존 P2(sign+크기가중). arm 라벨은 P2(gatew_v2/t2_signw) 유지, 표기만 P1w. dir1(W-A)은 참조로 병기. ⚠stack = cifar10 iid의 clean/fr/gn/lf T1 앵커는 B200 원본(track_g/rundirs_cnn) — W-A 드리프트≈0 판정 하에 병기.

- flirds rows: 804 (T2 retrain rows: 240 — 0이면 sbatch_cnn_p1w.sh 미실행, T1만 표시)

## 절대 acc (overview §3.2.3 스타일)

**online** — 절대 test acc (3-seed mean; P1=gate_v2 / P1w=gatew_v2 온라인)

| dataset/partition | policy | clean | free_rider | frrand | grad_noise | lf@0.7 | strmain |
|---|---|---|---|---|---|---|---|
| cifar10/shard | P1 | 0.4533 | 0.3976 | 0.4267 | 0.3597 | 0.2518 | 0.2225 |
| cifar10/shard | P1w | 0.4260 | 0.4694 | 0.4128 | 0.3469 | 0.2124 | 0.2716 |
| cifar10/qskew | P1 | 0.6449 | 0.6340 | 0.6329 | 0.6300 | 0.6028 | 0.5480 |
| cifar10/qskew | P1w | 0.6557 | 0.6331 | 0.6347 | 0.5991 | 0.5854 | 0.5959 |
| fmnist/iid | P1 | 0.8553 | 0.8512 | 0.8476 | 0.8658 | 0.8525 | 0.8538 |
| fmnist/iid | P1w | 0.8543 | 0.8516 | 0.8545 | 0.8653 | 0.8536 | 0.8531 |
| fmnist/dir1 | P1 | 0.8321 | 0.8389 | 0.8216 | 0.8561 | 0.8418 | 0.8414 |
| fmnist/dir1 | P1w | 0.8456 | 0.8421 | 0.8416 | 0.8532 | 0.8420 | 0.8409 |
| cifar10/iid ⚠stack | P1 | 0.6428 | 0.6308 | 0.6233 | 0.6143 | 0.5967 | 0.6096 |
| cifar10/iid ⚠stack | P1w | 0.6412 | 0.6317 | 0.6308 | 0.6185 | 0.6032 | 0.5830 |

**retrain** — 절대 test acc (3-seed mean; P1=gate_v2 / P1w=gatew_v2 재학습)

| dataset/partition | policy | clean | free_rider | frrand | grad_noise | lf@0.7 | strmain |
|---|---|---|---|---|---|---|---|
| cifar10/shard | P1 | 0.4410 | 0.4564 | 0.4308 | 0.4684 | 0.2520 | 0.2920 |
| cifar10/shard | P1w | 0.4020 | 0.4245 | 0.3761 | 0.4299 | 0.2907 | 0.2362 |
| cifar10/qskew | P1 | 0.6511 | 0.6402 | 0.6305 | 0.6167 | 0.6403 | 0.6418 |
| cifar10/qskew | P1w | 0.6415 | 0.6319 | 0.6322 | 0.6036 | 0.6366 | 0.6392 |
| fmnist/iid | P1 | 0.8548 | 0.8560 | 0.8432 | 0.8559 | 0.8570 | 0.8570 |
| fmnist/iid | P1w | 0.8521 | 0.8547 | 0.8549 | 0.8554 | 0.8555 | 0.8556 |
| fmnist/dir1 | P1 | 0.8485 | 0.8438 | 0.8218 | 0.8456 | 0.8387 | 0.8387 |
| fmnist/dir1 | P1w | 0.8453 | 0.8356 | 0.8331 | 0.8460 | 0.8399 | 0.8402 |
| cifar10/iid ⚠stack | P1 | 0.6419 | 0.6326 | 0.6180 | 0.6258 | 0.6305 | 0.6305 |
| cifar10/iid ⚠stack | P1w | 0.6315 | 0.6251 | 0.6216 | 0.6115 | 0.6292 | 0.6298 |

## H-5 재현 — 오염 recovery P1 vs P1w + clean parity

- **online**: 오염평균 acc  P1=+0.650  P1w=+0.653  **gap(P1w-P1)=+0.003** (dir1 참조 +0.007) | clean dAcc  P1=-0.006  P1w=-0.007 (band +/-0.006)
    - recovery(guard|orc-van|>=0.02, dropped 4 cells): P1=+0.541  P1w=+0.693  gap=+0.152
- **retrain**: 오염평균 acc  P1=+0.667  P1w=+0.660  **gap(P1w-P1)=-0.007** (dir1 참조 -0.015) | clean dAcc  P1=-0.004  P1w=-0.017 (band +/-0.006)
    - recovery(guard|orc-van|>=0.02, dropped 4 cells): P1=+0.792  P1w=+0.741  gap=-0.050

## FedIF 역전 (00-INDEX §1 '타 소스 역전' 조항)

- dir1 (W-A, on disk): FedIF P1w 오염평균 online .6011 / retrain .6159 **> flirds .5913 / .5959** → 역전 확인('타 소스 역전' 조항 해당).
- 확장 무대: FedIF 셀 감지됨 — make_analysis competition CSV에서 fedif vs flirds P1w recovery 직접 대조 가능(W-D 착지).

## 판정 초안 (수록 규칙 = 00-INDEX §1)

> 사전 고정 규칙: 전 범위(W-A·W-B·L7)에서 이기면 본문 승격 / 동률이면 '부호가 가치의 대부분' ablation 1문장 / 열세·타 소스 역전 시 미수록(P1만).
> W-B 단독 판정 금지 — L7(LLM P1w)·W-A 종합 후 확정. 위 gap·역전으로 초안만.