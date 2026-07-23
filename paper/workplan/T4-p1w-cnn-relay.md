# T4 — P1w CNN 검증 (CNN 서버 세션 전달 스펙; W-A + W-B)

> 수신처: CNN 캠페인 서버 세션. 정책 정의·수록 규칙 = `00-INDEX.md` §1 + `T3-p1w-llm-impl.md` 서두.
> 핵심: **P1w ≡ Track H의 기존 P2(sign_weight)** — dir1은 재실행하지 않는다.

## W-A — dir1 기존 P2 실측 재사용 (재실행 0)

1. 캠페인 **restack 드리프트 표** 확인(cifar10 iid·dir1 12셀): 드리프트 ≈ 0이면 기존
   `runs/track_h/rundirs_cnn`의 **P2 arm(T1·T2, 8점수원, 4위협×3-seed)** 값을 P1w 결과로 귀속.
   드리프트 유의 시 dir1 P2 재실행 여부를 비용과 함께 보고(기본 = 재실행 금지, 보고 후 결정).
2. 알려진 실측(귀속 시 그대로 보고; overview §3.2.3 P2 표): flirds 오염 T1 .5913(P1 .5843 대비 +0.7pt) ·
   **오염 T2 .5959(P1 .6107 대비 −1.5pt)** · clean T1 .6188(P1 .6315 대비 −1.3pt); ⚠ **FedIF가 flirds 상회**
   (P2 오염평균 on .6011 / re .6159) — 수록 규칙의 "타 소스 역전" 조항에 해당하는 실측임을 판정 보고에 명시.

## W-B — 캠페인 확장 무대 flirds P1w twin leg (신규)

- 무대: 캠페인 그리드 중 dir1 제외 전부({cifar10 shard·qskew, fmnist iid·dir1(+cifar10 iid)} × 위협
  {clean, fr, frrand, gn, lf@0.70, lf strmain} × 3-seed) — downstream twin이 있는 셀에만 동반.
- arm: **flirds × {T1 P2-게이트, T2 P2-재학습}** (track_c2 기존 P2/`C2_T2` 기계 그대로 — 신규 코드 불필요 예상;
  arm 라벨은 기존 P2 명명 유지, 분석에서 P1w로 리네임). 게이트 하이퍼 = R1과 동일(셀별 튜닝 금지).
- 비-flirds 점수원(W-D)은 **후순위 별도 승인** — flirds leg 결과 보고 후.
- 비용: 그쪽 그리드 기준 산정·보고(개산 ~25–35 GPU-h; CNN 런 5–8분/arm). 산출 스키마 = 기존
  competition CSV 호환 + `policy` 컬럼.

## 보고·판정

- 완료 시: W-A 귀속 판정(드리프트 포함) + W-B 표(P1 vs P1w, 위협×파티션) + FedIF-역전이 확장 무대에서도
  재현되는지 1줄 → 00-INDEX §1 수록 규칙 적용 의견(승/동률/미수록)과 함께 커밋.
- 결과 기입처: overview §3.2.3 이웃(신규 소절) — paper·T2 페이지는 그로부터.
