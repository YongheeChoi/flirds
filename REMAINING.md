# REMAINING — 남은 실험 인수인계 (라우터)

> 2026-07-24: 실행처별로 **2개 파일로 분리**. 세션은 자기 하드웨어 파일만 보면 된다.
> 이 파일 = 라우터(공통·무GPU + 포인터)뿐. push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만.

## 실행처별 파일

| 파일 | 실행처 | 담는 실험 | 현재 상태 |
|---|---|---|---|
| **`REMAINING-b200.md`** | B200 컨테이너 | LLM 주무대 R4(gsm50k5, 1B): L1·L2·L4·L5·L6·L7·L9·L10 + anchor5 β0.3 재실행 | **R4 L1·L2 진행 중** |
| **`REMAINING-slurm.md`** | yonsei Slurm RTX3090 | CNN(c2fid·W-B·C-fr·C1 재실행) + 작은-N LLM(L8 gsm5·silo5 a-leg) | **유휴 — 전량 대기** |

- **ShapleyFL β0.3 재실행**은 두 파일에 걸침(같은 07-23 감사): anchor5 3셀 = B200(`REMAINING-b200.md` §4) ·
  C1 30셀 = Slurm(`REMAINING-slurm.md` §4). 둘 다 착지해야 paper B.5 주석 삭제로 완결.
- **구 §번호 매핑**(외부 문서 참조 완충): 구 REMAINING `§1.6`(LLM 캠페인) → `REMAINING-b200.md` ·
  구 `§1.4b/d`·`§1.6a(C-fr)`·`§1.6b(L8)` → `REMAINING-slurm.md` · 구 `§1.4c(β)` → 양 파일 §4.

## 공통 (무GPU·실행처 무관)

### P0(H1) 소급 재실행 스코프 (장기)

- 논문 인용 셀 한정으로 판단(그룹 카탈로그 = git 히스토리의 `RERUN_AFTER_REPRO_FIX_2026-07-21.md`).

### rundir 정체성 가드 (양 파일 β재실행이 사용)

정체성 allow-list(`check_identity`/`precheck`; 우회 `RUNDIR_REPLACE=1`) + β 단일화
(`shapleyfl.BETA = env SFL_BETA, 기본 0.3`). 배선 완료 = `track_g`·`phase2_matrix`, 테스트 6개.
**잔여 배선**(track_c1/c2/c2_fid/track_d/phase1 = `identity=None`)은 CNN track 몫 → `REMAINING-slurm.md` §6.
