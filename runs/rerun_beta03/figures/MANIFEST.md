# rerun_beta03/figures — ShapleyFL β0.5→0.3 대조·provenance figure

Generator: `runs/rerun_beta03/make_figures.py`. 이 폴더는 rundir이 없는 캠페인 드라이버 —
figure 입력은 ①git 히스토리의 β0.5 원본 rundir(`b1b95d0~1`) vs 현재 β0.3 rundir,
②전 그룹 rundir `meta.json` git_sha의 β커밋(`e89af94`, 2026-06-25) ancestry 분류.

**Coverage** (2026-07-16): before/after 쌍 6/6 (3B std20·anchor5 ×3seed; before=전부
β0.5-era sha, after=전부 β0.3-era sha 검증됨). provenance 스캔 294 rundir.

**스캔이 확정한 사실 (문서 주장과의 차이 포함 — 실측 우선)**:
- β0.5-era 잔존 154 rundir: **phase2 June 25**(RESUME 문서와 일치) + **track_d 9**
  (1B×6 + 7B_std20×3) + **track_c c1 30 + c2 120전부**.
- **7B_anchor5×3는 β0.3-era 코드**(git `f677427`, 06-26 > β커밋 06-25) — RESUME 문서의
  "7B×6 대기" 중 3셀은 이미 β0.3 코드 산출물(단, 전 rundir git_dirty=True → sha는 era를
  bound할 뿐 확정 아님; 어차피 7B는 stale-June 사유로 이월 배치가 전면 재실행 중).
- **track_c CNN 120셀 = β0.5-era**(c1 06-12 등) — RESUME 문서의 "CNN 완료(β0.3)" 주장과
  상충, 루트 CLAUDE.md의 "CNN은 e89af94 이후 재실행 커밋 없음·canon 미확보" 각주가 맞음.
  **β는 ShapleyFL method/arm 행에만 영향** — 다른 method 수치는 무관.
- phase2 July(rundirs_2026-07) 22 + B축 6 + probe_signal 전부 = β0.3-era ✓.

| figure | 내용 (1줄) | 데이터 출처 |
|---|---|---|
| `01_beta_contrast_3b_before_after.png` | 재실행된 3B 6셀의 ShapleyFL fidelity 전후 + method별 φ 전후 순위일치(비-ShapleyFL=재실행 노이즈 플로어; ShapleyFL도 같은 플로어 = β효과 미미, anchor5_seed2만 +0.1→+0.3) | git `b1b95d0~1` vs 현재 rundir |
| `02_beta_provenance_map.png` | 전 rundir β-era 지도(그룹별 적층 바; dirty 수 병기) | `*/meta.json` git_sha ancestry |
| `beta_contrast_3b.csv` / `beta_provenance.csv` | figure 정확 입력(294 rundir 분류 전체) | — |

재생성: `python runs/rerun_beta03/make_figures.py` (git 히스토리 필요; stdout에 쌍 검증+잔존 목록).
스팟체크(2026-07-16): 3B_anchor5_seed2 ShapleyFL Spearman before 0.10/after 0.30 =
metrics.json(git·현재) 손확인 일치; φ 전후 순위일치 1.0 = CSV 일치.
