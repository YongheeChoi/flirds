# overview-figures-2026-07 — overview 임베드용 figure 사본

`flirds-experiment-results-overview-2026-06-25.md`에 임베드된 그림들.
**원본(정본) = `runs/<exp>/figures/*.png`** — 각 실험 폴더의 `make_figures.py`가 rundir만으로
재생성(내용·출처·커버리지·스팟체크 = 각 `runs/<exp>/figures/MANIFEST.md`). Obsidian vault가
`research-wiki/`라 vault 밖 `runs/` 경로는 렌더링이 안 돼 여기 사본을 둔다
(선례: `removal-dose-2026-07/`).

파일명 = `<실험접두사>_<원본파일명>.png`. 접두사: `trackd`(track_d) · `trackc`(track_c) ·
`phase2`(phase2_matrix) · `probe`(probe_signal) · `cxni`(matrix_cxni) · `measured`(measured_2026-07;
04 acct만 — taylor/tf32/microbench는 이 overview가 아니라 precision·math-rigor survey 문서 소관) ·
`beta`(rerun_beta03).

재동기화 (figure 재생성 후):

```bash
cd <repo>
D=research-wiki/survey/overview-figures-2026-07
for f in runs/track_d/figures/0*.png;       do cp "$f" "$D/trackd_$(basename $f)"; done
for f in runs/track_c/figures/0*.png;       do cp "$f" "$D/trackc_$(basename $f)"; done
for f in runs/phase2_matrix/figures/0*.png; do cp "$f" "$D/phase2_$(basename $f)"; done
for f in runs/probe_signal/figures/0*.png;  do cp "$f" "$D/probe_$(basename $f)"; done
for f in runs/matrix_cxni/figures/0*.png;   do cp "$f" "$D/cxni_$(basename $f)"; done
cp runs/measured_2026-07/figures/04_cost_accounting.png "$D/measured_04_cost_accounting.png"
for f in runs/rerun_beta03/figures/0*.png;  do cp "$f" "$D/beta_$(basename $f)"; done
```
