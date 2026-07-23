# Track C2-FID 분석 (자동 생성 — `make_analysis.py`; 수치 canon = rundir)

셀 1 · 행 9. 지표 정의·사전등록 원문 = `../README.md`.
게임 캐비엇: 10/100 참여 부분게임 — C1(N=10 전원참여) 표와 직접 비교 금지.

## Spearman vs (b) — (dataset, partition) 평균

```
dataset   cifar10
partition    dir1
method           
ComFedSV    0.273
FedIF       0.774
FedSV       0.870
Flirds      0.846
Flirds1st   0.269
GTG         0.861
ShapleyFL   0.784
loss-heur   0.989
```

## 사전등록 대조 (F-1~F-4)

- **F-1 N/A** — Delta_rho(qskew-iid) family means uniform=+nan nprop=+nan taylor=+nan direct=+nan | per-method Flirds=+nan Flirds1st=+nan GTG=+nan FedSV=+nan ComFedSV=+nan ShapleyFL=+nan FedIF=+nan loss-heur=+nan
- **F-2 N/A** — fr max|phi| exact0-family=Flirds:nan,Flirds1st:nan,loss-heur:nan,FedIF:nan,(b)oracle:nan renorm=GTG:nan,FedSV:nan | frrand AUROC flirds=nan renorm=nan
- **F-3 MISS** — mean rho(b) flirds=+0.558 >= nprop=+0.866 >= uniform=+0.528 | Flirds=+0.846 Flirds1st=+0.269 GTG=+0.861 FedSV=+0.870 ComFedSV=+0.273 ShapleyFL=+0.784 FedIF=+0.774 loss-heur=+0.989
- **F-4 MISS** — spearman_vs_rate: (b)=+nan Flirds=+nan Flirds1st=+nan | spearman_vs_rate_corrupt: (b)=+nan Flirds=+nan Flirds1st=+nan
