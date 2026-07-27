# Track C2-FID 분석 (자동 생성 — `make_analysis.py`; 수치 canon = rundir)

셀 168 · 행 1512. 지표 정의·사전등록 원문 = `../README.md`.
게임 캐비엇: 10/100 참여 부분게임 — C1(N=10 전원참여) 표와 직접 비교 금지.

## Spearman vs (b) — (dataset, partition) 평균

```
dataset   cifar10                      fmnist         mnist       
partition    dir1    iid  qskew  shard   dir1    iid   dir1    iid
method                                                            
ComFedSV    0.177  0.164  0.130  0.202  0.273  0.232  0.188  0.203
FedIF       0.653  0.807  0.598  0.426  0.699  0.744  0.597  0.637
FedSV       0.502  0.534  0.524  0.472  0.628  0.633  0.617  0.718
Flirds      0.974  0.982  0.963  0.932  0.991  0.993  0.976  0.969
Flirds1st   0.606  0.841  0.622  0.554  0.765  0.889  0.744  0.815
GTG         0.509  0.698  0.635  0.513  0.698  0.834  0.746  0.844
ShapleyFL   0.447  0.430  0.351  0.439  0.471  0.339  0.320  0.238
loss-heur   0.923  0.973  0.941  0.866  0.960  0.977  0.970  0.973
```

## 사전등록 대조 (F-1~F-4)

- **F-1 MISS** — Delta_rho(qskew-iid) family means uniform=-0.057 nprop=-0.037 taylor=-0.149 direct=-0.032 | per-method Flirds=-0.019 Flirds1st=-0.219 GTG=-0.063 FedSV=-0.010 ComFedSV=-0.034 ShapleyFL=-0.079 FedIF=-0.209 loss-heur=-0.032
- **F-2 MISS** — fr max|phi| exact0-family=Flirds:1.12e+00,Flirds1st:1.64e+00,loss-heur:1.24e+00,FedIF:9.43e-01,(b)oracle:1.07e+00 renorm=GTG:7.46e-01,FedSV:9.12e+00 | frrand AUROC flirds=0.881 renorm=0.157
- **F-3 HIT** — mean rho(b) flirds=+0.848 >= nprop=+0.617 >= uniform=+0.295 | Flirds=+0.973 Flirds1st=+0.722 GTG=+0.669 FedSV=+0.566 ComFedSV=+0.196 ShapleyFL=+0.394 FedIF=+0.649 loss-heur=+0.944
- **F-4 MISS** — spearman_vs_rate: (b)=+0.678 Flirds=+0.717 Flirds1st=+0.779 | spearman_vs_rate_corrupt: (b)=+0.577 Flirds=+0.578 Flirds1st=+0.590
