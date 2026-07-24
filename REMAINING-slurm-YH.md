# REMAINING (Slurm · YH 계정 = chyoyhr) — CNN 주무대 완주 + LLM downstream 보조

> 실행처별 인수인계 **5-서버 분할** 중 **YH**(지금까지 쓰던 계정) 몫. 짝 = `REMAINING-b200.md`(HVP 전용)·`REMAINING-slurm-HJ.md`(silo5-a·L11)·`REMAINING-slurm-JW.md`(L4 renorm T2)·`REMAINING-slurm-JB.md`(L9 arms).
> **역할 = (1) CNN 주무대(c2fid·W-B·C-fr·C1) 완주** — 이미 실행 중 · **(2) CNN free(~07-25) 후 A6000 48GB에서 LLM downstream 보조**(HJ·JW 큐 work-steal). push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만.
> **마감: 실험 07-28 / 논문 07-29 21:00** — seed0 우선. 논문·문서 정본 = `paper/workplan/00-INDEX.md`. 기존 rundir은 read-only.

## 실행 현황 (2026-07-24 저녁 · 커밋 80ebf30/47680ec 시점)

| § | 실험 | 상태 | 진척 |
|---|---|---|---|
| §1 | c2fid CNN fidelity | 🟡 seed0·1 완료·seed2 진행 | **110/144**(s0 48✓·s1 46·s2 16) 실행 중·8슬롯 점유 |
| §2 | W-B P1w observer(T2) | 🟡 파일럿 29셀 완주·게이트 대기 | `p1wgate`가 GPU 확보 시 판정 → 통과면 30-89 자동 |
| §3 | C-fr frrand full-method | 🟡 파일럿 8/8 EXIT=0·커밋(`80ebf30`) | `frgate` PD(GPU 확보 시 seeds1-2 판정) |
| §3.5 | **fmnist 완결(신규 07-25)** | 🟢 CNN free 후 즉시 | Part A c2fid s2 16셀 + Part B competition 288 rundir — **YH lora4cl 전용**(스택 정합) |
| §4 | C1 β0.3 재실행(30셀) | ✅ 완주·커밋 `47680ec` | ShapleyFL만 β0.5→0.3 변화·타 10방법 비트동일 |
| §5 | LLM downstream 보조 | ⚪ CNN·§3.5 free 후 | HJ/JW overflow work-steal(A6000) |

- **한도**: 동시 8-GPU(QOSMaxGRESPerUser) + §1 array `%8`. 현재 §1 c2fid가 8슬롯 점유 → 게이트 2종(p1w·fr) PD 대기(GPU 확보 시 발화).
- **파일럿→GO 게이트**(§2·§3)는 Slurm `--dependency`로 자동(세션 무관). 완주 시 GPU-h 실측 보고 후 잔여 leg 자동 제출.
- **예상 종료**: §1~§3(CNN) ~07-25 → 이후 8슬롯이 **§5 LLM 보조**로 전환(A6000 파티션).
- **silo5-a는 여기 아님**: 24GB OOM으로 취소 → **HJ 계정 A6000**로 이관(`REMAINING-slurm-HJ.md` §1). YH의 3090/A6000 QOS 8슬롯이 CNN에 물려 동시 실행 불가라 = 다른 계정으로 뺀 이유.

## 0. 환경

- **CNN**: conda `lora4cl`(`/home/chyoyhr/anaconda3/envs/lora4cl/bin/python`, torch 2.11.0), partition `base_suma_rtx3090`, 8-GPU QOS. 리포 루트 = `/home/chyoyhr/projects/flirds/`.
- **더 큰 GPU**(2026-07-24 조사): `base_qos`로 **A6000 48GB**(`suma_a6000`·`gigabyte_a6000` ~96장)·**RTX6000Ada 48GB**(`asus_6000ada`)·**RTX4090 24GB**(`suma_rtx4090`) 즉시 접근(모든 파티션 AllowAccounts=ALL — QOS가 게이트). A100/PRO6000만 특수 QOS. **8-GPU 상한 per-user** → 계정 3개(YH·HJ·JW) = 동시 24.
- 공통: `HF_HUB_OFFLINE=1`, `codes/`에서 `PYTHONPATH=.`.

## 1–4. CNN 주무대 (실행 중 — 상세 런북은 각 정본 참조)

> 진행 중이라 **개입 불필요**(게이트 자동). 완료 시 분석·overview 기입만. 상세 = 각 런북.

- **§1 c2fid**(fidelity 주무대, 143셀·**seed-major**): `sbatch runs/track_c/c2fid/sbatch_fid.sh` → `runs/track_c/c2fid/make_analysis.py` → F-1~F-4 사전등록 대조. 실측 **1.05 GPU-h/셀**·본런 ≈150 GPU-h. **채우는 ⬚**: §5.2 메인 c2fid fidelity·§5.4 φ-AUROC·§5.6 F-4·부록 C·D·figF3/F6. 정본 = `runs/track_c/c2fid/README.md`.
- **§2 W-B**(P1w twin leg): 정본 = `runs/track_h/RUN_P1W_CNN.md`. W-A 판정 완료(드리프트≈0 귀속·FedIF 역전). W-B 신규 = **T2 leg만**(`sbatch runs/track_h/sbatch_cnn_p1w.sh` 파일럿 0-29 → 게이트 → 30-89). 전체 ~30–32 GPU-h. **채우는 ⬚**: §5.3 CNN P1w 규칙부.
- **§3 C-fr**(frrand full-method, dir1×3seed): 러너 `experiments/track_c2.py` `C2_THREAT=frrand C2_MODE=full`+`C2_T2` · 템플릿 `runs/track_h/sbatch_strmain.sh`(label_flip→frrand·strmain 특유부 제거) · 7소스+obs T2 = 24런(strmain 동형). 파일럿 8/8 완주(`80ebf30`)·`frgate` PD. **채우는 ⬚**: §5.3 CNN frrand 열·§5.4 frrand AUROC.
- **§4 C1**(β0.3 재실행 30셀): ✅ 완주·커밋 `47680ec`. **⚠ 실측**: `RUNDIR_REPLACE`가 canonical(하이픈·무해시)을 못 덮고 `*_<hash>` 생성 → 수동 canonical 승격 후 커밋(§6 잔여 배선과 연관). **갱신**: overview §3.1.2 C1 ShapleyFL 행·paper §5.2 sub·부록 C.

## 3.5. fmnist 완결 (신규 2026-07-25 Yonghee "fmnist 비어있는 부분 전부") — YH lora4cl 전용

> **왜 YH 전용**: fmnist 아티팩트 전량(`_g` grid 48셀·c2fid seed0/1) + cifar10/dir1 competition = **전부 torch 2.11**(freeze.txt 실측 2026-07-25). 신규 fmnist도 2.11에서 돌려야 ①c2fid 3-seed 세트가 seed별로 스택 안 갈리고 ②competition recovery 분모(`_g` vanilla/oracle_excl, 2.11)와 소스 arm이 셀 내부서 동일 스택. **2.11 = YH lora4cl 뿐** → A6000(2.12) 이관 시 오히려 스택 쪼갬. 둘 다 CNN·B200 독립. **CNN 주무대(§1–3) free 후 즉시**, §5 LLM 보조보다 **먼저**(CNN 네이티브 환경 그대로).

**Part A — c2fid fmnist seed2 (16셀, §5.2 fidelity = §5.4 탐지):** seed0·1 완비, seed2만 결측.
- 본 144-array의 인덱스 **128–143**과 동일 셀 = 별도 파일 `runs/track_c/c2fid/sbatch_fid_fmnist_s2.sh`(array 0-15).
- **이중실행 주의**: 본 `sbatch_fid.sh`(0-143)가 아직 살아있으면 → 본 array를 `--array=0-127%8`로 캡(cifar10 seed2만) 후 이 파일 제출, **또는** 이 파일 건너뛰고 본 array가 128–143 자체 완주. 같은 rundir명 last-writer-wins라 레이스는 GPU 낭비만.
  ```
  mkdir -p runs/track_c/c2fid/_logs
  sbatch runs/track_c/c2fid/sbatch_fid_fmnist_s2.sh          # 16셀, 3090/lora4cl
  ```
- 비용 ≈ **~11–16 GPU-h**(fmnist는 cifar10보다 쌈). 완료 → `make_analysis.py`가 fmnist seed2 흡수.

**Part B — §5.3 competition fmnist (288 rundir, 신규):** fmnist에 obsf(W-B)만 있고 8점수원 경쟁은 전무.
- `{fmnist iid,dir1} × 6위협(clean·fr·frrand·gn·lf@0.7·strmain) × (7 비-flirds + obs) × 3seed`.
- flirds arm·recovery 분모는 track_g fmnist `_g` grid(3-seed 전량 on disk) 재사용 → 신규는 비-flirds 소스만 = 자체 스코어·B200 독립.
  ```
  mkdir -p runs/track_h/_logs
  sbatch --array=0-95%8   runs/track_h/sbatch_cnn_fmnist_comp.sh   # seed0 파일럿(96)
  # GPU-h 실측 보고 → GO →
  sbatch --array=96-287%8 runs/track_h/sbatch_cnn_fmnist_comp.sh   # seeds 1-2
  ```
- 비용 ≈ **~15–40 GPU-h**(비-flirds 소스만; MC 소스[gtg·fedsv·comfedsv·shapleyfl]가 주비용이나 fmnist 저해상이라 쌈). 사전등록 = README H-16.
- 완료 → `make_analysis.py`(fmnist 행 cnn_competition.csv 편입) → overview §3.2.3 이웃 소절(fmnist 열) → paper §5.3. **판정 = 성능만**(§3).

**합계 ~30–55 GPU-h**(YH 8슬롯 3090 → wall ~반나절). B200 HVP ~2일 임계경로 하위라 마감 여유. 완료 후 §5 L11 seed2로 전환.

## 5. LLM downstream — L11 seed2 담당 + work-steal 보조 (CNN·§3.5 free 후 · A6000 48GB)

> CNN(§1-3) + **fmnist(§3.5)** 완주 후 YH 8슬롯이 비면: **(a) 담당 = L11 seed2**(HJ가 seed0·1을 맡고 부하분할한 나머지 = `REMAINING-slurm-HJ.md` §2) **+ (b) 보조 = HJ(L11 tail)·JW(L4)·JB(L9-arms)의 밀리는 잔여 흡수.** 전부 자체완결·B200 독립·arm-level idempotent → 계정 넘어 안전. **순서 = fmnist(§3.5) 먼저**(CNN 네이티브 2.11 환경 그대로) → 그다음 L11 seed2(A6000 48GB LLM). fmnist가 밀리면 L11 seed2를 JB/HJ가 work-steal.

- **(a) 담당 = L11 seed2**(7 비-flirds × {clean,noisy,frzero} × seed2 = **21런 ~92 GPU-h**): HJ와 동일 arm세트에 `SEED=2`. seed2 = 최저 우선(error-bar seed)이라 YH의 늦은 가용(~07-25)과 정합. 착지 root `rundirs_llm_yh`.
- **(b) 보조(우선순위)**: 남으면 HJ/JW/JB 밀리는 큐의 seeds 1-2 흡수. 큰 물량 순 = L11 tail(HJ) → L4(JW) → L9-arms(JB). flirds 가중 T2(L7·L1)는 B200 몫이라 여기 안 옴.
- **명령**: 해당 실험의 track_g 명령 그대로, **착지 root만 YH 전용**:
  ```
  RUNDIR_ROOT=$REPO/runs/track_h/rundirs_llm_yh \
  REGIME=gsm50k5 THREAT=<...> SEED=<1|2> ARMS=<...> T2=<0|1> T2_LEGACY=0 T2_P5=0 \
    PYTHONPATH=. $PY -u experiments/track_g.py
  ```
  (arm 세트 = 흡수하는 HJ/JW 항목과 동일. A6000 파티션.)
- **셋업**: YH는 repo·conda·HF캐시 이미 완비(§0) → A6000 파티션 지정만 추가. HJ/JW 대비 **셋업 마찰 0**.
- **분석**: `make_analysis.py`에 `rundirs_llm_yh` root 추가(dup-win) → 병합. **스택 캐비엇 = recovery 정규화**(W-A ≤0.006). **timing cost 금지**(B200 실측만).

## 6. rundir 정체성 — 잔여 배선 (CNN track)

처방 1+2 완료(07-23): 정체성 allow-list(`check_identity`/`precheck`; 우회 `RUNDIR_REPLACE=1`) + β 단일화(`shapleyfl.BETA=env SFL_BETA, 기본 0.3`). 배선 = `track_g`·`phase2_matrix`, 테스트 6개.
- **잔여**: ① `track_c1`·`track_c2`·`track_c2_fid`·`track_d`·`phase1_*`는 아직 `identity=None`(§4 C1 재실행이 `*_<hash>` 새 디렉토리 낸 원인 — 다음 재실행 전 네이밍 정합 필요). ② 처방 3(`superseded.json`) 미착수.
