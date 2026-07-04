# 논문화 전 검증 — 미해소 실측 인수인계 (2026-07-04)

이 세션 환경은 **GPU를 못 돌린다**(로컬 Windows). 아래 실측 3건은 스크립트·커맨드가 전부
준비돼 있으니 **서버에서 GPU가 비는 시점에** 순서 상관없이 돌리면 된다. 각 항목에 (1) 무엇을
왜, (2) 실행 커맨드, (3) 산출물 회수, (4) 결과를 어느 문서 어디에 채우는지 를 적어둔다.

공통:
- 원격 접속: `ssh "[tmp]korea_bupj"` (로컬 Git Bash에서는 `/c/Windows/System32/OpenSSH/ssh.exe` 명시).
- 원격 저장소 `/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds`, python
  `/home/korea_bupj/miniconda3/envs/flirds/bin/python`, 실행은 `codes/`에서 `PYTHONPATH=.`.
- **GPU 규약**: 물리 0-3만. 실행 전 `nvidia-smi`로 빈 GPU 확인 — probe std50k5(GPU0-2)가
  아직이면 그 GPU는 피할 것. GPU4-7은 프로젝트 규약상 미사용. B200 실 OOM ~160GB.
- 작업 파일은 신규 `flirds_verify_scratch/`(이미 존재; 원본 저장소 무수정). 스크립트 3개는 이미
  원격에 업로드돼 있음(taylor/, ripple/). 재업로드 필요 시 로컬 survey 폴더에서 scp.
- 이 실측들은 **결과에 영향 주는 추가 근거**일 뿐 기존 결론을 뒤집는 게 아니다 — 안 돌려도
  문서는 완결돼 있고(placeholder 명시), 돌리면 placeholder가 실측치로 채워진다.

---

## 실측 1 — Taylor 잔차 1B 본실행 (항목 1; 우선순위 최상)

**무엇·왜**: gpt2 CPU 스모크는 P1·P2의 *대수적 정확성*만 1e-12로 확인했다(closed-form φ =
`flirds_values`, per-round = 2^N oracle). gpt2는 이동량 ‖ΔW‖≈6e-4가 fp32 노이즈 바닥이라
**P3의 물리적 잔차 크기·2차>1차 우위·O(‖Δ‖³) 스케일링·P5 순위**는 못 봤다. realistic
‖ΔW‖(1B, max_steps=10)에서만 검증 가능.

**커맨드** (상세·예상비용은 `irds-fl-math-rigor-2026-07/RUN_1B.md`):
```bash
ssh "[tmp]korea_bupj" 'cd /NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/codes && \
  CUDA_VISIBLE_DEVICES=<X> OMP_NUM_THREADS=16 PYTHONPATH=. nohup nice -n 10 \
  /home/korea_bupj/miniconda3/envs/flirds/bin/python -u \
  /NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds_verify_scratch/taylor/measure_taylor_residual.py \
  --model meta-llama/Llama-3.2-1B-Instruct --device cuda --rounds 10 --val_size 100 \
  --seed 0 --renorm --check_inrun \
  --outdir /NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds_verify_scratch/taylor/llama1b_r10_seed0 \
  > /NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds_verify_scratch/taylor/llama1b_r10_seed0.log 2>&1 &'
```
- GPU 1장, **예상 ~55–75분**, peak ~20–30GB. `--renorm`(P5) `--check_inrun`(P1) 유지 권장.
- 선택: seed 1·2 추가로 3-seed(잔차가 seed-robust한지) — 각 동일 비용.

**산출물 회수**: `llama1b_r10_seed0/{coalitions.csv, coalitions.parquet, phi.csv, summary.json}`
→ 로컬 `research-wiki/survey/irds-fl-math-rigor-2026-07/`로 회수(gpt2 스모크 산출물과 동일 위치).

**결과 반영**: `irds-fl-math-rigor-2026-07/irds-fl-math-rigor.md` §7.2 "[실측 대기]" placeholder를
표로 교체 — 라운드별 [실제 u / û¹ / û² / 잔차 / ‖Δ_S‖], resid median의 loglog slope(기대 2/3에
근접하는지), frac(û²≤û¹)(2차 우위), P5 renorm 순위 vs 고정가중 순위. **판정 포인트**: gpt2에서
slope 0.30(노이즈)이었는데 1B에서 ~2/3로 올라오면 P3 상계가 타이트함을 실증. §9 결정 3(승인) 해소.

---

## 실측 2 — 2026-06-06 표 조건 통일 회계 스모크 (항목 6·2·3 공유; 우선순위 상)

**무엇·왜**: 세 항목의 placeholder를 한 run으로 채운다 — (6) 공유 로그 생성(=FL 학습) 시간을
명시 측정해 "학습 대비 valuation overhead %" 완성 + Ripple valuation-only 환산, (2) Ripple 분리
계측을 실 무대(1B)에서, (3) fp32-vs-bf16/tf32 forward·HVP 마이크로벤치로 **×3.1 배수 확정**(현재
"미검증 placeholder") + peak-mem.

**셋업**: 1B N=5 R=10 val=100 (2026-06-06 baseline과 동일). 이건 **단일 준비된 스크립트가 없다** —
아래 둘 중 택1:
- (a) 간단: 기존 러너(`codes/experiments/phase2_matrix.py`의 silo5 셀 또는 `phase1_baseline_compare.py`)를
  돌리되 FL 학습 구간을 `time`으로 감싸 로그 생성 시간을 별도 기록 + Ripple 포함. 산출 metrics.json의
  방법별 runtime과 합산.
- (b) 정밀: `cost-comparison-methodology.md` 후속제안 1·2 + `precision-audit-and-policy.md` 후속제안
  P3(fp32-vs-bf16 마이크로벤치)를 한 스크립트로. Ripple 분리 계측은 `ripple-audit-2026-07/`의
  `instrumented_ripple_cnn.py` 패턴을 LLM(ripple_llm.py)에 이식(GPU 필요).
  → **이 스크립트는 아직 안 만들었다**(GPU 없어 검증 불가라 보류). 서버 세션에서 작성 권장.

**커맨드**: 스크립트 확정 후. 1셀 **~30분급**(FL ~15–25분 + valuation). GPU 1장.

**결과 반영** (3곳):
- `cost-comparison-methodology.md` §1.4 "~15min 어림값"→실측 교체, overhead % 열 완성; §5.2
  valuation-only 표에 Ripple 환산치.
- `ripple-audit-2026-07/ripple-audit.md` §4 LLM 분리 계측(현재 CNN-CPU 실측만 있음; §3.3에서 LLM은
  "스케일업 추정"이라 명시한 부분을 실측으로).
- `precision-audit-and-policy.md` §3.1-(4)·후속 P3: ×3.1 배수 확정치로 C3 캡션 갱신.

---

## 실측 3 — CNN TF32 노출 A/B + yonsei 박스 판별 (항목 3; 우선순위 중)

**무엇·왜**: cuDNN conv TF32가 기본 on이라(B200 확인) **CNN 트랙이 진짜 fp32가 아니었을 수
있다**. CNN은 yonsei SLURM 박스에서 도는데 그 박스 기본값은 미확인. 결과 영향 판정에 필요.

**3-a. yonsei 박스 런타임 판별** (~1분, GPU 거의 불필요):
`precision-audit-and-policy.md` 부록 B의 스니펫(`torch.backends.cudnn.allow_tf32` 등 출력)을
yonsei 박스에서 1회 실행. → cuDNN conv가 실제 TF32인지 확정(전제).

**3-b. CNN TF32 A/B 스모크** (~1–2.5h, 3090/소형 GPU면 충분):
cifar10 iid + label_flip seed0을 `allow_tf32=True`(현행) vs `False`로 각 1회 → coalition diff·
φ·fidelity(Spearman)가 TF32 오차(~1e-3)에 흔들리는지. **headline 불변 확인이 목적**(흔들리면
`cudnn.allow_tf32=False` 1줄 채택 근거, 안 흔들리면 논문 각주로 충분). 스크립트는 기존 CNN
러너(`track_c*.py` 또는 phase0 CNN 스모크)에 `torch.backends.cudnn.allow_tf32=<bool>` 토글만
추가한 사본. → **미작성**(CNN 무대라 yonsei 세션에서. `runs/probe_signal/PROMPT_yonsei_cnn.md`
제출과 묶어도 됨).

**결과 반영**: `precision-audit-and-policy.md` §2.2·§2.4 판정을 실측으로 확정, 부록 B에 yonsei
런타임 값. 옵션 비교(§3)의 CNN 각주 근거.

---

## 실행 후 공통 마무리

1. 산출물을 로컬 각 survey 폴더로 회수(원본 저장소 flirds_verify_scratch/는 그대로 두거나 정리).
2. 해당 문서의 "[실측 대기]" placeholder를 실측 표로 교체 + 각 문서 §Yonghee 결정의 관련 항목 해소 표시.
3. `2026-07-verification-overview.md`의 "미해소 실측" 절에서 완료분 제거.
4. 커밋(코드·rundir 불변, survey 문서·산출물만).

**우선순위 요약**: 1(Taylor 1B, 준비완료·바로실행) > 2(통일 회계, 스크립트 작성 필요) >
3(CNN TF32, yonsei 박스·스크립트 작성 필요). 1은 지금 커맨드만 붙여넣으면 되고, 2·3은
서버 세션에서 스크립트를 마저 만들어야 한다(GPU 없이는 검증이 안 돼 이 세션에서 보류함).
