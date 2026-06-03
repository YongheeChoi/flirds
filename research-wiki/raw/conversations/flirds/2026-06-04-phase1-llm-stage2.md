---
type: conversation
date: 2026-06-04
topic: flirds
participants: [Yonghee, Claude]
tags: [flirds, phase-1, llm, validation, fedavg-core, llama, smoke]
---

# 2026-06-04 — Phase 1 LLM stage 2: validation lock + FedAvg core + LLM backend/FL-loop

새 세션에서 Phase 1 이어감. 다른 세션(fork)이 estimator/oracle을 backend-agnostic +
partial-participation + per-layer로 완성·커밋(`15d84b8`)한 뒤라, 이 세션은 (i) 그 작업
검토, (ii) validation(§3.4) 결정, (iii) **LLM stage 2**(backend + FL loop) 구현·검증.

## 파악 + 검토
- Phase 0/0.5 + plan + 다른 세션 wiki 업데이트(12 source 2025–26 scan, experiment plan
  #12–18, Phase 1 seam 3개) 흡수.
- 다른 세션 코드 검토 → **문제 없음**. backend 추상화 깨끗(`loss_fn(params,buffers)`+pkeys
  주입; estimator/oracle가 model/val/task 모름). per-round weight·per-layer 정확, CNN
  회귀 green, 기록/메모리 정리됨. fork는 별도 worktree 아닌 같은 main이었음(작업 종료 확인).

## Validation (§3.4) — Yonghee 결정
- **B1 trainset 크기**: cross-silo는 도메인 효과 isolate가 목적 → **도메인별 데이터 개수
  동일 통제**. 두 결정 분리: aggregate weight는 size-proportional 유지(lock), trainset
  크기만 통제변수. 값은 min 도메인에 맞춤(실데이터 확인); **FiQA까지 부족하면 code-domain
  으로 FiQA 대체** 옵션(Yonghee).
- **B2 대표 샘플링**: dev split이 대표성 보장(제작자 큐레이션) → random/stratified 200으로
  충분, coreset 최적화는 "대표성 선별이 또 다른 valuation"이라 순환 → 안 함. "차후 꼭
  검증"(#16 validation-sensitivity).
- **B3 split 없는 dataset(Dolly)**: category-stratified fixed-seed carve(train에서).
  일반 원칙 = 데이터셋이 주는 구조(라벨/카테고리)로 stratified holdout.
- **B4 (A) 방향 확정**: IRDS-held-out 관점(안정적 평균 val-loss가 utility), few-shot(50)
  기각(loss 추정 noisy). **validation 1000 / coalition-subset 1024 분리**(같은 "1024"가
  두 의미로 섞여 있던 걸 발견 → validation을 1000으로 고정해 혼동 원천 차단).
  plan §3.4·D6·flirds·protocol §8 **4곳 일치**시킴.
- OpenFedLLM eval harness(open_ended LLM-judge + FinGPT)는 val-loss utility로 부적합 →
  val-loss는 자체 구성, task-metric만 Phase 3에서 차용.

## LLM 환경
- transformers 5.9 / peft 0.19 / trl 1.5 / accelerate 1.13 / datasets 4.8 설치(torch
  2.12 유지). OpenFedLLM reqs(torch 2.0 기반)는 비호환 → reference로만, self-build.
- HF token(Yohez 계정) → `~/.cache/huggingface/token`(chmod 600, repo 밖). **채팅 노출
  됐으니 rotate 권장.** Llama-3.2 1B/3B access OK 확인.

## LLM stage 2 (A = 공통 core + backend wrapper; Yonghee 선택)
- **`fl/server.py` core 추출**: `_fedavg_core(init_state, local_train_fn, sample_nums,
  rounds, sample_frac, seed, on_round, eval_fn)` — 라운드 루프·partial-participation·
  per-round participant weight·on_round 로깅. CNN `fedavg`/`run_fedavg_logs`는 thin
  wrapper(시그니처 보존). **회귀 bit-identical**(이전 server.py로 대조: 1st 0.7381 /
  1st+2nd 0.8810 동일, cosine·relL2까지).
- **`backends/llm.py`**: `make_llm_loss(model, val_batch)` → (loss_fn, pkeys). LoRA-only
  주입, frozen base는 캡처 모델에서.
- **`fl/llm_server.py`**: `run_llm_fedavg_logs` — TRL SFTTrainer 1.x(`processing_class`,
  `SFTConfig.max_length`, `completion_only_loss=True`) + **forced SGD**
  (`optimizer_cls_and_kwargs=(SGD,{lr,momentum:0})`, constant lr). delta = named-key
  LoRA(after−global).
- OpenFedLLM 정독: state=`get_peft_model_state_dict`(LoRA), aggregate=참여자
  size-weighted(우리 per-round weight와 동치; delta는 local−global로 추출).

## 발견 3개 (LLM-specific, CNN엔 없던; 본실험/3B/7B 동일 적용)
1. **eager attention 필수** — SDPA/flash 커널이 forward-mode AD(jvp HVP) 미지원 →
   `attn_implementation="eager"`. (`backends/llm.py` docstring에 명시.)
2. **named_parameters key** — `get_peft_model_state_dict` key(`…lora_A.weight`)와
   `functional_call`/named key(`…lora_A.default.weight`) 불일치 → FL loop을 named-key
   + `load_state_dict(strict=False)`로(base 무관).
3. **embedding require-grad hook** — SFTTrainer가 gradient-checkpointing용으로 embedding
   에 건 hook(`make_inputs_require_grads`→`output.requires_grad_()`)이 functorch transform
   과 충돌 → `make_llm_loss`에서 `get_input_embeddings()._forward_hooks.clear()`
   + `use_cache=False`.

## 검증
- backend 스모크(Qwen2.5-0.5B, fake Δw): HVP 작동, est≈oracle 1.67e-4, **2차항>1차항**
  (1.67e-4<1.71e-3), per-layer invariant 2.08e-17(seam1 LLM 재확인).
- **LLM-FL 스모크(Llama-3.2-1B, real SFTTrainer 궤적)**: logs(2r×3c, LoRA-only,
  ‖Δw‖2e-3), **estimator≈oracle 1.70e-6**, finite → LLM FL loop **end-to-end OK**.

## 다음
- **3번 = 5-domain data layer**(validation 1000 stratified + seam 2 corruptor registry).
- 본실험: Llama-3.2-1B/3B + Llama-2-7B(7B는 bf16 train / fp32 eval 분리 필요).
