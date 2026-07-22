#!/usr/bin/env python
"""gnoise dose diagnosis (H1/H2/H3) -- forward-only perturbation-response curves.

Context (07-22): gn_full (GN_ABS, gamma*=5) failed its band checkpoint --
vanilla EM 0.3753 > oracle_excl 0.3735, i.e. the corruption is harmless.
Three non-exclusive hypotheses (Yonghee):
  H1 dose-control failure -- _add_gnoise sets sigma = gamma * RMS(per-round LoRA
     delta over A and B), but the model feels dW = (alpha/r) B A.  The effective
     perturbation is s[(B+eps)(A+eta) - BA] = s[B eta + eps A + eps eta], so
     ||A||,||B|| rescale the noise multiplicatively -> the dW-space dose was never
     specified.  peft default inits B=0, so GN_ABS freezing sigma at the FIRST
     corrupt round pins it where the dW contribution is smallest.
  H2 tr(H) too small -- converged pretrained+LoRA region is flat; 1/2 sigma^2 tr(H)
     never materializes.  Not fixable by gamma (regime problem).
  H3 stage ceiling -- val_loss across all arms sits in 0.6022~0.6025.

Method: take a trained LoRA state, then compare val-loss response to three
perturbation families at MATCHED per-module ||Xi||_F in dW space:
  (a) current scheme  -- independent Gaussian on A and B (what _add_gnoise does)
  (b) dW-isotropic    -- Gaussian directly on the effective weight
  (c) gradient dir    -- reference upper bound
(a) is applied as a LoRA-state perturbation (loss_fn takes params functionally);
(b)/(c) are applied to base_layer.weight and reverted, which functional_call picks
up because only LoRA params are passed explicitly.

Env: REGIME=gsm50k5 (default), R_TRAIN (default 100), SNAPS (csv round indices),
     MEM_FRAC (default 0.25 -- HARD cap so a co-resident production run can never
     be starved), OUT (json path).
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds/codes")
os.environ.setdefault("REGIME", "gsm50k5")
os.environ.setdefault("THREAT", "clean")

MEM_FRAC = float(os.environ.get("MEM_FRAC", "0.25"))
if torch.cuda.is_available():                    # cap BEFORE any allocation
    torch.cuda.set_per_process_memory_fraction(MEM_FRAC, 0)

import experiments.track_g as tg                                       # noqa: E402
from flirds.backends.llm import make_llm_loss                          # noqa: E402
from flirds.data.llm import build_val_batches                          # noqa: E402
from flirds.oracle.exact_sv_llm import _final_lora_state               # noqa: E402
from flirds.baselines.ripple import _flat                              # noqa: E402

DEV = "cuda"
R_TRAIN = int(os.environ.get("R_TRAIN", "100"))
SNAPS = [int(x) for x in os.environ.get("SNAPS", "25,50,100").split(",")]
OUT = os.environ.get("OUT", "gnoise_dose_diag.json")
CACHE = os.environ.get("CACHE", "")          # snapshot cache: skip retraining on rerun
GAMMAS = [float(x) for x in os.environ.get("GAMMAS", "1,5,20,50,100,200").split(",")]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def collect_lora(model):
    """module_path -> {A: pname, B: pname, mod: module}.  mod.base_layer.weight is
    the frozen W0; the effective update is scaling * B @ A."""
    mods = dict(model.named_modules())
    out = {}
    for pname, _ in model.named_parameters():
        for tag in ("A", "B"):
            key = f".lora_{tag}."
            if key in pname and pname.endswith(".weight"):
                out.setdefault(pname.split(key)[0], {})[tag] = pname
    for base in list(out):
        m = mods[base]
        if not hasattr(m, "base_layer") or "A" not in out[base] or "B" not in out[base]:
            out.pop(base)
            continue
        out[base]["mod"] = m
    return out


def scaling_of(mod):
    s = getattr(mod, "scaling", None)
    if isinstance(s, dict):
        s = s.get("default", next(iter(s.values())))
    return float(s if s is not None else 1.0)


def dw_of(state, info, base):
    """effective dW = scaling * B @ A for one module, from a LoRA state dict."""
    A = state[info[base]["A"]]
    B = state[info[base]["B"]]
    return scaling_of(info[base]["mod"]) * (B @ A)


def main():
    t0 = time.time()
    log(f"regime={tg.REGIME} model={tg.MODEL} R_TRAIN={R_TRAIN} snaps={SNAPS} "
        f"mem_frac={MEM_FRAC} gammas={GAMMAS}")
    tok, model, init, pkeys = tg._load(DEV)
    clients, val, test = tg.build_data(0, set())            # clean stage
    val_chunks = build_val_batches(val, tok, tg.MCFG["val_maxlen"], DEV,
                                   tg.MCFG["val_chunk"])
    loss_fn, _pk, _lc = make_llm_loss(model, val_chunks, DEV)
    info = collect_lora(model)
    log(f"lora modules={len(info)} scaling={scaling_of(next(iter(info.values()))['mod'])} "
        f"val_chunks={len(val_chunks)}")

    # ---- phase 1: train (clean FedAvg, no scoring) -------------------------- #
    # CACHE holds only the per-snapshot LoRA states + delta RMS, so a rerun of the
    # (cheap) perturbation phase never repeats the (expensive) training phase.
    cached = CACHE and os.path.exists(CACHE)
    if cached:
        blob = torch.load(CACHE, map_location="cpu", weights_only=False)
        snap_states = {int(k): v for k, v in blob["snaps"].items()}
        rms_by_round = blob["rms"]
        log(f"loaded cached snapshots {sorted(snap_states)} from {CACHE}")
    else:
        log(f"training {R_TRAIN} rounds ...")
        logs = tg._fl(model, tok, clients, init, seed=0, rounds=R_TRAIN)
        log(f"train done {time.time()-t0:.0f}s  "
            f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB")

        # per-round delta RMS = what _add_gnoise scales sigma by (the SPEC dose knob)
        keys = sorted(next(iter(logs[0][1].values()))[0].keys())
        rms_by_round = []
        for _w, dm in logs:
            for c in dm:
                v = _flat(dm[c][0], keys)
                rms_by_round.append(float(v.pow(2).mean().sqrt()))
                break                                        # one client per round is enough
        snap_states = {s: _final_lora_state(logs[:s] if s < len(logs) else logs)
                       for s in SNAPS if s <= len(logs)}
        if CACHE:
            torch.save({"snaps": snap_states, "rms": rms_by_round}, CACHE)
            log(f"cached snapshots {sorted(snap_states)} -> {CACHE}")
        del logs

    # Post-training hygiene (track_g._guard + make_llm_loss do the same): SFTTrainer
    # leaves train mode, an input-require-grad embedding hook, and gradient
    # checkpointing on -- the last one breaks the (c) backward through
    # functional_call (recompute sees different shapes than the saved forward).
    model.eval()
    for meth in ("gradient_checkpointing_disable", "disable_input_require_grads"):
        try:
            getattr(model, meth)()
        except Exception:
            pass
    model.config.use_cache = False
    emb = model.get_input_embeddings()
    if emb is not None:
        emb._forward_hooks.clear()

    sigma_first = rms_by_round[0]                            # GN_ABS freezes here
    log(f"delta RMS: first={sigma_first:.3e} last={rms_by_round[-1]:.3e} "
        f"min={min(rms_by_round):.3e}")

    res = {"config": {"regime": tg.REGIME, "model": tg.MODEL, "R_TRAIN": R_TRAIN,
                      "lora_r": tg.LORA_R, "lora_alpha": tg.LORA_ALPHA,
                      "n_modules": len(info), "gammas": GAMMAS,
                      "delta_rms_first": sigma_first,
                      "delta_rms_last": rms_by_round[-1],
                      "delta_rms_by_round": rms_by_round},
           "snapshots": {}}

    gen = torch.Generator(device=DEV)

    for snap in sorted(snap_states):
        state_cpu = snap_states[snap]
        state = {k: state_cpu[k].to(DEV) for k in pkeys}
        with torch.no_grad():
            base_loss = float(loss_fn(state, {}))
        # module geometry: ||A||, ||B||, ||dW||
        geo = {}
        for base in info:
            A, B = state[info[base]["A"]], state[info[base]["B"]]
            dW = dw_of(state, info, base)
            geo[base] = {"A_fro": float(A.norm()), "B_fro": float(B.norm()),
                         "dW_fro": float(dW.norm()),
                         "W0_fro": float(info[base]["mod"].base_layer.weight.norm())}
        tot = {k: sum(g[k] ** 2 for g in geo.values()) ** 0.5 for k in
               ("A_fro", "B_fro", "dW_fro", "W0_fro")}
        log(f"snap r={snap} val={base_loss:.5f} |A|={tot['A_fro']:.3f} "
            f"|B|={tot['B_fro']:.3f} |dW|={tot['dW_fro']:.3f} |W0|={tot['W0_fro']:.1f}")

        # ---- gradient direction (c), one backward reused for every dose ---- #
        # Per-CHUNK backward with grad accumulation: the full-val graph OOMs (that is
        # why make_llm_loss hands back loss_chunks -- "peak mem = one chunk").  Summing
        # weight_c * grad(lf_c) is exactly the full-val gradient (linearity).
        grads = {}
        for base in info:
            info[base]["mod"].base_layer.weight.requires_grad_(True)
        for lf, w_c in _lc:
            (w_c * lf(state, {})).backward()
        for base in info:
            w = info[base]["mod"].base_layer.weight
            grads[base] = w.grad.detach().clone()
            w.grad = None
            w.requires_grad_(False)
        torch.cuda.empty_cache()

        rows = []
        for gamma in GAMMAS:
            sigma = gamma * sigma_first                      # GN_ABS semantics
            gen.manual_seed(1234)
            # ---- (a) current scheme: independent Gaussian on A and B -------- #
            pert = dict(state)
            xis = {}
            for base in info:
                ka, kb = info[base]["A"], info[base]["B"]
                A, B = state[ka], state[kb]
                eta = torch.randn(A.shape, generator=gen, device=DEV, dtype=A.dtype) * sigma
                eps = torch.randn(B.shape, generator=gen, device=DEV, dtype=B.dtype) * sigma
                pert[ka], pert[kb] = A + eta, B + eps
                s = scaling_of(info[base]["mod"])
                xis[base] = s * ((B + eps) @ (A + eta) - B @ A)
            with torch.no_grad():
                loss_a = float(loss_fn(pert, {}))
            xi_norm = {b: float(x.norm()) for b, x in xis.items()}
            xi_tot = sum(v ** 2 for v in xi_norm.values()) ** 0.5
            del pert, xis
            torch.cuda.empty_cache()

            # ---- (b) dW-isotropic at MATCHED per-module norm --------------- #
            saved = {}
            with torch.no_grad():
                for base in info:
                    w = info[base]["mod"].base_layer.weight
                    saved[base] = w.detach().clone()
                    z = torch.randn(w.shape, generator=gen, device=DEV, dtype=w.dtype)
                    w += z * (xi_norm[base] / float(z.norm()))
                loss_b = float(loss_fn(state, {}))
                for base in info:                            # revert
                    info[base]["mod"].base_layer.weight.copy_(saved[base])

            # ---- (c) gradient direction at the same norm ------------------- #
            with torch.no_grad():
                for base in info:
                    w = info[base]["mod"].base_layer.weight
                    g = grads[base]
                    gn = float(g.norm())
                    if gn > 0:
                        w += g * (xi_norm[base] / gn)        # +grad = ascend = worst case
                loss_c = float(loss_fn(state, {}))
                for base in info:
                    info[base]["mod"].base_layer.weight.copy_(saved[base])
            del saved
            torch.cuda.empty_cache()

            row = {"gamma": gamma, "sigma": sigma, "xi_fro": xi_tot,
                   "xi_over_dW": xi_tot / tot["dW_fro"],
                   "xi_over_W0": xi_tot / tot["W0_fro"],
                   "val_a": loss_a, "val_b": loss_b, "val_c": loss_c,
                   "d_a": loss_a - base_loss, "d_b": loss_b - base_loss,
                   "d_c": loss_c - base_loss}
            rows.append(row)
            log(f"  g={gamma:6.1f} sig={sigma:.2e} |Xi|={xi_tot:9.3f} "
                f"(={row['xi_over_dW']:6.3f}|dW|) "
                f"dval a={row['d_a']:+.5f} b={row['d_b']:+.5f} c={row['d_c']:+.5f}")

        res["snapshots"][str(snap)] = {"round": snap, "base_val": base_loss,
                                       "totals": tot, "per_module": geo, "rows": rows}
        del grads, state
        torch.cuda.empty_cache()
        json.dump(res, open(OUT, "w"), indent=1)             # checkpoint each snapshot

    res["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    res["wall_s"] = time.time() - t0
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"DIAG DONE {res['wall_s']:.0f}s peak={res['peak_gib']:.1f}GiB -> {OUT}")


if __name__ == "__main__":
    main()
