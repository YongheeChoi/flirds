#!/bin/bash
# Reference: exact submit commands for REMAINING-slurm-HJ (run from repo root).
# Overrides: REPO (repo is /home/.../flirds, not $HOME/projects/flirds) + PY (rebuilt env).
# HF_HOME defaults to /scratch/chyoyhr/hf_home inside the sbatch (readable) -> not overridden.
export REPO=/home/rlaguswls186790/flirds
export PY=/home/rlaguswls186790/miniconda3/envs/flirds/bin/python
cd "$REPO"
mkdir -p runs/phase2_matrix/_logs runs/track_h/_logs runs/track_h/rundirs_llm_hj

# (1) silo5-a (a)-leg -- 9 legs, seed-major (array 0-8%8 built into the script)
sbatch runs/phase2_matrix/sbatch_silo5_a.sh

# (2) L11 online -- HJ = seed0,1 (42 runs); seed2 is YH's --array=42-62
sbatch --array=0-41%8 runs/track_h/sbatch_l11_online.sh
