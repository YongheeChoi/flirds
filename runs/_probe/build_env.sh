#!/bin/bash
# Recreate lora4cl-equivalent env (torch 2.11 stack) for REMAINING-slurm-HJ.
# Shared /home/chyoyhr/.../lora4cl is unreadable (home 700) -> §0-2 fallback: rebuild same spec.
# HJ = LLM-only (silo5-a + L11), so torchvision (CNN-only) is skipped.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"

ENV=flirds
echo "=== [1/3] conda create $ENV (python 3.11) $(date '+%T') ==="
conda create -y -n "$ENV" python=3.11 >/dev/null
conda activate "$ENV"
python -V

PIP="python -m pip"
$PIP install --quiet --upgrade pip

echo "=== [2/3] torch 2.11.0+cu128 $(date '+%T') ==="
$PIP install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128

echo "=== [3/3] HF stack + infra $(date '+%T') ==="
$PIP install \
  "transformers==5.5.4" "trl==1.2.0" peft accelerate datasets \
  numpy pandas pyyaml pyarrow scipy scikit-learn

echo "=== VERIFY imports $(date '+%T') ==="
python - <<'PY'
import torch, transformers, trl, peft, datasets, numpy, scipy, sklearn, pyarrow, pandas, yaml
print("torch       ", torch.__version__, "| cuda build", torch.version.cuda)
print("transformers", transformers.__version__)
print("trl         ", trl.__version__)
print("peft        ", peft.__version__)
print("datasets    ", datasets.__version__)
print("accelerate  ", __import__("accelerate").__version__)
from trl import SFTConfig, SFTTrainer  # the exact API the runners use
print("trl SFT API  OK")
print("ENV BUILD DONE")
PY
