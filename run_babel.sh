#!/bin/bash
#SBATCH --job-name=conti-safety-10r
#SBATCH --output=conti_%j.out
#SBATCH --error=conti_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL

# Usage: WANDB_API_KEY="your_key" sbatch run_babel.sh

set -euo pipefail

echo "=============================================="
echo " CONTI SAFETY — SLURM DEPLOYMENT"
echo " Job ID: $SLURM_JOB_ID"
echo " Node:   $(hostname)"
echo " GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo " Time:   $(date)"
echo "=============================================="

echo "[STEP 1/5] setting up environment..."

module purge 2>/dev/null || true
module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || module load conda 2>/dev/null || {
    echo "[WARN] No anaconda module found. Trying system conda..."
    export PATH="/opt/anaconda3/bin:/opt/conda/bin:$PATH"
}

# Nuke existing env
conda deactivate 2>/dev/null || true
conda env remove -n conti_env -y 2>/dev/null || true
echo "[OK] Old environment removed."

# Create fresh env
echo "[STEP 2/5] Creating fresh Python 3.10 environment..."
conda create -n conti_env python=3.10 -y --quiet
eval "$(conda shell.bash hook)"
conda activate conti_env

echo "[OK] Python version: $(python --version)"
echo "[OK] Pip version: $(pip --version)"

# -------------------------------------------------------
# 2. INSTALL DEPENDENCIES
# -------------------------------------------------------
echo "[STEP 3/5] Installing dependencies..."

# Install PyTorch first (CUDA 12.1 for L40/A100)
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project deps
pip install --quiet -r requirements.txt

# Install wandb (needed for logging)
pip install --quiet wandb

# Install the project itself in editable mode
pip install --quiet -e .

echo "[OK] All dependencies installed."
echo "[OK] torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "[OK] CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[OK] GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"

# -------------------------------------------------------
# 3. W&B SETUP
# -------------------------------------------------------
echo "[STEP 4/5] Configuring W&B..."

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[WARN] WANDB_API_KEY not set. W&B logging will be disabled."
    echo "[WARN] Results will still print to stdout (captured in SLURM .out file)."
else
    export WANDB_API_KEY="${WANDB_API_KEY}"
    wandb login --relogin "${WANDB_API_KEY}" 2>/dev/null || true
    echo "[OK] W&B authenticated."
fi

export WANDB_PROJECT="conti-safety"
export PYTHONPATH="."
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# -------------------------------------------------------
# 4. RUN THE EXPERIMENT
# -------------------------------------------------------
echo "[STEP 5/5] Launching 10-round Phase 1 experiment..."
echo "=============================================="
echo " Config: configs/phase1_10rounds_babel.yaml"
echo " Output: ./outputs/babel_phase1_10rounds"
echo " Start:  $(date)"
echo "=============================================="

python scripts/run_experiment.py \
    --config configs/phase1_10rounds_babel.yaml \
    --output-dir ./outputs/babel_phase1_10rounds

echo ""
echo "=============================================="
echo " EXPERIMENT COMPLETE"
echo " End: $(date)"
echo "=============================================="

# -------------------------------------------------------
# 5. BACKUP TO W&B (if available)
# -------------------------------------------------------
if [ -n "${WANDB_API_KEY:-}" ]; then
    python -c "
import wandb, glob, os
try:
    run = wandb.init(project='conti-safety', name='babel_artifacts', reinit=True)
    for f in glob.glob('outputs/babel_phase1_10rounds/**/*', recursive=True):
        if os.path.isfile(f) and not f.endswith('.bin'):
            wandb.save(f, policy='now')
            print(f'[W&B] Saved: {f}')
    wandb.finish()
except Exception as e:
    print(f'[WARN] W&B artifact upload failed: {e}')
" || true
fi

echo "[DONE] All results saved. Check SLURM output file: conti_${SLURM_JOB_ID}.out"
