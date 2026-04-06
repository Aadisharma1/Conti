#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Local dry run on Qwen2.5-0.5B-Instruct
# Tests the entire pipeline with tiny data on your laptop GPU
# ─────────────────────────────────────────────────────────────

set -e

export GROQ_API_KEY="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"

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
RESULTS_DIR="results/local_test"
CKPT_DIR="checkpoints/local_test"
DATA_DIR="data/local_test"

mkdir -p $RESULTS_DIR $CKPT_DIR $DATA_DIR

echo "=== Step 1: Generate self-edits (10 samples each) ==="
python src/groq_generator.py \
    --output-dir $DATA_DIR \
    --squad-samples 10 \
    --gsm8k-samples 10 \
    --dataset both

echo "=== Step 2: Baseline eval ==="
python src/eval.py \
    --base-model $MODEL \
    --arm-name "local_baseline" \
    --output $RESULTS_DIR/eval.json \
    --squad-samples 10 \
    --gsm8k-samples 10 \
    --advbench-samples 10 \
    --xstest-samples 10

echo "=== Step 3: Train with EWC (1 epoch, tiny data) ==="
python src/train_ewc.py \
    --base-model $MODEL \
    --train-data $DATA_DIR/squad_edits.jsonl \
    --output-dir $CKPT_DIR/ewc_test \
    --epochs 1 \
    --batch-size 2 \
    --lr 2e-5 \
    --ewc-lambda 0.5 \
    --fisher-samples 5 \
    --lora-rank 8 \
    --max-length 512

echo "=== Step 4: Post-training eval ==="
python src/eval.py \
    --base-model $MODEL \
    --adapter-path $CKPT_DIR/ewc_test \
    --arm-name "local_ewc_trained" \
    --output $RESULTS_DIR/eval.json \
    --squad-samples 10 \
    --gsm8k-samples 10 \
    --advbench-samples 10 \
    --xstest-samples 10

echo ""
echo "=== LOCAL DRY RUN COMPLETE ==="
echo "Results in: $RESULTS_DIR/eval.json"
