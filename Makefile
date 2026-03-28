.PHONY: test lint smoke sweep clean

# run all unit tests
test:
	python -m pytest tests/ -v --tb=short

# quick lint check
lint:
	python -m ruff check conti/ scripts/ tests/

# smoke test with gpt2 (no GPU needed)
smoke:
	python scripts/run_experiment.py --config configs/smoke_gpt2.yaml

# full sweep across all arms and 3 seeds
sweep:
	python scripts/run_sweep.py \
		--base-config configs/default.yaml \
		--out-root outputs/sweep_v1 \
		--experiments baseline_frozen baseline_single_sft naive_continual phase1_verifier phase2_verifier_buffer \
		--seeds 11 22 33

# aggregate sweep results
aggregate:
	python scripts/aggregate_results.py \
		--root outputs/sweep_v1 \
		--experiments baseline_frozen baseline_single_sft naive_continual phase1_verifier phase2_verifier_buffer \
		--out outputs/sweep_v1/aggregate.json

# generate figures
plot:
	python scripts/plot_results.py \
		--sweep-root outputs/sweep_v1 \
		--aggregate outputs/sweep_v1/aggregate.json

# clean outputs
clean:
	rm -rf outputs/ .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
