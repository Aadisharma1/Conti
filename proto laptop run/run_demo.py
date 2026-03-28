import sys
from pathlib import Path

# Add parent directory to Python path so we can import conti
sys.path.insert(0, str(Path(__file__).parent.parent))

from conti.config_schema import ContiConfig
from conti.loop.run import run_experiment


def main():
    print("==================================================")
    print("🚀 RUNNING LAPTOP PROTOTYPE DEMO")
    print("==================================================")
    print("Since an RTX 4050 has 6GB VRAM, trying to run Llama-3.1-8B")
    print("will instantly crash your laptop with an Out Of Memory error.")
    print("For this demo, we are dynamically overriding the config to use")
    print("Qwen2.5-0.5B, which fits easily in 6GB VRAM.\n")
    
    # Start with default config
    cfg = ContiConfig()
    
    # 1. Shrink the model so it fits in 6GB VRAM
    cfg.model.name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"
    cfg.model.max_seq_length = 512
    cfg.model.lora_r = 8
    
    # 2. Shrink the workload so the demo finishes fast (~2 minutes instead of 2 hours)
    cfg.loop.experiment = "baseline_single_sft"
    cfg.loop.num_self_improve_rounds = 1
    cfg.loop.max_train_samples_per_round = 5  # Only train on 5 examples
    cfg.loop.generation_max_new_tokens = 128
    
    # 3. Shrink evaluation so it doesn't take forever
    cfg.data.max_prompts_per_eval = 5  # Only eval on 5 safety prompts
    cfg.data.safety_eval_datasets = ["advbench_subset"]  # Just one benchmark for speed
    
    print(f"[*] Overriding model to: {cfg.model.name_or_path}")
    print(f"[*] Train samples reduced to: {cfg.loop.max_train_samples_per_round}")
    print("[*] Starting experiment...\n")
    
    try:
        run_experiment(cfg)
        print("\n✅ DEMO COMPLETED SUCCESSFULLY!")
        print("Check the 'outputs/run' folder for the generated metrics and logs.")
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {e}")

if __name__ == "__main__":
    main()
