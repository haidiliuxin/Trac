import os
import sys

# Set environment variables to bypass proxy and use mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Remove proxy settings
if "http_proxy" in os.environ: del os.environ["http_proxy"]
if "https_proxy" in os.environ: del os.environ["https_proxy"]
if "HTTP_PROXY" in os.environ: del os.environ["HTTP_PROXY"]
if "HTTPS_PROXY" in os.environ: del os.environ["HTTPS_PROXY"]
# Force no proxy
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

# Now import main
try:
    import main
except ImportError:
    # If running from different dir, add path
    sys.path.append(os.getcwd())
    import main

from src.utils import setup_seeds

def run_experiment(dataset_name, attack_type="default", data_num=5):
    print(f"\n\n{'='*50}")
    print(f"Running Experiment: {dataset_name} ({attack_type})")
    print(f"{'='*50}\n")
    
    # Mock sys.argv
    sys.argv = [
        "main.py",
        "--dataset_name", dataset_name,
        "--model_name", "deepseek-chat", # Will load from model_configs/deepseek_config.json
        "--prompt_injection_attack", attack_type,
        "--inject_times", "5",
        "--sh_N", "5", # Increased permutations for better defense
        "--K", "10", # Increase top-K removal to 10 for stronger defense
        "--data_num", "1", # Just 1 sample
        "--verbose", "1",
        "--gpu_id", "0",
        "--results_path", f"results_{dataset_name}_{attack_type}"
    ]

    print("Running reproduction script...")
    args = main.parse_args()
    setup_seeds(args.seed)
    
    # Run main
    try:
        main.main(args)
    except Exception as e:
        print(f"Error running {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Task 1: Prompt Injection on MuSiQue
    # run_experiment("musique", "default", data_num=2)
    
    # Task 2: Knowledge Corruption on NQ (Natural Questions)
    # dataset_name for poison is 'nq-poison'
    run_experiment("nq-poison", "default", data_num=1)
