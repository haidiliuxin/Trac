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

print("Environment variables set:")
print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
print(f"http_proxy: {os.environ.get('http_proxy')}")
print(f"https_proxy: {os.environ.get('https_proxy')}")

# Now import main
try:
    import main
except ImportError:
    # If running from different dir, add path
    sys.path.append(os.getcwd())
    import main

# Mock sys.argv
sys.argv = [
    "main.py",
    "--dataset_name", "hotpotqa-poison",
    "--model_name", "qwen0.5b",
    "--prompt_injection_attack", "default",
    "--inject_times", "5",
    "--sh_N", "5",
    "--data_num", "2",
    "--verbose", "1",
    "--gpu_id", "0"
]

if __name__ == "__main__":
    print("Running reproduction script...")
    args = main.parse_args()
    # Setup seeds like in main
    from src.utils import setup_seeds
    setup_seeds(args.seed)
    
    # Run main
    main.main(args)
