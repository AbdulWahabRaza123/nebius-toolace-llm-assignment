from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path

token = Path("/home/nebius-assignment/ml-ops/.env").read_text().strip().split("=", 1)[1]
api = HfApi(token=token)
user = api.whoami()
print("user", user.get("name"))

repos = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Team-ACE/ToolACE-8B",
]
for repo in repos:
    try:
        info = api.model_info(repo)
        print(repo, "gated=", getattr(info, "gated", False))
    except Exception as e:
        print(repo, "ERROR", str(e)[:200])

try:
    path = hf_hub_download(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "config.json",
        token=token,
    )
    print("config_download_ok", path)
except Exception as e:
    print("config_download_fail", type(e).__name__, str(e)[:300])
