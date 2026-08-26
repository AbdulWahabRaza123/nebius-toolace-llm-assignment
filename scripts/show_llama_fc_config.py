from pathlib import Path

text = Path(
    "/home/nebius-assignment/gorilla/berkeley-function-call-leaderboard/bfcl_eval/constants/model_config.py"
).read_text(encoding="utf-8")
key = '"meta-llama/Llama-3.1-8B-Instruct-FC"'
idx = text.find(key)
print(text[idx : idx + 700])
