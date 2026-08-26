from pathlib import Path
import re

p = Path("/home/nebius-assignment/gorilla/berkeley-function-call-leaderboard/bfcl_eval/constants/model_config.py")
text = p.read_text(encoding="utf-8")
keys = re.findall(r'"([^"]+)": ModelConfig\(', text)
print("n_keys", len(keys))
print("llama keys:")
for k in keys:
    if "llama" in k.lower() or "Llama" in k:
        print(" ", k)
print("openai-like sample:")
for k in keys:
    if k.endswith("-FC") and ("gpt" in k or "openai" in k.lower()):
        print(" ", k)
        break

# Find a concrete ModelConfig block using OpenAICompletionsHandler
idx = text.find("OpenAICompletionsHandler")
print("OpenAICompletionsHandler first idx", idx)
print(text[idx: idx + 500] if idx >= 0 else "missing")

# Find QuickTesting entry
idx2 = text.find("QuickTestingOSSHandler")
print("QuickTesting idx", idx2)
if idx2 > 0:
    start = text.rfind('"', 0, idx2)
    print(text[max(0, idx2 - 400): idx2 + 350])
