import os

_cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0")
os.environ.setdefault("CUDA_HOME", _cuda_home)
os.environ["PATH"] = f"{_cuda_home}/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{_cuda_home}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["TORCH_NATIVE_DISABLE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "models/qwen2.5-7b-instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
inputs = tok("hello", return_tensors="pt").to(model.device)
out = model(**inputs, labels=inputs["input_ids"])
print("loss", float(out.loss))
