import inspect
from trl import SFTConfig

params = list(inspect.signature(SFTConfig.__init__).parameters.keys())
print("count", len(params))
for name in params:
    if any(k in name for k in ("warm", "eval", "max", "dataset", "pack", "assistant")):
        print(name)
