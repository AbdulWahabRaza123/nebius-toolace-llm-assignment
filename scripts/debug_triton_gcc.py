import os
import subprocess
import tempfile
from pathlib import Path

triton_inc = Path(os.environ.get("VIRTUAL_ENV", "")) / "lib/python3.12/site-packages/triton/backends/nvidia/include"
triton_lib = Path(os.environ.get("VIRTUAL_ENV", "")) / "lib/python3.12/site-packages/triton/backends/nvidia/lib"
cuda_home = Path("/usr/local/cuda-13.0")

src = triton_inc / "cuda_utils.c"
if not src.exists():
    raise SystemExit(f"missing {src}")

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "cuda_utils.so"
    cmd = [
        "/usr/bin/gcc",
        str(src),
        "-O3",
        "-shared",
        "-fPIC",
        "-Wno-psabi",
        "-o",
        str(out),
        "-l:libcuda.so.1",
        f"-L{triton_lib}",
        "-L/lib/x86_64-linux-gnu",
        f"-I{triton_inc}",
        f"-I{tmp}",
        "-I/usr/include/python3.12",
        f"-I{cuda_home}/include",
    ]
    print("CMD:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("RC:", proc.returncode)
    print("STDOUT:", proc.stdout[-2000:])
    print("STDERR:", proc.stderr[-2000:])
