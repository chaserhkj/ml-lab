# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

# Script to setup kernel for a sub-lab
# Run with uv: uv run setup_kernel.py <sub-lab-name>

import os
import subprocess
import argparse
from sys import stderr

kernel_deps = [
    "ipykernel", "ipywidgets", "python-dotenv",
    "jupyter_paint_segment", "ipyevents", "ipycanvas", "orjson"
]

def sh(command: str) -> None:
    _ = subprocess.run(command, shell=True, check=True)

def main(sub_lab: str) -> int:
    # Set from environment
    base_dir = os.environ.get("DATA_ROOT", "/work")
    runtime_type = os.environ.get("RUNTIME_TYPE", "rocm")
    repo_dir = f"{base_dir}/ml-lab"
    if not os.path.isdir(repo_dir):
        sh(f"git clone https://github.com/chaserhkj/ml-lab/ {repo_dir}")
    lab_dir = f"{repo_dir}/{sub_lab}"
    if not os.path.isdir(lab_dir):
        print(f"Sub lab {sub_lab} not found")
        return 1
    os.chdir(lab_dir)
    os.environ["VIRTUAL_ENV"] = f"{lab_dir}/.venv"
    sh("uv venv --clear")
    sh(f"uv sync --extra {runtime_type}-runtime")
    sh(f"uv pip install --index-strategy unsafe-best-match {" ".join(kernel_deps)}")
    sh("uv run python3 -m ipykernel install --prefix /usr/local"+
        f" --env VIRTUAL_ENV {lab_dir}/.venv"+
        f" --name {sub_lab} --display-name 'Python ({sub_lab})'")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("sub_lab", type=str, help = "Sub Lab folder to setup kernel for")
    args = parser.parse_args()
    try:
        rtn = main(**args.__dict__) # pyright: ignore[reportAny]
        exit(rtn)
    except subprocess.CalledProcessError as e:
        print(f"{e.cmd} failed with rtn {e.returncode}", file=stderr)  # pyright: ignore[reportAny]
        exit(e.returncode)