from __future__ import annotations

import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("nnue-trainer")
CHECKPOINT_VOLUME = modal.Volume.from_name("nnue-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "datasets>=4.4.0",
        "numpy>=2.2.0",
        "pyarrow>=12.0.0",
        "python-chess>=1.999",
        "tqdm>=4.66.0",
    )
    .pip_install("torch==2.3.1", index_url="https://download.pytorch.org/whl/cu121")
)


@app.function(
    image=image.add_local_dir(".", remote_path="/root/app"),
    gpu="A100-40GB",
    timeout=60 * 60 * 24,
    volumes={"/outputs": CHECKPOINT_VOLUME},
)
def train_on_a100(
    run_name: str = "modal-run",
    batch_size: int = 8192,
    epochs: int = 6,
    limit_rows: int | None = None,
    lr: float = 3e-4,
    seed: int = 42,
    extra_args: str | None = None,
) -> None:
    CHECKPOINT_VOLUME.reload()
    output_dir = Path("/outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "src.bot.train_nnue",
        f"--batch-size={batch_size}",
        f"--epochs={epochs}",
        f"--lr={lr}",
        f"--seed={seed}",
        f"--output-dir={output_dir}",
        "--amp",
        "--num-workers=4",
        "--device=cuda",
    ]
    if limit_rows:
        cmd.extend(["--limit-rows", str(limit_rows)])
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd="/root/app", check=True, env=os.environ.copy())
    CHECKPOINT_VOLUME.commit()


@app.local_entrypoint()
def main(
    run_name: str | None = None,
    batch_size: int = 8192,
    epochs: int = 6,
    limit_rows: int | None = None,
    lr: float = 3e-4,
    seed: int = 42,
    extra_args: str | None = None,
) -> None:
    resolved = run_name or f"modal-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    train_on_a100.remote(
        run_name=resolved,
        batch_size=batch_size,
        epochs=epochs,
        limit_rows=limit_rows,
        lr=lr,
        seed=seed,
        extra_args=extra_args,
    )
