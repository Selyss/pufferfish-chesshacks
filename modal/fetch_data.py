from __future__ import annotations

import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("nnue-data-fetch")

RAW_VOLUME = modal.Volume.from_name("lichess-raw", create_if_missing=True)
PARQUET_VOLUME = modal.Volume.from_name("lichess-parquet", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("zstd", "stockfish")
    .pip_install("requests", "python-chess", "datasets", "zstandard", "tqdm")
)


@app.function(
    image=base_image.add_local_dir(".", remote_path="/root/app"),
    gpu="A100-40GB",
    timeout=60 * 60 * 12,
    volumes={
        "/data-raw": RAW_VOLUME,
        "/data-out": PARQUET_VOLUME,
    },
)
def fetch_on_modal(
    month: str,
    depth: int = 18,
    games: int = 1000,
    positions: int | None = None,
    run_name: str | None = None,
) -> None:
    RAW_VOLUME.reload()
    PARQUET_VOLUME.reload()
    output_name = run_name or f"lichess-{month}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    output_path = Path("/data-out") / f"{output_name}.parquet"
    workdir = Path("/data-raw") / month
    workdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "tools.data.fetch_lichess",
        f"--month={month}",
        f"--engine=stockfish",
        f"--depth={depth}",
        f"--games={games}",
        f"--output={output_path}",
        f"--workdir={workdir}",
    ]
    if positions:
        cmd.append(f"--positions={positions}")

    print("Running:", " ".join(cmd))
    env = {**os.environ, "STOCKFISH_PATH": "/usr/bin/stockfish"}
    subprocess.run(cmd, cwd="/root/app", check=True, env=env)
    PARQUET_VOLUME.commit()
    RAW_VOLUME.commit()
    print(f"Dataset stored at {output_path}")


@app.local_entrypoint()
def main(
    month: str,
    depth: int = 18,
    games: int = 1000,
    positions: int | None = None,
    run_name: str | None = None,
) -> None:
    fetch_on_modal.remote(
        month=month,
        depth=depth,
        games=games,
        positions=positions,
        run_name=run_name,
    )
