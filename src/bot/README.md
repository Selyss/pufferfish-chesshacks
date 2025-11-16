# ChessHacks Bot Training

This directory contains everything related to training a standalone NNUE model for your ChessHacks bot without polluting the main runtime environment.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r src/bot/requirements.txt
```

`python-chess`, `torch`, and Hugging Face `datasets` are the only heavy dependencies that are specific to training, which is why they live in `src/bot/requirements.txt`.

## Dataset

The trainer works with [`LegendaryAKx3/light-preprocessed`](https://huggingface.co/datasets/LegendaryAKx3/light-preprocessed). Each record looks like:

```text
{
  "fen": "r2qkb1r/pp1b1ppp/2np4/1Bpn4/8/1P3N2/PBPP1PPP/RN1Q1RK1 b kq -",
  "depth": 48,
  "knodes": 1830500,
  "cp_label": 23,
  "eval_bucket": 2
}
```

`cp_label` is treated as the regression target (scaled to pawns by default), though you can switch to `eval_bucket` classification labels with `--label-key eval_bucket`.

## Training

Run the CLI entry point with `python -m src.bot.train_nnue` (relative to the repo root). Common flags:

```bash
python -m src.bot.train_nnue \
  --batch-size 4096 \
  --epochs 5 \
  --limit-rows 500000 \
  --lr 3e-4 \
  --hidden-dims 2048 2048 1024 512 256 \
  --residual-repeats 3 3 2 2 1 \
  --output-dir src/bot/checkpoints/run1
```

Options worth knowing:

| Flag | Description |
| --- | --- |
| `--limit-rows N` | Train on the first `N` records to iterate quickly. |
| `--target-scale S` | Multiply labels by `S`. Default `0.01` converts centipawns to pawns. |
| `--skip-depth-feature` / `--skip-knodes-feature` | Drop metadata features if you only want pure board encodings. |
| `--weight-by-depth` / `--weight-by-knodes` | Increase the contribution of deeper or more expensive searches during training. |
| `--residual-repeats` | Number of residual refinement blocks to insert after each hidden layer. Defaults to 2 per layer. |
| `--amp` | Enables mixed precision when CUDA is available. |

Every run writes:

- `src/bot/checkpoints/<run>/training_config.json` – exact CLI + encoder config
- `src/bot/checkpoints/<run>/history.json` – per-epoch metrics
- `src/bot/checkpoints/<run>/nnue_epochXXX.pt` – serialized checkpoints

The checkpoints store `model_state`, `optimizer_state`, and stats, so you can resume training manually by loading them in a custom script later.

## Runtime Integration

`src/main.py` now loads the NNUE checkpoint and drives an alpha-beta searcher to produce moves. Configure it through environment variables (add them to `.env.local` in the repo root):

| Variable | Default | Description |
| --- | --- | --- |
| `CHESSBOT_CHECKPOINT` | `src/bot/checkpoints/nnue_epoch001.pt` | Path to the trained checkpoint containing `model_config`. |
| `CHESSBOT_DEVICE` | `cpu` | Torch device string (`cpu`, `mps`, `cuda`). |
| `CHESSBOT_MAX_DEPTH` | `4` | Maximum depth for the iterative deepening search. |
| `CHESSBOT_QUIESCENCE_DEPTH` | `4` | Maximum capture plies before quiescence terminates. |
| `CHESSBOT_TIME_FRACTION` | `0.02` | Fraction of the remaining clock devoted to the current move. |
| `CHESSBOT_MIN_TIME_MS` / `CHESSBOT_MAX_TIME_MS` | `100` / `4000` | Clamp the allocated think time in milliseconds. |
| `CHESSBOT_TEMPERATURE` | `0.6` | Softmax temperature for logging move probabilities. |

At runtime:

1. `NNUEEvaluator` loads the checkpoint and builds incrementally updatable features (`src/bot/nnue.py`). These features now include per-piece counts and a phase indicator that update without re-encoding the whole board.
2. `AlphaBetaSearch` (`src/bot/search.py`) performs iterative deepening with quiescence and logs the per-move softmax for transparency.
3. `chess_manager.entrypoint` delegates to the engine so the Next.js devtools receive the best move from the trained model automatically.

## Training on Modal (A100 GPU)

If you just want to run `src.bot.train_nnue` on a GPU without worrying about manual setup, the repo includes `modal/train.py`:

```bash
pip install modal
modal token new

modal run modal/train.py \
  --run-name modal-compact-a100 \
  --epochs 6 \
  --batch-size 8192 \
  --limit-rows 30000000 \
  --amp \
  --model compact \
  --log-interval 100 \
  --log-interval-eval \
  --extra-args "--skip-depth-feature --skip-knodes-feature"
```

The function:

- Mounts the repo at `/root/app`.
- Installs CUDA-enabled PyTorch and the training requirements.
- Runs `python -m src.bot.train_nnue ...` so all relative imports keep working.
- Writes checkpoints to `src/bot/checkpoints/<run-name>` inside the mounted tree (download via `modal run ...` stdout or `modal shell` if you need to fetch them).

Pass any extra CLI flags via `--extra-args`. You can also change defaults (`--run-name`, `--epochs`, `--batch-size`, etc.) directly on the Modal CLI call.

### Building Training Data on Modal

Generating millions of annotated positions is CPU-heavy. To offload it, use `modal/fetch_data.py`:

```bash
pip install modal
modal token new

modal run modal/fetch_data.py --month 2024-08 --depth 20 --games 3000 --positions 200000 --run-name lichess-2024-08
```

This mounts two volumes (`lichess-raw` for downloaded PGNs and `lichess-parquet` for the final dataset), installs Stockfish + zstd inside the container, and runs `tools/data/fetch_lichess`. After the job completes, pull the dataset locally:

```bash
modal volume get lichess-parquet/lichess-2024-08.parquet data/lichess-2024-08.parquet
```

You can then point the trainer at the downloaded Parquet file by adapting `train_nnue.py` or loading it via Hugging Face `datasets`.
