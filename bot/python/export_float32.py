"""Export a trained NNUEModel (256x2-32-32-1) to float32 for the C++ engine.

Why float32 rather than the existing int16 path: export_int16.py quantizes by
DIVIDING by the scale instead of multiplying --

    w_acc_f_q = np.round(acc_f_w / SCALE1)     # SCALE1 = 32.0

-- and the accumulator weights have absmax ~1.5, absmean ~0.16, so all 393,216 of
them round to zero. Every committed .bin has an all-zero accumulator and evaluates
every position as 0. Fixing that properly needs matching per-layer scales and
rescaling shifts in NNUE::evaluate too, so this exports the weights verbatim and
lets the engine do the arithmetic in float. The network is small and the input is
sparse (<=32 of 768 features set), so this is cheap.

Feature layout is taken from train.py's fen_to_features, which is what the weights
were actually trained against:

    feature index = square * 12 + piece_offset      (square-major)
    piece_offset  = 0..5 for WHITE P,N,B,R,Q,K and 6..11 for BLACK

Note this is colour-absolute and carries no side-to-move bit, so the model's output
is from White's point of view. The engine negates it when Black is to move.

Accumulator weights are written feature-major and interleaved (friendly then enemy
for each feature) so the engine reads one contiguous run per active feature.

Usage:
    python bot/python/export_float32.py --checkpoint bot/python/overnight.pt \
        --output bot/python/nnue_float.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import NNUEModel  # noqa: E402

MAGIC = b"PFNN"
VERSION = 1
FEATURE_DIM = 768
ACC_UNITS = 256
HIDDEN1 = 32
HIDDEN2 = 32


def load_model(path: Path) -> NNUEModel:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw
    if isinstance(raw, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            if key in raw:
                state = raw[key]
                break
    model = NNUEModel()
    model.load_state_dict(state)
    model.eval()
    return model


def _f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy().astype(np.float32)


def export(model: NNUEModel, out_path: Path) -> None:
    accf_w = _f32(model.acc_friendly.weight)   # (ACC_UNITS, FEATURE_DIM)
    accf_b = _f32(model.acc_friendly.bias)
    acce_w = _f32(model.acc_enemy.weight)
    acce_b = _f32(model.acc_enemy.bias)
    fc1_w = _f32(model.fc1.weight)             # (HIDDEN1, 2*ACC_UNITS)
    fc1_b = _f32(model.fc1.bias)
    fc2_w = _f32(model.fc2.weight)
    fc2_b = _f32(model.fc2.bias)
    out_w = _f32(model.fc_out.weight).reshape(-1)
    out_b = _f32(model.fc_out.bias).reshape(-1)

    assert accf_w.shape == (ACC_UNITS, FEATURE_DIM), accf_w.shape
    assert fc1_w.shape == (HIDDEN1, 2 * ACC_UNITS), fc1_w.shape
    assert fc2_w.shape == (HIDDEN2, HIDDEN1), fc2_w.shape
    assert out_w.shape == (HIDDEN2,), out_w.shape

    # Feature-major and interleaved: acc[f] = [friendly 256 | enemy 256]
    interleaved = np.concatenate([accf_w.T, acce_w.T], axis=1)  # (FEATURE_DIM, 512)
    assert interleaved.shape == (FEATURE_DIM, 2 * ACC_UNITS)

    with out_path.open("wb") as fh:
        fh.write(MAGIC)
        for v in (VERSION, FEATURE_DIM, ACC_UNITS, HIDDEN1, HIDDEN2):
            fh.write(struct.pack("<i", v))
        fh.write(np.ascontiguousarray(accf_b).tobytes())
        fh.write(np.ascontiguousarray(acce_b).tobytes())
        fh.write(np.ascontiguousarray(interleaved).tobytes())
        fh.write(np.ascontiguousarray(fc1_b).tobytes())
        fh.write(np.ascontiguousarray(fc1_w).tobytes())
        fh.write(np.ascontiguousarray(fc2_b).tobytes())
        fh.write(np.ascontiguousarray(fc2_w).tobytes())
        fh.write(np.ascontiguousarray(out_b).tobytes())
        fh.write(np.ascontiguousarray(out_w).tobytes())

    expected = (
        4 + 5 * 4
        + ACC_UNITS * 4 * 2
        + FEATURE_DIM * 2 * ACC_UNITS * 4
        + HIDDEN1 * 4 + HIDDEN1 * 2 * ACC_UNITS * 4
        + HIDDEN2 * 4 + HIDDEN2 * HIDDEN1 * 4
        + 4 + HIDDEN2 * 4
    )
    actual = out_path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"wrote {actual} bytes, expected {expected}")
    print(f"wrote {out_path} ({actual} bytes)")


def reference_eval(model: NNUEModel, fen: str) -> float:
    """The training-time forward pass, used to validate the C++ port."""
    import chess

    board = chess.Board(fen)
    v = torch.zeros(FEATURE_DIM, dtype=torch.float32)
    pidx = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            off = pidx[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
            v[sq * 12 + off] = 1.0
    with torch.no_grad():
        return float(model(v.unsqueeze(0)).item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(SCRIPT_DIR / "overnight.pt"))
    ap.add_argument("--output", default=str(SCRIPT_DIR / "nnue_float.bin"))
    ap.add_argument("--dump-refs", default="",
                    help="write 'fen<TAB>eval' reference lines here for the C++ check")
    args = ap.parse_args()

    model = load_model(Path(args.checkpoint))
    export(model, Path(args.output))

    if args.dump_refs:
        import chess
        import random

        rng = random.Random(11)
        fens = []
        for _ in range(300):
            b = chess.Board()
            for _ in range(rng.randint(0, 60)):
                moves = list(b.legal_moves)
                if not moves:
                    break
                b.push(rng.choice(moves))
            if not b.is_game_over():
                fens.append(b.fen())
        with open(args.dump_refs, "w") as fh:
            for f in fens:
                fh.write(f"{f}\t{reference_eval(model, f):.6f}\n")
        print(f"wrote {len(fens)} reference evaluations to {args.dump_refs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
