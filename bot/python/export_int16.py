import torch
import numpy as np

from config import (
    FEATURE_DIM,
    ACC_UNITS,
    HIDDEN1,
    HIDDEN2,
    RELU_CLIP,
    OUTPUT_SCALE_BITS,
    SCALE1,
    SCALE2,
)
from model import NNUEModel


def export_quantized(pt_path: str, bin_path: str):
    model = NNUEModel()
    state = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Extract weights
    acc_f_w = model.acc_friendly.weight.detach().numpy()  # (ACC_UNITS, F)
    acc_f_b = model.acc_friendly.bias.detach().numpy()    # (ACC_UNITS,)
    acc_e_w = model.acc_enemy.weight.detach().numpy()     # (ACC_UNITS, F)
    acc_e_b = model.acc_enemy.bias.detach().numpy()       # (ACC_UNITS,)

    fc1_w = model.fc1.weight.detach().numpy()             # (HIDDEN1, 2*ACC_UNITS)
    fc1_b = model.fc1.bias.detach().numpy()               # (HIDDEN1,)
    fc2_w = model.fc2.weight.detach().numpy()             # (HIDDEN2, HIDDEN1)
    fc2_b = model.fc2.bias.detach().numpy()               # (HIDDEN2,)
    out_w = model.fc_out.weight.detach().numpy()          # (1, HIDDEN2)
    out_b = model.fc_out.bias.detach().numpy()[0]         # scalar

    # First stage quantization
    w_acc_f_q = np.round(acc_f_w / SCALE1).astype(np.int16)
    w_acc_e_q = np.round(acc_e_w / SCALE1).astype(np.int16)
    b_acc_f_q = np.round(acc_f_b / SCALE1).astype(np.int32)
    b_acc_e_q = np.round(acc_e_b / SCALE1).astype(np.int32)

    # Fully connected parts. Fold SCALE1 in and scale down by SCALE2.
    w_fc1_q = np.round(fc1_w * SCALE1 / SCALE2).astype(np.int16)
    b_fc1_q = np.round(fc1_b / SCALE2).astype(np.int32)

    w_fc2_q = np.round(fc2_w * SCALE1 / SCALE2).astype(np.int16)
    b_fc2_q = np.round(fc2_b / SCALE2).astype(np.int32)

    w_out_q = np.round(out_w * SCALE1 / SCALE2).astype(np.int16)
    b_out_q = int(np.round(out_b / SCALE2))

    print("acc_f weights:", w_acc_f_q.min(), w_acc_f_q.max())
    print("acc_e weights:", w_acc_e_q.min(), w_acc_e_q.max())

    with open(bin_path, "wb") as f:
        # Header: feature count and dims
        f.write(np.int32(FEATURE_DIM).tobytes())
        f.write(np.int32(ACC_UNITS).tobytes())
        f.write(np.int32(HIDDEN1).tobytes())
        f.write(np.int32(HIDDEN2).tobytes())

        # First stage biases: friendly and enemy, each ACC_UNITS int32
        f.write(b_acc_f_q.astype(np.int32).tobytes())
        f.write(b_acc_e_q.astype(np.int32).tobytes())

        # First stage weights in feature-major layout:
        # For each feature f: ACC_UNITS friendly weights then ACC_UNITS enemy weights
        w_acc_f_feature_major = w_acc_f_q.T  # (F, ACC_UNITS)
        w_acc_e_feature_major = w_acc_e_q.T  # (F, ACC_UNITS)
        for f_idx in range(FEATURE_DIM):
            f.write(w_acc_f_feature_major[f_idx].astype(np.int16).tobytes())
            f.write(w_acc_e_feature_major[f_idx].astype(np.int16).tobytes())

        # FC1: 2*ACC_UNITS -> HIDDEN1
        f.write(b_fc1_q.astype(np.int32).tobytes())
        f.write(w_fc1_q.astype(np.int16).tobytes())

        # FC2: HIDDEN1 -> HIDDEN2
        f.write(b_fc2_q.astype(np.int32).tobytes())
        f.write(w_fc2_q.astype(np.int16).tobytes())

        # Output: HIDDEN2 -> 1
        f.write(np.int32(b_out_q).tobytes())
        f.write(w_out_q.astype(np.int16).tobytes())

    print(f"Exported quantized NNUE to {bin_path}")


if __name__ == "__main__":
    export_quantized("nnue_state_dict.pt", "nnue_weights.bin")
