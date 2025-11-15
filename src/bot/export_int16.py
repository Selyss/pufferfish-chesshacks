import torch
import numpy as np


class LegacyNNUE(torch.nn.Module):
    """Classic NNUE layout compatible with pufferfish engine."""

    def __init__(self, feature_dim: int = 768, acc_units: int = 256, hidden1: int = 32, hidden2: int = 32) -> None:
        super().__init__()
        self.acc_friendly = torch.nn.Linear(feature_dim, acc_units)
        self.acc_enemy = torch.nn.Linear(feature_dim, acc_units)
        self.fc1 = torch.nn.Linear(acc_units * 2, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc_out = torch.nn.Linear(hidden2, 1)

SQUARES = 64
PIECE_TYPES = 6      # P, N, B, R, Q, K per color
COLORS = 2          # white, black

ACC_UNITS = 256      # accumulator size per side
HIDDEN1 = 32
HIDDEN2 = 32

# Total feature count depends on your feature encoding.
# Using simple piece-square encoding: 64 squares * 12 piece types (6 white + 6 black)
FEATURE_DIM = 768  # 64 * 12

# Quantization parameters
RELU_CLIP = 255
OUTPUT_SCALE_BITS = 5    # 2^5 = 32
SCALE1 = 32.0            # scale for first stage
SCALE2 = 2.0 ** OUTPUT_SCALE_BITS


def export_quantized(pt_path: str, bin_path: str):
    """
    Export a trained PyTorch model to quantized int16 binary format.
    
    Args:
        pt_path: Path to the .pt file (e.g., 'nnue_state_dict.pt')
        bin_path: Output path for the binary file (e.g., 'nnue_weights.bin')
    """
    import os
    
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Model file not found: {pt_path}")
    
    print(f"Loading model from {pt_path}...")
    try:
        raw = torch.load(pt_path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(pt_path, map_location="cpu")

    if isinstance(raw, dict) and "model_state" in raw:
        state = raw["model_state"]
    else:
        state = raw

    model = LegacyNNUE(feature_dim=FEATURE_DIM, acc_units=ACC_UNITS, hidden1=HIDDEN1, hidden2=HIDDEN2)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "Checkpoint structure does not match legacy NNUE layout. "
            f"Missing keys: {missing}, unexpected: {unexpected}."
        )
    model.eval()
    print("Model loaded successfully!")

    # Extract weights
    print("Extracting weights from model...")
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
    
    print(f"Accumulator weights shape: {acc_f_w.shape}")
    print(f"FC1 weights shape: {fc1_w.shape}")
    print(f"FC2 weights shape: {fc2_w.shape}")
    print(f"Output weights shape: {out_w.shape}")

    # First stage quantization
    print("Quantizing weights to int16...")
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

    print("Quantization ranges:")
    print(f"  acc_friendly weights: [{w_acc_f_q.min()}, {w_acc_f_q.max()}]")
    print(f"  acc_enemy weights:    [{w_acc_e_q.min()}, {w_acc_e_q.max()}]")
    print(f"  fc1 weights:          [{w_fc1_q.min()}, {w_fc1_q.max()}]")
    print(f"  fc2 weights:          [{w_fc2_q.min()}, {w_fc2_q.max()}]")
    print(f"  output weights:       [{w_out_q.min()}, {w_out_q.max()}]")

    print(f"Writing binary file to {bin_path}...")
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

    import os
    file_size = os.path.getsize(bin_path)
    print(f"\n✓ Successfully exported quantized NNUE to {bin_path}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"\nBinary format:")
    print(f"  - FEATURE_DIM: {FEATURE_DIM}")
    print(f"  - ACC_UNITS: {ACC_UNITS}")
    print(f"  - HIDDEN1: {HIDDEN1}")
    print(f"  - HIDDEN2: {HIDDEN2}")
    print(f"  - Quantization scales: SCALE1={SCALE1}, SCALE2={SCALE2}")


if __name__ == "__main__":
    import sys
    
    # Allow custom paths via command line arguments
    if len(sys.argv) >= 2:
        pt_path = sys.argv[1]
        bin_path = sys.argv[2] if len(sys.argv) >= 3 else "nnue_weights.bin"
    else:
        pt_path = "nnue_state_dict.pt"
        bin_path = "nnue_weights.bin"
    
    try:
        export_quantized(pt_path, bin_path)
    except Exception as e:
        print(f"\n❌ Error during export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
