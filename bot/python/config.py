# Shared constants between Python and C++.
# Keep these in sync with cpp/nnue_constants.h.

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
