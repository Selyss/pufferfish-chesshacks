#pragma once

// NNUE layout constants. These must match the training/export pipeline.
// For now, we mirror the existing bot/cpp configuration.

#include <cstdint>

constexpr int SQUARES = 64;
constexpr int PIECE_TYPES = 6;
constexpr int COLORS = 2;

constexpr int FEATURE_DIM = 768; // 64 squares * 12 piece types
constexpr int ACC_UNITS = 256;   // per side
constexpr int HIDDEN1 = 32;
constexpr int HIDDEN2 = 32;

using Weight = int16_t;
using Accum = int32_t;
using Activation = int16_t;

constexpr int RELU_CLIP = 255;
constexpr int OUTPUT_SCALE_BITS = 5;
