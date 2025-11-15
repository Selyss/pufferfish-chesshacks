#pragma once

#include <cstdint>

// Must match python/config.py

constexpr int SQUARES = 64;
constexpr int PIECE_TYPES = 6;
constexpr int COLORS = 2;

constexpr int FEATURE_DIM = 768;
constexpr int ACC_UNITS   = 256;     // per side
constexpr int HIDDEN1     = 32;
constexpr int HIDDEN2     = 32;

// Quantization types and parameters
using Weight     = int16_t;
using Accum      = int32_t;
using Activation = int16_t;

constexpr int RELU_CLIP         = 255;
constexpr int OUTPUT_SCALE_BITS = 5;   // right shift bits
