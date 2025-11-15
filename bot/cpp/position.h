#pragma once

#include <string>

// Placeholder position type.
// Replace with your full board representation and side to move.
struct Position {
    std::string fen;

    Position() : fen("startpos") {}
    explicit Position(const std::string& f) : fen(f) {}
};
