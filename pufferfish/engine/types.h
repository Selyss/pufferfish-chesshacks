// Core engine-wide types and constants.

#pragma once

#include <cstdint>
#include <array>

namespace pf
{

    using Bitboard = std::uint64_t;
    using Key = std::uint64_t;

    constexpr int MAX_PLY = 128;
    constexpr int MAX_MOVES = 256;
    constexpr int INF_SCORE = 32000;
    constexpr int MATE_SCORE = 31000;
    constexpr int TB_WIN_SCORE = 30000; // reserve margin
    constexpr int DRAW_SCORE = 0;

    enum Color : int
    {
        WHITE = 0,
        BLACK = 1,
        COLOR_NB = 2
    };

    enum PieceType : int
    {
        PAWN = 0,
        KNIGHT,
        BISHOP,
        ROOK,
        QUEEN,
        KING,
        PIECE_TYPE_NB
    };

    enum Piece : int
    {
        NO_PIECE = 0,
        W_PAWN,
        W_KNIGHT,
        W_BISHOP,
        W_ROOK,
        W_QUEEN,
        W_KING,
        B_PAWN,
        B_KNIGHT,
        B_BISHOP,
        B_ROOK,
        B_QUEEN,
        B_KING,
        PIECE_NB
    };

    inline constexpr Color operator~(Color c) { return Color(c ^ 1); }

    // Packed move: from(6) | to(6) | piece(4) | promo(4) | flags(4)
    using Move = std::uint32_t;

    constexpr Move MOVE_NONE = 0u;

    enum MoveFlag : std::uint32_t
    {
        FLAG_NONE = 0,
        FLAG_CAPTURE = 1u << 0,
        FLAG_PROMOTION = 1u << 1,
        FLAG_ENPASSANT = 1u << 2,
        FLAG_CASTLING = 1u << 3
    };

    inline int from_sq(Move m) { return int(m & 0x3Fu); }
    inline int to_sq(Move m) { return int((m >> 6) & 0x3Fu); }
    inline int move_piece(Move m) { return int((m >> 12) & 0xFu); }
    inline int promo_piece(Move m) { return int((m >> 16) & 0xFu); }
    inline std::uint32_t move_flags(Move m) { return (m >> 20) & 0xFu; }

    inline Move make_move(int from, int to, int pc, int promo, std::uint32_t flags)
    {
        return Move((from & 0x3F) | ((to & 0x3F) << 6) | ((pc & 0xF) << 12) | ((promo & 0xF) << 16) | ((flags & 0xF) << 20));
    }

    struct MoveList
    {
        Move moves[MAX_MOVES];
        int count = 0;

        void clear() { count = 0; }
        void push(Move m) { moves[count++] = m; }
    };

    enum NodeType : int
    {
        NODE_PV,
        NODE_NON_PV,
        NODE_ROOT
    };

    enum BoundType : std::uint8_t
    {
        BOUND_NONE,
        BOUND_EXACT,
        BOUND_LOWER,
        BOUND_UPPER
    };

    struct SearchLimits
    {
        std::uint64_t time_ms = 0;      // per move
        std::uint64_t time_left_ms = 0; // remaining time in game (no increment)
        int depth = 0;                  // 0 = no depth limit (use time)
    };

    struct TimeManager
    {
        std::uint64_t start_ms = 0;
        std::uint64_t alloc_ms = 0;

        bool is_time_up(std::uint64_t now_ms) const
        {
            return alloc_ms && now_ms - start_ms >= alloc_ms;
        }
    };

} // namespace pf