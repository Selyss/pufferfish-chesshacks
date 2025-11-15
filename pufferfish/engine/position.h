// Bitboard-based position representation and move make/unmake.

#pragma once

#include <cstdint>
#include <array>
#include <string>

#include "types.h"
#include "bitboard.h"

namespace pf
{

    struct ZobristTables
    {
        Key piece[PIECE_NB][64];
        Key castling[16];
        Key ep_file[8];
        Key side;
    };

    extern ZobristTables Zobrist;
    void init_zobrist();

    struct UndoState
    {
        Move move;
        Piece captured;
        int castling_rights;
        int ep_square; // -1 if none
        int halfmove_clock;
        Key key;
    };

    struct Position
    {
        Bitboard pieceBB[PIECE_NB];
        Bitboard colorBB[COLOR_NB];
        Bitboard occupiedBB;

        Color side_to_move;
        int castling_rights; // bitmask
        int ep_square;       // 0..63 or -1
        int halfmove_clock;
        int fullmove_number;

        Key key;

        Piece board[64]; // mailbox for convenience

        Position();

        void set_startpos();

        // Load a position from FEN. Returns true on success.
        bool set_fen(const std::string &fen);

        bool is_square_attacked(int sq, Color by) const;

        bool in_check(Color c) const;

        void do_move(Move m, UndoState &u);

        void undo_move(const UndoState &u);
    };

} // namespace pf
