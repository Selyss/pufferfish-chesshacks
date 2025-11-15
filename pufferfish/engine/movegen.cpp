// Pseudolegal move generation using bitboards.

#include "movegen.h"

namespace pf
{

    static inline bool on_board(int sq) { return sq >= 0 && sq < 64; }

    static inline int rank_of(int sq) { return sq >> 3; }
    static inline int file_of(int sq) { return sq & 7; }

    static void gen_pawn_moves(const Position &pos, MoveList &list)
    {
        Color us = pos.side_to_move;
        Color them = Color(us ^ 1);
        Bitboard pawns = pos.pieceBB[us == WHITE ? W_PAWN : B_PAWN];
        Bitboard empty = ~pos.occupiedBB;

        int dir = (us == WHITE ? 8 : -8);
        int startRank = (us == WHITE ? 1 : 6);
        int promoRank = (us == WHITE ? 6 : 1);

        while (pawns)
        {
            int from = lsb(pawns);
            Bitboard fromBB = Bitboard(1) << from;
            pawns ^= fromBB;

            int r = rank_of(from);

            // Single push
            int to = from + dir;
            if (on_board(to) && (empty & (Bitboard(1) << to)))
            {
                bool promo = (r == promoRank);
                if (promo)
                {
                    int basePromo = (us == WHITE ? W_QUEEN : B_QUEEN);
                    for (int i = 0; i < 4; ++i)
                    {
                        list.push(make_move(from, to, us == WHITE ? W_PAWN : B_PAWN,
                                            basePromo - i, FLAG_PROMOTION));
                    }
                }
                else
                {
                    list.push(make_move(from, to, us == WHITE ? W_PAWN : B_PAWN, 0, FLAG_NONE));
                    // Double push
                    if (r == startRank)
                    {
                        int to2 = from + 2 * dir;
                        if (on_board(to2) && (empty & (Bitboard(1) << to2)))
                        {
                            list.push(make_move(from, to2, us == WHITE ? W_PAWN : B_PAWN, 0, FLAG_NONE));
                        }
                    }
                }
            }

            // Captures
            Bitboard caps = PawnAttacks[us][from] & pos.colorBB[them];
            while (caps)
            {
                int t = lsb(caps);
                caps ^= Bitboard(1) << t;
                bool promo = (rank_of(t) == (us == WHITE ? 7 : 0));
                if (promo)
                {
                    int basePromo = (us == WHITE ? W_QUEEN : B_QUEEN);
                    for (int i = 0; i < 4; ++i)
                    {
                        list.push(make_move(from, t, us == WHITE ? W_PAWN : B_PAWN,
                                            basePromo - i, FLAG_PROMOTION | FLAG_CAPTURE));
                    }
                }
                else
                {
                    list.push(make_move(from, t, us == WHITE ? W_PAWN : B_PAWN, 0, FLAG_CAPTURE));
                }
            }

            // En-passant
            if (pos.ep_square != -1)
            {
                Bitboard epMask = Bitboard(1) << pos.ep_square;
                if (PawnAttacks[us][from] & epMask)
                {
                    list.push(make_move(from, pos.ep_square, us == WHITE ? W_PAWN : B_PAWN, 0, FLAG_ENPASSANT | FLAG_CAPTURE));
                }
            }
        }
    }

    static void gen_knight_moves(const Position &pos, MoveList &list)
    {
        Color us = pos.side_to_move;
        Bitboard knights = pos.pieceBB[us == WHITE ? W_KNIGHT : B_KNIGHT];
        Bitboard ours = pos.colorBB[us];

        while (knights)
        {
            int from = lsb(knights);
            knights ^= Bitboard(1) << from;
            Bitboard moves = KnightAttacks[from] & ~ours;
            while (moves)
            {
                int to = lsb(moves);
                moves ^= Bitboard(1) << to;
                std::uint32_t flags = (pos.occupiedBB & (Bitboard(1) << to)) ? FLAG_CAPTURE : FLAG_NONE;
                list.push(make_move(from, to, us == WHITE ? W_KNIGHT : B_KNIGHT, 0, flags));
            }
        }
    }

    static void gen_king_moves(const Position &pos, MoveList &list)
    {
        Color us = pos.side_to_move;
        Bitboard kingBB = pos.pieceBB[us == WHITE ? W_KING : B_KING];
        Bitboard ours = pos.colorBB[us];
        if (!kingBB)
            return;
        int from = lsb(kingBB);
        Bitboard moves = KingAttacks[from] & ~ours;
        while (moves)
        {
            int to = lsb(moves);
            moves ^= Bitboard(1) << to;
            std::uint32_t flags = (pos.occupiedBB & (Bitboard(1) << to)) ? FLAG_CAPTURE : FLAG_NONE;
            list.push(make_move(from, to, us == WHITE ? W_KING : B_KING, 0, flags));
        }

        // Castling (very basic, assumes castling rights encoded externally)
        // This implementation trusts Position::do_move + legality filter to reject illegal castle through check.
        if (!pos.in_check(us))
        {
            // White: rights bits 0 (K) and 1 (Q); Black: 2 (K) and 3 (Q)
            if (us == WHITE)
            {
                // King side
                if (pos.castling_rights & 0b0001)
                {
                    if (!(pos.occupiedBB & ((Bitboard(1) << 5) | (Bitboard(1) << 6))))
                        list.push(make_move(4, 6, W_KING, 0, FLAG_CASTLING));
                }
                // Queen side
                if (pos.castling_rights & 0b0010)
                {
                    if (!(pos.occupiedBB & ((Bitboard(1) << 1) | (Bitboard(1) << 2) | (Bitboard(1) << 3))))
                        list.push(make_move(4, 2, W_KING, 0, FLAG_CASTLING));
                }
            }
            else
            {
                if (pos.castling_rights & 0b0100)
                {
                    if (!(pos.occupiedBB & ((Bitboard(1) << 61) | (Bitboard(1) << 62))))
                        list.push(make_move(60, 62, B_KING, 0, FLAG_CASTLING));
                }
                if (pos.castling_rights & 0b1000)
                {
                    if (!(pos.occupiedBB & ((Bitboard(1) << 57) | (Bitboard(1) << 58) | (Bitboard(1) << 59))))
                        list.push(make_move(60, 58, B_KING, 0, FLAG_CASTLING));
                }
            }
        }
    }

    static void gen_slider_moves(const Position &pos, MoveList &list, Piece bishop, Piece rook, Piece queen)
    {
        Color us = pos.side_to_move;
        Bitboard ours = pos.colorBB[us];
        Bitboard all = pos.occupiedBB;

        static const int dirsB[4] = {9, 7, -9, -7};
        static const int dirsR[4] = {8, -8, 1, -1};

        auto gen_from = [&](Bitboard pieces, const int *dirs, int dirCount, Piece p)
        {
            while (pieces)
            {
                int from = lsb(pieces);
                pieces ^= Bitboard(1) << from;
                for (int d = 0; d < dirCount; ++d)
                {
                    int s = from;
                    while (true)
                    {
                        int file = file_of(s);
                        int rank = rank_of(s);
                        int ns = s + dirs[d];
                        if (!on_board(ns))
                            break;
                        int nfile = file_of(ns);
                        int nrank = rank_of(ns);
                        if (std::abs(nfile - file) > 1 || std::abs(nrank - rank) > 1)
                            break;
                        s = ns;
                        Bitboard toBB = Bitboard(1) << s;
                        if (ours & toBB)
                            break;
                        std::uint32_t flags = (all & toBB) ? FLAG_CAPTURE : FLAG_NONE;
                        list.push(make_move(from, s, p, 0, flags));
                        if (all & toBB)
                            break;
                    }
                }
            }
        };

        gen_from(pos.pieceBB[bishop], dirsB, 4, bishop);
        gen_from(pos.pieceBB[rook], dirsR, 4, rook);
        gen_from(pos.pieceBB[queen], dirsB, 4, queen);
        gen_from(pos.pieceBB[queen], dirsR, 4, queen);
    }

    void generate_moves(const Position &pos, MoveList &list)
    {
        list.clear();
        gen_pawn_moves(pos, list);
        gen_knight_moves(pos, list);
        gen_slider_moves(pos, list,
                         pos.side_to_move == WHITE ? W_BISHOP : B_BISHOP,
                         pos.side_to_move == WHITE ? W_ROOK : B_ROOK,
                         pos.side_to_move == WHITE ? W_QUEEN : B_QUEEN);
        gen_king_moves(pos, list);
    }

    void generate_captures(const Position &pos, MoveList &list)
    {
        MoveList tmp;
        generate_moves(pos, tmp);
        list.clear();
        for (int i = 0; i < tmp.count; ++i)
        {
            Move m = tmp.moves[i];
            if (move_flags(m) & (FLAG_CAPTURE | FLAG_PROMOTION))
                list.push(m);
        }
    }

    void filter_legal_moves(Position &pos, MoveList &list)
    {
        int out = 0;
        for (int i = 0; i < list.count; ++i)
        {
            Move m = list.moves[i];
            UndoState u;
            pos.do_move(m, u);
            bool ok = !pos.in_check(Color(pos.side_to_move ^ 1));
            pos.undo_move(u);
            if (ok)
                list.moves[out++] = m;
        }
        list.count = out;
    }

} // namespace pf
