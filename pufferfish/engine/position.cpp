// Position implementation: initialization, Zobrist, attacks, make/unmake.

#include "position.h"

#include <cstring>
#include <random>

namespace pf
{

    ZobristTables Zobrist;

    static Key rand_key(std::mt19937_64 &rng)
    {
        std::uniform_int_distribution<Key> dist;
        return dist(rng);
    }

    void init_zobrist()
    {
        std::mt19937_64 rng(0xC0FFEEULL);
        for (int p = 0; p < PIECE_NB; ++p)
            for (int sq = 0; sq < 64; ++sq)
                Zobrist.piece[p][sq] = rand_key(rng);
        for (int i = 0; i < 16; ++i)
            Zobrist.castling[i] = rand_key(rng);
        for (int f = 0; f < 8; ++f)
            Zobrist.ep_file[f] = rand_key(rng);
        Zobrist.side = rand_key(rng);
    }

    Position::Position()
    {
        std::memset(pieceBB, 0, sizeof(pieceBB));
        std::memset(colorBB, 0, sizeof(colorBB));
        occupiedBB = 0;
        side_to_move = WHITE;
        castling_rights = 0;
        ep_square = -1;
        halfmove_clock = 0;
        fullmove_number = 1;
        key = 0;
        std::memset(board, 0, sizeof(board));
    }

    void Position::set_startpos()
    {
        *this = Position();
        // Simple classic start position.
        auto place = [&](int sq, Piece pc)
        {
            board[sq] = pc;
            pieceBB[pc] |= Bitboard(1) << sq;
            Color c = (pc <= W_KING) ? WHITE : BLACK;
            colorBB[c] |= Bitboard(1) << sq;
            occupiedBB |= Bitboard(1) << sq;
        };

        // Pawns
        for (int f = 0; f < 8; ++f)
        {
            place(8 + f, W_PAWN);
            place(48 + f, B_PAWN);
        }
        // Rooks
        place(0, W_ROOK);
        place(7, W_ROOK);
        place(56, B_ROOK);
        place(63, B_ROOK);
        // Knights
        place(1, W_KNIGHT);
        place(6, W_KNIGHT);
        place(57, B_KNIGHT);
        place(62, B_KNIGHT);
        // Bishops
        place(2, W_BISHOP);
        place(5, W_BISHOP);
        place(58, B_BISHOP);
        place(61, B_BISHOP);
        // Queens
        place(3, W_QUEEN);
        place(59, B_QUEEN);
        // Kings
        place(4, W_KING);
        place(60, B_KING);

        side_to_move = WHITE;
        castling_rights = 0b1111; // all rights
        ep_square = -1;
        halfmove_clock = 0;
        fullmove_number = 1;

        key = 0;
        for (int sq = 0; sq < 64; ++sq)
        {
            Piece pc = board[sq];
            if (pc != NO_PIECE)
                key ^= Zobrist.piece[pc][sq];
        }
        key ^= Zobrist.castling[castling_rights];
    }

    bool Position::is_square_attacked(int sq, Color by) const
    {
        Bitboard target = Bitboard(1) << sq;

        // Pawns
        if (PawnAttacks[by ^ 1][sq] & pieceBB[by == WHITE ? W_PAWN : B_PAWN])
            return true;

        // Knights
        if (KnightAttacks[sq] & (by == WHITE ? pieceBB[W_KNIGHT] : pieceBB[B_KNIGHT]))
            return true;

        // Kings
        if (KingAttacks[sq] & (by == WHITE ? pieceBB[W_KING] : pieceBB[B_KING]))
            return true;

        // Sliding pieces: bishops/queens (diagonals) and rooks/queens (orthogonals).
        Bitboard bishops = (by == WHITE ? pieceBB[W_BISHOP] | pieceBB[W_QUEEN]
                                        : pieceBB[B_BISHOP] | pieceBB[B_QUEEN]);
        Bitboard rooks = (by == WHITE ? pieceBB[W_ROOK] | pieceBB[W_QUEEN]
                                      : pieceBB[B_ROOK] | pieceBB[B_QUEEN]);

        // Simple ray attacks (not magic, but fine for clarity).
        static const int dirsB[4] = {9, 7, -9, -7};
        static const int dirsR[4] = {8, -8, 1, -1};

        auto attacked_by_rays = [&](const Bitboard &set, const int *dirs, int dirCount)
        {
            for (int d = 0; d < dirCount; ++d)
            {
                int s = sq;
                while (true)
                {
                    int file = s & 7;
                    int rank = s >> 3;
                    int ns = s + dirs[d];
                    if (ns < 0 || ns >= 64)
                        break;
                    int nfile = ns & 7;
                    int nrank = ns >> 3;
                    if (std::abs(nfile - file) > 1 || std::abs(nrank - rank) > 1)
                        break;
                    s = ns;
                    Bitboard bb = Bitboard(1) << s;
                    if (bb & occupiedBB)
                    {
                        if (bb & set)
                            return true;
                        break;
                    }
                }
            }
            return false;
        };

        if (attacked_by_rays(bishops, dirsB, 4))
            return true;
        if (attacked_by_rays(rooks, dirsR, 4))
            return true;

        (void)target;
        return false;
    }

    bool Position::in_check(Color c) const
    {
        Bitboard kingBB = pieceBB[c == WHITE ? W_KING : B_KING];
        if (!kingBB)
            return false;
        int sq = lsb(kingBB);
        return is_square_attacked(sq, Color(c ^ 1));
    }

    void Position::do_move(Move m, UndoState &u)
    {
        u.move = m;
        u.castling_rights = castling_rights;
        u.ep_square = ep_square;
        u.halfmove_clock = halfmove_clock;
        u.key = key;
        u.captured = NO_PIECE;

        int from = from_sq(m);
        int to = to_sq(m);
        int pc = move_piece(m);
        int promo = promo_piece(m);
        std::uint32_t flags = move_flags(m);

        Color us = side_to_move;
        Color them = Color(us ^ 1);

        Piece piece = Piece(pc);

        Bitboard fromBB = Bitboard(1) << from;
        Bitboard toBB = Bitboard(1) << to;

        // Remove piece from origin
        pieceBB[piece] ^= fromBB;
        colorBB[us] ^= fromBB;
        occupiedBB ^= fromBB;
        key ^= Zobrist.piece[piece][from];

        // Handle captures (including en-passant)
        if (flags & FLAG_ENPASSANT)
        {
            int capSq = to + (us == WHITE ? -8 : 8);
            Bitboard capBB = Bitboard(1) << capSq;
            Piece capPiece = us == WHITE ? B_PAWN : W_PAWN;
            pieceBB[capPiece] ^= capBB;
            colorBB[them] ^= capBB;
            occupiedBB ^= capBB;
            key ^= Zobrist.piece[capPiece][capSq];
            u.captured = capPiece;
        }
        else if (occupiedBB & toBB)
        {
            // capture on destination
            Piece captured = board[to];
            u.captured = captured;
            pieceBB[captured] ^= toBB;
            colorBB[them] ^= toBB;
            occupiedBB ^= toBB;
            key ^= Zobrist.piece[captured][to];
        }

        // Promotions
        Piece placed = piece;
        if (flags & FLAG_PROMOTION)
        {
            placed = Piece(promo);
        }

        pieceBB[placed] ^= toBB;
        colorBB[us] ^= toBB;
        occupiedBB ^= toBB;
        key ^= Zobrist.piece[placed][to];

        // Castling: move rook
        if (flags & FLAG_CASTLING)
        {
            int rookFrom, rookTo;
            if (to == 6 || to == 62)
            { // king side
                rookFrom = (to == 6 ? 7 : 63);
                rookTo = (to == 6 ? 5 : 61);
            }
            else
            {
                rookFrom = (to == 2 ? 0 : 56);
                rookTo = (to == 2 ? 3 : 59);
            }
            Bitboard rFromBB = Bitboard(1) << rookFrom;
            Bitboard rToBB = Bitboard(1) << rookTo;
            Piece rook = (us == WHITE ? W_ROOK : B_ROOK);
            pieceBB[rook] ^= (rFromBB | rToBB);
            colorBB[us] ^= (rFromBB | rToBB);
            occupiedBB ^= (rFromBB | rToBB);
            key ^= Zobrist.piece[rook][rookFrom];
            key ^= Zobrist.piece[rook][rookTo];
            board[rookFrom] = NO_PIECE;
            board[rookTo] = rook;
        }

        // Update castling rights (simplified: clear when king or rook moves or rook captured).
        // Here we just clear all rights for the moving side when king moves, or side's corresponding rook moves.
        key ^= Zobrist.castling[castling_rights];
        if (piece == (us == WHITE ? W_KING : B_KING))
        {
            if (us == WHITE)
                castling_rights &= ~0b0011;
            else
                castling_rights &= ~0b1100;
        }
        key ^= Zobrist.castling[castling_rights];

        // Update ep square and halfmove clock
        if (ep_square != -1)
            key ^= Zobrist.ep_file[ep_square & 7];
        ep_square = -1;
        if (piece == W_PAWN || piece == B_PAWN || (flags & FLAG_CAPTURE))
            halfmove_clock = 0;
        else
            ++halfmove_clock;

        // Double pawn push creates ep square
        if (piece == W_PAWN || piece == B_PAWN)
        {
            int diff = to - from;
            if (diff == 16 || diff == -16)
            {
                ep_square = (from + to) / 2;
                key ^= Zobrist.ep_file[ep_square & 7];
            }
        }

        board[from] = NO_PIECE;
        board[to] = placed;

        side_to_move = Color(side_to_move ^ 1);
        key ^= Zobrist.side;
        if (side_to_move == WHITE)
            ++fullmove_number;
    }

    void Position::undo_move(const UndoState &u)
    {
        Move m = u.move;
        int from = from_sq(m);
        int to = to_sq(m);
        int pc = move_piece(m);
        int promo = promo_piece(m);
        std::uint32_t flags = move_flags(m);

        side_to_move = Color(side_to_move ^ 1);

        Color us = side_to_move;
        Color them = Color(us ^ 1);

        Piece piece = Piece(pc);
        Piece placed = (flags & FLAG_PROMOTION) ? Piece(promo) : piece;

        Bitboard fromBB = Bitboard(1) << from;
        Bitboard toBB = Bitboard(1) << to;

        // Restore from Zobrist/state snapshot; it's the most reliable
        castling_rights = u.castling_rights;
        ep_square = u.ep_square;
        halfmove_clock = u.halfmove_clock;
        key = u.key;

        // Clear destination piece
        pieceBB[placed] ^= toBB;
        colorBB[us] ^= toBB;
        occupiedBB ^= toBB;

        // Restore moving piece to origin
        pieceBB[piece] ^= fromBB;
        colorBB[us] ^= fromBB;
        occupiedBB ^= fromBB;

        board[to] = NO_PIECE;
        board[from] = piece;

        // Restore captured piece
        if (flags & FLAG_ENPASSANT)
        {
            int capSq = to + (us == WHITE ? -8 : 8);
            Bitboard capBB = Bitboard(1) << capSq;
            Piece capPiece = u.captured;
            if (capPiece != NO_PIECE)
            {
                pieceBB[capPiece] ^= capBB;
                colorBB[them] ^= capBB;
                occupiedBB ^= capBB;
                board[capSq] = capPiece;
            }
        }
        else if (u.captured != NO_PIECE)
        {
            Piece capPiece = u.captured;
            pieceBB[capPiece] ^= toBB;
            colorBB[them] ^= toBB;
            occupiedBB ^= toBB;
            board[to] = capPiece;
        }
    }

} // namespace pf
