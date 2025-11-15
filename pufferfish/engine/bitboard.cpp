// Bitboard attack table initialization.

#include "bitboard.h"

namespace pf
{

    Bitboard KnightAttacks[64];
    Bitboard KingAttacks[64];
    Bitboard PawnAttacks[COLOR_NB][64];
    Bitboard BishopMasks[64];
    Bitboard RookMasks[64];

    static bool on_board(int sq) { return sq >= 0 && sq < 64; }
    static int file_of(int sq) { return sq & 7; }
    static int rank_of(int sq) { return sq >> 3; }

    void init_bitboards()
    {
        for (int sq = 0; sq < 64; ++sq)
        {
            Bitboard bb = Bitboard(1) << sq;

            // King attacks
            Bitboard king = 0;
            for (int dr = -1; dr <= 1; ++dr)
                for (int df = -1; df <= 1; ++df)
                {
                    if (!dr && !df)
                        continue;
                    int r = rank_of(sq) + dr;
                    int f = file_of(sq) + df;
                    if (r >= 0 && r < 8 && f >= 0 && f < 8)
                        king |= Bitboard(1) << (r * 8 + f);
                }
            KingAttacks[sq] = king;

            // Knight attacks
            Bitboard knight = 0;
            static const int kdr[8] = {2, 1, -1, -2, -2, -1, 1, 2};
            static const int kdf[8] = {1, 2, 2, 1, -1, -2, -2, -1};
            for (int i = 0; i < 8; ++i)
            {
                int r = rank_of(sq) + kdr[i];
                int f = file_of(sq) + kdf[i];
                if (r >= 0 && r < 8 && f >= 0 && f < 8)
                    knight |= Bitboard(1) << (r * 8 + f);
            }
            KnightAttacks[sq] = knight;

            // Pawn attacks
            Bitboard w = 0, b = 0;
            int r = rank_of(sq);
            int f = file_of(sq);
            if (r + 1 < 8)
            {
                if (f - 1 >= 0)
                    w |= Bitboard(1) << ((r + 1) * 8 + (f - 1));
                if (f + 1 < 8)
                    w |= Bitboard(1) << ((r + 1) * 8 + (f + 1));
            }
            if (r - 1 >= 0)
            {
                if (f - 1 >= 0)
                    b |= Bitboard(1) << ((r - 1) * 8 + (f - 1));
                if (f + 1 < 8)
                    b |= Bitboard(1) << ((r - 1) * 8 + (f + 1));
            }
            PawnAttacks[WHITE][sq] = w;
            PawnAttacks[BLACK][sq] = b;

            // Sliding masks (excluding edges)
            Bitboard bishop = 0, rook = 0;
            // Diagonals
            static const int dirsB[4] = {9, 7, -9, -7};
            for (int d = 0; d < 4; ++d)
            {
                int s = sq;
                while (true)
                {
                    int file = file_of(s);
                    int rank = rank_of(s);
                    int ns = s + dirsB[d];
                    if (!on_board(ns))
                        break;
                    int f2 = file_of(ns);
                    int r2 = rank_of(ns);
                    if (std::abs(f2 - file) > 1 || std::abs(r2 - rank) > 1)
                        break;
                    s = ns;
                    bishop |= Bitboard(1) << s;
                }
            }
            BishopMasks[sq] = bishop;

            static const int dirsR[4] = {8, -8, 1, -1};
            for (int d = 0; d < 4; ++d)
            {
                int s = sq;
                while (true)
                {
                    int file = file_of(s);
                    int rank = rank_of(s);
                    int ns = s + dirsR[d];
                    if (!on_board(ns))
                        break;
                    int f2 = file_of(ns);
                    int r2 = rank_of(ns);
                    if (std::abs(f2 - file) > 1 || std::abs(r2 - rank) > 1)
                        break;
                    s = ns;
                    rook |= Bitboard(1) << s;
                }
            }
            RookMasks[sq] = rook;

            (void)bb;
        }
    }

} // namespace pf
