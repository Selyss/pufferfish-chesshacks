// Bitboard utilities and attack tables.

#pragma once

#include "types.h"

namespace pf
{

    extern Bitboard KnightAttacks[64];
    extern Bitboard KingAttacks[64];
    extern Bitboard PawnAttacks[COLOR_NB][64];

    extern Bitboard BishopMasks[64];
    extern Bitboard RookMasks[64];

    void init_bitboards();

    inline int popcount(Bitboard b)
    {
#if defined(_MSC_VER)
        return (int)__popcnt64(b);
#else
        return (int)__builtin_popcountll(b);
#endif
    }

    inline int lsb(Bitboard b)
    {
#if defined(_MSC_VER)
        unsigned long idx;
        _BitScanForward64(&idx, b);
        return (int)idx;
#else
        return (int)__builtin_ctzll(b);
#endif
    }

    inline Bitboard pop_lsb(Bitboard &b)
    {
        Bitboard l = b & -b;
        b ^= l;
        return l;
    }

    inline Bitboard north(Bitboard b) { return b << 8; }
    inline Bitboard south(Bitboard b) { return b >> 8; }

} // namespace pf
