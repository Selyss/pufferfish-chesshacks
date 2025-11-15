#include <cassert>
#include <iostream>
#include <vector>

#include "engine/types.h"
#include "engine/bitboard.h"
#include "engine/position.h"
#include "engine/movegen.h"
#include "engine/tt.h"
#include "engine/nn_interface.h"
#include "engine/search.h"

using namespace pf;

static void test_bitboards()
{
    init_bitboards();

    // Knight from b1 (sq=1) -> a3(16), c3(18), d2(11)
    Bitboard n = KnightAttacks[1];
    assert(n & (Bitboard(1) << 16));
    assert(n & (Bitboard(1) << 18));
    assert(n & (Bitboard(1) << 11));

    // King from e4 (sq=28) has 8 neighbors minus edges
    Bitboard k = KingAttacks[28];
    int kcount = popcount(k);
    assert(kcount == 8);

    // Pawn attacks
    // White pawn from e2 (sq=12) attacks d3(19), f3(21)
    assert(PawnAttacks[WHITE][12] & (Bitboard(1) << 19));
    assert(PawnAttacks[WHITE][12] & (Bitboard(1) << 21));

    // popcount/lsb/pop_lsb
    Bitboard bb = (Bitboard(1) << 3) | (Bitboard(1) << 10) | (Bitboard(1) << 40);
    assert(popcount(bb) == 3);
    assert(lsb(bb) == 3);
    Bitboard first = pop_lsb(bb);
    (void)first;
    assert(popcount(bb) == 2);
}

static void test_position_make_unmake()
{
    init_zobrist();
    init_bitboards();

    Position pos;
    pos.set_startpos();

    Key initial = pos.key;

    MoveList ml;
    generate_moves(pos, ml);
    filter_legal_moves(pos, ml);
    assert(ml.count == 20); // start position legal moves

    // Do/undo a handful of legal moves and verify key and state
    int limit = std::min(ml.count, 10);
    for (int i = 0; i < limit; ++i)
    {
        UndoState u;
        pos.do_move(ml.moves[i], u);
        // not crashing implies make works; quick sanity: king not left in check for side that moved
        bool illegal = pos.in_check(Color(pos.side_to_move ^ 1));
        assert(!illegal);
        pos.undo_move(u);
        assert(pos.key == initial);
    }
}

static void test_movegen_startpos()
{
    init_zobrist();
    init_bitboards();

    Position pos;
    pos.set_startpos();

    MoveList ml;
    generate_moves(pos, ml);
    // Pseudolegal at start should be >= 20; exact depends on castles not generated yet
    assert(ml.count >= 20);

    filter_legal_moves(pos, ml);
    assert(ml.count == 20);
}

static void test_tt_probe_store()
{
    TranspositionTable tt;
    tt.resize(4); // MB

    Key key = 0x12345678ABCDEF00ull;
    int depth = 8;
    int score = 123;
    Move best = make_move(12, 20, W_PAWN, 0, FLAG_NONE);

    tt.store(key, depth, score, BOUND_EXACT, best, /*ply=*/0);

    TTEntry out{};
    bool hit = tt.probe(key, depth, -INF_SCORE, INF_SCORE, 0, out);
    assert(hit);
    assert(out.best == best);
    assert(out.bound == BOUND_EXACT);
    assert((int)out.score == score);
}

static void test_search_depth3()
{
    init_zobrist();
    init_bitboards();

    Position pos;
    pos.set_startpos();

    TranspositionTable tt;
    tt.resize(16);
    NNUEEvaluator nn;
    const char *weightPaths[] = {
        "bot/python/nnue_weights.bin",
        "../bot/python/nnue_weights.bin",
        "../../bot/python/nnue_weights.bin",
        "../../../bot/python/nnue_weights.bin",
        "nnue_weights.bin"};
    bool loaded = false;
    for (const char *p : weightPaths)
    {
        if (nn.load(p))
        {
            loaded = true;
            break;
        }
    }
    assert(loaded && "NNUE weights failed to load for tests");

    SearchContext ctx;
    ctx.tt = &tt;
    ctx.nn = &nn;
    ctx.limits.depth = 3;
    ctx.limits.time_ms = 0;

    SearchResult res = search(pos, ctx);
    assert(res.bestMove != MOVE_NONE);
    assert(res.depth >= 3);
}

int main()
{
    test_bitboards();
    test_position_make_unmake();
    test_movegen_startpos();
    test_tt_probe_store();
    test_search_depth3();

    std::cout << "All tests passed\n";
    return 0;
}
