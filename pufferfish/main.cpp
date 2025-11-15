// Minimal runner: initialize engine, run a short search from startpos.

#include <iostream>

#include "engine/types.h"
#include "engine/bitboard.h"
#include "engine/position.h"
#include "engine/movegen.h"
#include "engine/tt.h"
// #include "engine/nn_interface.h"
// #include "engine/search.h"

using namespace pf;

// struct DummyNN : NNEvaluator
// {
//     int evaluate(const Position &pos) override
//     {
//         // Very crude material-only eval as a stand-in for NN.
//         static const int val[PIECE_NB] = {
//             0,
//             100, 320, 330, 500, 900, 0,
//             -100, -320, -330, -500, -900, 0};
//         int score = 0;
//         for (int sq = 0; sq < 64; ++sq)
//             score += val[pos.board[sq]];
//         return (pos.side_to_move == WHITE ? score : -score);
//     }
// };

int main()
{
    init_zobrist();
    init_bitboards();

    Position pos;
    pos.set_startpos();

    TranspositionTable tt;
    tt.resize(64); // 64 MB

    NNUEEvaluator nn;
    nn.load("nnue_weights.bin");

    SearchContext ctx;
    ctx.tt = &tt;
    ctx.nn = &nn;
    ctx.limits.depth = 5;
    ctx.limits.time_ms = 0; // depth-limited

    SearchResult res = search(pos, ctx);

    std::cout << "bestmove from startpos: from " << from_sq(res.bestMove)
              << " to " << to_sq(res.bestMove) << " score " << res.score
              << " depth " << res.depth << " nodes " << ctx.stats.nodes
              << " qnodes " << ctx.stats.qnodes << "\n";

    return 0;
}
