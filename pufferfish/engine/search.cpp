// Alpha-beta search with NN evaluation, TT, LMR, null move, and quiescence.

#include "search.h"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace pf
{
    static inline int piece_mvv_value(int piece)
    {
        // Use conventional MVV values; indices are pf::Piece enums.
        static const int v[PIECE_NB] = {
            0,                             // NO_PIECE
            100,                           // W_PAWN
            320,                           // W_KNIGHT
            330,                           // W_BISHOP
            500,                           // W_ROOK
            900,                           // W_QUEEN
            20000,                         // W_KING (avoid preferring captures of king; large for ordering only)
            100, 320, 330, 500, 900, 20000 // black pieces
        };
        if (piece < 0 || piece >= PIECE_NB)
            return 0;
        return v[piece];
    }

    // ---------------------------------------------------------------------
    // Static exchange evaluation
    //
    // Move ordering used MVV-LVA alone, which only looks at what is captured and
    // with what. It cannot tell a free pawn from a pawn that is defended three
    // times, so QxP onto a guarded square was ordered as though it won material,
    // and quiescence then searched the whole losing sequence.
    //
    // SEE plays the capture sequence out statically, each side recapturing with
    // its least valuable attacker, and returns the material balance.
    // ---------------------------------------------------------------------

    static inline int see_value(int piece)
    {
        static const int v[PIECE_NB] = {
            0, 100, 320, 330, 500, 900, 20000,
            100, 320, 330, 500, 900, 20000};
        return (piece <= 0 || piece >= PIECE_NB) ? 0 : v[piece];
    }

    // First blocker along each ray from `sq` under occupancy `occ`. Recomputing
    // this after every capture is what makes x-rayed attackers appear.
    static Bitboard ray_blockers(int sq, Bitboard occ, bool diagonal)
    {
        static const int dB[4] = {9, 7, -9, -7};
        static const int dR[4] = {8, -8, 1, -1};
        const int *dirs = diagonal ? dB : dR;
        Bitboard res = 0;
        for (int d = 0; d < 4; ++d)
        {
            int s = sq;
            while (true)
            {
                const int file = s & 7, rank = s >> 3;
                const int ns = s + dirs[d];
                if (ns < 0 || ns >= 64)
                    break;
                if (std::abs((ns & 7) - file) > 1 || std::abs((ns >> 3) - rank) > 1)
                    break;
                s = ns;
                const Bitboard bb = Bitboard(1) << s;
                if (occ & bb)
                {
                    res |= bb;
                    break;
                }
            }
        }
        return res;
    }

    static Bitboard attackers_to(const Position &pos, int sq, Bitboard occ)
    {
        Bitboard a = 0;
        a |= PawnAttacks[BLACK][sq] & pos.pieceBB[W_PAWN];
        a |= PawnAttacks[WHITE][sq] & pos.pieceBB[B_PAWN];
        a |= KnightAttacks[sq] & (pos.pieceBB[W_KNIGHT] | pos.pieceBB[B_KNIGHT]);
        a |= KingAttacks[sq] & (pos.pieceBB[W_KING] | pos.pieceBB[B_KING]);
        a |= ray_blockers(sq, occ, true) &
             (pos.pieceBB[W_BISHOP] | pos.pieceBB[W_QUEEN] |
              pos.pieceBB[B_BISHOP] | pos.pieceBB[B_QUEEN]);
        a |= ray_blockers(sq, occ, false) &
             (pos.pieceBB[W_ROOK] | pos.pieceBB[W_QUEEN] |
              pos.pieceBB[B_ROOK] | pos.pieceBB[B_QUEEN]);
        return a & occ;
    }

    // Material won or lost by the capture sequence starting with `m`, in
    // centipawns, from the perspective of the side making it.
    static int see(const Position &pos, Move m)
    {
        const int to = to_sq(m);
        const int from = from_sq(m);
        const std::uint32_t flags = move_flags(m);

        int gain[32];
        int d = 0;
        gain[0] = (flags & FLAG_ENPASSANT) ? see_value(W_PAWN)
                                           : see_value(pos.board[to]);

        Bitboard occ = pos.occupiedBB ^ (Bitboard(1) << from);
        if (flags & FLAG_ENPASSANT)
            occ ^= Bitboard(1) << (to + (pos.side_to_move == WHITE ? -8 : 8));

        int attackerPiece = move_piece(m);
        Color side = Color(pos.side_to_move ^ 1);
        Bitboard attackers = attackers_to(pos, to, occ);

        while (d < 31)
        {
            ++d;
            gain[d] = see_value(attackerPiece) - gain[d - 1];
            if (std::max(-gain[d - 1], gain[d]) < 0)
                break; // already decided, whoever is to move will stop here

            const Bitboard mine = attackers & occ & pos.colorBB[side];
            if (!mine)
                break;

            // Recapture with the least valuable attacker available.
            Bitboard chosen = 0;
            const int base = (side == WHITE) ? W_PAWN : B_PAWN;
            for (int pt = 0; pt < 6; ++pt)
            {
                const Bitboard s = mine & pos.pieceBB[base + pt];
                if (s)
                {
                    chosen = s & (~s + 1); // lowest set bit
                    attackerPiece = base + pt;
                    break;
                }
            }
            if (!chosen)
                break;

            occ ^= chosen;
            attackers = attackers_to(pos, to, occ); // picks up x-rays
            side = Color(side ^ 1);
        }

        while (--d > 0)
            gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
        return gain[0];
    }

    int see_probe(const Position &pos, Move m) { return see(pos, m); }

    static std::uint64_t now_ms()
    {
        using namespace std::chrono;
        return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
    }

    struct Line
    {
        Move moves[MAX_PLY];
        int len = 0;
    };

    static int qsearch(Position &pos, SearchContext &ctx, int alpha, int beta, int ply);
    static int alphabeta(Position &pos, SearchContext &ctx, int depth, int alpha, int beta, int ply, NodeType nodeType, Line &pv);

    static void order_moves(const Position &pos, SearchContext &ctx, const MoveList &raw, MoveList &ordered, Move ttMove, Move prevBest, int ply)
    {
        struct Scored
        {
            Move m;
            int score;
        } buf[MAX_MOVES];

        int count = raw.count;
        for (int i = 0; i < count; ++i)
        {
            Move m = raw.moves[i];
            int s = 0;
            if (m == ttMove)
                s += 1000000;
            if (m == prevBest)
                s += 900000;
            const std::uint32_t flags = move_flags(m);
            const bool isCapture = (flags & FLAG_CAPTURE) != 0;
            const bool isPromotion = (flags & FLAG_PROMOTION) != 0;
            if (isCapture)
            {
                // MVV-LVA orders among captures, but cannot tell a winning capture
                // from a losing one. SEE splits them: captures that lose material
                // are pushed below every quiet move instead of being searched first.
                int victim = pos.board[to_sq(m)];
                int attacker = move_piece(m);
                const int mvvlva = piece_mvv_value(victim) - (piece_mvv_value(attacker) / 10);
                const int exchange = see(pos, m);
                if (exchange >= 0)
                    s += 200000 + mvvlva;
                else
                    s += -200000 + exchange;
                if (isPromotion)
                    s += 5000; // promote-capture is even better
            }
            else
            {
                if (isPromotion)
                {
                    s += 120000; // non-capture promotions are high priority
                }
                else if (m == ctx.killers[0][ply])
                    s += 80000;
                else if (m == ctx.killers[1][ply])
                    s += 70000;
                else
                    s += ctx.history[move_piece(m)][to_sq(m)];
            }
            buf[i] = {m, s};
        }

        std::sort(buf, buf + count, [](const Scored &a, const Scored &b)
                  { return a.score > b.score; });
        ordered.clear();
        for (int i = 0; i < count; ++i)
            ordered.push(buf[i].m);
    }

    static int estimate_moves_to_go(const Position &pos)
    {
        // Simple phase-based MTG: opening 28, middlegame 20, endgame 12
        auto cnt = [&](Piece p)
        { return popcount(pos.pieceBB[p]); };
        int nonPawnMaterial = (cnt(W_KNIGHT) + cnt(B_KNIGHT)) * 3 + (cnt(W_BISHOP) + cnt(B_BISHOP)) * 3 +
                              (cnt(W_ROOK) + cnt(B_ROOK)) * 5 + (cnt(W_QUEEN) + cnt(B_QUEEN)) * 9;
        int mtg = 20;
        if (nonPawnMaterial >= 40)
            mtg = 28; // opening
        else if (nonPawnMaterial <= 16)
            mtg = 12; // endgame
        return mtg;
    }

    SearchResult search(Position &pos, SearchContext &ctx)
    {
        ctx.stats = SearchStats{};
        std::uint64_t start = now_ms();
        ctx.tm.start_ms = start;
        if (ctx.limits.time_ms)
        {
            ctx.tm.alloc_ms = ctx.limits.time_ms;
        }
        else if (ctx.limits.time_left_ms)
        {
            std::uint64_t T = ctx.limits.time_left_ms;
            int mtg = estimate_moves_to_go(pos);
            // Keep a reserve: 12% of remaining time, capped at 4000 ms
            std::uint64_t reserve = std::min<std::uint64_t>(T / 8, 4000);
            std::uint64_t usable = (T > reserve) ? (T - reserve) : (T * 7 / 8);
            // Base allocation: usable / (mtg + 1) for safety
            std::uint64_t base = usable / (std::uint64_t)(mtg + 1);
            // Clamp between [10 ms, 18% of T]
            std::uint64_t hardMax = (T * 18) / 100;
            std::uint64_t alloc = std::max<std::uint64_t>(10, std::min<std::uint64_t>(base, hardMax));
            ctx.tm.alloc_ms = alloc;
            std::cerr << "info tm time_left_ms " << T << " mtg " << mtg
                      << " reserve_ms " << reserve << " alloc_ms " << alloc << std::endl;
        }

        // Seed the incremental accumulator from the root position. Every move the
        // search makes from here updates it by delta.
        if (ctx.nn)
            ctx.nn->acc_reset(pos);

        // The caller supplies the keys of the game so far; the root itself is part
        // of the path too, so push it here and drop it again on the way out.
        ctx.repetitionKeys.push_back(pos.key);
        const std::size_t repBase = ctx.repetitionKeys.size();

        SearchResult result;
        Line rootPV;
        int alpha = -INF_SCORE;
        int beta = INF_SCORE;
        Move bestSoFar = MOVE_NONE;

        int maxDepth = ctx.limits.depth ? ctx.limits.depth : 64;

        for (int depth = 1; depth <= maxDepth; ++depth)
        {
            // Light history decay each iteration to keep values bounded
            for (int p = 0; p < PIECE_NB; ++p)
                for (int sq = 0; sq < 64; ++sq)
                    ctx.history[p][sq] -= (ctx.history[p][sq] >> 3);

            Line pv;
            int window = 20; // aspiration window in cp
            int scoreLo = alpha;
            int scoreHi = beta;

            if (depth > 1 && result.score > -INF_SCORE && result.score < INF_SCORE)
            {
                scoreLo = result.score - window;
                scoreHi = result.score + window;
            }

            int score;
            while (true)
            {
                pv.len = 0;
                score = alphabeta(pos, ctx, depth, scoreLo, scoreHi, 0, NODE_ROOT, pv);
                if (score <= scoreLo && score > -INF_SCORE)
                {
                    scoreLo = -INF_SCORE;
                    continue;
                }
                if (score >= scoreHi && score < INF_SCORE)
                {
                    scoreHi = INF_SCORE;
                    continue;
                }
                break;
            }

            const bool timeUp = ctx.tm.is_time_up(now_ms());

            // Record the iteration before honouring the time check. Breaking first
            // meant that running out of time during depth 1 returned MOVE_NONE and
            // the engine emitted "bestmove 0000", forfeiting the game. A partial
            // iteration is trusted only when there is nothing better to fall back on.
            if (pv.len > 0 && (!timeUp || result.bestMove == MOVE_NONE))
            {
                result.bestMove = pv.moves[0];
                bestSoFar = pv.moves[0];
                result.score = score;
                result.depth = depth;
                rootPV = pv;
            }

            if (timeUp)
                break;
        }

        // Last resort: never return "no move" for a position that has one.
        if (result.bestMove == MOVE_NONE)
        {
            MoveList ml;
            generate_moves(pos, ml);
            filter_legal_moves(pos, ml);
            if (ml.count > 0)
            {
                result.bestMove = ml.moves[0];
                result.depth = 1;
            }
        }

        // Unwind the root key (and anything a timed-out iteration left behind).
        if (ctx.repetitionKeys.size() >= repBase)
            ctx.repetitionKeys.resize(repBase - 1);

        (void)rootPV; // could be logged/used for UI
        return result;
    }

    static bool should_abort(const SearchContext &ctx)
    {
        if (ctx.tm.alloc_ms == 0)
            return false;
        std::uint64_t now = now_ms();
        return ctx.tm.is_time_up(now);
    }

    static int qsearch(Position &pos, SearchContext &ctx, int alpha, int beta, int ply)
    {
        if (should_abort(ctx))
            return 0;
        ++ctx.stats.qnodes;

        int standPat = ctx.nn->evaluate(pos);
        if (standPat >= beta)
            return standPat;
        if (standPat > alpha)
            alpha = standPat;

        MoveList moves, ordered;
        generate_captures(pos, moves);
        if (moves.count == 0)
            return standPat;

        order_moves(pos, ctx, moves, ordered, MOVE_NONE, MOVE_NONE, ply);

        for (int i = 0; i < ordered.count; ++i)
        {
            Move m = ordered.moves[i];
            // Delta pruning: if even capturing the most valuable piece cannot raise alpha, skip.
            // Here we use a simple constant margin.
            const int delta = 900; // queen
            if (standPat + delta < alpha && !(move_flags(m) & FLAG_PROMOTION))
                continue;

            // Skip captures that lose material outright. Quiescence exists to
            // resolve exchanges, and searching a sequence the mover would never
            // choose only inflates the tree.
            if ((move_flags(m) & FLAG_CAPTURE) && !(move_flags(m) & FLAG_PROMOTION) &&
                see(pos, m) < 0)
                continue;

            UndoState u;
            // acc_begin_move only snapshots, so the illegal-move path below needs
            // no accumulator call: nothing has been applied yet.
            ctx.nn->acc_begin_move(pos);
            pos.do_move(m, u);
            if (pos.in_check(Color(pos.side_to_move ^ 1)))
            {
                pos.undo_move(u);
                continue;
            }
            ctx.nn->acc_end_move(pos);
            ctx.repetitionKeys.push_back(pos.key);
            int score = -qsearch(pos, ctx, -beta, -alpha, ply + 1);
            pos.undo_move(u);
            ctx.nn->acc_unmake();
            ctx.repetitionKeys.pop_back();
            if (score >= beta)
                return score;
            if (score > alpha)
                alpha = score;
        }
        return alpha;
    }

    static int alphabeta(Position &pos, SearchContext &ctx, int depth, int alpha, int beta, int ply, NodeType nodeType, Line &pv)
    {
        if (ply >= MAX_PLY - 1)
            return ctx.nn->evaluate(pos);

        if (should_abort(ctx))
            return 0;

        // Draw by repetition or by the fifty-move rule.
        //
        // Checked before the transposition probe on purpose: both depend on the
        // path taken to reach this position rather than on the position itself, so
        // the result must not be read from or written to the table. Skipped at the
        // root, where a score is no substitute for a move.
        if (ply > 0)
        {
            if (pos.halfmove_clock >= 100 ||
                ctx.is_repetition(pos.key, pos.halfmove_clock))
                return DRAW_SCORE;
        }

        bool inCheck = pos.in_check(pos.side_to_move);
        if (inCheck)
            ++depth; // check extension

        if (depth <= 0)
            return qsearch(pos, ctx, alpha, beta, ply);

        ++ctx.stats.nodes;

        int alphaOrig = alpha;
        TTEntry tte;
        Move ttMove = MOVE_NONE;
        if (ctx.tt && ctx.tt->probe(pos.key, depth, alpha, beta, ply, tte))
        {
            ttMove = tte.best;
            int tscore = tte.score;
            if (tte.bound == BOUND_EXACT)
                return tscore;
            if (tte.bound == BOUND_LOWER && tscore > alpha)
                alpha = tscore;
            else if (tte.bound == BOUND_UPPER && tscore < beta)
                beta = tscore;
            if (alpha >= beta)
                return tscore;
        }

        // Null move pruning
        if (!inCheck && depth >= 3 && nodeType != NODE_ROOT)
        {
            UndoState u;
            Move nullMove = MOVE_NONE;
            // No accumulator hooks here on purpose. A null move leaves every piece
            // bitboard untouched, and the feature encoding is colour-absolute with
            // no side-to-move input, so the accumulator is unchanged by definition.
            pos.do_move(nullMove, u); // flip side without moving pieces
            ctx.repetitionKeys.push_back(pos.key);
            int R = 2 + depth / 4;
            int score = -alphabeta(pos, ctx, depth - R, -beta, -beta + 1, ply + 1, NODE_NON_PV, pv);
            pos.undo_move(u);
            ctx.repetitionKeys.pop_back();
            if (score >= beta)
                return score;
        }

        MoveList moves, ordered;
        generate_moves(pos, moves);
        if (moves.count == 0)
        {
            if (inCheck)
                return -MATE_SCORE + ply;
            return DRAW_SCORE;
        }

        Move prevBest = ttMove;
        order_moves(pos, ctx, moves, ordered, ttMove, prevBest, ply);

        Line childPV;
        int bestScore = -INF_SCORE;
        Move bestMove = MOVE_NONE;
        int legalMoves = 0;

        for (int i = 0; i < ordered.count; ++i)
        {
            Move m = ordered.moves[i];
            UndoState u;
            ctx.nn->acc_begin_move(pos);
            pos.do_move(m, u);
            if (pos.in_check(Color(pos.side_to_move ^ 1)))
            {
                pos.undo_move(u);
                continue;
            }
            ctx.nn->acc_end_move(pos);
            ctx.repetitionKeys.push_back(pos.key);
            ++legalMoves;

            int newDepth = depth - 1;
            int score;
            NodeType childType = nodeType == NODE_PV && legalMoves == 1 ? NODE_PV : NODE_NON_PV;

            // Late move reductions
            int R = 0;
            if (childType == NODE_NON_PV && depth >= 3 && legalMoves > 3 && !(move_flags(m) & FLAG_CAPTURE))
                R = 1;

            if (childType == NODE_PV)
            {
                score = -alphabeta(pos, ctx, newDepth, -beta, -alpha, ply + 1, childType, childPV);
            }
            else
            {
                score = -alphabeta(pos, ctx, newDepth - R, -alpha - 1, -alpha, ply + 1, childType, childPV);
                if (score > alpha && R > 0)
                    score = -alphabeta(pos, ctx, newDepth, -alpha - 1, -alpha, ply + 1, childType, childPV);
                if (score > alpha && score < beta)
                    score = -alphabeta(pos, ctx, newDepth, -beta, -alpha, ply + 1, childType, childPV);
            }

            pos.undo_move(u);
            ctx.nn->acc_unmake();
            ctx.repetitionKeys.pop_back();

            if (score > bestScore)
            {
                bestScore = score;
                bestMove = m;
                if (score > alpha)
                {
                    alpha = score;
                    pv.len = 0;
                    pv.moves[0] = m;
                    pv.len = 1;
                    for (int j = 0; j < childPV.len && j + 1 < MAX_PLY; ++j)
                        pv.moves[j + 1] = childPV.moves[j];
                    pv.len += childPV.len;
                }
            }

            if (alpha >= beta)
            {
                // Store killer and history for quiet moves
                if (!(move_flags(m) & FLAG_CAPTURE))
                {
                    if (ctx.killers[0][ply] != m)
                    {
                        ctx.killers[1][ply] = ctx.killers[0][ply];
                        ctx.killers[0][ply] = m;
                    }
                    ctx.history[move_piece(m)][to_sq(m)] += depth * depth;
                }
                break;
            }
        }

        // No legal move: this node is checkmate or stalemate.
        //
        // generate_moves() is pseudo-legal, so the count check before the loop does
        // not fire here -- a mated or stalemated position still produces moves, they
        // are just all illegal. Without this, bestScore kept its -INF_SCORE seed and
        // was returned as a real score, which the parent negated to +INF_SCORE
        // (32000). That is larger than MATE_SCORE (31000), so stalemating the
        // opponent scored better than checkmating them, and the engine drew won
        // games on purpose: it stalemated K+Q vs K in 6 of 6 test positions.
        if (legalMoves == 0)
            return inCheck ? (-MATE_SCORE + ply) : DRAW_SCORE;

        BoundType bound = BOUND_EXACT;
        if (bestScore <= alphaOrig)
            bound = BOUND_UPPER;
        else if (bestScore >= beta)
            bound = BOUND_LOWER;

        if (ctx.tt)
            ctx.tt->store(pos.key, depth, bestScore, bound, bestMove, ply);

        return bestScore;
    }

} // namespace pf