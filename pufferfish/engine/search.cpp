// Alpha-beta search with NN evaluation, TT, LMR, null move, and quiescence.

#include "search.h"

#include <algorithm>
#include <chrono>

namespace pf
{

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
            if (move_flags(m) & FLAG_CAPTURE)
            {
                // Simple MVV-LVA using piece codes
                int victim = pos.board[to_sq(m)];
                int attacker = move_piece(m);
                s += 100000 + (victim * 10 - attacker);
            }
            else
            {
                if (m == ctx.killers[0][ply])
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

    SearchResult search(Position &pos, SearchContext &ctx)
    {
        ctx.stats = SearchStats{};
        std::uint64_t start = now_ms();
        ctx.tm.start_ms = start;
        if (ctx.limits.time_ms)
            ctx.tm.alloc_ms = ctx.limits.time_ms;

        SearchResult result;
        Line rootPV;
        int alpha = -INF_SCORE;
        int beta = INF_SCORE;
        Move bestSoFar = MOVE_NONE;

        int maxDepth = ctx.limits.depth ? ctx.limits.depth : 64;

        for (int depth = 1; depth <= maxDepth; ++depth)
        {
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

            if (ctx.tm.is_time_up(now_ms()))
                break;

            if (pv.len > 0)
            {
                result.bestMove = pv.moves[0];
                bestSoFar = pv.moves[0];
            }
            result.score = score;
            result.depth = depth;
            rootPV = pv;
        }

        (void)rootPV; // could be logged/used for UI
        return result;
    }

    static bool should_abort(const SearchContext &ctx)
    {
        if (!ctx.limits.time_ms)
            return false;
        std::uint64_t now = now_ms();
        return ctx.tm.is_time_up(now);
    }

    static int qsearch(Position &pos, SearchContext &ctx, int alpha, int beta, int ply)
    {
        if (should_abort(ctx))
            return 0;
        ++ctx.stats.qnodes;

        const bool inCheck = pos.in_check(pos.side_to_move);
        int standPat = -INF_SCORE;
        if (!inCheck)
        {
            standPat = ctx.nn->evaluate(pos);
            if (standPat >= beta)
                return standPat;
            if (standPat > alpha)
                alpha = standPat;
        }

        MoveList moves, ordered;
        if (inCheck)
        {
            generate_moves(pos, moves);
            filter_legal_moves(pos, moves);
        }
        else
        {
            generate_captures(pos, moves);
        }
        if (moves.count == 0)
            return inCheck ? -MATE_SCORE + ply : standPat;

        order_moves(pos, ctx, moves, ordered, MOVE_NONE, MOVE_NONE, ply);

        for (int i = 0; i < ordered.count; ++i)
        {
            Move m = ordered.moves[i];
            if (!inCheck)
            {
                // Delta pruning for non-promotion captures.
                const int delta = 900; // queen
                if (standPat + delta < alpha && !(move_flags(m) & FLAG_PROMOTION))
                    continue;
            }

            UndoState u;
            pos.do_move(m, u);
            if (!inCheck && pos.in_check(Color(pos.side_to_move ^ 1)))
            {
                pos.undo_move(u);
                continue;
            }
            int score = -qsearch(pos, ctx, -beta, -alpha, ply + 1);
            pos.undo_move(u);
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
            pos.do_move(nullMove, u); // flip side without moving pieces
            int R = 2 + depth / 4;
            int score = -alphabeta(pos, ctx, depth - R, -beta, -beta + 1, ply + 1, NODE_NON_PV, pv);
            pos.undo_move(u);
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
            pos.do_move(m, u);
            if (pos.in_check(Color(pos.side_to_move ^ 1)))
            {
                pos.undo_move(u);
                continue;
            }
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
