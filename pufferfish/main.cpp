// Minimal runner: initialize engine, run a short search from startpos.

#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <chrono>

#include "engine/types.h"
#include "engine/bitboard.h"
#include "engine/position.h"
#include "engine/movegen.h"
#include "engine/tt.h"
#include "engine/nn_interface.h"
#include "engine/simple_nnue.h"
#include "engine/search.h"

using namespace pf;

// Material fallback removed: NNUE is required.

static std::string sq_to_str(int sq)
{
    const char files[] = "abcdefgh";
    std::string s;
    s += files[sq & 7];
    s += char('1' + (sq >> 3));
    return s;
}

static char promo_char_from_piece(int promoPiece)
{
    int typeIdx = 0;
    if (promoPiece >= W_PAWN && promoPiece <= W_KING)
        typeIdx = promoPiece - W_PAWN;
    else if (promoPiece >= B_PAWN && promoPiece <= B_KING)
        typeIdx = promoPiece - B_PAWN;
    else
        return '\0';
    switch (typeIdx)
    {
    case KNIGHT:
        return 'n';
    case BISHOP:
        return 'b';
    case ROOK:
        return 'r';
    case QUEEN:
        return 'q';
    default:
        return '\0';
    }
}

static std::string move_to_uci(Move m)
{
    int from = from_sq(m);
    int to = to_sq(m);
    std::string uci = sq_to_str(from) + sq_to_str(to);
    if (move_flags(m) & FLAG_PROMOTION)
    {
        char pc = promo_char_from_piece(promo_piece(m));
        if (pc)
            uci += pc;
    }
    return uci;
}

static char piece_letter(Piece p)
{
    switch (p)
    {
    case W_KNIGHT:
    case B_KNIGHT:
        return 'N';
    case W_BISHOP:
    case B_BISHOP:
        return 'B';
    case W_ROOK:
    case B_ROOK:
        return 'R';
    case W_QUEEN:
    case B_QUEEN:
        return 'Q';
    case W_KING:
    case B_KING:
        return 'K';
    default:
        return '\0';
    }
}

// Produce simple SAN for a legal move in the given position.
static std::string move_to_san(Position &pos, Move m)
{
    if (m == MOVE_NONE)
        return "--";
    std::uint32_t flags = move_flags(m);
    int from = from_sq(m);
    int to = to_sq(m);
    Piece p = Piece(move_piece(m));

    // Castling
    if (flags & FLAG_CASTLING)
    {
        bool kingSide = to > from;
        std::string san = kingSide ? "O-O" : "O-O-O";
        UndoState u;
        pos.do_move(m, u);
        bool check = pos.in_check(pos.side_to_move);
        MoveList replies;
        generate_moves(pos, replies);
        filter_legal_moves(pos, replies);
        bool mate = check && replies.count == 0;
        pos.undo_move(u);
        if (mate)
            san += '#';
        else if (check)
            san += '+';
        return san;
    }

    std::string san;
    char pieceChar = piece_letter(p);
    bool isPawn = (pieceChar == '\0');
    bool isCapture = (flags & FLAG_CAPTURE) || (flags & FLAG_ENPASSANT);

    if (!isPawn)
    {
        // Disambiguation: find other same-type pieces that can reach 'to'.
        Position tmp = pos; // copy
        MoveList gen;
        generate_moves(tmp, gen);
        filter_legal_moves(tmp, gen);
        bool needFile = false, needRank = false;
        int fromFile = from & 7;
        int fromRank = from >> 3;
        for (int i = 0; i < gen.count; ++i)
        {
            Move om = gen.moves[i];
            if (om == m)
                continue;
            if (to_sq(om) == to && move_piece(om) == move_piece(m))
            {
                int of = from_sq(om) & 7;
                int orank = from_sq(om) >> 3;
                if (of == fromFile)
                    needRank = true;
                if (orank == fromRank)
                    needFile = true;
                if (needFile && needRank)
                    break;
            }
        }
        san += pieceChar;
        if (needFile)
            san += char('a' + fromFile);
        if (needRank)
            san += char('1' + fromRank);
    }
    else if (isCapture)
    {
        // Pawn capture includes source file
        san += char('a' + (from & 7));
    }

    if (isCapture)
        san += 'x';
    san += sq_to_str(to);

    if (flags & FLAG_PROMOTION)
    {
        Piece promo = Piece(promo_piece(m));
        char promoLetter = piece_letter(promo);
        if (promoLetter)
        {
            san += '=';
            san += promoLetter;
        }
    }

    UndoState u;
    pos.do_move(m, u);
    bool givesCheck = pos.in_check(pos.side_to_move);
    MoveList replies;
    generate_moves(pos, replies);
    filter_legal_moves(pos, replies);
    bool mate = givesCheck && replies.count == 0;
    pos.undo_move(u);
    if (mate)
        san += '#';
    else if (givesCheck)
        san += '+';
    return san;
}

int main(int argc, char **argv)
{
    init_zobrist();
    init_bitboards();

    Position pos;
    pos.set_startpos();

    // Parse CLI args: --fen <6 tokens>, --depth N, --movetime ms, --timeleft ms
    std::string fen;
    int depth = 5;
    int movetime = 0;
    long long timeleft = 0;
    bool bench = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--fen" && i + 1 < argc)
        {
            // Accept either a single quoted FEN ("--fen \"8/8/... w - - 0 1\"") or the
            // six fields as separate tokens. The old code required exactly six tokens
            // and silently fell back to the start position otherwise, which is very
            // easy to trigger by accident -- zsh, unlike bash, does not word-split
            // unquoted variables, so `--fen $FEN` arrives as one argument.
            std::string first = argv[i + 1];
            if (first.find(' ') != std::string::npos)
            {
                fen = first;
                i += 1;
            }
            else
            {
                std::ostringstream os;
                int consumed = 0;
                for (int k = 1; k <= 6 && i + k < argc; ++k)
                {
                    std::string tok = argv[i + k];
                    if (tok.rfind("--", 0) == 0)
                        break;
                    if (consumed)
                        os << ' ';
                    os << tok;
                    ++consumed;
                }
                fen = os.str();
                i += consumed;
            }
            if (fen.empty())
            {
                std::cerr << "error: --fen given no value" << std::endl;
                return 2;
            }
        }
        else if (a == "--depth" && i + 1 < argc)
        {
            depth = std::max(1, std::atoi(argv[++i]));
        }
        else if (a == "--movetime" && i + 1 < argc)
        {
            movetime = std::max(0, std::atoi(argv[++i]));
        }
        else if (a == "--bench")
        {
            bench = true;
        }
        else if (a == "--timeleft" && i + 1 < argc)
        {
            timeleft = std::max(0LL, std::atoll(argv[++i]));
        }
    }
    if (!fen.empty())
    {
        pos.set_fen(fen);
    }

    TranspositionTable tt;
    tt.resize(64); // 64 MB

    // Evaluator selection.
    //
    // Two evaluators exist. The int16 NNUE (256x2-32-32-1, nnue.h) has a real
    // incremental accumulator and matching weights committed to the repo, so it is
    // the default. SimpleNNUEEvaluator reads the float residual-nnue-v1 format,
    // whose weights are gitignored and generally absent; it is opt-in via
    // PUFFERFISH_SIMPLE_NNUE so a missing file can no longer take the engine down.
    //
    // Note SimpleNNUEEvaluator implements the older SimpleNNUE architecture, whose
    // residual blocks carry a LayerNorm. It cannot represent the newer "compact"
    // checkpoints (no LayerNorm, trailing ReLU) such as inference006.pt.
    NNUEEvaluator inn;
    SimpleNNUEEvaluator snn;
    NNEvaluator *evaluator = nullptr;

    bool loaded = false;
    const char *loadedPath = nullptr;

    const char *int16Paths[] = {
        "bot/python/nnue_weights.bin",
        "../bot/python/nnue_weights.bin",
        "../../bot/python/nnue_weights.bin",
        "../../../bot/python/nnue_weights.bin",
    };
    const char *simplePaths[] = {
        "bot/python/nnue_residual_rebalanced_preprocessed.bin",
        "../bot/python/nnue_residual_rebalanced_preprocessed.bin",
        "../../bot/python/nnue_residual_rebalanced_preprocessed.bin",
        "../../../bot/python/nnue_residual_rebalanced_preprocessed.bin",
    };

    if (std::getenv("PUFFERFISH_SIMPLE_NNUE") != nullptr)
    {
        for (const char *p : simplePaths)
        {
            if (snn.load(p))
            {
                loaded = true;
                loadedPath = p;
                evaluator = static_cast<NNEvaluator *>(&snn);
                break;
            }
        }
    }
    else
    {
        if (const char *envPath = std::getenv("PUFFERFISH_NNUE_PATH"))
        {
            if (inn.load(envPath))
            {
                loaded = true;
                loadedPath = envPath;
                evaluator = static_cast<NNEvaluator *>(&inn);
            }
        }
        if (!loaded)
        {
            for (const char *p : int16Paths)
            {
                if (inn.load(p))
                {
                    loaded = true;
                    loadedPath = p;
                    evaluator = static_cast<NNEvaluator *>(&inn);
                    break;
                }
            }
        }
    }

    if (!loaded)
    {
        std::cerr << "error nnue_load_failed" << std::endl;
        return 2;
    }
    else
    {
        std::cerr << "info nnue_loaded " << loadedPath
                  << (evaluator == static_cast<NNEvaluator *>(&snn) ? " simple" : " int16")
                  << std::endl;
    }

    SearchContext ctx;
    ctx.tt = &tt;
    ctx.nn = evaluator;
    if (movetime > 0)
    {
        ctx.limits.time_ms = static_cast<std::uint64_t>(movetime);
        ctx.limits.depth = 0;
    }
    else
    {
        ctx.limits.depth = depth;
        ctx.limits.time_ms = 0;
        if (timeleft > 0)
            ctx.limits.time_left_ms = static_cast<std::uint64_t>(timeleft);
    }

    if (bench)
    {
        // Fixed suite at fixed depth, so optimization work can be compared run to run.
        static const char *benchFens[] = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9",
            "r2q1rk1/pp1bbppp/2np1n2/4p3/2B1P3/2NP1N2/PPP1QPPP/R1B2RK1 w - - 0 10",
            "8/5pk1/6p1/7p/5P1P/6P1/5K2/8 w - - 0 1",
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
            "6rk/ppp2p1p/2b1p3/4bp2/7q/2P2P2/PP2B2P/R2Q3K b - - 0 22",
        };
        const int n = (int)(sizeof(benchFens) / sizeof(benchFens[0]));
        std::uint64_t totalNodes = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < n; ++k)
        {
            Position bp;
            bp.set_fen(benchFens[k]);
            TranspositionTable btt;
            btt.resize(64);
            SearchContext bctx;
            bctx.tt = &btt;
            bctx.nn = evaluator;
            bctx.limits.depth = depth;
            bctx.limits.time_ms = 0;
            SearchResult r = search(bp, bctx);
            totalNodes += bctx.stats.nodes + bctx.stats.qnodes;
            std::cout << "  pos " << (k + 1) << "  best " << move_to_uci(r.bestMove)
                      << "  score " << r.score
                      << "  nodes " << (bctx.stats.nodes + bctx.stats.qnodes) << "\n";
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "bench depth " << depth
                  << "  nodes " << totalNodes
                  << "  time " << (long long)ms << "ms"
                  << "  nps " << (long long)(totalNodes / (ms / 1000.0)) << "\n";
        return 0;
    }

    auto st0 = std::chrono::steady_clock::now();
    SearchResult res = search(pos, ctx);
    auto st1 = std::chrono::steady_clock::now();
    double sms = std::chrono::duration<double, std::milli>(st1 - st0).count();

    std::uint64_t nodes = ctx.stats.nodes + ctx.stats.qnodes;
    std::cerr << "info depth " << res.depth
              << " score " << res.score
              << " nodes " << nodes
              << " time " << (long long)sms
              << " nps " << (long long)(sms > 0 ? nodes / (sms / 1000.0) : 0)
              << std::endl;

    if (res.bestMove == MOVE_NONE)
        std::cout << "bestmove 0000\n";
    else
        std::cout << "bestmove " << move_to_uci(res.bestMove) << "\n";

    return 0;
}