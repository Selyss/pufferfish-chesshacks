// Minimal runner: initialize engine, run a short search from startpos.

#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <filesystem>
#include <system_error>
#include <cstdint>
#include <thread>
#include <atomic>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include "engine/types.h"
#include "engine/bitboard.h"
#include "engine/position.h"
#include "engine/movegen.h"
#include "engine/tt.h"
#include "engine/nn_interface.h"
#include "engine/simple_nnue.h"
#include "engine/nnue_float.h"
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

// Directory containing this executable.
//
// Weights used to be looked up relative to the working directory, which is fine
// when you run the binary from the repository but not when something else starts
// it. Chess GUIs launch engines from an arbitrary directory and generally offer
// no way to set one -- En Croissant, for instance, exposes only name, Elo, search
// limits and UCI options -- so the engine would start and immediately die with
// nnue_load_failed. Resolving relative to the binary makes it work wherever it is
// launched from, which lichess-bot will need too.
static std::string executable_dir(const char *argv0)
{
    std::error_code ec;
#if defined(__APPLE__)
    char buf[8192];
    std::uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) == 0)
    {
        auto p = std::filesystem::weakly_canonical(std::filesystem::path(buf), ec);
        if (!ec)
            return p.parent_path().string();
    }
#elif defined(__linux__)
    auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec)
        return p.parent_path().string();
#endif
    if (argv0 && *argv0)
    {
        auto p = std::filesystem::weakly_canonical(std::filesystem::path(argv0), ec);
        if (!ec)
            return p.parent_path().string();
    }
    return std::filesystem::current_path(ec).string();
}

// Candidate locations for a weights file, most specific first: an explicit
// override, then upward from the executable, then upward from the working
// directory (which keeps every existing invocation working).
static std::vector<std::string> weight_candidates(const char *argv0,
                                                  const char *relative)
{
    std::vector<std::string> out;
    const std::string exeDir = executable_dir(argv0);
    std::filesystem::path base(exeDir);
    for (int i = 0; i < 5; ++i)
    {
        out.push_back((base / relative).string());
        if (!base.has_parent_path() || base.parent_path() == base)
            break;
        base = base.parent_path();
    }
    std::string up;
    for (int i = 0; i < 5; ++i)
    {
        out.push_back(up + relative);
        up += "../";
    }
    return out;
}

// Find the legal move matching a UCI string such as "e2e4" or "e7e8q".
static Move move_from_uci(Position &pos, const std::string &s)
{
    if (s.size() < 4)
        return MOVE_NONE;
    auto sq = [](char file, char rank) -> int
    { return (rank - '1') * 8 + (file - 'a'); };
    const int from = sq(s[0], s[1]);
    const int to = sq(s[2], s[3]);
    const char promo = (s.size() >= 5) ? s[4] : '\0';

    MoveList ml;
    generate_moves(pos, ml);
    filter_legal_moves(pos, ml);
    for (int i = 0; i < ml.count; ++i)
    {
        Move m = ml.moves[i];
        if (from_sq(m) != from || to_sq(m) != to)
            continue;
        if (move_flags(m) & FLAG_PROMOTION)
        {
            if (promo_char_from_piece(promo_piece(m)) != promo)
                continue;
        }
        else if (promo != '\0')
        {
            continue;
        }
        return m;
    }
    return MOVE_NONE;
}

// Minimal UCI loop. Enough for lichess-bot and for tools/selfplay.py to run
// whole games in one process, which also keeps the transposition table warm
// across moves instead of rebuilding it per move like the one-shot CLI does.
static int uci_loop(NNEvaluator *evaluator, const char *loadedPath)
{
    TranspositionTable tt;
    tt.resize(64);
    Position pos;
    pos.set_startpos();

    // The search runs on its own thread so the loop can keep reading input.
    // UCI requires "go infinite" to search until "stop" arrives, and requires
    // "isready" to be answered even mid-search; both are impossible if the
    // search blocks the only thread. Without this the engine simply hung.
    std::atomic<bool> stopFlag{false};
    std::thread searchThread;
    SearchContext ctx;

    auto finishSearch = [&]()
    {
        stopFlag.store(true, std::memory_order_relaxed);
        if (searchThread.joinable())
            searchThread.join();
    };

    // Keys of every position the game passed through before the current one.
    // The search needs these to see a repetition that began before its root.
    std::vector<Key> gameKeys;

    std::string line;
    while (std::getline(std::cin, line))
    {
        std::istringstream is(line);
        std::string token;
        is >> token;

        if (token == "uci")
        {
            std::cout << "id name Pufferfish\n";
            std::cout << "id author ChessHacks\n";
            std::cout << "option name Hash type spin default 64 min 1 max 1024\n";
            std::cout << "uciok" << std::endl;
        }
        else if (token == "isready")
        {
            std::cout << "readyok" << std::endl;
        }
        else if (token == "ucinewgame")
        {
            finishSearch();
            tt.resize(64); // clears
            pos.set_startpos();
            gameKeys.clear();
        }
        else if (token == "setoption")
        {
            std::string w;
            std::string name;
            int value = 0;
            while (is >> w)
            {
                if (w == "name")
                    is >> name;
                else if (w == "value")
                    is >> value;
            }
            if (name == "Hash" && value > 0)
                tt.resize(value);
        }
        else if (token == "position")
        {
            finishSearch();
            std::string sub;
            is >> sub;
            gameKeys.clear();
            if (sub == "startpos")
            {
                pos.set_startpos();
            }
            else if (sub == "fen")
            {
                std::string fen, part;
                for (int i = 0; i < 6 && (is >> part); ++i)
                {
                    if (part == "moves")
                        break;
                    if (!fen.empty())
                        fen += ' ';
                    fen += part;
                }
                pos.set_fen(fen);
                if (part == "moves")
                {
                    std::string mv;
                    while (is >> mv)
                    {
                        Move m = move_from_uci(pos, mv);
                        if (m == MOVE_NONE)
                            break;
                        UndoState u;
                        gameKeys.push_back(pos.key);
                        pos.do_move(m, u);
                    }
                    continue;
                }
            }
            std::string w;
            while (is >> w)
            {
                if (w == "moves")
                    continue;
                Move m = move_from_uci(pos, w);
                if (m == MOVE_NONE)
                    break;
                UndoState u;
                gameKeys.push_back(pos.key);
                pos.do_move(m, u);
            }
        }
        else if (token == "go")
        {
            finishSearch(); // never run two searches at once
            ctx = SearchContext{};
            ctx.tt = &tt;
            ctx.nn = evaluator;
            ctx.repetitionKeys = gameKeys;
            ctx.stop = &stopFlag;
            long long movetime = 0, wtime = 0, btime = 0, winc = 0, binc = 0;
            int depth = 0;
            std::string w;
            while (is >> w)
            {
                if (w == "movetime")
                    is >> movetime;
                else if (w == "depth")
                    is >> depth;
                else if (w == "wtime")
                    is >> wtime;
                else if (w == "btime")
                    is >> btime;
                else if (w == "winc")
                    is >> winc;
                else if (w == "binc")
                    is >> binc;
                else if (w == "infinite")
                    depth = 64;
            }
            if (movetime > 0)
            {
                ctx.limits.time_ms = (std::uint64_t)movetime;
                ctx.limits.depth = 0;
            }
            else if (depth > 0)
            {
                ctx.limits.depth = depth;
                ctx.limits.time_ms = 0;
            }
            else
            {
                const long long left = (pos.side_to_move == WHITE) ? wtime : btime;
                if (left > 0)
                {
                    ctx.limits.depth = 0;
                    ctx.limits.time_ms = 0;
                    ctx.limits.time_left_ms = (std::uint64_t)left;
                }
                else
                {
                    ctx.limits.depth = 6; // nothing specified; keep it quick
                }
            }

            stopFlag.store(false, std::memory_order_relaxed);
            Position searchPos = pos;
            searchThread = std::thread(
                [&ctx, searchPos]() mutable
                {
                    SearchResult res = search(searchPos, ctx);
                    std::uint64_t nodes = ctx.stats.nodes + ctx.stats.qnodes;
                    std::cout << "info depth " << res.depth << " score cp " << res.score
                              << " nodes " << nodes << std::endl;
                    if (res.bestMove == MOVE_NONE)
                        std::cout << "bestmove 0000" << std::endl;
                    else
                        std::cout << "bestmove " << move_to_uci(res.bestMove) << std::endl;
                });
        }
        else if (token == "stop")
        {
            finishSearch();
        }
        else if (token == "ponderhit")
        {
            // Not implemented; the search is already running normally.
        }
        else if (token == "quit")
        {
            finishSearch();
            break;
        }
    }
    finishSearch();
    (void)loadedPath;
    return 0;
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
    bool uci = (argc == 1); // bare invocation behaves like a normal UCI engine
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
        else if (a == "--uci")
        {
            uci = true;
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
    FloatNNUEEvaluator fnn;
    NNUEEvaluator inn;
    SimpleNNUEEvaluator snn;
    NNEvaluator *evaluator = nullptr;

    bool loaded = false;
    std::string loadedPathStr;

    const std::vector<std::string> floatPaths =
        weight_candidates(argv[0], "bot/python/nnue_float.bin");
    const std::vector<std::string> int16Paths =
        weight_candidates(argv[0], "bot/python/nnue_weights.bin");
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
                loadedPathStr = p;
                evaluator = static_cast<NNEvaluator *>(&snn);
                break;
            }
        }
    }
    else if (std::getenv("PUFFERFISH_INT16_NNUE") != nullptr)
    {
        // Opt-in only: the committed int16 weights evaluate every position as 0,
        // because export_int16.py divides by the quantization scale instead of
        // multiplying and all 393,216 accumulator weights round away. Kept so a
        // corrected int16 export can be tested without touching this file.
        for (const std::string &p : int16Paths)
        {
            if (inn.load(p.c_str()))
            {
                loaded = true;
                loadedPathStr = p;
                evaluator = static_cast<NNEvaluator *>(&inn);
                break;
            }
        }
    }
    else
    {
        if (const char *envPath = std::getenv("PUFFERFISH_NNUE_PATH"))
        {
            if (fnn.load(envPath))
            {
                loaded = true;
                loadedPathStr = envPath;
                evaluator = static_cast<NNEvaluator *>(&fnn);
            }
        }
        if (!loaded)
        {
            for (const std::string &p : floatPaths)
            {
                if (fnn.load(p.c_str()))
                {
                    loaded = true;
                    loadedPathStr = p;
                    evaluator = static_cast<NNEvaluator *>(&fnn);
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
        std::cerr << "info nnue_loaded " << loadedPathStr
                  << (evaluator == static_cast<NNEvaluator *>(&snn) ? " simple"
                       : evaluator == static_cast<NNEvaluator *>(&inn) ? " int16" : " float32")
                  << std::endl;
    }

    if (uci)
        return uci_loop(evaluator, loadedPathStr.c_str());

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