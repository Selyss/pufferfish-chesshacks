#include "features.h"
#include <algorithm>
#include <iterator>

namespace pf
{

    void extract_features(const Position &pos, std::vector<int> &features)
    {
        features.clear();
        const bool stmWhite = (pos.side_to_move == WHITE);
        for (int sq = 0; sq < 64; ++sq)
        {
            Piece pc = pos.board[sq];
            if (pc == NO_PIECE)
                continue;
            bool isWhite = (pc <= W_KING);
            int typeIdx;
            switch (pc)
            {
            case W_PAWN:
            case B_PAWN:
                typeIdx = 0;
                break;
            case W_KNIGHT:
            case B_KNIGHT:
                typeIdx = 1;
                break;
            case W_BISHOP:
            case B_BISHOP:
                typeIdx = 2;
                break;
            case W_ROOK:
            case B_ROOK:
                typeIdx = 3;
                break;
            case W_QUEEN:
            case B_QUEEN:
                typeIdx = 4;
                break;
            case W_KING:
            case B_KING:
                typeIdx = 5;
                break;
            default:
                continue;
            }
            bool isFriendly = (stmWhite ? isWhite : !isWhite);
            int colorOffset = isFriendly ? 0 : 6; // 0..5 friendly, 6..11 enemy
            int feat = (colorOffset + typeIdx) * 64 + sq;
            if (feat >= 0 && feat < FEATURE_DIM)
                features.push_back(feat);
        }
    }

    void diff_features(const Position &before, const Position &after,
                       std::vector<int> &added, std::vector<int> &removed)
    {
        added.clear();
        removed.clear();
        // Naive diff via extraction; can be optimized later.
        std::vector<int> fb, fa;
        extract_features(before, fb);
        extract_features(after, fa);
        std::sort(fb.begin(), fb.end());
        std::sort(fa.begin(), fa.end());
        std::set_difference(fa.begin(), fa.end(), fb.begin(), fb.end(), std::back_inserter(added));
        std::set_difference(fb.begin(), fb.end(), fa.begin(), fa.end(), std::back_inserter(removed));
    }

    void extract_features_795(const Position &pos, std::vector<float> &out)
    {
        out.assign(795, 0.0f);

        // 0..767: PSQ one-hot (12x64): channels 0..5 = friendly Pawn..King, 6..11 = enemy Pawn..King
        const bool stmWhite = (pos.side_to_move == WHITE);
        auto psq_index = [&](bool friendly, int typeIdx, int sq) -> int
        {
            int channel = (friendly ? 0 : 6) + typeIdx;
            return channel * 64 + sq; // 0..767
        };
        for (int sq = 0; sq < 64; ++sq)
        {
            Piece pc = pos.board[sq];
            if (pc == NO_PIECE)
                continue;
            bool isWhite = (pc <= W_KING);
            int typeIdx;
            switch (pc)
            {
            case W_PAWN:
            case B_PAWN:
                typeIdx = 0;
                break;
            case W_KNIGHT:
            case B_KNIGHT:
                typeIdx = 1;
                break;
            case W_BISHOP:
            case B_BISHOP:
                typeIdx = 2;
                break;
            case W_ROOK:
            case B_ROOK:
                typeIdx = 3;
                break;
            case W_QUEEN:
            case B_QUEEN:
                typeIdx = 4;
                break;
            case W_KING:
            case B_KING:
                typeIdx = 5;
                break;
            default:
                continue;
            }
            bool friendly = (stmWhite ? isWhite : !isWhite);
            int idx = psq_index(friendly, typeIdx, sq);
            out[idx] = 1.0f;
        }

        int cursor = 768;
        // 768: side-to-move (1 for White, 0 for Black)
        out[cursor++] = (pos.side_to_move == WHITE) ? 1.0f : 0.0f; // 768

        // 769..772: castling rights K,Q,k,q
        int cr = pos.castling_rights;                // bitmask consistent with position.cpp usage
        out[cursor++] = (cr & 0b0001) ? 1.0f : 0.0f; // K
        out[cursor++] = (cr & 0b0010) ? 1.0f : 0.0f; // Q
        out[cursor++] = (cr & 0b0100) ? 1.0f : 0.0f; // k
        out[cursor++] = (cr & 0b1000) ? 1.0f : 0.0f; // q

        // 773..780: en-passant file one-hot (a..h) if any; else all zeros
        if (pos.ep_square != -1)
        {
            int file = pos.ep_square & 7;
            if (file >= 0 && file < 8)
                out[cursor + file] = 1.0f;
        }
        cursor += 8;

        // 781: material balance (white - black) in pawns=1.0, knights/bishops=3.0, rooks=5.0, queens=9.0
        auto count_piece = [&](Piece p) -> int
        {
            // Count bits in piece bitboard
            Bitboard bb = pos.pieceBB[p];
            return (int)__popcnt64(bb);
        };
        int wp = count_piece(W_PAWN), bp = count_piece(B_PAWN);
        int wn = count_piece(W_KNIGHT), bn = count_piece(B_KNIGHT);
        int wb = count_piece(W_BISHOP), bbp = count_piece(B_BISHOP);
        int wr = count_piece(W_ROOK), br = count_piece(B_ROOK);
        int wq = count_piece(W_QUEEN), bq = count_piece(B_QUEEN);
        float material = (float)((wp - bp) * 1.0 + (wn - bn) * 3.0 + (wb - bbp) * 3.0 + (wr - br) * 5.0 + (wq - bq) * 9.0);
        out[cursor++] = material; // 781

        // 782..793: per-piece counts (white then black): P,N,B,R,Q,K (K included as count 1)
        int countsW[6] = {wp, wn, wb, wr, wq, count_piece(W_KING)};
        int countsB[6] = {bp, bn, bbp, br, bq, count_piece(B_KING)};
        for (int i = 0; i < 6; ++i)
            out[cursor++] = (float)countsW[i];
        for (int i = 0; i < 6; ++i)
            out[cursor++] = (float)countsB[i];

        // 794: phase indicator (0..1). Use a simple taper: total minor+major material normalized.
        int totalPhase = 4 * 1 + 4 * 3 + 4 * 3 + 4 * 5 + 2 * 9; // rough max when both sides full: pawns excluded
        int curPhase = (wn + bn) * 3 + (wb + bbp) * 3 + (wr + br) * 5 + (wq + bq) * 9;
        float phase = totalPhase ? (float)curPhase / (float)totalPhase : 0.0f; // 1 opening -> 0 endgame
        out[cursor++] = phase;

        // Ensure we filled exactly 795
        (void)cursor;
    }

} // namespace pf
