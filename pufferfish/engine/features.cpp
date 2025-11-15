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

} // namespace pf
