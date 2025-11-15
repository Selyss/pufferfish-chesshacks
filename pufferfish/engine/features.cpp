#include "features.h"

namespace pf
{

    void extract_features(const Position &pos, std::vector<int> &features)
    {
        (void)pos;
        features.clear();
        // TODO: implement real NNUE feature extraction compatible with training.
    }

    void diff_features(const Position &before, const Position &after,
                       std::vector<int> &added, std::vector<int> &removed)
    {
        (void)before;
        (void)after;
        added.clear();
        removed.clear();
        // TODO: implement incremental feature diffs; currently unused by search.
    }

} // namespace pf
