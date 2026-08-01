#ifndef HARMONY_ENGINE_PRUNING_SCANNER_H
#define HARMONY_ENGINE_PRUNING_SCANNER_H

#include <vector>

#include "../index/dataset.h"
#include "../index/distance.h"
#include "slice_plan.h"
#include "topk_heap.h"

// Dimension-level pruning (paper Algorithm 1, lines 6-12).
//
// Add up the distance one slice at a time. Partial sums only grow, so once
// the sum passes heap.worst() this vector can never reach the top-K and the
// remaining slices can be skipped.
//
// prunedAt[s] counts the vectors dropped at slice s.

namespace harmony {

inline void scanAll(const Dataset& base, const float* query,
                    const SlicePlan& plan,
                    TopKHeap& heap, std::vector<long>& prunedAt) {
    for (int i = 0; i < base.getN(); ++i) {
        float sum = 0.0f;
        bool pruned = false;

        for (int s = 0; s < plan.nSlices; ++s) {
            sum = sum + l2DistanceSquaredRange(query, base.vec(i),
                                               plan.begin(s), plan.end(s));
            if (sum > heap.worst()) {
                prunedAt[s] = prunedAt[s] + 1;
                pruned = true;
                break;
            }
        }

        if (!pruned) {
            heap.push(i, sum);
        }
    }
}

}  // namespace harmony

#endif  // HARMONY_ENGINE_PRUNING_SCANNER_H
