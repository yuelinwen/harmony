#ifndef HARMONY_INDEX_DISTANCE_H
#define HARMONY_INDEX_DISTANCE_H

// Squared L2 distance, no sqrt: the ranking is the same either way.
//
// Pruning depends on the squared form. Every term is non-negative, so a
// partial sum over some of the dimensions can only grow, and a running total
// that has passed the threshold can never come back under it (paper §3.1).
// It also means the dimensions can be added in any order, which is why a
// cluster may enter the worker chain at any column.

namespace harmony {

inline float l2DistanceSquared(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum = sum + diff * diff;
    }
    return sum;
}

}  // namespace harmony

#endif  // HARMONY_INDEX_DISTANCE_H
