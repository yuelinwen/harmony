#ifndef HARMONY_INDEX_DISTANCE_H
#define HARMONY_INDEX_DISTANCE_H

// Squared L2 distance, no sqrt: the ranking is the same either way, and
// skipping the sqrt is faster.
//
// Pruning depends on the squared form. Because every term is non-negative,
// a partial sum over some of the dimensions can only grow as more are added,
// so a running total that has already passed the threshold can never come
// back under it -- which is what makes it safe to stop early (paper §3.1).
//
// This also means the dimensions can be added in any order, which is why a
// cluster may enter the worker chain at any column.

namespace harmony {

// Squared L2 distance over the full vector:
//     sum of (a[i] - b[i])^2 for i in [0, dim)
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
