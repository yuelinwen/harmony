#ifndef HARMONY_INDEX_DISTANCE_H
#define HARMONY_INDEX_DISTANCE_H

// Distance functions between vectors.
//
// We use SQUARED L2 distance (no sqrt). For nearest neighbor search the
// ordering is the same with or without sqrt, and skipping it is faster.
// The paper's dimension-level pruning also relies on the squared form:
// partial sums over dimension groups only grow, never shrink.

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

// Same, but only over dimensions [begin, end).
// This is the building block for dimension-based partition and pruning:
// the full distance equals the sum of partial distances over disjoint ranges.
inline float l2DistanceSquaredRange(const float* a, const float* b, int begin, int end) {
    float sum = 0.0f;
    for (int i = begin; i < end; ++i) {
        float diff = a[i] - b[i];
        sum = sum + diff * diff;
    }
    return sum;
}

}  // namespace harmony

#endif  // HARMONY_INDEX_DISTANCE_H
