#ifndef HARMONY_ENGINE_SLICE_PLAN_H
#define HARMONY_ENGINE_SLICE_PLAN_H

// How to cut a vector's dimensions into blocks.
// Slice s covers dimensions [begin(s), end(s)).
//
// Written as (s * dim) / nSlices, not s * (dim / nSlices), so the slices
// still cover all of [0, dim) when nSlices does not divide dim exactly.

namespace harmony {

struct SlicePlan {
    int dim;
    int nSlices;

    int begin(int s) const {
        return s * dim / nSlices;
    }

    int end(int s) const {
        return (s + 1) * dim / nSlices;
    }
};

}  // namespace harmony

#endif  // HARMONY_ENGINE_SLICE_PLAN_H
