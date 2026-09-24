#ifndef HARMONY_ENGINE_SLICE_PLAN_H
#define HARMONY_ENGINE_SLICE_PLAN_H

// How to cut a vector's dimensions into blocks: slice s covers
// [begin(s), end(s)). Written as (s * dim) / nSlices rather than
// s * (dim / nSlices) so the slices still cover all of [0, dim) when nSlices
// does not divide dim exactly.

namespace harmony {

struct SlicePlan {
    // Given values, so a default-constructed plan is harmless rather than
    // indeterminate: nSlices of 0 would divide by zero in both accessors.
    // The master overwrites both in splitGrid() before anything reads them.
    int dim = 0;
    int nSlices = 1;

    int begin(int s) const {
        return s * dim / nSlices;
    }

    int end(int s) const {
        return (s + 1) * dim / nSlices;
    }
};

}  // namespace harmony

#endif  // HARMONY_ENGINE_SLICE_PLAN_H
