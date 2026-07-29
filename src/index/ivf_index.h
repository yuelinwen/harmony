#ifndef HARMONY_INDEX_IVF_INDEX_H
#define HARMONY_INDEX_IVF_INDEX_H

#include <vector>

#include "dataset.h"
#include "../engine/topk_heap.h"

// IvfIndex: cluster-based index (IVF = InVerted File).
//
// Self-contained, no external library. The paper (Section 5) implements
// Harmony in plain C++20 and uses Faiss only as a comparison baseline in
// the evaluation, so the index itself is ours.
//
// build():  kmeans over the base vectors -> nlist clusters.
//           centroids_[c]  = the center of cluster c
//           invlists_[c]   = ids of all base vectors inside cluster c
//
// search(): 1. compare the query against the nlist centroids (cheap)
//           2. keep the nprobe nearest clusters
//           3. scan only the vectors in those clusters
//
// This is the base that the paper's contributions plug into later:
//   - vector-based partition  = split the clusters across workers
//   - dimension-based partition = split each vector's dims across workers
//   - pruning = replace the full distance in step 3 with a sliced,
//     early-exit computation

namespace harmony {

class IvfIndex {
public:
    IvfIndex() {
        nlist_ = 0;
        dim_ = 0;
    }

    // Clusters the base vectors into nlist groups.
    // iterations = number of kmeans rounds (10 is usually enough).
    void build(const Dataset& base, int nlist, int iterations);

    // Returns the k nearest neighbors of one query vector, scanning only
    // the nprobe nearest clusters.
    std::vector<Candidate> search(const Dataset& base, const float* query,
                                  int nprobe, int k);

    int getNlist() {
        return nlist_;
    }

    int clusterSize(int c) {
        return (int)invlists_[c].size();
    }

private:
    int nlist_;                                // number of clusters
    int dim_;                                  // vector dimension
    std::vector<float> centroids_;             // nlist * dim, row-major
    std::vector<std::vector<int>> invlists_;   // invlists_[c] = ids in cluster c

    // Returns the id of the centroid nearest to vector v.
    int nearestCentroid(const float* v);
};

}  // namespace harmony

#endif  // HARMONY_INDEX_IVF_INDEX_H
