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
// The clustering is global: it runs once, on the master, over the whole
// dataset, and only then is the result cut up for the workers. That is what
// lets the master send a query to just the few workers that can hold its
// neighbours, instead of broadcasting it to everyone.
//
// The distributed side builds on this index rather than replacing it:
//   - vector partition     = which worker row owns which of these clusters
//   - dimension partition  = which columns of each vector a worker keeps
//   - pruning              = the scan in step 3, spread across a worker chain
//                            and stopped early
//
// search() stays whole and single-machine, and is used as the reference the
// distributed answer is checked against.

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

    // The nprobe clusters whose centroids are nearest the query.
    // The master uses this to decide which workers a query has to visit.
    std::vector<int> nearestClusters(const float* query, int nprobe) const;

    int getNlist() const {
        return nlist_;
    }

    int clusterSize(int c) const {
        return (int)invlists_[c].size();
    }

    // The ids of the vectors inside cluster c.
    const std::vector<int>& clusterIds(int c) const {
        return invlists_[c];
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
