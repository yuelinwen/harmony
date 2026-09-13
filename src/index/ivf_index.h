#ifndef HARMONY_INDEX_IVF_INDEX_H
#define HARMONY_INDEX_IVF_INDEX_H

#include <string>
#include <vector>

#include "dataset.h"
#include "../engine/topk_heap.h"

// IvfIndex: cluster-based index (IVF = InVerted File). Self-contained, no
// external library.
//
// The clustering is global: it runs once, on the master, over the whole
// dataset, and only then is the result cut up for the workers. That is what
// lets the master send a query to just the few workers that could hold its
// neighbours instead of broadcasting it.
//
// The distributed side builds on this index rather than replacing it -- the
// vector partition decides which row owns which of these clusters, and
// search() stays single-machine, as the reference the distributed answer is
// checked against.

namespace harmony {

class IvfIndex {
public:
    IvfIndex() {
        nlist_ = 0;
        dim_ = 0;
    }

    // Clusters the base vectors into nlist groups.
    // iterations = number of kmeans rounds (10 is usually enough).
    void build(const Dataset& base, int nlist, int iterations,
               int perCentroid);

    // Writes the clustering to a file, and reads one back. A run with the
    // same data and the same parameters produces the same index, so it is
    // worth keeping: kmeans is the slowest part of startup, and reusing one
    // file also means two runs being compared share a clustering exactly.
    //
    // save returns false if the file cannot be written; load returns false if
    // it is missing, truncated, or was built for a different dataset, and
    // leaves the index untouched so the caller can just build instead.
    bool save(const std::string& path) const;
    bool load(const std::string& path, int expectN, int expectDim);

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
    int builtFrom_ = 0;                        // base vectors it was built on
    std::vector<float> centroids_;             // nlist * dim, row-major
    std::vector<std::vector<int>> invlists_;   // invlists_[c] = ids in cluster c

    // Returns the id of the centroid nearest to vector v.
    // const because build() calls it from a parallel loop: it must only read.
    int nearestCentroid(const float* v) const;
};

}  // namespace harmony

#endif  // HARMONY_INDEX_IVF_INDEX_H
