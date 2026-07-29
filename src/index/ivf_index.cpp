#include "ivf_index.h"

#include <cstdlib>
#include <iostream>

#include "distance.h"

namespace harmony {

// build: plain kmeans, then fill the inverted lists.
//
//   1. pick nlist random base vectors as the initial centroids
//   2. repeat `iterations` times:
//        assign: every vector joins its nearest centroid
//        update: every centroid moves to the mean of its members
//   3. final assign -> invlists_
void IvfIndex::build(const Dataset& base, int nlist, int iterations) {
    nlist_ = nlist;
    dim_ = base.getDim();
    int n = base.getN();

    // --- 1. init: copy nlist random vectors as starting centroids ---
    std::srand(42);   // fixed seed -> same clustering every run (reproducible)
    centroids_.resize((size_t)nlist_ * dim_);
    for (int c = 0; c < nlist_; ++c) {
        int pick = std::rand() % n;
        const float* v = base.vec(pick);
        for (int j = 0; j < dim_; ++j) {
            centroids_[(size_t)c * dim_ + j] = v[j];
        }
    }

    // assignment of every base vector, reused across iterations
    std::vector<int> owner(n);

    // scratch space for recomputing centroids
    std::vector<double> sum((size_t)nlist_ * dim_);   // double: avoids float rounding
    std::vector<int> count(nlist_);

    for (int it = 0; it < iterations; ++it) {

        // --- 2a. assign step: each vector -> nearest centroid ---
        for (int i = 0; i < n; ++i) {
            owner[i] = nearestCentroid(base.vec(i));
        }

        // --- 2b. update step: centroid = mean of its members ---
        for (size_t x = 0; x < sum.size(); ++x) {
            sum[x] = 0.0;
        }
        for (int c = 0; c < nlist_; ++c) {
            count[c] = 0;
        }

        for (int i = 0; i < n; ++i) {
            int c = owner[i];
            const float* v = base.vec(i);
            for (int j = 0; j < dim_; ++j) {
                sum[(size_t)c * dim_ + j] += v[j];
            }
            count[c] = count[c] + 1;
        }

        for (int c = 0; c < nlist_; ++c) {
            if (count[c] == 0) {
                // empty cluster: re-seed it with a random vector so it
                // does not stay empty forever
                int pick = std::rand() % n;
                const float* v = base.vec(pick);
                for (int j = 0; j < dim_; ++j) {
                    centroids_[(size_t)c * dim_ + j] = v[j];
                }
                continue;
            }
            for (int j = 0; j < dim_; ++j) {
                centroids_[(size_t)c * dim_ + j] =
                    (float)(sum[(size_t)c * dim_ + j] / count[c]);
            }
        }

        std::cout << "kmeans iteration " << (it + 1) << "/" << iterations << " done" << std::endl;
    }

    // --- 3. final assign -> inverted lists ---
    invlists_.clear();
    invlists_.resize(nlist_);
    for (int i = 0; i < n; ++i) {
        int c = nearestCentroid(base.vec(i));
        invlists_[c].push_back(i);
    }
}

int IvfIndex::nearestCentroid(const float* v) {
    int best = 0;
    float bestDist = l2DistanceSquared(v, &centroids_[0], dim_);
    for (int c = 1; c < nlist_; ++c) {
        float d = l2DistanceSquared(v, &centroids_[(size_t)c * dim_], dim_);
        if (d < bestDist) {
            bestDist = d;
            best = c;
        }
    }
    return best;
}

std::vector<Candidate> IvfIndex::search(const Dataset& base, const float* query,
                                        int nprobe, int k) {
    // 1. rank the centroids, keep the nprobe nearest clusters
    TopKHeap clusterHeap(nprobe);
    for (int c = 0; c < nlist_; ++c) {
        float d = l2DistanceSquared(query, &centroids_[(size_t)c * dim_], dim_);
        clusterHeap.push(c, d);
    }
    std::vector<Candidate> clusters = clusterHeap.results();

    // 2. scan only the vectors inside those clusters
    TopKHeap heap(k);
    for (int ci = 0; ci < (int)clusters.size(); ++ci) {
        int c = clusters[ci].id;
        const std::vector<int>& list = invlists_[c];
        for (int j = 0; j < (int)list.size(); ++j) {
            int vid = list[j];
            float d = l2DistanceSquared(query, base.vec(vid), dim_);
            heap.push(vid, d);
        }
    }
    return heap.results();
}

}  // namespace harmony
