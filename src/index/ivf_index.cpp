#include "ivf_index.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "distance.h"
#include "../engine/stopwatch.h"

namespace harmony {

// build: plain kmeans, then fill the inverted lists.
//
//   1. pick nlist random base vectors as the initial centroids
//   2. repeat `iterations` times, over a sample of the data:
//        assign: every vector joins its nearest centroid
//        update: every centroid moves to the mean of its members
//   3. final assign, over all of it -> invlists_
//
// perCentroid caps the training sample at nlist * perCentroid vectors, 0 for
// no cap. Step 2 only has to locate the centroids, and a sample locates them
// about as well as the whole set does -- this is faiss's
// max_points_per_centroid, which the authors' code keeps at 256. Step 3 is
// not sampled: every vector has to end up in a list.
void IvfIndex::build(const Dataset& base, int nlist, int iterations,
                     int perCentroid) {
    nlist_ = nlist;
    dim_ = base.getDim();
    int n = base.getN();
    builtFrom_ = n;

    // Split in two at phase 3, which is the line the sample draws between
    // train and add: everything above it works on the sample of vectors,
    // everything below it touches all n.
    Stopwatch watch;

    // --- 1. init: nlist distinct vectors as starting centroids ---
    //
    // The order every draw below comes from: shuffled once, so taking a
    // prefix is a sample without replacement. Picking with replacement
    // (rand() % n, which this did) can start two centroids on the same
    // vector, and the iterations only pull them apart if some third vector
    // happens to land between them -- until then the pair splits one
    // cluster's worth of points and one cluster is effectively lost.
    std::srand(42);   // fixed seed -> same clustering every run (reproducible)
    std::vector<int> train(n);
    for (int i = 0; i < n; ++i) {
        train[i] = i;
    }
    for (int i = n - 1; i > 0; --i) {
        int j = std::rand() % (i + 1);
        int t = train[i];
        train[i] = train[j];
        train[j] = t;
    }

    centroids_.resize((size_t)nlist_ * dim_);
    for (int c = 0; c < nlist_; ++c) {
        // % n only matters when nlist > n, which is a degenerate index but
        // must not read past the end of the shuffle.
        const float* v = base.vec(train[c % n]);
        for (int j = 0; j < dim_; ++j) {
            centroids_[(size_t)c * dim_ + j] = v[j];
        }
    }

    // Which vectors the rounds below train on: all of them, or a prefix of
    // the same shuffle.
    long cap = (long)nlist_ * perCentroid;
    if (perCentroid > 0 && cap < n) {
        train.resize((size_t)cap);
        std::cout << "kmeans trains on " << train.size() << " of " << n
                  << " vectors" << std::endl;
    }
    int nt = (int)train.size();

    // assignment of every training vector, reused across iterations
    std::vector<int> owner(nt);

    // scratch space for recomputing centroids
    std::vector<double> sum((size_t)nlist_ * dim_);   // double: avoids float rounding
    std::vector<int> count(nlist_);

    for (int it = 0; it < iterations; ++it) {

        // --- 2a. assign step: each vector -> nearest centroid ---
        //
        // Nearly all of the build time goes here: n * nlist distance
        // computations per round. Each vector writes only its own owner[i]
        // and reads nothing that changes, so the rounds parallelise as they
        // are.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nt; ++i) {
            owner[i] = nearestCentroid(base.vec(train[i]));
        }

        // --- 2b. update step: centroid = mean of its members ---
        for (size_t x = 0; x < sum.size(); ++x) {
            sum[x] = 0.0;
        }
        for (int c = 0; c < nlist_; ++c) {
            count[c] = 0;
        }

        for (int i = 0; i < nt; ++i) {
            int c = owner[i];
            const float* v = base.vec(train[i]);
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

    trainSeconds_ = watch.seconds(true);

    // --- 3. final assign -> inverted lists ---
    //
    // Two passes: the distances in parallel, then the lists filled in order.
    // Appending inside the parallel loop would have several threads pushing
    // onto the same list, and would leave the ids in a different order on
    // every run.
    // Every vector, not the sample: the lists have to hold the whole dataset.
    owner.assign(n, 0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        owner[i] = nearestCentroid(base.vec(i));
    }

    invlists_.clear();
    invlists_.resize(nlist_);
    for (int i = 0; i < n; ++i) {
        invlists_[owner[i]].push_back(i);
    }

    addSeconds_ = watch.seconds();
}

// Which cluster does this vector belong to: compare against all nlist
// centroids and keep the nearest. Called only from build(), once per vector
// per kmeans round, which is where nearly all of the build time goes.
int IvfIndex::nearestCentroid(const float* v) const {
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

// The nprobe clusters a query should visit, nearest centroid first. This is
// the only part of the index the distributed search calls: the master runs it
// per query, then hands the cluster ids to the workers that own them.
//
// Cheap -- nlist distance computations, against the millions the workers then
// do. A TopKHeap is reused here to keep the nprobe smallest, holding cluster
// ids rather than vector ids.
std::vector<int> IvfIndex::nearestClusters(const float* query, int nprobe) const {
    TopKHeap clusterHeap(nprobe);
    for (int c = 0; c < nlist_; ++c) {
        float d = l2DistanceSquared(query, &centroids_[(size_t)c * dim_], dim_);
        clusterHeap.push(c, d);
    }

    std::vector<Candidate> best = clusterHeap.results();
    std::vector<int> out;
    for (int i = 0; i < (int)best.size(); ++i) {
        out.push_back(best[i].id);
    }
    return out;
}

// A whole search on one machine: pick the clusters, scan every vector in
// them, keep the k nearest. No pruning and no partitioning, which is what
// makes it the reference -- the distributed answer has to match this exactly.
// Not part of the system itself; run() calls it only to check the result.
std::vector<Candidate> IvfIndex::search(const Dataset& base, const float* query,
                                        int nprobe, int k) {
    return search(base, query, nearestClusters(query, nprobe), k);
}

std::vector<Candidate> IvfIndex::search(const Dataset& base, const float* query,
                                        const std::vector<int>& clusters, int k) {
    // Scan only the vectors inside those clusters
    TopKHeap heap(k);
    for (int ci = 0; ci < (int)clusters.size(); ++ci) {
        int c = clusters[ci];
        const std::vector<int>& list = invlists_[c];
        for (int j = 0; j < (int)list.size(); ++j) {
            int vid = list[j];
            float d = l2DistanceSquared(query, base.vec(vid), dim_);
            heap.push(vid, d);
        }
    }
    return heap.results();
}

// On-disk layout, little-endian, the same shape as the .bin data files:
//   int32 magic, int32 nlist, int32 dim, int32 n
//   float32 centroids[nlist * dim]
//   then per cluster: int32 size, int32 ids[size]
//
// No version field. The magic doubles as one: change the layout and change
// the magic, and an old file is rejected rather than misread.
static const int kIndexMagic = 0x48494658;   // "HIFX"

bool IvfIndex::save(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }

    int header[4] = {kIndexMagic, nlist_, dim_, builtFrom_};
    bool ok = std::fwrite(header, sizeof(int), 4, f) == 4;
    if (ok) {
        ok = std::fwrite(centroids_.data(), sizeof(float), centroids_.size(), f)
             == centroids_.size();
    }
    for (int c = 0; ok && c < nlist_; ++c) {
        int size = (int)invlists_[c].size();
        ok = std::fwrite(&size, sizeof(int), 1, f) == 1;
        if (ok && size > 0) {
            ok = std::fwrite(invlists_[c].data(), sizeof(int), size, f)
                 == (size_t)size;
        }
    }

    std::fclose(f);
    if (!ok) {
        std::remove(path.c_str());   // a half-written index is worse than none
    }
    return ok;
}

bool IvfIndex::load(const std::string& path, int expectN, int expectDim) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }

    int header[4];
    if (std::fread(header, sizeof(int), 4, f) != 4 ||
        header[0] != kIndexMagic || header[2] != expectDim ||
        header[3] != expectN) {
        std::fclose(f);
        return false;
    }

    int nlist = header[1];
    std::vector<float> centroids((size_t)nlist * header[2]);
    bool ok = std::fread(centroids.data(), sizeof(float), centroids.size(), f)
              == centroids.size();

    std::vector<std::vector<int>> lists(nlist);
    long total = 0;
    for (int c = 0; ok && c < nlist; ++c) {
        int size = 0;
        ok = std::fread(&size, sizeof(int), 1, f) == 1 && size >= 0;
        if (ok && size > 0) {
            lists[c].resize(size);
            ok = std::fread(lists[c].data(), sizeof(int), size, f)
                 == (size_t)size;
        }
        total = total + size;
    }
    std::fclose(f);

    // Every base vector belongs to exactly one cluster, so the lists have to
    // account for all of them. A file that passes the header check but not
    // this one was truncated.
    if (!ok || total != expectN) {
        return false;
    }

    nlist_ = nlist;
    dim_ = header[2];
    builtFrom_ = expectN;
    centroids_.swap(centroids);
    invlists_.swap(lists);
    return true;
}

}  // namespace harmony
