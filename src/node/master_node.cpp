#include "master_node.h"

#include <algorithm>
#include <chrono>
#include <iostream>

#include "../index/distance.h"

namespace harmony {

bool MasterNode::loadData(const std::string& basePath, const std::string& queryPath) {
    if (!base_.load(basePath)) {
        return false;
    }
    if (!query_.load(queryPath)) {
        return false;
    }

    std::cout << "base:  n=" << base_.getN() << " dim=" << base_.getDim() << std::endl;
    std::cout << "query: n=" << query_.getN() << " dim=" << query_.getDim() << std::endl;
    return true;
}

void MasterNode::buildIndex(int nlist, int iterations) {
    std::cout << "building index (nlist=" << nlist << ")" << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    index_.build(base_, nlist, iterations);
    auto t1 = std::chrono::steady_clock::now();

    std::cout << "build time: "
              << std::chrono::duration<double>(t1 - t0).count() << " s" << std::endl;
}

// Cut the dimensions into equal slices, one per worker. The paper gives each
// machine a share of the dimensions proportional to its compute capacity,
// which on a homogeneous cluster is just d/N (Section 4.2).
void MasterNode::splitDimensions(int numWorkers) {
    numWorkers_ = numWorkers;
    plan_ = SlicePlan{base_.getDim(), numWorkers};

    std::cout << "split " << base_.getDim() << " dimensions over "
              << numWorkers << " workers" << std::endl;
    for (int w = 0; w < numWorkers; ++w) {
        std::cout << "  worker " << (w + 1) << ": dims ["
                  << plan_.begin(w) << "," << plan_.end(w) << ")" << std::endl;
    }
}

// Every worker gets every cluster, but only its own slice of each vector.
// The load is identical on all workers by construction, whatever the query
// pattern looks like -- that is the point of partitioning by dimension.
//
// The master does the cutting and hands over only the slice, so no worker
// ever holds a dimension it does not own. Under MPI the buffer built here is
// exactly what gets sent.
void MasterNode::createWorkers() {
    workers_.clear();
    for (int w = 0; w < numWorkers_; ++w) {
        workers_.emplace_back(w + 1);   // ids start at 1, the master is 0
        workers_[w].setDimCount(plan_.end(w) - plan_.begin(w));
    }

    for (int c = 0; c < index_.getNlist(); ++c) {
        const std::vector<int>& ids = index_.clusterIds(c);

        for (int w = 0; w < numWorkers_; ++w) {
            int begin = plan_.begin(w);
            int myDim = plan_.end(w) - begin;

            std::vector<float> data(ids.size() * myDim);
            for (size_t i = 0; i < ids.size(); ++i) {
                const float* v = base_.vec(ids[i]);
                for (int j = 0; j < myDim; ++j) {
                    data[i * myDim + j] = v[begin + j];
                }
            }

            workers_[w].addCluster(c, ids, data);
        }
    }

    std::cout << "created " << numWorkers_ << " workers" << std::endl;
    for (int w = 0; w < numWorkers_; ++w) {
        std::cout << "  worker " << workers_[w].getId() << ": "
                  << workers_[w].vectorCount() << " vectors x "
                  << (plan_.end(w) - plan_.begin(w)) << " dims" << std::endl;
    }
}

// Without this the heap starts empty, worst() is infinite, and nothing can be
// pruned until a whole cluster has been through the pipeline (paper Algorithm
// 1, lines 1-5).
//
// The samples come from the nearest cluster rather than at random: vectors in
// it are far more likely to be real neighbours, which makes the starting
// threshold much tighter.
//
// Distances here are over every dimension. A partial distance is smaller than
// the real one, and using one as the threshold would drop candidates that
// belonged in the top-K. Only the master can do this -- a worker holds a
// fraction of each vector.
int MasterNode::prewarm(const float* query, int clusterId, int count, TopKHeap& heap) {
    const std::vector<int>& ids = index_.clusterIds(clusterId);

    int n = (int)ids.size();
    if (count < n) {
        n = count;
    }

    for (int i = 0; i < n; ++i) {
        heap.push(ids[i], l2DistanceSquared(query, base_.vec(ids[i]), base_.getDim()));
    }

    return n;
}

std::vector<Candidate> MasterNode::search(const float* query, int nprobe, int k) {
    std::vector<int> clusters = index_.nearestClusters(query, nprobe);

    TopKHeap heap(k);
    int prewarmed = prewarm(query, clusters[0], 500, heap);

    // One cluster at a time. Survivors go into the heap right away, so the
    // threshold keeps tightening as the clusters go by.
    for (int i = 0; i < (int)clusters.size(); ++i) {
        const std::vector<int>& ids = index_.clusterIds(clusters[i]);

        std::vector<float> sums(ids.size(), 0.0f);
        std::vector<char> alive(ids.size(), 1);

        // prewarm already put these in the heap with their real distances;
        // running them through the pipeline again would push duplicates
        if (i == 0) {
            for (int j = 0; j < prewarmed; ++j) {
                alive[j] = 0;
            }
        }

        scanned_ = scanned_ + (long)ids.size();

        // Dimension pipeline: the workers run one after another, not in
        // parallel. Each adds its slice to the running total and drops what
        // has already passed the threshold, so later workers see fewer and
        // fewer candidates. Running them in parallel would compute every
        // slice in full and save nothing (paper Section 3.2, Challenge 3).
        for (int w = 0; w < numWorkers_; ++w) {
            // the master cuts the query the same way it cut the data
            workers_[w].accumulate(query + plan_.begin(w), clusters[i],
                                   heap.worst(), sums, alive);

            long n = 0;
            for (int j = 0; j < (int)alive.size(); ++j) {
                if (alive[j]) {
                    n = n + 1;
                }
            }
            aliveAfter_[w] = aliveAfter_[w] + n;
        }

        // whatever survived all the slices has a real distance now
        for (int j = 0; j < (int)ids.size(); ++j) {
            if (alive[j]) {
                heap.push(ids[j], sums[j]);
            }
        }
    }

    return heap.results();
}

int MasterNode::run() {
    running_ = true;
    std::cout << "master node started" << std::endl;

    if (!loadData("Data/sift_base.bin", "Data/sift_query.bin")) {
        return 1;
    }
    buildIndex(256, 10);
    splitDimensions(4);
    createWorkers();

    // v1 check: splitting the work over the workers and merging it back
    // must give exactly what one machine would have returned.
    //
    // Compared as sets, not position by position. Squared distances on SIFT
    // are integers and ties are common near rank k, and which of two equally
    // distant vectors comes first is not defined either way.
    int k = 100;
    int nprobe = 32;
    int nq = 100;
    int differing = 0;

    scanned_ = 0;
    aliveAfter_.assign(numWorkers_, 0);

    for (int q = 0; q < nq; ++q) {
        std::vector<Candidate> spread = search(query_.vec(q), nprobe, k);
        std::vector<Candidate> single = index_.search(base_, query_.vec(q), nprobe, k);

        std::vector<int> a;
        std::vector<int> b;
        for (int j = 0; j < k; ++j) {
            a.push_back(spread[j].id);
            b.push_back(single[j].id);
        }
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());

        if (a != b) {
            differing = differing + 1;
        }
    }

    std::cout << "queries differing from single machine: "
              << differing << "/" << nq << std::endl;

    // Pruning ratios in the shape of the paper's Table 3: the share of
    // candidates that never had to reach worker w. Worker 0 is always 0 --
    // everyone computes the first slice, there is nothing to skip yet.
    std::cout << "\npruning (" << numWorkers_ << " slices)" << std::endl;
    long done = 0;
    for (int w = 0; w < numWorkers_; ++w) {
        double skipped = 1.0 - (double)(w == 0 ? scanned_ : aliveAfter_[w - 1])
                                   / (double)scanned_;
        std::cout << "  worker " << (w + 1) << ": skipped " << (100.0 * skipped)
                  << "%" << std::endl;
        done = done + (w == 0 ? scanned_ : aliveAfter_[w - 1]);
    }
    std::cout << "distance work vs no pruning: "
              << (100.0 * done / (double)(scanned_ * numWorkers_)) << "%" << std::endl;

    return 0;
}

}  // namespace harmony
