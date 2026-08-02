#include "master_node.h"

#include <algorithm>
#include <chrono>
#include <iostream>

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
void MasterNode::createWorkers() {
    workers_.clear();
    for (int w = 0; w < numWorkers_; ++w) {
        workers_.emplace_back(w + 1);   // ids start at 1, the master is 0
        workers_[w].setDimensions(plan_.begin(w), plan_.end(w));
    }

    for (int c = 0; c < index_.getNlist(); ++c) {
        for (int w = 0; w < numWorkers_; ++w) {
            workers_[w].addCluster(c, index_.clusterIds(c), base_);
        }
    }

    std::cout << "created " << numWorkers_ << " workers" << std::endl;
    for (int w = 0; w < numWorkers_; ++w) {
        std::cout << "  worker " << workers_[w].getId() << ": "
                  << workers_[w].vectorCount() << " vectors x "
                  << (plan_.end(w) - plan_.begin(w)) << " dims" << std::endl;
    }
}

std::vector<Candidate> MasterNode::search(const float* query, int nprobe, int k) {
    std::vector<int> clusters = index_.nearestClusters(query, nprobe);

    // the candidate ids, in the order every worker walks them
    std::vector<int> ids;
    for (int i = 0; i < (int)clusters.size(); ++i) {
        const std::vector<int>& list = index_.clusterIds(clusters[i]);
        for (int j = 0; j < (int)list.size(); ++j) {
            ids.push_back(list[j]);
        }
    }

    // Each worker reports the distance over its own dimensions only. Summing
    // them gives the real distance, because the slices are disjoint and cover
    // everything: D^2 = sum of d_k^2 (paper Section 3.1).
    std::vector<float> sums(ids.size(), 0.0f);
    for (int w = 0; w < numWorkers_; ++w) {
        std::vector<float> part = workers_[w].partialDistances(query, clusters);
        for (int i = 0; i < (int)part.size(); ++i) {
            sums[i] = sums[i] + part[i];
        }
    }

    // only now is there a real distance to rank by
    TopKHeap heap(k);
    for (int i = 0; i < (int)ids.size(); ++i) {
        heap.push(ids[i], sums[i]);
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

    return 0;
}

}  // namespace harmony
