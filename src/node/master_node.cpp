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

// Round-robin: cluster 0 goes to worker 0, cluster 1 to worker 1, and so on.
// Simple, and good enough while all clusters are treated as equally hot.
// The paper assigns frequently accessed clusters more carefully (Section
// 6.2.2), but that needs query history we do not have yet.
void MasterNode::splitClusters(int numWorkers) {
    numWorkers_ = numWorkers;

    int nlist = index_.getNlist();
    clusterOwner_.resize(nlist);
    for (int c = 0; c < nlist; ++c) {
        clusterOwner_[c] = c % numWorkers;
    }

    // Report how even the split turned out. Uneven loads are the problem
    // the paper is about, so it is worth seeing the numbers from the start.
    std::vector<int> clusters(numWorkers, 0);
    std::vector<long> vectors(numWorkers, 0);
    for (int c = 0; c < nlist; ++c) {
        int w = clusterOwner_[c];
        clusters[w] = clusters[w] + 1;
        vectors[w] = vectors[w] + index_.clusterSize(c);
    }

    std::cout << "split " << nlist << " clusters over " << numWorkers
              << " workers" << std::endl;
    for (int w = 0; w < numWorkers; ++w) {
        std::cout << "  worker " << w << ": " << clusters[w] << " clusters, "
                  << vectors[w] << " vectors" << std::endl;
    }
}

// Each worker gets its own copy of the vectors in the clusters it owns, so
// that no worker reads the full dataset. That costs one extra copy of the
// base data here, but it is what a worker really holds under MPI.
void MasterNode::createWorkers() {
    workers_.clear();
    for (int w = 0; w < numWorkers_; ++w) {
        workers_.emplace_back(w + 1);   // ids start at 1, the master is 0
    }

    for (int c = 0; c < index_.getNlist(); ++c) {
        int w = clusterOwner_[c];
        workers_[w].addCluster(c, index_.clusterIds(c), base_);
    }

    std::cout << "created " << numWorkers_ << " workers" << std::endl;
    for (int w = 0; w < numWorkers_; ++w) {
        std::cout << "  worker " << workers_[w].getId() << ": "
                  << workers_[w].vectorCount() << " vectors" << std::endl;
    }
}

std::vector<Candidate> MasterNode::search(const float* query, int nprobe, int k) {
    std::vector<int> clusters = index_.nearestClusters(query, nprobe);

    // group the chosen clusters by the worker that owns them, so each
    // worker is asked once instead of once per cluster
    std::vector<std::vector<int>> perWorker(numWorkers_);
    for (int i = 0; i < (int)clusters.size(); ++i) {
        int c = clusters[i];
        perWorker[clusterOwner_[c]].push_back(c);
    }

    // merge every worker's answer into one heap
    TopKHeap heap(k);
    for (int w = 0; w < numWorkers_; ++w) {
        if (perWorker[w].empty()) {
            continue;
        }
        std::vector<Candidate> part = workers_[w].search(query, perWorker[w], k);
        for (int i = 0; i < (int)part.size(); ++i) {
            heap.push(part[i].id, part[i].dist);
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
    splitClusters(4);
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
