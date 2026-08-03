#include "master_node.h"

#include <algorithm>
#include <chrono>
#include <iostream>

#include <mpi.h>

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
// The master does the cutting and sends only the slice, so no worker ever
// holds a dimension it does not own.
void MasterNode::distributeData() {
    for (int w = 1; w <= numWorkers_; ++w) {
        int setup[3];
        setup[0] = plan_.end(w - 1) - plan_.begin(w - 1);
        setup[1] = index_.getNlist();
        setup[2] = numWorkers_;
        MPI_Send(setup, 3, MPI_INT, w, TAG_SETUP, MPI_COMM_WORLD);
    }

    for (int c = 0; c < index_.getNlist(); ++c) {
        const std::vector<int>& ids = index_.clusterIds(c);

        for (int w = 1; w <= numWorkers_; ++w) {
            int begin = plan_.begin(w - 1);
            int myDim = plan_.end(w - 1) - begin;

            std::vector<float> data(ids.size() * myDim);
            for (size_t i = 0; i < ids.size(); ++i) {
                const float* v = base_.vec(ids[i]);
                for (int j = 0; j < myDim; ++j) {
                    data[i * myDim + j] = v[begin + j];
                }
            }

            int header[2];
            header[0] = c;
            header[1] = (int)ids.size();
            MPI_Send(header, 2, MPI_INT, w, TAG_CLUSTER, MPI_COMM_WORLD);
            MPI_Send(ids.data(), (int)ids.size(), MPI_INT, w, TAG_IDS, MPI_COMM_WORLD);
            MPI_Send(data.data(), (int)data.size(), MPI_FLOAT, w, TAG_DATA, MPI_COMM_WORLD);
        }
    }

    std::cout << "sent every cluster to " << numWorkers_ << " workers" << std::endl;
}

void MasterNode::shutdown() {
    aliveAfter_.assign(numWorkers_, 0);

    for (int w = 1; w <= numWorkers_; ++w) {
        int job[3] = {JOB_SHUTDOWN, 0, 0};
        MPI_Send(job, 3, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);
    }
    for (int w = 1; w <= numWorkers_; ++w) {
        MPI_Recv(&aliveAfter_[w - 1], 1, MPI_LONG, w, TAG_STATS,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
int MasterNode::prewarmHeap(const float* query, int clusterId, int count, TopKHeap& heap) {
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

// The workers run one after another, not in parallel -- running them in
// parallel would compute every slice in full and save nothing (paper Section
// 3.2, Challenge 3).
void MasterNode::dimensionPipeline(int clusterId, int n, int skip, float threshold,
                                   std::vector<float>& sums, std::vector<char>& alive) {
    for (int w = 1; w <= numWorkers_; ++w) {
        int job[3] = {clusterId, n, skip};
        MPI_Send(job, 3, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);
        MPI_Send(&threshold, 1, MPI_FLOAT, w, TAG_THRESHOLD, MPI_COMM_WORLD);
    }

    sums.resize(n);
    alive.resize(n);
    MPI_Recv(sums.data(), n, MPI_FLOAT, numWorkers_, TAG_SUMS,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(alive.data(), n, MPI_CHAR, numWorkers_, TAG_ALIVE,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

std::vector<Candidate> MasterNode::queryPipeline(const float* query, int nprobe, int k) {
    std::vector<int> clusters = index_.nearestClusters(query, nprobe);

    // hand every worker its slice of the query, cut the same way as the data
    for (int w = 1; w <= numWorkers_; ++w) {
        int job[3] = {JOB_QUERY, 0, 0};
        MPI_Send(job, 3, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);

        int begin = plan_.begin(w - 1);
        int myDim = plan_.end(w - 1) - begin;
        MPI_Send(query + begin, myDim, MPI_FLOAT, w, TAG_QUERY, MPI_COMM_WORLD);
    }

    TopKHeap heap(k);
    int prewarmed = prewarmHeap(query, clusters[0], 500, heap);

    // One cluster at a time. Survivors go into the heap right away, so the
    // threshold keeps tightening as the clusters go by. (The paper batches by
    // vector partition instead, Algorithm 1 line 21, which needs B_vec > 1.)
    std::vector<float> sums;
    std::vector<char> alive;

    for (int i = 0; i < (int)clusters.size(); ++i) {
        const std::vector<int>& ids = index_.clusterIds(clusters[i]);
        int n = (int)ids.size();
        scanned_ = scanned_ + (long)n;

        // skip: prewarm already put these in the heap with their real
        // distances, so running them again would push duplicates
        int skip = (i == 0) ? prewarmed : 0;
        dimensionPipeline(clusters[i], n, skip, heap.worst(), sums, alive);

        // whatever survived every slice has a real distance now
        for (int j = 0; j < n; ++j) {
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
    splitDimensions(numWorkers_);
    distributeData();

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

    // Only the distributed search is timed. index_.search() below is the
    // single-machine reference used to check the answer, not part of the work.
    double seconds = 0.0;

    for (int q = 0; q < nq; ++q) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<Candidate> spread = queryPipeline(query_.vec(q), nprobe, k);
        auto t1 = std::chrono::steady_clock::now();
        seconds = seconds + std::chrono::duration<double>(t1 - t0).count();

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
    std::cout << "QPS: " << (nq / seconds)
              << "   (" << (1000.0 * seconds / nq) << " ms per query)" << std::endl;

    shutdown();   // stop the workers and collect their counters

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
