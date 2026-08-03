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

// The worker with rank w sits at row (w-1)/bDim, column (w-1)%bDim of the
// grid. Rows are vector partitions, columns are dimension slices.
//
// Clusters go to rows round-robin (c % bVec), the same simple rule the old
// vector-only split used. Dimensions are cut evenly across a row, which is
// what the paper does on a homogeneous cluster (Section 4.2).
void MasterNode::splitGrid(int bVec, int bDim) {
    bVec_ = bVec;
    bDim_ = bDim;
    plan_ = SlicePlan{base_.getDim(), bDim};

    int nlist = index_.getNlist();
    clusterOwner_.resize(nlist);
    for (int c = 0; c < nlist; ++c) {
        clusterOwner_[c] = c % bVec_;
    }

    std::cout << "grid: " << bVec_ << " vector partitions x "
              << bDim_ << " dimension slices" << std::endl;
    for (int r = 0; r < bVec_; ++r) {
        long vectors = 0;
        for (int c = 0; c < nlist; ++c) {
            if (clusterOwner_[c] == r) {
                vectors = vectors + index_.clusterSize(c);
            }
        }
        for (int col = 0; col < bDim_; ++col) {
            std::cout << "  worker " << (r * bDim_ + col + 1)
                      << ": partition " << r << ", dims ["
                      << plan_.begin(col) << "," << plan_.end(col)
                      << "), " << vectors << " vectors" << std::endl;
        }
    }
}

// Each cluster goes only to the workers in its row, and each of those gets
// only its own slice of the dimensions -- so a worker holds 1/bVec of the
// vectors x 1/bDim of the dimensions, one block of the paper's grid.
//
// The master does the cutting and sends only the slice, so no worker ever
// holds data it does not own.
void MasterNode::distributeData() {
    // how many clusters each row will receive
    std::vector<int> rowClusters(bVec_, 0);
    for (int c = 0; c < index_.getNlist(); ++c) {
        rowClusters[clusterOwner_[c]] = rowClusters[clusterOwner_[c]] + 1;
    }

    for (int w = 1; w <= numWorkers_; ++w) {
        int row = (w - 1) / bDim_;
        int col = (w - 1) % bDim_;
        int setup[3];
        setup[0] = plan_.end(col) - plan_.begin(col);
        setup[1] = rowClusters[row];
        setup[2] = bDim_;
        MPI_Send(setup, 3, MPI_INT, w, TAG_SETUP, MPI_COMM_WORLD);
    }

    for (int c = 0; c < index_.getNlist(); ++c) {
        const std::vector<int>& ids = index_.clusterIds(c);
        int row = clusterOwner_[c];

        for (int col = 0; col < bDim_; ++col) {
            int w = row * bDim_ + col + 1;
            int begin = plan_.begin(col);
            int myDim = plan_.end(col) - begin;

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

    std::cout << "distributed " << index_.getNlist() << " clusters over the grid"
              << std::endl;
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

// Hands one cluster to every worker in its row and returns immediately. The
// workers of a row then run one after another, not in parallel -- running
// them in parallel would compute every slice in full and save nothing (paper
// Section 3.2, Challenge 3).
void MasterNode::dispatchCluster(int clusterId, int n, int skip, float threshold) {
    int row = clusterOwner_[clusterId];

    for (int col = 0; col < bDim_; ++col) {
        int w = row * bDim_ + col + 1;
        int job[3] = {clusterId, n, skip};
        MPI_Send(job, 3, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);
        MPI_Send(&threshold, 1, MPI_FLOAT, w, TAG_THRESHOLD, MPI_COMM_WORLD);
    }
}

void MasterNode::collectCluster(int row, int n,
                                std::vector<float>& sums, std::vector<char>& alive) {
    int last = row * bDim_ + bDim_;   // final worker in the row

    sums.resize(n);
    alive.resize(n);
    MPI_Recv(sums.data(), n, MPI_FLOAT, last, TAG_SUMS,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(alive.data(), n, MPI_CHAR, last, TAG_ALIVE,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// One cluster per row at a time. Everything is dispatched before anything is
// collected, so while the master blocks on the first row the other rows are
// already computing -- that is where the parallelism comes from.
//
// Survivors go into the heap as soon as their row reports, so the threshold
// keeps tightening. (The paper only updates it after a whole partition,
// Algorithm 1 line 18; updating sooner prunes strictly more.)
void MasterNode::vectorPipeline(const std::vector<std::vector<int>>& perRow,
                                int prewarmClusterId, int prewarmed, TopKHeap& heap) {
    std::vector<size_t> next(bVec_, 0);   // per row: which cluster comes next
    std::vector<int> busy;                // rows dispatched this round
    std::vector<float> sums;
    std::vector<char> alive;

    while (true) {
        busy.clear();
        for (int r = 0; r < bVec_; ++r) {
            if (next[r] < perRow[r].size()) {
                int c = perRow[r][next[r]];
                int n = (int)index_.clusterIds(c).size();
                // skip: prewarm already put these in the heap with their real
                // distances, so running them again would push duplicates
                int skip = (c == prewarmClusterId) ? prewarmed : 0;
                dispatchCluster(c, n, skip, heap.worst());
                busy.push_back(r);
            }
        }
        if (busy.empty()) {
            break;
        }

        for (int i = 0; i < (int)busy.size(); ++i) {
            int r = busy[i];
            const std::vector<int>& ids = index_.clusterIds(perRow[r][next[r]]);
            int n = (int)ids.size();

            collectCluster(r, n, sums, alive);

            scanned_ = scanned_ + (long)n;
            scannedRow_[r] = scannedRow_[r] + (long)n;

            // whatever survived every slice has a real distance now
            for (int j = 0; j < n; ++j) {
                if (alive[j]) {
                    heap.push(ids[j], sums[j]);
                }
            }
            next[r] = next[r] + 1;
        }
    }
}

std::vector<Candidate> MasterNode::queryPipeline(const float* query, int nprobe, int k) {
    std::vector<int> clusters = index_.nearestClusters(query, nprobe);

    // hand every worker its slice of the query, cut the same way as the data
    for (int w = 1; w <= numWorkers_; ++w) {
        int job[3] = {JOB_QUERY, 0, 0};
        MPI_Send(job, 3, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);

        int col = (w - 1) % bDim_;
        int begin = plan_.begin(col);
        MPI_Send(query + begin, plan_.end(col) - begin, MPI_FLOAT, w,
                 TAG_QUERY, MPI_COMM_WORLD);
    }

    // Stage 0: prewarming (Algorithm 1 line 20)
    TopKHeap heap(k);
    int prewarmed = prewarmHeap(query, clusters[0], 500, heap);

    // Stage I: vector-level pipeline (Algorithm 1 lines 21-23). Group the
    // probed clusters by the vector partition that owns them, then run each
    // partition. (The paper's filterQueries picks the queries that need a
    // partition; with one query at a time, the same step becomes picking the
    // clusters of that query that live in the partition.)
    std::vector<std::vector<int>> perRow(bVec_);
    for (int i = 0; i < (int)clusters.size(); ++i) {
        perRow[clusterOwner_[clusters[i]]].push_back(clusters[i]);
    }

    vectorPipeline(perRow, clusters[0], prewarmed, heap);

    return heap.results();
}

int MasterNode::run() {
    running_ = true;
    std::cout << "master node started" << std::endl;

    if (!loadData("Data/sift_base.bin", "Data/sift_query.bin")) {
        return 1;
    }
    buildIndex(256, 10);

    // Grid shape: edit these two numbers to switch mode (their product must
    // equal the worker count). 2x2 = Harmony, 4x1 = Harmony-vector,
    // 1x4 = Harmony-dimension.
    int bVec = 2;
    int bDim = 2;
    if (bVec * bDim != numWorkers_) {
        bVec = 1;
        bDim = numWorkers_;   // fall back to pure dimension partition
    }
    splitGrid(bVec, bDim);
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
    int ties = 0;

    scanned_ = 0;
    scannedRow_.assign(bVec_, 0);

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
        std::vector<float> da;
        std::vector<float> db;
        for (int j = 0; j < k; ++j) {
            a.push_back(spread[j].id);
            b.push_back(single[j].id);
            da.push_back(spread[j].dist);
            db.push_back(single[j].dist);
        }
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        std::sort(da.begin(), da.end());
        std::sort(db.begin(), db.end());

        if (a != b) {
            // Same distances but different members means a tie at rank k was
            // broken the other way -- both answers are equally correct. Only
            // a differing distance sequence is an actual wrong result.
            if (da == db) {
                ties = ties + 1;
            } else {
                differing = differing + 1;
            }
        }
    }

    std::cout << "queries differing from single machine: "
              << differing << "/" << nq
              << "   (ties broken differently: " << ties << ")" << std::endl;
    std::cout << "QPS: " << (nq / seconds)
              << "   (" << (1000.0 * seconds / nq) << " ms per query)" << std::endl;

    shutdown();   // stop the workers and collect their counters

    // Pruning ratios in the shape of the paper's Table 3: the share of a
    // row's candidates that never had to reach each of its workers. The first
    // column is always 0 -- everyone computes the first slice, there is
    // nothing to skip yet.
    std::cout << "\npruning (" << bVec_ << " rows x " << bDim_ << " slices)"
              << std::endl;
    long done = 0;
    for (int r = 0; r < bVec_; ++r) {
        for (int col = 0; col < bDim_; ++col) {
            int w = r * bDim_ + col;   // worker index, 0-based
            long processed = (col == 0) ? scannedRow_[r] : aliveAfter_[w - 1];
            std::cout << "  worker " << (w + 1) << ": skipped "
                      << (100.0 * (1.0 - (double)processed / (double)scannedRow_[r]))
                      << "%" << std::endl;
            done = done + processed;
        }
    }
    std::cout << "distance work vs no pruning: "
              << (100.0 * done / (double)(scanned_ * bDim_)) << "%" << std::endl;

    return 0;
}

}  // namespace harmony
