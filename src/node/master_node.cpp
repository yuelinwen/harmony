#include "master_node.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
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

bool MasterNode::loadGroundtruth(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        std::cerr << "cannot open file: " << path << std::endl;
        return false;
    }

    int header[2];
    if (std::fread(header, sizeof(int), 2, file) != 2) {
        std::cerr << "cannot read header: " << path << std::endl;
        std::fclose(file);
        return false;
    }
    gtCount_ = header[0];
    gtDim_ = header[1];

    gt_.resize((size_t)gtCount_ * gtDim_);
    size_t got = std::fread(gt_.data(), sizeof(int), gt_.size(), file);
    std::fclose(file);

    if (got != gt_.size()) {
        std::cerr << "short read: " << path << std::endl;
        return false;
    }

    std::cout << "gt:    n=" << gtCount_ << " dim=" << gtDim_ << std::endl;
    return true;
}

// Counts how many of the true top-k this result found, as a fraction. Order
// does not matter -- a neighbour found at a different rank is still found.
double MasterNode::recallAt(int queryId, const std::vector<Candidate>& got, int k) const {
    if (queryId >= gtCount_ || k <= 0) {
        return 0.0;
    }
    int want = (k < gtDim_) ? k : gtDim_;   // the file may hold fewer than k
    const int* truth = &gt_[(size_t)queryId * gtDim_];

    int hits = 0;
    for (int i = 0; i < want; ++i) {
        for (int j = 0; j < (int)got.size(); ++j) {
            if (got[j].id == truth[i]) {
                hits = hits + 1;
                break;
            }
        }
    }
    return (double)hits / (double)want;
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
    aliveAfterStage_.assign(bDim_, 0);

    for (int w = 1; w <= numWorkers_; ++w) {
        int job[4] = {JOB_SHUTDOWN, 0, 0, 0};
        MPI_Send(job, 4, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);
    }

    // every worker reports its counts per chain position; sum them up
    std::vector<long> perWorker(bDim_);
    for (int w = 1; w <= numWorkers_; ++w) {
        MPI_Recv(perWorker.data(), bDim_, MPI_LONG, w, TAG_STATS,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int s = 0; s < bDim_; ++s) {
            aliveAfterStage_[s] = aliveAfterStage_[s] + perWorker[s];
        }
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

// Hands a batch of clusters to a row and returns immediately. The cluster at
// position p in the batch starts its chain at column p, so every worker is
// the first stop for one of them -- which matters because the first stop can
// prune nothing and therefore does the most work (paper Section 4.3, Load
// Balancing Strategies). Within one cluster the workers still run one after
// another; running them in parallel would compute every slice in full and
// save nothing (Section 3.2, Challenge 3).
//
// Each worker's jobs go out in the order it will actually reach them: at
// step t, the worker in column `col` handles batch[(col - t) mod m]. Sending
// them in any other order would leave a worker blocked on a cluster that has
// not arrived while one it could start sits behind it in the queue.
void MasterNode::dispatchBatch(int row, const std::vector<int>& batch,
                               int prewarmClusterId, int prewarmed, float threshold) {
    int m = (int)batch.size();

    for (int col = 0; col < bDim_; ++col) {
        int w = row * bDim_ + col + 1;

        for (int t = 0; t < m; ++t) {
            int p = ((col - t) % m + m) % m;
            int c = batch[p];
            int n = (int)index_.clusterIds(c).size();
            // skip: prewarm already put these in the heap with their real
            // distances, so running them again would push duplicates
            int skip = (c == prewarmClusterId) ? prewarmed : 0;

            int job[4] = {c, n, skip, p};
            MPI_Send(job, 4, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);
            MPI_Send(&threshold, 1, MPI_FLOAT, w, TAG_THRESHOLD, MPI_COMM_WORLD);
        }
    }
}

void MasterNode::collectCluster(int row, int startCol, int n,
                                std::vector<float>& sums, std::vector<char>& alive) {
    // the chain ends one step before it began, going round the row
    int lastCol = (startCol + bDim_ - 1) % bDim_;
    int last = row * bDim_ + lastCol + 1;

    sums.resize(n);
    alive.resize(n);
    MPI_Recv(sums.data(), n, MPI_FLOAT, last, TAG_SUMS,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(alive.data(), n, MPI_CHAR, last, TAG_ALIVE,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Two levels of overlap, which multiply out to one busy worker per machine:
//
//   across rows   -- every row is dispatched before anything is collected,
//                    so the rows compute at the same time (paper Fig. 5a)
//   within a row  -- up to bDim clusters are kept in flight, so while worker
//                    2 handles one cluster's second slice, worker 1 is
//                    already on the next cluster's first (paper Fig. 5b)
//
// Results come back in dispatch order: MPI keeps messages between a given
// pair of ranks ordered, and each worker handles its jobs in the order they
// arrive, so no request tracking is needed to match them up.
//
// Survivors go into the heap as soon as their cluster reports, so the
// threshold keeps tightening. (The paper only updates it after a whole
// partition, Algorithm 1 line 18; updating sooner prunes strictly more.)
void MasterNode::vectorPipeline(const std::vector<std::vector<int>>& perRow,
                                int prewarmClusterId, int prewarmed, TopKHeap& heap) {
    std::vector<size_t> pos(bVec_, 0);            // per row: clusters handled
    std::vector<std::vector<int>> batch(bVec_);   // per row: the batch in flight
    std::vector<float> sums;
    std::vector<char> alive;

    while (true) {
        // Send one batch into every row before collecting anything, so the
        // rows compute at the same time.
        bool any = false;
        for (int r = 0; r < bVec_; ++r) {
            batch[r].clear();
            for (int p = 0; p < bDim_ && pos[r] + p < perRow[r].size(); ++p) {
                batch[r].push_back(perRow[r][pos[r] + p]);
            }
            if (!batch[r].empty()) {
                // an infinite threshold means nothing is ever dropped, which
                // is the no-pruning arm of the ablation (paper Fig. 10)
                float threshold = cfg_.pruning ? heap.worst() : 1e30f;
                dispatchBatch(r, batch[r], prewarmClusterId, prewarmed, threshold);
                any = true;
            }
        }
        if (!any) {
            break;
        }

        for (int r = 0; r < bVec_; ++r) {
            for (int p = 0; p < (int)batch[r].size(); ++p) {
                const std::vector<int>& ids = index_.clusterIds(batch[r][p]);
                int n = (int)ids.size();

                collectCluster(r, p, n, sums, alive);

                scanned_ = scanned_ + (long)n;
                scannedRow_[r] = scannedRow_[r] + (long)n;

                // whatever survived every slice has a real distance now
                for (int j = 0; j < n; ++j) {
                    if (alive[j]) {
                        heap.push(ids[j], sums[j]);
                    }
                }
            }
            pos[r] = pos[r] + batch[r].size();
        }
    }
}

std::vector<Candidate> MasterNode::queryPipeline(const float* query, int nprobe, int k) {
    std::vector<int> clusters = index_.nearestClusters(query, nprobe);

    // hand every worker its slice of the query, cut the same way as the data
    for (int w = 1; w <= numWorkers_; ++w) {
        int job[4] = {JOB_QUERY, 0, 0, 0};
        MPI_Send(job, 4, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);

        int col = (w - 1) % bDim_;
        int begin = plan_.begin(col);
        MPI_Send(query + begin, plan_.end(col) - begin, MPI_FLOAT, w,
                 TAG_QUERY, MPI_COMM_WORLD);
    }

    // Stage 0: prewarming (Algorithm 1 line 20)
    TopKHeap heap(k);
    int prewarmed = prewarmHeap(query, clusters[0], cfg_.prewarm, heap);

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

    if (!loadData(cfg_.basePath, cfg_.queryPath) ||
        !loadGroundtruth(cfg_.gtPath)) {
        return 1;
    }
    buildIndex(cfg_.nlist, cfg_.iters);
    splitGrid(cfg_.bVec, cfg_.bDim);
    distributeData();

    // v1 check: splitting the work over the workers and merging it back
    // must give exactly what one machine would have returned.
    //
    // Compared as sets, not position by position. Squared distances on SIFT
    // are integers and ties are common near rank k, and which of two equally
    // distant vectors comes first is not defined either way.
    int k = cfg_.k;
    int nprobe = cfg_.nprobe;
    int nq = cfg_.nq;
    int differing = 0;
    int ties = 0;

    scanned_ = 0;
    scannedRow_.assign(bVec_, 0);

    // Only the distributed search is timed. index_.search() below is the
    // single-machine reference used to check the answer, not part of the work.
    double seconds = 0.0;
    double recallSum = 0.0;

    for (int q = 0; q < nq; ++q) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<Candidate> spread = queryPipeline(query_.vec(q), nprobe, k);
        auto t1 = std::chrono::steady_clock::now();
        seconds = seconds + std::chrono::duration<double>(t1 - t0).count();

        recallSum = recallSum + recallAt(q, spread, k);

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
    std::cout << "recall@" << k << ": " << (recallSum / nq) << std::endl;
    std::cout << "QPS: " << (nq / seconds)
              << "   (" << (1000.0 * seconds / nq) << " ms per query)" << std::endl;

    shutdown();   // stop the workers and collect their counters

    // Pruning ratios in the shape of the paper's Table 3: the share of
    // candidates that never had to reach the s-th slice of the chain. Slice 1
    // is always 0 -- everyone computes the first slice, there is nothing to
    // skip yet.
    std::cout << "\npruning (" << bDim_ << " slices per chain)" << std::endl;
    long done = 0;
    for (int s = 0; s < bDim_; ++s) {
        long processed = (s == 0) ? scanned_ : aliveAfterStage_[s - 1];
        std::cout << "  slice " << (s + 1) << ": skipped "
                  << (100.0 * (1.0 - (double)processed / (double)scanned_))
                  << "%" << std::endl;
        done = done + processed;
    }
    std::cout << "distance work vs no pruning: "
              << (100.0 * done / (double)(scanned_ * bDim_)) << "%" << std::endl;

    return 0;
}

}  // namespace harmony
