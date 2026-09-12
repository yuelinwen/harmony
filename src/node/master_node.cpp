#include "master_node.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

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
    std::cout << "\n===== 2. index =====" << std::endl;
    std::cout << "building index (nlist=" << nlist << ")" << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    index_.build(base_, nlist, iterations, cfg_.trainPoints);
    auto t1 = std::chrono::steady_clock::now();

    std::cout << "build time: "
              << std::chrono::duration<double>(t1 - t0).count() << " s" << std::endl;
}

// Centroid assignment only -- no worker is involved and nothing is searched,
// so this is cheap: it just records which clusters the workload touches.
void MasterNode::warmupPlan(int queries, int nprobe) {
    clusterHits_.assign(index_.getNlist(), 0);

    int n = query_.getN();
    if (queries < n) {
        n = queries;
    }
    for (int q = 0; q < n; ++q) {
        std::vector<int> clusters = index_.nearestClusters(query_.vec(q), nprobe);
        for (int i = 0; i < (int)clusters.size(); ++i) {
            clusterHits_[clusters[i]] = clusterHits_[clusters[i]] + 1;
        }
    }

    std::cout << "warmup: " << n << " queries profiled" << std::endl;
}

double MasterNode::imbalanceOf(int bVec, int bDim) const {
    // Work landing on each row: probes x cluster size x dimensions per machine.
    // Every machine in a row carries the same amount once the rotation has
    // evened out first-stop duty, so per-row is enough.
    std::vector<double> load(bVec, 0.0);
    for (int c = 0; c < index_.getNlist(); ++c) {
        long hits = clusterHits_.empty() ? 1 : clusterHits_[c];
        load[c % bVec] += (double)hits
                        * (double)index_.clusterIds(c).size()
                        * ((double)base_.getDim() / bDim);
    }

    double mean = 0.0;
    for (int r = 0; r < bVec; ++r) {
        mean = mean + load[r];
    }
    mean = mean / bVec;

    double var = 0.0;
    for (int r = 0; r < bVec; ++r) {
        var = var + (load[r] - mean) * (load[r] - mean);
    }
    return std::sqrt(var / bVec);
}

// Measured on Sift1M, nprobe=32, k=100. Between the tabulated points the
// value is interpolated; past the last one it is held flat.
double MasterNode::pruneFactor(int bDim) const {
    static const int slices[] = {1, 2, 4};
    static const double work[] = {1.00, 0.62, 0.51};
    int n = 3;

    if (bDim <= slices[0]) {
        return work[0];
    }
    for (int i = 1; i < n; ++i) {
        if (bDim <= slices[i]) {
            double t = (double)(bDim - slices[i - 1]) / (slices[i] - slices[i - 1]);
            return work[i - 1] + t * (work[i] - work[i - 1]);
        }
    }
    return work[n - 1];
}

double MasterNode::estimateCost(int bVec, int bDim) const {
    double totalWork = 0.0;   // multiply-adds over the whole grid
    double bytes = 0.0;

    for (int c = 0; c < index_.getNlist(); ++c) {
        long hits = clusterHits_.empty() ? 1 : clusterHits_[c];
        double candidates = (double)hits * (double)index_.clusterIds(c).size();

        totalWork = totalWork + candidates * base_.getDim();

        // A probed cluster is handed along its row once per dimension slice,
        // carrying one running total per candidate, so traffic scales with bDim.
        bytes = bytes + candidates * bDim * sizeof(float);
    }

    // Per machine, and reduced by however much pruning the slicing buys.
    double compute = totalWork * pruneFactor(bDim) / (bVec * bDim);

    return compute + cfg_.commCost * bytes + cfg_.alpha * imbalanceOf(bVec, bDim);
}

void MasterNode::choosePlan() {
    std::cout << "\n===== cost model =====" << std::endl;
    std::cout << "plan search (alpha=" << cfg_.alpha
              << ", commcost=" << cfg_.commCost << ")" << std::endl;

    int bestVec = 1;
    int bestDim = numWorkers_;
    double bestCost = -1.0;

    for (int bDim = 1; bDim <= numWorkers_; ++bDim) {
        if (numWorkers_ % bDim != 0) {
            continue;
        }
        int bVec = numWorkers_ / bDim;
        double cost = estimateCost(bVec, bDim);

        std::cout << "  " << bVec << " x " << bDim
                  << ":  work " << (100.0 * pruneFactor(bDim)) << "%"
                  << "   imbalance " << imbalanceOf(bVec, bDim)
                  << "   cost " << cost << std::endl;

        if (bestCost < 0.0 || cost < bestCost) {
            bestCost = cost;
            bestVec = bVec;
            bestDim = bDim;
        }
    }

    std::cout << "chosen grid: " << bestVec << " x " << bestDim << std::endl;
    cfg_.bVec = bestVec;
    cfg_.bDim = bestDim;
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
    chainOrder_ = SearchOrder(bDim_, bDim_, true);
    groupOrder_ = SearchOrder(bVec_, bVec_, true);

    int nlist = index_.getNlist();
    clusterOwner_.resize(nlist);
    for (int c = 0; c < nlist; ++c) {
        clusterOwner_[c] = c % bVec_;
    }

    std::cout << "\n===== 3. layout =====" << std::endl;
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
    std::cout << "\n===== 4. distribute =====" << std::endl;

    // how many clusters each row will receive
    std::vector<int> rowClusters(bVec_, 0);
    for (int c = 0; c < index_.getNlist(); ++c) {
        rowClusters[clusterOwner_[c]] = rowClusters[clusterOwner_[c]] + 1;
    }

    for (int w = 1; w <= numWorkers_; ++w) {
        int row = (w - 1) / bDim_;
        int col = (w - 1) % bDim_;
        int setup[4];
        setup[0] = plan_.end(col) - plan_.begin(col);
        setup[1] = rowClusters[row];
        setup[2] = bDim_;
        setup[3] = cfg_.batch;
        // MPI: blocking is fine for startup -- the order is fixed and nobody waits
        MPI_Send(setup, 4, MPI_INT, w, TAG_SETUP, MPI_COMM_WORLD);

        // This column's rows of the chain table: next, prev, stage.
        std::vector<int> table;
        table.insert(table.end(), chainOrder_.nextRow(col).begin(),
                     chainOrder_.nextRow(col).end());
        table.insert(table.end(), chainOrder_.prevRow(col).begin(),
                     chainOrder_.prevRow(col).end());
        for (int item = 0; item < bDim_; ++item) {
            const std::vector<int>& chain = chainOrder_.chain(item);
            int stage = 0;
            for (int p = 0; p < (int)chain.size(); ++p) {
                if (chain[p] == col) {
                    stage = p;
                    break;
                }
            }
            table.push_back(stage);
        }
        MPI_Send(table.data(), (int)table.size(), MPI_INT, w, TAG_ORDER,
                 MPI_COMM_WORLD);
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
    workerTimes_.assign(numWorkers_, std::vector<double>(6, 0.0));
    for (int w = 1; w <= numWorkers_; ++w) {
        MPI_Recv(perWorker.data(), bDim_, MPI_LONG, w, TAG_STATS,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int s = 0; s < bDim_; ++s) {
            aliveAfterStage_[s] = aliveAfterStage_[s] + perWorker[s];
        }
        MPI_Recv(workerTimes_[w - 1].data(), 6, MPI_DOUBLE, w, TAG_TIMES,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// One row per worker: how much of its run went to computing, and how much to
// waiting for somebody else. A worker that is mostly idle is being starved by
// the master; mostly in recv means its upstream is the slow one (paper
// Fig. 9).
void MasterNode::printWorkerTimes() const {
    if (workerTimes_.empty()) {
        return;
    }

    std::cout << "\n===== where each worker's time went =====" << std::endl;
    if (cfg_.check) {
        std::cout << "  (--check is on: idle also covers the reference pass"
                  << " the master runs between batches)" << std::endl;
    }
    std::cout << "  worker   grid    jobs     total    compute      idle"
              << "      recv      send" << std::endl;

    for (int w = 1; w <= numWorkers_; ++w) {
        const std::vector<double>& t = workerTimes_[w - 1];
        double total = t[0];
        if (total <= 0.0) {
            total = 1e-9;
        }

        std::string grid = std::to_string((w - 1) / bDim_) + "x"
                         + std::to_string((w - 1) % bDim_);
        std::cout << "  " << std::setw(6) << w
                  << std::setw(7) << grid
                  << std::setw(8) << (long)t[5]
                  << std::setw(9) << std::fixed << std::setprecision(2)
                  << t[0] << "s";

        // compute, idle, recv, send -- t[3], t[1], t[2], t[4]
        int order[4] = {3, 1, 2, 4};
        for (int i = 0; i < 4; ++i) {
            std::cout << std::setw(8) << std::setprecision(1)
                      << (100.0 * t[order[i]] / total) << "%";
        }
        std::cout << std::endl;
    }
}

// Without this the heap starts empty, worst() is infinite, and nothing can be
// pruned until a whole cluster has been through the pipeline (Algorithm 1,
// lines 1-5). Samples come from the nearest cluster, not at random, which
// makes the starting threshold much tighter.
//
// Only the master can do this: distances here are over every dimension, and a
// worker holds a fraction of each vector.
void MasterNode::prewarmHeap(const float* query, QueryState& state, TopKHeap& heap) {
    state.prewarmCluster.clear();
    state.prewarmed.clear();

    // Spread over the nearest few clusters rather than taking everything from
    // the first one, as the authors' code does. One cluster can miss: the
    // nearest centroid is not always where the nearest vectors are, and a seed
    // drawn only from there gives a loose threshold exactly when it matters.
    int lists = cfg_.prewarmLists;
    if (lists > (int)state.clusters.size()) {
        lists = (int)state.clusters.size();
    }

    for (int i = 0; i < lists; ++i) {
        int clusterId = state.clusters[i];
        const std::vector<int>& ids = index_.clusterIds(clusterId);

        int n = (int)ids.size();
        if (cfg_.prewarm < n) {
            n = cfg_.prewarm;
        }
        if (n <= 0) {
            continue;
        }

        for (int j = 0; j < n; ++j) {
            heap.push(ids[j], l2DistanceSquared(query, base_.vec(ids[j]),
                                                base_.getDim()));
        }
        state.prewarmCluster.push_back(clusterId);
        state.prewarmed.push_back(n);
    }
}

// Hands one cluster to every worker in its row and returns immediately. The
// workers of a row then run one after another, not in parallel: in parallel
// every slice would be computed in full and nothing saved (paper §3.2).
//
// Only the queries in `members` travel with the job, so a cluster wanted by
// three of thirty-two costs three rows of running totals, not thirty-two.
void MasterNode::dispatchOne(int row, int clusterId, int item,
                             const std::vector<int>& members,
                             const std::vector<float>& thresholds) {
    int n = (int)index_.clusterIds(clusterId).size();
    int m = (int)members.size();

    for (int col = 0; col < bDim_; ++col) {
        int w = row * bDim_ + col + 1;
        // MPI: three small messages per worker -- the job, who wants it, and
        // their thresholds
        int job[4] = {clusterId, n, item, m};
        MPI_Send(job, 4, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);
        MPI_Send(members.data(), m, MPI_INT, w, TAG_QIDX, MPI_COMM_WORLD);
        MPI_Send(thresholds.data(), m, MPI_FLOAT, w, TAG_THRESHOLD, MPI_COMM_WORLD);
    }
}

int MasterNode::lastRankOf(int row, int item) const {
    // whoever the table puts at the end of this item's chain
    const std::vector<int>& chain = chainOrder_.chain(item);
    return row * bDim_ + chain[chain.size() - 1] + 1;
}

// Three levels of overlap, which multiply out to one busy worker per machine:
// every query group is given work before anything is collected, each group
// holds one vector partition at a time (Fig. 5a), and within that partition up
// to bDim clusters are in flight (Fig. 5b). Groups refill on their own and
// MPI_Waitany takes whichever comes back first, so a slow group never holds up
// a fast one.
//
// Survivors enter their query's heap as soon as the cluster reports, so
// thresholds keep tightening -- sooner than Algorithm 1 line 18, which prunes
// strictly more and changes nothing else.
void MasterNode::vectorPipeline(const std::vector<std::vector<std::vector<int>>>& work,
                                const std::vector<std::vector<int>>& groupMembers,
                                const std::vector<QueryState>& batch,
                                std::vector<TopKHeap>& heaps) {
    struct Slot {
        int group;
        int row;
        int clusterId;
        int n;       // vectors in the cluster
        int kSend;   // how many of them the chain tail reports per query
        std::vector<int> members;
        std::vector<Candidate> top;   // members.size() rows of kSend
    };
    // One slot per worker is exactly enough -- a group only takes a new chunk
    // once its previous one has all come back, and at any stage the groups
    // hold one row each. The spare is so the search for a free slot below
    // cannot spin forever if that accounting ever changes.
    int maxInFlight = bVec_ * bDim_ + 1;
    std::vector<Slot> slot(maxInFlight);
    std::vector<MPI_Request> req(maxInFlight, MPI_REQUEST_NULL);

    std::vector<int> stage(bVec_, 0);         // which partition group g is on
    std::vector<size_t> pos(bVec_, 0);        // how far into that partition
    std::vector<int> inFlight(bVec_, 0);
    std::vector<int> chunkSize(bVec_, 0);
    int free = 0;

    std::vector<int> members;
    std::vector<float> thresholds;

    while (true) {
        for (int g = 0; g < bVec_; ++g) {
            // Keep taking chunks until one of them actually sends something
            // out, or the group has been through every partition.
            while (inFlight[g] == 0 && stage[g] < bVec_) {
                int r = groupOrder_.chain(g)[stage[g]];
                const std::vector<int>& list = work[g][r];

                // done with this partition: its distances are in the heaps, so
                // the next stage starts from a tighter threshold
                if (pos[g] >= list.size()) {
                    stage[g] = stage[g] + 1;
                    pos[g] = 0;
                    continue;
                }

                chunkSize[g] = 0;
                for (int p = 0; p < bDim_ && pos[g] + p < list.size(); ++p) {
                    int c = list[pos[g] + p];
                    chunkSize[g] = chunkSize[g] + 1;

                    // which queries of this group actually want this cluster
                    members.clear();
                    thresholds.clear();
                    for (int j = 0; j < (int)groupMembers[g].size(); ++j) {
                        int q = groupMembers[g][j];
                        const std::vector<int>& cl = batch[q].clusters;
                        if (std::find(cl.begin(), cl.end(), c) != cl.end()) {
                            members.push_back(q);
                            // an infinite threshold means nothing is dropped,
                            // the no-pruning arm of the ablation (Fig. 10)
                            thresholds.push_back(cfg_.pruning ? heaps[q].worst()
                                                              : PRUNED);
                        }
                    }
                    if (members.empty()) {
                        continue;
                    }

                    dispatchOne(r, c, p, members, thresholds);

                    while (req[free] != MPI_REQUEST_NULL) {
                        free = (free + 1) % maxInFlight;
                    }
                    int n = (int)index_.clusterIds(c).size();
                    int kSend = (cfg_.k < n) ? cfg_.k : n;
                    slot[free].group = g;
                    slot[free].row = r;
                    slot[free].clusterId = c;
                    slot[free].n = n;
                    slot[free].kSend = kSend;
                    slot[free].members = members;
                    slot[free].top.resize((size_t)members.size() * kSend);

                    // MPI: non-blocking, so the next cluster can be dispatched
                    // without waiting for this one. lastRankOf is who ends the
                    // chain; it sends back the k nearest per query rather than
                    // every running total, m*k instead of m*n floats.
                    MPI_Irecv(slot[free].top.data(),
                              (int)(slot[free].top.size() * sizeof(Candidate)),
                              MPI_BYTE, lastRankOf(r, p), TAG_TOPK,
                              MPI_COMM_WORLD, &req[free]);
                    inFlight[g] = inFlight[g] + 1;
                }

                // a chunk nobody in this group wanted still has to be stepped over
                if (inFlight[g] == 0) {
                    pos[g] = pos[g] + chunkSize[g];
                }
            }
        }

        // MPI: block until any one cluster reports, whichever it is. This is
        // what lets a slow row not hold up a fast one.
        int index = MPI_UNDEFINED;
        MPI_Waitany(maxInFlight, req.data(), &index, MPI_STATUS_IGNORE);
        if (index == MPI_UNDEFINED) {
            break;
        }

        const Slot& s = slot[index];
        const std::vector<int>& ids = index_.clusterIds(s.clusterId);

        for (int j = 0; j < (int)s.members.size(); ++j) {
            int q = s.members[j];
            const Candidate* top = &s.top[(size_t)j * s.kSend];

            // prewarm already pushed the leading ids of this query's first
            // cluster with their real distances; pushing them again would
            // duplicate them in the heap. The tail reports positions within
            // the cluster, so the same test still applies.
            //
            // Dropping one here can leave fewer than kSend from this cluster,
            // which is still the right answer: anything the tail left out has
            // kSend better candidates ahead of it, and those are either in
            // this list or were pushed by prewarm.
            int skip = 0;
            for (int i = 0; i < (int)batch[q].prewarmCluster.size(); ++i) {
                if (batch[q].prewarmCluster[i] == s.clusterId) {
                    skip = batch[q].prewarmed[i];
                    break;
                }
            }

            for (int t = 0; t < s.kSend; ++t) {
                if (top[t].dist >= PRUNED) {
                    break;   // the tail pads its unused slots, nearest first
                }
                int v = top[t].id;
                if (v < skip || v >= s.n) {
                    continue;
                }
                heaps[q].push(ids[v], top[t].dist);
            }
        }

        scanned_ = scanned_ + (long)s.members.size() * s.n;
        scannedRow_[s.row] = scannedRow_[s.row] + (long)s.members.size() * s.n;

        int g = s.group;
        inFlight[g] = inFlight[g] - 1;
        if (inFlight[g] == 0) {
            pos[g] = pos[g] + chunkSize[g];
        }
    }
}

std::vector<std::vector<Candidate>> MasterNode::queryPipeline(int firstQuery, int count,
                                                              int nprobe, int k) {
    // Stage 0: work out what each query needs, and prewarm its heap
    // (Algorithm 1 line 20).
    std::vector<QueryState> batch(count);
    std::vector<TopKHeap> heaps(count, TopKHeap(k));

    for (int q = 0; q < count; ++q) {
        const float* qv = query_.vec(firstQuery + q);
        batch[q].id = firstQuery + q;
        batch[q].clusters = index_.nearestClusters(qv, nprobe);
        prewarmHeap(qv, batch[q], heaps[q]);
    }

    // Every worker gets the whole batch's slices once; individual jobs then
    // name the queries they want by position.
    for (int w = 1; w <= numWorkers_; ++w) {
        int job[4] = {JOB_QUERY, 0, 0, 0};
        MPI_Send(job, 4, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);

        int col = (w - 1) % bDim_;
        int begin = plan_.begin(col);
        int myDim = plan_.end(col) - begin;

        std::vector<float> slices((size_t)cfg_.batch * myDim, 0.0f);
        for (int q = 0; q < count; ++q) {
            const float* src = query_.vec(batch[q].id) + begin;
            std::copy(src, src + myDim, &slices[(size_t)q * myDim]);
        }
        MPI_Send(slices.data(), (int)slices.size(), MPI_FLOAT, w,
                 TAG_QUERY, MPI_COMM_WORLD);
    }

    // Stage I: vector-level pipeline (Algorithm 1 lines 21-23).
    //
    // Fig. 5a splits the batch into as many query groups as there are vector
    // partitions and walks each group through the partitions one at a time, so
    // a group's later partitions prune against what its earlier ones found.
    // The split is contiguous; queries within a batch are interchangeable.
    std::vector<std::vector<int>> groupMembers(bVec_);
    for (int q = 0; q < count; ++q) {
        groupMembers[(int)((long)q * bVec_ / count)].push_back(q);
    }

    // work[g][r]: the clusters of partition r that group g probes. A cluster
    // wanted by two groups is visited once per group -- the same candidates
    // against fewer queries each time, so the distance work is unchanged.
    // Within one visit the queries still share it, which is the QueryBatch.
    std::vector<std::vector<std::vector<int>>> work(
        bVec_, std::vector<std::vector<int>>(bVec_));
    for (int g = 0; g < bVec_; ++g) {
        for (int j = 0; j < (int)groupMembers[g].size(); ++j) {
            const QueryState& qs = batch[groupMembers[g][j]];
            for (int i = 0; i < (int)qs.clusters.size(); ++i) {
                int c = qs.clusters[i];
                std::vector<int>& row = work[g][clusterOwner_[c]];
                if (std::find(row.begin(), row.end(), c) == row.end()) {
                    row.push_back(c);
                }
            }
        }
    }

    vectorPipeline(work, groupMembers, batch, heaps);

    std::vector<std::vector<Candidate>> out(count);
    for (int q = 0; q < count; ++q) {
        out[q] = heaps[q].results();
    }
    return out;
}

int MasterNode::run() {
    running_ = true;
    std::cout << "===== 1. data =====" << std::endl;

    if (!loadData(cfg_.data + "_base.bin", cfg_.data + "_query.bin") ||
        !loadGroundtruth(cfg_.data + "_gt.bin")) {
        return 1;
    }
    buildIndex(cfg_.nlist, cfg_.iters);

    // The layout has to be settled before any data moves, so the cost model
    // runs on a profiling pass first (paper Fig. 3, step 1).
    if (cfg_.costModel) {
        warmupPlan(cfg_.warmup, cfg_.nprobe);
        choosePlan();
    }

    splitGrid(cfg_.bVec, cfg_.bDim);
    distributeData();

    // The distributed answer must equal what one machine would have returned.
    // Compared as sets, not position by position: squared distances on SIFT
    // are integers, ties near rank k are common, and their order is not
    // defined either way.
    int k = cfg_.k;
    int nprobe = cfg_.nprobe;
    int nq = cfg_.nq;
    if (nq > query_.getN()) {
        std::cout << "only " << query_.getN() << " queries in the file, running those"
                  << std::endl;
        nq = query_.getN();
    }
    int differing = 0;
    int ties = 0;

    // Only the distributed search is timed. index_.search() below is the
    // single-machine reference used to check the answer, not part of the work.
    double seconds = 0.0;
    double recallSum = 0.0;

    // The query set can be run several times and the time averaged, which is
    // what a throughput number needs -- one pass on a cold cache is not
    // representative. Above one pass the first is a warm-up and is not timed.
    // Only the last pass is checked and counted, so the counters below mean
    // the same whatever --loop is.
    int passes = (cfg_.loop > 1) ? (cfg_.loop + 1) : 1;
    if (passes > 1) {
        std::cout << "\ntiming over " << cfg_.loop
                  << " passes, after one untimed warm-up" << std::endl;
    }

    for (int pass = 0; pass < passes; ++pass) {
    bool warmup = (passes > 1 && pass == 0);
    bool last = (pass == passes - 1);

    scanned_ = 0;
    scannedRow_.assign(bVec_, 0);
    recallSum = 0.0;
    differing = 0;
    ties = 0;

    // the workers keep their own counters, so they have to forget the earlier
    // passes too
    if (last && passes > 1) {
        for (int w = 1; w <= numWorkers_; ++w) {
            int job[4] = {JOB_RESET, 0, 0, 0};
            MPI_Send(job, 4, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);
        }
    }

    for (int start = 0; start < nq; start += cfg_.batch) {
        int count = (start + cfg_.batch <= nq) ? cfg_.batch : (nq - start);

        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::vector<Candidate>> spread = queryPipeline(start, count, nprobe, k);
        auto t1 = std::chrono::steady_clock::now();
        if (!warmup) {
            seconds = seconds + std::chrono::duration<double>(t1 - t0).count();
        }

        if (!last) {
            continue;   // intermediate passes are only there to be timed
        }

        for (int j = 0; j < count; ++j) {
            int q = start + j;
            recallSum = recallSum + recallAt(q, spread[j], k);

            // 这个参考对照比它检查的搜索还贵（单机、不剪枝），在计时区间之外，
            // 但跑上千条查询时它就是大部分墙钟时间，所以可以关掉。
            if (!cfg_.check) {
                continue;
            }
            std::vector<Candidate> single = index_.search(base_, query_.vec(q), nprobe, k);

            std::vector<int> a;
            std::vector<int> b;
            std::vector<float> da;
            std::vector<float> db;
            for (int i = 0; i < k; ++i) {
                a.push_back(spread[j][i].id);
                b.push_back(single[i].id);
                da.push_back(spread[j][i].dist);
                db.push_back(single[i].dist);
            }
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            std::sort(da.begin(), da.end());
            std::sort(db.begin(), db.end());

            if (a != b) {
                // Same distances but different members means a tie at rank k
                // was broken the other way -- both answers are equally
                // correct. Only a differing distance sequence is wrong.
                if (da == db) {
                    ties = ties + 1;
                } else {
                    differing = differing + 1;
                }
            }
        }
    }
    }

    if (cfg_.loop > 1) {
        seconds = seconds / cfg_.loop;
    }

    shutdown();   // stop the workers and collect their counters

    std::cout << "\n===== 5. results =====" << std::endl;
    std::cout << "setup: " << numWorkers_ << " workers, grid "
              << bVec_ << " x " << bDim_ << ", nlist " << index_.getNlist()
              << ", nprobe " << nprobe << ", k " << k
              << ", batch " << cfg_.batch << std::endl;

    // Is the distributed answer the same as one machine's? This is the check
    // that has to pass; everything below it is a measurement, not a verdict.
    std::cout << "\ncorrectness" << std::endl;
    std::cout << "  queries differing from single machine: "
              << differing << "/" << nq
              << "   (ties broken differently: " << ties << ")" << std::endl;
    std::cout << "  recall@" << k << ": " << (recallSum / nq)
              << "   (" << (k * (1.0 - recallSum / nq))
              << " of " << k << " true neighbours missed per query)" << std::endl;

    std::cout << "\nthroughput" << std::endl;
    std::cout << "  " << nq << " queries in " << seconds << " s" << std::endl;
    std::cout << "  QPS: " << (nq / seconds)
              << "   (" << (1000.0 * seconds / nq) << " ms per query)" << std::endl;

    // How much of the base each query actually touched, and how evenly that
    // work fell across the vector partitions. The spread here is what the
    // cost model's I(pi) term estimates in advance.
    double perQuery = (double)scanned_ / nq;
    std::cout << "\nwork" << std::endl;
    std::cout << "  candidates scanned: " << scanned_
              << "   (" << perQuery << " per query, "
              << (100.0 * perQuery / base_.getN()) << "% of the base)" << std::endl;
    if (bVec_ > 1) {
        std::cout << "  per vector partition   (even would be "
                  << (100.0 / bVec_) << "% each)" << std::endl;
        for (int r = 0; r < bVec_; ++r) {
            std::cout << "    partition " << r << ": " << scannedRow_[r]
                      << " candidates   "
                      << (100.0 * scannedRow_[r] / scanned_) << "%" << std::endl;
        }
    }

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
                  << "%   (" << processed << " candidates reached it)" << std::endl;
        done = done + processed;
    }
    std::cout << "distance work vs no pruning: "
              << (100.0 * done / (double)(scanned_ * bDim_)) << "%" << std::endl;

    printWorkerTimes();

    return 0;
}

}  // namespace harmony
