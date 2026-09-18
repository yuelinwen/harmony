#include "master_node.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <iomanip>
#include <ios>
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

// The file an index with these settings would be written to. Everything that
// changes the clustering is in the name, so two runs that differ in any of it
// cannot pick up each other's file. What the data itself is gets checked
// inside load(), against the base's own n and dim.
std::string MasterNode::indexPath(int nlist, int iterations) const {
    return cfg_.data + "_nlist" + std::to_string(nlist)
         + "_iters" + std::to_string(iterations)
         + "_tp" + std::to_string(cfg_.trainPoints) + ".idx";
}

void MasterNode::buildIndex(int nlist, int iterations) {
    std::cout << "\n===== 2. index =====" << std::endl;

    std::string path = indexPath(nlist, iterations);
    auto t0 = std::chrono::steady_clock::now();

    if (cfg_.cache && index_.load(path, base_.getN(), base_.getDim())) {
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "loaded index from " << path << " ("
                  << std::chrono::duration<double>(t1 - t0).count() << " s)"
                  << std::endl;
        return;
    }

    std::cout << "building index (nlist=" << nlist << ")" << std::endl;
    index_.build(base_, nlist, iterations, cfg_.trainPoints);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "build time: "
              << std::chrono::duration<double>(t1 - t0).count() << " s" << std::endl;

    if (cfg_.cache) {
        if (index_.save(path)) {
            std::cout << "saved index to " << path << std::endl;
        } else {
            std::cout << "could not write " << path << ", carrying on" << std::endl;
        }
    }
}

// Paper §6.6 wants a workload where some machines are asked for much more than
// others, which needs the probe lists themselves to be lopsided -- the real
// nearest clusters of real queries are not.
//
// --skew s sends that share of every query's probes into a hot eighth of the
// clusters and scatters the rest. Two things matter about how it is written.
// It depends on the cluster ids alone, not on the layout, so the same workload
// can be put to a 8x1 and a 1x8 and the numbers compared. And it is a pure
// function of the query id, so the profiling pass, the search and the
// reference all derive the same list without anybody storing or sending it --
// which is what keeps differing a verdict in the one experiment where recall
// has stopped being one.
//
// (The sample fakes its listidqueries the same way, but only in the
// distributed path, so recall there is meaningless and nothing checks the
// answer at all.)
std::vector<int> MasterNode::probesFor(int queryId, int nprobe) const {
    if (cfg_.skew <= 0.0) {
        return index_.nearestClusters(query_.vec(queryId), nprobe);
    }

    int nlist = index_.getNlist();
    if (nprobe > nlist) {
        nprobe = nlist;
    }
    int hot = nlist / 8;
    if (hot < 1) {
        hot = 1;
    }

    int wanted = (int)(cfg_.skew * nprobe + 0.5);   // probes aimed at the hot set
    if (wanted > hot) {
        wanted = hot;      // cannot take more distinct clusters than there are
    }

    // A generator seeded from the query id, so this is reproducible and every
    // caller gets the same answer for the same query.
    unsigned int state = (unsigned int)queryId * 2654435761u + 12345u;
    std::vector<char> taken(nlist, 0);
    std::vector<int> out;
    out.reserve(nprobe);

    for (int pass = 0; pass < 2; ++pass) {
        int want = (pass == 0) ? wanted : nprobe;
        int range = (pass == 0) ? hot : nlist;

        // Rejection sampling, then a linear walk, so a full range still
        // terminates rather than spinning on collisions.
        int tries = 0;
        while ((int)out.size() < want && tries < 8 * nprobe) {
            state = state * 1103515245u + 12345u;
            int c = (int)((state >> 16) % (unsigned int)range);
            if (!taken[c]) {
                taken[c] = 1;
                out.push_back(c);
            }
            tries = tries + 1;
        }
        for (int c = 0; (int)out.size() < want && c < range; ++c) {
            if (!taken[c]) {
                taken[c] = 1;
                out.push_back(c);
            }
        }
    }
    return out;
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
        std::vector<int> clusters = probesFor(q, nprobe);
        for (int i = 0; i < (int)clusters.size(); ++i) {
            clusterHits_[clusters[i]] = clusterHits_[clusters[i]] + 1;
        }
    }

    // The paper labels its workloads by "variance" and never says how it is
    // obtained; the sample computes it as the standard deviation of how often
    // each cluster is probed (query.cpp:735-746), which is what §4.2.1 calls
    // I(pi) in the text.
    //
    // That number cannot be matched to the paper's. It carries units of probe
    // counts, so it scales with how many queries were profiled: the most it
    // can reach is mean * sqrt((nlist - hot) / hot) with mean = n * nprobe /
    // nlist, which here tops out near 330 while Fig. 7 says 500. Fig. 7 states
    // none of n, nprobe or nlist, so there is nothing to match against. The
    // ratio to the mean is printed alongside because it is scale-free, and is
    // the figure worth comparing between runs.
    double mean = (double)n * nprobe / index_.getNlist();
    double var = 0.0;
    for (int c = 0; c < index_.getNlist(); ++c) {
        double d = (double)clusterHits_[c] - mean;
        var = var + d * d;
    }
    var = std::sqrt(var / index_.getNlist());

    std::cout << "warmup: " << n << " queries profiled, cluster probe counts"
              << " mean " << mean << " variance " << var
              << " (" << (mean > 0.0 ? var / mean : 0.0) << " x mean)" << std::endl;
    if (cfg_.skew > 0.0) {
        std::cout << "SKEWED WORKLOAD (--skew " << cfg_.skew << "): probe lists"
                  << " are synthetic, recall is not meaningful" << std::endl;
    }
}

// Longest-processing-time first: heaviest cluster to whichever partition is
// lightest so far. Round-robin ignores that kmeans produces clusters of very
// different sizes and that queries do not visit them equally often, so one
// partition ends up doing noticeably more work -- and a query is not done
// until its slowest partition reports. This greedy is the standard one for
// the problem and comes within 4/3 of the best possible split. Ties go to the
// lower cluster id so a run is reproducible.
//
// A cluster's weight is how often it is probed times how many vectors it
// holds, which is the number of distance computations it brings.
std::vector<double> MasterNode::partitionLoads(int bVec,
                                               std::vector<int>* owner) const {
    int nlist = index_.getNlist();
    if (owner != nullptr) {
        owner->assign(nlist, 0);
    }

    std::vector<std::pair<double, int> > order(nlist);
    for (int c = 0; c < nlist; ++c) {
        long hits = clusterHits_.empty() ? 1 : clusterHits_[c];
        order[c] = std::make_pair((double)hits * index_.clusterSize(c), c);
    }
    std::sort(order.begin(), order.end(),
              [](const std::pair<double, int>& a,
                 const std::pair<double, int>& b) {
                  if (a.first != b.first) {
                      return a.first > b.first;
                  }
                  return a.second < b.second;
              });

    std::vector<double> load(bVec, 0.0);
    for (int i = 0; i < nlist; ++i) {
        int lightest = 0;
        for (int r = 1; r < bVec; ++r) {
            if (load[r] < load[lightest]) {
                lightest = r;
            }
        }
        if (owner != nullptr) {
            (*owner)[order[i].second] = lightest;
        }
        load[lightest] = load[lightest] + order[i].first;
    }
    return load;
}

double MasterNode::imbalanceOf(int bVec, int bDim) const {
    // The split this bVec would actually get, since that is what the term is
    // meant to price. Each machine in a row carries its row's clusters over
    // its own slice of the dimensions.
    std::vector<double> load = partitionLoads(bVec, nullptr);
    for (int r = 0; r < bVec; ++r) {
        load[r] = load[r] * ((double)base_.getDim() / bDim);
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

    // Which partition gets which cluster. Round-robin ignores that kmeans
    // produces clusters of very different sizes, and that queries do not visit
    // them equally often, so one partition ends up doing noticeably more work
    // than another -- and since a query is not done until its slowest
    // partition reports, that difference is throughput.
    //
    // Longest-processing-time first instead: heaviest cluster to whichever
    // partition is lightest so far. It is the standard greedy for this and
    // comes within 4/3 of the best possible split. Ties go to the lower
    // cluster id so a run is reproducible.
    //
    // This is the assignment the cost model's I(pi) measures. Until now that
    // term could only rank grid shapes, since the assignment inside a grid was
    // fixed; the paper's pi is the partition plan itself (§4.2.1).
    int nlist = index_.getNlist();
    partitionLoads(bVec_, &clusterOwner_);

    // The chain tags are derived from the master's in-flight slots, so the
    // largest one this layout can produce is fixed here. Nothing near MPI's
    // ceiling is reachable at any worker count that fits on a cluster, but a
    // tag MPI rejects fails at the receive rather than at the send, which is
    // an unpleasant way to find out.
    if (!tagsFitMpi(bVec_ * bDim_ + 1)) {
        std::cerr << "chain tags exceed this MPI's maximum" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Vectors are what a partition stores; the share is what it works on.
    // They are not the same number, and it is the second one that decides how
    // long a query waits -- a small partition of much-probed clusters is the
    // slow one. The shares are what the split above evens out.
    std::vector<double> load = partitionLoads(bVec_, nullptr);
    double total = 0.0;
    for (int r = 0; r < bVec_; ++r) {
        total = total + load[r];
    }

    std::cout << "\n===== 3. layout =====" << std::endl;
    std::cout << "grid: " << bVec_ << " vector partitions x "
              << bDim_ << " dimension slices  (even share would be "
              << (100.0 / bVec_) << "%)" << std::endl;
    for (int r = 0; r < bVec_; ++r) {
        long vectors = 0;
        for (int c = 0; c < nlist; ++c) {
            if (clusterOwner_[c] == r) {
                vectors = vectors + index_.clusterSize(c);
            }
        }
        double share = (total > 0.0) ? (100.0 * load[r] / total) : 0.0;
        for (int col = 0; col < bDim_; ++col) {
            std::cout << "  worker " << (r * bDim_ + col + 1)
                      << ": partition " << r << ", dims ["
                      << plan_.begin(col) << "," << plan_.end(col)
                      << "), " << vectors << " vectors, "
                      << share << "% of the work" << std::endl;
        }
    }
}

// Each cluster goes only to the workers in its row, and each of those gets
// only its own slice of the dimensions -- so a worker holds 1/bVec of the
// vectors x 1/bDim of the dimensions, one block of the paper's grid.
//
// The master does the cutting and sends only the slice, so no worker ever
// holds data it does not own.
// One column's three rows of the chain table -- next, prev and stage, one
// entry per item -- which is all a worker needs to know about the order.
std::vector<int> MasterNode::chainTableFor(int col) const {
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
    return table;
}

// Paper §4.3, the paragraph after Fig. 5b: when the machine holding one
// dimension block becomes overloaded, later queries should process that block
// last. Later stages are the light ones -- by the end of a chain most
// candidates have been pruned -- so the position is the lever.
//
// Measure, decide, rewrite the table, send it. Only between batches: a block
// carries nothing but its item number, and both ends of a hop have to agree
// on what that item means, so the table can only change while no block is
// part-way along a chain.
//
// The paper gives no rule, so this one: rank the columns by the arithmetic
// they did since the last look, and make the heaviest one the last stop of
// every chain. The others keep rotating among the earlier stages, which is
// what stops any of them becoming the permanent head.
void MasterNode::reorderChains() {
    if (!cfg_.reorder || bDim_ < 2) {
        return;
    }

    collectStats();

    // Work done since the previous look. A --loop pass or a new --nprobes
    // value zeroes the workers' counters, which shows up as a drop.
    std::vector<double> colLoad(bDim_, 0.0);
    for (int w = 0; w < numWorkers_; ++w) {
        double now = workerTimes_[w][3];
        double before = (w < (int)prevCompute_.size()) ? prevCompute_[w] : 0.0;
        double delta = (now >= before) ? (now - before) : now;
        colLoad[w % bDim_] += delta;
    }
    prevCompute_.assign(numWorkers_, 0.0);
    for (int w = 0; w < numWorkers_; ++w) {
        prevCompute_[w] = workerTimes_[w][3];
    }

    // Exponential smoothing, so one unrepresentative batch cannot swing the
    // order, and a dead zone, so a spread that is already small leaves it
    // alone. Without both, the heaviest column is relieved, becomes the
    // lightest, and the order flips back and forth every batch.
    if (colSmooth_.empty()) {
        colSmooth_ = colLoad;
    } else {
        for (int c = 0; c < bDim_; ++c) {
            colSmooth_[c] = 0.7 * colSmooth_[c] + 0.3 * colLoad[c];
        }
    }

    double lo = colSmooth_[0];
    double hi = colSmooth_[0];
    for (int c = 1; c < bDim_; ++c) {
        lo = (colSmooth_[c] < lo) ? colSmooth_[c] : lo;
        hi = (colSmooth_[c] > hi) ? colSmooth_[c] : hi;
    }
    if (hi <= 0.0 || (hi - lo) / hi < kReorderDeadzone) {
        return;
    }

    // Columns lightest first; the last of them is the one to bury.
    std::vector<int> order(bDim_);
    for (int c = 0; c < bDim_; ++c) {
        order[c] = c;
    }
    std::sort(order.begin(), order.end(),
              [this](int a, int b) { return colSmooth_[a] < colSmooth_[b]; });

    int busiest = order[bDim_ - 1];
    std::vector<int> rest(order.begin(), order.end() - 1);

    std::vector<std::vector<int> > chains(bDim_);
    for (int item = 0; item < bDim_; ++item) {
        chains[item].clear();
        for (int p = 0; p < (int)rest.size(); ++p) {
            chains[item].push_back(rest[(item + p) % rest.size()]);
        }
        chains[item].push_back(busiest);
    }
    chainOrder_ = SearchOrder(chains, bDim_);

    for (int w = 1; w <= numWorkers_; ++w) {
        int job[5] = {JOB_ORDER, 0, 0, 0, 0};
        MPI_Send(job, 5, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);
        std::vector<int> table = chainTableFor((w - 1) % bDim_);
        MPI_Send(table.data(), (int)table.size(), MPI_INT, w, TAG_ORDER,
                 MPI_COMM_WORLD);
    }
    reorders_ = reorders_ + 1;
}

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

        std::vector<int> table = chainTableFor(col);
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

// Zeroes every worker's counters. Sent just before the stretch that gets
// reported, so an untimed --loop pass or a previous --nprobes value cannot
// leak into it.
void MasterNode::resetCounters() {
    scanned_ = 0;
    scannedRow_.assign(bVec_, 0);
    for (int w = 1; w <= numWorkers_; ++w) {
        int job[5] = {JOB_RESET, 0, 0, 0, 0};
        MPI_Send(job, 5, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);
    }
}

// Asks every worker for its counters without ending it, which is what running
// several nprobe values against one distribution needs.
void MasterNode::collectStats() {
    aliveAfterStage_.assign(bDim_, 0);

    for (int w = 1; w <= numWorkers_; ++w) {
        int job[5] = {JOB_STATS, 0, 0, 0, 0};
        MPI_Send(job, 5, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);
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

void MasterNode::shutdown() {
    for (int w = 1; w <= numWorkers_; ++w) {
        int job[5] = {JOB_SHUTDOWN, 0, 0, 0, 0};
        MPI_Send(job, 5, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);
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

    // setprecision and fixed stay on the stream, and with --nprobes there is
    // another run's output after this table.
    std::ios_base::fmtflags flags = std::cout.flags();
    std::streamsize digits = std::cout.precision();

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

    std::cout.flags(flags);
    std::cout.precision(digits);
}

// Without this the heap starts empty, worst() is infinite, and nothing can be
// pruned until a whole block has been through the pipeline (Algorithm 1,
// lines 1-5). Samples come from the nearest clusters, not at random, which
// makes the starting threshold much tighter.
//
// Only the master can do this: distances here are over every dimension, and a
// worker holds a fraction of each vector.
//
// What it keeps is the threshold, not the candidates. They all sit in
// clusters the search is about to visit, so they come back on their own, and
// throwing them away saves the pipeline from having to remember which ones
// the heap already holds. This is what the sample does -- warmUpSearch, then
// init_result to empty the heap again.
void MasterNode::prewarmHeap(const float* query, QueryState& state, TopKHeap& heap) {
    // Spread over the nearest few clusters rather than taking everything from
    // the first one, as the authors' code does. One cluster can miss: the
    // nearest centroid is not always where the nearest vectors are, and a seed
    // drawn only from there gives a loose threshold exactly when it matters.
    int lists = cfg_.prewarmLists;
    if (lists > (int)state.clusters.size()) {
        lists = (int)state.clusters.size();
    }

    TopKHeap seed(heap.capacity());
    for (int i = 0; i < lists; ++i) {
        const std::vector<int>& ids = index_.clusterIds(state.clusters[i]);

        int n = (int)ids.size();
        if (cfg_.prewarm < n) {
            n = cfg_.prewarm;
        }
        for (int j = 0; j < n; ++j) {
            seed.push(ids[j], l2DistanceSquared(query, base_.vec(ids[j]),
                                                base_.getDim()));
        }
    }

    heap.seedThreshold(seed.worst());
}

// Candidates a block of queries contributes in one vector partition. This has
// to walk the queries and their probe lists in exactly the order the workers
// of that row will, since that order is the buffer layout and the master uses
// the total to size its receive.
long MasterNode::blockLoad(int row, int firstQ, int len,
                           const std::vector<QueryState>& batch) const {
    long total = 0;
    for (int j = 0; j < len; ++j) {
        const std::vector<int>& cl = batch[firstQ + j].clusters;
        for (int i = 0; i < (int)cl.size(); ++i) {
            if (clusterOwner_[cl[i]] == row) {
                total = total + index_.clusterSize(cl[i]);
            }
        }
    }
    return total;
}

// Hands one block of queries to every worker in a row and returns immediately.
// The workers of a row then run one after another, not in parallel: in
// parallel every slice would be computed in full and nothing saved (§3.2).
//
// Two small messages rather than three. The worker already has the batch's
// probe lists, so it works out for itself which of this block's queries want
// which of its clusters -- there is no list of participants to send.
void MasterNode::dispatchBlock(int row, int firstQ, int len, int item, int slot,
                               const std::vector<float>& thresholds) {
    for (int col = 0; col < bDim_; ++col) {
        int w = row * bDim_ + col + 1;
        int job[5] = {JOB_BLOCK, firstQ, len, item, slot};
        MPI_Send(job, 5, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);
        MPI_Send(thresholds.data(), len, MPI_FLOAT, w, TAG_THRESHOLD,
                 MPI_COMM_WORLD);
    }
}

int MasterNode::lastRankOf(int row, int item) const {
    // whoever the table puts at the end of this item's chain
    const std::vector<int>& chain = chainOrder_.chain(item);
    return row * bDim_ + chain[chain.size() - 1] + 1;
}

// Three levels of overlap: every query group is given work before anything is
// collected, each group holds one vector partition at a time (Fig. 5a), and
// within that partition all of its blocks are in flight at once (Fig. 5b,
// stages X, Y and Z). Groups refill on their own and MPI_Waitany takes
// whichever comes back first, so a slow group never holds up a fast one.
//
// Having several blocks open is what lets a worker hide the wait for one
// block's upstream behind another block's arithmetic; with one, measured
// compute was 6.8% of a worker's time in dimension mode, the rest blocked.
//
// Survivors enter their query's heap as soon as a block reports, so
// thresholds keep tightening -- sooner than Algorithm 1 line 18, which prunes
// strictly more and changes nothing else.
void MasterNode::vectorPipeline(const std::vector<std::vector<int>>& groupMembers,
                                const std::vector<QueryState>& batch,
                                std::vector<TopKHeap>& heaps) {
    struct Slot {
        int group;
        int row;
        int firstQ;
        int len;
        std::vector<Candidate> top;   // len rows of k, nearest first, padded
    };

    int k = cfg_.k;
    int blocks = cfg_.block;

    // Enough slots for every group to have all of its blocks out at once,
    // plus a spare so the search for a free one below cannot spin forever.
    int maxInFlight = bVec_ * blocks + 1;
    std::vector<Slot> slot(maxInFlight);
    std::vector<MPI_Request> req(maxInFlight, MPI_REQUEST_NULL);

    std::vector<int> stage(bVec_, 0);     // which partition group g is on
    std::vector<int> pos(bVec_, 0);       // which of its blocks goes next
    std::vector<int> inFlight(bVec_, 0);
    int free = 0;

    std::vector<float> thresholds;

    while (true) {
        for (int g = 0; g < bVec_; ++g) {
            while (stage[g] < bVec_) {
                int r = groupOrder_.chain(g)[stage[g]];

                if (pos[g] >= blocks) {
                    // Every block of this partition is out. Only once they
                    // have all come back are their distances in the heaps, and
                    // only then does the next partition start from a tighter
                    // threshold, which is the whole point of Fig. 5a.
                    if (inFlight[g] > 0) {
                        break;
                    }
                    stage[g] = stage[g] + 1;
                    pos[g] = 0;
                    continue;
                }

                const std::vector<int>& mem = groupMembers[g];
                int b = pos[g];
                pos[g] = pos[g] + 1;
                if (mem.empty()) {
                    continue;
                }

                // Groups are contiguous runs of the batch and so are their
                // blocks, which is why a job needs only a first and a length.
                int gStart = mem[0];
                int gLen = (int)mem.size();
                int firstQ = gStart + (int)((long)b * gLen / blocks);
                int endQ = gStart + (int)((long)(b + 1) * gLen / blocks);
                int len = endQ - firstQ;
                if (len <= 0) {
                    continue;
                }

                // Nothing of this partition to do for these queries.
                if (blockLoad(r, firstQ, len, batch) == 0) {
                    continue;
                }

                thresholds.resize(len);
                for (int j = 0; j < len; ++j) {
                    // an infinite threshold means nothing is dropped, the
                    // no-pruning arm of the ablation (paper Fig. 10)
                    thresholds[j] = cfg_.pruning ? heaps[firstQ + j].worst()
                                                 : PRUNED;
                }

                // The slot is picked first because its number is what tags
                // this block's messages all the way down the chain.
                while (req[free] != MPI_REQUEST_NULL) {
                    free = (free + 1) % maxInFlight;
                }

                // Different blocks enter the row at different columns, so no
                // worker is always the first stop -- the one that can prune
                // nothing (paper §4.3).
                int item = b % bDim_;
                dispatchBlock(r, firstQ, len, item, free, thresholds);

                slot[free].group = g;
                slot[free].row = r;
                slot[free].firstQ = firstQ;
                slot[free].len = len;
                slot[free].top.assign((size_t)len * k, Candidate{-1, PRUNED});

                // MPI: non-blocking, so the next block can be dispatched
                // without waiting for this one. lastRankOf is who ends the
                // chain; it sends back the k nearest per query rather than
                // every running total.
                MPI_Irecv(slot[free].top.data(),
                          (int)(slot[free].top.size() * sizeof(Candidate)),
                          MPI_BYTE, lastRankOf(r, item), tagTopk(free),
                          MPI_COMM_WORLD, &req[free]);
                inFlight[g] = inFlight[g] + 1;
            }
        }

        // MPI: block until any one block reports, whichever it is. This is
        // what lets a slow group not hold up a fast one.
        int index = MPI_UNDEFINED;
        MPI_Waitany(maxInFlight, req.data(), &index, MPI_STATUS_IGNORE);
        if (index == MPI_UNDEFINED) {
            break;
        }

        const Slot& s = slot[index];
        for (int j = 0; j < s.len; ++j) {
            const Candidate* top = &s.top[(size_t)j * k];
            TopKHeap& heap = heaps[s.firstQ + j];
            for (int t = 0; t < k; ++t) {
                if (top[t].dist >= PRUNED) {
                    break;   // the tail pads its unused slots, nearest first
                }
                heap.push(top[t].id, top[t].dist);
            }
        }

        scanned_ = scanned_ + blockLoad(s.row, s.firstQ, s.len, batch);
        scannedRow_[s.row] =
            scannedRow_[s.row] + blockLoad(s.row, s.firstQ, s.len, batch);

        inFlight[s.group] = inFlight[s.group] - 1;
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
        batch[q].clusters = probesFor(batch[q].id, nprobe);
        prewarmHeap(qv, batch[q], heaps[q]);
    }

    // Every worker gets the batch's query slices and its probe lists once.
    // The probe lists are what let a worker lay out a block's buffer the same
    // way the rest of its row does, so a job can name a block and nothing
    // more.
    std::vector<int> probes((size_t)count * nprobe, 0);
    for (int q = 0; q < count; ++q) {
        const std::vector<int>& cl = batch[q].clusters;
        for (int i = 0; i < (int)cl.size(); ++i) {
            probes[(size_t)q * nprobe + i] = cl[i];
        }
        // a query with fewer than nprobe clusters pads with its last one,
        // which the workers skip as a repeat
        for (int i = (int)cl.size(); i < nprobe; ++i) {
            probes[(size_t)q * nprobe + i] = -1;
        }
    }

    for (int w = 1; w <= numWorkers_; ++w) {
        int job[5] = {JOB_QUERY, count, nprobe, 0, 0};
        MPI_Send(job, 5, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);

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
        MPI_Send(probes.data(), (int)probes.size(), MPI_INT, w,
                 TAG_PROBES, MPI_COMM_WORLD);
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

    vectorPipeline(groupMembers, batch, heaps);

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

    // The layout has to be settled before any data moves, so the profiling
    // pass comes first (paper Fig. 3, step 1). It runs whatever the mode is:
    // the cost model is one consumer of the cluster hit counts, splitGrid is
    // the other, and it only costs one centroid scan per profiled query.
    warmupPlan(cfg_.warmup, cfg_.nprobe);
    if (cfg_.costModel) {
        choosePlan();
    }

    splitGrid(cfg_.bVec, cfg_.bDim);
    distributeData();

    // The distributed answer must equal what one machine would have returned.
    // Compared as sets, not position by position: squared distances on SIFT
    // are integers, ties near rank k are common, and their order is not
    // defined either way.
    int k = cfg_.k;
    int nq = cfg_.nq;
    if (nq > query_.getN()) {
        std::cout << "only " << query_.getN() << " queries in the file, running those"
                  << std::endl;
        nq = query_.getN();
    }

    // --nprobes runs several values one after another. They share the index
    // and the distribution: nprobe only decides which clusters the master
    // dispatches, and nothing a worker holds depends on it. The grid was
    // already fixed above, by --mode or the cost model.
    std::vector<int> sweep = cfg_.nprobes;
    if (sweep.empty()) {
        sweep.push_back(cfg_.nprobe);
    }

    for (size_t ni = 0; ni < sweep.size(); ++ni) {
    int nprobe = sweep[ni];
    if (sweep.size() > 1) {
        std::cout << "\n########## nprobe " << nprobe << "  ("
                  << (ni + 1) << " of " << sweep.size() << ") ##########"
                  << std::endl;
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

    recallSum = 0.0;
    differing = 0;
    ties = 0;

    // Every pass, not just the counted one: the counters are sized here as
    // well as zeroed, and the warm-up pass reads them too.
    resetCounters();

    for (int start = 0; start < nq; start += cfg_.batch) {
        int count = (start + cfg_.batch <= nq) ? cfg_.batch : (nq - start);

        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::vector<Candidate>> spread = queryPipeline(start, count, nprobe, k);
        auto t1 = std::chrono::steady_clock::now();
        if (!warmup) {
            seconds = seconds + std::chrono::duration<double>(t1 - t0).count();
        }

        // Between batches, with the chains empty. Outside the timed section:
        // it measures and re-plans, which is not the search.
        reorderChains();

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
            std::vector<Candidate> single =
                index_.search(base_, query_.vec(q), probesFor(q, nprobe), k);

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

    collectStats();   // the workers keep running, the next nprobe needs them

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
    if (cfg_.reorder) {
        std::cout << "chain reordered " << reorders_ << " time(s)" << std::endl;
    }
    writeCsv(nprobe, nq, recallSum / nq, seconds, differing, ties);
    }   // end of the nprobe sweep

    shutdown();
    return 0;
}

// One row per run, appended, header written when the file is new. Everything
// that was varied over a set of runs has to be in the row, or the rows cannot
// be told apart later.
void MasterNode::writeCsv(int nprobe, int nq, double recall, double seconds,
                          int differing, int ties) const {
    if (cfg_.csv.empty()) {
        return;
    }

    bool isNew = true;
    std::FILE* probe = std::fopen(cfg_.csv.c_str(), "rb");
    if (probe != nullptr) {
        std::fseek(probe, 0, SEEK_END);
        isNew = (std::ftell(probe) == 0);
        std::fclose(probe);
    }

    std::FILE* f = std::fopen(cfg_.csv.c_str(), "ab");
    if (f == nullptr) {
        std::cout << "could not write " << cfg_.csv << std::endl;
        return;
    }

    if (isNew) {
        std::fprintf(f, "data,nlist,iters,trainpoints,workers,bvec,bdim,mode,"
                        "batch,threads,prewarm,prewarmlists,pruning,mkl,loop,"
                        "nprobe,k,nq,recall,qps,ms_per_query,differing,ties,"
                        "scanned,work_pct\n");
    }

    long done = 0;
    for (int s = 0; s < bDim_; ++s) {
        done = done + ((s == 0) ? scanned_ : aliveAfterStage_[s - 1]);
    }
    double work = (scanned_ > 0)
                ? (100.0 * done / (double)(scanned_ * bDim_)) : 0.0;

    std::fprintf(f,
        "%s,%d,%d,%d,%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,"
        "%d,%d,%d,%.6f,%.3f,%.4f,%d,%d,%ld,%.4f\n",
        cfg_.data.c_str(), cfg_.nlist, cfg_.iters, cfg_.trainPoints,
        numWorkers_, bVec_, bDim_, cfg_.mode.c_str(),
        cfg_.batch, cfg_.threads, cfg_.prewarm, cfg_.prewarmLists,
        cfg_.pruning ? 1 : 0, cfg_.mkl ? 1 : 0, cfg_.loop,
        nprobe, cfg_.k, nq, recall, nq / seconds, 1000.0 * seconds / nq,
        differing, ties, scanned_, work);

    std::fclose(f);
    std::cout << "appended to " << cfg_.csv << std::endl;
}

}  // namespace harmony
