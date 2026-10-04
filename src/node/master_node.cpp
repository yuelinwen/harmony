#include "master_node.h"

#include <algorithm>
#include <cmath>
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
    Stopwatch watch;

    if (cfg_.cache && index_.load(path, base_.getN(), base_.getDim())) {
        std::cout << "loaded index from " << path << " ("
                  << watch.seconds() << " s)" << std::endl;
        buildSizePrefix();
        return;
    }

    std::cout << "building index (nlist=" << nlist << ")" << std::endl;
    index_.build(base_, nlist, iterations, cfg_.trainPoints);
    std::cout << "build time: " << watch.seconds() << " s" << std::endl;

    buildSizePrefix();

    if (cfg_.cache) {
        if (index_.save(path)) {
            std::cout << "saved index to " << path << std::endl;
        } else {
            std::cout << "could not write " << path << ", carrying on" << std::endl;
        }
    }
}

// The pools --skew draws from, and their size prefix sums.
//
// The hot eighth is itself drawn proportional to size rather than taken as the
// first eighth of the cluster ids. Two reasons. Cluster ids come out of the
// kmeans initialisation order and mean nothing, so a prefix of them is an
// arbitrary set. And at a high --skew a query probes the whole hot set, which
// leaves no sampling freedom at all -- the candidate count is then just the
// hot set's total size, so that set has to be representative or the workload
// changes weight rather than only shape.
void MasterNode::buildSizePrefix() {
    int nlist = index_.getNlist();

    allIds_.resize(nlist);
    for (int c = 0; c < nlist; ++c) {
        allIds_[c] = c;
    }
    allPrefix_.assign(nlist + 1, 0.0);
    for (int c = 0; c < nlist; ++c) {
        allPrefix_[c + 1] = allPrefix_[c] + (double)index_.clusterSize(c);
    }

    int hot = nlist / 8;
    if (hot < 1) {
        hot = 1;
    }
    hotIds_.clear();
    std::vector<char> taken(nlist, 0);
    unsigned int state = 987654321u;          // fixed: the hot set is the same every run
    for (int tries = 0; (int)hotIds_.size() < hot && tries < 64 * hot; ++tries) {
        int c = drawWeighted(allIds_, allPrefix_, state);
        if (!taken[c]) {
            taken[c] = 1;
            hotIds_.push_back(c);
        }
    }
    for (int c = 0; (int)hotIds_.size() < hot && c < nlist; ++c) {
        if (!taken[c]) {
            taken[c] = 1;
            hotIds_.push_back(c);
        }
    }

    hotPrefix_.assign(hotIds_.size() + 1, 0.0);
    for (size_t i = 0; i < hotIds_.size(); ++i) {
        hotPrefix_[i + 1] = hotPrefix_[i] + (double)index_.clusterSize(hotIds_[i]);
    }

    // The test workload's hot set. Keeps the first (1 - shift) of the layout's
    // hot clusters and draws the rest from outside it, size-weighted like the
    // original so the candidate count stays comparable.
    //
    // A prefix of hotIds_ is kept rather than a random subset because the list
    // came out of a weighted draw already, so its order carries no meaning
    // worth preserving and a prefix makes the overlap exactly countable.
    testHotIds_.assign(hotIds_.begin(), hotIds_.end());
    int move = (int)(cfg_.skewShift * hot + 0.5);
    if (move > 0) {
        // A seed of its own, so which clusters move does not depend on the
        // shift being used -- raising --skewshift only ever moves more of
        // them, it does not reshuffle the ones already moved.
        unsigned int shiftState = 192837465u;
        for (int i = hot - move; i < hot; ++i) {
            int c = -1;
            for (int tries = 0; tries < 64 * nlist; ++tries) {
                int pick = drawWeighted(allIds_, allPrefix_, shiftState);
                if (!taken[pick]) {
                    c = pick;
                    break;
                }
            }
            if (c < 0) {
                // the pool is exhausted: fall back to a linear scan
                for (int pick = 0; pick < nlist; ++pick) {
                    if (!taken[pick]) {
                        c = pick;
                        break;
                    }
                }
            }
            if (c < 0) {
                break;      // every cluster is taken; nothing left to move to
            }
            taken[c] = 1;
            testHotIds_[i] = c;
        }
    }

    testHotPrefix_.assign(testHotIds_.size() + 1, 0.0);
    for (size_t i = 0; i < testHotIds_.size(); ++i) {
        testHotPrefix_[i + 1] =
            testHotPrefix_[i] + (double)index_.clusterSize(testHotIds_[i]);
    }
}

// One cluster out of a pool, with probability proportional to how many vectors
// it holds.
//
// Uniform draws would make the synthetic workload lighter than a real one:
// a query's nearest clusters are biased towards the big ones, since a bigger
// cluster covers more space and is more often the nearest. Measured on sift1M
// that bias is 16% of the candidates, which would show up as a throughput
// gain and have nothing to do with the skew being studied.
int MasterNode::drawWeighted(const std::vector<int>& ids,
                             const std::vector<double>& prefix,
                             unsigned int& state) const {
    state = state * 1103515245u + 12345u;
    double u = (double)((state >> 8) & 0xFFFFFFu) / 16777216.0;
    double target = u * prefix[prefix.size() - 1];

    size_t i = (size_t)(std::upper_bound(prefix.begin(), prefix.end(), target)
                        - prefix.begin());
    if (i == 0) {
        i = 1;
    }
    if (i > ids.size()) {
        i = ids.size();
    }
    return ids[i - 1];
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
std::vector<int> MasterNode::probesFor(int queryId, int nprobe,
                                      Workload which) const {
    bool test = (which == kTestWorkload);
    double skew = test ? cfg_.skew : cfg_.indexSkew;
    if (skew <= 0.0) {
        return index_.nearestClusters(query_.vec(queryId), nprobe);
    }

    int nlist = index_.getNlist();
    if (nprobe > nlist) {
        nprobe = nlist;
    }
    const std::vector<int>& hotPool = test ? testHotIds_ : hotIds_;
    const std::vector<double>& hotPoolPrefix = test ? testHotPrefix_ : hotPrefix_;
    int hot = (int)hotPool.size();

    int wanted = (int)(skew * nprobe + 0.5);   // probes aimed at the hot set
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
        const std::vector<int>& pool = (pass == 0) ? hotPool : allIds_;
        const std::vector<double>& prefix = (pass == 0) ? hotPoolPrefix : allPrefix_;

        // Rejection sampling, then a linear walk over the pool, so a pool that
        // is nearly exhausted still terminates rather than spinning.
        int tries = 0;
        while ((int)out.size() < want && tries < 8 * nprobe) {
            int c = drawWeighted(pool, prefix, state);
            if (!taken[c]) {
                taken[c] = 1;
                out.push_back(c);
            }
            tries = tries + 1;
        }
        for (size_t i = 0; (int)out.size() < want && i < pool.size(); ++i) {
            if (!taken[pool[i]]) {
                taken[pool[i]] = 1;
                out.push_back(pool[i]);
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
    std::vector<std::vector<int>> probes(n);
    for (int q = 0; q < n; ++q) {
        probes[q] = probesFor(q, nprobe, kIndexWorkload);
        for (int i = 0; i < (int)probes[q].size(); ++i) {
            clusterHits_[probes[q][i]] = clusterHits_[probes[q][i]] + 1;
        }
    }

    // How lopsided the profiling workload was. Same measure as the one
    // reported for the test workload (src/test/metrics.h) -- this one is the
    // distribution the layout was built from, that one is what was then
    // actually searched.
    //
    // Neither can be matched to the paper's "variance = 500". The number
    // carries units of probe counts, so it scales with how many queries were
    // profiled: the most it can reach is mean * sqrt((nlist - hot) / hot) with
    // mean = n * nprobe / nlist, which here tops out near 330. Fig. 7 states
    // none of n, nprobe or nlist, so there is nothing to match against. The
    // ratio to the mean is printed alongside because it is scale-free, and is
    // the figure worth comparing between runs.
    double mean = (double)n * nprobe / index_.getNlist();
    double var = workloadVariance(probes, index_.getNlist());

    std::cout << "warmup: " << n << " queries profiled at skew "
              << cfg_.indexSkew << ", cluster probe counts"
              << " mean " << mean << " variance " << var
              << " (" << (mean > 0.0 ? var / mean : 0.0) << " x mean)" << std::endl;
    if (cfg_.skew > 0.0 || cfg_.indexSkew > 0.0) {
        std::cout << "SYNTHETIC WORKLOAD (--indexskew " << cfg_.indexSkew
                  << " --skew " << cfg_.skew << "): probe lists are made up,"
                  << " recall is not meaningful" << std::endl;
        // How much of the layout's hot set the test workload still wants.
        // Printed rather than inferred from --skewshift, because it is the
        // number that decides whether anything can mismatch at all.
        int shared = 0;
        for (size_t i = 0; i < testHotIds_.size(); ++i) {
            for (size_t j = 0; j < hotIds_.size(); ++j) {
                if (testHotIds_[i] == hotIds_[j]) {
                    shared = shared + 1;
                    break;
                }
            }
        }
        std::cout << "  hot set: " << hotIds_.size() << " clusters, "
                  << shared << " of them still hot at test time"
                  << " (--skewshift " << cfg_.skewShift << ")" << std::endl;
        if (shared == (int)hotIds_.size()) {
            std::cout << "  the layout is built for exactly the clusters the"
                      << " queries want, and LPT spreads them, so the load"
                      << " stays balanced -- Fig. 8 needs --skewshift above 0"
                      << std::endl;
        }
    }
}

long MasterNode::clusterHitsOf(int c) const {
    long hits = clusterHits_.empty() ? 1 : clusterHits_[c];
    return (hits < 1) ? 1 : hits;
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
        order[c] = std::make_pair((double)clusterHitsOf(c) * index_.clusterSize(c), c);
    }

    // The arm to measure the greedy against: deal the clusters out in id
    // order, taking no notice of how big or how popular any of them is.
    if (cfg_.assign == "roundrobin") {
        std::vector<double> load(bVec, 0.0);
        for (int c = 0; c < nlist; ++c) {
            if (owner != nullptr) {
                (*owner)[c] = c % bVec;
            }
            load[c % bVec] = load[c % bVec] + order[c].first;
        }
        return load;
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
    // Measured on the cluster (Sift1M, nlist 256 / nprobe 32, gemm on), as
    // the "distance work vs no pruning" line reports it. Not monotonic: at
    // eight slices each one is only 16 dimensions wide, too little distance
    // per check to beat four.
    static const int slices[] = {1, 2, 4, 8};
    static const double work[] = {1.00, 0.76, 0.67, 0.69};
    int n = 4;

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
        double candidates = (double)clusterHitsOf(c) * index_.clusterSize(c);

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

    // The grid resolveGrid() derived --block from was only a starting point,
    // so derive it again now that this one is final. Skipped when the user
    // gave the flag, and a no-op when the cost model kept the starting grid.
    if (!cfg_.blockGiven) {
        cfg_.block = defaultBlock(cfg_, bestDim);
    }
}

// The worker with rank w sits at row (w-1)/bDim, column (w-1)%bDim of the
// grid. Rows are vector partitions, columns are dimension slices.
//
// Clusters go to rows by weight -- partitionLoads(), longest-processing-time
// first. Dimensions are cut evenly across a row, which is what the paper does
// on a homogeneous cluster (Section 4.2).
void MasterNode::splitGrid(int bVec, int bDim) {
    bVec_ = bVec;
    bDim_ = bDim;
    plan_ = SlicePlan{base_.getDim(), bDim};
    chainOrder_ = SearchOrder(bDim_, bDim_, true);
    groupOrder_ = SearchOrder(bVec_, bVec_, true);

    // This is the assignment the cost model's I(pi) measures -- the paper's pi
    // is the partition plan itself (§4.2.1), not just the grid shape.
    int nlist = index_.getNlist();
    partitionLoads(bVec_, &clusterOwner_);

    // The chain tags are derived from the master's in-flight slots, so the
    // largest one this layout can produce is fixed here. Nothing near MPI's
    // ceiling is reachable at any worker count that fits on a cluster, but a
    // tag MPI rejects fails at the receive rather than at the send, which is
    // an unpleasant way to find out.
    if (!tagsFitMpi(bVec_ * bVec_ * cfg_.block)) {
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
    if (bDim_ < 2) {
        return;      // a chain of one has no order to change
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

// What a worker sizes its outgoing buffer pool from. A row serves one group
// at a time, so a worker has that group's blocks open; without the
// vector-level barrier a group holds every partition at once, which squares
// the budget. The spare is slack, not a requirement.
int MasterNode::maxBlocksInFlight() const {
    int perGroup = cfg_.pruneVector ? cfg_.block : (bVec_ * cfg_.block);
    return bVec_ * perGroup + 1;
}

// Each cluster goes only to the workers in its row, and each of those gets
// only its own slice of the dimensions -- so a worker holds 1/bVec of the
// vectors x 1/bDim of the dimensions, one block of the paper's grid.
//
// The master does the cutting and sends only the slice, so no worker ever
// holds data it does not own.
void MasterNode::distributeData() {
    std::cout << "\n===== 4. distribute =====" << std::endl;
    Stopwatch watch;

    // how many clusters each row will receive
    std::vector<int> rowClusters(bVec_, 0);
    for (int c = 0; c < index_.getNlist(); ++c) {
        rowClusters[clusterOwner_[c]] = rowClusters[clusterOwner_[c]] + 1;
    }

    for (int w = 1; w <= numWorkers_; ++w) {
        int row = (w - 1) / bDim_;
        int col = (w - 1) % bDim_;
        int setup[9];
        setup[0] = plan_.end(col) - plan_.begin(col);
        setup[1] = rowClusters[row];
        setup[2] = bDim_;
        setup[3] = cfg_.batch;
        setup[4] = maxBlocksInFlight();
        setup[5] = bVec_;
        setup[6] = cfg_.block;
        setup[7] = cfg_.pipeline ? 1 : 0;
        setup[8] = cfg_.pruneVector ? 1 : 0;
        // MPI: blocking is fine for startup -- the order is fixed and nobody waits
        MPI_Send(setup, 9, MPI_INT, w, TAG_SETUP, MPI_COMM_WORLD);

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

    distributeSeconds_ = watch.seconds();
    std::cout << "distributed " << index_.getNlist() << " clusters over the grid"
              << " (" << distributeSeconds_ << " s)" << std::endl;
}



// Zeroes every worker's counters. Sent just before the stretch that gets
// reported, so an untimed --loop pass or a previous --nprobes value cannot
// leak into it.
void MasterNode::resetCounters() {
    scanned_ = 0;
    scannedRow_.assign(bVec_, 0);
    masterTotal_ = 0.0;
    stage0Seconds_ = 0.0;
    broadcastSeconds_ = 0.0;
    dispatchSeconds_ = 0.0;
    waitSeconds_ = 0.0;
    mergeSeconds_ = 0.0;
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
    workerTimes_.assign(numWorkers_, std::vector<double>(WORKER_TIMES, 0.0));
    for (int w = 1; w <= numWorkers_; ++w) {
        MPI_Recv(perWorker.data(), bDim_, MPI_LONG, w, TAG_STATS,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int s = 0; s < bDim_; ++s) {
            aliveAfterStage_[s] = aliveAfterStage_[s] + perWorker[s];
        }
        MPI_Recv(workerTimes_[w - 1].data(), WORKER_TIMES, MPI_DOUBLE, w, TAG_TIMES,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// The same index on one machine. This is the paper's Faiss column, except
// that it is this program's own single-machine form rather than another
// implementation's, so the comparison is only about the partitioning.
long MasterNode::singleMachineMemory() const {
    long n = base_.getN();
    long dim = base_.getDim();
    return n * dim * (long)sizeof(float)
         + n * (long)sizeof(int)
         + (long)index_.getNlist() * dim * (long)sizeof(float);
}

void MasterNode::shutdown() {
    for (int w = 1; w <= numWorkers_; ++w) {
        int job[5] = {JOB_SHUTDOWN, 0, 0, 0, 0};
        MPI_Send(job, 5, MPI_INT, w, TAG_JOB, MPI_COMM_WORLD);
    }
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
    // How many of the query's nearest clusters the seed is drawn from. One by
    // default, against the ten the authors' code uses: measured, spreading
    // the same budget wider leaves a looser threshold. See --prewarmlists.
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

// Starts a row on the query group that is due there, and returns. The
// workers of a row then run one after another, not in parallel: in parallel
// every slice would be computed in full and nothing saved (§3.2).
//
// One message per worker, holding nothing but the thresholds. Which group
// this is, where it starts, how it splits into blocks, who is first in each
// block's chain and what tag it carries are all things the worker works out
// for itself, from the group order, the batch's probe lists and the chain
// table it was given at setup -- see comm/messages.h. That is why the message
// count stops growing with --block, and why it no longer carries a schedule
// at all.
void MasterNode::sendThresholds(int row, int gStart, int gLen,
                                const std::vector<TopKHeap>& heaps) {
    Stopwatch watch;
    std::vector<float> t(gLen);
    for (int j = 0; j < gLen; ++j) {
        // an infinite threshold drops nothing, which is the no-dimension-level
        // arm of --pruning (paper Fig. 10)
        t[j] = cfg_.pruneDim ? heaps[gStart + j].worst() : PRUNED;
    }
    for (int col = 0; col < bDim_; ++col) {
        int w = row * bDim_ + col + 1;
        MPI_Send(t.data(), gLen, MPI_FLOAT, w, TAG_THRESHOLD, MPI_COMM_WORLD);
    }
    dispatchSeconds_ = dispatchSeconds_ + watch.seconds();
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
        int row;
        int firstQ;
        int len;
        long load;                    // candidates this block offers, counted once
        std::vector<Candidate> top;   // len rows of k, nearest first, padded
    };

    int k = cfg_.k;
    int blocks = cfg_.block;

    // One receive per block the batch can produce: group g, stage s, its b-th
    // query block. Named by blockTag(), which the workers derive too, so
    // nobody has to be told a block's tag -- which is what lets a batch run
    // without the master sending a schedule at all.
    int tagSpace = bVec_ * bVec_ * blocks;
    if (!tagsFitMpi(tagSpace)) {
        std::cerr << "chain tags exceed this MPI's maximum at " << tagSpace
                  << " blocks; lower --block" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<Slot> slot(tagSpace);
    std::vector<MPI_Request> req(tagSpace, MPI_REQUEST_NULL);

    int inFlight = 0;   // blocks dispatched and not yet reported

    // Take whichever block reports next and fold it into its queries' heaps.
    // Survivors enter as soon as a block comes back, so thresholds keep
    // tightening -- sooner than Algorithm 1 line 18, which prunes strictly
    // more and changes nothing else.
    auto collect = [&]() {
        Stopwatch watch;
        int index = MPI_UNDEFINED;
        MPI_Waitany(tagSpace, req.data(), &index, MPI_STATUS_IGNORE);
        waitSeconds_ = waitSeconds_ + watch.seconds(true);
        if (index == MPI_UNDEFINED) {
            inFlight = 0;      // nothing outstanding; cannot happen mid-stage
            return;
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
        mergeSeconds_ = mergeSeconds_ + watch.seconds();

        scanned_ = scanned_ + s.load;
        scannedRow_[s.row] = scannedRow_[s.row] + s.load;
        inFlight = inFlight - 1;
    };

    // Fig. 5a, in lockstep: every group moves to its next vector partition at
    // the same time, so at stage s the mapping group -> row is a permutation
    // and each row has exactly one group. That is what lets a worker know
    // which group reaches it without being told -- it counts stages and reads
    // the same rotation.
    for (int stage = 0; stage < bVec_; ++stage) {
        for (int g = 0; g < bVec_; ++g) {
            const std::vector<int>& mem = groupMembers[g];
            if (mem.empty()) {
                continue;      // fewer queries in the batch than groups
            }

            int r = groupOrder_.chain(g)[stage];
            int gStart = mem[0];
            int gLen = (int)mem.size();

            sendThresholds(r, gStart, gLen, heaps);

            // A receive per block that has anything in this partition. The
            // worker cuts the group up the same way and walks the same probe
            // lists, so a block that comes out empty here comes out empty
            // there and exists on neither side.
            for (int b = 0; b < blocks; ++b) {
                int firstQ = gStart + (int)((long)b * gLen / blocks);
                int endQ = gStart + (int)((long)(b + 1) * gLen / blocks);
                int len = endQ - firstQ;
                if (len <= 0) {
                    continue;
                }

                // Walking the queries and their probe lists is not free, so it
                // happens once here and the result rides along in the slot.
                long load = blockLoad(r, firstQ, len, batch);
                if (load == 0) {
                    continue;   // nothing of this partition for these queries
                }

                int item = b % bDim_;
                int id = blockTag(g, stage, b, bVec_, blocks);

                slot[id].row = r;
                slot[id].firstQ = firstQ;
                slot[id].len = len;
                slot[id].load = load;
                slot[id].top.assign((size_t)len * k, Candidate{-1, PRUNED});

                // MPI: non-blocking, so the rest of the stage can go out
                // without waiting for this one. lastRankOf is who ends the
                // chain; it sends back the k nearest per query rather than
                // every running total.
                MPI_Irecv(slot[id].top.data(),
                          (int)(slot[id].top.size() * sizeof(Candidate)),
                          MPI_BYTE, lastRankOf(r, item), tagTopk(id),
                          MPI_COMM_WORLD, &req[id]);
                inFlight = inFlight + 1;
            }

            // --disablepipeline: one group's blocks anywhere in the system at
            // a time, and the worker takes them one at a time inside that --
            // so exactly one block is ever open. Fig. 10's arm without the
            // overlap.
            while (!cfg_.pipeline && inFlight > 0) {
                collect();
            }
        }

        // Vector-level pruning: the stage has to finish before the next
        // begins, or the next would not start from a threshold this one
        // tightened. Without it every stage goes out first and nothing is
        // drained until the end, which is the arm --disablevectorpruning
        // measures.
        while (cfg_.pruneVector && inFlight > 0) {
            collect();
        }
    }

    while (inFlight > 0) {
        collect();
    }
}

std::vector<std::vector<Candidate>> MasterNode::queryPipeline(int firstQuery, int count,
                                                              int nprobe, int k) {
    // Stage 0: work out what each query needs, and prewarm its heap
    // (Algorithm 1 line 20).
    std::vector<QueryState> batch(count);
    std::vector<TopKHeap> heaps(count, TopKHeap(k));

    Stopwatch phase;

    // OpenMP: queries are independent here. Each iteration writes only its own
    // batch[q] and heaps[q], and everything it reads -- the index, the base
    // and query vectors -- is const, so the values come out the same whatever
    // the thread count. The loop's implicit barrier still separates this from
    // the dispatch below, which needs every threshold before it sends any.
    //
    // Serial, this cost 32% of throughput at 8x1: one core walking the batch
    // while eight workers wait for it, and the paper counts centroid
    // assignment as an ordinary ANNS cost rather than one of Harmony's
    // (§4.2.2). The sample parallelises both halves the same way
    // (Index.cpp, findNearNprobeOfCentroidIds and warmUpSearch).
    //
    // dynamic, because prewarmHeap's cost follows the size of the cluster its
    // query lands in.
    #pragma omp parallel for schedule(dynamic)
    for (int q = 0; q < count; ++q) {
        const float* qv = query_.vec(firstQuery + q);
        batch[q].id = firstQuery + q;
        batch[q].clusters = probesFor(batch[q].id, nprobe, kTestWorkload);
        prewarmHeap(qv, batch[q], heaps[q]);
    }
    stage0Seconds_ = stage0Seconds_ + phase.seconds(true);

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
        // A query with fewer than nprobe clusters pads with -1, which
        // blockOf() rejects. It must not pad with a real cluster id: the
        // workers derive a block's buffer layout by walking this list, so a
        // repeat would reserve the same run of totals twice.
        for (int i = (int)cl.size(); i < nprobe; ++i) {
            probes[(size_t)q * nprobe + i] = -1;
        }
    }

    // No thresholds go out with the batch. Every stage opens by sending the
    // group that reaches that row its own, and a worker computes nothing
    // until a stage has opened -- so a whole-batch copy here would be
    // overwritten before it could be read (paper §5).
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
    broadcastSeconds_ = broadcastSeconds_ + phase.seconds(true);

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
    wall_.reset();

    // Printed before anything can go wrong, so a run that dies half way still
    // says which binary died. The same three values go into every CSV row.
    std::cout << "run " << timestampNow()
              << "  build " << buildId()
              << "  host " << hostName() << std::endl;

    std::cout << "\n===== 1. data =====" << std::endl;

    if (!loadData(cfg_.data + "_base.bin", cfg_.data + "_query.bin") ||
        !truth_.loadIds(cfg_.data + "_gt.bin")) {
        return 1;
    }
    truth_.loadDistances(cfg_.data + "_gtd.bin");
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

    Agreement agree;

    // Only the distributed search is timed. The single-machine pass below is
    // both the baseline it is compared against and the answer key --check
    // uses, and is timed separately.
    double seconds = 0.0;
    double recallSum = 0.0;
    double r2Sum = 0.0;

    // Every probe list of this nprobe, worked out once. The search, the
    // single-machine reference and the workload statistic all read this same
    // set, which is what keeps `differing` meaningful under a synthetic
    // workload.
    std::vector<std::vector<int>> probes(nq);
    for (int q = 0; q < nq; ++q) {
        probes[q] = probesFor(q, nprobe, kTestWorkload);
    }

    // Before the search rather than inside it: the probe lists depend on
    // nprobe but not on anything the search does, so this can be computed
    // once, and doing it here keeps it out of the workers' idle time.
    double single = 0.0;
    if (cfg_.baseline || cfg_.check) {
        single = referencePass(index_, base_, query_, probes, k, cfg_.loop,
                              reference_);
    }

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
    r2Sum = 0.0;
    agree = Agreement();

    // Every pass, not just the counted one: the counters are sized here as
    // well as zeroed, and the warm-up pass reads them too.
    resetCounters();

    for (int start = 0; start < nq; start += cfg_.batch) {
        int count = (start + cfg_.batch <= nq) ? cfg_.batch : (nq - start);

        Stopwatch batchWatch;
        std::vector<std::vector<Candidate>> spread = queryPipeline(start, count, nprobe, k);
        double took = batchWatch.seconds();
        masterTotal_ = masterTotal_ + took;
        if (!warmup) {
            seconds = seconds + took;
        }

        // Between batches, with the chains empty. Outside the timed section:
        // it measures and re-plans, which is not the search.
        reorderChains();

        if (!last) {
            continue;   // intermediate passes are only there to be timed
        }

        for (int j = 0; j < count; ++j) {
            int q = start + j;
            recallSum = recallSum + truth_.recallAt(q, spread[j], k);
            r2Sum = r2Sum + truth_.r2Of(q, spread[j], k);

            // This reference costs more than the search it checks (one
            // machine, no pruning), and over a thousand queries it is most of
            // the wall clock, which is why it can be turned off.
            if (!cfg_.check) {
                continue;
            }
            compare(spread[j], reference_[q], k, &agree);
        }
    }
    }

    if (cfg_.loop > 1) {
        seconds = seconds / cfg_.loop;
    }

    collectStats();   // the workers keep running, the next nprobe needs them

    // Everything measured, handed to the one place that knows how to say it.
    Metrics m;
    m.cfg = cfg_;
    m.workers = numWorkers_;
    m.bVec = bVec_;
    m.bDim = bDim_;
    m.nlist = index_.getNlist();
    m.nprobe = nprobe;
    m.k = k;
    m.nq = nq;
    m.baseCount = base_.getN();

    m.differing = agree.differing;
    m.ties = agree.ties;
    m.recall = recallSum / nq;
    m.r2 = r2Sum / nq;
    m.haveR2 = truth_.haveDistances();

    m.seconds = seconds;
    m.single = single;
    m.elapsed = wall_.seconds();
    m.variance = workloadVariance(probes, index_.getNlist());
    m.trainSeconds = index_.trainSeconds();
    m.addSeconds = index_.addSeconds();
    m.distributeSeconds = distributeSeconds_;

    m.workerBytes = 0;
    for (int w = 0; w < (int)workerTimes_.size(); ++w) {
        m.workerBytes = m.workerBytes + (long)workerTimes_[w][9];
    }
    m.singleBytes = singleMachineMemory();

    m.scanned = scanned_;
    m.scannedRow = scannedRow_;
    m.aliveAfterStage = aliveAfterStage_;
    m.workerTimes = workerTimes_;
    m.reorders = reorders_;

    m.masterTotal = masterTotal_;
    m.masterStage0 = stage0Seconds_;
    m.masterBroadcast = broadcastSeconds_;
    m.masterDispatch = dispatchSeconds_;
    m.masterWait = waitSeconds_;
    m.masterMerge = mergeSeconds_;

    m.print();
    m.writeCsv();
    }   // end of the nprobe sweep

    shutdown();
    return 0;
}



}  // namespace harmony
