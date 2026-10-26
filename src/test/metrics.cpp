#include "metrics.h"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <ios>
#include <iostream>

#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// The fallback is for a build that went around scripts/build.sh.
#ifndef HARMONY_COMMIT
#define HARMONY_COMMIT "unknown"
#endif

namespace harmony {

const char* buildId() {
    return HARMONY_COMMIT;
}

std::string timestampNow() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return std::string(buf);
}

std::string hostName() {
    char buf[256];
    if (gethostname(buf, sizeof(buf)) != 0) {
        return "unknown";
    }
    buf[sizeof(buf) - 1] = '\0';
    return std::string(buf);
}

double workloadVariance(const std::vector<std::vector<int>>& probes, int nlist) {
    if (nlist <= 0 || probes.empty()) {
        return 0.0;
    }

    std::vector<long> hits(nlist, 0);
    long total = 0;
    for (size_t q = 0; q < probes.size(); ++q) {
        for (size_t i = 0; i < probes[q].size(); ++i) {
            hits[probes[q][i]] = hits[probes[q][i]] + 1;
            total = total + 1;
        }
    }

    double mean = (double)total / nlist;
    double var = 0.0;
    for (int c = 0; c < nlist; ++c) {
        double d = (double)hits[c] - mean;
        var = var + d * d;
    }
    return std::sqrt(var / nlist);
}

std::string Metrics::pruningLabel() const {
    if (cfg.pruneDim && cfg.pruneVector) {
        return "both";
    }
    if (cfg.pruneDim) {
        return "dim";
    }
    if (cfg.pruneVector) {
        return "vector";
    }
    return "none";
}

double Metrics::workPercent() const {
    if (scanned <= 0 || bDim <= 0) {
        return 0.0;
    }
    long done = 0;
    for (int s = 0; s < bDim; ++s) {
        done = done + ((s == 0) ? scanned : aliveAfterStage[s - 1]);
    }
    return 100.0 * done / (double)(scanned * bDim);
}

// communication   recv + send -- blocked moving partial sums along a chain.
//                 This is the sample's waitTime (node.cpp:275).
// computation     accumulate() and the top-k pick at a chain tail.
// other           everything else: waiting for the master to dispatch,
//                 opening a block, and polling.
//
// "other" is large here and small in the paper, and the reason is in the
// design rather than in the measurement: this master dispatches at run time,
// so a worker waits on it, while the sample's schedule is fixed at setup and
// its workers never wait for work at all.
void Metrics::timeBreakdown(double* comm, double* compute,
                            double* other) const {
    *comm = 0.0;
    *compute = 0.0;
    *other = 0.0;
    if (workerTimes.empty()) {
        return;
    }

    for (size_t w = 0; w < workerTimes.size(); ++w) {
        const std::vector<double>& t = workerTimes[w];
        *comm = *comm + t[2] + t[4];
        *compute = *compute + t[3];
        // By subtraction, so that whatever is not in a bucket still shows up
        // rather than quietly going missing.
        *other = *other + (t[0] - t[2] - t[4] - t[3]);
    }

    double n = (double)workerTimes.size();
    *comm = *comm / n;
    *compute = *compute / n;
    *other = *other / n;
}

// One row per worker: how much of its run went to computing, and how much to
// waiting for somebody else. A worker that is mostly idle is being starved by
// the master; mostly in recv means its upstream is the slow one (paper
// Fig. 9).
void Metrics::printWorkerTimes() const {
    if (workerTimes.empty()) {
        return;
    }

    std::cout << "\n===== where each worker's time went =====" << std::endl;
    std::cout << "  worker   grid    jobs     total    compute      idle"
              << "      recv      send     setup      poll     admin"
              << "     other" << std::endl;

    // setprecision and fixed stay on the stream, and with several nprobe
    // values there is another run's output after this table.
    std::ios_base::fmtflags flags = std::cout.flags();
    std::streamsize digits = std::cout.precision();

    for (int w = 1; w <= workers; ++w) {
        const std::vector<double>& t = workerTimes[w - 1];
        double total = t[0];
        if (total <= 0.0) {
            total = 1e-9;
        }

        std::string grid = std::to_string((w - 1) / bDim) + "x"
                         + std::to_string((w - 1) % bDim);
        std::cout << "  " << std::setw(6) << w
                  << std::setw(7) << grid
                  << std::setw(8) << (long)t[5]
                  << std::setw(9) << std::fixed << std::setprecision(2)
                  << t[0] << "s";

        // compute, idle, recv, send, setup, poll, admin
        int order[7] = {3, 1, 2, 4, 6, 7, 8};
        double named = 0.0;
        for (int i = 0; i < 7; ++i) {
            named = named + t[order[i]];
            std::cout << std::setw(8) << std::setprecision(1)
                      << (100.0 * t[order[i]] / total) << "%";
        }

        // Printed rather than left to be worked out, so it is obvious when
        // the seven above stop covering the run.
        std::cout << std::setw(8) << std::setprecision(1)
                  << (100.0 * (total - named) / total) << "%" << std::endl;
    }

    // How unevenly the arithmetic actually fell, as one number. The table
    // above has it per worker, but comparing eight rows by eye does not go
    // into a result table, and this is the figure a partitioning scheme is
    // judged on: every query waits for the slowest machine holding part of
    // it.
    //
    // Measured on compute seconds rather than total, because total is the
    // same for everyone by construction -- a worker that finishes early just
    // waits.
    double lo = -1.0;
    double hi = 0.0;
    double sum = 0.0;
    for (int w = 1; w <= workers; ++w) {
        double c = workerTimes[w - 1][3];
        if (lo < 0.0 || c < lo) {
            lo = c;
        }
        if (c > hi) {
            hi = c;
        }
        sum = sum + c;
    }
    if (lo > 0.0) {
        std::cout << std::fixed << std::setprecision(3)
                  << "  busy time: " << lo << "s to " << hi << "s"
                  << "   mean " << (sum / workers) << "s"
                  << std::setprecision(2)
                  << "   spread " << (hi / lo) << "x" << std::endl;
        std::cout.flags(flags);
        std::cout.precision(digits);
    }

    std::cout.flags(flags);
    std::cout.precision(digits);

    printTimeBreakdown();
}

void Metrics::printTimeBreakdown() const {
    double comm = 0.0;
    double compute = 0.0;
    double other = 0.0;
    timeBreakdown(&comm, &compute, &other);

    double total = comm + compute + other;
    if (total <= 0.0) {
        return;
    }

    std::ios_base::fmtflags flags = std::cout.flags();
    std::streamsize digits = std::cout.precision();

    std::cout << "  ---- paper Fig. 9, mean worker ----" << std::endl;
    std::cout << std::fixed << std::setprecision(3)
              << "  communication " << comm << "s (" << std::setprecision(1)
              << (100.0 * comm / total) << "%)   "
              << std::setprecision(3) << "computation " << compute << "s ("
              << std::setprecision(1) << (100.0 * compute / total) << "%)   "
              << std::setprecision(3) << "other " << other << "s ("
              << std::setprecision(1) << (100.0 * other / total) << "%)"
              << std::endl;

    std::cout.flags(flags);
    std::cout.precision(digits);
}

// Where the master's own time went, over the same pass as the worker table.
// That table says how long the workers waited; this says what they waited
// for. The remainder is MPI_Waitany plus pushing returned candidates into the
// heaps, which is the only part of a batch the master cannot hand to anyone.
// How much looser the threshold a round pruned against was than the one that
// round's own results turned out to support, as a percentage of the latter.
// A single machine reheaps after every candidate and would show 0% here;
// what is left is the price of deciding once per block instead.
//
// One row per round, because the interesting thing is the decay: the first
// round prunes against a prewarm guess and the gap is wide, later ones start
// from a threshold earlier rounds already tightened.
void Metrics::printThresholdGap() const {
    if (stalenessMean.empty()) {
        return;
    }

    std::ios_base::fmtflags flags = std::cout.flags();
    std::streamsize digits = std::cout.precision();

    std::cout << "\n===== how far behind the thresholds were =====" << std::endl;
    std::cout << "  " << std::setw(5) << "round"
              << std::setw(12) << "average"
              << std::setw(13) << "worst"
              << std::setw(13) << "queries" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    for (size_t s = 0; s < stalenessMean.size(); ++s) {
        if (stalenessCount[s] == 0) {
            continue;
        }
        std::cout << "  " << std::setw(5) << (s + 1)
                  << std::setw(11) << (100.0 * stalenessMean[s]) << "%"
                  << std::setw(12) << (100.0 * stalenessMax[s]) << "%"
                  << std::setw(13) << stalenessCount[s] << std::endl;
    }
    std::cout << "  how much looser the threshold a round pruned with was"
              << " than the one its own" << std::endl;
    std::cout << "  results turned out to support; one machine updates after"
              << " every candidate," << std::endl;
    std::cout << "  so it would show 0%" << std::endl;

    std::cout.flags(flags);
    std::cout.precision(digits);
}

void Metrics::printMasterTimes() const {
    if (masterTotal <= 0.0) {
        return;
    }

    const char* name[6] = {"stage 0 (probes + prewarm)",
                           "broadcast (queries + probes)",
                           "dispatch (jobs + thresholds)",
                           "wait (blocked in Waitany)",
                           "merge (candidates into heaps)",
                           "other"};
    double part[6] = {masterStage0, masterBroadcast, masterDispatch,
                      masterWait, masterMerge,
                      masterTotal - masterStage0 - masterBroadcast
                          - masterDispatch - masterWait - masterMerge};

    std::ios_base::fmtflags flags = std::cout.flags();
    std::streamsize digits = std::cout.precision();

    std::cout << "\n===== where the master's time went =====" << std::endl;
    for (int i = 0; i < 6; ++i) {
        std::cout << "  " << std::left << std::setw(30) << name[i]
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(8) << part[i] << "s"
                  << std::setw(8) << std::setprecision(1)
                  << (100.0 * part[i] / masterTotal) << "%" << std::endl;
    }
    std::cout << "  " << std::left << std::setw(30) << "total (batches only)"
              << std::right << std::setprecision(3) << std::setw(8)
              << masterTotal << "s" << std::endl;

    std::cout.flags(flags);
    std::cout.precision(digits);
}

void Metrics::print() const {
    std::cout << "\n===== 5. results =====" << std::endl;
    std::cout << "setup: " << workers << " workers, grid "
              << bVec << " x " << bDim << ", nlist " << nlist
              << ", nprobe " << nprobe << ", k " << k
              << ", batch " << cfg.batch
              << ", block " << cfg.block << std::endl;
    std::cout << "arms: pruning " << pruningLabel()
              << ", assign " << cfg.assign
              << ", pipeline " << (cfg.pipeline ? "on" : "OFF")
              << ", blocksend " << (cfg.blockSend ? "ON" : "off")
              << std::endl;

    // Is the distributed answer the same as one machine's? This is the check
    // that has to pass; everything below it is a measurement, not a verdict.
    std::cout << "\ncorrectness" << std::endl;
    if (cfg.check) {
        std::cout << "  queries differing from single machine: "
                  << differing << "/" << nq
                  << "   (ties broken differently: " << ties << ")" << std::endl;
    } else {
        // Saying 0/nq here would read as a check that passed. It is off,
        // which is a different thing, and the two are easy to confuse in a
        // log of a sweep where most runs skip it to save the time.
        std::cout << "  not checked against the single machine (--check 0)"
                  << std::endl;
    }
    std::cout << "  recall@" << k << ": " << recall
              << "   (" << (k * (1.0 - recall))
              << " of " << k << " true neighbours missed per query)" << std::endl;
    if (haveR2) {
        std::cout << "  r2: " << r2
                  << "   (squared distances this much further than the true"
                  << " top-" << k << ")" << std::endl;
    }

    std::cout << "\nthroughput" << std::endl;
    std::cout << "  " << nq << " queries in " << seconds << " s" << std::endl;
    std::cout << "  QPS: " << (nq / seconds)
              << "   (" << (1000.0 * seconds / nq) << " ms per query)" << std::endl;

    // What distributing bought, against the same clustering and the same
    // probe lists on one machine (paper §6.2.1). The thread count is printed
    // because the comparison only means anything with it: this is one node
    // using all of its cores against `workers` nodes using theirs.
    if (cfg.baseline && single > 0.0) {
        int cores = 1;
        #ifdef _OPENMP
        cores = omp_get_max_threads();
        #endif
        std::cout << "  single machine: " << single << " s   ("
                  << (nq / single) << " QPS, " << cores << " threads on "
                  << hostName() << ")" << std::endl;
        std::cout << "  speedup: " << (single / seconds) << "x over one machine"
                  << "   (" << workers << " workers)" << std::endl;
    }

    // How lopsided this workload was, in the units the paper labels its
    // workloads with. Index and test workload are separate numbers: the
    // layout was built from the profiling pass, this is what was then
    // actually searched.
    std::cout << "  test workload variance: " << variance
              << "   (mean " << ((double)nq * nprobe / nlist)
              << " probes per cluster)" << std::endl;

    std::cout << "\nindex build" << std::endl;
    std::cout << "  train " << trainSeconds << " s, add "
              << addSeconds << " s, distribute "
              << distributeSeconds << " s";
    if (trainSeconds == 0.0 && addSeconds == 0.0) {
        std::cout << "   (loaded from cache, so train and add are 0)";
    }
    std::cout << std::endl;

    // Paper Table 4. The interesting number is the ratio: the workers between
    // them should hold about what one machine would, and whatever they hold
    // beyond it is what the partitioning costs.
    if (workerBytes > 0 && singleBytes > 0) {
        std::cout << "  index memory: " << (workerBytes / 1048576) << " MB over "
                  << workers << " workers ("
                  << (workerBytes / 1048576 / workers) << " MB each), vs "
                  << (singleBytes / 1048576) << " MB on one machine" << std::endl;
        std::cout << "    " << (100.0 * workerBytes / singleBytes)
                  << "% of the single-machine index, so "
                  << (100.0 * (workerBytes - singleBytes) / singleBytes)
                  << "% overhead" << std::endl;
    }

    // How much of the base each query actually touched, and how evenly that
    // work fell across the vector partitions. The spread here is what the
    // cost model's I(pi) term estimates in advance.
    double perQuery = (double)scanned / nq;
    std::cout << "\nwork" << std::endl;
    std::cout << "  candidates scanned: " << scanned
              << "   (" << perQuery << " per query, "
              << (100.0 * perQuery / baseCount) << "% of the base)" << std::endl;
    if (bVec > 1) {
        std::cout << "  per vector partition   (even would be "
                  << (100.0 / bVec) << "% each)" << std::endl;
        for (int r = 0; r < bVec; ++r) {
            std::cout << "    partition " << r << ": " << scannedRow[r]
                      << " candidates   "
                      << (100.0 * scannedRow[r] / scanned) << "%" << std::endl;
        }
    }

    // One row per dimension slice, which is also one hop of the chain. The
    // paper's Table 3 is the `skipped` column; the two beside it are what a
    // pruning method costs once it is distributed, where computing less and
    // transmitting less are different things and can point opposite ways.
    //
    // Slice 1 always skips 0% -- everyone computes the first slice, there is
    // nothing to skip against yet.
    long dimOps = 0;
    long bytesSent = 0;
    for (int s = 0; s < bDim && s < (int)dimsAfterStage.size(); ++s) {
        dimOps = dimOps + dimsAfterStage[s];
        bytesSent = bytesSent + bytesAfterStage[s];
    }

    std::cout << "\npruning, by dimension slice" << std::endl;
    std::cout << "  slice        reached      survived   skipped"
              << "        distance ops      bytes sent" << std::endl;
    for (int s = 0; s < bDim; ++s) {
        long reached = (s == 0) ? scanned : aliveAfterStage[s - 1];
        std::cout << "  " << std::setw(5) << (s + 1)
                  << std::setw(15) << reached
                  << std::setw(14) << aliveAfterStage[s]
                  << std::setw(9) << std::fixed << std::setprecision(1)
                  << (100.0 * (1.0 - (double)reached / (double)scanned)) << "%";
        if (s < (int)dimsAfterStage.size()) {
            std::cout << std::setw(20) << dimsAfterStage[s]
                      << std::setw(16) << bytesAfterStage[s];
        }
        std::cout << std::endl;
        std::cout.unsetf(std::ios_base::floatfield);
        std::cout.precision(6);
    }

    // What the same search would have cost with nothing pruned: every
    // candidate offered, over every dimension.
    double full = (double)scanned * dim;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  distance ops: " << (double)dimOps / 1e9 << " G of "
              << full / 1e9 << " G without pruning   ("
              << ((full > 0.0) ? (100.0 * dimOps / full) : 0.0) << "%)"
              << std::endl;
    std::cout << "  bytes sent:   " << (double)bytesSent / 1e6 << " MB";
    if (nq > 0) {
        std::cout << "   (" << ((double)bytesSent / nq / 1024.0)
                  << " KB per query)";
    }
    std::cout << std::endl;
    std::cout.unsetf(std::ios_base::floatfield);
    std::cout.precision(6);

    printWorkerTimes();
    printMasterTimes();
    printThresholdGap();
}

// Six files under test/, one per measurement plus one for everything else.
// Each gets the grain its own quantity has -- per hop, per worker, per round,
// per run -- rather than one shape forced on all of them, and each repeats
// enough identity to be read on its own. run_id joins them back together.
//
// No flag turns this on. A run that was not recorded cannot be compared with
// another later, and remembering to pass a flag is exactly the thing that
// gets forgotten on the run that turns out to matter.
void Metrics::writeCsv() const {
    mkdir("test", 0755);   // already there is not an error worth reporting

    // Unique per run: two runs can start within the same second, but not
    // within the same second from the same process.
    char runId[64];
    std::snprintf(runId, sizeof(runId), "%s-%d",
                  timestampNow().c_str(), (int)getpid());

    // Repeated into every file so one can be read without a join. Five
    // columns is the most that earns its place: which strategy, which grid,
    // and where on the accuracy curve.
    char id[256];
    std::snprintf(id, sizeof(id), "%s,%s,%d,%d,%d,%d",
                  runId, cfg.backend.c_str(), bVec, bDim, nprobe, k);

    // Appends, writing the header only into a file that is new or empty.
    struct Open {
        static std::FILE* at(const char* path, const char* header) {
            bool isNew = true;
            std::FILE* probe = std::fopen(path, "rb");
            if (probe != nullptr) {
                std::fseek(probe, 0, SEEK_END);
                isNew = (std::ftell(probe) == 0);
                std::fclose(probe);
            }
            std::FILE* f = std::fopen(path, "ab");
            if (f != nullptr && isNew) {
                std::fprintf(f, "%s\n", header);
            }
            return f;
        }
    };

    // ---- 1. dimensions evaluated: per hop, and what it would have been ----
    std::FILE* f = Open::at("test/dims_evaluated.csv",
        "run_id,backend,bvec,bdim,nprobe,k,hop_id,dims_evaluated,"
        "dims_no_pruning,pct_of_no_pruning");
    if (f != nullptr) {
        for (int h = 0; h < bDim; ++h) {
            long ops = (h < (int)dimsAfterStage.size()) ? dimsAfterStage[h] : 0;
            // What this slice alone would have cost with nothing pruned.
            double bare = (double)scanned * (dim / (bDim > 0 ? bDim : 1));
            std::fprintf(f, "%s,%d,%ld,%.0f,%.4f\n", id, h + 1, ops, bare,
                         (bare > 0.0) ? (100.0 * ops / bare) : 0.0);
        }
        std::fclose(f);
    }

    // ---- 2. candidates forwarded per hop ----
    f = Open::at("test/candidates_per_hop.csv",
        "run_id,backend,bvec,bdim,nprobe,k,hop_id,reached,alive,skipped_pct");
    if (f != nullptr) {
        for (int h = 0; h < bDim; ++h) {
            long reached = (h == 0) ? scanned
                         : (h - 1 < (int)aliveAfterStage.size()
                            ? aliveAfterStage[h - 1] : 0);
            long alive = (h < (int)aliveAfterStage.size())
                       ? aliveAfterStage[h] : 0;
            // The share of every candidate offered that never got this far.
            double skipped = (scanned > 0)
                           ? (100.0 * (1.0 - (double)reached / (double)scanned))
                           : 0.0;
            std::fprintf(f, "%s,%d,%ld,%ld,%.4f\n", id, h + 1,
                         reached, alive, skipped);
        }
        std::fclose(f);
    }

    // ---- 3. bytes per hop ----
    f = Open::at("test/bytes_per_hop.csv",
        "run_id,backend,bvec,bdim,nprobe,k,hop_id,bytes_fwd,bytes_per_query");
    if (f != nullptr) {
        for (int h = 0; h < bDim; ++h) {
            long by = (h < (int)bytesAfterStage.size()) ? bytesAfterStage[h] : 0;
            std::fprintf(f, "%s,%d,%ld,%.2f\n", id, h + 1, by,
                         (nq > 0) ? ((double)by / nq) : 0.0);
        }
        std::fclose(f);
    }

    // ---- 4. node busy-time skew: the per-worker seconds it comes from ----
    double busyLo = -1.0;
    double busyHi = 0.0;
    for (size_t w = 0; w < workerTimes.size(); ++w) {
        double c = workerTimes[w][3];
        if (busyLo < 0.0 || c < busyLo) {
            busyLo = c;
        }
        if (c > busyHi) {
            busyHi = c;
        }
    }
    double busySkew = (busyLo > 0.0) ? (busyHi / busyLo) : 0.0;

    f = Open::at("test/busy_skew.csv",
        "run_id,backend,bvec,bdim,nprobe,k,worker,row,col,"
        "busy_s,total_s,idle_pct,setup_pct,run_skew");
    if (f != nullptr) {
        for (size_t w = 0; w < workerTimes.size(); ++w) {
            double tot = workerTimes[w][0];
            std::fprintf(f, "%s,%d,%d,%d,%.6f,%.6f,%.2f,%.2f,%.4f\n",
                         id, (int)w + 1,
                         (int)w / (bDim > 0 ? bDim : 1),
                         (int)w % (bDim > 0 ? bDim : 1),
                         workerTimes[w][3], tot,
                         (tot > 0.0) ? (100.0 * workerTimes[w][1] / tot) : 0.0,
                         (tot > 0.0) ? (100.0 * workerTimes[w][6] / tot) : 0.0,
                         busySkew);
        }
        std::fclose(f);
    }

    // ---- 5. threshold staleness: per vector-level round ----
    f = Open::at("test/threshold_staleness.csv",
        "run_id,backend,bvec,bdim,nprobe,k,round,mean_pct,worst_pct,queries");
    if (f != nullptr) {
        for (size_t r = 0; r < stalenessMean.size(); ++r) {
            std::fprintf(f, "%s,%d,%.4f,%.4f,%ld\n", id, (int)r + 1,
                         100.0 * stalenessMean[r],
                         100.0 * (r < stalenessMax.size() ? stalenessMax[r] : 0.0),
                         r < stalenessCount.size() ? stalenessCount[r] : 0L);
        }
        std::fclose(f);
    }

    // ---- 6. everything else: one row per run ----
    double comm = 0.0;
    double compute = 0.0;
    double other = 0.0;
    timeBreakdown(&comm, &compute, &other);

    long dimOps = 0;
    long bytesFwd = 0;
    for (size_t s2 = 0; s2 < dimsAfterStage.size(); ++s2) {
        dimOps = dimOps + dimsAfterStage[s2];
        bytesFwd = bytesFwd + bytesAfterStage[s2];
    }

    double staleSum = 0.0;
    long staleN = 0;
    for (size_t s2 = 0; s2 < stalenessMean.size(); ++s2) {
        staleSum = staleSum + stalenessMean[s2] * (double)stalenessCount[s2];
        staleN = staleN + stalenessCount[s2];
    }
    double staleMean = (staleN > 0) ? (staleSum / staleN) : 0.0;

    f = Open::at("test/runs.csv",
        "run_id,timestamp,build,host,data,"
        "backend,pruning,assign,pipeline,blocksend,gemmall,compact,adeps,dadesig,"
        "bvec,bdim,nlist,nprobe,k,nq,workers,skew,skewshift,variance,prewarm,"
        "qps,latency_ms,recall,differing,speedup,"
        "dim_ops,bytes_fwd,cand_offered,busy_skew,stale_mean,"
        "comm_time,compute_time,other_time,mem_mb");
    if (f != nullptr) {
        std::fprintf(f,
            "%s,%s,%s,%s,%s,"
            "%s,%s,%s,%d,%d,%d,%d,%.3f,%.3f,"
            "%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.2f,%d,"
            "%.3f,%.4f,%.6f,%d,%.3f,"
            "%ld,%ld,%ld,%.4f,%.6f,"
            "%.4f,%.4f,%.4f,%ld\n",
            runId, timestampNow().c_str(), buildId(), hostName().c_str(),
            cfg.data.c_str(), cfg.backend.c_str(), pruningLabel().c_str(), cfg.assign.c_str(),
            cfg.pipeline ? 1 : 0, cfg.blockSend ? 1 : 0, cfg.gemmAll ? 1 : 0,
            cfg.compact ? 1 : 0, cfg.adEps, cfg.dadeSig,
            bVec, bDim, nlist, nprobe, k, nq, workers,
            cfg.skew, cfg.skewShift, variance, cfg.prewarm,
            nq / seconds, 1000.0 * seconds / nq, recall,
            // -1 rather than 0 when the check was off: a sweep leaves it off
            // on most runs, and a column of zeroes would read as passes.
            cfg.check ? differing : -1,
            (single > 0.0) ? (single / seconds) : 0.0,
            dimOps, bytesFwd, scanned, busySkew, staleMean,
            comm, compute, other, workerBytes / 1048576);
        std::fclose(f);
    }
}

}  // namespace harmony
