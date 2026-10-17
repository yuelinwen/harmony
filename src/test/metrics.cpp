#include "metrics.h"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <ios>
#include <iostream>

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
    std::cout << "  queries differing from single machine: "
              << differing << "/" << nq
              << "   (ties broken differently: " << ties << ")" << std::endl;
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
    printFive();
}

// The five numbers everything above exists to produce, together, so that one
// run can be put beside another without reading the whole output.
//
// They decompose "why was this fast or slow" into four separate questions:
// how much arithmetic the pruning removed, what it cost to spread that
// decision over machines, how evenly the work landed, and how far behind the
// information each decision was made on. A method that evaluates fewer
// dimensions but forwards more state can lose, and only these together say
// so.
void Metrics::printFive() const {
    std::ios_base::fmtflags flags = std::cout.flags();
    std::streamsize digits = std::cout.precision();

    long dimOps = 0;
    long bytesSent = 0;
    for (size_t s = 0; s < dimsAfterStage.size(); ++s) {
        dimOps = dimOps + dimsAfterStage[s];
        bytesSent = bytesSent + bytesAfterStage[s];
    }
    double full = (double)scanned * dim;

    double lo = -1.0;
    double hi = 0.0;
    for (int w = 1; w <= (int)workerTimes.size(); ++w) {
        double c = workerTimes[w - 1][3];
        if (lo < 0.0 || c < lo) {
            lo = c;
        }
        if (c > hi) {
            hi = c;
        }
    }

    std::cout << "\n===== five numbers =====" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "  1. dimensions evaluated   " << (double)dimOps / 1e9 << " G";
    if (full > 0.0) {
        std::cout << "   (" << (100.0 * dimOps / full) << "% of no pruning)";
    }
    std::cout << std::endl;

    std::cout << "  2. candidates per hop     ";
    if (!aliveAfterStage.empty()) {
        std::cout << (double)scanned / 1e6 << " M offered, "
                  << (double)aliveAfterStage[aliveAfterStage.size() - 1] / 1e6
                  << " M alive at the tail   (" << bDim << " hop"
                  << (bDim == 1 ? "" : "s") << ")";
    }
    std::cout << std::endl;

    std::cout << "  3. bytes per hop          " << (double)bytesSent / 1e6
              << " MB";
    if (nq > 0) {
        std::cout << "   (" << ((double)bytesSent / nq / 1024.0)
                  << " KB per query)";
    }
    std::cout << std::endl;

    std::cout << "  4. node busy-time skew    ";
    if (lo > 0.0) {
        std::cout << (hi / lo) << "x"
                  << std::setprecision(3) << "   (" << lo << "s to " << hi
                  << "s over " << workerTimes.size() << " workers)"
                  << std::setprecision(2);
    }
    std::cout << std::endl;

    std::cout << "  5. threshold gap          ";
    if (!stalenessMean.empty()) {
        std::cout << (100.0 * stalenessMean[0]) << "% at round 1";
        if (stalenessMean.size() > 1) {
            std::cout << ", " << (100.0 * stalenessMean[stalenessMean.size() - 1])
                      << "% at round " << stalenessMean.size();
        }
        std::cout << "   (average)";
    }
    std::cout << std::endl;

    std::cout.flags(flags);
    std::cout.precision(digits);
}

// One row per run, appended, header written when the file is new. Everything
// that was varied over a set of runs has to be in the row, or the rows cannot
// be told apart later.
void Metrics::writeCsv() const {
    if (cfg.csv.empty()) {
        return;
    }

    bool isNew = true;
    std::FILE* probe = std::fopen(cfg.csv.c_str(), "rb");
    if (probe != nullptr) {
        std::fseek(probe, 0, SEEK_END);
        isNew = (std::ftell(probe) == 0);
        std::fclose(probe);
    }

    std::FILE* f = std::fopen(cfg.csv.c_str(), "ab");
    if (f == nullptr) {
        std::cout << "could not write " << cfg.csv << std::endl;
        return;
    }

    // The four identity columns come first: a row has to say when it was
    // taken, from what, and where, or a file of them cannot be sorted, and a
    // repeat of the same settings is indistinguishable from the original.
    if (isNew) {
        std::fprintf(f, "timestamp,build,host,elapsed,"
                        "data,nlist,iters,trainpoints,workers,bvec,bdim,mode,"
                        "assign,skew,indexskew,skewshift,batch,block,blocksend,pipeline,"
                        "threads,prewarm,prewarmlists,"
                        "pruning,loop,nprobe,k,nq,recall,r2,qps,ms_per_query,"
                        "single_time,speedup,variance,"
                        "comm_time,compute_time,other_time,"
                        "train_time,add_time,distribute_time,"
                        "mem_mb,mem_pct,"
                        "differing,ties,scanned,work_pct\n");
    }

    double comm = 0.0;
    double compute = 0.0;
    double other = 0.0;
    timeBreakdown(&comm, &compute, &other);

    std::fprintf(f,
        "%s,%s,%s,%.1f,"
        "%s,%d,%d,%d,%d,%d,%d,%s,%s,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%d,%d,%s,%d,"
        "%d,%d,%d,%.6f,%.6f,%.3f,%.4f,"
        "%.4f,%.3f,%.2f,"
        "%.4f,%.4f,%.4f,"
        "%.3f,%.3f,%.3f,"
        "%ld,%.2f,"
        "%d,%d,%ld,%.4f\n",
        timestampNow().c_str(), buildId(), hostName().c_str(), elapsed,
        cfg.data.c_str(), nlist, cfg.iters, cfg.trainPoints,
        workers, bVec, bDim, cfg.mode.c_str(),
        cfg.assign.c_str(), cfg.skew, cfg.indexSkew, cfg.skewShift,
        cfg.batch, cfg.block, cfg.blockSend ? 1 : 0, cfg.pipeline ? 1 : 0,
        cfg.threads, cfg.prewarm, cfg.prewarmLists,
        pruningLabel().c_str(), cfg.loop,
        nprobe, k, nq, recall, r2, nq / seconds, 1000.0 * seconds / nq,
        single, (single > 0.0) ? (single / seconds) : 0.0, variance,
        comm, compute, other,
        trainSeconds, addSeconds, distributeSeconds,
        workerBytes / 1048576,
        (singleBytes > 0) ? (100.0 * workerBytes / singleBytes) : 0.0,
        differing, ties, scanned, workPercent());

    std::fclose(f);
    std::cout << "appended to " << cfg.csv << std::endl;
}

}  // namespace harmony
