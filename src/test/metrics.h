#ifndef HARMONY_TEST_METRICS_H
#define HARMONY_TEST_METRICS_H

#include <string>
#include <vector>

#include "../config.h"

// Every number a run produces, and the only place any of them is printed or
// written out. Modelled on the sample's stats.h: a plain struct the caller
// fills in, plus print() and writeCsv().
//
// It is a struct rather than a class on purpose. There is no invariant to
// protect -- the master measures, this holds and formats -- and hiding the
// fields behind thirty setters would be worse than the problem.

namespace harmony {

// Which source this binary was built from, stamped in by scripts/build.sh as
// <commit>+<digest of the compiled sources>.
const char* buildId();

// Local wall-clock time, to the second. Not steady_clock: this exists to line
// a result up against a shell history or a log, not to measure anything.
std::string timestampNow();

// Which machine rank 0 ran on. Worth recording because a run that landed on
// the wrong master, or shared a machine with somebody else's leftovers, looks
// perfectly normal in every other column.
std::string hostName();

// How lopsided a workload is: the standard deviation of how often each
// cluster is probed. The sample's formula exactly (query.cpp:735-746), which
// is where the paper's unexplained "variance = 500" comes from.
double workloadVariance(const std::vector<std::vector<int>>& probes, int nlist);

struct Metrics {
    Config cfg;              // echoed into the CSV, so a row can be reproduced

    // what was run
    int workers = 0;
    int bVec = 1;
    int bDim = 1;
    int nlist = 0;
    int nprobe = 0;
    int k = 0;
    int nq = 0;
    long baseCount = 0;

    // is it right
    int differing = 0;       // must be 0; the only verdict here
    int ties = 0;
    double recall = 0.0;
    double r2 = 0.0;
    bool haveR2 = false;

    // how fast
    double seconds = 0.0;    // the distributed search
    double single = 0.0;     // the same queries on one machine, 0 if not run
    double elapsed = 0.0;    // the whole run, loading and clustering included
    double variance = 0.0;
    double trainSeconds = 0.0;
    double addSeconds = 0.0;
    double distributeSeconds = 0.0;

    // how big (paper Table 4)
    long workerBytes = 0;
    long singleBytes = 0;

    // how much work, and how much of it pruning removed (paper Table 3)
    long scanned = 0;
    std::vector<long> scannedRow;
    std::vector<long> aliveAfterStage;

    // [total, idle, recv, compute, send, jobs, setup, poll, admin, bytes]
    // per worker, as collected over TAG_TIMES
    std::vector<std::vector<double>> workerTimes;
    int reorders = 0;

    // The whole "5. results" section, in order.
    void print() const;

    // One row appended to cfg.csv, header first if the file is new. Nothing
    // happens when --csv was not given.
    void writeCsv() const;

    // The two pruning switches as one word: both | dim | vector | none. dim
    // and vector both come out at 100% of the distance work, since nothing
    // reads the threshold either way -- they differ in the pipeline, not in
    // the arithmetic.
    std::string pruningLabel() const;

    // The paper's Fig. 9 buckets, averaged over the workers, in seconds.
    void timeBreakdown(double* comm, double* compute, double* other) const;

    // Share of the distance work still done, against doing all of it.
    double workPercent() const;

private:
    void printWorkerTimes() const;
    void printTimeBreakdown() const;
};

}  // namespace harmony

#endif  // HARMONY_TEST_METRICS_H
