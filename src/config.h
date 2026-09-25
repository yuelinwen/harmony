#ifndef HARMONY_CONFIG_H
#define HARMONY_CONFIG_H

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Run-time settings, so an experiment does not need a recompile.
//
// The measurement behind a default is in the commit that set it, and in the
// comment next to it when it is short enough to be worth repeating.

namespace harmony {

struct Config {
    // data and index
    std::string data = "Data/sift";  // prefix of _base.bin, _query.bin, _gt.bin
    int nlist = 256;                 // clusters kmeans builds
    int iters = 25;                  // kmeans rounds
    int trainPoints = 256;           // kmeans training points per centroid, 0 = all
    bool cache = false;              // reuse a saved index, and save one if absent

    // worker layout: a bVec x bDim grid (paper Fig. 4a)
    std::string mode = "harmony";    // harmony | vector | dimension (paper §5)
    int bVec = 0;                    // vector partitions, overrides mode
    int bDim = 0;                    // dimension slices, overrides mode
    bool costModel = false;          // set by resolveGrid, not by a flag

    // the search
    // Clusters visited per query: recall against speed. Several may be given,
    // and are then run one after another against one index and one
    // distribution -- nprobe only decides which clusters the master dispatches,
    // so nothing a worker holds depends on it. nprobe is the first of them,
    // which is what the profiling pass and the cost model read.
    int nprobe = 32;
    std::vector<int> nprobes;
    int k = 100;       // neighbours returned
    int nq = 100;      // queries to run

    // speed
    int threads = 1;      // OpenMP threads inside each worker (paper §5)
    int batch = 1024;       // queries handled together (Algorithm 1 line 13)
    // Heap seeding, Algorithm 1 lines 1-5. prewarmLists clusters of the query's
    // nprobe, prewarm vectors out of each. 0 either way turns it off.
    //
    // One cluster, not the ten the authors' code seeds from: measured on
    // sift1M, spreading the same budget over more clusters leaves a looser
    // threshold and prunes less, at every budget tried. The nearest cluster is
    // where the nearest vectors are. Since pruning is lossless, a looser
    // threshold costs speed and never recall.
    int prewarm = 500;
    int prewarmLists = 1;
    // The paper's two pruning levels (§5's --Pruning_Configuration), both on
    // unless switched off:
    //
    //   dim     a candidate whose running total has passed the threshold is
    //           dropped where it stands (§4.3, Fig. 10)
    //   vector  a query group finishes one vector partition before starting
    //           the next, so the next one starts from a threshold the first
    //           has already tightened (Fig. 5a)
    //
    // Named the way the sample names its own, as switches that are either
    // present or absent rather than flags taking a value. A value flag whose
    // meaning has changed reads as valid and quietly does something else; an
    // absent one cannot.
    //
    // Turning dim off while leaving vector on is the control for measuring
    // what dim pruning is worth: the threshold still tightens between
    // partitions, nothing reads it, and the pipeline is otherwise untouched.
    bool pruneDim = true;
    bool pruneVector = true;
    // Wait for each forward to land before starting the next cluster, which
    // is the blocking arm of the paper's Fig. 2(b) comparison.
    bool blockSend = false;
    // Query blocks a group is cut into. A block is the unit that travels the
    // dimension chain, and all of a partition's blocks are in flight at once,
    // so this is also how much a worker has to overlap the wait for one
    // block's upstream with another block's arithmetic. The sample's examples
    // use four to eight. More blocks costs memory -- a worker holds one
    // running total per (query, candidate) pair of every open block -- and
    // reads thresholds a little earlier, so it prunes slightly less.
    // 0 means "not given": resolveGrid() works it out from the grid, because
    // the right value depends on bDim and a single default is wrong for every
    // layout but one. An explicit --block always wins.
    int block = 0;
    // Whether --block was given. The default depends on the grid, and under
    // --mode harmony the grid is not final until choosePlan() has run, so the
    // default has to be worked out twice -- and only when the user left it to
    // us.
    bool blockGiven = false;

    // Hand out one block at a time, waiting for it to come back before
    // dispatching the next: the arm Fig. 10 calls "without pipeline and
    // asynchronous execution". Blocks still travel a chain and the forwards
    // are still non-blocking -- this only stops several being out at once, so
    // what it measures is the overlap and nothing else.
    //
    // Named like --disablepruning: the switch is present or absent, never
    // takes a value, so it cannot quietly change meaning later.
    bool pipeline = true;

    bool check = true;    // re-run each query on one machine and compare

    // Time that single-machine pass and report the speedup over it (paper
    // §6.2.1, the sample's faiss_query_time). It is the same clustering and
    // the same probe lists as the distributed run, so it measures what
    // distributing bought and nothing else -- a fairer baseline than the
    // paper's, which compares against a different implementation.
    bool baseline = true;
    int loop = 1;         // timed passes over the query set, averaged
    std::string csv;      // append one row per run here, "" = off

    // Synthetic skewed workload (paper §6.6, Fig. 7/8; the sample's
    // --HardInBalance). 0 uses the real nearest clusters. Above that, this
    // share of every query's probes is drawn from a hot eighth of the
    // clusters and the rest uniformly, which concentrates the work without
    // touching the index or the layout.
    double skew = 0.0;

    // The skew of the workload the layout is built from, which is a separate
    // thing from the skew of the workload it is then measured on.
    //
    // Paper §6.2.2 and the cost-model evaluation both build the index from a
    // historical workload at variance 500 and then test at a different
    // variance: what degrades a vector-only layout is the *mismatch* between
    // the distribution the clusters were assigned for and the one that
    // arrives. With one knob for both there is never any mismatch, the layout
    // always fits, and measured QPS went up with skew rather than down --
    // the opposite of Fig. 8.
    //
    // -1 means "follow --skew", resolved in resolveGrid(). Defaulting to 0
    // instead would silently turn every existing --skew run into the
    // mismatched case and change what the old numbers meant.
    double indexSkew = -1.0;

    // How much of the test workload's hot set is somewhere else than the one
    // the layout was built for. 0 means the same clusters are hot in both.
    //
    // This is the knob Fig. 8 actually turns, and it took a measurement to
    // find that out. Concentration alone changes nothing, because LPT hands
    // the heaviest clusters to the lightest partitions and so spreads the hot
    // set one per partition -- making the same clusters hotter keeps the load
    // balanced (measured: shares stay within 11.1-13.6% at skew 0.8, while
    // --assign roundrobin goes 2.4-19.8% on the same workload). What hurts a
    // vector-only layout is the clusters it optimised for not being the ones
    // the queries ask for.
    //
    // The replacements are drawn size-weighted from outside the hot set, like
    // the hot set itself, so the candidate count stays comparable and the
    // change is in the shape rather than the weight.
    double skewShift = 0.0;

    // How clusters are handed to vector partitions. "lpt" gives the heaviest
    // to whichever partition is lightest so far; "roundrobin" is c % bVec,
    // which is what this used to do and what the sample still does (as
    // contiguous ranges). The second is only there to measure the first.
    std::string assign = "lpt";

    // cost model, read only when mode is "harmony" (paper §4.2.1)
    double commCost = 100.0;  // a transferred byte, in multiply-adds. Hardware.
    double alpha = 0.3;       // weight on the imbalance term of C(pi,Q)
    int warmup = 1000;        // queries profiled to find the hot clusters
};

inline void printUsage(const char* prog) {
    std::cerr
        << "usage: mpirun -n <N+1> " << prog << " [options]\n"
        << "\n"
        << "data and index\n"
        << "  --data <prefix>    reads <prefix>_base/_query/_gt.bin (Data/sift)\n"
        << "  --nlist <int>      clusters in the index              (256)\n"
        << "  --iters <int>      kmeans rounds                      (25)\n"
        << "  --trainpoints <int>  kmeans training points per centroid (256)\n"
        << "                     0 uses every base vector, which is much slower\n"
        << "  --cache <0|1>      reuse the index file beside the data       (0)\n"
        << "                     writes it on the first run with these settings\n"
        << "\n"
        << "how the workers are laid out\n"
        << "  --mode <name>      harmony | vector | dimension       (harmony)\n"
        << "                       harmony   = the cost model picks the grid\n"
        << "                       vector    = N x 1, no dimension pipeline\n"
        << "                       dimension = 1 x N, no vector partitions\n"
        << "  --bvec <int>       vector partitions, overrides --mode\n"
        << "  --bdim <int>       dimension slices,  overrides --mode\n"
        << "\n"
        << "the search\n"
        << "  --nq <int>         queries to run                     (100)\n"
        << "  --nprobe <int>...  clusters visited per query         (32)\n"
        << "                     several values run one after another, sharing\n"
        << "                     one index: --nprobe 8 16 32\n"
        << "  --k <int>          neighbours returned                (100)\n"
        << "\n"
        << "speed\n"
        << "  --threads <int>    OpenMP threads per worker          (1)\n"
        << "  --batch <int>      queries processed together        (1024)\n"
        << "  --prewarm <int>    heap seed vectors per cluster      (500)\n"
        << "  --prewarmlists <int>  clusters seeded per query         (1)\n"
        << "                     either set to 0 turns seeding off\n"
        << "  --disablepruning   turn off dimension-level pruning\n"
        << "                     (drop a candidate past the threshold)\n"
        << "  --disablevectorpruning  turn off vector-level pruning\n"
        << "                     (tighten the threshold between partitions)\n"        << "  --blocksend <0|1>  wait for each forward to land        (0)\n"
        << "  --block <int>      query blocks per group        (from bDim)\n"
        << "                     default keeps about 256/bDim^2 queries in a\n"
        << "                     block: 4 at 8x1, 16 at 4x2, 64 at 2x4,\n"
        << "                     128 at 1x8, which is what measured fastest\n"
        << "  --blocksend <0|1>  wait for each forward to land        (0)\n"
        << "  --disablepipeline  one block out at a time, no overlap\n"
        << "                     otherwise all of a partition's blocks fly\n"
        << "                     at once\n"
        << "  --check <0|1>      verify against a single machine    (1)\n"
        << "                     costs more than the search it checks\n"
        << "  --baseline <0|1>   time that pass, report the speedup  (1)\n"
        << "  --loop <int>       timed passes, averaged               (1)\n"
        << "                     above 1 adds an untimed warm-up pass\n"
        << "  --csv <path>       append one row per run to this file\n"
        << "\n"
        << "skewed workload (paper 6.6)\n"
        << "  --assign <name>    lpt | roundrobin                    (lpt)\n"
        << "                     roundrobin is the arm to measure lpt against\n"
        << "  --skew <float>     share of probes aimed at a hot eighth  (0)\n"
        << "                     0 searches for real; above it the probe lists\n"
        << "                     are synthetic, so recall stops meaning anything\n"
        << "                     while differing still does\n"
        << "  --indexskew <f>    the skew the layout is built from (--skew)\n"
        << "  --skewshift <f>    share of the test hot set that moved     (0)\n"
        << "                     0 = the same clusters are hot in both, which\n"
        << "                     LPT absorbs; raise it to make the layout miss\n"
        << "                     the clusters the queries want (Fig. 8)\n"
        << "\n"
        << "cost model, only read when --mode harmony\n"
        << "  --commcost <float> a transferred byte, in multiply-adds (100)\n"
        << "  --alpha <float>    weight on the imbalance term        (0.3)\n"
        << "  --warmup <int>     queries used to find hot clusters   (1000)\n";
}

// Returns false on an unknown or incomplete option.
inline bool parseArgs(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];
        bool hasValue = (i + 1 < argc);

        if (opt == "--data" && hasValue) {
            cfg.data = argv[++i];
        } else if (opt == "--nlist" && hasValue) {
            cfg.nlist = std::atoi(argv[++i]);
        } else if (opt == "--iters" && hasValue) {
            cfg.iters = std::atoi(argv[++i]);
        } else if (opt == "--cache" && hasValue) {
            cfg.cache = (std::atoi(argv[++i]) != 0);
        } else if (opt == "--trainpoints" && hasValue) {
            cfg.trainPoints = std::atoi(argv[++i]);
            if (cfg.trainPoints < 0) {
                cfg.trainPoints = 0;
            }
        } else if (opt == "--mode" && hasValue) {
            cfg.mode = argv[++i];
        } else if (opt == "--bvec" && hasValue) {
            cfg.bVec = std::atoi(argv[++i]);
        } else if (opt == "--bdim" && hasValue) {
            cfg.bDim = std::atoi(argv[++i]);
        } else if (opt == "--nprobe" && hasValue) {
            // eats values until the next option, so one or several both work
            cfg.nprobes.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg.nprobes.push_back(std::atoi(argv[++i]));
            }
            if (!cfg.nprobes.empty()) {
                cfg.nprobe = cfg.nprobes[0];
            }
        } else if (opt == "--csv" && hasValue) {
            cfg.csv = argv[++i];
        } else if (opt == "--assign" && hasValue) {
            cfg.assign = argv[++i];
            if (cfg.assign != "lpt" && cfg.assign != "roundrobin") {
                std::cerr << "unknown assign: " << cfg.assign << std::endl;
                printUsage(argv[0]);
                return false;
            }
        } else if (opt == "--skew" && hasValue) {
            cfg.skew = std::atof(argv[++i]);
            if (cfg.skew < 0.0) {
                cfg.skew = 0.0;
            }
            if (cfg.skew > 1.0) {
                cfg.skew = 1.0;
            }
        } else if (opt == "--skewshift" && hasValue) {
            cfg.skewShift = std::atof(argv[++i]);
            if (cfg.skewShift < 0.0) {
                cfg.skewShift = 0.0;
            }
            if (cfg.skewShift > 1.0) {
                cfg.skewShift = 1.0;
            }
        } else if (opt == "--indexskew" && hasValue) {
            cfg.indexSkew = std::atof(argv[++i]);
            if (cfg.indexSkew < 0.0) {
                cfg.indexSkew = 0.0;
            }
            if (cfg.indexSkew > 1.0) {
                cfg.indexSkew = 1.0;
            }
        } else if (opt == "--k" && hasValue) {
            cfg.k = std::atoi(argv[++i]);
        } else if (opt == "--nq" && hasValue) {
            cfg.nq = std::atoi(argv[++i]);
        } else if (opt == "--prewarm" && hasValue) {
            cfg.prewarm = std::atoi(argv[++i]);
        } else if (opt == "--prewarmlists" && hasValue) {
            cfg.prewarmLists = std::atoi(argv[++i]);
        } else if (opt == "--disablepruning") {
            cfg.pruneDim = false;
        } else if (opt == "--disablevectorpruning") {
            cfg.pruneVector = false;
        } else if (opt == "--loop" && hasValue) {
            cfg.loop = std::atoi(argv[++i]);
        } else if (opt == "--check" && hasValue) {
            cfg.check = (std::atoi(argv[++i]) != 0);
        } else if (opt == "--baseline" && hasValue) {
            cfg.baseline = (std::atoi(argv[++i]) != 0);
        } else if (opt == "--threads" && hasValue) {
            cfg.threads = std::atoi(argv[++i]);
        } else if (opt == "--batch" && hasValue) {
            cfg.batch = std::atoi(argv[++i]);
        } else if (opt == "--block" && hasValue) {
            cfg.block = std::atoi(argv[++i]);
            if (cfg.block < 1) {
                cfg.block = 1;
            }
            cfg.blockGiven = true;
        } else if (opt == "--disablepipeline") {
            cfg.pipeline = false;
        } else if (opt == "--blocksend" && hasValue) {
            cfg.blockSend = (std::atoi(argv[++i]) != 0);
        } else if (opt == "--alpha" && hasValue) {
            cfg.alpha = std::atof(argv[++i]);
        } else if (opt == "--commcost" && hasValue) {
            cfg.commCost = std::atof(argv[++i]);
        } else if (opt == "--warmup" && hasValue) {
            cfg.warmup = std::atoi(argv[++i]);
        } else {
            std::cerr << "bad option: " << opt << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

// The default --block for a given grid.
//
// One value cannot fit every layout: measured at nq 1024, the best is 4 at 8x1
// but 128 at 1x8, and using 4 everywhere -- which this did once -- costs 44% at
// 2x4 and 75% at 1x8. What holds steady across the four is the number of
// queries in a block, not the number of blocks: 256, 64, 16, 8 as bDim goes
// 1, 2, 4, 8. A block's running totals are (queries x candidates) and that
// buffer travels bDim hops, so a deeper chain needs a smaller block to keep
// the pipeline moving. 256/bDim^2 reproduces all four measured optima; the
// floor of 8 is where 1x8 stopped improving.
//
// min(batch, nq), not batch: a batch holds whatever is left of the query set,
// so --nq 256 with the default --batch 1024 runs batches of 256.
inline int defaultBlock(const Config& cfg, int bDim) {
    int perBlock = 256 / (bDim * bDim);
    if (perBlock < 8) {
        perBlock = 8;
    }
    int perBatch = (cfg.batch < cfg.nq) ? cfg.batch : cfg.nq;
    int block = perBatch / perBlock;
    return (block < 1) ? 1 : block;
}

// Turns --mode into a grid, unless --bvec/--bdim were given. bVec * bDim has
// to come out equal to the worker count.
inline bool resolveGrid(Config& cfg, int numWorkers) {

    // -n 1 leaves no worker, and every layout below would then divide by
    // zero. Here rather than in main because both nodes pass through it.
    if (numWorkers < 1) {
        std::cerr << "no workers: -n is workers plus one, so it needs to be "
                  << "at least 2\n" << std::endl;
        return false;
    }

    // Counts with a floor. Every one of these was reachable and every one of
    // them failed badly rather than loudly: --nprobe 0, --k 0 and --nlist 0
    // each segfaulted (an empty TopKHeap's push reads heap_.top(), an empty
    // centroid array gets indexed), --batch 0 turned the batch loop into
    // `start += 0` and hung looking like it was computing, and --warmup -1
    // reached `std::vector probes(-1)` and aborted on length_error.
    //
    // The last three floor at 0 rather than 1 because "none" is a real
    // setting for them: --warmup 0 skips profiling and leaves the split to
    // cluster size alone, --prewarm 0 starts every query from an empty
    // threshold. A negative is a typo, and used to be a silent no-op.
    //
    // Checked here rather than in parseArgs so one place covers them all, and
    // so the message names the flag the way the user typed it.
    struct Least { const char* flag; int value; int floor; };
    Least least[] = {
        {"--nlist", cfg.nlist, 1},
        {"--nprobe", cfg.nprobe, 1},
        {"--k", cfg.k, 1},
        {"--nq", cfg.nq, 1},
        {"--batch", cfg.batch, 1},
        {"--iters", cfg.iters, 1},
        {"--loop", cfg.loop, 1},
        {"--threads", cfg.threads, 1},
        {"--warmup", cfg.warmup, 0},
        {"--prewarm", cfg.prewarm, 0},
        {"--prewarmlists", cfg.prewarmLists, 0},
    };
    for (size_t i = 0; i < sizeof(least) / sizeof(least[0]); ++i) {
        if (least[i].value < least[i].floor) {
            std::cerr << least[i].flag << " must be at least " << least[i].floor
                      << ", got " << least[i].value << std::endl;
            return false;
        }
    }

    // Not given: the layout is built from the same workload it is measured
    // on, which is what this did when there was only one knob.
    if (cfg.indexSkew < 0.0) {
        cfg.indexSkew = cfg.skew;
    }

    // --nprobe takes several values; cfg.nprobe above is only the first.
    for (size_t i = 0; i < cfg.nprobes.size(); ++i) {
        if (cfg.nprobes[i] < 1) {
            std::cerr << "--nprobe must be at least 1, got "
                      << cfg.nprobes[i] << std::endl;
            return false;
        }
    }

    if (cfg.bVec > 0 || cfg.bDim > 0) {
        if (cfg.bVec <= 0) {
            cfg.bVec = (cfg.bDim > 0) ? numWorkers / cfg.bDim : 0;
        }
        if (cfg.bDim <= 0) {
            cfg.bDim = numWorkers / cfg.bVec;
        }
    } else if (cfg.mode == "vector") {
        cfg.bVec = numWorkers;
        cfg.bDim = 1;
    } else if (cfg.mode == "dimension") {
        cfg.bVec = 1;
        cfg.bDim = numWorkers;
    } else if (cfg.mode == "harmony" || cfg.mode == "auto") {   // "auto": old name
        // The paper's Harmony is the cost-model-driven mode (§5). Start where
        // §4.2.1 says to -- every machine holding d/N dimensions, so
        // bVec = 1 -- and let choosePlan() raise bVec from there.
        cfg.bVec = 1;
        cfg.bDim = numWorkers;
        cfg.costModel = true;
    } else {
        std::cerr << "unknown mode: " << cfg.mode << std::endl;
        return false;
    }

    if (cfg.bVec <= 0 || cfg.bDim <= 0 || cfg.bVec * cfg.bDim != numWorkers) {
        std::cerr << "grid " << cfg.bVec << "x" << cfg.bDim
                  << " does not match " << numWorkers << " workers" << std::endl;
        return false;
    }

    // --block, once the grid is known. Under --mode harmony this grid is only
    // a starting point and choosePlan() will replace it, so the master works
    // the default out again there; this value is what the other modes use and
    // what a worker would see.
    if (!cfg.blockGiven) {
        cfg.block = defaultBlock(cfg, cfg.bDim);
    }
    return true;
}

}  // namespace harmony

#endif  // HARMONY_CONFIG_H
