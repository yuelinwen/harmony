#ifndef HARMONY_CONFIG_H
#define HARMONY_CONFIG_H

#include <cstdlib>
#include <iostream>
#include <string>

// Run-time settings, so an experiment does not need a recompile.
// Measurements behind the defaults are in README.md.

namespace harmony {

struct Config {
    // data and index
    std::string data = "Data/sift";  // prefix of _base.bin, _query.bin, _gt.bin
    int nlist = 256;                 // clusters kmeans builds
    int iters = 25;                  // kmeans rounds
    int trainPoints = 256;           // kmeans training points per centroid, 0 = all

    // worker layout: a bVec x bDim grid (paper Fig. 4a)
    std::string mode = "harmony";    // harmony | vector | dimension (paper §5)
    int bVec = 0;                    // vector partitions, overrides mode
    int bDim = 0;                    // dimension slices, overrides mode
    bool costModel = false;          // set by resolveGrid, not by a flag

    // the search
    int nprobe = 32;   // clusters visited per query: recall against speed
    int k = 100;       // neighbours returned
    int nq = 100;      // queries to run

    // speed
    int threads = 1;      // OpenMP threads inside each worker (paper §5)
    int batch = 32;       // queries handled together (Algorithm 1 line 13)
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
    bool pruning = true;  // dimension-level early exit (paper Fig. 10)
    bool mkl = true;      // gemm for the first slice, if MKL was compiled in
    bool check = true;    // re-run each query on one machine and compare
    int loop = 1;         // timed passes over the query set, averaged

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
        << "  --nprobe <int>     clusters visited per query         (32)\n"
        << "  --k <int>          neighbours returned                (100)\n"
        << "\n"
        << "speed\n"
        << "  --threads <int>    OpenMP threads per worker          (1)\n"
        << "  --batch <int>      queries processed together         (32)\n"
        << "  --prewarm <int>    heap seed vectors per cluster      (500)\n"
        << "  --prewarmlists <int>  clusters seeded per query         (1)\n"
        << "                     either set to 0 turns seeding off\n"
        << "  --pruning <0|1>    dimension-level pruning            (1)\n"
        << "  --mkl <0|1>        gemm for the first slice           (1)\n"
        << "  --check <0|1>      verify against a single machine    (1)\n"
        << "                     costs more than the search it checks\n"
        << "  --loop <int>       timed passes, averaged               (1)\n"
        << "                     above 1 adds an untimed warm-up pass\n"
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
            cfg.nprobe = std::atoi(argv[++i]);
        } else if (opt == "--k" && hasValue) {
            cfg.k = std::atoi(argv[++i]);
        } else if (opt == "--nq" && hasValue) {
            cfg.nq = std::atoi(argv[++i]);
        } else if (opt == "--prewarm" && hasValue) {
            cfg.prewarm = std::atoi(argv[++i]);
        } else if (opt == "--prewarmlists" && hasValue) {
            cfg.prewarmLists = std::atoi(argv[++i]);
        } else if (opt == "--pruning" && hasValue) {
            cfg.pruning = (std::atoi(argv[++i]) != 0);
        } else if (opt == "--threads" && hasValue) {
            cfg.threads = std::atoi(argv[++i]);
        } else if (opt == "--alpha" && hasValue) {
            cfg.alpha = std::atof(argv[++i]);
        } else if (opt == "--commcost" && hasValue) {
            cfg.commCost = std::atof(argv[++i]);
        } else if (opt == "--warmup" && hasValue) {
            cfg.warmup = std::atoi(argv[++i]);
        } else if (opt == "--batch" && hasValue) {
            cfg.batch = std::atoi(argv[++i]);
        } else if (opt == "--mkl" && hasValue) {
            cfg.mkl = (std::atoi(argv[++i]) != 0);
        } else if (opt == "--loop" && hasValue) {
            cfg.loop = std::atoi(argv[++i]);
            if (cfg.loop < 1) {
                cfg.loop = 1;
            }
        } else if (opt == "--check" && hasValue) {
            cfg.check = (std::atoi(argv[++i]) != 0);
        } else {
            std::cerr << "bad option: " << opt << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

// Turns --mode into a grid, unless --bvec/--bdim were given. bVec * bDim has
// to come out equal to the worker count.
inline bool resolveGrid(Config& cfg, int numWorkers) {
    // -n 1 留不下 worker，下面每种布局都会除以零。放在这里是因为
    // master 和 worker 都要经过这个函数。
    if (numWorkers < 1) {
        std::cerr << "no workers: -n is workers plus one, so it needs to be "
                  << "at least 2\n" << std::endl;
        return false;
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
    return true;
}

}  // namespace harmony

#endif  // HARMONY_CONFIG_H
