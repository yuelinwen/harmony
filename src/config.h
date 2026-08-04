#ifndef HARMONY_CONFIG_H
#define HARMONY_CONFIG_H

#include <cstdlib>
#include <iostream>
#include <string>

// Run-time settings, so an experiment does not need a recompile.
//
// The paper exposes NMachine, Pruning_Configuration and Mode on top of the
// usual Faiss knobs (Section 5). NMachine is not needed here -- the worker
// count comes from mpirun -n.

namespace harmony {

struct Config {
    std::string basePath = "Data/sift_base.bin";
    std::string queryPath = "Data/sift_query.bin";
    std::string gtPath = "Data/sift_gt.bin";   // groundtruth, for recall

    int nlist = 256;    // clusters in the index
    int iters = 10;     // kmeans rounds

    std::string mode = "harmony";   // harmony | vector | dimension
    int bVec = 0;       // set directly to override the mode
    int bDim = 0;

    int nprobe = 32;    // clusters visited per query
    int k = 100;        // neighbours returned
    int nq = 100;       // queries to run
    int prewarm = 500;  // vectors used to seed the heap; 0 turns it off
    bool pruning = true;
};

inline void printUsage(const char* prog) {
    std::cerr
        << "usage: mpirun -n <N+1> " << prog << " [options]\n"
        << "  --base <path>      base vectors            (Data/sift_base.bin)\n"
        << "  --query <path>     query vectors           (Data/sift_query.bin)\n"
        << "  --gt <path>        groundtruth ids         (Data/sift_gt.bin)\n"
        << "  --nlist <int>      clusters in the index   (256)\n"
        << "  --iters <int>      kmeans rounds           (10)\n"
        << "  --mode <name>      harmony | vector | dimension  (harmony)\n"
        << "  --bvec <int>       vector partitions, overrides --mode\n"
        << "  --bdim <int>       dimension slices, overrides --mode\n"
        << "  --nprobe <int>     clusters per query      (32)\n"
        << "  --k <int>          neighbours returned     (100)\n"
        << "  --nq <int>         queries to run          (100)\n"
        << "  --prewarm <int>    heap seed size, 0 = off (500)\n"
        << "  --pruning <0|1>    dimension-level pruning (1)\n";
}

// Returns false on an unknown or incomplete option.
inline bool parseArgs(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];
        bool hasValue = (i + 1 < argc);

        if (opt == "--base" && hasValue) {
            cfg.basePath = argv[++i];
        } else if (opt == "--query" && hasValue) {
            cfg.queryPath = argv[++i];
        } else if (opt == "--gt" && hasValue) {
            cfg.gtPath = argv[++i];
        } else if (opt == "--nlist" && hasValue) {
            cfg.nlist = std::atoi(argv[++i]);
        } else if (opt == "--iters" && hasValue) {
            cfg.iters = std::atoi(argv[++i]);
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
        } else if (opt == "--pruning" && hasValue) {
            cfg.pruning = (std::atoi(argv[++i]) != 0);
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
//   vector    -> N x 1   (Harmony-vector, no dimension pipeline)
//   dimension -> 1 x N   (Harmony-dimension, no vector partitions)
//   harmony   -> the most square split N allows
inline bool resolveGrid(Config& cfg, int numWorkers) {
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
    } else if (cfg.mode == "harmony") {
        cfg.bDim = 1;
        for (int d = 1; d * d <= numWorkers; ++d) {
            if (numWorkers % d == 0) {
                cfg.bDim = d;
            }
        }
        cfg.bVec = numWorkers / cfg.bDim;
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
