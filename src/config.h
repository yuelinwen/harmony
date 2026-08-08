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
    // One prefix stands for the three files convert_hdf5.py writes:
    //   <prefix>_base.bin  <prefix>_query.bin  <prefix>_gt.bin
    std::string data = "Data/sift";

    int nlist = 256;    // clusters in the index
    int iters = 10;     // kmeans rounds

    std::string mode = "harmony";   // harmony | vector | dimension | auto
    int bVec = 0;       // set directly to override the mode
    int bDim = 0;

    // Cost model (paper Section 4.2.1), used when mode is "auto".
    //   alpha    weight on the imbalance term of C(pi,Q)
    //   commCost cost of one byte between workers, relative to one multiply-
    //            add. This is the number that decides whether splitting by
    //            dimension is worth its extra hops, and it is a property of
    //            the hardware, so it has to be measured.
    //
    //            Two ways to get it, which agreed on the test cluster:
    //              flops/s divided by bytes/s -- 12.7 GFLOP/s over ~109 MB/s
    //              gives about 116
    //              sweeping it until the model ranks the three grids the way
    //              they actually measure -- anything from 10 upwards
    //
    //            The default suits that cluster (1 Gb/s links). Shared memory
    //            is nearer 1, and a 100 Gb/s fabric like the paper's would be
    //            around 1-10, which is why the paper can afford to split by
    //            dimension where this cluster cannot.
    //   warmup   queries used to learn which clusters are hot before the plan
    //            is fixed (the paper's pre-query phase, Section 6.2.1)
    double alpha = 0.3;
    double commCost = 100.0;
    int warmup = 1000;

    int nprobe = 32;    // clusters visited per query
    int k = 100;        // neighbours returned
    int nq = 100;       // queries to run

    // Queries processed together (paper Algorithm 1 line 13, QueryBatch).
    // Queries that probe the same cluster share one visit to it, so the
    // vectors are read once and used for all of them. That reuse is also what
    // makes the gemm path worth taking: measured 0.64x at batch 1 but 3.16x
    // at batch 32, against the plain loop.
    int batch = 32;

    // Use MKL for the first dimension slice, where nothing can be pruned yet
    // and the whole block has to be computed anyway. Later slices stay on the
    // scalar loop, which is what can stop early. Ignored if MKL is not
    // compiled in.
    bool mkl = true;
    int prewarm = 500;  // vectors used to seed the heap; 0 turns it off
    bool pruning = true;

    // OpenMP threads inside each worker (paper Section 5). One MPI process
    // per node with as many threads as it has cores is the layout the paper
    // runs; the default of 1 suits a laptop, where the processes already
    // occupy every core.
    int threads = 1;
};

inline void printUsage(const char* prog) {
    std::cerr
        << "usage: mpirun -n <N+1> " << prog << " [options]\n"
        << "\n"
        << "data and index\n"
        << "  --data <prefix>    reads <prefix>_base/_query/_gt.bin (Data/sift)\n"
        << "  --nlist <int>      clusters in the index              (256)\n"
        << "  --iters <int>      kmeans rounds                      (10)\n"
        << "\n"
        << "how the workers are laid out\n"
        << "  --mode <name>      vector | dimension | harmony | auto (harmony)\n"
        << "                       vector    = N x 1, no dimension pipeline\n"
        << "                       dimension = 1 x N, no vector partitions\n"
        << "                       harmony   = the most square split of N\n"
        << "                       auto      = let the cost model decide\n"
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
        << "  --prewarm <int>    heap seed size, 0 = off            (500)\n"
        << "  --pruning <0|1>    dimension-level pruning            (1)\n"
        << "  --mkl <0|1>        gemm for the first slice           (1)\n"
        << "\n"
        << "cost model, only read when --mode auto\n"
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
    } else if (cfg.mode == "auto") {
        cfg.bVec = 1;          // provisional; choosePlan() replaces it
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
