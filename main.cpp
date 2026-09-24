#include <iostream>
#include <memory>

#include <mpi.h>

#include "src/config.h"
#include "src/node/master_node.h"
#include "src/node/worker_node.h"

// Build and run (from the project root, so Data/ resolves):
//   scripts/build.sh
//   mpirun -n 5 ./main                 1 master + 4 workers
//   mpirun -n 5 ./main --mode auto     let the cost model pick the layout
//   ./main --help-ish                  any bad option prints the full list
//
// Every process runs this same main. Rank 0 becomes the master and drives
// everything; ranks 1..N become workers and sit in a receive loop.

int main(int argc, char** argv) {
    // MPI: every process starts here, mpirun made them. FUNNELED says what is
    // true of this program -- OpenMP is used inside accumulate() and the
    // top-K pick, but every MPI call is made from the main thread. The sample
    // asks for THREAD_MULTIPLE because it calls MPI from inside its OpenMP
    // regions; that level makes MPI take locks internally, which would show
    // up in the measurements here for no reason.
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // which process am I -> master or worker
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // how many of us -> workers is size - 1

    if (provided < MPI_THREAD_FUNNELED && rank == harmony::MASTER_RANK) {
        std::cerr << "warning: MPI gave thread level " << provided
                  << ", below the requested FUNNELED" << std::endl;
    }

    // every rank sees the same argv, so they all parse it rather than having
    // the master broadcast the settings
    harmony::Config cfg;
    if (!harmony::parseArgs(argc, argv, cfg) ||
        !harmony::resolveGrid(cfg, size - 1)) {
        MPI_Finalize();
        return 1;
    }

    std::unique_ptr<harmony::Node> node;
    if (rank == harmony::MASTER_RANK) {
        node = std::make_unique<harmony::MasterNode>(size - 1, cfg);
    } else {
        node = std::make_unique<harmony::WorkerNode>(rank, cfg);
    }

    int ret = node->run();

    MPI_Finalize();
    return ret;
}
