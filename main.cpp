#include <memory>

#include <mpi.h>

#include "src/config.h"
#include "src/node/master_node.h"
#include "src/node/worker_node.h"

// Build and run (from the project root, so Data/ resolves):
//   ./build.sh
//   mpirun -n 5 ./main                 1 master + 4 workers
//   mpirun -n 5 ./main --mode auto     let the cost model pick the layout
//   ./main --help-ish                  any bad option prints the full list
//
// Every process runs this same main. Rank 0 becomes the master and drives
// everything; ranks 1..N become workers and sit in a receive loop.

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
