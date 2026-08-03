#include <memory>

#include <mpi.h>

#include "src/node/master_node.h"
#include "src/node/worker_node.h"

// Run with:  mpirun -n 5 ./harmony     (1 master + 4 workers)
//
// Every process runs this same main. Rank 0 becomes the master and drives
// everything; ranks 1..N become workers and sit in a receive loop.

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::unique_ptr<harmony::Node> node;
    if (rank == harmony::MASTER_RANK) {
        node = std::make_unique<harmony::MasterNode>(size - 1);
    } else {
        node = std::make_unique<harmony::WorkerNode>(rank);
    }

    int ret = node->run();

    MPI_Finalize();
    return ret;
}
