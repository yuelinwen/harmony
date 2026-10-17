// netbench: what the interconnect does for the traffic this program makes.
//
// Run from the project root, two ranks on two machines:
//
//   mpirun -n 2 --hostfile scripts/hosts.txt -N 1 ./netbench
//
// Two patterns, because they answer different questions:
//   ping-pong   latency, and bandwidth when nothing overlaps
//   stream      several Isends in flight, which is what a chain hop looks like
//
// Why it exists. --commcost prices a transferred byte in multiply-adds, and
// it was set to 100 from a nominal link speed rather than a measurement. It
// also decides nothing on its own: the grid it picks is the same for any
// value above about 1, so a wrong figure here has never changed an answer.
// What it does change is the reading of every result, and the next pruning
// backend will move message sizes around, so the numbers need to be real.
//
// Measured on this cluster (two VMs, OpenMPI). Two runs, because one is not
// enough -- the 16 MB figure moved 28% between them, so read these as a
// range and re-run before quoting any of them:
//
//     message      ping-pong             stream
//        4 KB     10 - 12 MB/s     330 -  356 MB/s
//       64 KB     57 - 60 MB/s      81 -   94 MB/s
//        1 MB    590 - 670 MB/s    590 -  863 MB/s
//        4 MB    980 -1180 MB/s   1063 - 1257 MB/s
//       16 MB   1310 -1616 MB/s   1444 - 2011 MB/s
//     one-way latency 395 - 414 us
//
// Three things to read off it. Bandwidth depends strongly on message size,
// so a single number is meaningless -- the chain forwards about 2 MB at a
// time, which lands near 1000 MB/s and makes --commcost closer to 13 than
// 100. Latency is 395 us, an order of magnitude worse than a physical 10GbE
// link, because the network is virtual. And the gap between the two columns
// at 4 KB, 30x, is the non-blocking send earning its keep: with several
// buffers in flight the latency is hidden, and with one it is not.

#include <cstdio>
#include <vector>

#include <mpi.h>

namespace {

const int kTag = 7;

double pingpong(int rank, size_t bytes, int iters) {
    std::vector<char> buf(bytes);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) {
        if (rank == 0) {
            MPI_Send(buf.data(), (int)bytes, MPI_BYTE, 1, kTag, MPI_COMM_WORLD);
            MPI_Recv(buf.data(), (int)bytes, MPI_BYTE, 1, kTag, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(buf.data(), (int)bytes, MPI_BYTE, 0, kTag, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Send(buf.data(), (int)bytes, MPI_BYTE, 0, kTag, MPI_COMM_WORLD);
        }
    }
    return MPI_Wtime() - t0;
}

// Several buffers outstanding at once, the way a worker forwards blocks.
double stream(int rank, size_t bytes, int iters, int depth) {
    std::vector<std::vector<char> > buf(depth, std::vector<char>(bytes));
    std::vector<MPI_Request> req(depth, MPI_REQUEST_NULL);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) {
        int s = i % depth;
        if (rank == 0) {
            MPI_Wait(&req[s], MPI_STATUS_IGNORE);
            MPI_Isend(buf[s].data(), (int)bytes, MPI_BYTE, 1, kTag,
                      MPI_COMM_WORLD, &req[s]);
        } else {
            MPI_Recv(buf[s].data(), (int)bytes, MPI_BYTE, 0, kTag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    if (rank == 0) {
        for (int s = 0; s < depth; ++s) {
            MPI_Wait(&req[s], MPI_STATUS_IGNORE);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime() - t0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        if (rank == 0) {
            std::printf("needs exactly 2 ranks, got %d\n", size);
        }
        MPI_Finalize();
        return 1;
    }

    char name[MPI_MAX_PROCESSOR_NAME];
    int len = 0;
    MPI_Get_processor_name(name, &len);
    std::printf("rank %d on %s\n", rank, name);
    std::fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    size_t sizes[] = {4096, 65536, 1u << 20, 4u << 20, 16u << 20};
    int n = 5;

    if (rank == 0) {
        std::printf("\n%12s %10s %14s %14s\n", "bytes", "iters",
                    "pingpong MB/s", "stream MB/s");
    }

    for (int i = 0; i < n; ++i) {
        size_t b = sizes[i];
        int iters = (b <= 65536) ? 2000 : (b <= (1u << 20) ? 500 : 100);

        pingpong(rank, b, 5);                       // warm up
        double tp = pingpong(rank, b, iters);
        stream(rank, b, 20, 8);                     // warm up
        double ts = stream(rank, b, iters, 8);

        if (rank == 0) {
            // ping-pong moves the buffer twice per iteration
            double mbPP = 2.0 * b * iters / tp / 1e6;
            double mbST = (double)b * iters / ts / 1e6;
            std::printf("%12zu %10d %14.1f %14.1f\n", b, iters, mbPP, mbST);
        }
    }

    // Latency, from the smallest round trip.
    double t = pingpong(rank, 8, 5000);
    if (rank == 0) {
        std::printf("\none-way latency: %.1f us\n", t / 5000 / 2 * 1e6);
    }

    MPI_Finalize();
    return 0;
}
