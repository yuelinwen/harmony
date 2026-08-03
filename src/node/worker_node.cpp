#include "worker_node.h"

#include <iostream>

#include <mpi.h>

#include "../comm/messages.h"
#include "../index/distance.h"

namespace harmony {
    void WorkerNode::setDimCount(int myDim) {
        myDim_ = myDim;
    }

    void WorkerNode::addCluster(int clusterId, const std::vector<int>& ids,
                                const std::vector<float>& data) {
        ClusterBlock block;
        block.clusterId = clusterId;
        block.ids = ids;
        block.data = data;      // already sliced by the master

        blocks_.push_back(block);
    }

    long WorkerNode::vectorCount() const {
        long n = 0;
        for (size_t b = 0; b < blocks_.size(); ++b) {
            n = n + (long)blocks_[b].ids.size();
        }
        return n;
    }

    void WorkerNode::accumulate(const float* querySlice, int clusterId, float threshold,
                                std::vector<float>& sums, std::vector<char>& alive) {
        for (int b = 0; b < (int)blocks_.size(); ++b) {
            if (blocks_[b].clusterId != clusterId) {
                continue;
            }
            const ClusterBlock& block = blocks_[b];
            for (int v = 0; v < (int)block.ids.size(); ++v) {
                if (!alive[v]) {
                    continue;      // an earlier worker already dropped it
                }
                sums[v] = sums[v] + l2DistanceSquared(querySlice,
                                                      &block.data[(size_t)v * myDim_],
                                                      myDim_);
                if (sums[v] > threshold) {
                    alive[v] = 0;  // cannot reach the top-K, stop here
                }
            }
        }
    }



    void WorkerNode::receiveSetup() {
        int setup[3];
        MPI_Recv(setup, 3, MPI_INT, MASTER_RANK, TAG_SETUP,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        myDim_ = setup[0];
        int nClusters = setup[1];   // only the clusters of this worker's row
        int bDim = setup[2];

        // My place in the row is (id_-1) % bDim. The first slice starts the
        // running totals; the last one reports back to the master; everyone
        // else forwards to the next slice in the row.
        int col = (id_ - 1) % bDim;
        first_ = (col == 0);
        nextRank_ = (col == bDim - 1) ? MASTER_RANK : id_ + 1;

        for (int c = 0; c < nClusters; ++c) {
            int header[2];
            MPI_Recv(header, 2, MPI_INT, MASTER_RANK, TAG_CLUSTER,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int clusterId = header[0];
            int nIds = header[1];

            std::vector<int> ids(nIds);
            MPI_Recv(ids.data(), nIds, MPI_INT, MASTER_RANK, TAG_IDS,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::vector<float> data((size_t)nIds * myDim_);
            MPI_Recv(data.data(), (int)data.size(), MPI_FLOAT, MASTER_RANK, TAG_DATA,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            addCluster(clusterId, ids, data);
        }

        std::cout << "worker " << id_ << " ready: " << vectorCount()
                  << " vectors x " << myDim_ << " dims" << std::endl;
    }

    int WorkerNode::run() {
        running_ = true;
        receiveSetup();

        std::vector<float> querySlice(myDim_);
        std::vector<float> sums;
        std::vector<char> alive;

        while (true) {
            int job[3];
            MPI_Recv(job, 3, MPI_INT, MASTER_RANK, TAG_JOB,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (job[0] == JOB_SHUTDOWN) {
                MPI_Send(&aliveCount_, 1, MPI_LONG, MASTER_RANK, TAG_STATS,
                         MPI_COMM_WORLD);
                break;
            }

            if (job[0] == JOB_QUERY) {
                MPI_Recv(querySlice.data(), myDim_, MPI_FLOAT, MASTER_RANK, TAG_QUERY,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                continue;
            }

            int clusterId = job[0];
            int n = job[1];
            int skip = job[2];

            float threshold = 0.0f;
            MPI_Recv(&threshold, 1, MPI_FLOAT, MASTER_RANK, TAG_THRESHOLD,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (first_) {
                // first slice in the row: start the running totals from scratch
                sums.assign(n, 0.0f);
                alive.assign(n, 1);
                for (int j = 0; j < skip; ++j) {
                    alive[j] = 0;   // the master prewarmed these already
                }
            } else {
                sums.resize(n);
                alive.resize(n);
                MPI_Recv(sums.data(), n, MPI_FLOAT, id_ - 1, TAG_SUMS,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(alive.data(), n, MPI_CHAR, id_ - 1, TAG_ALIVE,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            accumulate(querySlice.data(), clusterId, threshold, sums, alive);

            for (int j = 0; j < n; ++j) {
                if (alive[j]) {
                    aliveCount_ = aliveCount_ + 1;
                }
            }

            MPI_Send(sums.data(), n, MPI_FLOAT, nextRank_, TAG_SUMS, MPI_COMM_WORLD);
            MPI_Send(alive.data(), n, MPI_CHAR, nextRank_, TAG_ALIVE, MPI_COMM_WORLD);
        }

        return 0;
    }

}  // namespace harmony
