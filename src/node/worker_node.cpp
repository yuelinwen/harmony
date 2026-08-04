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
        bDim_ = setup[2];

        // Where this worker sits in its row. Which end of a chain it is
        // depends on the job, since clusters start at different columns.
        myCol_ = (id_ - 1) % bDim_;
        rowBase_ = id_ - myCol_;
        aliveAtStage_.assign(bDim_, 0);

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

        // Forwarding has to be non-blocking. Clusters start at different
        // columns, so two workers can be sending to each other at the same
        // moment; with blocking sends both would sit in MPI_Send waiting for
        // the other to post a receive, and the row would deadlock. This is
        // why the paper uses MPI_Isend / MPI_Irecv (Section 5).
        //
        // An outgoing buffer must stay untouched until its send completes, and
        // sums/alive are reused by the next job, so sends go out of a small
        // rotating pool instead.
        int slots = bDim_ + 1;
        std::vector<std::vector<float>> outSums(slots);
        std::vector<std::vector<char>> outAlive(slots);
        std::vector<MPI_Request> reqSums(slots, MPI_REQUEST_NULL);
        std::vector<MPI_Request> reqAlive(slots, MPI_REQUEST_NULL);
        int slot = 0;

        while (true) {
            int job[4];
            MPI_Recv(job, 4, MPI_INT, MASTER_RANK, TAG_JOB,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (job[0] == JOB_SHUTDOWN) {
                MPI_Send(aliveAtStage_.data(), bDim_, MPI_LONG, MASTER_RANK,
                         TAG_STATS, MPI_COMM_WORLD);
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
            int startCol = job[3];

            // This cluster's chain runs startCol, startCol+1, ... around the
            // row. Where this worker sits in that chain decides everything.
            int stage = (myCol_ - startCol + bDim_) % bDim_;   // 0 = first stop
            bool isFirst = (stage == 0);
            bool isLast = (myCol_ == (startCol + bDim_ - 1) % bDim_);
            int prevRank = rowBase_ + (myCol_ - 1 + bDim_) % bDim_;
            int nextRank = isLast ? MASTER_RANK
                                  : rowBase_ + (myCol_ + 1) % bDim_;

            float threshold = 0.0f;
            MPI_Recv(&threshold, 1, MPI_FLOAT, MASTER_RANK, TAG_THRESHOLD,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (isFirst) {
                // first stop of the chain: start the running totals from scratch
                sums.assign(n, 0.0f);
                alive.assign(n, 1);
                for (int j = 0; j < skip; ++j) {
                    alive[j] = 0;   // the master prewarmed these already
                }
            } else {
                sums.resize(n);
                alive.resize(n);
                MPI_Recv(sums.data(), n, MPI_FLOAT, prevRank, TAG_SUMS,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(alive.data(), n, MPI_CHAR, prevRank, TAG_ALIVE,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            accumulate(querySlice.data(), clusterId, threshold, sums, alive);

            for (int j = 0; j < n; ++j) {
                if (alive[j]) {
                    aliveAtStage_[stage] = aliveAtStage_[stage] + 1;
                }
            }

            // reclaim this slot before overwriting it
            MPI_Wait(&reqSums[slot], MPI_STATUS_IGNORE);
            MPI_Wait(&reqAlive[slot], MPI_STATUS_IGNORE);

            outSums[slot] = sums;
            outAlive[slot] = alive;
            MPI_Isend(outSums[slot].data(), n, MPI_FLOAT, nextRank, TAG_SUMS,
                      MPI_COMM_WORLD, &reqSums[slot]);
            MPI_Isend(outAlive[slot].data(), n, MPI_CHAR, nextRank, TAG_ALIVE,
                      MPI_COMM_WORLD, &reqAlive[slot]);
            slot = (slot + 1) % slots;
        }

        // let any still-outstanding sends finish before the process goes away
        for (int s = 0; s < slots; ++s) {
            MPI_Wait(&reqSums[s], MPI_STATUS_IGNORE);
            MPI_Wait(&reqAlive[s], MPI_STATUS_IGNORE);
        }

        return 0;
    }

}  // namespace harmony
