#include "worker_node.h"

#include <iostream>

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HARMONY_USE_MKL
#include <mkl.h>
#endif

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

    // ||v||^2 once per vector, so the gemm path only has to produce the dot
    // products. Cheap here, and it never changes.
    block.norm.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const float* v = &block.data[i * myDim_];
        float s = 0.0f;
        for (int j = 0; j < myDim_; ++j) {
            s = s + v[j] * v[j];
        }
        block.norm[i] = s;
    }

    blocks_.push_back(block);
}

long WorkerNode::vectorCount() const {
    long n = 0;
    for (size_t b = 0; b < blocks_.size(); ++b) {
        n = n + (long)blocks_[b].ids.size();
    }
    return n;
}

void WorkerNode::accumulate(const float* queries, int m, int clusterId,
                            const float* thresholds, bool first,
                            std::vector<float>& sums) {
#ifndef HARMONY_USE_MKL
    (void)first;   // only the gemm path cares which end of the chain this is
#endif
    for (int b = 0; b < (int)blocks_.size(); ++b) {
        if (blocks_[b].clusterId != clusterId) {
            continue;
        }
        const ClusterBlock& block = blocks_[b];
        int n = (int)block.ids.size();

#ifdef HARMONY_USE_MKL
        // At the head of the chain every candidate has to be computed -- there
        // is no running total yet to prune against -- so the whole m x n block
        // can go through one gemm. Expanding ||a-b||^2 into
        // ||a||^2 + ||b||^2 - 2ab turns it into a matrix multiply, and the
        // vector norms were precomputed when the block arrived.
        //
        // Later stages do not take this path: their whole point is to skip
        // most of the candidates, which a dense multiply cannot do.
        if (first && useMkl_ && m > 1) {
            std::vector<float> qn(m);
            for (int q = 0; q < m; ++q) {
                const float* qv = &queries[(size_t)q * myDim_];
                qn[q] = cblas_sdot(myDim_, qv, 1, qv, 1);
            }

            std::vector<float> dot((size_t)m * n);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        m, n, myDim_, 1.0f,
                        queries, myDim_, block.data.data(), myDim_,
                        0.0f, dot.data(), n);

            #pragma omp parallel for schedule(static)
            for (int q = 0; q < m; ++q) {
                float t = thresholds[q];
                float* row = &sums[(size_t)q * n];
                const float* d = &dot[(size_t)q * n];
                for (int v = 0; v < n; ++v) {
                    row[v] = qn[q] + block.norm[v] - 2.0f * d[v];
                    if (row[v] > t) {
                        row[v] = PRUNED;
                    }
                }
            }
            return;
        }
#endif

        // The scalar path, and the only one that can stop early. Iterations
        // touch different sums entries, so the threads never write the same
        // memory and no locking is needed (paper Section 5, node-level
        // parallelism; across nodes the work is already split by MPI).
        #pragma omp parallel for schedule(static)
        for (int q = 0; q < m; ++q) {
            const float* qv = &queries[(size_t)q * myDim_];
            float t = thresholds[q];
            float* row = &sums[(size_t)q * n];

            for (int v = 0; v < n; ++v) {
                if (row[v] >= PRUNED) {
                    continue;      // an earlier worker already dropped it
                }
                row[v] = row[v] + l2DistanceSquared(qv, &block.data[(size_t)v * myDim_],
                                                    myDim_);
                if (row[v] > t) {
                    row[v] = PRUNED;   // cannot reach the top-K, stop here
                }
            }
        }
        return;
    }
}

void WorkerNode::receiveSetup() {
    int setup[4];
    MPI_Recv(setup, 4, MPI_INT, MASTER_RANK, TAG_SETUP,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    myDim_ = setup[0];
    int nClusters = setup[1];   // only the clusters of this worker's row
    bDim_ = setup[2];
    batch_ = setup[3];

    // Where this worker sits in its row. Which end of a chain it is depends
    // on the job, since clusters start at different columns.
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

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char* kernel = "loop";
#ifdef HARMONY_USE_MKL
    if (useMkl_) {
        kernel = "loop+mkl";
    }
#endif
    std::cout << "worker " << id_ << " ready: " << vectorCount()
              << " vectors x " << myDim_ << " dims, "
              << threads << " thread(s), " << kernel << std::endl;
}

int WorkerNode::run() {
    running_ = true;
    useMkl_ = cfg_.mkl;

#ifdef _OPENMP
    omp_set_num_threads(cfg_.threads);
#endif

    receiveSetup();

    std::vector<float> queries;          // batch_ slices of myDim_ floats
    std::vector<int> qidx;               // which of them this job wants
    std::vector<float> picked;           // those slices, packed together
    std::vector<float> thresholds;
    std::vector<float> sums;

    // Forwarding has to be non-blocking. Clusters start at different columns,
    // so two workers can be sending to each other at the same moment; with
    // blocking sends both would sit in MPI_Send waiting for the other to post
    // a receive, and the row would deadlock. This is why the paper uses
    // MPI_Isend / MPI_Irecv (Section 5).
    //
    // An outgoing buffer must stay untouched until its send completes, and
    // sums is reused by the next job, so sends go out of a small rotating
    // pool instead.
    int slots = bDim_ + 1;
    std::vector<std::vector<float>> outSums(slots);
    std::vector<MPI_Request> reqSums(slots, MPI_REQUEST_NULL);
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
            queries.resize((size_t)batch_ * myDim_);
            MPI_Recv(queries.data(), (int)queries.size(), MPI_FLOAT,
                     MASTER_RANK, TAG_QUERY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
        }

        int clusterId = job[0];
        int n = job[1];
        int startCol = job[2];
        int m = job[3];

        // This cluster's chain runs startCol, startCol+1, ... around the row.
        // Where this worker sits in that chain decides everything.
        int stage = (myCol_ - startCol + bDim_) % bDim_;   // 0 = first stop
        bool isFirst = (stage == 0);
        bool isLast = (myCol_ == (startCol + bDim_ - 1) % bDim_);
        int prevRank = rowBase_ + (myCol_ - 1 + bDim_) % bDim_;
        int nextRank = isLast ? MASTER_RANK
                              : rowBase_ + (myCol_ + 1) % bDim_;

        qidx.resize(m);
        MPI_Recv(qidx.data(), m, MPI_INT, MASTER_RANK, TAG_QIDX,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        thresholds.resize(m);
        MPI_Recv(thresholds.data(), m, MPI_FLOAT, MASTER_RANK, TAG_THRESHOLD,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // gemm wants the participating slices contiguous
        picked.resize((size_t)m * myDim_);
        for (int q = 0; q < m; ++q) {
            const float* src = &queries[(size_t)qidx[q] * myDim_];
            std::copy(src, src + myDim_, &picked[(size_t)q * myDim_]);
        }

        size_t total = (size_t)m * n;
        if (isFirst) {
            sums.assign(total, 0.0f);
        } else {
            sums.resize(total);
            MPI_Recv(sums.data(), (int)total, MPI_FLOAT, prevRank, TAG_SUMS,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        accumulate(picked.data(), m, clusterId, thresholds.data(), isFirst, sums);

        for (size_t j = 0; j < total; ++j) {
            if (sums[j] < PRUNED) {
                aliveAtStage_[stage] = aliveAtStage_[stage] + 1;
            }
        }

        // reclaim this slot before overwriting it
        MPI_Wait(&reqSums[slot], MPI_STATUS_IGNORE);

        outSums[slot] = sums;
        MPI_Isend(outSums[slot].data(), (int)total, MPI_FLOAT, nextRank, TAG_SUMS,
                  MPI_COMM_WORLD, &reqSums[slot]);
        slot = (slot + 1) % slots;
    }

    // let any still-outstanding sends finish before the process goes away
    for (int s = 0; s < slots; ++s) {
        MPI_Wait(&reqSums[s], MPI_STATUS_IGNORE);
    }

    return 0;
}

}  // namespace harmony
