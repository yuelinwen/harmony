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
#include "../engine/stopwatch.h"
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
    // A worker is only ever sent clusters its row owns, so not finding one
    // means the grid and the dispatch disagree. Left unnoticed it would return
    // whatever the previous worker accumulated -- a plausible-looking but
    // wrong answer -- so it stops the whole job instead.
    const ClusterBlock* found = nullptr;
    for (int b = 0; b < (int)blocks_.size(); ++b) {
        if (blocks_[b].clusterId == clusterId) {
            found = &blocks_[b];
            break;
        }
    }
    if (found == nullptr) {
        std::cerr << "worker " << id_ << ": cluster " << clusterId
                  << " was never sent here" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const ClusterBlock& block = *found;
    int n = (int)block.ids.size();

#ifdef HARMONY_USE_MKL
    // At the head of the chain nothing can be pruned yet, so the whole
    // m x n block has to be computed and can go through one gemm.
    // Expanding ||a-b||^2 into ||a||^2 + ||b||^2 - 2ab makes it a matrix
    // multiply; the norms were precomputed when the block arrived.
    //
    // Later stages skip most candidates, which a dense multiply cannot.
    if (first && useMkl_ && m > 1) {
        // MKL: ||q||^2 for each query in the batch
        std::vector<float> qn(m);
        for (int q = 0; q < m; ++q) {
            const float* qv = &queries[(size_t)q * myDim_];
            qn[q] = cblas_sdot(myDim_, qv, 1, qv, 1);
        }

        // MKL: one matrix multiply produces all m x n dot products
        std::vector<float> dot((size_t)m * n);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, n, myDim_, 1.0f,
                    queries, myDim_, block.data.data(), myDim_,
                    0.0f, dot.data(), n);

        // OpenMP: turn the dot products into distances, one thread per query
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
    // OpenMP: one thread per query, no locking needed (see above)
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

    // The chain table, built by the master and handed over row by row. A
    // worker never builds it itself: only the master can change it, which is
    // what any load-aware reordering would need.
    std::vector<int> table(3 * bDim_);
    MPI_Recv(table.data(), 3 * bDim_, MPI_INT, MASTER_RANK, TAG_ORDER,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    nextOf_.assign(table.begin(), table.begin() + bDim_);
    prevOf_.assign(table.begin() + bDim_, table.begin() + 2 * bDim_);
    stageOf_.assign(table.begin() + 2 * bDim_, table.end());

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
    omp_set_num_threads(cfg_.threads);   // OpenMP: --threads takes effect here
#endif

    receiveSetup();

    std::vector<float> queries;          // batch_ slices of myDim_ floats
    std::vector<float> picked;           // the slices one job wants, packed

    // Forwarding has to be non-blocking. Clusters start at different columns,
    // so two workers can be sending to each other at once; with blocking sends
    // both would wait in MPI_Send for the other to receive, and the row would
    // deadlock. This is why the paper uses MPI_Isend / MPI_Irecv (§5).
    //
    // An outgoing buffer must stay untouched until its send completes, so
    // sends go out of a rotating pool.
    int slots = 2 * bDim_ + 2;
    std::vector<std::vector<float>> outSums(slots);       // mid-chain: totals
    std::vector<std::vector<Candidate>> outTop(slots);    // chain tail: top-k
    std::vector<MPI_Request> reqSums(slots, MPI_REQUEST_NULL);
    int slot = 0;

    // A cluster this worker has been handed but not finished. Several stay
    // open at once on purpose: one still waiting on its upstream must not hold
    // up another whose turn has already come. Taking them strictly in the
    // order the master sent them was costing 34% of a worker's time in
    // harmony mode and 70% in dimension mode, all of it blocked in one
    // MPI_Recv with other work sitting right there. The sample avoids the
    // same trap by testing every pending block instead of taking them in
    // order; the tags in messages.h are what make that safe.
    struct Pending {
        int clusterId;
        int n;
        int m;
        int stage;
        int slotTag;             // names this cluster's chain tags
        bool isFirst;
        bool isLast;
        int prevRank;
        int nextRank;
        std::vector<int> qidx;
        std::vector<float> thresholds;
        std::vector<float> sums;
    };
    std::vector<Pending> pend;
    std::vector<MPI_Request> pendReq;   // upstream receive; NULL for a head

    // The clock starts at the first job, not at setup: loading the index is
    // measured separately and would otherwise swamp everything.
    Stopwatch run;
    Stopwatch phase;
    bool started = false;
    bool done = false;

    while (!done) {
        // ---- 1. take every job already waiting, and block only when there
        //         is nothing open to work on ----
        while (true) {
            int waiting = 1;
            if (!pend.empty()) {
                // MPI: peek rather than block -- there is real work in hand
                MPI_Iprobe(MASTER_RANK, TAG_JOB, MPI_COMM_WORLD, &waiting,
                           MPI_STATUS_IGNORE);
                if (!waiting) {
                    break;
                }
            }

            phase.reset();
            int job[5];
            MPI_Recv(job, 5, MPI_INT, MASTER_RANK, TAG_JOB,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // only a wait with nothing else to do is idleness
            double waited = pend.empty() ? phase.seconds() : 0.0;
            if (!started) {
                run.reset();   // the wait for the very first job is setup
                started = true;
                waited = 0.0;
            }

            // total_ stops at the last real job rather than at the shutdown
            // message: with --check the master runs its single-machine
            // reference afterwards, and counting that wait would read as idle
            // workers. The master sends the bookkeeping jobs only once every
            // cluster has reported, so nothing is open when they arrive.
            if (job[0] == JOB_SHUTDOWN) {
                done = true;
                break;
            }

            // The wait for either bookkeeping message is not idle time: the
            // master is checking answers or printing, not withholding work.
            if (job[0] == JOB_STATS) {
                MPI_Send(aliveAtStage_.data(), bDim_, MPI_LONG, MASTER_RANK,
                         TAG_STATS, MPI_COMM_WORLD);
                double times[6] = {total_, idle_, recv_, compute_, send_,
                                   (double)jobs_};
                MPI_Send(times, 6, MPI_DOUBLE, MASTER_RANK, TAG_TIMES,
                         MPI_COMM_WORLD);
                continue;
            }

            // Start of a counted stretch: forget everything before it. Sent
            // before the pass that gets reported, so an earlier --loop pass or
            // an earlier --nprobes value does not leak into these numbers.
            if (job[0] == JOB_RESET) {
                aliveAtStage_.assign(bDim_, 0);
                idle_ = 0.0;
                recv_ = 0.0;
                compute_ = 0.0;
                send_ = 0.0;
                jobs_ = 0;
                total_ = 0.0;
                run.reset();
                continue;
            }

            idle_ = idle_ + waited;

            if (job[0] == JOB_QUERY) {
                queries.resize((size_t)batch_ * myDim_);
                MPI_Recv(queries.data(), (int)queries.size(), MPI_FLOAT,
                         MASTER_RANK, TAG_QUERY, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                total_ = run.seconds();
                continue;
            }

            // ---- a cluster: open it and post its upstream receive ----
            Pending p;
            p.clusterId = job[0];
            p.n = job[1];
            int item = job[2];
            p.m = job[3];
            p.slotTag = job[4];

            // Everything about this worker's part in the chain comes out of
            // the table: which item it is decides where the chain starts, and
            // the table says who is on either side.
            int prevCol = prevOf_[item];
            int nextCol = nextOf_[item];
            p.stage = stageOf_[item];        // 0 = first stop
            p.isFirst = (prevCol < 0);
            p.isLast = (nextCol < 0);
            p.prevRank = p.isFirst ? MASTER_RANK : rowBase_ + prevCol;
            p.nextRank = p.isLast ? MASTER_RANK : rowBase_ + nextCol;

            p.qidx.resize(p.m);
            MPI_Recv(p.qidx.data(), p.m, MPI_INT, MASTER_RANK, TAG_QIDX,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            p.thresholds.resize(p.m);
            MPI_Recv(p.thresholds.data(), p.m, MPI_FLOAT, MASTER_RANK,
                     TAG_THRESHOLD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            size_t total = (size_t)p.m * (size_t)p.n;
            MPI_Request up = MPI_REQUEST_NULL;
            if (p.isFirst) {
                p.sums.assign(total, 0.0f);   // nothing upstream to wait for
            } else {
                p.sums.resize(total);
                // MPI: posted now, collected later -- the point of the whole
                // arrangement is that this wait overlaps other clusters.
                MPI_Irecv(p.sums.data(), (int)total, MPI_FLOAT, p.prevRank,
                          tagSums(p.slotTag), MPI_COMM_WORLD, &up);
            }
            pend.push_back(std::move(p));
            pendReq.push_back(up);
        }

        if (done) {
            break;
        }
        if (pend.empty()) {
            continue;
        }

        // ---- 2. work on whichever open cluster can be worked on ----
        int pick = -1;
        for (size_t i = 0; i < pendReq.size(); ++i) {
            if (pendReq[i] == MPI_REQUEST_NULL) {
                pick = (int)i;   // a chain head, nothing to wait for
                break;
            }
        }
        if (pick < 0) {
            // MPI: has anyone's upstream landed while we were busy?
            int flag = 0;
            int idx = MPI_UNDEFINED;
            MPI_Testany((int)pendReq.size(), pendReq.data(), &idx, &flag,
                        MPI_STATUS_IGNORE);
            if (flag && idx != MPI_UNDEFINED) {
                pick = idx;
            }
        }
        if (pick < 0) {
            // Nothing ready and nothing else to do. Block, rather than spin
            // the way the sample does with MPI_Test and usleep(100).
            phase.reset();
            int idx = MPI_UNDEFINED;
            MPI_Waitany((int)pendReq.size(), pendReq.data(), &idx,
                        MPI_STATUS_IGNORE);
            recv_ = recv_ + phase.seconds();
            pick = idx;
        }

        Pending& p = pend[pick];
        size_t total = (size_t)p.m * (size_t)p.n;

        // gemm wants the participating slices contiguous
        picked.resize((size_t)p.m * myDim_);
        for (int q = 0; q < p.m; ++q) {
            const float* src = &queries[(size_t)p.qidx[q] * myDim_];
            std::copy(src, src + myDim_, &picked[(size_t)q * myDim_]);
        }

        phase.reset();
        accumulate(picked.data(), p.m, p.clusterId, p.thresholds.data(),
                   p.isFirst, p.sums);
        compute_ = compute_ + phase.seconds();
        jobs_ = jobs_ + 1;

        for (size_t j = 0; j < total; ++j) {
            if (p.sums[j] < PRUNED) {
                aliveAtStage_[p.stage] = aliveAtStage_[p.stage] + 1;
            }
        }

        // reclaim this slot before overwriting it
        phase.reset();
        MPI_Wait(&reqSums[slot], MPI_STATUS_IGNORE);
        send_ = send_ + phase.seconds();

        if (p.isLast) {
            // End of the chain. This is the only worker that ever sees these
            // candidates' full distances, so it can pick the k nearest itself
            // and send just those, instead of handing every running total back
            // for the master to sift (paper §4.3). k is around a hundred
            // against a cluster's few thousand vectors.
            //
            // The ids kept are positions in the cluster, which is what lets
            // the master map them back and still drop the ones prewarm
            // already pushed.
            Stopwatch pickWatch;
            int kSend = (k_ < p.n) ? k_ : p.n;
            Candidate pad;
            pad.id = -1;
            pad.dist = PRUNED;          // a query with fewer than kSend
            outTop[slot].assign((size_t)p.m * kSend, pad);   // survivors pads

            // OpenMP: one thread per query, each writing its own row
            #pragma omp parallel for schedule(static)
            for (int q = 0; q < p.m; ++q) {
                TopKHeap heap(kSend);
                const float* row = &p.sums[(size_t)q * p.n];
                for (int v = 0; v < p.n; ++v) {
                    if (row[v] < PRUNED) {
                        heap.push(v, row[v]);
                    }
                }

                std::vector<Candidate> best = heap.results();   // nearest first
                Candidate* out = &outTop[slot][(size_t)q * kSend];
                for (int j = 0; j < (int)best.size(); ++j) {
                    out[j] = best[j];
                }
            }

            compute_ = compute_ + pickWatch.seconds();

            // MPI: as bytes, since a Candidate is an int beside a float and
            // every rank is the same build (see topk_heap.h).
            MPI_Isend(outTop[slot].data(),
                      (int)((size_t)p.m * kSend * sizeof(Candidate)), MPI_BYTE,
                      MASTER_RANK, tagTopk(p.slotTag), MPI_COMM_WORLD,
                      &reqSums[slot]);
        } else {
            outSums[slot] = std::move(p.sums);
            MPI_Isend(outSums[slot].data(), (int)total, MPI_FLOAT, p.nextRank,
                      tagSums(p.slotTag), MPI_COMM_WORLD, &reqSums[slot]);
        }

        // The ablation arm of Fig. 2(b): waiting here instead of letting the
        // send ride alongside the next cluster is what blocking communication
        // would cost. The send itself stays non-blocking, or a row of workers
        // sending to each other would deadlock.
        if (cfg_.blockSend) {
            phase.reset();
            MPI_Wait(&reqSums[slot], MPI_STATUS_IGNORE);
            send_ = send_ + phase.seconds();
        }

        pend.erase(pend.begin() + pick);
        pendReq.erase(pendReq.begin() + pick);

        total_ = run.seconds();
        slot = (slot + 1) % slots;
    }

    // let any still-outstanding sends finish before the process goes away
    for (int s = 0; s < slots; ++s) {
        MPI_Wait(&reqSums[s], MPI_STATUS_IGNORE);
    }

    return 0;
}

}  // namespace harmony
