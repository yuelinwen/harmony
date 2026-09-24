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

    if (clusterId >= (int)where_.size()) {
        where_.resize(clusterId + 1, -1);
    }
    where_[clusterId] = (int)blocks_.size();
    blocks_.push_back(block);
}

long WorkerNode::vectorCount() const {
    long n = 0;
    for (size_t b = 0; b < blocks_.size(); ++b) {
        n = n + (long)blocks_[b].ids.size();
    }
    return n;
}

// What a worker holds for the duration: its slice of the vectors, the global
// ids that go with them, and the norms the gemm path needs.
//
// ids and norm are what makes a dimension-sliced layout cost more than a
// vector-sliced one. Every worker in a row holds the same clusters and so the
// same id list, cut only across dimensions, so the ids are stored bDim times
// over; and norm is one float per vector per slice, which a single machine
// does not keep at all.
long WorkerNode::memoryBytes() const {
    long b = 0;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const ClusterBlock& cb = blocks_[i];
        b = b + (long)(cb.ids.size() * sizeof(int));
        b = b + (long)(cb.data.size() * sizeof(float));
        b = b + (long)(cb.norm.size() * sizeof(float));
    }
    return b + (long)(where_.size() * sizeof(int));
}

// Head of the chain: sums are all zero and nothing is pruned, so every
// (query, candidate) pair has to be computed. Regrouping those pairs by
// cluster turns them into one dense matrix multiply per cluster --
// this block's queries that probe it, against all of its vectors -- which is
// the case MKL is built for and the scalar loop is not.
//
//   ||q - v||^2 = ||q||^2 + ||v||^2 - 2 (q . v)
//
// the dot products being the matrix product. ||v||^2 came with the cluster
// and never changes.
//
// Only here. Later slices skip most candidates, and a dense multiply cannot
// skip anything, so grouping by cluster would compute what pruning just saved.
bool WorkerNode::accumulateGemm(int firstQ, int len,
                                const std::vector<size_t>& qOff,
                                std::vector<float>& sums, long* alive) {
#ifndef HARMONY_USE_MKL
    (void)firstQ; (void)len; (void)qOff; (void)sums; (void)alive;
    return false;
#else
    // Which of this block's queries probe each cluster, and where each one's
    // run of totals starts. This is the same walk accumulate() does, read by
    // cluster instead of by query.
    byCluster_.resize(blocks_.size());
    for (size_t b = 0; b < byCluster_.size(); ++b) {
        byCluster_[b].clear();
    }
    for (int j = 0; j < len; ++j) {
        size_t off = qOff[j];
        for (int i = 0; i < nprobe_; ++i) {
            int bi = blockOf(probes_[(size_t)(firstQ + j) * nprobe_ + i]);
            if (bi < 0) {
                continue;
            }
            byCluster_[bi].push_back(std::make_pair(j, off));
            off += blocks_[bi].ids.size();
        }
    }

    // One cluster per thread. Two queries probing the same cluster write
    // different runs of sums, and a run belongs to one query, so no two
    // threads ever touch the same element.
    long survivors = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+ : survivors)
    for (int bi = 0; bi < (int)byCluster_.size(); ++bi) {
        const std::vector<std::pair<int, size_t> >& mem = byCluster_[bi];
        int m = (int)mem.size();
        if (m == 0) {
            continue;
        }

        const ClusterBlock& cb = blocks_[bi];
        int n = (int)cb.ids.size();

        // gemm wants the participating queries contiguous
        std::vector<float> q((size_t)m * myDim_);
        std::vector<float> qn(m);
        for (int a = 0; a < m; ++a) {
            const float* src = &queries_[(size_t)(firstQ + mem[a].first) * myDim_];
            std::copy(src, src + myDim_, &q[(size_t)a * myDim_]);
            qn[a] = cblas_sdot(myDim_, src, 1, src, 1);
        }

        std::vector<float> dot((size_t)m * n);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, n, myDim_, 1.0f,
                    q.data(), myDim_, cb.data.data(), myDim_,
                    0.0f, dot.data(), n);

        for (int a = 0; a < m; ++a) {
            float t = thresholds_[firstQ + mem[a].first];
            float* row = &sums[mem[a].second];
            const float* d = &dot[(size_t)a * n];
            for (int v = 0; v < n; ++v) {
                float dist = qn[a] + cb.norm[v] - 2.0f * d[v];
                if (dist > t) {
                    row[v] = PRUNED;
                } else {
                    row[v] = dist;
                    survivors = survivors + 1;
                }
            }
        }
    }

    *alive = survivors;
    return true;
#endif
}

long WorkerNode::accumulate(int firstQ, int len,
                            const std::vector<size_t>& qOff,
                            std::vector<float>& sums, bool first) {
    long survivors = 0;
    if (first && accumulateGemm(firstQ, len, qOff, sums, &survivors)) {
        return survivors;
    }

    // One query at a time, walking its probe list and stopping at the
    // clusters this row holds. That walk is the buffer layout: every worker
    // in the row derives the same one, so nobody has to send an index.
    //
    // OpenMP: a query owns its own run of sums, so the threads never write
    // the same memory and no locking is needed (paper §5, node-level
    // parallelism; across nodes the work is already split by MPI).
    #pragma omp parallel for schedule(static) reduction(+ : survivors)
    for (int j = 0; j < len; ++j) {
        const float* qv = &queries_[(size_t)(firstQ + j) * myDim_];
        float t = thresholds_[firstQ + j];
        size_t off = qOff[j];

        for (int i = 0; i < nprobe_; ++i) {
            int bi = blockOf(probes_[(size_t)(firstQ + j) * nprobe_ + i]);
            if (bi < 0) {
                continue;   // padding, or a cluster another partition holds
            }

            const ClusterBlock& cb = blocks_[bi];
            int n = (int)cb.ids.size();
            float* row = &sums[off];

            for (int v = 0; v < n; ++v) {
                if (row[v] >= PRUNED) {
                    continue;      // an earlier worker already dropped it
                }
                row[v] = row[v] + l2DistanceSquared(qv, &cb.data[(size_t)v * myDim_],
                                                    myDim_);
                if (row[v] > t) {
                    row[v] = PRUNED;   // cannot reach the top-K, stop here
                } else {
                    survivors = survivors + 1;
                }
            }
            off += n;
        }
    }

    return survivors;
}

void WorkerNode::receiveSetup() {
    int setup[5];
    MPI_Recv(setup, 5, MPI_INT, MASTER_RANK, TAG_SETUP,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    myDim_ = setup[0];
    int nClusters = setup[1];   // only the clusters of this worker's row
    bDim_ = setup[2];
    batch_ = setup[3];
    sendSlots_ = setup[4];
    if (sendSlots_ < 1) {
        sendSlots_ = 1;
    }

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
    kernel = "loop+mkl";
#endif
    std::cout << "worker " << id_ << " ready: " << vectorCount()
              << " vectors x " << myDim_ << " dims, "
              << threads << " thread(s), " << kernel << std::endl;
}

int WorkerNode::run() {

#ifdef _OPENMP
    omp_set_num_threads(cfg_.threads);   // OpenMP: --threads takes effect here
#endif

    receiveSetup();

    // Setup is not part of the search, so the clock starts at the first job.
    Stopwatch run;
    Stopwatch phase;
    bool started = false;
    bool done = false;

    // Forwarding has to be non-blocking. Blocks enter the row at different
    // columns, so two workers can be sending to each other at once; with
    // blocking sends both would wait in MPI_Send for the other to receive,
    // and the row would deadlock. This is why the paper uses MPI_Isend /
    // MPI_Irecv (§5).
    //
    // An outgoing buffer must stay untouched until its send completes, so
    // sends go out of a rotating pool.
    //
    // As many slots as the master can have blocks in flight, which it works
    // out and sends at setup. Anything smaller and this worker can run out
    // while a neighbour in the same row has also run out, each waiting for the
    // other to take what it sent -- see TAG_SETUP in comm/messages.h. The pool
    // used to be 2 * bDim + 2, which happened to be enough while --block
    // defaulted to 4 and deadlocked about one run in six once it did not.
    //
    // Costs a vector header per slot; the buffers themselves arrive by move
    // when a block is forwarded, so only the ones actually in flight hold
    // memory.
    int slots = sendSlots_;
    std::vector<std::vector<float>> outSums(slots);       // mid-chain: totals
    std::vector<std::vector<Candidate>> outTop(slots);    // chain tail: top-k
    std::vector<MPI_Request> reqSums(slots, MPI_REQUEST_NULL);
    int slot = 0;

    // A block this worker has been handed but not finished. Several stay open
    // at once on purpose: one still waiting on its upstream must not hold up
    // another whose turn has already come.
    struct Pending {
        int firstQ;
        int len;
        int stage;
        int slotTag;
        bool isFirst;
        bool isLast;
        int prevRank;
        int nextRank;
        std::vector<size_t> qOff;   // where each query's totals start
        std::vector<float> sums;
    };
    std::vector<Pending> pend;
    std::vector<MPI_Request> pendReq;   // upstream receive; NULL for a head

    while (!done) {
        // ---- 1. take every job already waiting, and block only when there
        //         is nothing open to work on ----
        while (true) {
            int waiting = 1;
            if (!pend.empty()) {
                // MPI: peek rather than block -- there is real work in hand
                phase.reset();
                MPI_Iprobe(MASTER_RANK, TAG_JOB, MPI_COMM_WORLD, &waiting,
                           MPI_STATUS_IGNORE);
                poll_ = poll_ + phase.seconds();
                if (!waiting) {
                    break;
                }
            }

            phase.reset();
            int job[5];
            MPI_Recv(job, 5, MPI_INT, MASTER_RANK, TAG_JOB,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Only a wait with nothing else to do is idleness. With a block
            // open this receive is still time spent, just not time lost --
            // it belongs to dispatch, not to waiting.
            double took = phase.seconds();
            double waited = pend.empty() ? took : 0.0;
            if (!pend.empty()) {
                poll_ = poll_ + took;
            }
            if (!started) {
                run.reset();   // the wait for the very first job is setup
                started = true;
                waited = 0.0;
            }

            // total_ stops at the last real job rather than at the shutdown
            // message: with --check the master runs its single-machine
            // reference afterwards, and counting that wait would read as idle
            // workers. The master sends the bookkeeping jobs only once every
            // block has reported, so nothing is open when they arrive.
            if (job[0] == JOB_SHUTDOWN) {
                done = true;
                break;
            }

            // Every wait for the master is idleness, whichever message
            // ends it. Charging the bookkeeping ones elsewhere made idle mean
            // different things at different bDim: at bDim = 1 nothing
            // collects stats per batch, so the batch-boundary wait landed in
            // idle, and at bDim = 8 the same wait landed in admin. The two
            // then could not be compared, and admin looked like a cost of the
            // instrumentation when it was the boundary itself. admin below is
            // only the handling -- the messages, which are small.
            idle_ = idle_ + waited;

            if (job[0] == JOB_STATS) {
                total_ = run.seconds();
                Stopwatch adminWatch;
                MPI_Send(aliveAtStage_.data(), bDim_, MPI_LONG, MASTER_RANK,
                         TAG_STATS, MPI_COMM_WORLD);
                double times[WORKER_TIMES] = {total_, idle_, recv_, compute_,
                                              send_, (double)jobs_,
                                              setup_, poll_, admin_,
                                              (double)memoryBytes()};
                MPI_Send(times, WORKER_TIMES, MPI_DOUBLE, MASTER_RANK,
                         TAG_TIMES, MPI_COMM_WORLD);
                admin_ = admin_ + adminWatch.seconds();
                continue;
            }

            // A new chain table. Safe here and only here: the master sends
            // it between batches, when no block is part-way along a chain.
            if (job[0] == JOB_ORDER) {
                total_ = run.seconds();
                Stopwatch adminWatch;
                std::vector<int> table(3 * bDim_);
                MPI_Recv(table.data(), 3 * bDim_, MPI_INT, MASTER_RANK,
                         TAG_ORDER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                nextOf_.assign(table.begin(), table.begin() + bDim_);
                prevOf_.assign(table.begin() + bDim_, table.begin() + 2 * bDim_);
                stageOf_.assign(table.begin() + 2 * bDim_, table.end());
                admin_ = admin_ + adminWatch.seconds();
                continue;
            }

            // Start of a counted stretch: forget everything before it. Sent
            // before the pass that gets reported, so an earlier --loop pass or
            // an earlier --nprobes value does not leak into these numbers.
            if (job[0] == JOB_RESET) {
                total_ = run.seconds();
                aliveAtStage_.assign(bDim_, 0);
                idle_ = 0.0;
                recv_ = 0.0;
                compute_ = 0.0;
                send_ = 0.0;
                setup_ = 0.0;
                poll_ = 0.0;
                admin_ = 0.0;
                jobs_ = 0;
                total_ = 0.0;
                run.reset();
                continue;
            }

            // Thresholds for a run of the batch, refreshed between
            // partitions. job[1] is where the run starts and job[2] how long.
            if (job[0] == JOB_THRESH) {
                MPI_Recv(&thresholds_[job[1]], job[2], MPI_FLOAT, MASTER_RANK,
                         TAG_THRESHOLD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                continue;
            }

            if (job[0] == JOB_QUERY) {
                int count = job[1];
                nprobe_ = job[2];
                queries_.resize((size_t)batch_ * myDim_);
                MPI_Recv(queries_.data(), (int)queries_.size(), MPI_FLOAT,
                         MASTER_RANK, TAG_QUERY, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                probes_.resize((size_t)count * nprobe_);
                MPI_Recv(probes_.data(), (int)probes_.size(), MPI_INT,
                         MASTER_RANK, TAG_PROBES, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                thresholds_.assign(batch_, PRUNED);
                MPI_Recv(thresholds_.data(), count, MPI_FLOAT, MASTER_RANK,
                         TAG_THRESHOLD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                total_ = run.seconds();
                continue;
            }

            // ---- a block of queries: open it and post its upstream receive ----
            Pending p;
            p.firstQ = job[1];
            p.len = job[2];
            int item = job[3];
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


            // The layout: each query's run of running totals is its probe
            // list restricted to the clusters this row holds, in probe order.
            //
            // This and the buffer below are the setup_ bucket. Not free: the
            // loop is blockLen * nprobe lookups, and the buffer is one float
            // per (query, candidate) pair, freshly allocated and zeroed for
            // every block -- which at bDim = 1, where every block is a chain
            // head, is every block.
            phase.reset();
            p.qOff.resize(p.len);
            size_t total = 0;
            for (int j = 0; j < p.len; ++j) {
                p.qOff[j] = total;
                for (int i = 0; i < nprobe_; ++i) {
                    int bi = blockOf(probes_[(size_t)(p.firstQ + j) * nprobe_ + i]);
                    if (bi >= 0) {
                        total += blocks_[bi].ids.size();
                    }
                }
            }

            MPI_Request up = MPI_REQUEST_NULL;
            if (p.isFirst) {
                p.sums.assign(total, 0.0f);   // nothing upstream to wait for
            } else {
                p.sums.resize(total);
                // MPI: posted now, collected later -- the point of the whole
                // arrangement is that this wait overlaps other blocks.
                MPI_Irecv(p.sums.data(), (int)total, MPI_FLOAT, p.prevRank,
                          tagSums(p.slotTag), MPI_COMM_WORLD, &up);
            }
            setup_ = setup_ + phase.seconds();
            pend.push_back(std::move(p));
            pendReq.push_back(up);
        }

        if (done) {
            break;
        }
        if (pend.empty()) {
            continue;
        }

        // ---- 2. work on whichever open block can be worked on ----
        int pick = -1;
        for (size_t i = 0; i < pendReq.size(); ++i) {
            if (pendReq[i] == MPI_REQUEST_NULL) {
                pick = (int)i;   // a chain head, nothing to wait for
                break;
            }
        }
        if (pick < 0) {
            // MPI: has anyone's upstream landed while we were busy?
            phase.reset();
            int flag = 0;
            int idx = MPI_UNDEFINED;
            MPI_Testany((int)pendReq.size(), pendReq.data(), &idx, &flag,
                        MPI_STATUS_IGNORE);
            poll_ = poll_ + phase.seconds();
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
        size_t total = p.sums.size();

        phase.reset();
        aliveAtStage_[p.stage] = aliveAtStage_[p.stage]
                               + accumulate(p.firstQ, p.len, p.qOff, p.sums,
                                            p.stage == 0);
        compute_ = compute_ + phase.seconds();
        jobs_ = jobs_ + 1;

        // reclaim this slot before overwriting it. A large send_ is back
        // pressure: the next hop is not taking what this worker sends.
        phase.reset();
        MPI_Wait(&reqSums[slot], MPI_STATUS_IGNORE);
        send_ = send_ + phase.seconds();

        if (p.isLast) {
            // End of the chain. This is the only worker that ever sees these
            // candidates' full distances, so it can pick the k nearest itself
            // and send just those, instead of handing every running total back
            // for the master to sift (paper §4.3). k is around a hundred
            // against the thousands of candidates a query has here.
            Stopwatch pickWatch;
            outTop[slot].assign((size_t)p.len * k_, Candidate{-1, PRUNED});

            // OpenMP: one thread per query, each writing its own row
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < p.len; ++j) {
                TopKHeap heap(k_);
                size_t off = p.qOff[j];

                for (int i = 0; i < nprobe_; ++i) {
                    int bi = blockOf(probes_[(size_t)(p.firstQ + j) * nprobe_ + i]);
                    if (bi < 0) {
                        continue;
                    }
                    const ClusterBlock& cb = blocks_[bi];
                    int n = (int)cb.ids.size();
                    for (int v = 0; v < n; ++v) {
                        if (p.sums[off + v] < PRUNED) {
                            heap.push(cb.ids[v], p.sums[off + v]);
                        }
                    }
                    off += n;
                }

                std::vector<Candidate> best = heap.results();   // nearest first
                Candidate* out = &outTop[slot][(size_t)j * k_];
                for (int t = 0; t < (int)best.size(); ++t) {
                    out[t] = best[t];
                }
            }

            compute_ = compute_ + pickWatch.seconds();

            // MPI: as bytes, since a Candidate is an int beside a float and
            // every rank is the same build (see topk_heap.h).
            MPI_Isend(outTop[slot].data(),
                      (int)((size_t)p.len * k_ * sizeof(Candidate)), MPI_BYTE,
                      MASTER_RANK, tagTopk(p.slotTag), MPI_COMM_WORLD,
                      &reqSums[slot]);
        } else {
            outSums[slot] = std::move(p.sums);
            MPI_Isend(outSums[slot].data(), (int)total, MPI_FLOAT, p.nextRank,
                      tagSums(p.slotTag), MPI_COMM_WORLD, &reqSums[slot]);
        }

        // The ablation arm of Fig. 2(b): waiting here instead of letting the
        // send ride alongside the next block is what blocking communication
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
