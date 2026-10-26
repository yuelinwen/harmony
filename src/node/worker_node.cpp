#include "worker_node.h"

#include <iostream>

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <mkl.h>

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

// ---- compaction ------------------------------------------------------------
//
// A hop forwards one float per (query, candidate) pair whether or not the
// candidate is still alive, because pruning marks it PRUNED in place and
// leaves the array the same length. That makes pruning save arithmetic and
// nothing else: measured on Sift1M at 1x4, the bytes on the wire are
// identical under every backend, including the one that drops three times
// as many candidates.
//
// Compacting sends a bitmap of who survived followed by only their values.
// At survival rate s the cost is total/8 + 4*s*total against 4*total, so it
// pays whenever s < 0.97 -- which is every hop after the first.
//
// The receiver does not need to be told the length. The bitmap is a fixed
// size for a known candidate count, and its population count says how many
// floats follow, so one Irecv into a worst-case buffer is enough.

size_t bitmapBytes(size_t n) {
    // Rounded up to four so the values that follow start aligned.
    return (((n + 7) / 8) + 3) & ~(size_t)3;
}

// Both halves run over the whole candidate array, which is the same array
// accumulate() just walked -- and accumulate walks it on eight threads. Left
// serial, this scan alone cost a third of the run (measured: "other" went
// from 9% to 33%), so it is split the same way.
//
// A chunk is a whole number of bitmap bytes, so no two threads ever write the
// same byte. The first pass counts survivors per chunk, the prefix sum says
// where each chunk's values go, and the second pass writes them with no
// coordination.
const size_t kPackChunk = 4096;

// [bitmap][surviving values], returning the length actually used.
//
// The store is guarded rather than branchless. Writing every value and
// advancing by 0 or 1 is the faster shape and is safe in one sequential
// pass -- the stray write is overwritten by the next live value -- but not
// once the passes are chunked: a chunk whose last entries are all dead
// writes past its own region and into the first slot of the next chunk,
// which another thread has already filled. That cost two wrong answers in
// 1024 and 12,832 missing dim_ops before it was caught.
size_t packSurvivors(const std::vector<float>& sums, unsigned char* out,
                     std::vector<size_t>& starts) {
    size_t n = sums.size();
    size_t bits = bitmapBytes(n);
    size_t chunks = (n + kPackChunk - 1) / kPackChunk;
    float* values = reinterpret_cast<float*>(out + bits);

    // bitmapBytes rounds up to four; zero the bytes no chunk will write, so
    // what goes on the wire does not depend on what was in the buffer last.
    for (size_t b = (n + 7) / 8; b < bits; ++b) {
        out[b] = 0;
    }

    starts.assign(chunks + 1, 0);
    #pragma omp parallel for schedule(static)
    for (long c = 0; c < (long)chunks; ++c) {
        size_t lo = (size_t)c * kPackChunk;
        size_t hi = (lo + kPackChunk < n) ? (lo + kPackChunk) : n;
        size_t alive = 0;
        for (size_t i = lo; i < hi; ++i) {
            alive = alive + ((sums[i] < PRUNED) ? 1 : 0);
        }
        starts[c + 1] = alive;
    }
    for (size_t c = 0; c < chunks; ++c) {
        starts[c + 1] = starts[c + 1] + starts[c];
    }

    #pragma omp parallel for schedule(static)
    for (long c = 0; c < (long)chunks; ++c) {
        size_t lo = (size_t)c * kPackChunk;
        size_t hi = (lo + kPackChunk < n) ? (lo + kPackChunk) : n;
        size_t kept = starts[c];
        for (size_t i = lo; i < hi; i += 8) {
            size_t end = (i + 8 < hi) ? (i + 8) : hi;
            unsigned mask = 0;
            for (size_t j = i; j < end; ++j) {
                if (sums[j] < PRUNED) {
                    mask = mask | (1u << (j - i));
                    values[kept] = sums[j];
                    kept = kept + 1;
                }
            }
            out[i >> 3] = (unsigned char)mask;
        }
    }
    return bits + starts[chunks] * sizeof(float);
}

// The inverse. Everything the bitmap does not claim is PRUNED, which is what
// the uncompacted array would have carried. Here the per-chunk count is a
// popcount of the bitmap, so the first pass touches bytes rather than floats.
//
// This half can stay branchless: the overshooting access is a read, and a
// read of the next chunk's first value races with nothing.
void unpackSurvivors(const unsigned char* in, std::vector<float>& sums,
                     std::vector<size_t>& starts) {
    size_t n = sums.size();
    size_t bits = bitmapBytes(n);
    size_t chunks = (n + kPackChunk - 1) / kPackChunk;
    const float* values = reinterpret_cast<const float*>(in + bits);

    starts.assign(chunks + 1, 0);
    #pragma omp parallel for schedule(static)
    for (long c = 0; c < (long)chunks; ++c) {
        size_t lo = (size_t)c * kPackChunk;
        size_t hi = (lo + kPackChunk < n) ? (lo + kPackChunk) : n;
        size_t alive = 0;
        for (size_t b = lo >> 3; b < (hi + 7) >> 3; ++b) {
            alive = alive + (size_t)__builtin_popcount(in[b]);
        }
        starts[c + 1] = alive;
    }
    for (size_t c = 0; c < chunks; ++c) {
        starts[c + 1] = starts[c + 1] + starts[c];
    }

    #pragma omp parallel for schedule(static)
    for (long c = 0; c < (long)chunks; ++c) {
        size_t lo = (size_t)c * kPackChunk;
        size_t hi = (lo + kPackChunk < n) ? (lo + kPackChunk) : n;
        size_t kept = starts[c];
        for (size_t i = lo; i < hi; ++i) {
            unsigned alive = (in[i >> 3] >> (i & 7)) & 1u;
            // The unconditional read is in range: kept never passes n - 1.
            sums[i] = alive ? values[kept] : PRUNED;
            kept = kept + alive;
        }
    }
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
// Later slices are a different case: a dense multiply cannot skip, so it
// recomputes what pruning has already dropped. That is a race between the
// multiply's speed and the prune rate -- 3.1% at the second slice and 71.2%
// at the fourth, measured -- so --gemmall decides it rather than this
// comment. first says which semantics to write back with: the head assigns
// into a zero buffer, a later slice accumulates into what it received and
// has to leave already-pruned entries alone.
HopCount WorkerNode::accumulateGemm(int firstQ, int len, bool first,
                                    const std::vector<size_t>& qOff,
                                    std::vector<float>& sums) {
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
    long evaluated = 0;
    #pragma omp parallel for schedule(dynamic) \
        reduction(+ : survivors) reduction(+ : evaluated)
    for (int bi = 0; bi < (int)byCluster_.size(); ++bi) {
        const std::vector<std::pair<int, size_t> >& mem = byCluster_[bi];
        int m = (int)mem.size();
        if (m == 0) {
            continue;
        }

        const ClusterBlock& cb = blocks_[bi];
        int n = (int)cb.ids.size();

        // A dense multiply cannot skip, so every pair here is evaluated --
        // which is the whole reason this path is only ever a chain head.
        evaluated = evaluated + (long)m * n;

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
            float t = thresholds_[firstQ + mem[a].first] * curScale_;
            float* row = &sums[mem[a].second];
            const float* d = &dot[(size_t)a * n];
            for (int v = 0; v < n; ++v) {
                float part = qn[a] + cb.norm[v] - 2.0f * d[v];
                if (first) {
                    row[v] = part;
                } else if (row[v] >= PRUNED) {
                    continue;      // dropped upstream; the multiply computed
                                   // it anyway, which is what this costs
                } else {
                    row[v] = row[v] + part;
                }
                if (row[v] > t) {
                    row[v] = PRUNED;
                } else {
                    survivors = survivors + 1;
                }
            }
        }
    }

    HopCount out;
    out.evaluated = evaluated;
    out.survivors = survivors;
    return out;
}

HopCount WorkerNode::accumulate(int firstQ, int len,
                                const std::vector<size_t>& qOff,
                                std::vector<float>& sums, bool first) {
    if (first || cfg_.gemmAll) {
        return accumulateGemm(firstQ, len, first, qOff, sums);
    }

    long survivors = 0;
    long evaluated = 0;
    // One query at a time, walking its probe list and stopping at the
    // clusters this row holds. That walk is the buffer layout: every worker
    // in the row derives the same one, so nobody has to send an index.
    //
    // OpenMP: a query owns its own run of sums, so the threads never write
    // the same memory and no locking is needed (paper §5, node-level
    // parallelism; across nodes the work is already split by MPI).
    #pragma omp parallel for schedule(static) \
        reduction(+ : survivors) reduction(+ : evaluated)
    for (int j = 0; j < len; ++j) {
        const float* qv = &queries_[(size_t)(firstQ + j) * myDim_];
        float t = thresholds_[firstQ + j] * curScale_;
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
                evaluated = evaluated + 1;
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

    HopCount out;
    out.evaluated = evaluated;
    out.survivors = survivors;
    return out;
}

void WorkerNode::receiveSetup() {
    int setup[9];
    MPI_Recv(setup, 9, MPI_INT, MASTER_RANK, TAG_SETUP,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    myDim_ = setup[0];
    int nClusters = setup[1];   // only the clusters of this worker's row
    bDim_ = setup[2];
    batch_ = setup[3];
    sendSlots_ = setup[4];
    bVec_ = setup[5];
    blockCount_ = setup[6];
    pipeline_ = (setup[7] != 0);
    pruneVector_ = (setup[8] != 0);
    if (sendSlots_ < 1) {
        sendSlots_ = 1;
    }

    // Where this worker sits in its row. Which end of a chain it is depends
    // on the job, since clusters start at different columns.
    myCol_ = (id_ - 1) % bDim_;
    rowBase_ = id_ - myCol_;
    myRow_ = (id_ - 1) / bDim_;
    aliveAtStage_.assign(bDim_, 0);
    dimsAtStage_.assign(bDim_, 0);
    bytesAtStage_.assign(bDim_, 0);

    // Both orders are built here rather than sent, from the one number each
    // needs. The master builds the same two from the same numbers, so the two
    // sides always agree without a message: at stage s the group on this row
    // is stepsOf(myRow_)[s], and item i enters the row at chain(i)[0].
    //
    // This works because neither order changes once the grid is fixed. It did
    // not always: §4.3's load-aware reordering rewrote the chain order from
    // measured load, which only the master can see, so the table had to be
    // shipped. That was measured and removed (engine/search_order.h).
    groupOrder_ = SearchOrder(bVec_, bVec_, true);
    SearchOrder chainOrder(bDim_, bDim_, true);

    nextOf_ = chainOrder.nextRow(myCol_);
    prevOf_ = chainOrder.prevRow(myCol_);
    stageOf_.assign(bDim_, 0);
    for (int item = 0; item < bDim_; ++item) {
        const std::vector<int>& chain = chainOrder.chain(item);
        for (int p = 0; p < (int)chain.size(); ++p) {
            if (chain[p] == myCol_) {
                stageOf_[item] = p;
                break;
            }
        }
    }

    // What to multiply the pruning threshold by on this worker's hop for
    // each item, straight after the setup message. One float per item and
    // no explanation: the master works them out, because what sets them can
    // depend on the data, which only the master has.
    scaleOf_.assign(bDim_, 1.0f);
    MPI_Recv(scaleOf_.data(), bDim_, MPI_FLOAT, MASTER_RANK, TAG_SETUP,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // --netprobe: the master times a few round trips against rank 1 before
    // any data moves. The counts come from comm/messages.h so both sides
    // agree; getting them wrong would hang here rather than fail.
    if (cfg_.netProbe && id_ == 1) {
        std::vector<char> buf(PROBE_BYTES);
        char ping[8] = {0};
        for (int i = 0; i < PROBE_ROUNDS + 1; ++i) {
            MPI_Recv(buf.data(), PROBE_BYTES, MPI_BYTE, MASTER_RANK, TAG_PROBE,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buf.data(), PROBE_BYTES, MPI_BYTE, MASTER_RANK, TAG_PROBE,
                     MPI_COMM_WORLD);
        }
        for (int i = 0; i < PROBE_PINGS; ++i) {
            MPI_Recv(ping, 8, MPI_BYTE, MASTER_RANK, TAG_PROBE, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Send(ping, 8, MPI_BYTE, MASTER_RANK, TAG_PROBE, MPI_COMM_WORLD);
        }
    }

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
    std::cout << "worker " << id_ << " ready: " << vectorCount()
              << " vectors x " << myDim_ << " dims, "
              << threads << " thread(s), loop+mkl" << std::endl;
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
    // Under --compact the totals are packed into these instead. Grown to the
    // largest block seen and then left alone: a fresh allocation per block
    // costs a malloc and a zero-fill of an array this size, which is the
    // same order as the bytes compacting saves.
    std::vector<std::vector<unsigned char>> outWire(slots);
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
        int item;     // which of the row's items, so the hop knows its scale
        int slotTag;
        bool isFirst;
        bool isLast;
        int prevRank;
        int nextRank;
        std::vector<size_t> qOff;   // where each query's totals start
        std::vector<float> sums;
        // Under --compact the wire form arrives here and is expanded into
        // sums once the receive completes. Sized for the worst case, which
        // is everything surviving, so one receive covers any survival rate.
        bool packed = false;
        std::vector<unsigned char> wire;
    };
    std::vector<Pending> pend;
    // Recycled across blocks, for the reason outWire is.
    std::vector<std::vector<unsigned char>> wireFree;
    std::vector<size_t> wireChunks;   // scratch for pack/unpack
    std::vector<MPI_Request> pendReq;   // upstream receive; NULL for a head

    // A block named but not yet opened. A group arrives as one message and
    // expands into blockCount_ of these; they turn into Pendings as there is
    // room, which is the only place --disablepipeline bites.
    struct Todo {
        int firstQ;
        int len;
        int item;
        int tag;
    };
    std::vector<Todo> todo;

    // Stages of the current batch still to be taken. A batch is bVec of
    // them, each bringing one query group to this row.
    int stagesLeft = 0;

    while (!done) {
        // ---- 1. take the next stage of the batch, if one is due ----
        //
        // Nothing announces it. After JOB_QUERY there are exactly bVec
        // stages, and the group order says which query group each of them
        // brings to this row; the group's queries, its blocks, and their tags
        // follow from that (comm/messages.h). All that arrives is the
        // thresholds.
        //
        // Only once the previous stage is finished, which is the vector-level
        // barrier (Fig. 5a) seen from this side: the next partition starts
        // from a threshold this one tightened. With --disablevectorpruning
        // every stage is taken as soon as it arrives instead, which is what
        // the master does on its side too.
        bool stageDue = false;
        if (stagesLeft > 0) {
            if (pend.empty() && todo.empty()) {
                stageDue = true;
            } else if (!pruneVector_) {
                // Work in hand, so take the stage only if it has already
                // arrived. Under --disablepipeline the master does not send
                // the next one until the blocks still sitting in todo come
                // back, and blocking here waits for a message it is waiting
                // on this worker to earn.
                int ready = 0;
                MPI_Iprobe(MASTER_RANK, TAG_THRESHOLD, MPI_COMM_WORLD, &ready,
                           MPI_STATUS_IGNORE);
                stageDue = (ready != 0);
            }
        }
        if (stageDue) {
            int s = bVec_ - stagesLeft;
            stagesLeft = stagesLeft - 1;

            int g = groupOrder_.stepsOf(myRow_)[s];
            int gStart = (int)(((long)g * count_ + bVec_ - 1) / bVec_);
            int gEnd = (int)(((long)(g + 1) * count_ + bVec_ - 1) / bVec_);
            int gLen = gEnd - gStart;

            if (gLen > 0) {
                // Blocked on the master, which sends this once the row before
                // has finished with the group -- so it is waiting, not work.
                phase.reset();
                MPI_Recv(&thresholds_[gStart], gLen, MPI_FLOAT, MASTER_RANK,
                         TAG_THRESHOLD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                idle_ = idle_ + phase.seconds();

                for (int b = 0; b < blockCount_; ++b) {
                    Todo t;
                    t.firstQ = gStart + (int)((long)b * gLen / blockCount_);
                    int endQ = gStart + (int)((long)(b + 1) * gLen / blockCount_);
                    t.len = endQ - t.firstQ;
                    if (t.len <= 0) {
                        continue;
                    }
                    t.item = b % bDim_;
                    t.tag = blockTag(g, s, b, bVec_, blockCount_);
                    todo.push_back(t);
                }
            }
        }

        // ---- 2. take every job already waiting, and block only when there
        //         is nothing open to work on ----
        while (true) {
            int waiting = 1;
            if (!pend.empty() || !todo.empty() || stagesLeft > 0) {
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
            bool holding = !pend.empty() || !todo.empty() || stagesLeft > 0;
            double waited = holding ? 0.0 : took;
            if (holding) {
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
                std::vector<long> stats;
                stats.reserve((size_t)STAT_ROWS * bDim_);
                stats.insert(stats.end(), aliveAtStage_.begin(),
                             aliveAtStage_.end());
                stats.insert(stats.end(), dimsAtStage_.begin(),
                             dimsAtStage_.end());
                stats.insert(stats.end(), bytesAtStage_.begin(),
                             bytesAtStage_.end());
                MPI_Send(stats.data(), STAT_ROWS * bDim_, MPI_LONG,
                         MASTER_RANK, TAG_STATS, MPI_COMM_WORLD);
                double times[WORKER_TIMES] = {total_, idle_, recv_, compute_,
                                              send_, (double)jobs_,
                                              setup_, poll_, admin_,
                                              (double)memoryBytes()};
                MPI_Send(times, WORKER_TIMES, MPI_DOUBLE, MASTER_RANK,
                         TAG_TIMES, MPI_COMM_WORLD);
                admin_ = admin_ + adminWatch.seconds();
                continue;
            }

            // Start of a counted stretch: forget everything before it. Sent
            // before the pass that gets reported, so an earlier --loop pass or
            // an earlier --nprobes value does not leak into these numbers.
            if (job[0] == JOB_RESET) {
                aliveAtStage_.assign(bDim_, 0);
                dimsAtStage_.assign(bDim_, 0);
                bytesAtStage_.assign(bDim_, 0);
                clearCounters();
                run.reset();
                continue;
            }

            if (job[0] == JOB_QUERY) {
                count_ = job[1];
                int count = count_;
                nprobe_ = job[2];
                stagesLeft = bVec_;   // a batch is bVec stages, and no job
                                      // will say so again
                queries_.resize((size_t)batch_ * myDim_);
                MPI_Recv(queries_.data(), (int)queries_.size(), MPI_FLOAT,
                         MASTER_RANK, TAG_QUERY, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                probes_.resize((size_t)count * nprobe_);
                MPI_Recv(probes_.data(), (int)probes_.size(), MPI_INT,
                         MASTER_RANK, TAG_PROBES, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                // Sized, not filled: each stage receives the thresholds of
                // the group it brings, and nothing is computed before a
                // stage has opened.
                thresholds_.assign(batch_, PRUNED);
                total_ = run.seconds();
                continue;
            }

            // Nothing else reaches a worker during a batch.
        }

        if (done) {
            break;
        }

        // ---- 3. open what there is room for ----
        //
        // Normally every named block at once, so the wait for one block's
        // upstream overlaps another block's arithmetic. One at a time under
        // --disablepipeline, which is exactly the overlap that arm removes.
        while (!todo.empty() && (pipeline_ || pend.empty())) {
            Pending p;
            p.firstQ = todo.front().firstQ;
            p.len = todo.front().len;
            int item = todo.front().item;
            p.slotTag = todo.front().tag;
            todo.erase(todo.begin());

            // Everything about this worker's part in the chain comes out of
            // the table: which item it is decides where the chain starts, and
            // the table says who is on either side.
            int prevCol = prevOf_[item];
            int nextCol = nextOf_[item];
            p.stage = stageOf_[item];        // 0 = first stop
            p.item = item;
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

            if (total == 0) {
                // This partition holds nothing for these queries. The master
                // adds up the same probe lists and posts no receive for it, so
                // the block exists on neither side. This is the one place the
                // two sides have to agree without being told, and they agree
                // because they are adding up the same numbers.
                setup_ = setup_ + phase.seconds();
                continue;
            }

            MPI_Request up = MPI_REQUEST_NULL;
            if (p.isFirst) {
                p.sums.assign(total, 0.0f);   // nothing upstream to wait for
            } else {
                p.sums.resize(total);
                // MPI: posted now, collected later -- the point of the whole
                // arrangement is that this wait overlaps other blocks.
                if (cfg_.compact) {
                    // Worst case, so one receive is enough however many
                    // survive. MPI delivers fewer bytes without complaint.
                    size_t need = bitmapBytes(total) + total * sizeof(float);
                    if (!wireFree.empty()) {
                        p.wire = std::move(wireFree.back());
                        wireFree.pop_back();
                    }
                    if (p.wire.size() < need) {
                        p.wire.resize(need);
                    }
                    p.packed = true;
                    MPI_Irecv(p.wire.data(), (int)need, MPI_BYTE,
                              p.prevRank, tagSums(p.slotTag), MPI_COMM_WORLD,
                              &up);
                } else {
                    MPI_Irecv(p.sums.data(), (int)total, MPI_FLOAT, p.prevRank,
                              tagSums(p.slotTag), MPI_COMM_WORLD, &up);
                }
            }
            setup_ = setup_ + phase.seconds();
            pend.push_back(std::move(p));
            pendReq.push_back(up);
        }

        if (pend.empty()) {
            continue;
        }

        // ---- 4. work on whichever open block can be worked on ----
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

        if (p.packed) {
            // The bitmap puts the survivors back where they came from and
            // marks everything else PRUNED, which is what the upstream hop
            // decided. Charged to recv: taking a block off the wire is not
            // finished until it has been expanded.
            phase.reset();
            unpackSurvivors(p.wire.data(), p.sums, wireChunks);
            recv_ = recv_ + phase.seconds();
        }

        phase.reset();
        curScale_ = scaleOf_[p.item];
        HopCount hop = accumulate(p.firstQ, p.len, p.qOff, p.sums,
                                  p.stage == 0);
        aliveAtStage_[p.stage] = aliveAtStage_[p.stage] + hop.survivors;
        // One "dimension evaluated" is one (q[i] - v[i])^2, so a candidate
        // this hop computed costs the width of this worker's slice. Counting
        // it here rather than as candidates x width on the master is what
        // makes an uneven split (128 over 3) add up correctly.
        dimsAtStage_[p.stage] = dimsAtStage_[p.stage]
                              + hop.evaluated * (long)myDim_;
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
            size_t bytes = (size_t)p.len * k_ * sizeof(Candidate);
            bytesAtStage_[p.stage] = bytesAtStage_[p.stage] + (long)bytes;
            MPI_Isend(outTop[slot].data(), (int)bytes, MPI_BYTE,
                      MASTER_RANK, tagTopk(p.slotTag), MPI_COMM_WORLD,
                      &reqSums[slot]);
        } else if (cfg_.compact) {
            size_t need = bitmapBytes(total) + total * sizeof(float);
            if (outWire[slot].size() < need) {
                outWire[slot].resize(need);
            }
            phase.reset();
            size_t wireBytes = packSurvivors(p.sums, outWire[slot].data(),
                                             wireChunks);
            send_ = send_ + phase.seconds();   // the other half of the same cost
            // What actually went out, not what the array could have held --
            // the whole point of compacting is that these differ.
            bytesAtStage_[p.stage] = bytesAtStage_[p.stage] + (long)wireBytes;
            MPI_Isend(outWire[slot].data(), (int)wireBytes, MPI_BYTE,
                      p.nextRank, tagSums(p.slotTag), MPI_COMM_WORLD,
                      &reqSums[slot]);
        } else {
            bytesAtStage_[p.stage] = bytesAtStage_[p.stage]
                                   + (long)(total * sizeof(float));
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

        if (!p.wire.empty()) {
            wireFree.push_back(std::move(p.wire));
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
