#ifndef HARMONY_NODE_WORKER_NODE_H
#define HARMONY_NODE_WORKER_NODE_H

#include <utility>
#include <vector>

#include "node.h"
#include "../config.h"
#include "../engine/search_order.h"
#include "../engine/stopwatch.h"
#include "../engine/topk_heap.h"

// WorkerNode (rank >= 1): where essentially all the arithmetic happens. It
// owns no plan and makes no decisions -- the master sends only the slice, so a
// worker never learns which dimensions it holds, just how many.
//
// Workers in the same row form a chain. A cluster enters at some column, goes
// round the row one worker at a time, and the last reports back to the master.
// The entry point rotates, so no worker is always the first stop.
//
// Paper Fig. 3 right side, Fig. 5b, Algorithm 1 lines 6-12.

namespace harmony {

// One cluster living on this worker.
struct ClusterBlock {
    int clusterId;            // id in the master's global clustering
    std::vector<int> ids;     // global vector ids
    std::vector<float> data;  // ids.size() * myDim, row-major
    std::vector<float> norm;  // ||v||^2 per vector, for the gemm path
};

class WorkerNode : public Node {
public:
    WorkerNode(int id, const Config& cfg) : Node(id) {
        cfg_ = cfg;
        myDim_ = 0;
        bDim_ = 1;
        myCol_ = 0;
        rowBase_ = id;
        batch_ = 1;
        sendSlots_ = 1;
        nprobe_ = 0;
        k_ = cfg.k;
        clearCounters();
    }
    ~WorkerNode() override = default;

    // Receive loop: take setup from the master, then serve jobs until told to
    // stop.
    int run() override;

    // Takes one cluster: the global ids, and their vectors already cut down to
    // this worker's dimensions.
    void addCluster(int clusterId, const std::vector<int>& ids,
                    const std::vector<float>& data);

    // How many vectors this worker ended up with.
    long vectorCount() const;

    // Bytes the cluster blocks occupy (paper Table 4). What this worker holds
    // for good, not the per-block scratch, which comes and goes with the
    // queries rather than with the index.
    long memoryBytes() const;

    // Adds this worker's slice of the distance to what the previous worker
    // accumulated, and drops any candidate whose total has passed its
    // threshold -- partial sums only grow, so one that has passed can never
    // come back (paper Algorithm 1, lines 6-12).
    //
    // Works on one block of queries: [firstQ, firstQ+len) of the batch, each
    // against every cluster of its probe list that this worker holds. qOff
    // says where each query's run of running totals starts in sums.
    //
    // Returns how many candidates came out of it still alive, which is what
    // the paper's Table 3 ratios are built from. Counted here rather than by a
    // pass over sums afterwards: this loop already visits every element, and
    // has threads, whereas that pass was serial over millions of floats and
    // measured 2.5% of the run at bDim = 1 and 14% at bDim = 4.
    long accumulate(int firstQ, int len,
                    const std::vector<size_t>& qOff, std::vector<float>& sums,
                    bool first);

private:
    // Takes myDim, the cluster count, and then every cluster block.
    void receiveSetup();

    // The head of a chain, where nothing has been pruned yet and every pair
    // has to be computed. Grouped by cluster instead of by query, that is a
    // dense matrix multiply, which is what MKL is for (paper §5).
    //
    // Always taken when MKL is compiled in; returns false when it is not, so
    // the caller falls back to the scalar loop. There is no switch because
    // measured on the cluster it is never worse: +49% QPS at myDim 128, +22%
    // at 64, and inside the noise at 32 and 16, the gain shrinking with the
    // slice width exactly as arithmetic intensity says it should. On true,
    // alive holds the survivor count, as accumulate() returns.
    bool accumulateGemm(int firstQ, int len,
                        const std::vector<size_t>& qOff,
                        std::vector<float>& sums, long* alive);

    // Scratch for the above: byCluster_[bi] lists the queries of the current
    // block that probe cluster bi, and where each one's run starts in sums.
    // Held across calls so the allocation happens once.
    std::vector<std::vector<std::pair<int, size_t>>> byCluster_;

    Config cfg_;
    int myDim_;
    int bDim_;      // how many workers share this row
    int myCol_;     // which of them this one is

    // This worker's rows of the chain table (engine/search_order.h), given by
    // the master at setup: for each item, who it takes the partial sums from,
    // who it passes them to (-1 for the ends of the chain), and how far along
    // the chain it sits.
    std::vector<int> nextOf_;
    std::vector<int> prevOf_;
    std::vector<int> stageOf_;

    // tau^2 per query of the current batch, kept between jobs and refreshed by
    // JOB_THRESH. Indexed by position in the batch, like queries_ and probes_.
    std::vector<float> thresholds_;

    // The batch's query slices and probe lists, sent once per batch. The
    // probe lists are what let this worker work out a block's buffer layout
    // for itself, the same way the rest of its row does.
    std::vector<float> queries_;
    std::vector<int> probes_;
    int nprobe_;

    // clusterId -> index into blocks_, or -1 for one this row does not hold.
    // It only reaches as far as the largest id this row was given, so an id
    // past its end is simply one of somebody else's -- which is why every
    // lookup goes through blockOf rather than indexing it directly.
    std::vector<int> where_;

    int blockOf(int c) const {
        if (c < 0 || c >= (int)where_.size()) {
            return -1;
        }
        return where_[c];
    }
    int rowBase_;   // rank of column 0 in this row
    int batch_;     // queries the master sends slices for
    int sendSlots_; // outgoing buffers to rotate through, from the master
    int k_;         // neighbours to keep when this worker ends a chain

    // Back to zero, for the constructor and for JOB_RESET, which starts a
    // counted stretch and has to forget whatever ran before it.
    void clearCounters() {
        total_ = 0.0;
        idle_ = 0.0;
        recv_ = 0.0;
        compute_ = 0.0;
        send_ = 0.0;
        setup_ = 0.0;
        poll_ = 0.0;
        admin_ = 0.0;
        jobs_ = 0;
    }

    // Where the run went, in seconds. Reported at shutdown and printed by the
    // master as one row per worker (paper Fig. 9).
    //
    // The seven below are meant to add up to total_ with only a little left
    // over. They did not at first: only compute/idle/recv/send existed and
    // between 15% and 45% of the run was unaccounted for, which would have
    // made Fig. 9 mostly one unlabelled block. setup_ and poll_ are that
    // remainder, and they are named rather than subtracted because the first
    // of them turned out to be avoidable work rather than measurement noise.
    //
    // The split is by waiting versus working, not by which message arrived:
    // every block on the master goes to idle_, whichever job it was waiting
    // for. Splitting it by message type instead just moved the same seconds
    // between idle_ and admin_ and made admin_ look like a new cost.
    double total_;     // first job to shutdown
    double idle_;      // blocked waiting for the master to hand over a job
    double recv_;      // blocked receiving partial sums from upstream
    double compute_;   // accumulate(), plus the top-k pick at a chain tail
    double send_;      // blocked reclaiming a send slot
    double setup_;     // opening a block: its buffer layout and its buffer
    double poll_;      // asking whether anything has arrived, and taking it
    double admin_;     // handling the master's stats / re-plan messages, not
                       // the wait for them
    long jobs_;        // clusters served

    // Survivors by position in the chain, not by worker: rotation makes a
    // worker the first stop for some clusters and the last for others.
    std::vector<long> aliveAtStage_;
    std::vector<ClusterBlock> blocks_;
};

}  // namespace harmony

#endif  // HARMONY_NODE_WORKER_NODE_H
