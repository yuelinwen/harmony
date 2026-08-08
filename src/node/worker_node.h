#ifndef HARMONY_NODE_WORKER_NODE_H
#define HARMONY_NODE_WORKER_NODE_H

#include <vector>

#include "node.h"
#include "../config.h"

// WorkerNode (rank >= 1): where essentially all the arithmetic happens.
//
// It owns no plan and makes no decisions. The master cuts the data and the
// query the same way and sends only the slice, so a worker never even learns
// which dimensions it is working on -- just how many. All it does is:
//
//   run()         wait for a job, do it, pass the result on, repeat
//   accumulate()  add my dimensions to the running total, drop what is now
//                 too far to reach the top-K
//
// Workers in the same row form a chain. A cluster enters at some column,
// goes round the row one worker at a time, and the last one reports back to
// the master. Which end of a chain this worker is depends on the job: the
// entry point rotates, so no worker is always the first stop.
//
// Paper Fig. 3 right side, Fig. 5b, Algorithm 1 lines 6-12.

namespace harmony {

// One cluster living on this worker: the global ids of its vectors, plus the
// slice of those vectors the master sent.
struct ClusterBlock {
    int clusterId;            // id in the master's global clustering
    std::vector<int> ids;     // global vector ids
    std::vector<float> data;  // ids.size() * myDim, row-major
};

class WorkerNode : public Node {
public:
    WorkerNode(int id, const Config& cfg) : Node(id) {
        cfg_ = cfg;
        myDim_ = 0;
        bDim_ = 1;
        myCol_ = 0;
        rowBase_ = id;
    }
    ~WorkerNode() override = default;

    // Receive loop: take setup from the master, then serve jobs until told
    // to stop.
    int run() override;

    // How many dimensions per vector the master will be sending.
    // Must be called before addCluster.
    void setDimCount(int myDim);

    // Takes one cluster: the global ids, and their vectors already cut down
    // to this worker's dimensions. Under MPI this is the buffer the master
    // sent, received as-is.
    void addCluster(int clusterId, const std::vector<int>& ids,
                    const std::vector<float>& data);

    // How many vectors this worker ended up with.
    long vectorCount() const;

    // Adds this worker's slice of the distance on top of what the previous
    // worker in the pipeline already accumulated, and drops any candidate
    // whose running total has passed the threshold: partial sums only grow,
    // so it can never come back and reach the top-K.
    //
    // querySlice is the master's cut of the query, myDim floats long. sums
    // holds one running total per vector in the cluster and is carried from
    // worker to worker; a pruned entry is marked with PRUNED
    // (paper Algorithm 1, lines 6-12).
    void accumulate(const float* querySlice, int clusterId, float threshold,
                    std::vector<float>& sums);

private:
    // Takes myDim, the cluster count, and then every cluster block.
    void receiveSetup();

    Config cfg_;
    int myDim_;
    int bDim_;      // how many workers share this row
    int myCol_;     // which of them this one is
    int rowBase_;   // rank of column 0 in this row

    // Survivors counted by position in the chain, not by worker: with
    // rotation a worker is the first stop for some clusters and the last for
    // others, so a single per-worker total would mix the two.
    std::vector<long> aliveAtStage_;
    std::vector<ClusterBlock> blocks_;
};

}  // namespace harmony

#endif  // HARMONY_NODE_WORKER_NODE_H
