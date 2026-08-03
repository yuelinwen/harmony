#ifndef HARMONY_NODE_WORKER_NODE_H
#define HARMONY_NODE_WORKER_NODE_H

#include <vector>

#include "node.h"

// WorkerNode (id >= 1): a compute node.
// Holds one VxD data block, computes partial distances, and participates
// in pipeline pruning. (Paper Fig.3 right side, §4.3 Execution Engine.)
//
// The worker owns no plan and makes no decisions. The master cuts the data
// and the query the same way and sends only the slice, so a worker never even
// learns which dimensions it is working on -- just how many.

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
    explicit WorkerNode(int id) : Node(id) {
        myDim_ = 0;
        nextRank_ = 0;
        aliveCount_ = 0;
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
    // querySlice is the master's cut of the query, myDim floats long.
    // sums and alive hold one entry per vector in the cluster and are carried
    // from worker to worker (paper Algorithm 1, lines 6-12).
    void accumulate(const float* querySlice, int clusterId, float threshold,
                    std::vector<float>& sums, std::vector<char>& alive);

private:
    // Takes myDim, the cluster count, and then every cluster block.
    void receiveSetup();

    int myDim_;
    int nextRank_;    // where partial results go: the next worker, or the master
    long aliveCount_; // survivors after this worker, summed over every job
    std::vector<ClusterBlock> blocks_;
};

}  // namespace harmony

#endif  // HARMONY_NODE_WORKER_NODE_H
