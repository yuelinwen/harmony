#ifndef HARMONY_NODE_WORKER_NODE_H
#define HARMONY_NODE_WORKER_NODE_H

#include <algorithm>
#include <vector>

#include "node.h"
#include "../config.h"

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
        useMkl_ = false;
    }
    ~WorkerNode() override = default;

    // Receive loop: take setup from the master, then serve jobs until told to
    // stop.
    int run() override;

    // How many dimensions per vector the master will be sending.
    // Must be called before addCluster.
    void setDimCount(int myDim);

    // Takes one cluster: the global ids, and their vectors already cut down to
    // this worker's dimensions.
    void addCluster(int clusterId, const std::vector<int>& ids,
                    const std::vector<float>& data);

    // How many vectors this worker ended up with.
    long vectorCount() const;

    // Adds this worker's slice of the distance to what the previous worker
    // accumulated, and drops any candidate whose total has passed its
    // threshold -- partial sums only grow, so one that has passed can never
    // come back (paper Algorithm 1, lines 6-12).
    //
    // Works on m queries at once: `queries` is m slices of myDim floats,
    // `sums` is m rows of n running totals. `first` means the head of the
    // chain, where nothing is pruned yet and the gemm path applies.
    void accumulate(const float* queries, int m, int clusterId,
                    const float* thresholds, bool first, std::vector<float>& sums);

private:
    // Takes myDim, the cluster count, and then every cluster block.
    void receiveSetup();

    Config cfg_;
    int myDim_;
    int bDim_;      // how many workers share this row
    int myCol_;     // which of them this one is
    int rowBase_;   // rank of column 0 in this row
    int batch_;     // queries the master sends slices for
    bool useMkl_;   // gemm path enabled (and compiled in)

    // Survivors by position in the chain, not by worker: rotation makes a
    // worker the first stop for some clusters and the last for others.
    std::vector<long> aliveAtStage_;
    std::vector<ClusterBlock> blocks_;
};

}  // namespace harmony

#endif  // HARMONY_NODE_WORKER_NODE_H
