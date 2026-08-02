#ifndef HARMONY_NODE_WORKER_NODE_H
#define HARMONY_NODE_WORKER_NODE_H

#include <vector>

#include "node.h"
#include "../index/dataset.h"
#include "../engine/topk_heap.h"

// WorkerNode (id >= 1): a compute node.
// Holds one VxD data block, computes partial distances, and participates
// in pipeline pruning. (Paper Fig.3 right side, §4.3 Execution Engine.)

namespace harmony {

// One cluster living on this worker: the global ids of its vectors, plus
// this worker's copy of them, cut down to the dimensions it owns.
struct ClusterBlock {
    int clusterId;            // id in the master's global clustering
    std::vector<int> ids;     // global vector ids
    std::vector<float> data;  // ids.size() * (dimEnd - dimBegin), row-major
};

class WorkerNode : public Node {
public:
    explicit WorkerNode(int id) : Node(id) {
        dimBegin_ = 0;
        dimEnd_ = 0;
    }
    ~WorkerNode() override = default;

    int run() override;

    // Which dimensions of every vector this worker is responsible for.
    // Must be called before addCluster.
    void setDimensions(int dimBegin, int dimEnd);

    // Copies one cluster out of base, keeping only this worker's dimensions.
    // Under MPI this becomes "receive the block the master sent".
    void addCluster(int clusterId, const std::vector<int>& ids, const Dataset& base);

    // How many vectors this worker ended up with.
    long vectorCount() const;

    // Distance over this worker's dimensions only: one value per vector in
    // the given clusters, in cluster order. A partial distance is not enough
    // to rank anything, so unlike v1 the worker cannot return a top-k here --
    // the master adds up what every worker sends before it can sort.
    std::vector<float> partialDistances(const float* query,
                                        const std::vector<int>& clusterIds);

private:
    int dimBegin_;
    int dimEnd_;
    std::vector<ClusterBlock> blocks_;

    // ---- v3: pruning ----
    // TODO 5. keep the tau^2 last sent by the master
    // TODO 6. add up slices, stop early when the sum passes tau^2
    //         (scanAll in engine/pruning_scanner.h is this loop)
    // TODO 7. if not pruned, pass the partial sum to the next worker

    // ---- v4: MPI ----
    // TODO 8. receive queries and send results as messages
};

}  // namespace harmony

#endif  // HARMONY_NODE_WORKER_NODE_H
