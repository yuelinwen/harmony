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
// this worker's own copy of those vectors.
struct ClusterBlock {
    int clusterId;            // id in the master's global clustering
    std::vector<int> ids;     // global vector ids
    std::vector<float> data;  // ids.size() * dim, row-major
};

class WorkerNode : public Node {
public:
    explicit WorkerNode(int id) : Node(id) {
        dim_ = 0;
    }
    ~WorkerNode() override = default;

    int run() override;

    // Copies one cluster's vectors out of base and keeps them here.
    // Under MPI this becomes "receive the block the master sent".
    void addCluster(int clusterId, const std::vector<int>& ids, const Dataset& base);

    // How many vectors this worker ended up with.
    long vectorCount() const;

    // Scans only the given clusters and returns the best k found here.
    // The master merges what every worker returns.
    std::vector<Candidate> search(const float* query,
                                  const std::vector<int>& clusterIds, int k);

private:
    int dim_;
    std::vector<ClusterBlock> blocks_;

    // ---- v2: dimension partition ----
    // TODO 3. hold only some dimensions of each vector, not the whole vector
    // TODO 4. return a partial distance instead of a full one

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
