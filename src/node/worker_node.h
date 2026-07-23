#ifndef HARMONY_NODE_WORKER_NODE_H
#define HARMONY_NODE_WORKER_NODE_H

#include "node.h"

// WorkerNode (id >= 1): a compute node.
// Holds one VxD data block, computes partial distances, and participates
// in pipeline pruning. (Paper Fig.3 right side, §4.3 Execution Engine.)

namespace harmony {

class WorkerNode : public Node {
public:
    explicit WorkerNode(int id) : Node(id) {}
    ~WorkerNode() override = default;

    int run() override;

private:
    // ---- added in later phases ----
    // blockStore_;      // phase 3/4: this node's VxD slice + inverted lists
    // partialDist();    // phase 4: compute partial distance for a query chunk
    // forwardOrPrune(); // phase 5: accumulate, prune if over threshold, else forward
    // pruneState_;      // phase 5: cached tau^2 from master broadcast
};

}  // namespace harmony

#endif  // HARMONY_NODE_WORKER_NODE_H
