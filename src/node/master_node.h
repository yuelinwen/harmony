#ifndef HARMONY_NODE_MASTER_NODE_H
#define HARMONY_NODE_MASTER_NODE_H

#include "node.h"

// MasterNode (id 0): the coordinator.
// Plans partitions, builds/distributes the index, routes queries, and
// merges the global Top-K. (Paper Fig.3 left side, §4.2 Query Planner.)

namespace harmony {

class MasterNode : public Node {
public:
    MasterNode() : Node(0) {}
    ~MasterNode() override = default;

    int run() override;

private:
    // ---- added in later phases ----
    // buildAndDistributeIndex();  // phase 3: kmeans + IVF, send blocks to workers
    // dispatchBatch();            // phase 3: route queries, collect results
    // broadcastThreshold();       // phase 5: push tightened tau^2 to workers
    // planner_ / merger_ / loadMonitor_ / perfMeter_
};

}  // namespace harmony

#endif  // HARMONY_NODE_MASTER_NODE_H
