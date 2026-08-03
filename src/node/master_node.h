#ifndef HARMONY_NODE_MASTER_NODE_H
#define HARMONY_NODE_MASTER_NODE_H

#include <string>
#include <vector>

#include "node.h"
#include "../comm/messages.h"
#include "../engine/slice_plan.h"
#include "../index/dataset.h"
#include "../index/ivf_index.h"

// MasterNode (id 0): the coordinator.
// Plans partitions, builds/distributes the index, routes queries, and
// merges the global Top-K. (Paper Fig.3 left side, §4.2 Query Planner.)

namespace harmony {

class MasterNode : public Node {
public:
    explicit MasterNode(int numWorkers) : Node(0) {
        numWorkers_ = numWorkers;
        scanned_ = 0;
    }
    ~MasterNode() override = default;

    int run() override;

    // Reads the base and query vectors. Returns false if either file fails.
    bool loadData(const std::string& basePath, const std::string& queryPath);

    // Clusters the base vectors into nlist groups. This runs once, over the
    // whole dataset, before anything is handed to the workers.
    void buildIndex(int nlist, int iterations);

    // Gives each worker an equal slice of the dimensions. Every worker holds
    // every cluster, so there is nothing to route: B_vec = 1, B_dim = numWorkers.
    void splitDimensions(int numWorkers);

    // Cuts every cluster into per-worker slices and sends them out.
    void distributeData();

    // Tells the workers to stop, and collects their pruning counters.
    void shutdown();

    // Picks the nprobe nearest clusters, asks the workers that own them,
    // and merges their answers into one top-K.
    std::vector<Candidate> search(const float* query, int nprobe, int k);

    // Seeds the heap with real distances so there is a threshold to prune
    // against from the very first candidate. Returns how many it computed.
    int prewarm(const float* query, int clusterId, int count, TopKHeap& heap);

private:
    Dataset base_;    // the vectors being searched
    Dataset query_;   // the vectors to search for
    IvfIndex index_;  // global clustering: centroids + inverted lists

    int numWorkers_;
    SlicePlan plan_;   // how the dimensions are cut up among the workers

    // pruning counters (paper Table 3)
    long scanned_;                  // candidates offered in total
    std::vector<long> aliveAfter_;  // still alive after worker w's slice

    // ---- v1: vector partition, one process, direct method calls ----
    //
    // TODO 5. search(query, k)  centroids -> nprobe clusters
    //                           -> ask the workers owning those clusters
    //                           -> merge their candidates into one top-K
    //
    // members: IvfIndex index_;  std::vector<WorkerNode> workers_;
    //          std::vector<int> clusterOwner_;   // cluster id -> worker id
    //
    // check: top-K here must equal IvfIndex::search() on one machine

    // ---- v2: dimension partition ----
    // TODO 6. each worker gets only some dimensions, not whole vectors
    // TODO 7. a query is split by dimension and sent to several workers

    // ---- v3: pruning ----
    // TODO 8.  prewarm()             fill the heap early to get a first tau^2
    // TODO 9.  broadcastThreshold()  send tau^2 to workers now and then
    // TODO 10. workers pass partial distances along a pipeline

    // ---- v4: MPI ----
    // TODO 11. replace direct calls with messages
};

}  // namespace harmony

#endif  // HARMONY_NODE_MASTER_NODE_H
