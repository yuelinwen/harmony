#ifndef HARMONY_NODE_MASTER_NODE_H
#define HARMONY_NODE_MASTER_NODE_H

#include <string>
#include <vector>

#include "node.h"
#include "../config.h"
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
    MasterNode(int numWorkers, const Config& cfg) : Node(0) {
        numWorkers_ = numWorkers;
        cfg_ = cfg;
        bVec_ = cfg.bVec;
        bDim_ = cfg.bDim;
        gtCount_ = 0;
        gtDim_ = 0;
        scanned_ = 0;
    }
    ~MasterNode() override = default;

    int run() override;

    // Reads the base and query vectors. Returns false if either file fails.
    bool loadData(const std::string& basePath, const std::string& queryPath);

    // Reads the true nearest neighbours: same layout as the vector files, but
    // int32 ids. Row q holds the real answer for query q, nearest first.
    bool loadGroundtruth(const std::string& path);

    // Share of the true top-k this result actually found (paper Section 6).
    double recallAt(int queryId, const std::vector<Candidate>& got, int k) const;

    // Clusters the base vectors into nlist groups. This runs once, over the
    // whole dataset, before anything is handed to the workers.
    void buildIndex(int nlist, int iterations);

    // Lays the workers out as a bVec x bDim grid (paper Fig. 4a). Row r holds
    // the clusters with c % bVec == r; within a row, each worker holds one
    // slice of the dimensions. bVec * bDim must equal the worker count.
    //   bVec=1          -> pure dimension partition (Harmony-dimension)
    //   bDim=1          -> pure vector partition    (Harmony-vector)
    //   both > 1        -> the hybrid the paper calls Harmony
    void splitGrid(int bVec, int bDim);

    // Cuts every cluster into per-worker slices and sends them out.
    void distributeData();

    // Tells the workers to stop, and collects their pruning counters.
    void shutdown();

    // Algorithm 1, lines 19-23. Picks the nprobe nearest clusters, runs each
    // one through the dimension pipeline, and merges what survives into one
    // top-K. The paper's version takes a whole query set; this one takes a
    // single query, since each query needs its own heap.
    std::vector<Candidate> queryPipeline(const float* query, int nprobe, int k);

    // Algorithm 1, lines 1-5. Seeds the heap with real distances so there is
    // a threshold to prune against from the very first candidate. Returns how
    // many it computed.
    int prewarmHeap(const float* query, int clusterId, int count, TopKHeap& heap);

    // Algorithm 1, lines 13-18. Runs the clusters of every vector partition
    // through the dimension pipeline and pushes the survivors into the heap.
    //
    // The pseudocode calls this once per partition (line 21-23), but Fig. 5a
    // has the partitions running at the same time -- "Stage B does not need to
    // follow Stage A" -- so all rows are driven together here instead: one
    // cluster is dispatched into every row before anything is collected.
    void vectorPipeline(const std::vector<std::vector<int>>& perRow,
                        int prewarmClusterId, int prewarmed, TopKHeap& heap);

    // Algorithm 1, lines 6-12, split in two so a row can be started without
    // waiting on it. Together they are the paper's "foreach d in DSet": the
    // job goes to every worker in the cluster's row, those workers pass the
    // running totals down the chain, and only the last one reports back.
    void dispatchBatch(int row, const std::vector<int>& batch,
                       int prewarmClusterId, int prewarmed, float threshold);

    // Which worker reports the result of a cluster that started at startCol.
    int lastRankOf(int row, int startCol) const;

    // ---- cost model, paper Section 4.2.1 ----

    // Runs queries through centroid assignment only, to learn which clusters
    // are hot before the layout is fixed (the paper's pre-query phase).
    void warmupPlan(int queries, int nprobe);

    // I(pi): the spread of computation across the vector partitions, worked
    // out from how often each cluster was probed and how big it is. Nothing
    // has to be executed -- pi only decides which row owns which cluster.
    double imbalanceOf(int bVec, int bDim) const;

    // Share of the distance work that still gets done when the dimensions are
    // cut into bDim slices. More slices mean more chances to stop early.
    //
    // The paper states that computation per machine barely moves between
    // layouts (Section 4.3) and its cost model has no term for this. Measured
    // on Sift1M that does not hold once pruning is on -- the work ranges from
    // 100% at one slice to about 50% at four -- and ignoring it makes the
    // model prefer exactly the layout that turns out to be slowest. These are
    // measured numbers, not a derivation: the paper gives no way to predict a
    // pruning ratio.
    double pruneFactor(int bDim) const;

    // C(pi,Q): computation, communication, and the imbalance penalty. Cast in
    // multiply-add equivalents, so commCost is the price of one transferred
    // byte relative to one multiply-add.
    double estimateCost(int bVec, int bDim) const;

    // Picks the grid with the lowest cost. Only a handful of factorisations
    // of the worker count are legal, so they are simply enumerated.
    void choosePlan();

private:
    Dataset base_;    // the vectors being searched
    Dataset query_;   // the vectors to search for
    IvfIndex index_;  // global clustering: centroids + inverted lists

    std::vector<int> gt_;   // gtCount_ rows of gtDim_ ids, row-major
    int gtCount_;
    int gtDim_;

    Config cfg_;
    int numWorkers_;
    int bVec_;                       // rows: vector partitions
    int bDim_;                       // columns: dimension slices per row
    SlicePlan plan_;                 // how a row cuts up the dimensions
    std::vector<int> clusterOwner_;  // cluster id -> vector partition (row)
    std::vector<long> clusterHits_;  // how often each cluster has been probed

    // pruning counters (paper Table 3). Counted by position in the chain
    // rather than by worker: rotation makes each worker the first stop for
    // some clusters and the last for others.
    long scanned_;                      // candidates offered in total
    std::vector<long> scannedRow_;      // per vector partition
    std::vector<long> aliveAfterStage_; // still alive after the s-th slice

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
