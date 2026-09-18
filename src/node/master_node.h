#ifndef HARMONY_NODE_MASTER_NODE_H
#define HARMONY_NODE_MASTER_NODE_H

#include <string>
#include <vector>

#include "node.h"
#include "../config.h"
#include "../comm/messages.h"
#include "../engine/search_order.h"
#include "../engine/slice_plan.h"
#include "../index/dataset.h"
#include "../index/ivf_index.h"

// MasterNode (rank 0): decides and coordinates, but does almost no distance
// work. Paper Fig. 3 left side; Algorithm 1 is spread across the *Pipeline
// methods.

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

    // Load, build, distribute, then search -- and check the answer against a
    // single machine and report recall, QPS and the pruning ratios.
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

    // Running totals of cluster sizes, once the index exists.
    void buildSizePrefix();

    // Where a cached index with these settings lives.
    std::string indexPath(int nlist, int iterations) const;

    // Lays the workers out as a bVec x bDim grid (paper Fig. 4a). Clusters go
    // to rows by weight, heaviest first to the lightest row; within a row,
    // each worker holds one slice of the dimensions.
    void splitGrid(int bVec, int bDim);

    // Cuts every cluster into per-worker slices and sends them out.
    void distributeData();

    // One column's rows of the chain table, as the worker expects them.
    std::vector<int> chainTableFor(int col) const;

    // Paper §4.3: moves the column that has been doing the most arithmetic to
    // the end of every chain, where most candidates have already been pruned.
    // Call between batches only -- see the comment on the definition.
    void reorderChains();

    // Zeroes the counters, here and on every worker.
    void resetCounters();

    // Collects the pruning counters and timings without stopping anybody.
    void collectStats();

    // Tells the workers to stop.
    void shutdown();

    // Appends one row describing this run to cfg_.csv, writing the header
    // first if the file is new. Nothing happens when --csv was not given.
    void writeCsv(int nprobe, int nq, double recall, double seconds,
                  int differing, int ties) const;

    // Prints the per-worker time breakdown gathered by shutdown().
    void printWorkerTimes() const;

    // What one query of the current batch needs.
    struct QueryState {
        int id;                       // row in query_
        std::vector<int> clusters;    // its nprobe nearest
    };

    // Algorithm 1, lines 19-23. Runs a batch of queries, the paper's
    // QueryBatch: those probing the same cluster share one visit to it. One
    // heap per query, unlike line 20 -- see CLAUDE.md.
    std::vector<std::vector<Candidate>> queryPipeline(int firstQuery, int count,
                                                      int nprobe, int k);

    // Algorithm 1, lines 1-5. Seeds the heap with real distances so there is
    // a threshold to prune against from the very first candidate. Returns how
    // many it computed.
    void prewarmHeap(const float* query, QueryState& state, TopKHeap& heap);

    // Algorithm 1, lines 13-18. Runs the clusters of every vector partition
    // through the dimension pipeline and pushes the survivors into the heaps.
    //
    // work[g][r] holds the clusters of vector partition r that query group g
    // probes. Group g visits the partitions in the order r = (g + stage) %
    // bVec, one stage at a time, so by the time it reaches its second
    // partition its heaps already carry the first one's distances (Fig. 5a).
    // Groups advance independently -- at any stage the mapping group -> row is
    // a permutation, so every row stays busy.
    void vectorPipeline(const std::vector<std::vector<int>>& groupMembers,
                        const std::vector<QueryState>& batch,
                        std::vector<TopKHeap>& heaps);

    // Algorithm 1, lines 6-12. Sends one cluster to every worker in its row
    // and returns; the workers pass the running totals down the chain and only
    // the last reports back. `members` are the batch positions that probed it.
    void dispatchBlock(int row, int firstQ, int len, int item, int slot,
                       const std::vector<float>& thresholds);

    // Candidates a block of queries contributes in one vector partition, in
    // the order the workers of that row will lay them out.
    long blockLoad(int row, int firstQ, int len,
                   const std::vector<QueryState>& batch) const;

    // Which worker ends this item's chain, and so reports its result.
    int lastRankOf(int row, int item) const;

    // ---- cost model, paper Section 4.2.1 ----

    // The clusters one query visits. The only place probe lists come from --
    // the profiling pass, the search, and the single-machine reference all go
    // through it, so a synthetic workload stays consistent between them and
    // differing keeps its meaning.
    std::vector<int> probesFor(int queryId, int nprobe) const;

    // Learns which clusters the workload favours, by centroid assignment only,
    // before the layout is fixed (the paper's pre-query phase).
    void warmupPlan(int queries, int nprobe);

    // Assigns clusters to bVec partitions, heaviest first to whichever is
    // lightest so far, and returns what each ends up carrying. Fills owner
    // when one is given. Used both to make the split and, before the split
    // exists, to tell the cost model what a given bVec would cost.
    std::vector<double> partitionLoads(int bVec, std::vector<int>* owner) const;

    // I(pi): the spread of computation across the vector partitions, from how
    // often each cluster was probed and how big it is.
    double imbalanceOf(int bVec, int bDim) const;

    // Share of the distance work still done when the dimensions are cut into
    // bDim slices: more slices, more chances to stop early. Measured values,
    // and a term the paper's model does not have -- see CLAUDE.md.
    double pruneFactor(int bDim) const;

    // C(pi,Q): computation, communication, and the imbalance penalty, all in
    // multiply-add equivalents.
    double estimateCost(int bVec, int bDim) const;

    // Picks the grid with the lowest cost, by enumerating the factorisations
    // of the worker count.
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

    // The two pools --skew draws from, with running totals of their cluster
    // sizes so a draw can be made proportional to size. allIds_ is every
    // cluster; hotIds_ is the eighth that the skew concentrates on. Both are
    // built once the index exists.
    std::vector<int> allIds_;
    std::vector<double> allPrefix_;
    std::vector<int> hotIds_;
    std::vector<double> hotPrefix_;

    // Draws one cluster from a pool with probability proportional to size,
    // advancing the caller's generator state.
    int drawWeighted(const std::vector<int>& ids,
                     const std::vector<double>& prefix,
                     unsigned int& state) const;

    // pruning counters (paper Table 3), by position in the chain rather than
    // by worker, since rotation moves each worker between the two.
    long scanned_;                      // candidates offered in total
    std::vector<long> scannedRow_;      // per vector partition
    std::vector<long> aliveAfterStage_;  // still alive after the s-th slice

    // Who passes what to whom, at both levels (engine/search_order.h).
    // chainOrder_ runs a cluster across the columns of a row, groupOrder_ runs
    // a query group across the vector partitions. Both are built in
    // splitGrid(); chainOrder_ is also shipped to the workers.
    SearchOrder chainOrder_;
    SearchOrder groupOrder_;

    // [total, idle, recv, compute, send, jobs] per worker, filled by shutdown()
    std::vector<std::vector<double>> workerTimes_;

    // Chain reordering state (§4.3). prevCompute_ is what each worker had
    // done at the previous look, so the difference is this batch's work;
    // colSmooth_ is the per-column load with the swings taken out.
    static constexpr double kReorderDeadzone = 0.10;
    std::vector<double> prevCompute_;
    std::vector<double> colSmooth_;
    int reorders_ = 0;
};

}  // namespace harmony

#endif  // HARMONY_NODE_MASTER_NODE_H
