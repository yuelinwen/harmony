#ifndef HARMONY_NODE_MASTER_NODE_H
#define HARMONY_NODE_MASTER_NODE_H

#include <string>
#include <vector>

#include "node.h"
#include "../config.h"
#include "../comm/messages.h"
#include "../engine/search_order.h"
#include "../engine/slice_plan.h"
#include "../engine/stopwatch.h"
#include "../index/dataset.h"
#include "../index/ivf_index.h"
#include "../test/metrics.h"
#include "../test/verify.h"

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
        scanned_ = 0;
    }
    ~MasterNode() override = default;

    // Load, build, distribute, then search -- and check the answer against a
    // single machine and report recall, QPS and the pruning ratios.
    int run() override;

    // Reads the base and query vectors. Returns false if either file fails.
    bool loadData(const std::string& basePath, const std::string& queryPath);

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

    // The most blocks one query batch can have in flight at once.
    //
    // The only place this is worked out. A worker sizes its outgoing buffer
    // pool from the value it is told at setup, because the two numbers have to
    // satisfy pool >= in-flight or the wait before reusing a buffer can
    // deadlock against a neighbour in the same row -- see TAG_SETUP in
    // comm/messages.h. They used to be two independent formulas that happened
    // to agree at the old --block default, and stopped agreeing when that
    // default changed.
    int maxBlocksInFlight() const;

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

    // What one machine would need for this index: the vectors, one id per
    // vector in the inverted lists, and the centroids. The denominator of
    // paper Table 4, and the only half of it the master can work out on its
    // own -- what the workers hold, they report.
    long singleMachineMemory() const;

    // What one query of the current batch needs.
    struct QueryState {
        int id;                       // row in query_
        std::vector<int> clusters;    // its nprobe nearest
    };

    // Algorithm 1, lines 19-23. Runs a batch of queries, the paper's
    // QueryBatch: those probing the same cluster share one visit to it. One
    // heap per query, where line 20 reads as one heap for the whole set --
    // a shared heap would prune one query against another's neighbours.
    std::vector<std::vector<Candidate>> queryPipeline(int firstQuery, int count,
                                                      int nprobe, int k);

    // Algorithm 1, lines 1-5. Seeds the heap's threshold from real distances,
    // so there is something to prune against from the very first candidate.
    void prewarmHeap(const float* query, QueryState& state, TopKHeap& heap);

    // Algorithm 1, lines 13-18. Runs the clusters of every vector partition
    // through the dimension pipeline and pushes the survivors into the heaps.
    //
    // groupMembers[g] holds the batch positions of query group g, a
    // contiguous run. Group g visits the vector partitions in the order
    // groupOrder_ gives it, one at a time, so by the time it reaches its
    // second partition its heaps already carry the first one's distances
    // (Fig. 5a).
    // Groups advance independently -- at any stage the mapping group -> row is
    // a permutation, so every row stays busy.
    void vectorPipeline(const std::vector<std::vector<int>>& groupMembers,
                        const std::vector<QueryState>& batch,
                        std::vector<TopKHeap>& heaps);

    // Algorithm 1, lines 6-12. Sends one block of queries -- [firstQ,
    // firstQ+len) of the batch -- to every worker in row `row` and returns;
    // they pass the running totals down the chain and only the last reports
    // back. `item` picks the chain, `slot` names its tags.
    void dispatchBlock(int row, int firstQ, int len, int item, int slot);

    // One query group's thresholds to the row about to work on it (paper §5).
    // Workers keep them between jobs, so this goes once per partition a group
    // enters rather than with every block.
    void sendThresholds(int row, int firstQ, int len,
                        const std::vector<TopKHeap>& heaps);

    // Candidates a block of queries contributes in one vector partition, in
    // the order the workers of that row will lay them out.
    long blockLoad(int row, int firstQ, int len,
                   const std::vector<QueryState>& batch) const;

    // Which worker ends this item's chain, and so reports its result.
    int lastRankOf(int row, int item) const;

    // ---- cost model, paper Section 4.2.1 ----

    // Which of the two workloads a probe list belongs to. One parameter rather
    // than a skew and a hot set, because the two always go together: using the
    // profiling skew with the test hot set, or the other way round, would be a
    // silent bug and this makes it unspellable.
    enum Workload {
        kIndexWorkload,   // what the layout was built from (--indexskew)
        kTestWorkload     // what it is then measured on (--skew, --skewshift)
    };

    // The clusters one query visits. The only place probe lists come from --
    // the profiling pass, the search, and the single-machine reference all go
    // through it, so a synthetic workload stays consistent between them and
    // differing keeps its meaning.
    std::vector<int> probesFor(int queryId, int nprobe, Workload which) const;

    // Learns which clusters the workload favours, by centroid assignment only,
    // before the layout is fixed (the paper's pre-query phase).
    void warmupPlan(int queries, int nprobe);

    // Assigns clusters to bVec partitions, heaviest first to whichever is
    // lightest so far, and returns what each ends up carrying. Fills owner
    // when one is given. Used both to make the split and, before the split
    // exists, to tell the cost model what a given bVec would cost.
    std::vector<double> partitionLoads(int bVec, std::vector<int>* owner) const;

    // How often a cluster was probed in the profiling pass, never below one.
    //
    // The floor matters: with --warmup 0 nothing is profiled, every count is
    // zero, and a weight of zero makes partitionLoads() find no lightest
    // partition at all -- measured, every cluster went to the first one
    // (100% / 0% / ... / 0%) at a quarter of the throughput. Treating an
    // unprofiled workload as "all clusters equally likely" degrades to plain
    // size-weighted LPT, which is the sensible reading of "do not profile".
    long clusterHitsOf(int c) const;

    // I(pi): the spread of computation across the vector partitions, from how
    // often each cluster was probed and how big it is.
    double imbalanceOf(int bVec, int bDim) const;

    // Share of the distance work still done when the dimensions are cut into
    // bDim slices: more slices, more chances to stop early. Measured values,
    // and a term the paper's C(pi,Q) does not have -- it prices the work
    // before pruning, so without this every extra slice looks like pure cost.
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

    // The test workload's hot set: hotIds_ with --skewshift of its members
    // replaced by clusters from outside it. Equal to hotIds_ when the shift is
    // 0, which is why the layout absorbs the skew in that case.
    std::vector<int> testHotIds_;
    std::vector<double> testHotPrefix_;

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

    // WORKER_TIMES numbers per worker, in the order comm/messages.h lists
    // them, filled by collectStats().
    std::vector<std::vector<double>> workerTimes_;

    // Wall time from the start of run(), so a CSV row can say how long the
    // whole thing took -- loading, clustering and distribution included, none
    // of which the search timer sees.
    Stopwatch wall_;

    // What distributeData() took: the paper's index-build cost that is not
    // clustering, and the sample's preSearchTime.
    double distributeSeconds_ = 0.0;

    // The true neighbours, for recall and r2 (src/test/verify.h).
    Groundtruth truth_;

    // The single-machine answers for the nprobe being run, one entry per
    // query. Computed once by referencePass() rather than per query inside the
    // batch loop, which is where it used to live: there it ran between
    // batches and the workers' idle time absorbed it, so the Fig. 9
    // breakdown could not be read with --check on.
    std::vector<std::vector<Candidate>> reference_;

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
