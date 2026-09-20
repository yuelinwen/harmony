#ifndef HARMONY_TEST_VERIFY_H
#define HARMONY_TEST_VERIFY_H

#include <string>
#include <vector>

#include "../engine/topk_heap.h"
#include "../index/dataset.h"
#include "../index/ivf_index.h"

// Everything that decides whether an answer is right.
//
// Two different notions of "right" live here and they are not the same thing:
//
//   the groundtruth file   what the true nearest neighbours are, brute-forced
//                          over the whole dataset. Missing some of them is
//                          normal -- IVF only visits nprobe clusters -- so
//                          this gives recall and r2, which are measurements.
//
//   the single machine     what this program's own index returns for the same
//                          query and the same clusters, with no partitioning
//                          and no pruning. Missing any of these is a bug, so
//                          this gives `differing`, which is a verdict.
//
// Nothing in here is on the query path. It is the reason the numbers can be
// believed, not part of what produces them.

namespace harmony {

// The true nearest neighbours, and how far away they are.
class Groundtruth {
public:
    // Row q of _gt.bin holds the true top-k of query q, nearest first.
    bool loadIds(const std::string& path);

    // Their squared distances, same shape. Optional -- the file was added
    // after the first datasets were converted, and only r2 needs it.
    bool loadDistances(const std::string& path);

    bool haveDistances() const {
        return !dist_.empty();
    }

    // Share of the true top-k this result found. Order does not matter: a
    // neighbour found at a different rank is still found.
    double recallAt(int queryId, const std::vector<Candidate>& got, int k) const;

    // The sample's r2 (utils.h:734): sum(returned distances) / sum(true
    // distances) - 1, rank against rank. 0 means the result is as close as
    // the truth. Named r2 because the sample calls it that; it is not a
    // coefficient of determination.
    double r2Of(int queryId, const std::vector<Candidate>& got, int k) const;

private:
    std::vector<int> ids_;     // count_ rows of dim_ ids
    std::vector<float> dist_;  // the same shape, squared
    int count_ = 0;
    int dim_ = 0;
};

// How a batch of results compared against the single machine.
struct Agreement {
    int differing = 0;  // a different set of distances: wrong, must stay 0
    int ties = 0;       // the same distances, a tie at rank k broken the
                        // other way: both answers equally correct
};

// The whole query set on one machine, using the clusters it is given rather
// than picking its own -- so a synthetic workload stays the same on both
// sides and `differing` keeps its meaning.
//
// Parallel over queries, because that is how a single-node engine uses a
// machine and a serial baseline would hand the cluster a speedup equal to the
// core count before any distribution happened (paper §6.2.1, the sample's
// faiss_query_time). Timed like the distributed search: one untimed warm-up
// when loop is above 1, then loop passes averaged.
//
// Returns the seconds it took and fills `out` with one answer per query.
double referencePass(IvfIndex& index, const Dataset& base, const Dataset& query,
                     const std::vector<std::vector<int>>& probes, int k,
                     int loop, std::vector<std::vector<Candidate>>& out);

// Compares one result against the single machine's, as sets rather than
// position by position: squared distances on SIFT are integers, ties near
// rank k are common, and their order is not defined either way.
void compare(const std::vector<Candidate>& got,
             const std::vector<Candidate>& reference, int k, Agreement* into);

}  // namespace harmony

#endif  // HARMONY_TEST_VERIFY_H
