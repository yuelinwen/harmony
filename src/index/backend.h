#ifndef HARMONY_INDEX_BACKEND_H
#define HARMONY_INDEX_BACKEND_H

#include <memory>
#include <string>

#include "dataset.h"
#include "ivf_index.h"

// How the pruning decision gets its dimensions.
//
// Pruning stops computing a candidate as soon as its running sum passes the
// threshold, so how early it can stop depends on which dimensions are
// summed first. The baseline takes them in the order they arrive, which on
// Sift1M leaves the second of four slices pruning 3.1%. Putting the
// directions that separate the data most up front should do better, and the
// point of this interface is to measure by how much rather than argue.
//
// Two halves, because that is how the methods divide:
//
//   fit     look at the data and work out a transform
//   apply   rewrite the vectors, the queries and the centroids in place
//
// Everything downstream -- the grid, the distribution, the chain, the
// threshold test -- is untouched. A rotation is applied on the master before
// a byte is sent, so a worker never learns which backend is running.
//
// All three have to be rewritten together. The base vectors and the queries
// are obvious; the centroids are the one that is easy to miss, and missing
// them sends every query to the wrong clusters while still returning
// plausible-looking results.
//
// An orthogonal transform preserves distances exactly, so backends 1-4 must
// leave `differing` at 0. That is the regression test for this whole file:
// a non-zero differing after a rotation means the transform was applied
// somewhere incompletely, not that the method is approximate.

namespace harmony {

class PruningBackend {
public:
    virtual ~PruningBackend() {}

    // Learn whatever the transform needs from the data. Called once, after
    // the index is built -- residual PCA needs the centroids to exist.
    virtual void fit(const Dataset& base, const IvfIndex& index) {
        (void)base;
        (void)index;
    }

    // Rewrite all three in place.
    virtual void apply(Dataset& base, Dataset& query, IvfIndex& index) {
        (void)base;
        (void)query;
        (void)index;
    }

    // As given to --backend, and as recorded in the backend column.
    virtual const char* name() const = 0;
};

// Returns nullptr for a name that is not a backend, so the caller can print
// the list rather than run something the user did not ask for.
std::unique_ptr<PruningBackend> makeBackend(const std::string& name);

// Every name makeBackend accepts, for the error message and for --help.
std::string backendNames();

}  // namespace harmony

#endif  // HARMONY_INDEX_BACKEND_H
