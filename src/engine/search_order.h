#ifndef HARMONY_ENGINE_SEARCH_ORDER_H
#define HARMONY_ENGINE_SEARCH_ORDER_H

#include <vector>

namespace harmony {

// Who handles what, and in which order, when several workers pass one item
// along a chain.
//
// The point is that no worker is always the first stop: the first stop is the
// one that can prune nothing, since it has no partial sum to test yet
// (paper §4.3). Every item therefore enters the chain at a different worker.
//
// Held as a table rather than computed from a formula. The rotation below is
// the only order built so far, but a table can express any order, which is
// what §4.3's load-aware reordering -- put the overloaded worker last -- would
// need.
//
// Workers and items are both 0-based here.
class SearchOrder {
public:
    SearchOrder() { build(1, 0, true); }

    SearchOrder(int workers, int items, bool rotate) {
        build(workers, items, rotate);
    }

    int workers() const { return workers_; }
    int items() const { return items_; }

    // The workers handling this item, first stop first.
    const std::vector<int>& chain(int item) const { return chain_[item]; }

    // Who this worker passes the item to, and who it takes it from.
    // -1 means there is nobody: the end and the start of the chain.
    int next(int worker, int item) const { return next_[worker][item]; }
    int prev(int worker, int item) const { return prev_[worker][item]; }

    // Flattened [worker][item] table, for handing a worker its own row over
    // MPI. Row w is nextRow(w), which is items() long.
    const std::vector<int>& nextRow(int worker) const { return next_[worker]; }
    const std::vector<int>& prevRow(int worker) const { return prev_[worker]; }

private:
    void build(int workers, int items, bool rotate) {
        workers_ = workers;
        items_ = items;

        // step_[w][t]: the item worker w takes at step t. Every worker walks
        // the same list of items, offset by gap per worker, so at any step the
        // workers are on different items and none of them collide.
        step_.assign(workers_, std::vector<int>(items_, 0));
        int gap = rotate ? ((items_ + workers_ - 1) / workers_) : 0;
        for (int w = 0; w < workers_; ++w) {
            for (int i = 0; i < items_; ++i) {
                int t = (items_ > 0) ? ((i + gap * w) % items_) : 0;
                step_[w][t] = i;
            }
        }

        // Reading the same table by item gives the chain: the workers that
        // take item i, in the order they take it.
        chain_.assign(items_, std::vector<int>());
        for (int t = 0; t < items_; ++t) {
            for (int w = 0; w < workers_; ++w) {
                chain_[step_[w][t]].push_back(w);
            }
        }

        next_.assign(workers_, std::vector<int>(items_, -1));
        prev_.assign(workers_, std::vector<int>(items_, -1));
        for (int i = 0; i < items_; ++i) {
            const std::vector<int>& c = chain_[i];
            for (int p = 0; p + 1 < (int)c.size(); ++p) {
                next_[c[p]][i] = c[p + 1];
                prev_[c[p + 1]][i] = c[p];
            }
        }
    }

    int workers_;
    int items_;
    std::vector<std::vector<int>> step_;
    std::vector<std::vector<int>> chain_;
    std::vector<std::vector<int>> next_;
    std::vector<std::vector<int>> prev_;
};

}  // namespace harmony

#endif  // HARMONY_ENGINE_SEARCH_ORDER_H
