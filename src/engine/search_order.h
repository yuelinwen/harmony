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

    // Built from chains given outright, one per item, each listing every
    // worker in the order that item visits them. The rotation above is the
    // only order this class makes on its own; anything else -- such as §4.3's
    // "put the overloaded machine last" -- is a policy, and policies live
    // where the load is measured, not here.
    SearchOrder(const std::vector<std::vector<int> >& chains, int workers) {
        workers_ = workers;
        items_ = (int)chains.size();
        chain_ = chains;
        link();
    }

    // The workers handling this item, first stop first.
    const std::vector<int>& chain(int item) const { return chain_[item]; }

    // One worker's row of the table, which is what gets handed to it over
    // MPI: entry i is who it passes item i to, or takes item i from, and -1
    // means it is the end or the start of that item's chain.
    const std::vector<int>& nextRow(int worker) const { return next_[worker]; }
    const std::vector<int>& prevRow(int worker) const { return prev_[worker]; }

private:
    void build(int workers, int items, bool rotate) {
        workers_ = workers;
        items_ = items;

        // step[w][t]: the item worker w takes at step t. Every worker walks
        // the same list of items, offset by gap per worker, so at any step the
        // workers are on different items and none of them collide. Only the
        // chains read off it below outlive this function.
        std::vector<std::vector<int> > step(workers_, std::vector<int>(items_, 0));
        int gap = rotate ? ((items_ + workers_ - 1) / workers_) : 0;
        for (int w = 0; w < workers_; ++w) {
            for (int i = 0; i < items_; ++i) {
                int t = (items_ > 0) ? ((i + gap * w) % items_) : 0;
                step[w][t] = i;
            }
        }

        // Reading the same table by item gives the chain: the workers that
        // take item i, in the order they take it.
        chain_.assign(items_, std::vector<int>());
        for (int t = 0; t < items_; ++t) {
            for (int w = 0; w < workers_; ++w) {
                chain_[step[w][t]].push_back(w);
            }
        }

        link();
    }

    // next/prev tables, read off the chains.
    void link() {
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
    std::vector<std::vector<int>> chain_;
    std::vector<std::vector<int>> next_;
    std::vector<std::vector<int>> prev_;
};

}  // namespace harmony

#endif  // HARMONY_ENGINE_SEARCH_ORDER_H
