#ifndef HARMONY_ENGINE_TOPK_HEAP_H
#define HARMONY_ENGINE_TOPK_HEAP_H

#include <algorithm>
#include <queue>
#include <vector>

// TopKHeap: keeps the K nearest candidates seen so far, as a max-heap of size
// K, so its top is the WORST of the K kept and a new candidate is compared
// against that one value. worst() is the admission threshold, which is the
// paper's pruning threshold tau^2 (Algorithm 1).

namespace harmony {

// One search result: which vector, and how far it is.
struct Candidate {
    int id;
    float dist;
};

// Comparison for the priority_queue: largest dist ends up on top.
struct CandidateLess {
    bool operator()(const Candidate& a, const Candidate& b) const {
        return a.dist < b.dist;
    }
};

class TopKHeap {
public:
    TopKHeap(int k) {
        k_ = k; // top k records
    }

    // Offers a candidate to the heap. Kept only if it is among the K best.
    void push(int id, float dist) {
        if ((int)heap_.size() < k_) {
            Candidate c;
            c.id = id;
            c.dist = dist;
            heap_.push(c);
            return;
        }

        if (dist < heap_.top().dist) {
            heap_.pop();
            Candidate c;
            c.id = id;
            c.dist = dist;
            heap_.push(c);
        }
    }

    // The worst kept candidate's distance, or infinity while the heap is not
    // full and every candidate is accepted.
    float worst() const {
        if ((int)heap_.size() < k_) {
            return 1e30f;
        }
        return heap_.top().dist;
    }

    // The kept candidates, nearest first. Copies the heap, so the object stays
    // usable afterwards.
    std::vector<Candidate> results() const {
        std::vector<Candidate> out;
        std::priority_queue<Candidate, std::vector<Candidate>, CandidateLess> copy = heap_;
        while (!copy.empty()) {
            out.push_back(copy.top());
            copy.pop();
        }
        // The heap gives them worst-first, so reverse into best-first order.
        std::reverse(out.begin(), out.end());
        return out;
    }

    int size() const {
        return (int)heap_.size();
    }

private:
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateLess> heap_;
    int k_;
};

}  // namespace harmony

#endif  // HARMONY_ENGINE_TOPK_HEAP_H
