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

// A Candidate travels between machines as raw bytes (TAG_TOPK), which needs
// it to be exactly these two 4-byte fields with no padding. Every rank is the
// same build on the same architecture, so that is all this has to hold.
static_assert(sizeof(Candidate) == 8, "Candidate must pack into 8 bytes");

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
        seed_ = 1e30f;
    }

    // Offers a candidate to the heap. Kept only if it is among the K best.
    void push(int id, float dist) {
        if ((int)heap_.size() >= k_) {
            // Negated rather than written as >=, so that a dist that
            // compares unordered against the top is rejected, exactly as the
            // straight `dist < top` admission test used to reject it.
            if (!(dist < heap_.top().dist)) {
                return;   // no better than the K already kept
            }
            heap_.pop();
        }
        Candidate c;
        c.id = id;
        c.dist = dist;
        heap_.push(c);
    }

    // A threshold to use until the heap has K of its own. Prewarming measures
    // a few real distances, keeps the K-th as this, and throws the candidates
    // away -- they are all in clusters the search visits anyway, so they come
    // back on their own. That is what the sample does, and it saves the
    // pipeline from having to remember which ones it already has.
    void seedThreshold(float t) {
        seed_ = t;
    }

    // The worst kept candidate's distance, or the seeded threshold while the
    // heap is not yet full -- infinity if nothing seeded it.
    float worst() const {
        if ((int)heap_.size() < k_) {
            return seed_;
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

    int capacity() const {
        return k_;
    }

private:
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateLess> heap_;
    int k_;
    float seed_;
};

}  // namespace harmony

#endif  // HARMONY_ENGINE_TOPK_HEAP_H
