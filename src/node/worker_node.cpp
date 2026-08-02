#include "worker_node.h"

#include <iostream>

#include "../index/distance.h"

namespace harmony {
    void WorkerNode::setDimCount(int myDim) {
        myDim_ = myDim;
    }

    void WorkerNode::addCluster(int clusterId, const std::vector<int>& ids,
                                const std::vector<float>& data) {
        ClusterBlock block;
        block.clusterId = clusterId;
        block.ids = ids;
        block.data = data;      // already sliced by the master

        blocks_.push_back(block);
    }

    long WorkerNode::vectorCount() const {
        long n = 0;
        for (size_t b = 0; b < blocks_.size(); ++b) {
            n = n + (long)blocks_[b].ids.size();
        }
        return n;
    }

    void WorkerNode::accumulate(const float* querySlice, int clusterId, float threshold,
                                std::vector<float>& sums, std::vector<char>& alive) {
        for (int b = 0; b < (int)blocks_.size(); ++b) {
            if (blocks_[b].clusterId != clusterId) {
                continue;
            }
            const ClusterBlock& block = blocks_[b];
            for (int v = 0; v < (int)block.ids.size(); ++v) {
                if (!alive[v]) {
                    continue;      // an earlier worker already dropped it
                }
                sums[v] = sums[v] + l2DistanceSquared(querySlice,
                                                      &block.data[(size_t)v * myDim_],
                                                      myDim_);
                if (sums[v] > threshold) {
                    alive[v] = 0;  // cannot reach the top-K, stop here
                }
            }
        }
    }



int WorkerNode::run() {
    running_ = true;
    std::cout << "worker node started, id=" << id_ << std::endl;

    // In v1 the worker has no loop of its own: the master builds it, calls
    // setData() once, then calls search() per query. run() only matters
    // from v4 on, when the worker becomes its own process and waits for
    // messages instead.
    //
    //   TODO(v4): loop { receive query -> search -> send back }

    return 0;
}

}  // namespace harmony
