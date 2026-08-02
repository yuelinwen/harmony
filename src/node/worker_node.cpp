#include "worker_node.h"

#include <iostream>

#include "../index/distance.h"

namespace harmony {
    void WorkerNode::setDimensions(int dimBegin, int dimEnd) {
        dimBegin_ = dimBegin;
        dimEnd_ = dimEnd;
    }

    void WorkerNode::addCluster(int clusterId, const std::vector<int>& ids,
                            const Dataset& base) {
        int myDim = dimEnd_ - dimBegin_;

        ClusterBlock block;
        block.clusterId = clusterId;
        block.ids = ids;
        block.data.resize(ids.size() * myDim);

        for (size_t i = 0; i < ids.size(); ++i) {
            const float* v = base.vec(ids[i]);
            for (int j = 0; j < myDim; ++j) {
                block.data[i * myDim + j] = v[dimBegin_ + j];   // only my slice
            }
        }

        blocks_.push_back(block);
    }

    long WorkerNode::vectorCount() const {
        long n = 0;
        for (size_t b = 0; b < blocks_.size(); ++b) {
            n = n + (long)blocks_[b].ids.size();
        }
        return n;
    }

    std::vector<float> WorkerNode::partialDistances(const float* query,
                                                    const std::vector<int>& clusterIds) {
        int myDim = dimEnd_ - dimBegin_;
        std::vector<float> out;

        for (int i = 0; i < (int)clusterIds.size(); ++i) {
            for (int b = 0; b < (int)blocks_.size(); ++b) {
                if (blocks_[b].clusterId != clusterIds[i]) {
                    continue;
                }
                const ClusterBlock& block = blocks_[b];
                for (int v = 0; v < (int)block.ids.size(); ++v) {
                    // query is the full vector, so skip to my slice of it
                    out.push_back(l2DistanceSquared(query + dimBegin_,
                                                    &block.data[(size_t)v * myDim],
                                                    myDim));
                }
            }
        }

        return out;
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
