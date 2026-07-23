#include "master_node.h"

#include <iostream>

namespace harmony {

int MasterNode::run() {
    running_ = true;
    std::cout << "master node started" << std::endl;

    // TODO(phase 2): setup() — connect to all workers, handshake
    // TODO(phase 3): buildAndDistributeIndex(); loop dispatchBatch()

    return 0;
}

}  // namespace harmony
