#include "worker_node.h"

#include <iostream>

namespace harmony {

int WorkerNode::run() {
    running_ = true;
    std::cout << "worker node started, id=" << id_ << std::endl;

    // TODO(phase 2): setup() — listen on control/data ports, wait for master
    // TODO(phase 3): receive data block, serve partial distance computations

    return 0;
}

}  // namespace harmony
