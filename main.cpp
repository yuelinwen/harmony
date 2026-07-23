#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "src/node/master_node.h"
#include "src/node/worker_node.h"

int main(int argc, char** argv) {
    int id = -1;

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "-id=", 4) == 0) {
            id = std::atoi(argv[i] + 4);
        } else if (std::strncmp(argv[i], "--id=", 5) == 0) {
            id = std::atoi(argv[i] + 5);
        }
    }

    if (id < 0) {
        std::cout << "usage: " << argv[0] << " -id=0 (master) | -id=1,2,... (worker)" << std::endl;
        return 1;
    }

    std::unique_ptr<harmony::Node> node;
    if (id == 0) {
        node = std::make_unique<harmony::MasterNode>();
    } else {
        node = std::make_unique<harmony::WorkerNode>(id);
    }

    return node->run();
}
