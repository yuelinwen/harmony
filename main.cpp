#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "src/index/dataset.h"
#include "src/node/master_node.h"
#include "src/node/worker_node.h"
#include "src/index/distance.h"
#include "src/engine/topk_heap.h"

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

    // --- temporary: check that the dataset loads correctly ---
    harmony::Dataset base;
    if (!base.load("Data/sift_base.bin")) {
        return 1;
    }
    std::cout << "loaded base: n=" << base.getN() << " dim=" << base.getDim() << std::endl;

    const float* v = base.vec(0);
    for (int j = 0; j < base.getDim(); ++j) {
        std::cout << v[j] << " ";
    }
    std::cout << std::endl;

    std::unique_ptr<harmony::Node> node;
    if (id == 0) {
        node = std::make_unique<harmony::MasterNode>();
    } else {
        node = std::make_unique<harmony::WorkerNode>(id);
    }

    // vector calculation verfiy
    float a[2] = {1.0f, 2.0f};
    float b[2] = {4.0f, 6.0f};
    std::cout << harmony::l2DistanceSquared(a, b, 2) << std::endl;         // 应输出 25
    std::cout << harmony::l2DistanceSquaredRange(a, b, 0, 1) << std::endl; // 应输出 9
    std::cout << harmony::l2DistanceSquaredRange(a, b, 1, 2) << std::endl; // 应输出 16

    // top k heap
    harmony::TopKHeap heap(3);
    heap.push(10, 5.0f);
    heap.push(11, 2.0f);
    heap.push(12, 8.0f);
    heap.push(13, 1.0f);
    heap.push(14, 9.0f);

    std::cout << "worst = " << heap.worst() << std::endl;   // should be 5
    for (auto& c : heap.results()) {
        std::cout << "id=" << c.id << " dist=" << c.dist << std::endl;
    }
    return node->run();
}
