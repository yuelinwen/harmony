#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "src/index/dataset.h"
#include "src/index/distance.h"
#include "src/index/ivf_index.h"
#include "src/engine/pruning_scanner.h"
#include "src/engine/slice_plan.h"
#include "src/engine/topk_heap.h"
#include "src/node/master_node.h"
#include "src/node/worker_node.h"

// Reads the groundtruth file (same layout as base/query, but int32 data):
//     int32 n, int32 dim, then n * dim int32 values.
// Row i = the ids of the true nearest neighbors of query i.
static bool loadGroundtruth(const std::string& path, std::vector<int>& data, int& n, int& dim) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        std::cerr << "cannot open file: " << path << std::endl;
        return false;
    }
    int header[2];
    if (std::fread(header, sizeof(int), 2, file) != 2) {
        std::fclose(file);
        return false;
    }
    n = header[0];
    dim = header[1];
    data.resize((size_t)n * dim);
    size_t got = std::fread(data.data(), sizeof(int), data.size(), file);
    std::fclose(file);
    return got == data.size();
}

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

    // --- temporary: brute-force search on query 0, check against groundtruth ---

    harmony::Dataset base;
    harmony::Dataset query;
    if (!base.load("Data/sift_base.bin") || !query.load("Data/sift_query.bin")) {
        return 1;
    }
    std::cout << "base:  n=" << base.getN() << " dim=" << base.getDim() << std::endl;
    std::cout << "query: n=" << query.getN() << " dim=" << query.getDim() << std::endl;

    std::vector<int> gt;
    int gtN = 0;
    int gtDim = 0;
    if (!loadGroundtruth("Data/sift_gt.bin", gt, gtN, gtDim)) {
        return 1;
    }
    std::cout << "gt:    n=" << gtN << " dim=" << gtDim << std::endl;

    // Brute force: compare query 0 against every base vector, keep top 100.
    int k = 100;
    harmony::TopKHeap heap(k);
    for (int i = 0; i < base.getN(); ++i) {
        float d = harmony::l2DistanceSquared(query.vec(0), base.vec(i), base.getDim());
        heap.push(i, d);
    }

    std::vector<harmony::Candidate> mine = heap.results();

    // Compare: my top 10 ids vs groundtruth row 0 top 10 ids.
    std::cout << "\nmine:  ";
    for (int j = 0; j < 10; ++j) {
        std::cout << mine[j].id << " ";
    }
    std::cout << "\ngt:    ";
    for (int j = 0; j < 10; ++j) {
        std::cout << gt[j] << " ";   // row 0 starts at index 0
    }
    std::cout << std::endl;

    // Full check over all 100: count how many match position by position.
    int match = 0;
    for (int j = 0; j < k; ++j) {
        if (mine[j].id == gt[j]) {
            match = match + 1;
        }
    }
    std::cout << "match: " << match << "/" << k << std::endl;

    // See why the nearest neighbour is "near": print the query next to the
    // closest and the farthest of the 100 results, first 20 dimensions.
    int nearestId = mine[0].id;
    int farthestId = mine[k - 1].id;

    std::cout << "\nquery[0]        :";
    for (int j = 0; j < 20; ++j) {
        std::cout << " " << query.vec(0)[j];
    }
    std::cout << "\nnearest  " << nearestId << " :";
    for (int j = 0; j < 20; ++j) {
        std::cout << " " << base.vec(nearestId)[j];
    }
    std::cout << "\nfarthest " << farthestId << " :";
    for (int j = 0; j < 20; ++j) {
        std::cout << " " << base.vec(farthestId)[j];
    }
    std::cout << "\n\ndist to nearest  = " << mine[0].dist << std::endl;
    std::cout << "dist to farthest = " << mine[k - 1].dist << std::endl;

    // --- A4: dimension-level pruning experiment (paper Fig.2(a), Table 3) ---
    //
    // Same brute-force search, but the 128 dims are split into 4 slices.
    // After each slice we check the accumulated partial distance against
    // the current threshold (heap.worst()). Partial sums only grow, so if
    // the sum already exceeds the threshold, the remaining slices can be
    // skipped safely: this vector can never enter the top-K.

    int nSlices = 4;
    harmony::SlicePlan plan{base.getDim(), nSlices};
    std::vector<long> prunedAt(nSlices, 0);           // pruned after slice s

    auto t0 = std::chrono::steady_clock::now();

    harmony::TopKHeap pheap(k);
    harmony::scanAll(base, query.vec(0), plan, pheap, prunedAt);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Correctness: pruning must not change the result at all.
    std::vector<harmony::Candidate> pruneResults = pheap.results();
    int pmatch = 0;
    for (int j = 0; j < k; ++j) {
        if (pruneResults[j].id == gt[j]) {
            pmatch = pmatch + 1;
        }
    }

    std::cout << "\n--- pruning experiment (" << nSlices << " slices) ---" << std::endl;
    long total = base.getN();
    long cumulative = 0;
    for (int s = 0; s < nSlices; ++s) {
        cumulative = cumulative + prunedAt[s];
        std::cout << "pruned after slice " << (s + 1) << ": " << prunedAt[s]
                  << "  (cumulative " << (100.0 * cumulative / total) << "%)" << std::endl;
    }
    long survived = total - cumulative;               // computed all slices
    std::cout << "survived all slices: " << survived
              << "  (" << (100.0 * survived / total) << "%)" << std::endl;
    std::cout << "match with gt: " << pmatch << "/" << k << "  (must be " << k << ")" << std::endl;
    std::cout << "time: " << ms << " ms" << std::endl;

    // --- A5: IVF index (own kmeans + own scan, no dependencies) ---

    harmony::IvfIndex ivf;
    int nlist = 256;
    int nprobe = 32;

    std::cout << "\n--- building IVF index (nlist=" << nlist << ") ---" << std::endl;
    auto b0 = std::chrono::steady_clock::now();
    ivf.build(base, nlist, 10);
    auto b1 = std::chrono::steady_clock::now();
    std::cout << "build time: "
              << std::chrono::duration<double>(b1 - b0).count() << " s" << std::endl;

    // search query 0 through the index
    auto s0 = std::chrono::steady_clock::now();
    std::vector<harmony::Candidate> ivfResults = ivf.search(base, query.vec(0), nprobe, k);
    auto s1 = std::chrono::steady_clock::now();
    double ivfMs = std::chrono::duration<double, std::milli>(s1 - s0).count();

    // recall@100: how many of my 100 ids appear anywhere in the gt 100.
    // (not position-by-position: IVF is approximate, order can shift)
    int hit = 0;
    for (int a = 0; a < k; ++a) {
        for (int g = 0; g < k; ++g) {
            if (ivfResults[a].id == gt[g]) {
                hit = hit + 1;
                break;
            }
        }
    }
    std::cout << "ivf search (nprobe=" << nprobe << "): " << ivfMs << " ms" << std::endl;
    std::cout << "recall@100: " << hit << "/" << k
              << " = " << (100.0 * hit / k) << "%" << std::endl;

    // --- end temporary ---

    std::unique_ptr<harmony::Node> node;
    if (id == 0) {
        node = std::make_unique<harmony::MasterNode>();
    } else {
        node = std::make_unique<harmony::WorkerNode>(id);
    }

    return node->run();
}
