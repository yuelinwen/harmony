#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <faiss/IndexFlat.h>

#include "../src/engine/stopwatch.h"
#include "../src/index/dataset.h"

// maketruth: the true k nearest neighbours of every query, by brute force.
//
// Datasets from ann-benchmarks carry their own groundtruth and
// scripts/data.sh pulls it out of the hdf5. The ones the paper uses that do
// not -- msong, the UCR series, sift1B -- arrive as bare vectors, and ./main
// refuses to start without a _gt.bin. This writes one.
//
// A separate binary on purpose, not a mode of ./main. faiss is linked here
// and nowhere else, so the nine workers still need nothing installed, the
// binary mpirun copies to them does not grow, and scripts/build.sh keeps
// building ./main on a machine that has no faiss at all.
//
// The two files it writes are the ones Groundtruth::loadIds and
// loadDistances read (src/test/verify.cpp):
//
//     _gt.bin   int32 n, int32 k, int32   ids[n * k]
//     _gtd.bin  int32 n, int32 k, float32 dist[n * k]
//
// nearest first, distances squared -- the same convention as every other
// .bin file here.

namespace {

// Two int32 of header then the body, the layout scripts/data.sh writes.
bool writeBin(const std::string& path, int n, int k, const void* body,
              size_t itemSize) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        std::cerr << "cannot write " << path << std::endl;
        return false;
    }

    int header[2] = {n, k};
    bool ok = std::fwrite(header, sizeof(int), 2, f) == 2;
    if (ok) {
        ok = std::fwrite(body, itemSize, (size_t)n * k, f) == (size_t)n * k;
    }
    std::fclose(f);

    if (!ok) {
        // A half-written answer key is worse than none: it would load, pass
        // the header check, and quietly make recall wrong.
        std::remove(path.c_str());
        std::cerr << "short write: " << path << std::endl;
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::string data = "Data/sift";
    std::string out;
    int k = 100;

    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];
        bool hasValue = (i + 1 < argc);

        if (opt == "--data" && hasValue) {
            data = argv[++i];
        } else if (opt == "--out" && hasValue) {
            out = argv[++i];
        } else if (opt == "--k" && hasValue) {
            k = std::atoi(argv[++i]);
        } else {
            std::cerr << "usage: " << argv[0]
                      << " --data <prefix> [--out <prefix>] [--k <int>]\n"
                      << "  reads  <data>_base.bin and <data>_query.bin\n"
                      << "  writes <out>_gt.bin and <out>_gtd.bin  (--out"
                      << " defaults to --data)\n";
            return 1;
        }
    }

    if (out.empty()) {
        out = data;   // in place, which is what a new dataset wants
    }
    if (k < 1) {
        std::cerr << "--k must be at least 1, got " << k << std::endl;
        return 1;
    }

    harmony::Dataset base;
    harmony::Dataset query;
    if (!base.load(data + "_base.bin") || !query.load(data + "_query.bin")) {
        return 1;
    }
    if (base.getDim() != query.getDim()) {
        std::cerr << "base is " << base.getDim() << "-dimensional but query is "
                  << query.getDim() << std::endl;
        return 1;
    }
    if (k > base.getN()) {
        std::cerr << "--k " << k << " is more than the " << base.getN()
                  << " vectors there are" << std::endl;
        return 1;
    }

    int nb = base.getN();
    int nq = query.getN();
    int d = base.getDim();
    std::cout << "base " << nb << " x " << d << ", query " << nq << " x " << d
              << ", k " << k << std::endl;

    // IndexFlatL2 is exhaustive -- every query against every base vector, no
    // clustering and no approximation. That is what makes this an answer key
    // rather than one more thing that has to be checked. It returns squared
    // distances, which is the convention everywhere else here.
    //
    // It holds its own copy of the vectors, so this peaks at twice the base
    // file in memory.
    harmony::Stopwatch watch;
    faiss::IndexFlatL2 index(d);
    index.add(nb, base.vec(0));
    std::cout << "added in " << watch.seconds(true) << " s" << std::endl;

    std::vector<float> dist((size_t)nq * k);
    std::vector<faiss::idx_t> label((size_t)nq * k);
    index.search(nq, query.vec(0), k, dist.data(), label.data());
    std::cout << "searched in " << watch.seconds() << " s" << std::endl;

    // The reader wants int32. faiss counts in int64 because it indexes
    // datasets larger than this program could hold anyway.
    std::vector<int> ids((size_t)nq * k);
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = (int)label[i];
    }

    if (!writeBin(out + "_gt.bin", nq, k, ids.data(), sizeof(int)) ||
        !writeBin(out + "_gtd.bin", nq, k, dist.data(), sizeof(float))) {
        return 1;
    }

    std::cout << "wrote " << out << "_gt.bin and " << out << "_gtd.bin"
              << std::endl;
    return 0;
}
