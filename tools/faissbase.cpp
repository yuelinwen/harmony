#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>

#include "../src/engine/stopwatch.h"
#include "../src/index/dataset.h"
#include "../src/test/verify.h"

// faissbase: the single-node baseline the paper measures against.
//
// The paper's headline is "4.93x over Faiss" (section 6.2.1), so a speedup is
// only comparable with it when the denominator is actually Faiss. This
// program's own single-machine pass is a different thing: it shares the
// clustering, the probe lists and the distance kernel with the distributed
// side, which is what makes it a correctness reference and also what stops it
// being an independent baseline.
//
// Fair in the sense section 6.1 asks for -- the same clustering algorithm
// (kmeans), the same nlist, and the same thread count one worker machine gets
// -- with each side building its own index, as the paper has them do.
//
// Recall and r2 come from src/test/verify.cpp, the same code the distributed
// run uses. A second copy here would be shorter and would let the two
// definitions drift apart without anything failing.
//
// A separate binary, like ./maketruth and for the same reason: faiss is
// linked here and nowhere else, so ./main stays something the workers can be
// handed without installing anything.
//
//   ./faissbase --data Data/sift/sift --nq 1024 --k 100 --loop 3
//               --threads 8 --csv faiss.csv --nprobe 1 2 4 8 16 32 64 128 256
//
// --nprobe eats values until the next option, so it goes last.

namespace {

struct Options {
    std::string data = "Data/sift/sift";
    std::string csv;
    std::vector<int> nprobes;
    int nlist = 256;
    int k = 100;
    int nq = 1024;
    int loop = 1;
    int threads = 8;
};

void usage(const char* prog) {
    std::cerr
        << "usage: " << prog << " [options]\n"
        << "  --data <prefix>    reads <prefix>_base/_query/_gt.bin"
        << "   (Data/sift/sift)\n"
        << "  --nlist <int>      clusters faiss builds        (256)\n"
        << "  --nprobe <int>...  one search per value          (32)\n"
        << "  --k <int>          neighbours returned          (100)\n"
        << "  --nq <int>         queries to run              (1024)\n"
        << "  --loop <int>       timed passes, averaged         (1)\n"
        << "                     above 1 adds an untimed warm-up\n"
        << "  --threads <int>    what one worker machine gets   (8)\n"
        << "  --csv <path>       append one row per nprobe\n";
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];
        bool hasValue = (i + 1 < argc);

        if (opt == "--data" && hasValue) {
            o.data = argv[++i];
        } else if (opt == "--csv" && hasValue) {
            o.csv = argv[++i];
        } else if (opt == "--nlist" && hasValue) {
            o.nlist = std::atoi(argv[++i]);
        } else if (opt == "--k" && hasValue) {
            o.k = std::atoi(argv[++i]);
        } else if (opt == "--nq" && hasValue) {
            o.nq = std::atoi(argv[++i]);
        } else if (opt == "--loop" && hasValue) {
            o.loop = std::atoi(argv[++i]);
        } else if (opt == "--threads" && hasValue) {
            o.threads = std::atoi(argv[++i]);
        } else if (opt == "--nprobe" && hasValue) {
            // eats values until the next option, the way ./main does
            o.nprobes.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                o.nprobes.push_back(std::atoi(argv[++i]));
            }
        } else {
            usage(argv[0]);
            return false;
        }
    }
    if (o.nprobes.empty()) {
        o.nprobes.push_back(32);
    }
    return true;
}

// Header written only when the file is new, so a sweep can append to one.
void appendRow(const std::string& path, bool* isNew, const Options& o,
               int nprobe, double recall, double r2, double seconds,
               double trainSeconds, double addSeconds) {
    if (path.empty()) {
        return;
    }
    std::FILE* f = std::fopen(path.c_str(), "ab");
    if (f == nullptr) {
        std::cout << "could not write " << path << std::endl;
        return;
    }
    if (*isNew) {
        std::fprintf(f, "data,nlist,nprobe,k,nq,loop,recall,r2,"
                        "seconds,qps,ms_per_query,train_time,add_time\n");
        *isNew = false;
    }
    std::fprintf(f, "%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.4f,%.3f,%.4f,%.3f,%.3f\n",
                 o.data.c_str(), o.nlist, nprobe, o.k, o.nq, o.loop,
                 recall, r2, seconds, o.nq / seconds,
                 1000.0 * seconds / o.nq, trainSeconds, addSeconds);
    std::fclose(f);
}

// The index this run will search, built or read back. Returns null and says
// why on failure. train/add seconds come back 0 for one off disk, the same
// convention ./main uses for a cached index.
faiss::IndexIVFFlat* openIndex(const Options& o, const harmony::Dataset& base,
                               const std::string& path, double* trainSeconds,
                               double* addSeconds) {
    int d = base.getDim();
    *trainSeconds = 0.0;
    *addSeconds = 0.0;

    std::FILE* probe = std::fopen(path.c_str(), "rb");
    if (probe != nullptr) {
        std::fclose(probe);
        std::cout << "loading " << path << std::endl;
        faiss::Index* loaded = faiss::read_index(path.c_str());
        faiss::IndexIVFFlat* ivf = dynamic_cast<faiss::IndexIVFFlat*>(loaded);
        // A file built for other settings would load and then quietly answer
        // a different question, so the shape is checked rather than trusted.
        if (ivf == nullptr || (int)ivf->d != d ||
            (int)ivf->nlist != o.nlist || (long)ivf->ntotal != base.getN()) {
            std::cerr << path << " was built for other settings; delete it"
                      << std::endl;
            delete loaded;
            return nullptr;
        }
        ivf->own_fields = true;    // so the quantizer goes with it
        std::cout << "loaded " << ivf->ntotal << " vectors" << std::endl;
        return ivf;
    }

    std::cout << "building faiss IVF (nlist " << o.nlist << ")" << std::endl;
    faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(d);
    faiss::IndexIVFFlat* ivf = new faiss::IndexIVFFlat(quantizer, d, o.nlist);
    ivf->own_fields = true;

    harmony::Stopwatch watch;
    ivf->train(base.getN(), base.vec(0));
    *trainSeconds = watch.seconds(true);
    ivf->add(base.getN(), base.vec(0));
    *addSeconds = watch.seconds();
    std::cout << "train " << *trainSeconds << " s, add " << *addSeconds << " s"
              << std::endl;

    faiss::write_index(ivf, path.c_str());
    std::cout << "saved " << path << std::endl;
    return ivf;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse(argc, argv, o)) {
        return 1;
    }

#ifdef _OPENMP
    omp_set_num_threads(o.threads);
#endif

    harmony::Dataset base;
    harmony::Dataset query;
    harmony::Groundtruth truth;
    if (!base.load(o.data + "_base.bin") ||
        !query.load(o.data + "_query.bin") ||
        !truth.loadIds(o.data + "_gt.bin")) {
        return 1;
    }
    truth.loadDistances(o.data + "_gtd.bin");

    if (o.nq > query.getN()) {
        o.nq = query.getN();
    }
    std::cout << "faiss baseline: " << base.getN() << " x " << base.getDim()
              << ", nq " << o.nq << ", k " << o.k << ", nlist " << o.nlist
              << ", " << o.threads << " thread(s)" << std::endl;

    std::string path = o.data + "_faissivf_nlist"
                     + std::to_string(o.nlist) + ".index";
    double trainSeconds = 0.0;
    double addSeconds = 0.0;
    faiss::IndexIVFFlat* index = openIndex(o, base, path, &trainSeconds,
                                           &addSeconds);
    if (index == nullptr) {
        return 1;
    }

    bool isNew = true;
    std::vector<float> dist((size_t)o.nq * o.k);
    std::vector<faiss::idx_t> label((size_t)o.nq * o.k);

    for (size_t p = 0; p < o.nprobes.size(); ++p) {
        index->nprobe = o.nprobes[p];

        // One untimed pass when averaging, matching ./main's --loop.
        if (o.loop > 1) {
            index->search(o.nq, query.vec(0), o.k, dist.data(), label.data());
        }
        harmony::Stopwatch watch;
        for (int l = 0; l < o.loop; ++l) {
            index->search(o.nq, query.vec(0), o.k, dist.data(), label.data());
        }
        double seconds = watch.seconds() / o.loop;

        double recallSum = 0.0;
        double r2Sum = 0.0;
        for (int q = 0; q < o.nq; ++q) {
            std::vector<harmony::Candidate> got;
            for (int t = 0; t < o.k; ++t) {
                long id = label[(size_t)q * o.k + t];
                if (id < 0) {
                    break;      // faiss pads a short result with -1
                }
                got.push_back(harmony::Candidate{
                    (int)id, dist[(size_t)q * o.k + t]});
            }
            recallSum = recallSum + truth.recallAt(q, got, o.k);
            r2Sum = r2Sum + truth.r2Of(q, got, o.k);
        }

        std::cout << "nprobe " << o.nprobes[p] << ": " << seconds << " s   "
                  << (o.nq / seconds) << " QPS   recall " << (recallSum / o.nq)
                  << "   r2 " << (r2Sum / o.nq) << std::endl;
        appendRow(o.csv, &isNew, o, o.nprobes[p], recallSum / o.nq,
                  r2Sum / o.nq, seconds, trainSeconds, addSeconds);
    }

    delete index;
    return 0;
}
