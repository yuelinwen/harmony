#include "verify.h"

#include <algorithm>
#include <cstdio>
#include <iostream>

#include "../engine/stopwatch.h"

namespace harmony {

bool Groundtruth::loadIds(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        std::cerr << "cannot open file: " << path << std::endl;
        return false;
    }

    int header[2];
    if (std::fread(header, sizeof(int), 2, file) != 2) {
        std::cerr << "cannot read header: " << path << std::endl;
        std::fclose(file);
        return false;
    }
    count_ = header[0];
    dim_ = header[1];

    ids_.resize((size_t)count_ * dim_);
    size_t got = std::fread(ids_.data(), sizeof(int), ids_.size(), file);
    std::fclose(file);

    if (got != ids_.size()) {
        std::cerr << "short read: " << path << std::endl;
        return false;
    }

    std::cout << "gt:    n=" << count_ << " dim=" << dim_ << std::endl;
    return true;
}

// Same file layout as the ids, and the same shape, so it is checked against
// those rather than trusted.
bool Groundtruth::loadDistances(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        std::cout << "no " << path << ", so no r2"
                  << "   (scripts/data.sh writes it)" << std::endl;
        return false;
    }

    int header[2];
    if (std::fread(header, sizeof(int), 2, file) != 2) {
        std::cerr << "cannot read header: " << path << std::endl;
        std::fclose(file);
        return false;
    }

    if (header[0] != count_ || header[1] != dim_) {
        std::cerr << path << " is " << header[0] << "x" << header[1]
                  << " but the ids are " << count_ << "x" << dim_
                  << "; ignoring it" << std::endl;
        std::fclose(file);
        return false;
    }

    dist_.resize((size_t)count_ * dim_);
    size_t got = std::fread(dist_.data(), sizeof(float), dist_.size(), file);
    std::fclose(file);

    if (got != dist_.size()) {
        std::cerr << "short read: " << path << std::endl;
        dist_.clear();
        return false;
    }
    return true;
}

double Groundtruth::recallAt(int queryId, const std::vector<Candidate>& got,
                             int k) const {
    if (queryId >= count_ || k <= 0) {
        return 0.0;
    }
    int want = (k < dim_) ? k : dim_;   // the file may hold fewer than k
    const int* truth = &ids_[(size_t)queryId * dim_];

    int hits = 0;
    for (int i = 0; i < want; ++i) {
        for (int j = 0; j < (int)got.size(); ++j) {
            if (got[j].id == truth[i]) {
                hits = hits + 1;
                break;
            }
        }
    }
    return (double)hits / (double)want;
}

double Groundtruth::r2Of(int queryId, const std::vector<Candidate>& got,
                         int k) const {
    if (dist_.empty() || queryId >= count_ || k <= 0) {
        return 0.0;
    }
    int want = (k < dim_) ? k : dim_;
    const float* truth = &dist_[(size_t)queryId * dim_];

    double mine = 0.0;
    double real = 0.0;
    for (int i = 0; i < want && i < (int)got.size(); ++i) {
        mine = mine + got[i].dist;
        real = real + truth[i];
    }
    return (real > 0.0) ? (mine / real - 1.0) : 0.0;
}

double referencePass(IvfIndex& index, const Dataset& base, const Dataset& query,
                     const std::vector<std::vector<int>>& probes, int k,
                     int loop, std::vector<std::vector<Candidate>>& out) {
    int nq = (int)probes.size();
    out.assign(nq, std::vector<Candidate>());

    int passes = (loop > 1) ? (loop + 1) : 1;
    double seconds = 0.0;

    for (int pass = 0; pass < passes; ++pass) {
        Stopwatch watch;
        #pragma omp parallel for schedule(dynamic)
        for (int q = 0; q < nq; ++q) {
            out[q] = index.search(base, query.vec(q), probes[q], k);
        }
        double took = watch.seconds();
        if (!(passes > 1 && pass == 0)) {
            seconds = seconds + took;
        }
    }

    return (loop > 1) ? (seconds / loop) : seconds;
}

void compare(const std::vector<Candidate>& got,
             const std::vector<Candidate>& reference, int k, Agreement* into) {
    // Both sides can be shorter than k: a heap returns only as many as it was
    // offered, and the clusters one query probes can hold fewer than k vectors
    // between them -- `--nprobe 1 --k 5000` on sift1M is enough to do it.
    // Reading to k regardless segfaulted.
    int m = k;
    if ((int)got.size() < m) {
        m = (int)got.size();
    }
    if ((int)reference.size() < m) {
        m = (int)reference.size();
    }

    std::vector<int> a;
    std::vector<int> b;
    std::vector<float> da;
    std::vector<float> db;
    for (int i = 0; i < m; ++i) {
        a.push_back(got[i].id);
        b.push_back(reference[i].id);
        da.push_back(got[i].dist);
        db.push_back(reference[i].dist);
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::sort(da.begin(), da.end());
    std::sort(db.begin(), db.end());

    if (a == b) {
        return;
    }
    if (da == db) {
        into->ties = into->ties + 1;
    } else {
        into->differing = into->differing + 1;
    }
}

}  // namespace harmony
