#include "backend.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <mkl.h>

namespace harmony {

namespace {

// ---- shared by every backend that changes the basis -----------------------

// Rewrites n vectors of d dimensions as v' = R v, in place.
//
// Done as a matrix multiply rather than a loop of dot products: it is the
// same arithmetic either way, and one n x d by d x d gemm is the shape MKL
// is built for. In chunks because gemm cannot write into its own input, and
// a whole copy of Sift1M would be half a gigabyte of scratch to avoid a
// second pass over 32 MB.
void rotateInPlace(float* data, int n, int d, const std::vector<float>& r) {
    const int kChunk = 65536;
    std::vector<float> out((size_t)kChunk * d);
    for (int start = 0; start < n; start += kChunk) {
        int m = (start + kChunk <= n) ? kChunk : (n - start);
        // out = data * R^T, which is row i of data read as a column vector
        // and multiplied by R: out[i][j] = sum_k data[i][k] * R[j][k].
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, d, d, 1.0f,
                    data + (size_t)start * d, d,
                    r.data(), d,
                    0.0f, out.data(), d);
        std::copy(out.begin(), out.begin() + (size_t)m * d,
                  data + (size_t)start * d);
    }
}

// All three, which is the part that is easy to get wrong. A query reaches a
// cluster by its distance to the centroid, so a basis change that rewrites
// the vectors and not the centroids sends every query to the wrong clusters
// and still returns results that look plausible.
void rotateAll(Dataset& base, Dataset& query, IvfIndex& index,
               const std::vector<float>& r, int d) {
    rotateInPlace(base.data(), base.getN(), d, r);
    rotateInPlace(query.data(), query.getN(), d, r);
    rotateInPlace(index.centroids(), index.getNlist(), d, r);
}

// ---- backends -------------------------------------------------------------

// The order the dimensions already arrive in, which is the paper's. Both
// halves do nothing, so this is also the proof that the interface costs
// nothing: a run with it has to match a run from before it existed, to the
// digit.
class HarmonyBackend : public PruningBackend {
public:
    const char* name() const override {
        return "harmony";
    }
};

// A random orthogonal basis. It does not concentrate anything -- it spreads
// the variance evenly instead -- so on its own it is not expected to prune
// better than the baseline. It is here as the control: an orthogonal matrix
// preserves every distance exactly, so this backend has to return the same
// answers as the baseline, and `differing` staying at 0 is what says the
// transform reached all three of the base vectors, the queries and the
// centroids. A non-zero differing here is a bug in the plumbing, not a
// property of the method.
//
// It is also the floor the two PCA backends are read against: whatever PCA
// gains has to be over this, not over the unrotated order, or the gain could
// just be "the original axes were a bad basis".
class RotationBackend : public PruningBackend {
public:
    const char* name() const override {
        return "rotation";
    }

    void fit(const Dataset& base, const IvfIndex& index) override {
        (void)index;
        int d = base.getDim();
        // Fixed seed, and its own generator: kmeans seeds std::rand, and
        // sharing that would make the clustering depend on whether a
        // rotation was asked for.
        std::mt19937 gen(12345);
        std::normal_distribution<float> normal(0.0f, 1.0f);

        r_.assign((size_t)d * d, 0.0f);
        for (size_t i = 0; i < r_.size(); ++i) {
            r_[i] = normal(gen);
        }

        // Modified Gram-Schmidt. A matrix of gaussians is orthogonal only
        // after this; without it the transform still mixes the dimensions
        // but no longer preserves distance, which would show up as a
        // non-zero differing rather than as anything obviously wrong.
        for (int i = 0; i < d; ++i) {
            float* row = &r_[(size_t)i * d];
            for (int j = 0; j < i; ++j) {
                const float* prev = &r_[(size_t)j * d];
                float dot = cblas_sdot(d, row, 1, prev, 1);
                cblas_saxpy(d, -dot, prev, 1, row, 1);
            }
            float norm = cblas_snrm2(d, row, 1);
            if (norm > 1e-12f) {
                cblas_sscal(d, 1.0f / norm, row, 1);
            }
        }
    }

    void apply(Dataset& base, Dataset& query, IvfIndex& index) override {
        rotateAll(base, query, index, r_, base.getDim());
    }

private:
    std::vector<float> r_;   // d x d, row-major, orthogonal
};

}  // namespace

std::unique_ptr<PruningBackend> makeBackend(const std::string& name) {
    if (name == "harmony") {
        return std::unique_ptr<PruningBackend>(new HarmonyBackend());
    }
    if (name == "rotation") {
        return std::unique_ptr<PruningBackend>(new RotationBackend());
    }
    return nullptr;
}

std::string backendNames() {
    return "harmony, rotation";
}

}  // namespace harmony
