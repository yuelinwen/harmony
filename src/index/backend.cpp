#include "backend.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <mkl.h>
#include <algorithm>

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


// The directions the data actually varies along, most first.
//
// This is the one that should pay. Pruning stops when the running sum passes
// the threshold, so it stops sooner when the early dimensions carry more of
// the distance -- and the early dimensions carry more of the distance
// exactly when they are the directions of greatest variance. The baseline's
// order is whatever the dataset was written in, which has no reason to be
// that.
//
// Still orthogonal, so still exact: the eigenvectors of a symmetric matrix
// are orthonormal, and the transform is a change of basis, not a projection
// onto fewer dimensions. Nothing is discarded. Dropping the trailing
// components would be a different method with a different error, and would
// cost the exactness this whole line of backends is built on.
//
// The mean is subtracted to find the directions and then not applied.
// Subtracting the same vector from a query and a candidate leaves the
// distance between them unchanged, so centring would make no difference to
// anything downstream -- it belongs in fit, where it decides the directions,
// and nowhere else.
class PcaBackend : public PruningBackend {
public:
    const char* name() const override {
        return "pca";
    }

    void fit(const Dataset& base, const IvfIndex& index) override {
        (void)index;
        fitFrom(base, nullptr, index);
    }

    void apply(Dataset& base, Dataset& query, IvfIndex& index) override {
        rotateAll(base, query, index, r_, base.getDim());
    }

protected:
    // centreOn: when given, each vector is centred on the centroid of its own
    // cluster rather than on the global mean -- which is what residual PCA
    // is. Null means the global mean.
    void fitFrom(const Dataset& base, const std::vector<int>* owner,
                 const IvfIndex& index) {
        int d = base.getDim();
        int n = base.getN();

        // One pass for the mean, a second to accumulate the covariance. The
        // second is chunked: syrk wants the centred vectors, and centring a
        // million of them in place would mean either a second copy of the
        // dataset or modifying the one about to be searched.
        std::vector<double> mean((size_t)d, 0.0);
        if (owner == nullptr) {
            for (int i = 0; i < n; ++i) {
                const float* v = base.vec(i);
                for (int j = 0; j < d; ++j) {
                    mean[j] = mean[j] + v[j];
                }
            }
            for (int j = 0; j < d; ++j) {
                mean[j] = mean[j] / (n > 0 ? n : 1);
            }
        }

        const int kChunk = 65536;
        std::vector<float> centred((size_t)kChunk * d);
        std::vector<float> cov((size_t)d * d, 0.0f);
        for (int start = 0; start < n; start += kChunk) {
            int m = (start + kChunk <= n) ? kChunk : (n - start);
            for (int i = 0; i < m; ++i) {
                const float* v = base.vec(start + i);
                const float* c = nullptr;
                if (owner != nullptr) {
                    c = index.centroid((*owner)[start + i]);
                }
                float* dst = &centred[(size_t)i * d];
                for (int j = 0; j < d; ++j) {
                    dst[j] = v[j] - (c ? c[j] : (float)mean[j]);
                }
            }
            // cov += centred^T * centred, upper triangle only.
            cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        d, m, 1.0f, centred.data(), d, 1.0f, cov.data(), d);
        }

        // syrk filled one triangle; mirror it into the other. Only half a
        // matrix is layout-dependent -- a row-major upper triangle is a
        // column-major lower one -- and handing ssyev the half it does not
        // read leaves it decomposing a near-diagonal matrix, whose
        // eigenvectors come back as a permuted identity. That fails
        // silently: the run still works and still returns the right answers,
        // it just does not rotate anything.
        for (int i = 0; i < d; ++i) {
            for (int j = i + 1; j < d; ++j) {
                cov[(size_t)j * d + i] = cov[(size_t)i * d + j];
            }
        }

        // Full and symmetric now, so the layout does not matter going in;
        // coming out, column-major eigenvectors are contiguous rows in
        // memory, which is the shape rotateInPlace wants.
        std::vector<float> eig((size_t)d);
        int info = LAPACKE_ssyev(LAPACK_COL_MAJOR, 'V', 'U', d,
                                 cov.data(), d, eig.data());
        if (info != 0) {
            std::fprintf(stderr, "ssyev failed (%d); leaving the basis alone\n",
                         info);
            r_.assign((size_t)d * d, 0.0f);
            for (int i = 0; i < d; ++i) {
                r_[(size_t)i * d + i] = 1.0f;
            }
            return;
        }

        // ssyev returns them ascending; the whole point is to take the
        // largest first.
        r_.assign((size_t)d * d, 0.0f);
        for (int i = 0; i < d; ++i) {
            const float* src = &cov[(size_t)(d - 1 - i) * d];
            std::copy(src, src + d, &r_[(size_t)i * d]);
        }
    }

    std::vector<float> r_;
};


// PCA on what is left after the cluster centroid is subtracted.
//
// The reasoning is that global PCA finds the directions the whole dataset
// varies along, and most of that variance is between clusters -- which the
// IVF has already dealt with, because a query only ever scans the handful
// of clusters nearest it. Inside one of those clusters the between-cluster
// spread is a constant offset and contributes nothing to telling two
// candidates apart. What does is the residual, so those are the directions
// worth putting first.
//
// One basis for all of them, not one per cluster. Per-cluster bases would
// concentrate harder, but then a query would have to be rotated once per
// cluster it probes, and the dimension slice would mean a different thing
// in every cluster -- a much larger change for a gain that has not been
// measured yet.
class ResidualPcaBackend : public PcaBackend {
public:
    const char* name() const override {
        return "residual_pca";
    }

    void fit(const Dataset& base, const IvfIndex& index) override {
        // Which cluster each vector landed in, inverted from the lists.
        std::vector<int> owner((size_t)base.getN(), 0);
        for (int c = 0; c < index.getNlist(); ++c) {
            const std::vector<int>& ids = index.clusterIds(c);
            for (size_t i = 0; i < ids.size(); ++i) {
                owner[(size_t)ids[i]] = c;
            }
        }
        fitFrom(base, &owner, index);
    }
};

}  // namespace

std::unique_ptr<PruningBackend> makeBackend(const std::string& name) {
    if (name == "harmony") {
        return std::unique_ptr<PruningBackend>(new HarmonyBackend());
    }
    if (name == "rotation") {
        return std::unique_ptr<PruningBackend>(new RotationBackend());
    }
    if (name == "pca") {
        return std::unique_ptr<PruningBackend>(new PcaBackend());
    }
    if (name == "residual_pca") {
        return std::unique_ptr<PruningBackend>(new ResidualPcaBackend());
    }
    return nullptr;
}

std::string backendNames() {
    return "harmony, rotation, pca, residual_pca";
}

}  // namespace harmony
