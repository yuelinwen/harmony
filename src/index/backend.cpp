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
        lambda_.assign((size_t)d, 0.0f);
        for (int i = 0; i < d; ++i) {
            const float* src = &cov[(size_t)(d - 1 - i) * d];
            std::copy(src, src + d, &r_[(size_t)i * d]);
            // Kept in the same order as the rows above: how much of the
            // data's variance the i-th direction accounts for. PCA alone
            // does not need these, but a backend that has to say how much
            // of a distance the first few dimensions already explain does.
            // A near-singular covariance can return small negatives.
            float l = eig[(size_t)(d - 1 - i)];
            lambda_[i] = (l > 0.0f) ? l : 0.0f;
        }
    }

    std::vector<float> r_;
    std::vector<float> lambda_;   // descending, matching the rows of r_
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

// ADSampling (Gao & Long, SIGMOD 2023), as far as this chain can carry it.
//
// It wants a random orthogonal basis and nothing more, so it inherits one
// from rotation and the two differ in exactly one thing: what they are
// willing to conclude from a partial sum.
//
// Harmony drops a candidate once its partial sum is already past the
// threshold. That is exact, and it is also late -- it waits for proof. The
// random projection is what turns the partial sum into an estimator of the
// whole: after d' of D dimensions the full squared distance is around
// S * D / d', so a candidate can go as soon as that estimate is
// confidently above the threshold rather than once it is certainly so.
//
// The paper's test, written for squared distances:
//
//     S / d'  >  (T / D) * (1 + eps / sqrt(d'))^2
//
// which is S > T * m with m = (d'/D) * (1 + eps/sqrt(d'))^2. That m is all
// this backend has to return. eps (--adeps) is the confidence knob: larger
// is more cautious and prunes later.
//
// This is the first backend that is allowed to be wrong, and it will be:
// the estimate can clear a candidate's true neighbour out of the way, so
// recall drops and differing stops being 0. That is the method, not a bug
// in applying it, and sweeping eps is how the trade is measured.
//
// What the chain costs it: the paper tests every 32 dimensions, this can
// only test where one worker hands over to the next, so bDim times. At 1x4
// on Sift the slice is 32 dimensions and the two agree; at 2x2 there are
// two tests; at 4x1 the first and only sum is already the whole distance
// and this degenerates into the exact test. Its headroom here is bounded
// by bDim, which is worth knowing before reading the numbers.
class AdSamplingBackend : public RotationBackend {
public:
    explicit AdSamplingBackend(float eps) {
        eps_ = eps;
    }

    const char* name() const override {
        return "adsampling";
    }

    float thresholdScale(int dimsSoFar, int dimsTotal) const override {
        if (dimsSoFar <= 0 || dimsSoFar >= dimsTotal) {
            return 1.0f;   // nothing summed yet, or all of it: no guessing
        }
        float f = 1.0f + eps_ / std::sqrt((float)dimsSoFar);
        return ((float)dimsSoFar / (float)dimsTotal) * f * f;
    }

private:
    float eps_;
};

// DADE (Deng et al., PVLDB 18), which is ADSampling's idea with the basis
// and the slack both read off the data instead of assumed.
//
// Its transform is PCA, so it inherits one and pca is the right thing to
// read it against -- same basis, same everything, differing only in what
// it concludes from a partial sum.
//
// ADSampling can say the first d of D dimensions carry d/D of the distance
// because a random rotation spreads the variance evenly, which is the
// price it pays for needing no training. PCA does the opposite on purpose:
// it piles the variance into the leading directions, so after d dimensions
// the share already accounted for is not d/D but the eigenvalue ratio
// L_d/L_D, and on Sift1M that reaches half the variance in well under half
// the dimensions. Estimating the full squared distance as S * L_D/L_d and
// rejecting when it clears the threshold gives
//
//     m = (L_d / L_D) * (1 + eps_d)^2
//
// the same shape as ADSampling with the variance share in place of the
// dimension share.
//
// eps_d is the other half. ADSampling has a closed form from a
// concentration bound; DADE has none, and measures instead: how far the
// estimate at d dimensions overshoots the true distance, over a sample of
// pairs, at the quantile the significance level asks for. That is why this
// backend's scales are worked out on the master -- they are learned from
// the data, and a worker has none.
class DadeBackend : public PcaBackend {
public:
    explicit DadeBackend(float sig) {
        sig_ = sig;
    }

    const char* name() const override {
        return "dade";
    }

    void fit(const Dataset& base, const IvfIndex& index) override {
        PcaBackend::fit(base, index);
        estimateDeviation(base, index);
    }

    float thresholdScale(int dimsSoFar, int dimsTotal) const override {
        if (dimsSoFar <= 0 || dimsSoFar >= dimsTotal) {
            return 1.0f;
        }
        if ((int)eps_.size() < dimsSoFar || prefix_[dimsTotal] <= 0.0f) {
            return 1.0f;   // fit found nothing to work with
        }
        float f = 1.0f + eps_[dimsSoFar - 1];
        return (prefix_[dimsSoFar] / prefix_[dimsTotal]) * f * f;
    }

private:
    // Pairs are drawn from inside a cluster, not from the dataset at large.
    // The threshold only ever decides between candidates a query is already
    // scanning, which are the near pairs; the overshoot on a random far
    // pair is a different distribution and would set the slack for a case
    // that never comes up.
    void estimateDeviation(const Dataset& base, const IvfIndex& index) {
        int d = base.getDim();
        prefix_.assign((size_t)d + 1, 0.0f);
        for (int k = 0; k < d; ++k) {
            prefix_[k + 1] = prefix_[k] + lambda_[k];
        }
        eps_.assign((size_t)d, 0.0f);
        if (prefix_[d] <= 0.0f) {
            return;
        }

        std::vector<int> usable;
        for (int c = 0; c < index.getNlist(); ++c) {
            if (index.clusterSize(c) >= 2) {
                usable.push_back(c);
            }
        }
        if (usable.empty()) {
            return;
        }

        const int pairs = 2000;
        std::mt19937 gen(54321);   // its own, for the reason rotation has one
        std::vector<float> over((size_t)d * pairs, 0.0f);
        std::vector<float> diff((size_t)d);
        std::vector<float> proj((size_t)d);

        for (int s = 0; s < pairs; ++s) {
            const std::vector<int>& ids =
                index.clusterIds(usable[gen() % usable.size()]);
            size_t ia = gen() % ids.size();
            size_t ib = (ia + 1 + (gen() % (ids.size() - 1))) % ids.size();
            const float* va = base.vec(ids[ia]);
            const float* vb = base.vec(ids[ib]);
            for (int k = 0; k < d; ++k) {
                diff[k] = va[k] - vb[k];
            }
            // The basis is fitted but not yet applied to the dataset, so
            // the difference is rotated here rather than read off.
            cblas_sgemv(CblasRowMajor, CblasNoTrans, d, d, 1.0f, r_.data(),
                        d, diff.data(), 1, 0.0f, proj.data(), 1);

            float cum = 0.0f;
            for (int k = 0; k < d; ++k) {
                cum = cum + proj[k] * proj[k];
                over[(size_t)k * pairs + s] = cum;   // partial, scaled below
            }
            float full = cum;
            if (full <= 0.0f) {
                for (int k = 0; k < d; ++k) {
                    over[(size_t)k * pairs + s] = 0.0f;
                }
                continue;
            }
            for (int k = 0; k < d; ++k) {
                float est2 = over[(size_t)k * pairs + s] * prefix_[d]
                           / prefix_[k + 1];
                over[(size_t)k * pairs + s] =
                    std::sqrt(est2 / full) - 1.0f;
            }
        }

        int at = (int)((1.0f - sig_) * pairs);
        if (at < 0) {
            at = 0;
        }
        if (at > pairs - 1) {
            at = pairs - 1;
        }
        for (int k = 0; k < d; ++k) {
            float* row = &over[(size_t)k * pairs];
            std::nth_element(row, row + at, row + pairs);
            // Negative is the normal case here, and it must be kept. It
            // says the estimate runs low at this many dimensions, so the
            // threshold has to come down to match -- that is the whole
            // point of measuring the deviation instead of bounding it.
            // Clamping it away was worth 3.5% where keeping it is worth
            // far more, and made --dadesig do nothing at all.
            //
            // The floor only stops the square below going negative.
            eps_[k] = (row[at] > -0.99f) ? row[at] : -0.99f;
        }
    }

    float sig_;
    std::vector<float> prefix_;   // prefix sums of lambda_, size d + 1
    std::vector<float> eps_;      // eps_[k] is the slack at k + 1 dimensions
};

std::unique_ptr<PruningBackend> makeBackend(const std::string& name,
                                            const Config& cfg) {
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
    if (name == "adsampling") {
        return std::unique_ptr<PruningBackend>(new AdSamplingBackend(cfg.adEps));
    }
    if (name == "dade") {
        return std::unique_ptr<PruningBackend>(new DadeBackend(cfg.dadeSig));
    }
    return nullptr;
}

std::string backendNames() {
    return "harmony, rotation, pca, residual_pca, adsampling, dade";
}

}  // namespace harmony
