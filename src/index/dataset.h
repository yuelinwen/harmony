#ifndef HARMONY_INDEX_DATASET_H
#define HARMONY_INDEX_DATASET_H

#include <string>
#include <vector>

// Dataset: holds a set of vectors loaded from a flat binary file.
//
// File format (produced by scripts/data.sh):
//     int32 n        number of vectors
//     int32 dim      dimension of each vector
//     float32 data[n * dim]   row-major
//
// Data is stored in one contiguous array, so vector i occupies
// data_[i * dim_ .. i * dim_ + dim_ - 1].

namespace harmony {

class Dataset {
public:
    Dataset() {
        n_ = 0;
        dim_ = 0;
    }

    // Reads the file into memory. Returns false if the file cannot be read.
    bool load(const std::string& path);

    // Pointer to the first element of vector i.
    const float* vec(int i) const {
        return &data_[(size_t)i * dim_];
    }

    // The whole array, to be rewritten in place by a backend's transform.
    // Non-const on purpose: a transform rotates every vector where it lies
    // rather than building a second copy of the dataset.
    float* data() {
        return data_.data();
    }

    int getN() const {
        return n_;
    }

    int getDim() const {
        return dim_;
    }

private:
    std::vector<float> data_;  // n_ * dim_ values, row-major
    int n_;                    // number of vectors
    int dim_;                  // dimension
};

}  // namespace harmony

#endif  // HARMONY_INDEX_DATASET_H
