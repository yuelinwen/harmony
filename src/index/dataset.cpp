#include "dataset.h"

#include <cstdio>
#include <iostream>

namespace harmony {

bool Dataset::load(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        std::cerr << "cannot open file: " << path << std::endl;
        return false;
    }

    // Header: two int32 values, n and dim.
    int header[2];
    if (std::fread(header, sizeof(int), 2, file) != 2) {
        std::cerr << "cannot read header: " << path << std::endl;
        std::fclose(file);
        return false;
    }
    n_ = header[0];
    dim_ = header[1];

    if (n_ <= 0 || dim_ <= 0) {
        std::cerr << "bad header: n=" << n_ << " dim=" << dim_ << std::endl;
        std::fclose(file);
        return false;
    }

    // Body: n * dim float values, read in one go.
    size_t count = (size_t)n_ * dim_;
    data_.resize(count);
    size_t got = std::fread(data_.data(), sizeof(float), count, file);
    std::fclose(file);

    if (got != count) {
        std::cerr << "short read: expected " << count << " floats, got " << got << std::endl;
        return false;
    }

    return true;
}

}  // namespace harmony
