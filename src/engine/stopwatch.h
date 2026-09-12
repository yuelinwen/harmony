#ifndef HARMONY_ENGINE_STOPWATCH_H
#define HARMONY_ENGINE_STOPWATCH_H

#include <chrono>

namespace harmony {

// Elapsed wall time. steady_clock, so a clock adjustment cannot make an
// interval come out negative.
//
// Used to split a worker's run into the four things it can be doing -- idle,
// receiving, computing, sending -- which is the breakdown behind the paper's
// Fig. 9.
class Stopwatch {
public:
    Stopwatch() : start_(std::chrono::steady_clock::now()) {}

    void reset() { start_ = std::chrono::steady_clock::now(); }

    // Seconds since construction or the last reset. Pass true to restart in
    // the same call, which is what accumulating consecutive phases wants.
    double seconds(bool restart = false) {
        std::chrono::steady_clock::time_point now =
            std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(now - start_).count();
        if (restart) {
            start_ = now;
        }
        return s;
    }

private:
    std::chrono::steady_clock::time_point start_;
};

}  // namespace harmony

#endif  // HARMONY_ENGINE_STOPWATCH_H
