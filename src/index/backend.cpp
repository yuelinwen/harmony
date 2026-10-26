#include "backend.h"

namespace harmony {

namespace {

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

}  // namespace

std::unique_ptr<PruningBackend> makeBackend(const std::string& name) {
    if (name == "harmony") {
        return std::unique_ptr<PruningBackend>(new HarmonyBackend());
    }
    return nullptr;
}

std::string backendNames() {
    return "harmony";
}

}  // namespace harmony
