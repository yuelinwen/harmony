#include <cstring>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    int id = -1;

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "-id=", 4) == 0) {
            id = std::atoi(argv[i] + 4);
        } else if (std::strncmp(argv[i], "--id=", 5) == 0) {
            id = std::atoi(argv[i] + 5);
        }
    }

    if (id < 0) {
        std::cerr << "usage: " << argv[0] << " -id=0 (master) | -id=1,2,... (worker)" << std::endl;
        return 1;
    }

    if (id == 0) {
        std::cout << "master node started" << std::endl;
    } else {
        std::cout << "worker node started, id=" << id << std::endl;
    }
    return 0;
}
