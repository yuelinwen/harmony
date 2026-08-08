cmake_minimum_required(VERSION 3.16)
project(harmony CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main
        main.cpp
        src/node/master_node.cpp
        src/node/worker_node.cpp
        src/index/dataset.cpp
        src/index/ivf_index.cpp
)

# so that includes like "src/node/..." in main.cpp resolve from the project root
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR})

find_package(MPI REQUIRED)
target_link_libraries(main PRIVATE MPI::MPI_CXX)

# OpenMP parallelises the distance loop inside each worker (paper Section 5).
# Optional: without it every worker just runs single-threaded.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found - workers will be single-threaded")
endif()

# On Windows/MinGW: bundle the C++ runtime into the .exe so it runs standalone
# (no missing libstdc++-6.dll / libgcc_s / libwinpthread when run outside CLion)
if(MINGW)
    target_link_options(main PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# build:
#   mpicxx -std=c++20 -O2 -march=native -I. main.cpp src/node/*.cpp src/index/*.cpp -o main
# run (1 master + 4 workers, from the project root so Data/ resolves):
#   mpirun -n 5 ./main
# dataset script:
#   python3 util/convert_hdf5.py Data/sift-128-euclidean.hdf5 Data/sift
