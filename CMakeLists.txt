cmake_minimum_required(VERSION 3.16)
project(harmony CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(harmony
        main.cpp
        src/node/master_node.cpp
        src/node/worker_node.cpp
)

# so that includes like "src/node/..." in main.cpp resolve from the project root
target_include_directories(harmony PRIVATE ${CMAKE_SOURCE_DIR})

# On Windows/MinGW: bundle the C++ runtime into the .exe so it runs standalone
# (no missing libstdc++-6.dll / libgcc_s / libwinpthread when run outside CLion)
if(MINGW)
    target_link_options(harmony PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# g++ -std=c++20 -I. main.cpp src/node/master_node.cpp src/node/worker_node.cpp -static -static-libgcc -static-libstdc++ -o harmony.exe
#
# macos: ./cmake-build-debug/harmony -id=0