cmake_minimum_required(VERSION 3.15)
project(nnue_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT MSVC)
    add_compile_options(-O3 -march=native -ffast-math)
endif()

add_executable(nnue_engine
    main.cpp
    nnue.cpp
    position.cpp
    features.cpp
)

target_include_directories(nnue_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
