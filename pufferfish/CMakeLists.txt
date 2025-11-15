cmake_minimum_required(VERSION 3.14)
project(pufferfish_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(pufferfish_engine
	engine/bitboard.h
	engine/bitboard.cpp
	engine/types.h
	engine/position.h
	engine/position.cpp
	engine/movegen.h
	engine/movegen.cpp
	engine/tt.h
	engine/tt.cpp
	engine/nnue_constants.h
	engine/nnue.h
	engine/nnue.cpp
	engine/features.h
	engine/features.cpp
	engine/nn_interface.h
	engine/simple_nnue.h
	engine/simple_nnue.cpp
	engine/search.h
	engine/search.cpp
)

add_executable(pufferfish main.cpp)
target_link_libraries(pufferfish PRIVATE pufferfish_engine)

# Tests target
add_executable(pufferfish_tests tests.cpp)
target_link_libraries(pufferfish_tests PRIVATE pufferfish_engine)

