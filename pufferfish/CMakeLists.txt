cmake_minimum_required(VERSION 3.14)
project(pufferfish_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# A bare `cmake ..` leaves CMAKE_BUILD_TYPE empty, which means no optimization at
# all. Measured on the bench suite at depth 10, that is 50,653 nps versus 825,687
# for Release -- a 16x difference on identical node counts. An engine is useless
# built that way, so default to Release when the caller has not chosen.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		Debug Release RelWithDebInfo MinSizeRel)
	message(STATUS "No CMAKE_BUILD_TYPE specified; defaulting to Release")
endif()

# Tune for the building machine. Off by default for portable/distributable builds.
option(PUFFERFISH_NATIVE "Optimize for the host CPU (-march=native)" OFF)

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
	engine/nnue_float.h
	engine/nnue_float.cpp
	engine/features.h
	engine/features.cpp
	engine/nn_interface.h
	engine/simple_nnue.h
	engine/simple_nnue.cpp
	engine/search.h
	engine/search.cpp
)

if(PUFFERFISH_NATIVE AND NOT MSVC)
	include(CheckCXXCompilerFlag)
	check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
	if(HAS_MARCH_NATIVE)
		target_compile_options(pufferfish_engine PUBLIC -march=native)
	endif()
endif()

add_executable(pufferfish main.cpp)
target_link_libraries(pufferfish PRIVATE pufferfish_engine)

# Tests target
add_executable(pufferfish_tests tests.cpp)
target_link_libraries(pufferfish_tests PRIVATE pufferfish_engine)

