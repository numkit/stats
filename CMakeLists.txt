cmake_minimum_required(VERSION 3.20)
project(numkit_stats CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NUMKIT_WITH_SIMD "Enable Highway SIMD acceleration" ON)

include(FetchContent)

# Dependency: numkit/value
if(NOT TARGET numkit_value)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../value/CMakeLists.txt")
        message(STATUS "Using local numkit/value from ${CMAKE_CURRENT_SOURCE_DIR}/../value")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../value" _deps/numkit_value)
    else()
        FetchContent_Declare(
            numkit_value
            GIT_REPOSITORY https://github.com/numkit/value.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_value)
    endif()
endif()

# Dependency: numkit/sys
if(NOT TARGET numkit_sys)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../sys/CMakeLists.txt")
        message(STATUS "Using local numkit/sys from ${CMAKE_CURRENT_SOURCE_DIR}/../sys")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../sys" _deps/numkit_sys)
    else()
        FetchContent_Declare(
            numkit_sys
            GIT_REPOSITORY https://github.com/numkit/sys.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_sys)
    endif()
endif()

# Dependency: numkit/ops
if(NOT TARGET numkit_ops)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ops/CMakeLists.txt")
        message(STATUS "Using local numkit/ops from ${CMAKE_CURRENT_SOURCE_DIR}/../ops")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ops" _deps/numkit_ops)
    else()
        FetchContent_Declare(
            numkit_ops
            GIT_REPOSITORY https://github.com/numkit/ops.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_ops)
    endif()
endif()

file(GLOB_RECURSE STATS_COMPUTE_SOURCES
    "src/anova/*.cpp"
    "src/cluster/*.cpp"
    "src/copulas/*.cpp"
    "src/descriptive/*.cpp"
    "src/dim/*.cpp"
    "src/distributions/*.cpp"
    "src/fit/*.cpp"
    "src/lda/*.cpp"
    "src/moments/*.cpp"
    "src/moving/*.cpp"
    "src/mvdist/*.cpp"
    "src/nan_aware/*.cpp"
    "src/qmc/*.cpp"
    "src/regress/*.cpp"
    "src/resample/*.cpp"
    "src/sampling/*.cpp"
    "src/spline/*.cpp"
    "src/test/*.cpp"
)

file(GLOB_RECURSE STATS_PLUGIN_SOURCES
    "src/register/*.cpp"
    "src/module.cpp"
)

if(NOT NUMKIT_WITH_SIMD)
    list(FILTER STATS_COMPUTE_SOURCES EXCLUDE REGEX ".*_highway\\.cpp$")
else()
    list(FILTER STATS_COMPUTE_SOURCES EXCLUDE REGEX ".*_portable\\.cpp$")
endif()

add_library(numkit_stats_core STATIC ${STATS_COMPUTE_SOURCES})
add_library(numkit::stats_core ALIAS numkit_stats_core)
set_target_properties(numkit_stats_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(numkit_stats_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(numkit_stats_core PUBLIC numkit_value numkit_sys numkit_ops)

if(NUMKIT_WITH_SIMD)
    target_compile_definitions(numkit_stats_core PUBLIC NUMKIT_WITH_SIMD=1)
    target_link_libraries(numkit_stats_core PRIVATE hwy hwy_contrib)
endif()

if(MSVC)
    target_compile_options(numkit_stats_core PRIVATE /W4 /MP /bigobj)
else()
    target_compile_options(numkit_stats_core PRIVATE -Wall -Wextra -fPIC)
endif()

# Shared plugin module
add_library(numkit_stats SHARED ${STATS_PLUGIN_SOURCES})
add_library(numkit::stats ALIAS numkit_stats)

target_compile_definitions(numkit_stats PRIVATE NUMKIT_PLUGIN_BUILD=1)

target_include_directories(numkit_stats
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/register
)

target_link_libraries(numkit_stats PUBLIC numkit_stats_core)

if(MSVC)
    target_compile_options(numkit_stats PRIVATE /W4 /MP /bigobj)
else()
    target_compile_options(numkit_stats PRIVATE -Wall -Wextra -fPIC)
endif()
