cmake_minimum_required(VERSION 3.25)

project(
  PDEProject
  VERSION 0.1.0
  DESCRIPTION "C++ numerical kernels exposed to Python for PDE/CFD experiments"
  LANGUAGES CXX
)

option(PDE_ENABLE_NATIVE_OPT "Enable native CPU optimization flags for local experiments" OFF)
option(PDE_ENABLE_CUDA "Enable CUDA backend compilation" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

pybind11_add_module(cfd_solver MODULE solver.cpp mesh.cpp)

target_include_directories(cfd_solver PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(
  cfd_solver
  PRIVATE
    Eigen3::Eigen
    fmt::fmt
    nlohmann_json::nlohmann_json
)

target_compile_features(cfd_solver PRIVATE cxx_std_20)

target_compile_options(
  cfd_solver
  PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

if(PDE_ENABLE_NATIVE_OPT AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(cfd_solver PRIVATE -march=native)
endif()

if(PDE_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  target_sources(cfd_solver PRIVATE cuda_backend.cu)
  target_link_libraries(cfd_solver PRIVATE CUDA::cudart)
  target_compile_definitions(cfd_solver PRIVATE PDE_ENABLE_CUDA=1)
  target_compile_features(cfd_solver PRIVATE cuda_std_17)

  set_target_properties(
    cfd_solver
    PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

set_target_properties(
  cfd_solver
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

enable_testing()

add_test(
  NAME python_import_cfd_solver
  COMMAND
    "${Python3_EXECUTABLE}"
    -c
    "import sys; sys.path.insert(0, r'${CMAKE_BINARY_DIR}/python'); import cfd_solver; print(cfd_solver.version())"
)
