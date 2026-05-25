#include "mesh.hpp"
#include "solver.hpp"

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

