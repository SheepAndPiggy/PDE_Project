#include "mesh.hpp"
#include "solver.hpp"

#ifdef PDE_ENABLE_CUDA
#include "cuda_backend.hpp"
#endif

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

std::string dependency_summary()
{
    const nlohmann::json summary = {
        {"eigen", "enabled"},
        {"fmt", "enabled"},
        {"nlohmann_json", "enabled"},
        {"pybind11", "enabled"},
#ifdef PDE_ENABLE_CUDA
        {"cuda", "enabled"},
#else
        {"cuda", "disabled"},
#endif
    };

    return fmt::format("{}", summary.dump());
}

} // namespace

PYBIND11_MODULE(cfd_solver, m)
{
    m.doc() = "C++ CFD/PDE numerical kernels exposed to Python";

    m.def("version", [] { return "0.5.0"; });
    m.def("dependency_summary", &dependency_summary);
    m.def("cuda_enabled", [] {
#ifdef PDE_ENABLE_CUDA
        return true;
#else
        return false;
#endif
    });
    m.def("cuda_device_count", [] {
#ifdef PDE_ENABLE_CUDA
        return cuda_backend::device_count();
#else
        return 0;
#endif
    });
    m.def("cuda_runtime_version", [] {
#ifdef PDE_ENABLE_CUDA
        return cuda_backend::runtime_version();
#else
        return 0;
#endif
    });

    py::class_<mesh::Mesh2D>(m, "Mesh2D")
        .def_readonly("nx", &mesh::Mesh2D::nx)
        .def_readonly("ny", &mesh::Mesh2D::ny)
        .def_readonly("x_min", &mesh::Mesh2D::x_min)
        .def_readonly("x_max", &mesh::Mesh2D::x_max)
        .def_readonly("y_min", &mesh::Mesh2D::y_min)
        .def_readonly("y_max", &mesh::Mesh2D::y_max)
        .def_readonly("heated_x_min", &mesh::Mesh2D::heated_x_min)
        .def_readonly("heated_x_max", &mesh::Mesh2D::heated_x_max)
        .def_readonly("dx_min", &mesh::Mesh2D::dx_min)
        .def_readonly("dx_max", &mesh::Mesh2D::dx_max)
        .def_readonly("dy_min", &mesh::Mesh2D::dy_min)
        .def_readonly("dy_max", &mesh::Mesh2D::dy_max)
        .def_readonly("first_layer_height", &mesh::Mesh2D::first_layer_height)
        .def_readonly("y_growth_ratio", &mesh::Mesh2D::y_growth_ratio)
        .def_readonly("x_nodes", &mesh::Mesh2D::x_nodes)
        .def_readonly("y_nodes", &mesh::Mesh2D::y_nodes)
        .def_readonly("x", &mesh::Mesh2D::x)
        .def_readonly("y", &mesh::Mesh2D::y);

    py::class_<mesh::Mesh2DGenerator>(m, "Mesh2DGenerator")
        .def(py::init<int, int, double, double, double, double, double, double, double, double, double, double>(),
             py::arg("nx"),
             py::arg("ny"),
             py::arg("x_min"),
             py::arg("x_max"),
             py::arg("y_min"),
             py::arg("y_max"),
             py::arg("heated_x_min"),
             py::arg("heated_x_max"),
             py::arg("first_layer_height") = -1.0,
             py::arg("x_refinement_strength") = 4.0,
             py::arg("x_refinement_width") = -1.0,
             py::arg("max_y_growth_ratio") = 1.25)
        .def("generate", &mesh::Mesh2DGenerator::generate)
        .def_static("first_layer_height_from_y_plus",
                    &mesh::Mesh2DGenerator::first_layer_height_from_y_plus,
                    py::arg("rho"),
                    py::arg("mu"),
                    py::arg("friction_velocity"),
                    py::arg("target_y_plus") = 1.0)
        .def_static("first_layer_height_from_thermal_boundary_layer",
                    &mesh::Mesh2DGenerator::first_layer_height_from_thermal_boundary_layer,
                    py::arg("thermal_boundary_layer_thickness"),
                    py::arg("points_in_boundary_layer") = 10);

    py::class_<solver::SolverOption>(m, "SolverOption")
        .def(py::init<>())
        .def_readwrite("reynolds", &solver::SolverOption::reynolds)
        .def_readwrite("richardson", &solver::SolverOption::richardson)
        .def_readwrite("peclet", &solver::SolverOption::peclet)
        .def_readwrite("u_in", &solver::SolverOption::u_in)
        .def_readwrite("v_in", &solver::SolverOption::v_in)
        .def_readwrite("T_in", &solver::SolverOption::T_in)
        .def_readwrite("wall_heat_flux", &solver::SolverOption::wall_heat_flux)
        .def_readwrite("time_sep", &solver::SolverOption::time_sep)
        .def_readwrite("step", &solver::SolverOption::step)
        .def_readwrite("save_sep", &solver::SolverOption::save_sep)
        .def_readwrite("poisson_max_interval", &solver::SolverOption::poisson_max_interval)
        .def_readwrite("poisson_tolerance", &solver::SolverOption::poisson_tolerance)
        .def_readwrite("poisson_omega", &solver::SolverOption::poisson_omega)
        .def_readwrite("use_cuda", &solver::SolverOption::use_cuda);

    py::class_<solver::FlowField>(m, "FlowField")
        .def_readonly("step", &solver::FlowField::step)
        .def_readonly("time", &solver::FlowField::time)
        .def_readonly("u", &solver::FlowField::u)
        .def_readonly("v", &solver::FlowField::v)
        .def_readonly("p", &solver::FlowField::p)
        .def_readonly("T", &solver::FlowField::T)
        .def_readonly("divergence", &solver::FlowField::divergence);

    py::class_<solver::Solver>(m, "Solver")
        .def(py::init<const mesh::Mesh2D&, const solver::SolverOption&>(),
             py::arg("mesh"),
             py::arg("option"))
        .def("run", &solver::Solver::run);
}
