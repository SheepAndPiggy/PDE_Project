#include "mesh.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <fmt/format.h>

namespace mesh {

Mesh2DGenerator::Mesh2DGenerator(int nx,
                                 int ny,
                                 double x_min,
                                 double x_max,
                                 double y_min,
                                 double y_max,
                                 double heated_x_min,
                                 double heated_x_max,
                                 double first_layer_height,
                                 double x_refinement_strength,
                                 double x_refinement_width,
                                 double max_y_growth_ratio)
    : nx_(nx),
      ny_(ny),
      x_min_(x_min),
      x_max_(x_max),
      y_min_(y_min),
      y_max_(y_max),
      heated_x_min_(heated_x_min),
      heated_x_max_(heated_x_max),
      first_layer_height_(first_layer_height),
      x_refinement_strength_(x_refinement_strength),
      x_refinement_width_(x_refinement_width),
      max_y_growth_ratio_(max_y_growth_ratio)
{
}

Mesh2D Mesh2DGenerator::generate() const
{
    validate();

    Mesh2D mesh;
    mesh.nx = nx_;
    mesh.ny = ny_;
    mesh.x_min = x_min_;
    mesh.x_max = x_max_;
    mesh.y_min = y_min_;
    mesh.y_max = y_max_;
    mesh.heated_x_min = heated_x_min_;
    mesh.heated_x_max = heated_x_max_;

    mesh.x_nodes = generate_x_nodes();
    mesh.y_nodes = generate_y_nodes();

    mesh.dx_min = min_spacing(mesh.x_nodes);
    mesh.dx_max = max_spacing(mesh.x_nodes);
    mesh.dy_min = min_spacing(mesh.y_nodes);
    mesh.dy_max = max_spacing(mesh.y_nodes);
    mesh.first_layer_height = mesh.y_nodes[1] - mesh.y_nodes[0];
    mesh.y_growth_ratio = estimate_first_growth_ratio(mesh.y_nodes);

    mesh.x.resize(ny_, nx_);
    mesh.y.resize(ny_, nx_);
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            mesh.x(j, i) = mesh.x_nodes[i];
            mesh.y(j, i) = mesh.y_nodes[j];
        }
    }

    return mesh;
}

double Mesh2DGenerator::first_layer_height_from_y_plus(double rho,
                                                       double mu,
                                                       double friction_velocity,
                                                       double target_y_plus)
{
    if (rho <= 0.0 || mu <= 0.0 || friction_velocity <= 0.0 || target_y_plus <= 0.0) {
        throw std::invalid_argument("rho, mu, friction_velocity, and target_y_plus must be positive");
    }

    return target_y_plus * mu / (rho * friction_velocity);
}

double Mesh2DGenerator::first_layer_height_from_thermal_boundary_layer(double thermal_boundary_layer_thickness,
                                                                       int points_in_boundary_layer)
{
    if (thermal_boundary_layer_thickness <= 0.0) {
        throw std::invalid_argument("thermal_boundary_layer_thickness must be positive");
    }
    if (points_in_boundary_layer < 2) {
        throw std::invalid_argument("points_in_boundary_layer must be at least 2");
    }

    return thermal_boundary_layer_thickness / static_cast<double>(points_in_boundary_layer);
}

void Mesh2DGenerator::validate() const
{
    if (nx_ < 3) {
        throw std::invalid_argument("nx must be at least 3");
    }
    if (ny_ < 3) {
        throw std::invalid_argument("ny must be at least 3");
    }
    if (!(x_min_ < x_max_)) {
        throw std::invalid_argument("x_min must be smaller than x_max");
    }
    if (!(y_min_ < y_max_)) {
        throw std::invalid_argument("y_min must be smaller than y_max");
    }
    if (heated_x_min_ < x_min_ || heated_x_max_ > x_max_ || !(heated_x_min_ < heated_x_max_)) {
        throw std::invalid_argument("heated wall segment must be inside the x range");
    }
    if (first_layer_height_ == 0.0) {
        throw std::invalid_argument("first_layer_height cannot be zero; use a positive value or -1");
    }
    if (x_refinement_strength_ < 0.0) {
        throw std::invalid_argument("x_refinement_strength must be non-negative");
    }
    if (max_y_growth_ratio_ < 1.0) {
        throw std::invalid_argument("max_y_growth_ratio must be at least 1");
    }
}

Eigen::VectorXd Mesh2DGenerator::generate_x_nodes() const
{
    constexpr int sample_count = 4097;

    const double length = x_max_ - x_min_;
    const double heated_length = heated_x_max_ - heated_x_min_;
    const double width = x_refinement_width_ > 0.0
                             ? x_refinement_width_
                             : std::max(0.05 * length, 0.5 * heated_length);

    Eigen::VectorXd sample_x(sample_count);
    Eigen::VectorXd cumulative_weight(sample_count);

    sample_x[0] = x_min_;
    cumulative_weight[0] = 0.0;

    double previous_density = x_density(x_min_, width);
    for (int k = 1; k < sample_count; ++k) {
        const double t = static_cast<double>(k) / static_cast<double>(sample_count - 1);
        const double x = x_min_ + t * length;
        const double density = x_density(x, width);
        const double dx = x - sample_x[k - 1];

        sample_x[k] = x;
        cumulative_weight[k] = cumulative_weight[k - 1] + 0.5 * (previous_density + density) * dx;
        previous_density = density;
    }

    Eigen::VectorXd nodes(nx_);
    nodes[0] = x_min_;
    nodes[nx_ - 1] = x_max_;

    const double total_weight = cumulative_weight[sample_count - 1];
    int lower = 0;

    for (int i = 1; i + 1 < nx_; ++i) {
        const double target_weight = total_weight * static_cast<double>(i) / static_cast<double>(nx_ - 1);

        while (lower + 1 < sample_count && cumulative_weight[lower + 1] < target_weight) {
            ++lower;
        }

        const double w0 = cumulative_weight[lower];
        const double w1 = cumulative_weight[lower + 1];
        const double t = (target_weight - w0) / (w1 - w0);
        nodes[i] = sample_x[lower] + t * (sample_x[lower + 1] - sample_x[lower]);
    }

    return nodes;
}

Eigen::VectorXd Mesh2DGenerator::generate_y_nodes() const
{
    const double height = y_max_ - y_min_;
    const int intervals = ny_ - 1;

    const double first_spacing = first_layer_height_ > 0.0
                                     ? first_layer_height_
                                     : height / (5.0 * static_cast<double>(intervals));

    if (first_spacing * static_cast<double>(intervals) > height) {
        throw std::invalid_argument("first_layer_height is too large for this domain and ny");
    }

    const double ratio = solve_growth_ratio(first_spacing, height, intervals);
    if (ratio > max_y_growth_ratio_ + 1.0e-12) {
        throw std::invalid_argument(fmt::format(
            "required y growth ratio {:.4f} exceeds max_y_growth_ratio {:.4f}; increase ny or first_layer_height",
            ratio,
            max_y_growth_ratio_));
    }

    Eigen::VectorXd nodes(ny_);
    nodes[0] = y_min_;

    double y = y_min_;
    double spacing = first_spacing;
    for (int j = 1; j < ny_; ++j) {
        y += spacing;
        nodes[j] = y;
        spacing *= ratio;
    }

    nodes[ny_ - 1] = y_max_;
    return nodes;
}

double Mesh2DGenerator::x_density(double x, double width) const
{
    const double distance =
        x < heated_x_min_ ? heated_x_min_ - x : (x > heated_x_max_ ? x - heated_x_max_ : 0.0);

    const double q = distance / width;
    return 1.0 + x_refinement_strength_ * std::exp(-0.5 * q * q);
}

double Mesh2DGenerator::solve_growth_ratio(double first_spacing, double total_length, int intervals)
{
    const double uniform_length = first_spacing * static_cast<double>(intervals);
    if (std::abs(uniform_length - total_length) <= 1.0e-12 * total_length) {
        return 1.0;
    }

    double low = 1.0;
    double high = 1.1;
    while (geometric_sum(first_spacing, high, intervals) < total_length) {
        high *= 1.5;
        if (high > 10.0) {
            throw std::invalid_argument("could not find a reasonable y growth ratio");
        }
    }

    for (int iter = 0; iter < 100; ++iter) {
        const double mid = 0.5 * (low + high);
        if (geometric_sum(first_spacing, mid, intervals) < total_length) {
            low = mid;
        } else {
            high = mid;
        }
    }

    return 0.5 * (low + high);
}

double Mesh2DGenerator::geometric_sum(double first_spacing, double ratio, int intervals)
{
    if (std::abs(ratio - 1.0) < 1.0e-12) {
        return first_spacing * static_cast<double>(intervals);
    }

    return first_spacing * (std::pow(ratio, intervals) - 1.0) / (ratio - 1.0);
}

double Mesh2DGenerator::min_spacing(const Eigen::VectorXd& nodes)
{
    double value = nodes[1] - nodes[0];
    for (Eigen::Index i = 1; i + 1 < nodes.size(); ++i) {
        value = std::min(value, nodes[i + 1] - nodes[i]);
    }
    return value;
}

double Mesh2DGenerator::max_spacing(const Eigen::VectorXd& nodes)
{
    double value = nodes[1] - nodes[0];
    for (Eigen::Index i = 1; i + 1 < nodes.size(); ++i) {
        value = std::max(value, nodes[i + 1] - nodes[i]);
    }
    return value;
}

double Mesh2DGenerator::estimate_first_growth_ratio(const Eigen::VectorXd& nodes)
{
    if (nodes.size() < 3) {
        return 1.0;
    }

    const double dy0 = nodes[1] - nodes[0];
    const double dy1 = nodes[2] - nodes[1];
    return dy1 / dy0;
}

} // namespace mesh
