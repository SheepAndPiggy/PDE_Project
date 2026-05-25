#pragma once

#include <Eigen/Dense>

namespace mesh {

struct Mesh2D {
    int nx = 0;
    int ny = 0;

    double x_min = 0.0;
    double x_max = 0.0;
    double y_min = 0.0;
    double y_max = 0.0;

    double heated_x_min = 0.0;
    double heated_x_max = 0.0;

    double dx_min = 0.0;
    double dx_max = 0.0;
    double dy_min = 0.0;
    double dy_max = 0.0;

    double first_layer_height = 0.0;
    double y_growth_ratio = 1.0;

    Eigen::VectorXd x_nodes;
    Eigen::VectorXd y_nodes;
    Eigen::MatrixXd x;
    Eigen::MatrixXd y;
};

class Mesh2DGenerator {
public:
    Mesh2DGenerator(int nx,
                    int ny,
                    double x_min,
                    double x_max,
                    double y_min,
                    double y_max,
                    double heated_x_min,
                    double heated_x_max,
                    double first_layer_height = -1.0,
                    double x_refinement_strength = 4.0,
                    double x_refinement_width = -1.0,
                    double max_y_growth_ratio = 1.25);

    Mesh2D generate() const;

    static double first_layer_height_from_y_plus(double rho,
                                                 double mu,
                                                 double friction_velocity,
                                                 double target_y_plus = 1.0);

    static double first_layer_height_from_thermal_boundary_layer(double thermal_boundary_layer_thickness,
                                                                 int points_in_boundary_layer = 10);

private:
    int nx_;
    int ny_;

    double x_min_;
    double x_max_;
    double y_min_;
    double y_max_;

    double heated_x_min_;
    double heated_x_max_;
    double first_layer_height_;
    double x_refinement_strength_;
    double x_refinement_width_;
    double max_y_growth_ratio_;

    void validate() const;
    Eigen::VectorXd generate_x_nodes() const;
    Eigen::VectorXd generate_y_nodes() const;
    double x_density(double x, double width) const;

    static double solve_growth_ratio(double first_spacing, double total_length, int intervals);
    static double geometric_sum(double first_spacing, double ratio, int intervals);
    static double min_spacing(const Eigen::VectorXd& nodes);
    static double max_spacing(const Eigen::VectorXd& nodes);
    static double estimate_first_growth_ratio(const Eigen::VectorXd& nodes);
};

} // namespace mesh
