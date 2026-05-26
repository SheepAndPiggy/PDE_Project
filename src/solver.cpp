#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace solver {

Solver::Solver(const mesh::Mesh2D& mesh_field, const SolverOption& option)
    : mesh_field_(mesh_field), option_(option)
{
    if (option_.reynolds <= 0.0F) {
        throw std::invalid_argument("reynolds must be positive");
    }
    if (option_.richardson < 0.0F) {
        throw std::invalid_argument("richardson must be non-negative");
    }
    if (option_.peclet <= 0.0F) {
        throw std::invalid_argument("peclet must be positive");
    }
    if (option_.time_sep <= 0.0) {
        throw std::invalid_argument("time_sep must be positive");
    }
    if (option_.step <= 0) {
        throw std::invalid_argument("step must be positive");
    }
    if (option_.poisson_max_interval <= 0) {
        throw std::invalid_argument("poisson_max_interval must be positive");
    }
    if (option_.poisson_tolerance <= 0.0) {
        throw std::invalid_argument("poisson_tolerance must be positive");
    }
    if (option_.poisson_omega <= 0.0 || option_.poisson_omega >= 2.0) {
        throw std::invalid_argument("poisson_omega must be in (0, 2)");
    }
}

FlowField Solver::run() const
{
    return option_.use_cuda ? run_cuda() : run_cpu();
}

FlowField Solver::run_cpu() const
{
    FlowField current = initialize_field();
    apply_velocity_boundary_conditions(current);
    apply_temperature_boundary_conditions(current);
    apply_pressure_boundary_conditions(current);
    current.divergence = compute_divergence(current);

    FlowField next = current;

    for (int n = 1; n <= option_.step; ++n) {
        next = current;
        next.step = n;
        next.time = static_cast<double>(n) * option_.time_sep;

        predict_velocity(current, next);
        apply_velocity_boundary_conditions(next);
        next.divergence = compute_divergence(next);

        solve_pressure_poisson(next);
        apply_pressure_boundary_conditions(next);

        correct_velocity(next);
        apply_velocity_boundary_conditions(next);

        advance_temperature(current, next);
        apply_temperature_boundary_conditions(next);

        next.divergence = compute_divergence(next);
        current = next;
    }

    return current;
}

FlowField Solver::run_cuda() const
{
    throw std::runtime_error("CUDA solver backend is reserved but not implemented yet");
}

FlowField Solver::initialize_field() const
{
    const int nx_c = nx_cells();
    const int ny_c = ny_cells();
    const Eigen::VectorXd y_c = y_cell_centers();

    FlowField field;
    field.step = 0;
    field.time = 0.0;
    field.u = Eigen::MatrixXd::Zero(ny_c, nx_c + 1);
    field.v = Eigen::MatrixXd::Zero(ny_c + 1, nx_c);
    field.p = Eigen::MatrixXd::Zero(ny_c, nx_c);
    field.T = Eigen::MatrixXd::Constant(ny_c, nx_c, option_.T_in);
    field.divergence = Eigen::MatrixXd::Zero(ny_c, nx_c);

    // 用近似层流的抛物线速度分布初始化整个速度场
    const double height = mesh_field_.y_max - mesh_field_.y_min;
    for (int j = 0; j < ny_c; ++j) {
        const double eta = (y_c[j] - mesh_field_.y_min) / height;
        const double profile = 6.0 * option_.u_in * eta * (1.0 - eta);
        for (int i = 0; i < nx_c + 1; ++i) {
            field.u(j, i) = profile;
        }
    }

    return field;
}

void Solver::apply_velocity_boundary_conditions(FlowField& field) const
{
    const int nx_c = nx_cells();
    const int ny_c = ny_cells();
    const Eigen::VectorXd y_c = y_cell_centers();
    const double height = mesh_field_.y_max - mesh_field_.y_min;

    // 充分发展层流入口条件
    for (int j = 0; j < ny_c; ++j) {
        const double eta = (y_c[j] - mesh_field_.y_min) / height;
        field.u(j, 0) = 6.0 * option_.u_in * eta * (1.0 - eta);
        field.u(j, nx_c) = field.u(j, nx_c - 1);
    }

    // 上下壁无穿透。切向无滑移不直接覆盖第一层 u，而是在 u 动量方程的
    // 近壁扩散项中用 ghost-cell 反射实现：u_ghost = -u_inner。
    for (int i = 0; i < nx_c; ++i) {
        field.v(0, i) = 0.0;
        field.v(ny_c, i) = 0.0;
    }

    // 施加进出口速度边界条件
    for (int j = 1; j < ny_c; ++j) {
        field.v(j, 0) = option_.v_in;
        // 出口v沿x方向导数为0
        field.v(j, nx_c - 1) = field.v(j, nx_c - 2);
    }
}

void Solver::apply_temperature_boundary_conditions(FlowField& field) const
{
    const int nx_c = nx_cells();
    const int ny_c = ny_cells();

    for (int j = 0; j < ny_c; ++j) {
        field.T(j, 0) = option_.T_in;
        field.T(j, nx_c - 1) = field.T(j, nx_c - 2);
    }
}

void Solver::apply_pressure_boundary_conditions(FlowField& field) const
{
    const int nx_c = nx_cells();
    const int ny_c = ny_cells();

    for (int j = 0; j < ny_c; ++j) {
        field.p(j, 0) = field.p(j, 1);
        field.p(j, nx_c - 1) = 0.0;
    }

    for (int i = 0; i < nx_c; ++i) {
        field.p(0, i) = field.p(1, i);
        field.p(ny_c - 1, i) = field.p(ny_c - 2, i);
    }
}

void Solver::predict_velocity(const FlowField& current, FlowField& next) const
{
    const int nx_c = nx_cells();
    const int ny_c = ny_cells();
    const double viscosity = 1.0 / static_cast<double>(option_.reynolds);
    const Eigen::VectorXd x_u = mesh_field_.x_nodes;
    const Eigen::VectorXd y_u = y_cell_centers();
    const Eigen::VectorXd x_v = x_cell_centers();
    const Eigen::VectorXd y_v = mesh_field_.y_nodes;

    next.u = current.u;
    next.v = current.v;

    Eigen::MatrixXd adv_u_on_u = current.u;
    Eigen::MatrixXd adv_v_on_u = Eigen::MatrixXd::Zero(ny_c, nx_c + 1);
    for (int j = 0; j < ny_c; ++j) {
        for (int i = 1; i < nx_c; ++i) {
            adv_v_on_u(j, i) = v_velocity_at_u_face(current, i, j);
        }
    }

    for (int j = 0; j < ny_c; ++j) {
        for (int i = 1; i < nx_c; ++i) {
            const double conv_u = convection_u_no_slip(current.u, adv_u_on_u, adv_v_on_u, x_u, y_u, i, j);
            next.u(j, i) = current.u(j, i) +
                           option_.time_sep *
                               (-conv_u + viscosity * laplacian_u_no_slip(current.u, x_u, y_u, i, j));
        }
    }

    Eigen::MatrixXd adv_u_on_v = Eigen::MatrixXd::Zero(ny_c + 1, nx_c);
    Eigen::MatrixXd adv_v_on_v = current.v;
    for (int j = 1; j < ny_c; ++j) {
        for (int i = 1; i + 1 < nx_c; ++i) {
            adv_u_on_v(j, i) = u_velocity_at_v_face(current, i, j);
        }
    }

    for (int j = 1; j < ny_c; ++j) {
        for (int i = 1; i + 1 < nx_c; ++i) {
            const double conv_v = convection(current.v, adv_u_on_v, adv_v_on_v, x_v, y_v, i, j);
            const double buoyancy = 0.5 * static_cast<double>(option_.richardson) *
                                    (current.T(j - 1, i) + current.T(j, i));
            next.v(j, i) = current.v(j, i) +
                           option_.time_sep *
                               (-conv_v + viscosity * laplacian(current.v, x_v, y_v, i, j) + buoyancy);
        }
    }
}

void Solver::solve_pressure_poisson(FlowField& field) const
{
    const int nx_c = nx_cells();
    const int ny_c = ny_cells();
    const Eigen::VectorXd x_c = x_cell_centers();
    const Eigen::VectorXd y_c = y_cell_centers();

    for (int iter = 0; iter < option_.poisson_max_interval; ++iter) {
        double max_update = 0.0;
        apply_pressure_boundary_conditions(field);

        for (int j = 1; j + 1 < ny_c; ++j) {
            for (int i = 1; i + 1 < nx_c; ++i) {
                const double dx_w = x_c[i] - x_c[i - 1];
                const double dx_e = x_c[i + 1] - x_c[i];
                const double dy_s = y_c[j] - y_c[j - 1];
                const double dy_n = y_c[j + 1] - y_c[j];

                const double aw = 2.0 / (dx_w * (dx_w + dx_e));
                const double ae = 2.0 / (dx_e * (dx_w + dx_e));
                const double as = 2.0 / (dy_s * (dy_s + dy_n));
                const double an = 2.0 / (dy_n * (dy_s + dy_n));
                const double ap = aw + ae + as + an;

                const double rhs = field.divergence(j, i) / option_.time_sep;
                const double raw = (aw * field.p(j, i - 1) + ae * field.p(j, i + 1) +
                                    as * field.p(j - 1, i) + an * field.p(j + 1, i) - rhs) /
                                   ap;
                const double updated =
                    (1.0 - option_.poisson_omega) * field.p(j, i) + option_.poisson_omega * raw;

                max_update = std::max(max_update, std::abs(updated - field.p(j, i)));
                field.p(j, i) = updated;
            }
        }

        if (max_update < option_.poisson_tolerance) {
            break;
        }
    }

    apply_pressure_boundary_conditions(field);
}

void Solver::correct_velocity(FlowField& field) const
{
    const int nx_c = nx_cells();
    const int ny_c = ny_cells();

    for (int j = 0; j < ny_c; ++j) {
        for (int i = 1; i < nx_c; ++i) {
            field.u(j, i) -= option_.time_sep * pressure_gradient_x(field.p, i, j);
        }
    }

    for (int j = 1; j < ny_c; ++j) {
        for (int i = 0; i < nx_c; ++i) {
            field.v(j, i) -= option_.time_sep * pressure_gradient_y(field.p, i, j);
        }
    }
}

void Solver::advance_temperature(const FlowField& current, FlowField& next) const
{
    const int nx_c = nx_cells();
    const int ny_c = ny_cells();
    const double thermal_diffusion = 1.0 / static_cast<double>(option_.peclet);
    const Eigen::VectorXd x_c = x_cell_centers();
    const Eigen::VectorXd y_c = y_cell_centers();

    Eigen::MatrixXd adv_u = Eigen::MatrixXd::Zero(ny_c, nx_c);
    Eigen::MatrixXd adv_v = Eigen::MatrixXd::Zero(ny_c, nx_c);
    for (int j = 0; j < ny_c; ++j) {
        for (int i = 0; i < nx_c; ++i) {
            adv_u(j, i) = u_velocity_at_cell(next, i, j);
            adv_v(j, i) = v_velocity_at_cell(next, i, j);
        }
    }

    next.T = current.T;
    for (int j = 0; j < ny_c; ++j) {
        for (int i = 1; i + 1 < nx_c; ++i) {
            const double conv_T = convection_with_horizontal_walls(current.T, adv_u, adv_v, x_c, y_c, i, j);
            next.T(j, i) = current.T(j, i) +
                           option_.time_sep *
                               (-conv_T + thermal_diffusion *
                                              laplacian_temperature_with_walls(current.T, x_c, y_c, i, j));
        }
    }
}

Eigen::MatrixXd Solver::compute_divergence(const FlowField& field) const
{
    const int nx_c = nx_cells();
    const int ny_c = ny_cells();
    Eigen::MatrixXd div = Eigen::MatrixXd::Zero(ny_c, nx_c);

    for (int j = 0; j < ny_c; ++j) {
        const double dy = mesh_field_.y_nodes[j + 1] - mesh_field_.y_nodes[j];
        for (int i = 0; i < nx_c; ++i) {
            const double dx = mesh_field_.x_nodes[i + 1] - mesh_field_.x_nodes[i];
            div(j, i) = (field.u(j, i + 1) - field.u(j, i)) / dx +
                        (field.v(j + 1, i) - field.v(j, i)) / dy;
        }
    }

    return div;
}

int Solver::nx_cells() const
{
    return mesh_field_.nx - 1;
}

int Solver::ny_cells() const
{
    return mesh_field_.ny - 1;
}

Eigen::VectorXd Solver::x_cell_centers() const
{
    Eigen::VectorXd centers(nx_cells());
    for (int i = 0; i < nx_cells(); ++i) {
        centers[i] = 0.5 * (mesh_field_.x_nodes[i] + mesh_field_.x_nodes[i + 1]);
    }
    return centers;
}

Eigen::VectorXd Solver::y_cell_centers() const
{
    Eigen::VectorXd centers(ny_cells());
    for (int j = 0; j < ny_cells(); ++j) {
        centers[j] = 0.5 * (mesh_field_.y_nodes[j] + mesh_field_.y_nodes[j + 1]);
    }
    return centers;
}

double Solver::pressure_gradient_x(const Eigen::MatrixXd& p, int i_face, int j) const
{
    const Eigen::VectorXd x_c = x_cell_centers();
    return (p(j, i_face) - p(j, i_face - 1)) / (x_c[i_face] - x_c[i_face - 1]);
}

double Solver::pressure_gradient_y(const Eigen::MatrixXd& p, int i, int j_face) const
{
    const Eigen::VectorXd y_c = y_cell_centers();
    return (p(j_face, i) - p(j_face - 1, i)) / (y_c[j_face] - y_c[j_face - 1]);
}

double Solver::u_velocity_at_cell(const FlowField& field, int i, int j) const
{
    return 0.5 * (field.u(j, i) + field.u(j, i + 1));
}

double Solver::v_velocity_at_cell(const FlowField& field, int i, int j) const
{
    return 0.5 * (field.v(j, i) + field.v(j + 1, i));
}

double Solver::v_velocity_at_u_face(const FlowField& field, int i_face, int j) const
{
    const int i_left = std::max(0, i_face - 1);
    const int i_right = std::min(nx_cells() - 1, i_face);
    return 0.25 * (field.v(j, i_left) + field.v(j + 1, i_left) +
                   field.v(j, i_right) + field.v(j + 1, i_right));
}

double Solver::u_velocity_at_v_face(const FlowField& field, int i, int j_face) const
{
    const int j_down = std::max(0, j_face - 1);
    const int j_up = std::min(ny_cells() - 1, j_face);
    return 0.25 * (field.u(j_down, i) + field.u(j_down, i + 1) +
                   field.u(j_up, i) + field.u(j_up, i + 1));
}

double Solver::ddx_centered(const Eigen::MatrixXd& q, const Eigen::VectorXd& x, int i, int j) const
{
    return (q(j, i + 1) - q(j, i - 1)) / (x[i + 1] - x[i - 1]);
}

double Solver::ddy_centered(const Eigen::MatrixXd& q, const Eigen::VectorXd& y, int i, int j) const
{
    return (q(j + 1, i) - q(j - 1, i)) / (y[j + 1] - y[j - 1]);
}

double Solver::laplacian(const Eigen::MatrixXd& q, const Eigen::VectorXd& x, const Eigen::VectorXd& y, int i, int j) const
{
    const double dx_w = x[i] - x[i - 1];
    const double dx_e = x[i + 1] - x[i];
    const double dy_s = y[j] - y[j - 1];
    const double dy_n = y[j + 1] - y[j];

    const double d2x = 2.0 * ((q(j, i + 1) - q(j, i)) / dx_e - (q(j, i) - q(j, i - 1)) / dx_w) /
                       (dx_w + dx_e);
    const double d2y = 2.0 * ((q(j + 1, i) - q(j, i)) / dy_n - (q(j, i) - q(j - 1, i)) / dy_s) /
                       (dy_s + dy_n);

    return d2x + d2y;
}

double Solver::laplacian_u_no_slip(const Eigen::MatrixXd& u,
                                   const Eigen::VectorXd& x,
                                   const Eigen::VectorXd& y,
                                   int i,
                                   int j) const
{
    const double dx_w = x[i] - x[i - 1];
    const double dx_e = x[i + 1] - x[i];
    const double d2x = 2.0 * ((u(j, i + 1) - u(j, i)) / dx_e - (u(j, i) - u(j, i - 1)) / dx_w) /
                       (dx_w + dx_e);

    double dy_s = 0.0;
    double dy_n = 0.0;
    double u_s = 0.0;
    double u_n = 0.0;

    if (j == 0) {
        dy_s = 2.0 * (y[0] - mesh_field_.y_min);
        dy_n = y[1] - y[0];
        u_s = -u(j, i);
        u_n = u(j + 1, i);
    } else if (j + 1 == u.rows()) {
        dy_s = y[j] - y[j - 1];
        dy_n = 2.0 * (mesh_field_.y_max - y[j]);
        u_s = u(j - 1, i);
        u_n = -u(j, i);
    } else {
        dy_s = y[j] - y[j - 1];
        dy_n = y[j + 1] - y[j];
        u_s = u(j - 1, i);
        u_n = u(j + 1, i);
    }

    const double d2y = 2.0 * ((u_n - u(j, i)) / dy_n - (u(j, i) - u_s) / dy_s) /
                       (dy_s + dy_n);
    return d2x + d2y;
}

double Solver::laplacian_temperature_with_walls(const Eigen::MatrixXd& T,
                                                const Eigen::VectorXd& x,
                                                const Eigen::VectorXd& y,
                                                int i,
                                                int j) const
{
    const double dx_w = x[i] - x[i - 1];
    const double dx_e = x[i + 1] - x[i];
    const double d2x = 2.0 * ((T(j, i + 1) - T(j, i)) / dx_e - (T(j, i) - T(j, i - 1)) / dx_w) /
                       (dx_w + dx_e);

    double dy_s = 0.0;
    double dy_n = 0.0;
    double T_s = 0.0;
    double T_n = 0.0;

    if (j == 0) {
        const double wall_distance = y[0] - mesh_field_.y_min;
        const bool heated = x[i] >= mesh_field_.heated_x_min && x[i] <= mesh_field_.heated_x_max;
        dy_s = 2.0 * wall_distance;
        dy_n = y[1] - y[0];
        T_s = heated ? T(j, i) + 2.0 * wall_distance * option_.wall_heat_flux : T(j, i);
        T_n = T(j + 1, i);
    } else if (j + 1 == T.rows()) {
        dy_s = y[j] - y[j - 1];
        dy_n = 2.0 * (mesh_field_.y_max - y[j]);
        T_s = T(j - 1, i);
        T_n = T(j, i);
    } else {
        dy_s = y[j] - y[j - 1];
        dy_n = y[j + 1] - y[j];
        T_s = T(j - 1, i);
        T_n = T(j + 1, i);
    }

    const double d2y = 2.0 * ((T_n - T(j, i)) / dy_n - (T(j, i) - T_s) / dy_s) /
                       (dy_s + dy_n);
    return d2x + d2y;
}

double Solver::convection_with_horizontal_walls(const Eigen::MatrixXd& q,
                                                const Eigen::MatrixXd& adv_u,
                                                const Eigen::MatrixXd& adv_v,
                                                const Eigen::VectorXd& x,
                                                const Eigen::VectorXd& y,
                                                int i,
                                                int j) const
{
    const double u_e = 0.5 * (adv_u(j, i) + adv_u(j, i + 1));
    const double u_w = 0.5 * (adv_u(j, i - 1) + adv_u(j, i));
    const double q_e = muscl_x_face(q, i, j, u_e);
    const double q_w = muscl_x_face(q, i - 1, j, u_w);
    const double dx = 0.5 * (x[i + 1] - x[i - 1]);

    double v_n = 0.0;
    double v_s = 0.0;
    double q_n = 0.0;
    double q_s = 0.0;
    double dy = 0.0;

    if (j == 0) {
        v_s = 0.0;
        q_s = q(j, i);
        v_n = 0.5 * (adv_v(j, i) + adv_v(j + 1, i));
        q_n = muscl_y_face(q, i, j, v_n);
        dy = 0.5 * (y[j + 1] - y[j]) + (y[j] - mesh_field_.y_min);
    } else if (j + 1 == q.rows()) {
        v_s = 0.5 * (adv_v(j - 1, i) + adv_v(j, i));
        q_s = muscl_y_face(q, i, j - 1, v_s);
        v_n = 0.0;
        q_n = q(j, i);
        dy = 0.5 * (y[j] - y[j - 1]) + (mesh_field_.y_max - y[j]);
    } else {
        v_n = 0.5 * (adv_v(j, i) + adv_v(j + 1, i));
        v_s = 0.5 * (adv_v(j - 1, i) + adv_v(j, i));
        q_n = muscl_y_face(q, i, j, v_n);
        q_s = muscl_y_face(q, i, j - 1, v_s);
        dy = 0.5 * (y[j + 1] - y[j - 1]);
    }

    return (u_e * q_e - u_w * q_w) / dx + (v_n * q_n - v_s * q_s) / dy;
}

double Solver::convection_u_no_slip(const Eigen::MatrixXd& q,
                                    const Eigen::MatrixXd& adv_u,
                                    const Eigen::MatrixXd& adv_v,
                                    const Eigen::VectorXd& x,
                                    const Eigen::VectorXd& y,
                                    int i,
                                    int j) const
{
    const double u_e = 0.5 * (adv_u(j, i) + adv_u(j, i + 1));
    const double u_w = 0.5 * (adv_u(j, i - 1) + adv_u(j, i));
    const double q_e = muscl_x_face(q, i, j, u_e);
    const double q_w = muscl_x_face(q, i - 1, j, u_w);
    const double dx = 0.5 * (x[i + 1] - x[i - 1]);

    double v_n = 0.0;
    double v_s = 0.0;
    double q_n = 0.0;
    double q_s = 0.0;
    double dy = 0.0;

    if (j == 0) {
        v_s = 0.0;
        q_s = 0.0;
        v_n = 0.5 * (adv_v(j, i) + adv_v(j + 1, i));
        q_n = muscl_y_face(q, i, j, v_n);
        dy = 0.5 * (y[j + 1] - y[j]) + (y[j] - mesh_field_.y_min);
    } else if (j + 1 == q.rows()) {
        v_s = 0.5 * (adv_v(j - 1, i) + adv_v(j, i));
        q_s = muscl_y_face(q, i, j - 1, v_s);
        v_n = 0.0;
        q_n = 0.0;
        dy = 0.5 * (y[j] - y[j - 1]) + (mesh_field_.y_max - y[j]);
    } else {
        v_n = 0.5 * (adv_v(j, i) + adv_v(j + 1, i));
        v_s = 0.5 * (adv_v(j - 1, i) + adv_v(j, i));
        q_n = muscl_y_face(q, i, j, v_n);
        q_s = muscl_y_face(q, i, j - 1, v_s);
        dy = 0.5 * (y[j + 1] - y[j - 1]);
    }

    return (u_e * q_e - u_w * q_w) / dx + (v_n * q_n - v_s * q_s) / dy;
}

double Solver::convection(const Eigen::MatrixXd& q,
                          const Eigen::MatrixXd& adv_u,
                          const Eigen::MatrixXd& adv_v,
                          const Eigen::VectorXd& x,
                          const Eigen::VectorXd& y,
                          int i,
                          int j) const
{
    const double u_e = 0.5 * (adv_u(j, i) + adv_u(j, i + 1));
    const double u_w = 0.5 * (adv_u(j, i - 1) + adv_u(j, i));
    const double v_n = 0.5 * (adv_v(j, i) + adv_v(j + 1, i));
    const double v_s = 0.5 * (adv_v(j - 1, i) + adv_v(j, i));

    const double q_e = muscl_x_face(q, i, j, u_e);
    const double q_w = muscl_x_face(q, i - 1, j, u_w);
    const double q_n = muscl_y_face(q, i, j, v_n);
    const double q_s = muscl_y_face(q, i, j - 1, v_s);

    const double dx = 0.5 * (x[i + 1] - x[i - 1]);
    const double dy = 0.5 * (y[j + 1] - y[j - 1]);

    return (u_e * q_e - u_w * q_w) / dx + (v_n * q_n - v_s * q_s) / dy;
}

double Solver::minmod(double a, double b) const
{
    if (a * b <= 0.0) {
        return 0.0;
    }
    return std::copysign(std::min(std::abs(a), std::abs(b)), a);
}

double Solver::muscl_x_face(const Eigen::MatrixXd& q, int face_i, int j, double face_velocity) const
{
    if (face_velocity >= 0.0) {
        const int c = face_i;
        if (c <= 0 || c + 1 >= q.cols()) {
            return q(j, std::clamp(c, 0, static_cast<int>(q.cols()) - 1));
        }
        const double slope = minmod(q(j, c) - q(j, c - 1), q(j, c + 1) - q(j, c));
        return q(j, c) + 0.5 * slope;
    }

    const int c = face_i + 1;
    if (c <= 0 || c + 1 >= q.cols()) {
        return q(j, std::clamp(c, 0, static_cast<int>(q.cols()) - 1));
    }
    const double slope = minmod(q(j, c) - q(j, c - 1), q(j, c + 1) - q(j, c));
    return q(j, c) - 0.5 * slope;
}

double Solver::muscl_y_face(const Eigen::MatrixXd& q, int i, int face_j, double face_velocity) const
{
    if (face_velocity >= 0.0) {
        const int c = face_j;
        if (c <= 0 || c + 1 >= q.rows()) {
            return q(std::clamp(c, 0, static_cast<int>(q.rows()) - 1), i);
        }
        const double slope = minmod(q(c, i) - q(c - 1, i), q(c + 1, i) - q(c, i));
        return q(c, i) + 0.5 * slope;
    }

    const int c = face_j + 1;
    if (c <= 0 || c + 1 >= q.rows()) {
        return q(std::clamp(c, 0, static_cast<int>(q.rows()) - 1), i);
    }
    const double slope = minmod(q(c, i) - q(c - 1, i), q(c + 1, i) - q(c, i));
    return q(c, i) - 0.5 * slope;
}

} // namespace solver
