#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include "mesh.hpp"

namespace solver {
// 保存时间层数据，包含整个流场物理量在指定时间层的网格数据。
struct FlowField{        
    int step = 0;  // 当前时间步
    double time = 0.0;  // 当前时间

    Eigen::MatrixXd u;          // MAC: vertical faces, shape (ny-1, nx)
    Eigen::MatrixXd v;          // MAC: horizontal faces, shape (ny, nx-1)
    Eigen::MatrixXd p;          // MAC: cell centers, shape (ny-1, nx-1)
    Eigen::MatrixXd T;          // MAC: cell centers, shape (ny-1, nx-1)
    Eigen::MatrixXd divergence; // MAC: cell centers, shape (ny-1, nx-1)
};

// 求解器参数
struct SolverOption {
    // 无量纲常数
    float reynolds = 100.0;
    float richardson = 0.05;
    float peclet = 800.0;

    // 边界条件，入口给定速度温度；壁面除加热区域绝热；加热区给定热通量；出口充分发展
    double u_in = 1.0;  // 入口无量纲化速度
    double v_in = 0;  // 默认v = 0
    double T_in = 0.0;  // 入口无量纲化温度
    double wall_heat_flux = 8.0;  // 壁面加热区域无量纲热流量

    // 求解过程相关参数
    double time_sep = 1e-3;  // 求解时间步间隔
    int step = 1000;  // 求解步长
    int save_sep = 100;  // 保存数据时间间隔
    int poisson_max_interval = 50;  // 泊松方程迭代最大次数
    double poisson_tolerance = 1e-4;  // 泊松方程允许误差
    double poisson_omega = 1.6;  // SOR松弛因子
    bool use_cuda = false;  // 预留CUDA后端开关
};

// 求解器类
class Solver {
public:
    Solver(
        const mesh::Mesh2D& mesh_field,
        const SolverOption& option
    );

    FlowField run() const;  // 运行求解计算
private:
    mesh::Mesh2D mesh_field_;
    SolverOption option_;

    FlowField run_cpu() const;
    FlowField run_cuda() const;  // CUDA版本接口预留，具体实现可在cuda_backend.cu中完成

    // 流场初始化
    FlowField initialize_field() const;

    // 应用流场边界条件
    void apply_velocity_boundary_conditions(FlowField& field) const;
    void apply_temperature_boundary_conditions(FlowField& field) const;
    void apply_pressure_boundary_conditions(FlowField& field) const;

    // 算法实现部分
    void predict_velocity(const FlowField& current, FlowField& next) const;  // 投影法中预测速度部分
    void solve_pressure_poisson(FlowField& field) const;  // 投影法中求解泊松方程部分
    void correct_velocity(FlowField& field) const;  // 投影法中最终速度修正部分
    void advance_temperature(const FlowField& current, FlowField& next) const;  // 单独解温度方程的推进部分
    Eigen::MatrixXd compute_divergence(const FlowField& field) const;

    int nx_cells() const;
    int ny_cells() const;
    Eigen::VectorXd x_cell_centers() const;
    Eigen::VectorXd y_cell_centers() const;

    double pressure_gradient_x(const Eigen::MatrixXd& p, int i_face, int j) const;
    double pressure_gradient_y(const Eigen::MatrixXd& p, int i, int j_face) const;
    double u_velocity_at_cell(const FlowField& field, int i, int j) const;
    double v_velocity_at_cell(const FlowField& field, int i, int j) const;
    double v_velocity_at_u_face(const FlowField& field, int i_face, int j) const;
    double u_velocity_at_v_face(const FlowField& field, int i, int j_face) const;

    double ddx_centered(const Eigen::MatrixXd& q, const Eigen::VectorXd& x, int i, int j) const;
    double ddy_centered(const Eigen::MatrixXd& q, const Eigen::VectorXd& y, int i, int j) const;
    double laplacian(const Eigen::MatrixXd& q, const Eigen::VectorXd& x, const Eigen::VectorXd& y, int i, int j) const;
    double laplacian_u_no_slip(const Eigen::MatrixXd& u, const Eigen::VectorXd& x, const Eigen::VectorXd& y, int i, int j) const;
    double laplacian_temperature_with_walls(const Eigen::MatrixXd& T, const Eigen::VectorXd& x, const Eigen::VectorXd& y, int i, int j) const;
    double convection_with_horizontal_walls(
        const Eigen::MatrixXd& q,
        const Eigen::MatrixXd& adv_u,
        const Eigen::MatrixXd& adv_v,
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& y,
        int i,
        int j
    ) const;
    double convection_u_no_slip(
        const Eigen::MatrixXd& q,
        const Eigen::MatrixXd& adv_u,
        const Eigen::MatrixXd& adv_v,
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& y,
        int i,
        int j
    ) const;
    double convection(
        const Eigen::MatrixXd& q,
        const Eigen::MatrixXd& adv_u,
        const Eigen::MatrixXd& adv_v,
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& y,
        int i,
        int j
    ) const;

    double minmod(double a, double b) const;
    double muscl_x_face(const Eigen::MatrixXd& q, int face_i, int j, double face_velocity) const;
    double muscl_y_face(const Eigen::MatrixXd& q, int i, int face_j, double face_velocity) const;
};
}  // namespace solver
