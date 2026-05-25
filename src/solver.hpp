#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include "mesh.hpp"

namespace solver {
// 保存时间层数据，包含整个流场物理量在指定时间层的网格数据。
struct FlowField{        
    double time_step;  // 当前时间步
    double time_real;  // 当前时间

    Eigen::MatrixXd u;
    Eigen::MatrixXd v;
    Eigen::MatrixXd p;
    Eigen::MatrixXd T;
};

// 求解器参数
struct SolverOption {
    // 无量纲常数
    float reynolds = 100.0;
    float richardson = 0.05;
    float peclet = 800.0;

    // 边界条件，入口给定速度温度；壁面除加热区域绝热；加热区给定热通量；出口充分发展
    double u_in;  // 入口无量纲化速度
    double v_in = 0;  // 默认v = 0
    double T_in;  // 入口无量纲化温度
    double wall_heat_flux;  // 壁面加热区域无量纲热流量

    // 求解过程相关参数
    double time_sep = 1e-3;  // 求解时间步间隔
    int step = 1000;  // 求解步长
    int save_sep = 100;  // 保存数据时间间隔
    int possion_max_interval = 50;  // 泊松方程迭代最大次数
    double possion_tolerance = 1e-4;  // 泊松方程允许误差
};

// 求解器类
class Solver {
public:
    Solver(
        const mesh::Mesh2D& mesh_field,
        const SolverOption& option
    );

    void run();  // 运行求解计算
private:
    const mesh::Mesh2D& _mesh_field;
    const SolverOption& _option;

    // 计算中心差分
    inline double centralDiffFirst(const FlowField& field, int i, int j);

    // 计算拉普拉斯导数
    inline double laplaceDiff(const FlowField& field, int i, int j);

    // 计算迎风格式的对流项
    inline double convectionTerm(const FlowField& field, int i, int j);
};
}  // namespace solver