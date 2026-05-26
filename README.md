# CFD 课程大作业最小项目结构

这个目录刻意保持简单，目标是完成课程大作业，而不是一开始就搭建复杂工程框架。

```text
src/
  CMakeLists.txt       CMake 构建配置，用来编译 C++ Python 扩展模块
  CMakePresets.json    CMake 预设配置，方便统一执行 configure/build
  vcpkg.json           C++ 依赖配置，由 vcpkg 管理
  mesh.hpp             网格数据结构和网格生成器声明
  mesh.cpp             非均匀矩形结构化网格生成实现
  solver.hpp           求解器参数、结果和求解器类声明
  solver.cpp           热对流-扩散求解器实现，以及 pybind11 绑定入口
  stubs/cfd_solver.pyi Python 类型提示文件，帮助编辑器识别 C++ 模块接口
  test.py              Python 调用示例，用来设置参数、调用 C++、画图
  README.md            当前说明文件
```

## 代码分工

- `mesh.cpp`：只负责生成网格，不处理温度场和求解过程，后续 CUDA 扩展时可以直接复用。
- `solver.cpp`：负责温度方程推进、边界条件、结果记录和 Python 绑定。
- `PYBIND11_MODULE`：只暴露 Python 需要调用的函数或类。
- `stubs/cfd_solver.pyi`：手动声明 C++ 模块暴露给 Python 的函数，方便补全和类型检查。
- `test.py`：放参数设置、算例运行、结果可视化和简单后处理。

当前求解器对应报告中的无量纲 Boussinesq 方程组：

```text
div(u) = 0
du/dt + u grad(u) = -grad(p) + 1/Re * Laplacian(u)
dv/dt + u grad(v) = -dp/dy + 1/Re * Laplacian(v) + Ri * T
dT/dt + u grad(T) = 1/Pe * Laplacian(T)
```

时间推进采用 Chorin projection method：先显式预测速度，再解压力泊松方程，最后用压力梯度修正速度，使速度场近似满足不可压连续方程。温度方程使用修正后的速度场推进。

## 编译

```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```

## CUDA 编译

如果要启用 CUDA 后端，使用：

```bash
cmake --preset cuda
cmake --build --preset cuda
```

然后在 Python 中检查：

```python
import sys
sys.path.insert(0, "build/cuda/python")
import cfd_solver

print(cfd_solver.cuda_enabled())
print(cfd_solver.cuda_device_count())
print(cfd_solver.cuda_runtime_version())
```

当前项目把 CUDA 代码放在：

```text
cuda_backend.hpp
cuda_backend.cu
```

后续可以把温度推进、速度预测、压力泊松迭代等计算密集 kernel 逐步迁移到 `.cu` 文件中。`mesh.cpp` 不依赖 CUDA，可以直接复用。

## 运行

```bash
../.venv/bin/python test.py
```

## 设计原则

C++ 负责速度，Python 负责方便。

也就是说，循环多、计算重、数组访问频繁的部分尽量写在 C++ 中；参数配置、批量实验、画图和结果分析放在 Python 中。
