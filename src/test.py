import sys
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
for build_name in ("default", "release"):
    module_dir = ROOT / "build" / build_name / "python"
    if module_dir.exists():
        sys.path.insert(0, str(module_dir))

import cfd_solver

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False

if __name__ == "__main__":
    generator = cfd_solver.Mesh2DGenerator(
        nx=100,
        ny=20,
        x_min=0.0,
        x_max=20.0,
        y_min=0.0,
        y_max=1.0,
        heated_x_min=4.0,
        heated_x_max=6.0,
        first_layer_height=0.008,
        x_refinement_strength=5.0,
        x_refinement_width=1.0,
        max_y_growth_ratio=1.25,
    )
    mesh = generator.generate()

    print(f"mesh: {mesh.nx} x {mesh.ny}")
    print(f"dx_min={mesh.dx_min:.6g}, dx_max={mesh.dx_max:.6g}")
    print(f"dy_min={mesh.dy_min:.6g}, dy_max={mesh.dy_max:.6g}")
    print(f"first_layer_height={mesh.first_layer_height:.6g}")
    print(f"y_growth_ratio={mesh.y_growth_ratio:.6g}")

    fig, ax = plt.subplots(figsize=(20, 1))

    for i in range(mesh.nx):
        ax.plot(mesh.x[:, i], mesh.y[:, i], color="0.75", linewidth=0.6)
    for j in range(mesh.ny):
        ax.plot(mesh.x[j, :], mesh.y[j, :], color="0.75", linewidth=0.6)

    ax.plot(
        [mesh.heated_x_min, mesh.heated_x_max],
        [mesh.y_min, mesh.y_min],
        color="tab:red",
        linewidth=4,
        solid_capstyle="butt",
        label="壁面热斑",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.legend()

    plt.savefig("../images/local_mesh.png", dpi=1000, bbox_inches="tight")
