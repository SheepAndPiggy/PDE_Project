from __future__ import annotations

from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimSun", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_mesh(mesh, output_path: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 2.4), constrained_layout=True)

    for i in range(mesh.nx):
        ax.plot(mesh.x[:, i], mesh.y[:, i], color="0.75", linewidth=0.55)
    for j in range(mesh.ny):
        ax.plot(mesh.x[j, :], mesh.y[j, :], color="0.75", linewidth=0.55)

    ax.plot(
        [mesh.heated_x_min, mesh.heated_x_max],
        [mesh.y_min, mesh.y_min],
        color="tab:red",
        linewidth=4,
        solid_capstyle="butt",
        label="heated wall",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Non-uniform structured mesh")
    ax.legend()

    _finish_figure(fig, output_path)


def plot_plume_case(mesh, field, output_path: str | Path | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    xc, yc = cell_center_grid(mesh)
    xv, yv = v_face_grid(mesh)
    u_center, v_center = velocity_at_cell_centers(field)

    temperature = np.asarray(field.T)
    tmax = float(np.max(temperature))
    temperature_scaled = temperature / tmax if tmax > 0.0 else temperature

    temp_plot = axes[0, 0].contourf(
        xc,
        yc,
        temperature_scaled,
        levels=np.linspace(0.0, 1.0, 41),
        cmap="turbo",
        norm=colors.PowerNorm(gamma=0.45, vmin=0.0, vmax=1.0),
    )
    _mark_heated_wall(axes[0, 0], mesh, color="white")
    axes[0, 0].set_title("温度相对值 T / Tmax")
    fig.colorbar(temp_plot, ax=axes[0, 0], label="T / Tmax")

    v_abs = max(abs(float(np.min(field.v))), abs(float(np.max(field.v))), 1.0e-12)
    v_plot = axes[0, 1].contourf(
        xv,
        yv,
        field.v,
        levels=np.linspace(-v_abs, v_abs, 41),
        cmap="coolwarm",
    )
    _mark_heated_wall(axes[0, 1], mesh, color="black")
    axes[0, 1].set_title("垂直速度 v")
    fig.colorbar(v_plot, ax=axes[0, 1], label="v")

    speed = np.sqrt(u_center**2 + v_center**2)
    speed_plot = axes[1, 0].contourf(xc, yc, speed, levels=35, cmap="viridis")
    skip = (slice(None, None, 2), slice(None, None, 3))
    axes[1, 0].quiver(
        xc[skip],
        yc[skip],
        u_center[skip],
        v_center[skip],
        color="white",
        scale=12,
        width=0.002,
    )
    _mark_heated_wall(axes[1, 0], mesh, color="tab:red")
    axes[1, 0].set_title("速度矢量")
    fig.colorbar(speed_plot, ax=axes[1, 0], label="|u|")

    div_abs = max(
        abs(float(np.min(field.divergence))),
        abs(float(np.max(field.divergence))),
        1.0e-12,
    )
    div_plot = axes[1, 1].contourf(
        xc,
        yc,
        field.divergence,
        levels=np.linspace(-div_abs, div_abs, 41),
        cmap="coolwarm",
    )
    axes[1, 1].set_title("连续性误差")
    fig.colorbar(div_plot, ax=axes[1, 1], label="div(u)")

    for ax in axes.flat:
        ax.set_xlim(mesh.heated_x_min - 2.0, mesh.heated_x_max + 5.0)
        ax.set_ylim(mesh.y_min, mesh.y_min + 0.75 * (mesh.y_max - mesh.y_min))
        ax.set_aspect("auto")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    _finish_figure(fig, output_path)


def print_plume_diagnostics(mesh, field) -> None:
    xc, yc = cell_center_grid(mesh)
    xv, yv = v_face_grid(mesh)
    temperature = np.maximum(np.asarray(field.T), 0.0)
    total_heat = float(np.sum(temperature))
    if total_heat > 0.0:
        y_center = float(np.sum(yc * temperature) / total_heat)
        x_center = float(np.sum(xc * temperature) / total_heat)
    else:
        x_center = 0.0
        y_center = 0.0

    max_v_index = np.unravel_index(np.argmax(field.v), field.v.shape)
    max_v_y = float(yv[max_v_index])
    max_v_x = float(xv[max_v_index])

    print(f"Tmax={float(np.max(field.T)):.6g}")
    print(f"Tmean={float(np.mean(field.T)):.6g}")
    print(f"max_v={float(np.max(field.v)):.6g} at x={max_v_x:.4g}, y={max_v_y:.4g}")
    print(f"thermal_center=({x_center:.4g}, {y_center:.4g})")
    print(f"max|div(u)|={float(np.max(np.abs(field.divergence))):.6g}")
    if field.divergence.shape[0] > 2 and field.divergence.shape[1] > 2:
        interior_divergence = np.asarray(field.divergence)[1:-1, 1:-1]
        print(f"interior max|div(u)|={float(np.max(np.abs(interior_divergence))):.6g}")


def cell_center_grid(mesh) -> tuple[np.ndarray, np.ndarray]:
    x = 0.5 * (np.asarray(mesh.x_nodes[:-1]) + np.asarray(mesh.x_nodes[1:]))
    y = 0.5 * (np.asarray(mesh.y_nodes[:-1]) + np.asarray(mesh.y_nodes[1:]))
    return np.meshgrid(x, y)


def v_face_grid(mesh) -> tuple[np.ndarray, np.ndarray]:
    x = 0.5 * (np.asarray(mesh.x_nodes[:-1]) + np.asarray(mesh.x_nodes[1:]))
    y = np.asarray(mesh.y_nodes)
    return np.meshgrid(x, y)


def velocity_at_cell_centers(field) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(field.u)
    v = np.asarray(field.v)
    return 0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :])


def _mark_heated_wall(ax, mesh, color: str) -> None:
    ax.plot(
        [mesh.heated_x_min, mesh.heated_x_max],
        [mesh.y_min, mesh.y_min],
        color=color,
        linewidth=4,
        solid_capstyle="butt",
    )


def _finish_figure(fig, output_path: str | Path | None) -> None:
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
    else:
        plt.show()
