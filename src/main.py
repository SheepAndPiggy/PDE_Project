import sys
from pathlib import Path

from plot import plot_mesh, plot_plume_case, print_plume_diagnostics

ROOT = Path(__file__).resolve().parent
for build_name in reversed(("cuda", "cuda-release", "default", "release")):
    module_dir = ROOT / "build" / build_name / "python"
    if module_dir.exists():
        sys.path.insert(0, str(module_dir))

import cfd_solver


def build_plume_case():
    generator = cfd_solver.Mesh2DGenerator(
        nx=82,
        ny=42,
        x_min=0.0,
        x_max=20.0,
        y_min=0.0,
        y_max=1.0,
        heated_x_min=7.0,
        heated_x_max=9.0,
        first_layer_height=0.006,
        x_refinement_strength=5.0,
        x_refinement_width=1.0,
        max_y_growth_ratio=1.25,
    )
    mesh = generator.generate()

    option = cfd_solver.SolverOption()
    option.reynolds = 120.0
    option.richardson = 1.5
    option.peclet = 250.0

    option.u_in = 0.35
    option.v_in = 0.0
    option.T_in = 0.0
    option.wall_heat_flux = 18.0

    option.time_sep = 8.0e-5
    option.step = 1000
    option.save_sep = 100

    option.poisson_max_interval = 120
    option.poisson_tolerance = 1.0e-4
    option.poisson_omega = 1.6
    option.use_cuda = False

    return mesh, option


if __name__ == "__main__":
    mesh, option = build_plume_case()

    print(f"cfd_solver version: {cfd_solver.version()}")
    print(f"cuda enabled module: {cfd_solver.cuda_enabled()}")
    print(f"mesh: {mesh.nx} x {mesh.ny}")
    print(f"dx_min={mesh.dx_min:.6g}, dx_max={mesh.dx_max:.6g}")
    print(f"dy_min={mesh.dy_min:.6g}, dy_max={mesh.dy_max:.6g}")

    field = cfd_solver.Solver(mesh, option).run()
    print_plume_diagnostics(mesh, field)

    output_dir = ROOT.parent / "images"
    plot_plume_case(mesh, field, output_dir / "plume_case.png")
