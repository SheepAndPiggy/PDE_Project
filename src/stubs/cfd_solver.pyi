from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

def version() -> str: ...
def dependency_summary() -> str: ...
def cuda_enabled() -> bool: ...
def cuda_device_count() -> int: ...
def cuda_runtime_version() -> int: ...

class Mesh2D:
    nx: int
    ny: int

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    heated_x_min: float
    heated_x_max: float
    first_layer_height: float
    y_growth_ratio: float

    dx_min: float
    dx_max: float
    dy_min: float
    dy_max: float

    x_nodes: FloatArray
    y_nodes: FloatArray
    x: FloatArray
    y: FloatArray

class Mesh2DGenerator:
    def __init__(
        self,
        nx: int,
        ny: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        heated_x_min: float,
        heated_x_max: float,
        first_layer_height: float = -1.0,
        x_refinement_strength: float = 4.0,
        x_refinement_width: float = -1.0,
        max_y_growth_ratio: float = 1.25,
    ) -> None: ...
    def generate(self) -> Mesh2D: ...
    @staticmethod
    def first_layer_height_from_y_plus(
        rho: float,
        mu: float,
        friction_velocity: float,
        target_y_plus: float = 1.0,
    ) -> float: ...
    @staticmethod
    def first_layer_height_from_thermal_boundary_layer(
        thermal_boundary_layer_thickness: float,
        points_in_boundary_layer: int = 10,
    ) -> float: ...

class SolverOption:
    reynolds: float
    richardson: float
    peclet: float
    u_in: float
    v_in: float
    T_in: float
    wall_heat_flux: float
    time_sep: float
    step: int
    save_sep: int
    poisson_max_interval: int
    poisson_tolerance: float
    poisson_omega: float
    use_cuda: bool

    def __init__(self) -> None: ...

class FlowField:
    step: int
    time: float
    u: FloatArray
    v: FloatArray
    p: FloatArray
    T: FloatArray
    divergence: FloatArray

class Solver:
    def __init__(self, mesh: Mesh2D, option: SolverOption) -> None: ...
    def run(self) -> FlowField: ...
