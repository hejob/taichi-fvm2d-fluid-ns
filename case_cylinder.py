import taichi as ti
import numpy as np
import time

from multiblocksolver.multiblock_solver import MultiBlockSolver
from multiblocksolver.block_solver import BlockSolver
from multiblocksolver.drawer import Drawer


real = ti.f32

### initialize solver with simulation settings
gamma = 1.4
ma0 = 2.9
# re0 = 1.225 * (ma0 * 343) * 1.0 / (1.81e-5)
re0 = 7.2e5
p0 = 1.0 / gamma / ma0 / ma0
e0 = p0 / (gamma - 1.0) + 0.5 * 1.0


## sizes
ni = 240             # points on the cylinder surface
nj = 116            # points in normal direction
radius = 1.0        # cylinder radius (use as dimensionless unit)
# length_wall = 0.05  # cell height nearest wall
# stretch_rate = 1.005 # cell stretch rate
length_wall = 0.005  # cell height nearest wall
stretch_rate = 1.03 # cell stretch rate


@ti.func
def generate_edge_points_exp_stretch(d_start, stretch_rate, n) -> real:
    return d_start * (stretch_rate**n - 1.0) / (stretch_rate - 1.0)

### calculate angle start from left to right then down then back to left
@ti.func
def generate_surface_angle(n, ni) -> real:
    ## n: index in [0, nj - 1]
    alpha = 2.0 * np.pi / (ni - 1) * n
    return np.pi - alpha

@ti.kernel
def generate_circular_grids(
        x: ti.template(), i0: ti.i32, i1: ti.i32, j0: ti.i32, j1: ti.i32):
    ## EXAMPLE: supersonic cylinder
    ## NOTICE: virtual voxels are not generated here
    ## TODO: generate bc by program?

    field_radius = radius + generate_edge_points_exp_stretch(length_wall, stretch_rate, nj)

    for I in ti.grouped(ti.ndrange((i0, i1), (j0, j1))):
        angle = generate_surface_angle(I[0], ni)
        wall_distance = generate_edge_points_exp_stretch(length_wall, stretch_rate, I[1])

        vector_unit = ti.Vector([ti.cos(angle), ti.sin(angle)])
        pos_circle = vector_unit * (radius + wall_distance)

        ## transform into (x, y) coord start from left bottom
        pos_global = pos_circle + ti.Vector([field_radius, field_radius])

        x[I] = pos_global


field_distance = length_wall * (stretch_rate**nj - 1.0) / (stretch_rate - 1.0)
width = height = 2.0 * (radius + field_distance)


solver = MultiBlockSolver(
    BlockSolver,
    Drawer,
    width=width,
    height=height,
    n_blocks=1,
    block_dimensions=[(ni, nj)],
    ma0=ma0,
    # dt=5e-4,
    dt=1e-3,
    is_dual_time=True,
    convect_method=2,
    is_viscous=False,
    temp0_raw=273,
    re0=re0,
    gui_size=(800, 640),
    display_field=True,
    display_value_min=0.0,
    display_value_max=3.0,
    output_line=True,
    output_line_ends=((-1.0 + width / 2.0, 1.1 + height / 2.0), (1.0 + width / 2.0, 1.1 + height / 2.0)),
    output_line_num_points=50,
    output_line_var=7,  # Mach number. 0~7: rho/u/v/et/uu/p/a/ma
    output_line_plot_var=0)  # output along x-axis on plot

# solver.set_debug(True)

### generate grids in Solver's x tensor
(i_range, j_range) = solver.solvers[0].range_grid
generate_circular_grids(solver.solvers[0].x, i_range[0], i_range[1], j_range[0], j_range[1])


### boundary conditions
###     0/1/2/3/4: inlet(super)/outlet(super)/symmetry/wall(noslip)/wall(slip)
bc_q_values=[
    (1.0, 1.0 * 1.0, 1.0 * 0.0, 1.0 * e0)
]

bc_array = [
    ## move to bc connection
    # (2, 1, nj + 1, 0, 0, None),     # left interconnection, upper side, symmetry for now
    # (2, 1, nj + 1, 0, 1, None),     # left interconnection, lower side, symmetry for now

    (3, 1, ni + 1, 1, 0, None),       # wall cylinder

    (0, 1, ni // 4 + 1, 1, 1, 0),  # far-field, left upper side, inlet for left half
    (0, ni + 2 - ni // 4, ni + 1, 1, 1, 0),  # far-field, left lower side, inlet for left half
    (1, ni // 4 + 1, ni + 2 - ni // 4, 1, 1, None),     # right half far-field, outlet
]
solver.solvers[0].set_bc(bc_array, bc_q_values)


bc_connection_array = [
    ## ((block,   start, march_plus_or_minus_direction(1/-1), surface direction (0/1, i or j), surface start or end(0/1, surf i0/j0 or iend/jend)))
    ((0,  1, 1, 0, 0), (0,  1, 1, 0, 1), nj),   # left interconnection, upper side, copy from lower side
    ((0,  1, 1, 0, 1), (0,  1, 1, 0, 0), nj),   # left interconnection, lower side
]
solver.set_bc_connection(bc_connection_array)

solver.set_display_options(
        display_color_map=1,
        display_steps=20,
        display_show_grid=False,
        display_show_xc=False,
        display_show_velocity=False,
        display_show_velocity_skip=(4,4),
        display_show_surface=False,
        display_show_surface_norm=False,
        display_gif_files=False
    )

### start simulation loop
t = time.time()
solver.run()

### output statistics
print(f'Solver time: {time.time() - t:.3f}s')
ti.kernel_profiler_print()
ti.core.print_profile_info()
ti.core.print_stat()
