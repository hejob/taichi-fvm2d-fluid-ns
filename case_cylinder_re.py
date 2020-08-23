import taichi as ti
import numpy as np
import time

from multiblocksolver.multiblock_solver import MultiBlockSolver
from multiblocksolver.block_solver import BlockSolver
from multiblocksolver.drawer import Drawer


real = ti.f32

### initialize solver with simulation settings
gamma = 1.4
ma0 = 0.5
re0 = 100
p0 = 1.0 / gamma / ma0 / ma0
e0 = p0 / (gamma - 1.0) + 0.5 * 1.0


## sizes
radius = 0.5        # cylinder radius (d as dimensionless unit)
ni = 40             # cells on the cylinder surface (1/4 region)
nj = 60            # cells in normal direction
ni_after = 80       # after-cylinder right region
nj_after = ni
dx_after = radius / 6
# length_wall = 0.05  # cell height nearest wall
# stretch_rate = 1.005 # cell stretch rate
length_wall = 0.005  # cell height nearest wall
stretch_rate = 1.05 # cell stretch rate

## coord
# width = 2 * radius * 4
# height = 2 * radius * 2
# coord_shift = (2 * radius, 2 * radius)
width = 2 * radius * 5.5
height = 2 * radius * 3
coord_shift = (3 * radius, 3 * radius)


## template edges for one quater region around cylinder
edge_surface = ti.Vector.field(2, dtype=real, shape=(ni + 1))  ## on cylinder
edge_border = ti.Vector.field(2, dtype=real, shape=(ni + 1)) ## on far border
edge_left = ti.Vector.field(2, dtype=real, shape=(nj + 1)) ## one straight line in i0 direciton, template for nj distribution

@ti.func
def generate_edge_points_exp_stretch(d_start, stretch_rate, n) -> real:
    return d_start * (stretch_rate**n - 1.0) / (stretch_rate - 1.0)

### calculate angle start from left to right then down then back to left
### in cynlinder coord
@ti.func
def generate_surface_edge(angle_start):
    for i in range(ni + 1):
        alpha = 2.0 * np.pi * (angle_start - i * 90.0 / ni) / 360.0
        edge_surface[i] = ti.Vector([ti.cos(alpha), ti.sin(alpha)]) * radius

### straight line points in cylinder coord
@ti.func
def generate_left_edge(angle_start):
    alpha = angle_start * 2.0 * np.pi / 360.0
    for j in range(nj + 1):
        l = radius + generate_edge_points_exp_stretch(length_wall, stretch_rate, j)
        edge_left[j] = l * ti.Vector([ti.cos(alpha), ti.sin(alpha)])

## NOTICE: virtual voxels are not generated here
@ti.kernel
def generate_circular_quarter_region(
        x: ti.template(), i0: ti.i32, i1: ti.i32, j0: ti.i32, j1: ti.i32, angle_start: real): ## angle_start, i0 edge angle
    ## TODO: generate bc by program?

    generate_surface_edge(angle_start)
    generate_left_edge(angle_start)

    pb0 = edge_left[nj]
    pb1 = ti.Vector([pb0[1], -1.0 * pb0[0]])  # rotate -> 90deg

    # generate border line
    for i in range(ni + 1):
        edge_border[i] = pb0 + (pb1 - pb0) * (1.0 * i / ni)

    # generate grid with 3 edge templates
    for i in range(i0, i1):
        p0 = edge_surface[i]
        p1 = edge_border[i]
        for j in range(j0, j1):
            p = p0 + (p1 - p0) * (edge_left[j] - edge_left[0]).norm() / (edge_left[nj] - edge_left[0]).norm()
            ### transform into screen coord
            x[i, j] = p + ti.Vector([coord_shift[0], coord_shift[1]])

@ti.kernel
def generate_rectangle_region_after(
        x: ti.template(), i0: ti.i32, i1: ti.i32, j0: ti.i32, j1: ti.i32):
    ## TODO: generate bc by program?

    generate_left_edge(45)

    pb1 = edge_left[nj]
    pb0 = ti.Vector([pb1[1], -1.0 * pb1[0]])  # rotate -> 90deg

    # generate grid with 3 edge templates
    for i, j in ti.ndrange((i0, i1), (j0, j1)):
            p = pb0 + (pb1 - pb0) * j / (j1 - j0 - 1) + i * ti.Vector([dx_after, 0])
            x[i, j] = p + ti.Vector([coord_shift[0], coord_shift[1]])


solver = MultiBlockSolver(
    BlockSolver,
    Drawer,
    width=width,
    height=height,
    n_blocks=5,
    block_dimensions=[(ni, nj), (ni, nj), (ni, nj), (ni, nj), (ni_after, nj_after)],
    ma0=ma0,
    dt=1e-4,
    convect_method=2,
    is_viscous=True,
    temp0_raw=273,
    re0=re0,
    gui_size=(800, 400),
    display_field=True,
    display_value_min=-0.1,
    display_value_max=1.0,
    output_line=False,
    output_line_ends=((1.1 + radius * 2.0, height / 2.0), (1.9 + radius * 2.0, height / 2.0)),
    output_line_num_points=5,
    output_line_var=1,  # Mach number. 0~7: rho/u/v/et/uu/p/a/ma
    output_line_plot_var=0)  # output along x-axis on plot

# solver.set_debug(True)

### generate grids in Solver's x tensor
for block in range(4):
    (i_range, j_range) = solver.solvers[block].range_grid
    generate_circular_quarter_region(solver.solvers[block].x, i_range[0], i_range[1], j_range[0], j_range[1], 225.0 - 90.0 * block)

(i_range, j_range) = solver.solvers[4].range_grid
generate_rectangle_region_after(solver.solvers[4].x, i_range[0], i_range[1], j_range[0], j_range[1])


### boundary conditions
###     0/1/2/3/4: inlet(super)/outlet(super)/symmetry/wall(noslip)/wall(slip)
bc_q_values=[
    (1.0, 1.0 * 1.0, 1.0 * 0.0, 1.0 * e0)
]

bc_array = [
    [
        # LEFT REGION
        # (2, 1, nj + 1, 0, 0, None),     # conn to 4
        # (2, 1, nj + 1, 0, 1, None),     # conn to 2
        (3, 1, ni + 1, 1, 0, None),       # wall cylinder
        (10, 1, ni + 1, 1, 1, 0),  # left far-field, conn or inlet
    ],
    [
        # UP REGION
        # (2, 1, nj + 1, 0, 0, None),     # conn to 1
        # (2, 1, nj + 1, 0, 1, None),     # conn to 3
        (3, 1, ni + 1, 1, 0, None),       # wall cylinder
        (3, 1, ni + 1, 1, 1, 0),  # upper wall
    ],
    [
        # RIGHT REGION
        # (2, 1, nj + 1, 0, 0, None),     # conn to 2
        # (2, 1, nj + 1, 0, 1, None),     # conn to 4
        (3, 1, ni + 1, 1, 0, None),       # wall cylinder
        # (1, 1, ni + 1, 1, 1, 0),  # right far-field, conn or outlet
        # conn to 5
    ],
    [
        # DOWN REGION
        # (2, 1, nj + 1, 0, 0, None),     # conn to 3
        # (2, 1, nj + 1, 0, 1, None),     # conn to 1
        (3, 1, ni + 1, 1, 0, None),       # wall cylinder
        (3, 1, ni + 1, 1, 1, 0),  # lower wall
    ],

    [
        # RIGHT AFTER REGION
        # (2, 1, nj_after + 1, 0, 0, None),     # conn to 3
        (1, 1, nj_after + 1, 1, 1, 0),  # right far-field, conn or outlet
        (3, 1, ni_after + 1, 1, 0, None),       # down wall
        (3, 1, ni_after + 1, 1, 1, None),       # up wall
    ]
]
for i in range(4):
    solver.solvers[i].set_bc(bc_array[i], bc_q_values)

bc_connection_array = [
    ## ((block,   start, march_plus_or_minus_direction(1/-1), surface direction (0/1, i or j), surface start or end(0/1, surf i0/j0 or iend/jend)))
    
    ((0,  1, 1, 0, 1), (1,  1, 1, 0, 0), nj),   # 1-2
    ((1,  1, 1, 0, 0), (0,  1, 1, 0, 1), nj),   # 1-2

    ((1,  1, 1, 0, 1), (2,  1, 1, 0, 0), nj),   # 2-3
    ((2,  1, 1, 0, 0), (1,  1, 1, 0, 1), nj),   # 3-2

    ((2,  1, 1, 0, 1), (3,  1, 1, 0, 0), nj),   # 2-3
    ((3,  1, 1, 0, 0), (2,  1, 1, 0, 1), nj),   # 3-2

    ((3,  1, 1, 0, 1), (0,  1, 1, 0, 0), nj),   # 2-3
    ((0,  1, 1, 0, 0), (3,  1, 1, 0, 1), nj),   # 3-2


    ((2,  ni, -1, 1, 1), (4,  1, 1, 0, 0), ni),   # 3-5
    ((4,  1, 1, 0, 0), (2,  ni, -1, 1, 1), ni),   # 5-3
]
solver.set_bc_connection(bc_connection_array)

solver.set_display_options(
        display_color_map=1,
        display_steps=1,
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
