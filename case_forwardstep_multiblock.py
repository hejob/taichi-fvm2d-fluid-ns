#========================================
# Supersonic Forward Facing Step TestCase
# two-blocks invisid flow with shockwaves
#
# @author hejob.moyase@gmail.com
#========================================

import taichi as ti
import numpy as np
import time

from multiblock.multiblock_solver import MultiBlockSolver
from multiblock.block_solver import BlockSolver
from multiblock.drawer import Drawer

real = ti.f32
# ti.init(arch=ti.cpu, default_fp=real, kernel_profiler=True)

### initialize solver with simulation settings
gamma = 1.4
ma0 = 3.0
p0 = 1.0 / gamma / ma0 / ma0
e0 = p0 / (gamma - 1.0) + 0.5 * 1.0

## sizes
# ni_h = 16           # dh = 0.2 / ni_h, step height
ni_h = 24
dh = 0.2 / ni_h

width = 3.0
height = 1.0

eps = 1e-10
ni_step = int((0.6 + eps) // dh)
nj_step = int((0.2 + eps) // dh)
ni_main = int((3.0 + eps) // dh)
nj_main = int((0.8 + eps) // dh)

@ti.kernel
def generate_grid_step(
        x: ti.template(), i0: ti.i32, i1: ti.i32, j0: ti.i32, j1: ti.i32):
    for I in ti.grouped(ti.ndrange((i0, i1), (j0, j1))):
        tx = dh * I[0]
        ty = dh * I[1]
        px = tx
        py = ty
        x[I] = ti.Vector([px, py])

@ti.kernel
def generate_grid_main(
        x: ti.template(), i0: ti.i32, i1: ti.i32, j0: ti.i32, j1: ti.i32):
    for I in ti.grouped(ti.ndrange((i0, i1), (j0, j1))):
        tx = dh * I[0]
        ty = dh * I[1]
        px = tx
        py = ty + 0.2
        x[I] = ti.Vector([px, py])

solver = MultiBlockSolver(
    BlockSolver,
    Drawer,
    width=width,
    height=height,
    n_blocks=2,
    block_dimensions=[(ni_step, nj_step), (ni_main, nj_main)],
    ma0=ma0,
    dt=1e-3,
    is_dual_time=False,
    convect_method=2,
    is_viscous=False,
    temp0_raw=273,
    re0=1e5, # dummy
    gui_size=(600, 200),
    display_field=True,
    display_value_min=1.0,
    display_value_max=3.0,
    output_line=True,
    output_line_ends=((0.05, 0.5), (2.95, 0.5)),
    output_line_num_points=50,
    output_line_var=7,  # Mach number. 0~7: rho/u/v/et/uu/p/a/ma
    output_line_plot_var=0)  # output along x-axis on plot

# solver.set_debug(True)

### generate grids in Solver's x tensor
(i_range, j_range) = solver.solvers[0].range_grid
generate_grid_step(solver.solvers[0].x, i_range[0], i_range[1], j_range[0], j_range[1])

(i_range, j_range) = solver.solvers[1].range_grid
generate_grid_main(solver.solvers[1].x, i_range[0], i_range[1], j_range[0], j_range[1])

### boundary conditions
###     0/1/2/3/4: inlet(super)/outlet(super)/symmetry/wall(noslip)/wall(slip)
bc_q_values=[
    (1.0, 1.0 * 1.0, 1.0 * 0.0, 1.0 * e0)
]

bc_array_step = [
    ## move to bc connection
    (0, 1, nj_step + 1, 0, 0, 0),        # left inlet
    (3, 1, nj_step + 1, 0, 1, None),     # right wall
    

    # (3, 1, ni_step + 1, 1, 0, None),     # down wall
    (2, 1, ni_step + 1, 1, 0, None),     # down sym?

    # (3, 1, ni_step + 1, 1, 1, None),     # up wall -> connect
]
solver.solvers[0].set_bc(bc_array_step, bc_q_values)

bc_array_main = [
    ## move to bc connection
    (0, 1, nj_main + 1, 0, 0, 0),        # left inlet
    (1, 1, nj_main + 1, 0, 1, None),     # right outlet

    # (3, 1, ni_step + 1, 1, 0, None),     # down wall -> connect
    (4, ni_step + 1, ni_main + 1, 1, 0, None),     # down wall

    (4, 1, ni_main + 1, 1, 1, None),     # up wall
]
solver.solvers[1].set_bc(bc_array_main, bc_q_values)


bc_connection_array = [
    ## ((block,   start, march_plus_or_minus_direction(1/-1), surface direction (0/1, i or j), surface start or end(0/1, surf i0/j0 or iend/jend)))
    ((0,  1, 1, 1, 1), (1,  1, 1, 1, 0), ni_step),   # step bc: connect to upper main
    ((1,  1, 1, 1, 0), (0,  1, 1, 1, 1), ni_step),   # main bc: connect to lower step block
]
solver.set_bc_connection(bc_connection_array)

solver.set_display_options(
        display_steps=40,
        display_color_map=1,
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

