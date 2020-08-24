#========================================
# SOD Shockwave Tube Test Case
# 1D shockwave validation
#
# @author hejob.moyase@gmail.com
#========================================

import taichi as ti
import numpy as np
import time

from multiblocksolver.multiblock_solver import MultiBlockSolver
from multiblocksolver.block_solver import BlockSolver
from multiblocksolver.drawer import Drawer

real = ti.f32
# ti.init(arch=ti.cpu, default_fp=real, kernel_profiler=True)

### initialize solver with simulation settings
gamma = 1.4
ma0 = 1.0 # dummy
p0 = 1.0 / gamma / ma0 / ma0
e0 = p0 / (gamma - 1.0) + 0.5 * 1.0

p1 = p0
p2 = 0.1 * p1
e1 = p1 / (gamma - 1.0) + 0.5 * 0.0
e2 = p2 / (gamma - 1.0) + 0.5 * 0.0

## sizes
ni = 100
nj = 3

width = 10.0
height = 1.0

eps = 1e-10
dw = width / ni
dh = height /nj

@ti.kernel
def generate_grid(
        x: ti.template(), i0: ti.i32, i1: ti.i32, j0: ti.i32, j1: ti.i32):
    for I in ti.grouped(ti.ndrange((i0, i1), (j0, j1))):
        px = dw * I[0]
        py = dh * I[1]
        x[I] = ti.Vector([px, py])


@ti.kernel
def init_q_sod(solver: ti.template()):
    for I in ti.grouped(ti.ndrange(*solver.range_elems)):
        if solver.x[I][0] < 0.5 * width:
            solver.q[I] = ti.Vector([
                1.0,
                0.0,
                0.0,
                e1,
            ])
        else:
            solver.q[I] = ti.Vector([
                0.125,
                0.0,
                0.0,
                e2,
            ])
# sod initialize
def custom_init_q_sod(solver):
    init_q_sod(solver)


solver = MultiBlockSolver(
    BlockSolver,
    Drawer,
    width=width,
    height=height,
    n_blocks=1,
    block_dimensions=[(ni, nj)],
    ma0=ma0,
    dt=1e-3,
    is_dual_time=False,
    convect_method=2,
    is_viscous=False,
    temp0_raw=273,
    re0=1e5, # dummy
    gui_size=(600, 60),
    display_field=True,
    display_value_min=0.0,
    display_value_max=1.0,
    output_line=True,
    output_line_ends=((0.1, 0.5), (9.9, 0.5)),
    output_line_num_points=50,
    output_line_var=0,  # Mach number. 0~7: rho/u/v/et/uu/p/a/ma
    output_line_plot_var=0)  # output along x-axis on plot

# solver.set_debug(True)

### generate grids in Solver's x tensor
(i_range, j_range) = solver.solvers[0].range_grid
generate_grid(solver.solvers[0].x, i_range[0], i_range[1], j_range[0], j_range[1])

### boundary conditions
###     0/1/2/3/4: inlet(super)/outlet(super)/symmetry/wall(noslip)/wall(slip)
bc_q_values=[
    (1.0, 0.0, 0.0, 1.0 * e0)
]

bc_array = [
    ## move to bc connection
    (1, 1, nj + 1, 0, 0, 0),        # left outlet
    (1, 1, nj + 1, 0, 1, None),     # right outlet
    
    (3, 1, ni + 1, 1, 0, None),     # down sym
    (3, 1, ni + 1, 1, 1, None),     # up sym
]
solver.solvers[0].set_bc(bc_array, bc_q_values)

solver.set_custom_simulations(custom_init_q_sod)

solver.set_display_options(
        display_steps=20,
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

