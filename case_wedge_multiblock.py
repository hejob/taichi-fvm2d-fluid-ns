#========================================
# Supersonic Shockwave Wedge Test Case
#
# @author hejob.moyase@gmail.com
#========================================

import taichi as ti
import time

from multiblock.multiblock_solver import MultiBlockSolver
from multiblock.block_solver import BlockSolver
from multiblock.drawer import Drawer

##################
# TODO: is multiple taichi instance ok?
real = ti.f32
# ti.init(arch=ti.cpu, default_fp=real, kernel_profiler=True)

width = 1.0
height = 0.8
ni = 400
nj = 200

@ti.kernel
def generate_wedge_grids(
        x: ti.template(), i0: ti.i32, i1: ti.i32, j0: ti.i32, j1: ti.i32):
    ## EXAMPLE: supersonic wedge
    ## NOTICE: virtual voxels are not generated here
    ## TODO: generate bc by program?
    dx = width / ni
    dy = height / nj
    for I in ti.grouped(ti.ndrange((i0, i1), (j0, j1))):
        tx = dx * I[0]
        ty = dy * I[1]
        px = tx
        py = ty
        if (px > 0.2 * width):
            s = (1.0 * I[0] / ni) - 0.2
            py = 0.2 * s + (ty / 0.8) * (0.8 - 0.2 * s)
        x[I] = ti.Vector([px, py])

##################
# Main Test Case
##################
if __name__ == '__main__':

    ### initialize solver with simulation settings
    gamma = 1.4
    ma0 = 2.4
    re0 = 1.225 * (ma0 * 343) * 1.0 / (1.81e-5)
    p0 = 1.0 / gamma / ma0 / ma0
    e0 = p0 / (gamma - 1.0) + 0.5 * 1.0

    solver = MultiBlockSolver(
        BlockSolver,
        Drawer,
        width=width,
        height=height,
        n_blocks=1,
        block_dimensions=[(ni, nj)],
        ma0=ma0,
        dt=1e-5,
        convect_method=2,
        is_viscous=True,
        temp0_raw=273,
        re0=re0,
        gui_size=(400, 320),
        display_field=True,
        display_value_min=0.0,
        display_value_max=3.0,
        output_line=True,
        output_line_ends=((0.1, 0.4), (0.9, 0.4)),
        output_line_num_points=50,
        output_line_var=7,  # Mach number. 0~7: rho/u/v/et/uu/p/a/ma
        output_line_plot_var=0)  # output along x-axis on plot


    ### generate grids in Solver's x tensor
    (i_range, j_range) = solver.solvers[0].range_grid
    generate_wedge_grids(solver.solvers[0].x, i_range[0], i_range[1], j_range[0], j_range[1])

    ### boundary conditions
    ###     0/1/2/3/4: inlet(super)/outlet(super)/symmetry/wall(noslip)/wall(slip)
    i_bc = int(0.2 * ni // 1) + 1

    bc_q_values=[
        (1.0, 1.0 * 1.0, 1.0 * 0.0, 1.0 * e0)
    ]

    bc_array = [
        (0, 1, nj + 1, 0, 0, 0),        # left super inlet
        (1, 1, nj + 1, 0, 1, None),     # right, super outlet
        (2, 1, i_bc, 1, 0, None),       # down, symmetry before wedge
        (4, i_bc, ni + 1, 1, 0, None),  # down, wall wedge
        (1, 1, ni + 1, 1, 1, None),     # top, super outlet (right?)
    ]
    solver.solvers[0].set_bc(bc_array, bc_q_values)

    solver.set_display_options(
            display_steps=20,
            display_show_grid=False,
            display_show_xc=False,
            display_show_velocity=True,
            display_show_velocity_skip=(16,8),
            display_show_surface=False,
            display_show_surface_norm=False
        )

    ### start simulation loop
    t = time.time()
    solver.run()

    ### output statistics
    print(f'Solver time: {time.time() - t:.3f}s')
    ti.kernel_profiler_print()
    ti.core.print_profile_info()
    ti.core.print_stat()
