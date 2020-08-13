#========================================
# Supersonic Shockwave Cylinder Test Case
# one-block with interconnection bc
#     and farfield bc
#
# @author hejob.moyase@gmail.com
#========================================

import taichi as ti
import numpy as np
import time

from solver.Solver import Solver

##################
# TODO: is multiple taichi instance ok?
real = ti.f32
# ti.init(arch=ti.cpu, default_fp=real, kernel_profiler=True)

## sizes
ni = 240             # points on the cylinder surface
nj = 110            # points in normal direction
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
    return -1.0 * alpha

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

##################
# Main Test Case
##################
if __name__ == '__main__':

    ### initialize solver with simulation settings
    gamma = 1.4
    ma0 = 2.5
    # re0 = 1.225 * (ma0 * 343) * 1.0 / (1.81e-5)
    re0 = 7.2e5
    p0 = 1.0 / gamma / ma0 / ma0
    e0 = p0 / (gamma - 1.0) + 0.5 * 1.0

    field_distance = length_wall * (stretch_rate**nj - 1.0) / (stretch_rate - 1.0)
    width = height = 2.0 * (radius + field_distance)
 
    solver = Solver(
        width=width,
        height=height,
        ni=ni,
        nj=nj,
        ma0=ma0,
        dt=1e-4,
        convect_method=1,
        is_viscous=False,
        temp0_raw=300,
        re0=re0,
        gui_size=(400, 400),
        display_field=True,
        display_value_min=0.0,
        display_value_max=3.0,
        output_line=True,
        output_line_ends=((-1.0 + width / 2.0, 1.1 + height / 2.0), (1.0 + width / 2.0, 1.1 + height / 2.0)),
        output_line_num_points=200,
        output_line_var=7,  # Mach number. 0~7: rho/u/v/et/uu/p/a/ma
        output_line_plot_var=0)  # output along x-axis on plot

    ### generate grids in Solver's x tensor
    (i_range, j_range) = solver.range_grid
    generate_circular_grids(solver.x, i_range[0], i_range[1], j_range[0], j_range[1])

    ### boundary conditions
    ###     0/1/2/3/4: inlet(super)/outlet(super)/symmetry/wall(noslip)/wall(slip)
    bc_q_values=[
        (1.0, 1.0 * 1.0, 1.0 * 0.0, 1.0 * e0)
    ]

    bc_array = [
        ## move to bc connection
        # (2, 1, nj + 1, 0, 0, None),
        # (2, 1, nj + 1, 0, 1, None),
        (3, 1, ni + 1, 1, 0, None),       # wall cylinder

        (1, 1, ni // 4 + 1, 1, 1, None),  # far-field, right lower side, outlet
        (1, ni + 2 - ni // 4, ni + 1, 1, 1, None),  # far-field, right upper side, outlet
        (0, ni // 4 + 1, ni + 2 - ni // 4, 1, 1, 0),     # left half far-field, inlet
    ]
    solver.set_bc(bc_array, bc_q_values)

    bc_connection_array = [
        ## ((start, march_plus_or_minus_direction(1/-1), surface direction (0/1, i or j), surface start or end(0/1, surf i0/j0 or iend/jend)))
        ((1, 1, 0, 0), (1, 1, 0, 1), nj),   # right interconnection, lower side, copy from upper side
        ((1, 1, 0, 1), (1, 1, 0, 0), nj),   # right interconnection, upper side
    ]
    solver.set_bc_connection(bc_connection_array)

    solver.set_display_options(
            display_steps=20,
            display_show_grid=False,
            display_show_xc=False,
            display_show_velocity=False,
            display_show_velocity_skip=(4,4),
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
