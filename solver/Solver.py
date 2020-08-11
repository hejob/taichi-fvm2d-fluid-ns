# ==============================================
#  structured 2D compressible FVM fluid solver
#
#  @author hejob.moyase@gmail.com
# ==============================================

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import math
import time

real = ti.f32
ti.init(arch=ti.cpu, default_fp=real, kernel_profiler=True)

##############
# if we need convenient global debug variables
# dd = ti.field(dtype=ti.i32, shape=())
# dd2 = ti.Vector.field(2, dtype=ti.i32, shape=())
# dd3 = ti.Vector.field(2, dtype=real, shape=())
# dd4 = ti.Vector.field(3, dtype=real, shape=())


###############################
### Main Solver
###############################
@ti.data_oriented
class Solver:

    #--------------------------------------------------------------------------
    #  Initialize functions
    #
    #    Solver settings, grid definitions, etc.
    #--------------------------------------------------------------------------

    ################
    # constructor:
    #    simulation settings, memory allocations
    def __init__(
            self,
            # geometry
            width,
            height,
            ni,
            nj,
            # physics
            ma0,
            # simulation
            dt,
            # viscous
            is_viscous=False,
            temp0_raw=273,
            re0=1e5,
            # output: realtime field display
            gui_size=(400, 400),
            display_field=True,
            display_value_min=0.0,
            display_value_max=1.0,
            output_line=False,
            output_line_ends=((), ()),
            output_line_num_points=200,
            output_line_var=7,  # Mach number. 0~7: rho/u/v/et/uu/p/a/ma
            output_line_plot_var=0):
        ### TODO: read from input files, but are there perforamance issues?

        ## base properties
        self.dim = 2
        self.n_virtual_voxels = 1  # numbers of virtual voxels at boundary

        ## geometry attributes for grids
        self.width = width
        self.height = height
        self.ni = ni
        self.nj = nj

        ## props
        self.ma0 = ma0
        self.gamma = 1.4
        self.p0 = 1.0 / self.gamma / self.ma0 / self.ma0
        self.e0 = self.p0 / (self.gamma - 1.0) + 0.5 * 1.0

        ## simulation
        self.dt = dt
        self.t = 0.0

        ## viscous Navier-Stokes simulation properties
        self.is_convect_calculated = True  # a switch to skip convect flux only for debug
        self.is_viscous = is_viscous
        if self.is_viscous:
            self.temp0_raw = temp0_raw
            self.re0 = re0  # re = rho * u * L / miu  = u * L / niu
            self.cs_sthland = 117.0 / self.temp0_raw
            self.pr_laminar = 0.72  # laminar prantl number
            self.cp0 = self.gamma / (self.gamma - 1.0) * self.p0

        ## realtime outputs
        ##  display simulation field
        self.gui_size = gui_size
        self.display_field = display_field
        self.display_value_min = display_value_min
        self.display_value_max = display_value_max
        ## switches, can be set later
        self.display_steps = 20
        self.display_show_grid = False
        self.display_show_xc = False
        self.display_show_velocity = False
        self.display_show_velocity_skip = (1, 1)
        self.display_show_surface = False
        self.display_show_surface_norm = False

        ##  plots one quantity along one line
        self.output_line = output_line
        self.output_line_ends = output_line_ends
        self.output_line_num_points = output_line_num_points
        self.output_line_var = output_line_var  # Mach number. 0~7: rho/u/v/et/uu/p/a/ma
        self.output_line_plot_var = output_line_plot_var  # output x on plot

        ###############################
        ## taichi tensors and enumeration helpers
        ## all variable values are dimensionless
        self.init_allocations()

        #########
        ## still needs:
        ##      GRIDS, Boundary Conditions
        ##      Deal with these in CALLER before running solve
        ## bc_info is an array of boundary informations in:
        ##      (bc_type, rng0, rng1, direction, start_or_end, q_index)
        ##      with q_index directing to q(rho, rho*u, rho*v, rho*et)
        ##          value item in bc_q_values array
        self.bc_info = []
        self.bc_q_values = []
        ## bc_connection_info stores boundaries that is interconnected
        ##      ((bc_def), (bc_other_side_def), number of cells)
        ##      with bc_def:
        ##              (start, march_plus_or_minus_direction(1/-1), 
        ##                 surface direction (0/1, i or j), surface start or end(0/1, surf i0/j0 or iend/jend))
        ##      sample: ((1, +1, 0, 0), (1, +1, 0, 1), nj)
        ##              means a boundary on i-surf-start side, from (1, 1) to (1, nj + 1), goes in j+ direction
        ##              connects to a boundary on i-surf-end side, from (ni, 1) to (ni, nj + 1), goes in j+ direction
        self.bc_connection_info = []

    ###############################
    # Taichi tensors allocations
    def init_allocations(self):

        #===================== GRIDS ==================================
        # size: (ni+1, nj+1)
        self.size_grid = (self.ni + self.n_virtual_voxels,
                          self.nj + self.n_virtual_voxels)
        # range loop helpers: [0~ni+1), [0~nj+1]
        self.range_grid = ((0, self.ni + self.n_virtual_voxels),
                           (0, self.nj + self.n_virtual_voxels))
        # nodes, size: (ni + 1) * (nj + 1), index: [0 ~ ni+1), [0 ~ nj+1)
        self.x = ti.Vector(2, dt=real)
        ti.root.dense(ti.ij, self.size_grid).place(self.x)

        #===================== ELEMENTS(cells) ==========================
        ### index from [1~ni+1), [1~nj+1), count (ni, nj) within internal region of elems
        self.range_elems = ((self.n_virtual_voxels,
                             self.n_virtual_voxels + self.ni),
                            (self.n_virtual_voxels,
                             self.n_virtual_voxels + self.nj))
        ### index from [0~ni+2), [0~nj+2), count (ni+2, nj+2) of all elems including virtual voxels
        self.range_elems_virtual = ((1 - self.n_virtual_voxels,
                                     2 * self.n_virtual_voxels + self.ni),
                                    (1 - self.n_virtual_voxels,
                                     2 * self.n_virtual_voxels + self.nj))
        self.size_elems_virtual = (self.ni + 2 * self.n_virtual_voxels,
                                   self.nj + 2 * self.n_virtual_voxels)
        ### center position
        self.xc = ti.Vector(2, dt=real)
        ### cell area
        self.elem_area = ti.field(dtype=real)
        ### main quantities to solve
        self.q = ti.Vector(4, dt=real)  # Q = (rho, rho*u, rho*v, rho*Et)
        self.flux = ti.Vector(4, dt=real)  # flux density per area * dtime
        self.elem_nodes = ti.root.dense(ti.ij, self.size_elems_virtual)
        ### time marching cache for RK3
        self.w = ti.Vector(4, dt=real)
        ### making use of offsets
        ###     we put internal region within index [1~), [1~)
        self.elem_nodes.place(self.xc,
                              self.elem_area,
                              self.q,
                              self.flux,
                              self.w,
                              offset=(1 - self.n_virtual_voxels,
                                      1 - self.n_virtual_voxels))

        #===================== SURFACES (between cells) ==========================
        ### surf direction vectors with norm = surf area, lengh in z = 1.0
        ### direction is positive to x/y increasing direction
        ###     [i, j ,dir], (i, j) in (0~n), dir is x/y ~ 0/1
        ### the [i, j, 0] is the surface between elem [i, j] and [i+1,j] in i direction
        ###     notice the left of elem [1, 1] is indexed [0, 1, 0]

        ### TODO: for better cache-hit, should we relate this to elements better although they are not indexed the same?
        # size: (ni+1, nj+1, 2)
        self.size_surfs = (self.ni + self.n_virtual_voxels,
                           self.nj + self.n_virtual_voxels, 2)
        ### loop for all surfs, range [0~ni+1), [0~nj+1), [0/1], size (ni+1, nj+1)
        self.range_surfs_ij_all = ((1 - self.n_virtual_voxels,
                                    self.n_virtual_voxels + self.ni),
                                   (1 - self.n_virtual_voxels,
                                    self.n_virtual_voxels + self.nj))
        ### when loops for i direction surfs, range [0~ni+1), [1~nj+1]], size (ni+1, nj)
        self.range_surfs_ij_i = ((1 - self.n_virtual_voxels,
                                  self.n_virtual_voxels + self.ni),
                                 (1, self.n_virtual_voxels + self.nj))
        ### when loops for j direction surfs, range [1~ni+1), [0~nj+1]], size (ni, nj+1)
        self.range_surfs_ij_j = ((1, self.n_virtual_voxels + self.ni),
                                 (1 - self.n_virtual_voxels,
                                  self.n_virtual_voxels + self.nj))
        ### Surface normal vectors, with vec norm = length of the surface edge
        self.vec_surf = ti.Vector(2, dt=real)
        self.surf_nodes = ti.root.dense(ti.ijk, self.size_surfs)
        ### use offset, we put internal region in
        ###     index [0~), [1~), 0 in i direction
        ###     and [1~), [0~), 1 in j direction
        self.surf_nodes.place(self.vec_surf,
                              offset=(1 - self.n_virtual_voxels,
                                      1 - self.n_virtual_voxels, 0))
        ### geom elem widths in 2 directions
        ### for interpolation coefs (in fact, only diffusion uses this for now)
        self.elem_width = ti.Vector(2, dt=real)
        self.elem_nodes.place(self.elem_width,
                              offset=(1 - self.n_virtual_voxels,
                                      1 - self.n_virtual_voxels))

        #===================== NS only ==========================
        if self.is_viscous:
            ###############
            ### Extra vairables for viscous flows
            ### Needed for calculation of velocity and temperature graidents

            ### primitive variables (only save velocity and T for their gradients in diffusion terms)
            ### TODO: here we can instead calculate on the way to not save center primitive vars to reduce memory?
            self.v_c = ti.Vector(2, dt=real)  # primitive velocity
            self.temp_c = ti.field(dtype=real)  # temperature
            self.elem_nodes.place(self.v_c,
                                  self.temp_c,
                                  offset=(1 - self.n_virtual_voxels,
                                          1 - self.n_virtual_voxels))
            self.v_surf = ti.Vector(2, dt=real)
            self.temp_surf = ti.field(dtype=real)
            self.surf_nodes.place(self.v_surf,
                                  self.temp_surf,
                                  offset=(1 - self.n_virtual_voxels,
                                          1 - self.n_virtual_voxels, 0))
            ### gradients on center
            self.gradient_v_c = ti.Matrix(2, 2, dt=real)
            self.gradient_temp_c = ti.Vector(2, dt=real)
            self.elem_nodes.place(self.gradient_v_c,
                                  self.gradient_temp_c,
                                  offset=(1 - self.n_virtual_voxels,
                                          1 - self.n_virtual_voxels))
            ### gradients on surface
            ## TODO: here we can instead calculate on the way to not save center primitive vars to reduce memory
            self.gradient_v_surf = ti.Matrix(2, 2, dt=real)
            self.gradient_temp_surf = ti.Vector(2, dt=real)
            self.surf_nodes.place(self.gradient_v_surf,
                                  self.gradient_temp_surf,
                                  offset=(1 - self.n_virtual_voxels,
                                          1 - self.n_virtual_voxels, 0))

        #===================== OUTPUT: Field ==========================
        if self.display_field:
            # element triangles and fields to display
            self.display_img = ti.field(dtype=real, shape=(self.ni, self.nj))
            self.display_elems_triangle_a = ti.Vector.field(
                2, dtype=real, shape=(2 * self.ni * self.nj))
            self.display_elems_triangle_b = ti.Vector.field(
                2, dtype=real, shape=(2 * self.ni * self.nj))
            self.display_elems_triangle_c = ti.Vector.field(
                2, dtype=real, shape=(2 * self.ni * self.nj))
            self.display_elems_q = ti.field(dtype=ti.i32,
                                            shape=(2 * self.ni * self.nj))

        #===================== OUTPUT: Plot ==========================
        if (self.output_line):
            self.output_line_q = ti.field(dtype=real,
                                          shape=(self.output_line_num_points))
            self.output_line_points = ti.Vector.field(
                2, dtype=real, shape=(self.output_line_num_points))
            self.output_line_interp_index = ti.Vector.field(
                2, dtype=ti.i32, shape=(self.output_line_num_points))
            self.output_line_interp_coef = ti.Vector.field(
                4, dtype=real,
                shape=(self.output_line_num_points
                       ))  # coef for interpolation of 4 points in an rectangle

    ###############################
    # Dummy grids generation function
    # SHOULD BE DONE in the CALLER
    # For every case has different grids
    # TODO: read from input files?
    @ti.kernel
    def init_geom(self):
        ### TODO:  EXAMPLE: rectangular
        # TODO: generate virtual voxels
        pass

    ########################
    # Call this before solve to set boundary conditions
    # We use array clone here
    def set_bc(self, bc, bc_q_values):
        self.bc_info = bc[:]
        self.bc_q_values = bc_q_values[:]

    ########################
    # Call this before solve to set boundary connections
    # We use array clone here
    def set_bc_connection(self, connections):
        self.bc_connection_info = connections[:]

    ########################
    # Set extra display
    def set_display_options(self, display_steps=20,
                            display_show_grid=False,
                            display_show_xc=False,
                            display_show_velocity=False,
                            display_show_velocity_skip=(4, 4),
                            display_show_surface=False,
                            display_show_surface_norm=False):
        self.display_steps = display_steps
        self.display_show_grid = display_show_grid
        self.display_show_xc = display_show_xc
        self.display_show_velocity = display_show_velocity
        self.display_show_velocity_skip = display_show_velocity_skip
        self.display_show_surface = display_show_surface
        self.display_show_surface_norm = display_show_surface_norm

    #--------------------------------------------------------------------------
    #  Preparations before main simulation loop
    #
    #    Geometry caculations, etc.
    #--------------------------------------------------------------------------

    @ti.kernel
    def calc_geom(self):
        ## x center
        ##  NOTICE: virtual ones are calculated here
        for I in ti.grouped(ti.ndrange(*self.range_elems)):
            x11 = self.x[I + ti.Vector([-1, -1])]
            x21 = self.x[I + ti.Vector([0, -1])]
            x12 = self.x[I + ti.Vector([-1, 0])]
            x22 = self.x[I + ti.Vector([0, 0])]
            self.xc[I] = 0.25 * (x11 + x21 + x12 + x22)
            r12 = x21 - x12
            r21 = x11 - x22
            prod_cross = ti.abs(r12[0] * r21[1] - r12[1] * r21[0])
            self.elem_area[I] = 0.5 * prod_cross

        ### surf direction
        ## x dir to the right
        for I in ti.grouped(ti.ndrange(*self.range_surfs_ij_i)):
            vec_diff = self.x[I + ti.Vector([0, -1])] - self.x[I]
            ## rotate vec_diff by 90 degrees
            ## NOTICE: we assume S > 0, we do not deal with degenerated grid for now
            self.vec_surf[I, 0] = ti.Vector([-vec_diff[1], vec_diff[0]])
        ## y dir to the top
        for I in ti.grouped(ti.ndrange(*self.range_surfs_ij_j)):
            vec_diff = self.x[I] - self.x[I + ti.Vector(
                [-1, 0])]  # notice offset is negative in x direction
            # rotate vec_diff by 90 degrees
            # NOTICE: we assume S > 0, we do not deal with degenerated grid for now
            self.vec_surf[I, 1] = ti.Vector([-vec_diff[1], vec_diff[0]])

    #####################
    ## geom elem size in i/j, useful for interpolation on surface
    ## now diffusion only
    @ti.kernel
    def calc_geom_width(self):
        for I in ti.grouped(ti.ndrange(*self.range_elems)):
            x11 = self.x[I + ti.Vector([-1, -1])]
            x21 = self.x[I + ti.Vector([0, -1])]
            x12 = self.x[I + ti.Vector([-1, 0])]
            x22 = self.x[I + ti.Vector([0, 0])]

            x_left = 0.5 * (x11 + x12)
            x_right = 0.5 * (x21 + x22)
            x_down = 0.5 * (x11 + x21)
            x_up = 0.5 * (x12 + x22)

            d_i = x_right - x_left
            d_j = x_up - x_down

            self.elem_width[I] = ti.Vector([d_i.norm(), d_j.norm()])

    ################
    ## initialize quantities
    ## TODO: some cases may need non-homogeneous field
    @ti.kernel
    def init_q(self):
        ## internal elems only
        ## virtual elems should be initilized by bc (except 4 corner elems, not used)
        for I in ti.grouped(ti.ndrange(*self.range_elems)):
            self.q[I] = ti.Vector([
                1.0,
                1.0 * 1.0,
                1.0 * 0.0,
                1.0 * self.e0,
            ])

    ##################
    ## call all init functions here
    def init(self):

        ## init_geom should be dealt with by caller case by case
        # self.init_geom()

        self.calc_geom()

        if self.is_viscous:
            self.calc_geom_width()
            # bc: elem width and area, now only diffusion uses this
            self.bc(-1)

        self.init_q()

    #--------------------------------------------------------------------------
    #  Boundary Conditions (BC)
    #
    #    Inlet, Outlet, Symmetry, Slip Wall, Non-slip Wall
    #--------------------------------------------------------------------------

    # range, (start, end) notice end is +1
    # return value: ((x0, x1), (y0, y1), offset)
    # TODO: if expressions should be static compiled, but dir/end are not constants
    @ti.func
    def calc_bc_range(self, rng: ti.template(), dir, end) -> ti.template():
        range_bc_x = ti.Vector([0, 0])
        range_bc_y = ti.Vector([0, 0])
        offset = ti.Vector([0, 0])
        if (dir == 0):
            if (end == 0):
                range_bc_x = ti.Vector([1, 2])
                range_bc_y = ti.Vector([rng[0], rng[1]])
                offset = ti.Vector([-1, 0])
            else:
                range_bc_x = ti.Vector([self.ni, self.ni + 1])
                range_bc_y = ti.Vector([rng[0], rng[1]])
                offset = ti.Vector([1, 0])
        else:
            if (end == 0):
                range_bc_x = ti.Vector([rng[0], rng[1]])
                range_bc_y = ti.Vector([1, 2])
                offset = ti.Vector([0, -1])
            else:
                range_bc_x = ti.Vector([rng[0], rng[1]])
                range_bc_y = ti.Vector([self.nj, self.nj + 1])
                offset = ti.Vector([0, 1])
        return (range_bc_x, range_bc_y, offset)

    @ti.kernel
    def bc_inlet_super(self, rng_x: ti.template(), rng_y: ti.template(),
                       dir: ti.template(), end: ti.template(), q0: real,
                       q1: real, q2: real, q3: real, stage: ti.i32):
        rng = ti.Vector([rng_x, rng_y])
        range_bc_x, range_bc_y, offset = self.calc_bc_range(rng, dir, end)
        for I in ti.grouped(
                ti.ndrange((range_bc_x[0], range_bc_x[1]),
                           (range_bc_y[0], range_bc_y[1]))):
            bc_I = I + offset
            # far end inlet
            if stage == -1:
                self.elem_area[bc_I] = self.elem_area[I]
                self.elem_width[bc_I] = self.elem_width[I]
            elif stage == 0:
                self.q[bc_I] = ti.Vector([q0, q1, q2, q3])
            elif stage == 1:
                if ti.static(self.is_viscous):
                    self.gradient_v_c[bc_I] = 1.0 * self.gradient_v_c[I]
                    self.gradient_temp_c[bc_I] = 1.0 * self.gradient_temp_c[I]

    @ti.kernel
    def bc_symmetry(self, rng_x: ti.template(), rng_y: ti.template(),
                    dir: ti.template(), end: ti.template(), stage: ti.i32):
        rng = ti.Vector([rng_x, rng_y])
        range_bc_x, range_bc_y, offset = self.calc_bc_range(rng, dir, end)
        for I in ti.grouped(
                ti.ndrange((range_bc_x[0], range_bc_x[1]),
                           (range_bc_y[0], range_bc_y[1]))):
            bc_I = I + offset
            if stage == -1:
                self.elem_area[bc_I] = self.elem_area[I]
                self.elem_width[bc_I] = self.elem_width[I]
            elif stage == 0:
                self.q[bc_I] = self.q[I]
            elif stage == 1:
                if ti.static(self.is_viscous):
                    self.gradient_v_c[bc_I] = -1.0 * self.gradient_v_c[I]
                    self.gradient_temp_c[bc_I] = -1.0 * self.gradient_temp_c[I]

    @ti.kernel
    def bc_wall_slip(self, rng_x: ti.template(), rng_y: ti.template(),
                     dir: ti.template(), end: ti.template(), stage: ti.i32):
        rng = ti.Vector([rng_x, rng_y])
        range_bc_x, range_bc_y, offset = self.calc_bc_range(rng, dir, end)

        offset_surf_range = ti.Vector([0, 0])
        if dir == 0:  #x
            if end == 0:  #right
                offset_surf_range = ti.Vector([0, 0])
            else:
                offset_surf_range = ti.Vector([-1, 0])
        else:
            if end == 0:  #right
                offset_surf_range = ti.Vector([0, 0])
            else:
                offset_surf_range = ti.Vector([0, -1])
        for I in ti.grouped(
                ti.ndrange((range_bc_x[0], range_bc_x[1]),
                           (range_bc_y[0], range_bc_y[1]))):
            bc_I = I + offset

            if stage == -1:
                self.elem_area[bc_I] = self.elem_area[I]
                self.elem_width[bc_I] = self.elem_width[I]
            elif stage == 0:
                surf_inside_normal = self.vec_surf[I + offset_surf_range, dir]

                prim = self.q_to_primitive(self.q[I])
                rho = prim[0]
                u = ti.Vector([prim[0], prim[1]])
                e = prim[3]
                p = e * (self.gamma - 1.0)
                surf_inside_dir = surf_inside_normal.normalized()
                u_slip = u - u.dot(surf_inside_dir) * surf_inside_dir * 2.0
                self.q[bc_I] = ti.Vector([
                    rho, rho * u_slip[0], rho * u_slip[1],
                    rho * e + 0.5 * u_slip.norm_sqr()
                ])
            elif stage == 1:
                if ti.static(self.is_viscous):
                    self.gradient_v_c[bc_I] = 1.0 * self.gradient_v_c[I]
                    self.gradient_temp_c[bc_I] = -1.0 * self.gradient_temp_c[I]

    @ti.kernel
    def bc_wall_noslip(self, rng_x: ti.template(), rng_y: ti.template(),
                       dir: ti.template(), end: ti.template(), stage: ti.i32):
        rng = ti.Vector([rng_x, rng_y])
        range_bc_x, range_bc_y, offset = self.calc_bc_range(rng, dir, end)

        offset_surf_range = ti.Vector([0, 0])
        if dir == 0:  #x
            if end == 0:  #right
                offset_surf_range = ti.Vector([0, 0])
            else:
                offset_surf_range = ti.Vector([-1, 0])
        else:
            if end == 0:  #right
                offset_surf_range = ti.Vector([0, 0])
            else:
                offset_surf_range = ti.Vector([0, -1])
        for I in ti.grouped(
                ti.ndrange((range_bc_x[0], range_bc_x[1]),
                           (range_bc_y[0], range_bc_y[1]))):
            bc_I = I + offset

            if stage == -1:
                self.elem_area[bc_I] = self.elem_area[I]
                self.elem_width[bc_I] = self.elem_width[I]
            elif stage == 0:
                surf_inside_normal = self.vec_surf[I + offset_surf_range, dir]

                prim = self.q_to_primitive(self.q[I])
                rho = prim[0]
                v = ti.Vector([prim[0], prim[1]])
                e = prim[3]
                p = e * (self.gamma - 1.0)
                self.q[bc_I] = ti.Vector([rho, 0.0, 0.0, rho * e + 0.5 * 0.0])
            elif stage == 1:
                if ti.static(self.is_viscous):
                    self.gradient_v_c[bc_I] = 1.0 * self.gradient_v_c[I]
                    self.gradient_temp_c[bc_I] = -1.0 * self.gradient_temp_c[I]

    @ti.kernel
    def bc_outlet_super(self, rng0: ti.template(), rng1: ti.template(),
                        dir: ti.template(), end: ti.template(), stage: ti.i32):
        rng = ti.Vector([rng0, rng1])
        range_bc_x, range_bc_y, offset = self.calc_bc_range(rng, dir, end)
        for I in ti.grouped(
                ti.ndrange((range_bc_x[0], range_bc_x[1]),
                           (range_bc_y[0], range_bc_y[1]))):
            bc_I = I + offset
            int_I = I - offset

            if stage == -1:
                self.elem_area[bc_I] = self.elem_area[I]
                self.elem_width[bc_I] = self.elem_width[I]
            elif stage == 0:
                ## TODO: sometimes will cause back poisoning?
                self.q[bc_I] = ti.max(0.0, 2 * self.q[I] - self.q[int_I])
                # self.q[bc_I] = self.q[I]
            elif stage == 1:
                if ti.static(self.is_viscous):
                    self.gradient_v_c[bc_I] = 1.0 * self.gradient_v_c[I]
                    self.gradient_temp_c[bc_I] = 1.0 * self.gradient_temp_c[I]

    @ti.kernel
    def bc_inlet_subsonic(self, rng_x: ti.template(), rng_y: ti.template(),
                       dir: ti.template(), end: ti.template(), q0: real,
                       q1: real, q2: real, q3: real, stage: ti.i32):
        rng = ti.Vector([rng_x, rng_y])
        range_bc_x, range_bc_y, offset = self.calc_bc_range(rng, dir, end)
        for I in ti.grouped(
                ti.ndrange((range_bc_x[0], range_bc_x[1]),
                           (range_bc_y[0], range_bc_y[1]))):
            bc_I = I + offset
            # far end inlet
            if stage == -1:
                self.elem_area[bc_I] = self.elem_area[I]
                self.elem_width[bc_I] = self.elem_width[I]
            elif stage == 0:
                self.q[bc_I] = ti.Vector([q0, q1, q2, q3])
            elif stage == 1:
                if ti.static(self.is_viscous):
                    self.gradient_v_c[bc_I] = 1.0 * self.gradient_v_c[I]
                    self.gradient_temp_c[bc_I] = 1.0 * self.gradient_temp_c[I]


    @ti.kernel
    def bc_outlet_subsonic(self, rng_x: ti.template(), rng_y: ti.template(),
                       dir: ti.template(), end: ti.template(), q0: real,
                       q1: real, q2: real, q3: real, stage: ti.i32):
        rng = ti.Vector([rng_x, rng_y])
        range_bc_x, range_bc_y, offset = self.calc_bc_range(rng, dir, end)
        for I in ti.grouped(
                ti.ndrange((range_bc_x[0], range_bc_x[1]),
                           (range_bc_y[0], range_bc_y[1]))):
            bc_I = I + offset
            # far end inlet
            if stage == -1:
                self.elem_area[bc_I] = self.elem_area[I]
                self.elem_width[bc_I] = self.elem_width[I]
            elif stage == 0:
                # self.q[bc_I] = ti.Vector([q0, q1, q2, q3])
                ## TEMP: velocity points to normal side
                v = ti.Vector([q1, q2]).normalized()
                v_bc = v.dot(v) * v
                q_bc = ti.Vector([q0, v_bc[0], v_bc[1], q3])
                self.q[bc_I] = q_bc
            elif stage == 1:
                if ti.static(self.is_viscous):
                    self.gradient_v_c[bc_I] = 1.0 * self.gradient_v_c[I]
                    self.gradient_temp_c[bc_I] = 1.0 * self.gradient_temp_c[I]

    @ti.func
    def calc_bc_connection_positions(self, conn_info: ti.template()) -> ti.template():
        ## conn_info: (start index, march direction, surface direction, surface index (start/end))
        pos_start = ti.Vector([0, 0])
        direction_offset = ti.Vector([0, 0])   # bc marches in this direction from start elem
        bc_offset = ti.Vector([0, 0])          # virtual voxels indexes are pos + bc_offset for every bc cell

        index_start = conn_info[0]
        direction = conn_info[1]
        surf_ij = conn_info[2]
        surf_0n = conn_info[3]

        if (surf_ij == 0):     # i-surf
            direction_offset = ti.Vector([0, direction])
            if (surf_0n == 0): # i-0-surf
                pos_start = ti.Vector([1, index_start])
                bc_offset = ti.Vector([-1, 0])
            else:                   # i-n-surf
                pos_start = ti.Vector([self.ni, index_start])
                bc_offset = ti.Vector([1, 0])
        else:                       # j-surf
            direction_offset = ti.Vector([direction, 0])
            if (surf_0n == 0): # j-0-surf
                pos_start = ti.Vector([index_start, 1])
                bc_offset = ti.Vector([0, -1])
            else:                   # j-n-surf
                pos_start = ti.Vector([index_start, self.nj])
                bc_offset = ti.Vector([0, 1])

        return (pos_start, direction_offset, bc_offset)

    @ti.kernel
    def bc_connection(self, bc_conn: ti.template(), bc_other: ti.template(), num: ti.i32, stage: ti.i32):
        conn_pos_start, conn_direction_offset, conn_offset = self.calc_bc_connection_positions(bc_conn)
        other_pos_start, other_direction_offset, other_offset = self.calc_bc_connection_positions(bc_other)
        for i in range(num):
            bc_I = conn_pos_start + i * conn_direction_offset + conn_offset  # bc on this side
            I_other = other_pos_start + i * other_direction_offset          # real cell on other side

            if stage == -1:
                self.elem_area[bc_I] = self.elem_area[I_other]
                self.elem_width[bc_I] = self.elem_width[I_other]
            elif stage == 0:
                self.q[bc_I] = self.q[I_other]
            elif stage == 1:
                if ti.static(self.is_viscous):
                    self.gradient_v_c[bc_I] = self.gradient_v_c[I_other]
                    self.gradient_temp_c[bc_I] = self.gradient_temp_c[I_other]

    ###############
    ## Calls all bondary conditions from here
    ##
    ## bc (set quantity on virtual voxels)
    ##
    ## stage: before and in loop, there're several points to set bc values
    ##   -1: geom patch (area, elem_width)
    ##    0: q on center
    ##    1: gradient uvt on center
    def bc(self, stage):
        # local bc
        for bc in self.bc_info:
            (bc_type, rng0, rng1, direction, start_or_end, q_index) = bc
            if bc_type == 0:
                ## super inlet
                q_bc_item = self.bc_q_values[q_index]
                ## TODO: transfer q_bc directly slows down drastically?
                # q_bc = ti.Vector([q_bc_item[0], q_bc_item[1], q_bc_item[2], q_bc_item[3]])
                self.bc_inlet_super(rng0, rng1, direction, start_or_end,
                                    q_bc_item[0], q_bc_item[1], q_bc_item[2],
                                    q_bc_item[3], stage)
            elif bc_type == 1:
                ## super outlet
                self.bc_outlet_super(rng0, rng1, direction, start_or_end, stage)
            elif bc_type == 2:
                ## symmetry
                self.bc_symmetry(rng0, rng1, direction, start_or_end, stage)
            elif bc_type == 3:
                ## nosliop wall
                self.bc_wall_noslip(rng0, rng1, direction, start_or_end, stage)
            elif bc_type == 4:
                ## slip wall
                self.bc_wall_slip(rng0, rng1, direction, start_or_end, stage)
            elif bc_type == 10:
                ## subsonic inlet
                q_bc_item = self.bc_q_values[q_index]
                self.bc_inlet_subsonic(rng0, rng1, direction, start_or_end,
                                    q_bc_item[0], q_bc_item[1], q_bc_item[2],
                                    q_bc_item[3], stage)
            elif bc_type == 11:
                ## subsonic outlet
                q_bc_item = self.bc_q_values[q_index]
                self.bc_outlet_subsonic(rng0, rng1, direction, start_or_end,
                                    q_bc_item[0], q_bc_item[1], q_bc_item[2],
                                    q_bc_item[3], stage)
            else:
                raise ValueError("unknown bc type")

        # connection transfers
        for (bc_conn, bc_other, num) in self.bc_connection_info:
            self.bc_connection(bc_conn, bc_other, num, stage)


    #--------------------------------------------------------------------------
    #  Utility Functions
    #--------------------------------------------------------------------------
    #############
    ## translate (rho, rho*u, rho*v, rho*et) into primitive ones (rho, u, v, e)
    @ti.func
    def q_to_primitive(self, q: ti.template()) -> ti.template():
        rho = q[0]
        prim = ti.Vector([0.0, 0.0, 0.0, 0.0])  # (rho, u, v, e)
        if rho < 1e-10:  # this should not happen, rho < 0
            prim = ti.Vector([0.0, 0.0, 0.0, 0.0])
        else:
            rho_inv = 1.0 / rho
            ux = q[1] * rho_inv
            uy = q[2] * rho_inv
            e = ti.abs(q[3] * rho_inv - 0.5 *
                       (ux**2 + uy**2))  # TODO: abs or clamp?
            ### TODO: limit e to be > 0
            # TODO: h -> e/p/h?
            prim = ti.Vector([rho, ux, uy, e])
            ## others
            # p_on_rho = e * (self.gamma - 1.0)
            # p = p_on_rho * rho
            # a = (self.gamma * p_on_rho)**0.5
            # uu = (ux**2 + uy**2)**0.5
            # ma = uu / a
            # h = et + p/rho = e + 0.5 * uu + p / rho
            # t = Ma_far**2 * gamma * p / rho
        return prim

    #############
    ##  q to [u, v, t]
    @ti.func
    def q_to_primitive_u_t(self, q: ti.template()) -> ti.template():
        rho = q[0]
        uvt = ti.Vector([0.0, 0.0, 0.0])
        if rho < 1e-10:  # this should not happen in elems, rho < 0. Maybe in 4 corner elems (not used later)
            uvt = ti.Vector([0.0, 0.0, 0.0])
        else:
            rho_inv = 1.0 / rho
            ux = q[1] * rho_inv
            uy = q[2] * rho_inv
            e = ti.abs(q[3] * rho_inv - 0.5 *
                       (ux**2 + uy**2))  # TODO: abs or clamp?
            ### TODO: limit e to be > 0
            p_on_rho = e * (self.gamma - 1.0)
            temp = self.ma0**2 * self.gamma * p_on_rho  # = p0 * p/rho?
            uvt = ti.Vector([ux, uy, temp])
        return uvt

    @ti.func
    def util_ext_product_scalar_vec2d(self, q,
                                      vec: ti.template()) -> ti.template():
        return q * vec

    @ti.func
    def util_ext_product_vec2d_vec2d(
        self, q: ti.template(), vec: ti.template()) -> ti.template():
        v = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        for i in ti.static(range(2)):
            for j in ti.static(range(2)):
                v[i, j] = q[i] * vec[j]
        return v

    #######
    ## TODO: cannot achieve general ext product here: q is both var or vector
    # @ti.func
    # def util_ext_product(self, q: ti.template(), vec: ti.template(), w: ti.template()) -> ti.template():
    #     if ti.static(q.n == 2): # vector
    #         v = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
    #         for i in ti.static(range(2)):
    #             for j in ti.static(range(2)):
    #                 v[i, j] = q[i] * vec[j]
    #         return v
    #     else:
    #         # if ti.static(q.n == 1): CANNOT USE THIS
    #         return q * vec

    #--------------------------------------------------------------------------
    #  Convection Fluxes
    #
    #       van Leer split
    #--------------------------------------------------------------------------

    @ti.kernel
    def clear_flux(self):
        for I in ti.grouped(ti.ndrange(*self.range_elems_virtual)):
            self.flux[I] = ti.Vector([0.0, 0.0, 0.0, 0.0])

    ########
    ## caculate van Leer flux on one surface
    ## dir: 1.0/-1.0 for plus/minus split flux
    @ti.func
    def calc_van_leer_flux_split(self, q: ti.template(),
                                 vec_dir: ti.template(), dir) -> ti.template():
        prim = self.q_to_primitive(q)
        rho = prim[0]
        v = ti.Vector([prim[1], prim[2]])
        e = prim[3]
        p_on_rho = e * (self.gamma - 1.0)
        p = p_on_rho * rho
        v_across = v.dot(vec_dir)  # scalar

        # f_a = (self.gamma / (self.gamma - 1.0) * e)**0.5 # sqrt(gamma / (gamma - 1) * e) dimensionless?
        f_a = (self.gamma * p_on_rho)**0.5
        f_ma = v_across / f_a

        flux = ti.Vector([0.0, 0.0, 0.0, 0.0])

        if f_ma >= 1.0:
            # f+ = f, f- = 0
            # et = e + 0.5 * v.norm_sqr(), h = et + p/rho
            # h = (self.gamma / (self.gamma - 1.0)) * e + 0.5 * v.norm_sqr()
            # h = self.gamma * e + 0.5 * v.norm_sqr()
            h = (self.gamma /
                 (self.gamma - 1.0)) * p_on_rho + 0.5 * v.norm_sqr()
            fq = ti.Vector([
                rho * v_across, rho * v_across * v[0] + p * vec_dir[0],
                rho * v_across * v[1] + p * vec_dir[1], rho * v_across * h
            ])
            flux = 0.5 * (1.0 + dir) * fq
        elif f_ma <= -1.0:
            # h = self.gamma * e + 0.5 * v.norm_sqr()
            h = (self.gamma /
                 (self.gamma - 1.0)) * p_on_rho + 0.5 * v.norm_sqr()
            fq = ti.Vector([
                rho * v_across, rho * v_across * v[0] + p * vec_dir[0],
                rho * v_across * v[1] + p * vec_dir[1], rho * v_across * h
            ])
            flux = 0.5 * (1.0 - dir) * fq
        else:  # sub sonic
            f_mass = dir * rho * f_a / 4.0 * (f_ma + dir)**2
            u_part = (-1.0 * v_across + dir * 2.0 * f_a) / self.gamma
            flux = ti.Vector([
                f_mass, f_mass * (u_part * vec_dir[0] + v[0]),
                f_mass * (u_part * vec_dir[1] + v[1]),
                f_mass * (f_a**2 / (self.gamma - 1.0) + 0.5 * v.norm_sqr())
            ])

        return flux

    @ti.func
    def calc_van_leer_flux(
        self, ql: ti.template(), qr: ti.template(), vec_normal: ti.template()
    ) -> ti.template():
        # calculate van leer flux flowing from ql to qr, outside cell normal is vec_normal from left to right
        vec_dir = vec_normal.normalized()
        flux_plus = self.calc_van_leer_flux_split(ql, vec_dir, 1.0)
        flux_minus = self.calc_van_leer_flux_split(qr, vec_dir, -1.0)
        return (flux_plus + flux_minus)

    @ti.func
    def flux_van_leer(self):
        ## x dir to the right, flux across the same surf is positive/negative into left/right cells respectively
        for I in ti.grouped(ti.ndrange(*self.range_surfs_ij_i)):
            offset_right = ti.Vector([1, 0])
            flux = self.calc_van_leer_flux(self.q[I], self.q[I + offset_right],
                                           self.vec_surf[I, 0])
            # TODO: maybe we can check if is adding to virtual element, but bcs will update them later, no need?
            self.flux[I] += flux
            self.flux[I + offset_right] -= flux

        ## y dir to the top, flux across the same surf is positive/negative into left/right cells respectively
        for I in ti.grouped(ti.ndrange(*self.range_surfs_ij_j)):
            offset_top = ti.Vector([0, 1])
            flux = self.calc_van_leer_flux(self.q[I], self.q[I + offset_top],
                                           self.vec_surf[I, 1])
            # TODO: maybe we can check if is adding to virtual element, but bcs will update them later, no need?
            self.flux[I] += flux
            self.flux[I + offset_top] -= flux

    @ti.kernel
    def flux_advect(self):
        # SAMPLE: UPWIND
        # for i, j in ti.ndrange(*self.range_elems):
        #     self.flux[i, j] = self.q[i - 1, j] - self.q[i, j]

        # TODO: calculate ql, qr used on surface from left/right
        # now we use first-order approximation directly from left/right cell center
        if self.is_convect_calculated:
            self.flux_van_leer()

    #--------------------------------------------------------------------------
    #  Interpolations for gradients on surface/center
    #--------------------------------------------------------------------------

    @ti.kernel
    def calc_u_temp_center(self):
        for I in ti.grouped(ti.ndrange(*self.range_elems_virtual)):
            uvt = self.q_to_primitive_u_t(self.q[I])
            self.v_c[I] = ti.Vector([uvt[0], uvt[1]])
            self.temp_c[I] = uvt[2]

    @ti.func
    def interpolate_surf_by_neighbor_width(self, w_c: ti.template(),
                                           w_surf: ti.template(),
                                           I: ti.template(),
                                           I_right: ti.template(),
                                           dir: ti.template()):
        width_l = self.elem_width[I][dir]
        width_r = self.elem_width[I_right][dir]
        w_surf[I, dir] = (w_c[I] * width_r +
                          w_c[I_right] * width_l) / (width_l + width_r)

    @ti.func
    def interpolate_center_to_surf(self, w_c: ti.template(),
                                   w_surf: ti.template()):
        ## x dir interpolate to the right surf
        for I in ti.grouped(ti.ndrange(*self.range_surfs_ij_i)):
            offset_right = ti.Vector([1, 0])
            I_right = I + offset_right
            self.interpolate_surf_by_neighbor_width(w_c, w_surf, I, I_right, 0)

        ## y dir interpolate to the top surf
        for I in ti.grouped(ti.ndrange(*self.range_surfs_ij_j)):
            offset_top = ti.Vector([0, 1])
            I_top = I + offset_top
            self.interpolate_surf_by_neighbor_width(w_c, w_surf, I, I_top, 1)

    @ti.kernel
    def interpolate_u_surf(self):
        self.interpolate_center_to_surf(self.v_c, self.v_surf)

    @ti.kernel
    def interpolate_temp_surf(self):
        self.interpolate_center_to_surf(self.temp_c, self.temp_surf)

    @ti.func
    def integrate_calc_gradient_center_scalar(self, w_surf: ti.template(),
                                              w_graident_c: ti.template()):
        ## internal elems only
        for I in ti.grouped(ti.ndrange(*self.range_elems)):
            I_surf_left = I + ti.Vector([-1, 0])
            I_surf_right = I + ti.Vector([0, 0])
            I_surf_down = I + ti.Vector([-1, 0])
            I_surf_up = I + ti.Vector([0, 0])

            w_graident_c[I].fill(0.0)

            ## TODO: make this general
            # w_graident_c[I] += self.util_ext_product(w_surf[I_surf_right, 0], self.vec_surf[I_surf_right, 0], w_surf)
            # w_graident_c[I] -= self.util_ext_product(w_surf[I_surf_left, 0], self.vec_surf[I_surf_left, 0], w_surf)
            # w_graident_c[I] += self.util_ext_product(w_surf[I_surf_up, 1], self.vec_surf[I_surf_up, 1], w_surf)
            # w_graident_c[I] -= self.util_ext_product(w_surf[I_surf_down, 1], self.vec_surf[I_surf_down, 1], w_surf)
            w_graident_c[I] += self.util_ext_product_scalar_vec2d(
                w_surf[I_surf_right, 0], self.vec_surf[I_surf_right, 0])
            w_graident_c[I] -= self.util_ext_product_scalar_vec2d(
                w_surf[I_surf_left, 0], self.vec_surf[I_surf_left, 0])
            w_graident_c[I] += self.util_ext_product_scalar_vec2d(
                w_surf[I_surf_up, 1], self.vec_surf[I_surf_up, 1])
            w_graident_c[I] -= self.util_ext_product_scalar_vec2d(
                w_surf[I_surf_down, 1], self.vec_surf[I_surf_down, 1])

            w_graident_c[I] /= self.elem_area[I]

    @ti.func
    def integrate_calc_gradient_center_vec2d(self, w_surf: ti.template(),
                                             w_graident_c: ti.template()):
        ## internal elems only
        for I in ti.grouped(ti.ndrange(*self.range_elems)):
            I_surf_left = I + ti.Vector([-1, 0])
            I_surf_right = I + ti.Vector([0, 0])
            I_surf_down = I + ti.Vector([-1, 0])
            I_surf_up = I + ti.Vector([0, 0])

            w_graident_c[I].fill(0.0)
            w_graident_c[I] += self.util_ext_product_vec2d_vec2d(
                w_surf[I_surf_right, 0], self.vec_surf[I_surf_right, 0])
            w_graident_c[I] -= self.util_ext_product_vec2d_vec2d(
                w_surf[I_surf_left, 0], self.vec_surf[I_surf_left, 0])
            w_graident_c[I] += self.util_ext_product_vec2d_vec2d(
                w_surf[I_surf_up, 1], self.vec_surf[I_surf_up, 1])
            w_graident_c[I] -= self.util_ext_product_vec2d_vec2d(
                w_surf[I_surf_down, 1], self.vec_surf[I_surf_down, 1])

            w_graident_c[I] /= self.elem_area[I]

    @ti.kernel
    def integrate_calc_gradient_u_center(self):
        self.integrate_calc_gradient_center_vec2d(self.v_surf,
                                                  self.gradient_v_c)

    @ti.kernel
    def integrate_calc_gradient_temp_center(self):
        self.integrate_calc_gradient_center_scalar(self.temp_surf,
                                                   self.gradient_temp_c)

    @ti.kernel
    def interpolate_gradient_u_surf(self):
        self.interpolate_center_to_surf(self.gradient_v_c,
                                        self.gradient_v_surf)

    @ti.kernel
    def interpolate_gradient_temp_surf(self):
        self.interpolate_center_to_surf(self.gradient_temp_c,
                                        self.gradient_temp_surf)

    #--------------------------------------------------------------------------
    #  Diffusion Fluxes for viscous flow
    #--------------------------------------------------------------------------

    @ti.func
    def calc_flux_diffusion_surf(
        self, v: ti.template(), temp, gradient_v: ti.template(),
        gradient_temp: ti.template(), vec_surf: ti.template()
    ) -> ti.template():
        miu_laminar = (1.0 + self.cs_sthland) / (temp +
                                                 self.cs_sthland) * (temp**1.5)
        lambda_laminar = -2.0 / 3.0 * miu_laminar
        divergence = gradient_v[0, 0] + gradient_v[1, 1]
        identity = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        stress = miu_laminar * (gradient_v + gradient_v.transpose()) + (
            lambda_laminar * divergence) * identity

        flux_uv = stress @ vec_surf

        k = miu_laminar * self.cp0 / self.pr_laminar

        theta = stress @ v + k * gradient_temp

        return ti.Vector([0.0, flux_uv[0], flux_uv[1], theta.dot(vec_surf)])

    @ti.kernel
    def calc_flux_diffusion(self):
        for I in ti.grouped(ti.ndrange(*self.range_elems)):
            I_surf_left = I + ti.Vector([-1, 0])
            I_surf_right = I + ti.Vector([0, 0])
            I_surf_down = I + ti.Vector([-1, 0])
            I_surf_up = I + ti.Vector([0, 0])

            flux = ti.Vector([0.0, 0.0, 0.0, 0.0])
            flux += self.calc_flux_diffusion_surf(
                self.v_surf[I_surf_right, 0], self.temp_surf[I_surf_right, 0],
                self.gradient_v_surf[I_surf_right, 0],
                self.gradient_temp_surf[I_surf_right,
                                        0], self.vec_surf[I_surf_right, 0])
            flux -= self.calc_flux_diffusion_surf(
                self.v_surf[I_surf_left, 0], self.temp_surf[I_surf_left, 0],
                self.gradient_v_surf[I_surf_left, 0],
                self.gradient_temp_surf[I_surf_left,
                                        0], self.vec_surf[I_surf_left, 0])
            flux += self.calc_flux_diffusion_surf(
                self.v_surf[I_surf_up, 0], self.temp_surf[I_surf_up, 0],
                self.gradient_v_surf[I_surf_up, 0],
                self.gradient_temp_surf[I_surf_up, 0], self.vec_surf[I_surf_up,
                                                                     0])
            flux -= self.calc_flux_diffusion_surf(
                self.v_surf[I_surf_down, 0], self.temp_surf[I_surf_down, 0],
                self.gradient_v_surf[I_surf_down, 0],
                self.gradient_temp_surf[I_surf_down,
                                        0], self.vec_surf[I_surf_down, 0])

            flux /= self.re0

            self.flux[I] += flux

    ######
    ## flux for ns diffusion term (laminar flow)
    def flux_diffusion(self):
        self.calc_u_temp_center()

        self.interpolate_u_surf()
        self.interpolate_temp_surf()

        self.integrate_calc_gradient_u_center()
        self.integrate_calc_gradient_temp_center()

        self.bc(1)

        self.interpolate_gradient_u_surf()
        self.interpolate_gradient_temp_surf()

        self.calc_flux_diffusion()

    #--------------------------------------------------------------------------
    #  Time marching methods
    #
    #  Explicit
    #  Runge-Kutta 3rd
    #--------------------------------------------------------------------------

    @ti.kernel
    def time_march(self):
        ## TODO: do we need to caculate virtual voxels?
        for I in ti.grouped(ti.ndrange(*self.range_elems_virtual)):
            self.flux[I] *= self.dt / self.elem_area[I]
            self.q[I] -= self.flux[I]

    @ti.kernel
    def time_save_q(self):
        # save q to w
        for i, j in self.q:
            self.w[i, j] = self.q[i, j]

    @ti.kernel
    def time_march_rk3(self, stage: ti.i32):
        coef = 1.0

        if stage == 0:
            coef = 0.1481
        elif stage == 1:
            coef = 0.4

        for I in ti.grouped(ti.ndrange(*self.range_elems)):
            self.flux[I] *= coef * self.dt / self.elem_area[I]
            self.q[I] = self.w[I] - self.flux[I]

    ### DEBUGS
    # def debug_vector2d_point(self, q, i, j):
    #     return q[i, j]
    # def debug_q_slice(self, q, i0, i1, j0, j1):
    #     r = np.zeros((i1 - i0, j1 - j0, 4), dtype=np.float32)
    #     for i in range(i0, i1):
    #         for j in range(j0, j1):
    #             for k in range(4):
    #                 r[i - i0][j - j0][k] = q[i, j][k]
    #     print(r)
    # def debug_v_slice(self, q, i0, i1, j0, j1):
    #     r = np.zeros((i1 - i0, j1 - j0, 2), dtype=np.float32)
    #     for i in range(i0, i1):
    #         for j in range(j0, j1):
    #             for k in range(2):
    #                 r[i - i0][j - j0][k] = q[i, j][k]
    #     print(r)

    def step(self):
        # RK-3
        self.time_save_q()
        for i in range(3):
            # calc from new q
            self.bc(0)
            self.clear_flux()

            if self.is_viscous:
                self.flux_diffusion()

            self.flux_advect()

            ## DEBUG samples
            # print('debug')
            # self.debug_q_slice(self.q, self.ni, self.ni + 1, self.nj, self.nj + 1)
            # self.debug_q_slice(self.flux, self.ni, self.ni + 1, self.nj, self.nj + 1)
            # print(self.q[I[0], I[1]][0],
            #         self.q[I[0], I[1]][1],
            #         self.q[I[0], I[1]][2],
            #         self.q[I[0], I[1]][3])
            # print(self.vec_surf[I_surf[0], I_surf[1], 0][0],
            #         self.vec_surf[I_surf[0], I_surf[1], 0][1])
            self.time_march_rk3(i)

    #--------------------------------------------------------------------------
    #  Display field
    #--------------------------------------------------------------------------

    @ti.kernel
    def display_setimage(self):
        for I in ti.grouped(ti.ndrange(*self.range_elems)):
            pos = self.xc[I]
            px = ti.min(
                self.ni,
                ti.max(0, ti.cast(pos[0] / self.width * self.ni, ti.i32)))
            py = ti.min(
                self.nj,
                ti.max(0, ti.cast(pos[1] / self.height * self.nj, ti.i32)))
            # sample: u
            # self.display_img[px, py] = self.q[I][1] / self.q[I][0] / 2.0
            value = self.util_output_line_getvalue(I)
            value = (value - self.display_value_min) / (self.display_value_max - self.display_value_min)
            value = ti.max(0.0, ti.min(1.0, value))
            self.display_img[px, py] = value

    def scale_to_screen(self, p):
        return (p[0] / self.width, p[1] / self.height)

    def line_arrow(self, p, v, coef):
        return (p[0] + v[0] * coef, p[1] + v[1] * coef)

    def display_grid(self):
        np_x = self.x.to_numpy()[0:1 + self.ni, 0:1 + self.nj]
        np_x = np.reshape(np_x, ((self.ni + 1) * (self.nj + 1), 2))
        for i in range(np_x.shape[0]):
            px, py = self.scale_to_screen((np_x[i, 0], np_x[i, 1]))
            np_x[i] = [px, py]
        # node
        self.gui.circles(np_x, radius=2, color=0x999999)

    def display_surf_norm_edge(self, p0, p1, norm, display_norm=True):
        self.gui.line(self.scale_to_screen(p0),
                      self.scale_to_screen(p1),
                      radius=1,
                      color=0xaaaaaa)
        if not display_norm:
            return
        pc = (0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]))
        arrow = self.line_arrow(pc, norm, 0.4)
        self.gui.line(self.scale_to_screen(pc),
                      self.scale_to_screen(arrow),
                      radius=1,
                      color=0xffaa00)

    def display_surf_norm(self, display_norm=True):
        for i in range(*self.range_surfs_ij_i[0]):
            for j in range(*self.range_surfs_ij_i[1]):
                p0 = (self.x[i, j - 1][0], self.x[i, j - 1][1])
                p1 = (self.x[i, j][0], self.x[i, j][1])
                norm = (self.vec_surf[i, j, 0][0], self.vec_surf[i, j, 0][1])
                self.display_surf_norm_edge(p0, p1, norm, display_norm)

        for i in range(*self.range_surfs_ij_j[0]):
            for j in range(*self.range_surfs_ij_j[1]):
                p0 = (self.x[i - 1, j][0], self.x[i - 1, j][1])
                p1 = (self.x[i, j][0], self.x[i, j][1])
                norm = (self.vec_surf[i, j, 1][0], self.vec_surf[i, j, 1][1])
                self.display_surf_norm_edge(p0, p1, norm, display_norm)

    def util_vec2_to_tuple(self, v):
        return (v[0], v[1])

    def scale_value_to_color(self, c, min_value=0.0, max_value=1.0):
        v = (c - min_value) / (max_value - min_value)
        # map [0,1] ~ gray color 0~255
        color_gray = (v * 256.0) // 1
        color_gray = min(255, max(0, color_gray))
        c = color_gray * 0x010101
        return c

    def display_elem_q_raw_writeelems(self):
        n_elems_x = self.range_elems[0][1] - self.range_elems[0][0]
        n_elems_y = self.range_elems[1][1] - self.range_elems[1][0]
        n_elems = n_elems_x * n_elems_y

        self.display_pas = np.zeros((2 * n_elems, 2), dtype=np.float32)
        self.display_pbs = np.zeros((2 * n_elems, 2), dtype=np.float32)
        self.display_pcs = np.zeros((2 * n_elems, 2), dtype=np.float32)
        self.display_colors = np.zeros((2 * n_elems), dtype=np.float32)

        index = 0
        for i in range(*self.range_elems[0]):
            for j in range(*self.range_elems[1]):
                pa = self.scale_to_screen(
                    self.util_vec2_to_tuple(self.x[i - 1, j - 1]))
                pb = self.scale_to_screen(
                    self.util_vec2_to_tuple(self.x[i, j - 1]))
                pc = self.scale_to_screen(
                    self.util_vec2_to_tuple(self.x[i - 1, j]))
                pd = self.scale_to_screen(self.util_vec2_to_tuple(self.x[i,
                                                                         j]))
                index = 2 * ((i - 1) * n_elems_y + j - 1)
                self.display_pas[index] = [pa[0], pa[1]]
                self.display_pbs[index] = [pb[0], pb[1]]
                self.display_pcs[index] = [pc[0], pc[1]]
                # index += 1
                index = 2 * ((i - 1) * n_elems_y + j - 1) + 1
                self.display_pas[index] = [pb[0], pb[1]]
                self.display_pbs[index] = [pc[0], pc[1]]
                self.display_pcs[index] = [pd[0], pd[1]]
                # index += 1

    def display_elem_q_raw(self, q_index):
        n_elems_x = self.range_elems[0][1] - self.range_elems[0][0]
        n_elems_y = self.range_elems[1][1] - self.range_elems[1][0]
        n_elems = n_elems_x * n_elems_y

        index = 0
        for i in range(*self.range_elems[0]):
            for j in range(*self.range_elems[1]):
                c = self.scale_value_to_color(self.q[i, j][q_index],
                                              min_value=self.display_value_min,
                                              max_value=self.display_value_max)
                self.display_colors[index] = c
                index += 1
                self.display_colors[index] = c
                index += 1

        self.gui.triangles(self.display_pas,
                           self.display_pbs,
                           self.display_pcs,
                           color=self.display_colors)

    @ti.func
    def util_ti_scale_to_screen(self, p: ti.template()) -> ti.template():
        return p * ti.Vector([1.0 / self.width, 1.0 / self.height])

    @ti.func
    def util_ti_scale_value_to_color(self, c) -> ti.i32:
        max_value = self.display_value_max
        min_value = self.display_value_min
        v = (c - min_value) / (max_value - min_value)
        # map [0,1] ~ gray color 0~255
        color_gray = ti.min(255, ti.max(0, v * 256.0))
        value = ti.cast(color_gray, ti.i32) * 0x010101
        return value

    @ti.kernel
    def display_elem_q_writeelems(self):
        index = 0
        for I in ti.grouped(ti.ndrange(*self.range_elems)):
            Ia = I + ti.Vector([-1, -1])
            Ib = I + ti.Vector([0, -1])
            Ic = I + ti.Vector([-1, 0])
            Id = I
            index = 2 * ((I[0] - 1) * self.nj + (I[1] - 1))
            index2 = index + 1
            self.display_elems_triangle_a[
                index] = self.util_ti_scale_to_screen(self.x[Ia])
            self.display_elems_triangle_b[
                index] = self.util_ti_scale_to_screen(self.x[Ib])
            self.display_elems_triangle_c[
                index] = self.util_ti_scale_to_screen(self.x[Ic])
            self.display_elems_triangle_a[
                index2] = self.util_ti_scale_to_screen(self.x[Ib])
            self.display_elems_triangle_b[
                index2] = self.util_ti_scale_to_screen(self.x[Ic])
            self.display_elems_triangle_c[
                index2] = self.util_ti_scale_to_screen(self.x[Id])

    @ti.func
    def util_calc_ma_from_q(self, q: ti.template()) -> real:
        rho = q[0]
        rho_inv = 1.0 / rho
        u = q[1] * rho_inv
        v = q[2] * rho_inv
        u2 = u**2 + v**2
        e = q[3] * rho_inv - 0.5 * u2

        ### TODO: limit e to be > 0
        # PP(I,J,K)=0.4*U(5,I,J,K)-0.2*VV/U(1,I,J,K)

        p_on_rho = e * (self.gamma - 1.0)
        a = (self.gamma * p_on_rho)**0.5
        ma = (u2**0.5) / a

        return ma

    @ti.kernel
    def display_elem_q_writeq(self):
        index = 0
        for I in ti.grouped(ti.ndrange(*self.range_elems)):
            index = 2 * ((I[0] - 1) * self.nj + (I[1] - 1))
            index2 = index + 1
            value = self.util_output_line_getvalue(I)
            self.display_elems_q[index] = self.util_ti_scale_value_to_color(
                value)
            self.display_elems_q[index2] = self.util_ti_scale_value_to_color(
                value)

    def display_elem_q(self):
        self.display_elem_q_writeq()
        self.gui.triangles(self.display_elems_triangle_a.to_numpy(),
                           self.display_elems_triangle_b.to_numpy(),
                           self.display_elems_triangle_c.to_numpy(),
                           color=self.display_elems_q.to_numpy())

    def display_xc(self):
        np_xc = self.xc.to_numpy()[1:1 + self.ni, 1:1 + self.nj]
        np_xc = np.reshape(np_xc, (self.ni * self.nj, 2))

        for i in range(np_xc.shape[0]):
            np_xc[i, 0] /= self.width
            np_xc[i, 1] /= self.height

        ## grid centers
        self.gui.circles(np_xc, radius=1, color=0xaa0000)

    def util_rotate_vector(self, uv, deg):
        deg = deg / 180.0 * np.pi
        cd = math.cos(deg)
        sd = math.sin(deg)
        u, v = uv
        return (cd * u + sd * v, -1 * sd * u + cd * v)

    def display_draw_arrow(self, xc, uv, coef):
        p0 = self.scale_to_screen(xc)
        p1 = self.scale_to_screen(self.line_arrow(xc, uv, 0.02))
        su1 = self.util_rotate_vector(uv, 140)
        su2 = self.util_rotate_vector(uv, -140)
        p2 = self.line_arrow(p1, su1, 0.02 * 0.3)
        p3 = self.line_arrow(p1, su2, 0.02 * 0.3)
        self.gui.line(p0, p1, radius=1, color=0x669922)
        self.gui.line(p1, p2, radius=1, color=0x669922)
        self.gui.line(p1, p3, radius=1, color=0x669922)

    def display_v(self):
        for i in range(1, self.ni + 1, self.display_show_velocity_skip[0]):
            for j in range(1, self.nj + 1, self.display_show_velocity_skip[1]):
                # elem v
                u = self.q[i, j][1] / self.q[i, j][0]
                v = self.q[i, j][2] / self.q[i, j][0]
                xc = (self.xc[i, j][0], self.xc[i, j][1])
                self.display_draw_arrow(xc, (u, v), 0.01)

    def display(self):
        ### These are slow, to be removed
        # self.display_img.fill(0.0)
        # self.display_setimage()
        # self.gui.set_image(self.display_img)
        ## self.display_elem_q_raw(3)

        ## field should be displayed at the bottom, draw grids/arrows later
        if self.display_field:
            self.display_elem_q()

        ## grid and center, velocity arrows
        if self.display_show_grid:
            self.display_grid()
        if self.display_show_xc:
            self.display_xc()
        if self.display_show_velocity:
            self.display_v()
        if self.display_show_surface:
            self.display_surf_norm(self.display_show_surface_norm)

        self.gui.show()

    #--------------------------------------------------------------------------
    #  Plot values on a line
    #
    #  Needs to interpolate field to get values on the points
    #  Caches interpolation coefficients before main loop
    #--------------------------------------------------------------------------

    @ti.func
    def util_value_is_positive(self, v) -> ti.i32:
        return v > 0.0

    @ti.func
    def util_coefs_in_triangle(self, coefs: ti.template()) -> ti.i32:
        a = coefs[0]
        b = coefs[1]
        c = coefs[2]
        eps = 1e-8
        # a + b + c = 1.0
        return (self.util_value_is_positive(a + eps)
                and self.util_value_is_positive(1.0 - a + eps)
                and self.util_value_is_positive(b + eps)
                and self.util_value_is_positive(1.0 - b + eps)
                and self.util_value_is_positive(c + eps))

    @ti.func
    def triangle_interp_coef(
        self, p: ti.template(), pa: ti.template(), pb: ti.template(),
        pc: ti.template()
    ) -> ti.template():
        ## pa,pb,pc cannot be degenerated
        A = ti.Matrix([
            [1.0, 1.0, 1.0],
            [pa[0], pb[0], pc[0]],
            [pa[1], pb[1], pc[1]],
        ])
        b = ti.Vector([1.0, p[0], p[1]])
        return A.inverse() @ b

    @ti.kernel
    def display_output_line_init(self) -> ti.i32:
        ti.static_print("init output line search")

        ## calc output point coords, evenly spaced on the line
        point_start = ti.Vector(
            [self.output_line_ends[0][0], self.output_line_ends[0][1]])
        point_end = ti.Vector(
            [self.output_line_ends[1][0], self.output_line_ends[1][1]])
        dx = (point_end - point_start) / (self.output_line_num_points - 1)
        for i in range(self.output_line_num_points):
            self.output_line_points[i] = point_start + dx * i

        ####
        ## calcs and caches interpolation pivot point and coefs for quicker calculation in simulation
        ## search for which center grid (grid consisted of inner center points) the point is in
        ## notice we do not support point outside grid for now (even outside center (0, 0) at the node grid edge)
        ## one-time naive search for now, may use DFS or search tree for better performance
        ##
        ## bilinear inverse is quadratic, not effective, use triangle interpolation
        all_found = 1  # keep notice of the parallel reduction problem
        for i in range(self.output_line_num_points):
            found = False
            point = self.output_line_points[i]
            for I in ti.grouped(ti.ndrange((1, self.ni), (1, self.nj))):
                pa = self.xc[I]
                pb = self.xc[I + ti.Vector([1, 0])]
                pc = self.xc[I + ti.Vector([0, 1])]
                pd = self.xc[I + ti.Vector([1, 1])]
                # coefs = ti.Vector([0.0, 0.0, 0.0])
                coefs = self.triangle_interp_coef(point, pa, pb, pd)
                if self.util_coefs_in_triangle(coefs):
                    self.output_line_interp_index[i] = I
                    self.output_line_interp_coef[i] = ti.Vector(
                        [coefs[0], coefs[1], 0.0, coefs[2]])
                    found = True
                    continue
                coefs = self.triangle_interp_coef(point, pa, pc, pd)
                if self.util_coefs_in_triangle(coefs):
                    self.output_line_interp_index[i] = I
                    self.output_line_interp_coef[i] = ti.Vector(
                        [coefs[0], 0.0, coefs[1], coefs[2]])
                    found = True
                    continue
            if not found:
                # cannot print in kernel func
                # print(f'Output line point ({I[0]}, {I[1]}): ({point[0]}, {point[1]}) not in region.')
                # dd[None] = i
                # dd3[None] = point
                all_found *= 0  # reduction style
                break
        return all_found == 1

    ###
    # @ti.kernel
    # def debug_output_line(self):
    #     point = self.output_line_points[10] # (0.1120603010058403 0.4000000059604645)
    #     I = ti.Vector([22, 50]) # xc[I] = (0.10750000178813934 0.3959999978542328)
    #     pa = self.xc[I]
    #     pb = self.xc[I + ti.Vector([1, 0])]
    #     pc = self.xc[I + ti.Vector([0, 1])]
    #     pd = self.xc[I + ti.Vector([1, 1])]
    #     coefs = self.triangle_interp_coef(point, pa, pc, pd)
    #     dd3[None] = pd
    #     dd4[None] = coefs

    ####
    ## @param hidden: self.output_line_var index, to determine which quantity to save
    @ti.func
    def util_output_line_getvalue(self, I: ti.template()) -> real:
        # get value on point xc[I]
        q = self.q[I]
        v = 0.0
        if ti.static(self.output_line_var == 0):  # rho
            v = q[0]
        if ti.static(self.output_line_var == 1):  # u
            v = q[1] / q[0]
        if ti.static(self.output_line_var == 2):  # v
            v = q[2] / q[0]
        if ti.static(self.output_line_var == 3):  # et
            v = q[3] / q[0]
        if ti.static(self.output_line_var == 4):  # u.norm
            v = (q[1]**2 + q[2]**2)**0.5 / q[0]
        if ti.static(self.output_line_var == 5):  # p
            prim = self.q_to_primitive(q)
            rho = prim[0]
            e = prim[3]
            p_on_rho = e * (self.gamma - 1.0)
            p = p_on_rho * rho
            v = p
        if ti.static(self.output_line_var == 6):  # a
            prim = self.q_to_primitive(q)
            rho = prim[0]
            e = prim[3]
            p_on_rho = e * (self.gamma - 1.0)
            a = (self.gamma * p_on_rho)**0.5
            v = a
        if ti.static(self.output_line_var == 7):  # ma
            prim = self.q_to_primitive(q)
            rho = prim[0]
            uu = (prim[1]**2 + prim[2]**2)**0.5
            e = prim[3]
            p_on_rho = e * (self.gamma - 1.0)
            a = (self.gamma * p_on_rho)**0.5
            v = uu / a
        ## TODO: T, etc.
        return v

    @ti.kernel
    def display_output_line_save(self):
        ## calc and save output var on the line
        values = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for i in range(self.output_line_num_points):
            # va, vb, vc, vd
            Ia = self.output_line_interp_index[i]
            Ib = Ia + ti.Vector([1, 0])
            Ic = Ia + ti.Vector([0, 1])
            Id = Ia + ti.Vector([1, 1])
            coefs = self.output_line_interp_coef[i]  # vector 4
            values = ti.Vector([
                self.util_output_line_getvalue(Ia),
                self.util_output_line_getvalue(Ib),
                self.util_output_line_getvalue(Ic),
                self.util_output_line_getvalue(Id)
            ])
            self.output_line_q[i] = values.dot(coefs)

    def display_output_line(self):
        # calc vars
        self.display_output_line_save()

        # output to console
        # print("var on line:", self.output_line_q.to_numpy()[:])

        # plot var
        np_x = self.output_line_points.to_numpy()[:, self.output_line_plot_var]
        np_var = self.output_line_q.to_numpy()
        plt.clf()
        # plt.plot(np_x, np_var)
        ax = plt.gca()
        ax.plot(np_x, np_var, label='Ma')
        ax.legend()
        ax.grid(True)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        plt.show()
        plt.pause(0.001)

    #--------------------------------------------------------------------------
    #  Main loop
    #--------------------------------------------------------------------------

    def run(self):
        self.gui = ti.GUI("2D FVM Supersonic",
                          res=self.gui_size,
                          background_color=0x4f9297)

        self.init()

        ## init for field display
        if self.display_field:
            # self.display_elem_q_raw_writeelems()
            self.display_elem_q_writeelems()

        ## init for output line
        if self.output_line:
            all_points_ok = self.display_output_line_init()
            if not all_points_ok:
                print("Found some output point not in region, stops.")
                # # print(dd[None])
                exit(0)
            # init plot
            plt.ion()
            self.plot_fig = plt.figure()
            plt.axis([0, 1.0, 0, 3.0])

        pause = False
        while self.gui.running:
            for e in self.gui.get_events(ti.GUI.PRESS):
                if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                    self.gui.running = False
                elif e.key in [ti.GUI.SPACE]:
                    pause = not pause

            if pause:
                ## TODO: this will cause problems?
                # # time.sleep(1)
                if self.display_field:
                    self.display()
                continue

            ## simulation step
            for step in range(self.display_steps):
                self.step()
                self.t += self.dt

            ## TODO: more useful information to console
            print(f't: {self.t:.03f}')
            if self.display_field:
                self.display()
            if self.output_line:
                self.display_output_line()
