# ==============================================
#  structured 2D compressible FVM fluid solver
#  Solver class (one-block)
#
#  @author hejob.moyase@gmail.com
# ==============================================

import taichi as ti
import math

real = ti.f32
### init in multil block entrance
# ti.init(arch=ti.cpu, default_fp=real, kernel_profiler=True)


###############################
### Main Block Solver
###############################
@ti.data_oriented
class BlockSolver:

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
            is_dual_time=False,
            # method
            convect_method=1,  # 0~1, van Leer/Roe/Roe-RHLL
            # viscous
            is_viscous=False,
            temp0_raw=273,
            re0=1e5):
        ### TODO: read from input files, but are there perforamance issues?

        self.is_debug = False

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
        self.is_dual_time=is_dual_time
        self.dt = dt
        self.t = 0.0

        ## convect flux
        self.convect_method = convect_method

        ## viscous Navier-Stokes simulation properties
        self.is_convect_calculated = True  # a switch to skip convect flux only for debug
        self.is_viscous = is_viscous
        if self.is_viscous:
            self.temp0_raw = temp0_raw
            self.re0 = re0  # re = rho * u * L / miu  = u * L / niu
            self.cs_sthland = 117.0 / self.temp0_raw
            self.pr_laminar = 0.72  # laminar prantl number
            self.cp0 = self.gamma / (self.gamma - 1.0) * self.p0

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
        ## connections are inter-blocked, govened by Multiblock Solver

        ## custom function injections for various custom simulations
        self.custom_init_func = None

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
        ### dual time
        if self.is_dual_time:
            self.w0 = ti.Vector(4, dt=real)
            self.wsub = ti.Vector(4, dt=real)
            self.elem_nodes.place(self.w0, self.wsub,
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
    # Set custom simulations
    def set_custom_simulations(self, custom_init_func):
        self.custom_init_func = custom_init_func

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
            # self.q[I] = ti.Vector([
            #     1.0,
            #     1.0 * 0.0,
            #     1.0 * 0.0,
            #     1.0 * self.e0,
            # ])

    @ti.kernel
    def init_w(self):
        for I in ti.grouped(ti.ndrange(*self.range_elems)):
            self.w[I] = self.q[I]
            if ti.static(self.is_dual_time):
                self.w0[I] = self.q[I]
                self.wsub[I] = self.q[I]


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

        if not self.custom_init_func is None:
            self.custom_init_func(self)

        self.init_w()

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

    # offset to calc bc surf's index
    # NOTICE this is surf's plus direction, not always points to the bc side (positive/negative)
    # surf_inside_normal = self.vec_surf[I + offset_surf_range, dir]
    @ti.func
    def calc_bc_surf_range(self, dir, end) -> ti.template():
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
        return offset_surf_range

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
        offset_surf_range = self.calc_bc_surf_range(dir, end)
        for I in ti.grouped(
                ti.ndrange((range_bc_x[0], range_bc_x[1]),
                           (range_bc_y[0], range_bc_y[1]))):
            bc_I = I + offset
            surf_inside_normal = self.vec_surf[I + offset_surf_range, dir]
            if ti.static(end == 1):
                surf_inside_normal *= -1.0
            surf_inside_dir = surf_inside_normal.normalized()

            if stage == -1:
                self.elem_area[bc_I] = self.elem_area[I]
                self.elem_width[bc_I] = self.elem_width[I]
            elif stage == 0:
                # self.q[bc_I] = self.q[I]
                rho_uv = ti.Vector([self.q[I][1], self.q[I][2]])
                rho_reflect = rho_uv - 2.0 * rho_uv.dot(surf_inside_dir) * surf_inside_dir
                self.q[bc_I] = ti.Vector([
                                    self.q[I][0],
                                    rho_reflect[0], rho_reflect[1],
                                    self.q[I][3],
                                ])
            elif stage == 1:
                if ti.static(self.is_viscous):
                    ### TODO: gradient v is not right, should check normal/t directions
                    self.gradient_v_c[bc_I] = 1.0 * self.gradient_v_c[I]
                    # self.gradient_temp_c[bc_I] = -1.0 * self.gradient_temp_c[I]
                    self.gradient_temp_c[bc_I] = self.gradient_temp_c[I] * self.gradient_temp_c[I].dot(surf_inside_dir) * surf_inside_dir

    @ti.kernel
    def bc_wall_slip(self, rng_x: ti.template(), rng_y: ti.template(),
                     dir: ti.template(), end: ti.template(), stage: ti.i32):
        rng = ti.Vector([rng_x, rng_y])
        range_bc_x, range_bc_y, offset = self.calc_bc_range(rng, dir, end)

        offset_surf_range = self.calc_bc_surf_range(dir, end)
        for I in ti.grouped(
                ti.ndrange((range_bc_x[0], range_bc_x[1]),
                           (range_bc_y[0], range_bc_y[1]))):
            bc_I = I + offset
            surf_inside_normal = self.vec_surf[I + offset_surf_range, dir]
            if ti.static(end == 1):
                surf_inside_normal *= -1.0
            surf_inside_dir = surf_inside_normal.normalized()

            if stage == -1:
                self.elem_area[bc_I] = self.elem_area[I]
                self.elem_width[bc_I] = self.elem_width[I]
            elif stage == 0:
                ### this is basically the same with symmetry?
                # prim = self.q_to_primitive(self.q[I])
                # rho = prim[0]
                # u = ti.Vector([prim[0], prim[1]])
                # e = prim[3]
                # p = e * (self.gamma - 1.0)
                # u_slip = u - u.dot(surf_inside_dir) * surf_inside_dir * 2.0
                # self.q[bc_I] = ti.Vector([
                #     rho, rho * u_slip[0], rho * u_slip[1],
                #     rho * e + 0.5 * u_slip.norm_sqr()
                # ])
                rho_uv = ti.Vector([self.q[I][1], self.q[I][2]])
                rho_reflect = rho_uv - 2.0 * rho_uv.dot(surf_inside_dir) * surf_inside_dir
                self.q[bc_I] = ti.Vector([
                                    self.q[I][0],
                                    rho_reflect[0], rho_reflect[1],
                                    self.q[I][3],
                                ])                
            elif stage == 1:
                if ti.static(self.is_viscous):
                    self.gradient_v_c[bc_I] = 1.0 * self.gradient_v_c[I]
                    # self.gradient_temp_c[bc_I] = -1.0 * self.gradient_temp_c[I]
                    self.gradient_temp_c[bc_I] = self.gradient_temp_c[I] * self.gradient_temp_c[I].dot(surf_inside_dir) * surf_inside_dir

    @ti.kernel
    def bc_wall_noslip(self, rng_x: ti.template(), rng_y: ti.template(),
                       dir: ti.template(), end: ti.template(), stage: ti.i32):
        rng = ti.Vector([rng_x, rng_y])
        range_bc_x, range_bc_y, offset = self.calc_bc_range(rng, dir, end)

        offset_surf_range = self.calc_bc_surf_range(dir, end)
        for I in ti.grouped(
                ti.ndrange((range_bc_x[0], range_bc_x[1]),
                           (range_bc_y[0], range_bc_y[1]))):
            bc_I = I + offset
            surf_inside_normal = self.vec_surf[I + offset_surf_range, dir]
            if ti.static(end == 1):
                surf_inside_normal *= -1.0
            surf_inside_dir = surf_inside_normal.normalized()

            if stage == -1:
                self.elem_area[bc_I] = self.elem_area[I]
                self.elem_width[bc_I] = self.elem_width[I]
            elif stage == 0:
                prim = self.q_to_primitive(self.q[I])
                rho = prim[0]
                v = ti.Vector([prim[0], prim[1]])
                e = prim[3]
                p = e * (self.gamma - 1.0)
                self.q[bc_I] = ti.Vector([rho, 0.0, 0.0, rho * e + 0.5 * 0.0])
            elif stage == 1:
                if ti.static(self.is_viscous):
                    self.gradient_v_c[bc_I] = 1.0 * self.gradient_v_c[I]
                    # self.gradient_temp_c[bc_I] = -1.0 * self.gradient_temp_c[I]
                    self.gradient_temp_c[bc_I] = self.gradient_temp_c[I] * self.gradient_temp_c[I].dot(surf_inside_dir) * surf_inside_dir

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
                    # self.gradient_v_c[bc_I] = 1.0 * self.gradient_v_c[I]
                    # self.gradient_temp_c[bc_I] = 1.0 * self.gradient_temp_c[I]
                    self.gradient_v_c[bc_I] = 2.0 * self.gradient_v_c[I] - self.gradient_v_c[int_I]
                    self.gradient_temp_c[bc_I] = 2.0 * self.gradient_temp_c[I] - self.gradient_temp_c[int_I]

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

        # connection transfers are govened in MultiBlock class in a separate phase

    #--------------------------------------------------------------------------
    #  Utility Functions
    #
    #  TODO: move to a common module
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

    @ti.func
    def q_to_primitive_ruvpah(self, q: ti.template()) -> ti.template():
        rho = q[0]
        prim = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # (rho, u, v, e)
        if rho < 1e-10:  # this should not happen, rho < 0
            prim = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            rho_inv = 1.0 / rho
            ux = q[1] * rho_inv
            uy = q[2] * rho_inv
            # e = ti.abs(q[3] * rho_inv - 0.5 *
                       # (ux**2 + uy**2))  # TODO: abs or clamp?
            # p_on_rho = e * (self.gamma - 1.0)
            # p = p_on_rho * rho
            p = ti.abs(q[3] - 0.5 * rho * (ux**2 + uy**2)) * ti.static(self.gamma - 1.0)
            a = (ti.static(self.gamma) * p * rho_inv)**0.5
            h = ti.static(self.gamma / (self.gamma - 1.0)) * p * rho_inv + 0.5 * (ux**2 + uy**2)

            prim = ti.Vector([rho, ux, uy, p, a, h])
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
    def calc_roe_flux(
        self, ql: ti.template(), qr: ti.template(), vec_normal: ti.template()
    ) -> ti.template():
        ## calculate roe flux flowing from ql to qr, outside cell normal is vec_normal from left to right
        vec_dir = vec_normal.normalized()

        R = self.p0  # R=Cp-Cv
        prim_l = self.q_to_primitive_ruvpah(ql)
        prim_r = self.q_to_primitive_ruvpah(qr)

        rho_l = prim_l[0]
        rho_r = prim_r[0]
        v_l = ti.Vector([prim_l[1], prim_l[2]])
        v_r = ti.Vector([prim_r[1], prim_r[2]])
        p_l = prim_l[3]
        p_r = prim_r[3]
        a_l = prim_l[4]
        a_r = prim_r[4]
        h_l = prim_l[5]
        h_r = prim_r[5]


        r = (rho_r / rho_l)**0.5

        rho_m = (rho_r * rho_l)**0.5
        v_m = (v_l + v_r * r) / (1.0 + r)
        p_m = (p_l + p_r * r) / (1.0 + r)
        # a_m = (ti.static(self.gamma) * p_m / rho_m)**0.5
        # h_m = ti.static(self.gamma / (self.gamma - 1.0)) * p_m / rho_m + 0.5 * v_m.norm_sqr()
        h_m = (h_l + h_r * r) / (1.0 + r)
        a_m = (ti.static(self.gamma - 1.0) * (h_m - 0.5 * v_m.norm_sqr()))**0.5

        v_normal = v_m.dot(vec_dir)
        a_normal = a_m

        dv_normal = (v_r - v_l).dot(vec_dir)
        v_limit = 0.5 * (ti.abs(dv_normal) + ti.abs(a_r - a_l))
        eigen1 = ti.abs(v_normal)
        eigen2 = ti.max(v_limit, ti.abs(v_normal + a_normal))
        eigen3 = ti.max(v_limit, ti.abs(v_normal - a_normal))

        a_star = 0.5 * (eigen2 + eigen3)
        ma_star = 0.5 * (eigen2 - eigen3) / a_normal

        drho = rho_r - rho_l
        dv = v_r - v_l
        dp = p_r - p_l
        drho_v = rho_m * dv + drho * v_m
        drho_e = dp / ti.static(self.gamma - 1.0) + 0.5 * v_m.norm_sqr() * drho + rho_m * (v_m.dot(dv))

        dv_roe = ma_star * dv_normal + (a_star - eigen1) * dp / rho_m / (a_m**2)
        dp_roe = ma_star * dp + (a_star - eigen1) * rho_m * dv_normal

        drho1 = eigen1 * drho + dv_roe * rho_m
        drho_v1 = eigen1 * drho_v + rho_m * dv_roe * v_m + dp_roe * vec_dir
        drho_e1 = eigen1 * drho_e + rho_m * dv_roe * h_m + dp_roe * v_normal

        v_normal_l = v_l.dot(vec_dir)
        v_normal_r = v_r.dot(vec_dir)
        rho_v_normal_l = rho_l * v_normal_l
        rho_v_normal_r = rho_r * v_normal_r

        drho_flux = -0.5 * (rho_v_normal_l + rho_v_normal_r - drho1)
        dv_flux = -0.5 * (rho_v_normal_l * v_l + rho_v_normal_r * v_r + (p_l + p_r) * vec_dir - drho_v1)
        de_flux = -0.5 * (rho_v_normal_l * h_l + rho_v_normal_r * h_r - drho_e1)

        # debug
        if ti.static(self.is_debug):
            print('ms', r, rho_m, v_m, p_m, a_m, h_m)
            print('normal', v_normal, a_normal, dv_normal)
            print('eigen', v_limit, eigen1, eigen2, eigen3, a_star, ma_star)
            print('dq', drho, dv, dp, drho_v, drho_e)
            print('droe', dv_roe, dp_roe)
            print('d1', drho1, drho_v1, drho_e1)
            print('lrnormal', v_normal_l, v_normal_r, rho_v_normal_l, rho_v_normal_r)
            print('flux', drho_flux, dv_flux, de_flux)

        ### TODO: minus 1.0?
        return -1.0 * ti.Vector([drho_flux, dv_flux[0], dv_flux[1], de_flux])

    @ti.func
    def calc_roe_eigen_modified(self, eig) -> real:
        # return eig
        limit = 0.2
        v = eig # eig >= 0
        if eig < limit:
            v = (eig**2 + limit**2) * 0.5 / limit
        return v

    @ti.func
    def calc_roe_rhll_flux(
        self, ql: ti.template(), qr: ti.template(), vec_normal: ti.template()
    ) -> ti.template():
        ## calculate roe flux flowing from ql to qr, outside cell normal is vec_normal from left to right
        vec_dir = vec_normal.normalized()

        ## flux according to surf l/r quantities
        # R = self.p0  # R=Cp-Cv
        prim_l = self.q_to_primitive_ruvpah(ql)
        prim_r = self.q_to_primitive_ruvpah(qr)

        rho_l = prim_l[0]
        v_l = ti.Vector([prim_l[1], prim_l[2]])
        p_l = prim_l[3]
        a_l = prim_l[4]
        h_l = prim_l[5]
        rho_r = prim_r[0]
        v_r = ti.Vector([prim_r[1], prim_r[2]])
        p_r = prim_r[3]
        a_r = prim_r[4]
        h_r = prim_r[5]

        v_normal_l = v_l.dot(vec_dir)
        v_normal_r = v_r.dot(vec_dir)
        rho_v_normal_l = rho_l * v_normal_l
        rho_v_normal_r = rho_r * v_normal_r

        d_flux_l = ti.Vector([
                        rho_v_normal_l,
                        rho_v_normal_l * v_l[0] + p_l * vec_dir[0],
                        rho_v_normal_l * v_l[1] + p_l * vec_dir[1],
                        rho_v_normal_l * h_l
                    ])
        d_flux_r = ti.Vector([
                        rho_v_normal_r,
                        rho_v_normal_r * v_r[0] + p_r * vec_dir[0],
                        rho_v_normal_r * v_r[1] + p_r * vec_dir[1],
                        rho_v_normal_r * h_r
                    ])
        ### original roe
        # drho_flux = 0.5 * (rho_v_normal_l + rho_v_normal_r)
        # dv_flux = 0.5 * (rho_v_normal_l * v_l + rho_v_normal_r * v_r + (p_l + p_r) * vec_dir)
        # de_flux = 0.5 * (rho_v_normal_l * h_l + rho_v_normal_r * h_r)

        ## roe averages of l/r
        r = (rho_r / rho_l)**0.5
        rho_m = (rho_r * rho_l)**0.5
        v_m = (v_l + v_r * r) / (1.0 + r)
        p_m = (p_l + p_r * r) / (1.0 + r)
        h_m = (h_l + h_r * r) / (1.0 + r)
        a_m = (ti.static(self.gamma - 1.0) * (h_m - 0.5 * v_m.norm_sqr()))**0.5

        ## HLL average
        v_m_normal = v_m.dot(vec_dir)
        s_l = ti.min(v_normal_l - a_l, v_m_normal - a_m)
        s_r = ti.max(v_normal_r + a_r, v_m_normal + a_m)
        s_r_plus  = ti.max(0.0, s_r)
        s_l_minus = ti.min(0.0, s_l)
        d_flux_hhl = (s_r_plus * d_flux_l - s_l_minus * d_flux_r) / (s_r_plus - s_l_minus)

        ## split in 2 directions of roe/hll
        drho = rho_r - rho_l
        dv = v_r - v_l
        dp = p_r - p_l

        # n1 - across shock or parallel to shear
        vec_dir1 = vec_dir
        if (dv.norm() > 1e-6):
            vec_dir1 = dv.normalized()
        vec_dir2 = ti.Vector([-1.0 * vec_dir[1], vec_dir[0]]) # rotate 90 degrees
        alpha1 = vec_dir.dot(vec_dir1)
        alpha2 = vec_dir.dot(vec_dir2)

        # needs to be on the same side to normal dir
        if (alpha1 < 0.0):
            alpha1 *= -1.0
            vec_dir1 *= -1.0
        if (alpha2 < 0.0):
            alpha2 *= -1.0
            vec_dir2 *= -1.0



        # vec_dir1 = vec_dir
        # vec_dir2 = ti.Vector([-1.0 * vec_dir[1], vec_dir[0]]) # rotate 90 degrees
        # alpha1 = 1.0
        # alpha2 = 0.0





        ## 4 eigen values in n2 direction
        vec_dir2_tang = ti.Vector([-1.0 * vec_dir2[1], vec_dir2[0]])  # tang normal 2, may be negative to n1
        v_m_normal2 = v_m.dot(vec_dir2)
        v_m_tang2   = v_m.dot(vec_dir2_tang)

        eig1 = ti.abs(v_m_normal2 - a_m)
        eig2 = ti.abs(v_m_normal2)
        eig3 = ti.abs(v_m_normal2 + a_m)
        eig4 = eig2

        eig1_limit = self.calc_roe_eigen_modified(eig1)
        eig2_limit = self.calc_roe_eigen_modified(eig2)
        eig3_limit = self.calc_roe_eigen_modified(eig3)
        eig4_limit = self.calc_roe_eigen_modified(eig4)

        ## rhll coef
        coef_rhll_dsrlinv = 1.0 / (s_r_plus - s_l_minus)
        coef_rhll1 = alpha2 * (s_r_plus + s_l_minus) * coef_rhll_dsrlinv
        coef_rhll2 = 2.0 * alpha1 * s_r_plus * s_l_minus * coef_rhll_dsrlinv

        s_rhll1 = alpha2 * eig1_limit - coef_rhll1 * eig1 - coef_rhll2
        s_rhll2 = alpha2 * eig2_limit - coef_rhll1 * eig2 - coef_rhll2
        s_rhll3 = alpha2 * eig3_limit - coef_rhll1 * eig3 - coef_rhll2
        s_rhll4 = alpha2 * eig4_limit - coef_rhll1 * eig4 - coef_rhll2

        dv_normal2 = dv.dot(vec_dir2)
        dv_tang2   = dv.dot(vec_dir2_tang)
        w1 = (dp - rho_m * a_m * dv_normal2) * 0.5 / (a_m**2)
        w2 = drho - dp / (a_m**2)
        w3 = (dp + rho_m * a_m * dv_normal2) * 0.5 / (a_m**2)
        w4 = rho_m * dv_tang2

        ## roe matrix column in direction 2
        R1 = ti.Vector([
                        1.0,
                        v_m[0] - a_m * vec_dir2[0],
                        v_m[1] - a_m * vec_dir2[1],
                        h_m - v_m_normal2 * a_m
                    ])
        R2 = ti.Vector([
                        1.0,
                        v_m[0],
                        v_m[1],
                        0.5 * v_m.norm_sqr()
                    ])
        R3 = ti.Vector([
                        1.0,
                        v_m[0] + a_m * vec_dir2[0],
                        v_m[1] + a_m * vec_dir2[1],
                        h_m + v_m_normal2 * a_m
                    ])
        R4 = ti.Vector([
                        0.0,
                        -1.0 * vec_dir2[1],
                        vec_dir2[0],
                        v_m_tang2
                    ])

        d_flux_rhll_dissipation = (s_rhll1 * w1 * R1 +
                                s_rhll2 * w2 * R2 +
                                s_rhll3 * w3 * R3 +
                                s_rhll4 * w4 * R4)

        d_flux = d_flux_hhl - 0.5 * d_flux_rhll_dissipation

        # debug
        if ti.static(self.is_debug):
            print('ms', r, rho_m, v_m, p_m, a_m, h_m)
            print('normal', v_normal, a_normal, dv_normal)
            print('eigen', eig1, eig2, eig3, eig4)
            print('lrnormal', v_normal_l, v_normal_r, rho_v_normal_l, rho_v_normal_r)
            print('flux', d_flux_l, d_flux_r, d_flux_hhl, d_flux_rhll_dissipation, d_flux)

        ### TODO: minus 1.0?
        return d_flux # no -1.0* ?


    @ti.func
    def calc_flux_advect(self):
        flux = ti.Vector([0.0, 0.0, 0.0, 0.0])
        ## x dir to the right, flux across the same surf is positive/negative into left/right cells respectively
        for I in ti.grouped(ti.ndrange(*self.range_surfs_ij_i)):
            offset_right = ti.Vector([1, 0])
            if ti.static(self.convect_method == 0):  # van Leer
                flux = self.calc_van_leer_flux(self.q[I],
                                               self.q[I + offset_right],
                                               self.vec_surf[I, 0])
            elif ti.static(self.convect_method == 1): # roe modified (TODO: validation)
                flux = self.calc_roe_flux(self.q[I], self.q[I + offset_right],
                                          self.vec_surf[I, 0])
            else: # 2, Roe-RHLL
                flux = self.calc_roe_rhll_flux(self.q[I], self.q[I + offset_right],
                                                self.vec_surf[I, 0])
            # TODO: maybe we can check if is adding to virtual element, but bcs will update them later, no need?
            self.flux[I] += flux
            self.flux[I + offset_right] -= flux

        ## y dir to the top, flux across the same surf is positive/negative into left/right cells respectively
        for I in ti.grouped(ti.ndrange(*self.range_surfs_ij_j)):
            offset_top = ti.Vector([0, 1])
            if ti.static(self.convect_method == 0):  # van Leer
                flux = self.calc_van_leer_flux(self.q[I],
                                               self.q[I + offset_top],
                                               self.vec_surf[I, 1])
            if ti.static(self.convect_method == 1):  # roe modified
                flux = self.calc_roe_flux(self.q[I], self.q[I + offset_top],
                                          self.vec_surf[I, 1])
            else:
                flux = self.calc_roe_rhll_flux(self.q[I], self.q[I + offset_top],
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
            self.calc_flux_advect()

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
    # def flux_diffusion(self):
    #     self.calc_u_temp_center()

    #     self.interpolate_u_surf()
    #     self.interpolate_temp_surf()

    #     self.integrate_calc_gradient_u_center()
    #     self.integrate_calc_gradient_temp_center()

    #     self.bc(1)

    #     self.interpolate_gradient_u_surf()
    #     self.interpolate_gradient_temp_surf()

    #     self.calc_flux_diffusion()


    def flux_diffusion_init(self):
        self.calc_u_temp_center()

        self.interpolate_u_surf()
        self.interpolate_temp_surf()

        self.integrate_calc_gradient_u_center()
        self.integrate_calc_gradient_temp_center()


    def flux_diffusion_calc(self):
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

    def step_oneblock(self):
        # RK-3
        self.time_save_q()
        for i in range(5):
            # calc from new q
            self.bc(0)
            self.clear_flux()

            if self.is_viscous:
                # self.flux_diffusion()
                self.flux_diffusion_init()
                self.bc(1)
                self.flux_diffusion_calc()

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


    ### in multi-block solver, bc and interconnection bc calls are inserted between loops
    ### step() is rearranged in multi-block class
    ### TODO: better methods?


    @ti.kernel
    def time_save_q_dual(self):
        # save q to w, w0
        for i, j in self.q:
            self.w0[i, j] = self.w[i, j]
            self.w[i, j] = self.q[i, j]

    @ti.kernel
    def time_save_q_dual_sub(self):
        # save q to w, w0
        for i, j in self.q:
            self.wsub[i, j] = self.q[i, j]

    @ti.kernel
    def time_march_rk3_dual(self, stage: ti.i32):
        coef = 1.0

        if stage == 0:
            coef = 0.1481
        elif stage == 1:
            coef = 0.4

        dt_sub = 0.1 * self.dt
        cdt_sub = 1.0 / (1.0 + 3.0 / 2.0 / self.dt * coef * dt_sub)

        for I in ti.grouped(ti.ndrange(*self.range_elems)):
            dq = 1.0 / self.dt * self.elem_area[I] * 3.0 / 2.0 * self.w[I]
            # dq = 1.0 / self.dt * self.elem_area[I] * (2.0 * self.w[I] - 0.5 * self.w0[I])
            self.flux[I] *= coef * dt_sub / self.elem_area[I] * cdt_sub
            self.q[I] = (1.0 + 3.0 / 2.0 / self.dt * self.elem_area[I]) * self.wsub[I] - self.flux[I] - dq

    @ti.kernel
    def time_march_rk3_dual_last(self):
        coef = 1.0

        for I in ti.grouped(ti.ndrange(*self.range_elems)):
            self.flux[I] *= coef * self.dt / self.elem_area[I]
            self.q[I] = self.wsub[I] - self.flux[I]

