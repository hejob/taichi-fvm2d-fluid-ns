# ==============================================
#  structured 2D compressible FVM fluid solver
#  Drawer class with gui and plot outputs
#
#  @author hejob.moyase@gmail.com
# ==============================================

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import math

real = ti.f32
### init in multil block entrance
# ti.init(arch=ti.cpu, default_fp=real, kernel_profiler=True)

@ti.data_oriented
class Drawer:
    def __init__(
            self,
            width,
            height,
            ### multiblock info
            n_blocks,
            block_dimensions,
            solvers,
            ### output: realtime field display
            gui_size=(400, 400),
            display_field=True,
            display_value_min=0.0,
            display_value_max=1.0,
            display_color_map=0,
            output_line=False,
            output_line_ends=((), ()),
            output_line_num_points=200,
            output_line_var=7,  # Mach number. 0~7: rho/u/v/et/uu/p/a/ma
            output_line_plot_var=0,
            display_gif_files=False):

        self.gamma = 1.4

        self.width = width
        self.height = height

        self.n_blocks = n_blocks
        self.block_dimensions = block_dimensions
        self.solvers = solvers

        ## realtime outputs
        ##  display simulation field
        self.gui_size = gui_size
        self.display_field = display_field
        self.display_value_min = display_value_min
        self.display_value_max = display_value_max
        self.display_color_map = display_color_map
        ## switches, can be set later
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

        self.output_line_label = ('Rho', 'u', 'v', 'et', 'unorm', 'p', 'a', 'Ma')[self.output_line_var]

        ## monitor points to console
        self.output_monitor_points = []

        ## output gif for animation
        self.display_gif_files = display_gif_files

        ## taichi vars
        self.init_allocations()


    ###############################
    # Taichi tensors allocations
    def init_allocations(self):

        #===================== OUTPUT: Field ==========================
        if self.display_field:
            ### TODO: replace 2d array by accumulated array of all block's triangle into 1d array
            # element triangles and fields to display for every block
            self.display_elems_triangle_a = [ti.Vector(2, dt=real) for _ in range(self.n_blocks)]
            self.display_elems_triangle_b = [ti.Vector(2, dt=real) for _ in range(self.n_blocks)]
            self.display_elems_triangle_c = [ti.Vector(2, dt=real) for _ in range(self.n_blocks)]
            self.display_elems_q = [ti.field(dtype=ti.i32) for _ in range(self.n_blocks)]

            for block in range(self.n_blocks):
                ni, nj = self.block_dimensions[block]
                ti.root.dense(ti.i, (2 * ni * nj)).place(
                        self.display_elems_triangle_a[block],
                        self.display_elems_triangle_b[block],
                        self.display_elems_triangle_c[block],
                        self.display_elems_q[block]
                    )

        #===================== OUTPUT: Plot ==========================
        if (self.output_line):
            self.output_line_q = ti.field(dtype=real,
                                          shape=(self.output_line_num_points))
            self.output_line_points = ti.Vector.field(
                2, dtype=real, shape=(self.output_line_num_points))
            self.output_line_block_index = ti.field(
                dtype=real, shape=(self.output_line_num_points))
            self.output_line_interp_index = ti.Vector.field(
                2, dtype=ti.i32, shape=(self.output_line_num_points))
            self.output_line_interp_coef = ti.Vector.field(
                4, dtype=real,
                shape=(self.output_line_num_points
                       ))  # coef for interpolation of 4 points in an rectangle

    ########################
    # Set extra display
    def set_display_options(self,
                            display_color_map=0,
                            display_show_grid=False,
                            display_show_xc=False,
                            display_show_velocity=False,
                            display_show_velocity_skip=(4, 4),
                            display_show_surface=False,
                            display_show_surface_norm=False,
                            output_monitor_points=[],
                            display_gif_files=False):
        self.display_color_map = display_color_map
        self.display_show_grid = display_show_grid
        self.display_show_xc = display_show_xc
        self.display_show_velocity = display_show_velocity
        self.display_show_velocity_skip = display_show_velocity_skip
        self.display_show_surface = display_show_surface
        self.display_show_surface_norm = display_show_surface_norm
        self.output_monitor_points = output_monitor_points
        self.display_gif_files = display_gif_files

    ########################
    # Set GUI handle
    # TODO: manage gui in this class
    def set_gui(self, gui):
        self.gui = gui


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


    #--------------------------------------------------------------------------
    #  Display field
    #--------------------------------------------------------------------------
    # @ti.kernel
    # def display_setimage(self):
    #     for I in ti.grouped(ti.ndrange(*self.range_elems)):
    #         pos = self.xc[I]
    #         px = ti.min(
    #             self.ni,
    #             ti.max(0, ti.cast(pos[0] / self.width * self.ni, ti.i32)))
    #         py = ti.min(
    #             self.nj,
    #             ti.max(0, ti.cast(pos[1] / self.height * self.nj, ti.i32)))
    #         # sample: u
    #         # self.display_img[px, py] = self.q[I][1] / self.q[I][0] / 2.0
    #         value = self.util_output_line_getvalue(I)
    #         value = (value - self.display_value_min) / (self.display_value_max - self.display_value_min)
    #         value = ti.max(0.0, ti.min(1.0, value))
    #         self.display_img[px, py] = value

    def scale_to_screen(self, p):
        return (p[0] / self.width, p[1] / self.height)

    def line_arrow(self, p, v, coef):
        return (p[0] + v[0] * coef, p[1] + v[1] * coef)

    ### needs multiblock update
    def display_grid(self):
        for solver in self.solvers:
            np_x = solver.x.to_numpy()[0:1 + solver.ni, 0:1 + solver.nj]
            np_x = np.reshape(np_x, ((solver.ni + 1) * (solver.nj + 1), 2))
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

    ### needs multiblock update
    def display_surf_norm(self, display_norm=True):
        for solver in self.solvers:
            for i in range(*solver.range_surfs_ij_i[0]):
                for j in range(*solver.range_surfs_ij_i[1]):
                    p0 = (solver.x[i, j - 1][0], solver.x[i, j - 1][1])
                    p1 = (solver.x[i, j][0], solver.x[i, j][1])
                    norm = (solver.vec_surf[i, j, 0][0], solver.vec_surf[i, j, 0][1])
                    self.display_surf_norm_edge(p0, p1, norm, display_norm)

            for i in range(*solver.range_surfs_ij_j[0]):
                for j in range(*solver.range_surfs_ij_j[1]):
                    p0 = (solver.x[i - 1, j][0], solver.x[i - 1, j][1])
                    p1 = (solver.x[i, j][0], solver.x[i, j][1])
                    norm = (solver.vec_surf[i, j, 1][0], solver.vec_surf[i, j, 1][1])
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

    ### needs multiblock update
    # def display_elem_q_raw_writeelems(self):
    #     n_elems_x = self.range_elems[0][1] - self.range_elems[0][0]
    #     n_elems_y = self.range_elems[1][1] - self.range_elems[1][0]
    #     n_elems = n_elems_x * n_elems_y

    #     self.display_pas = np.zeros((2 * n_elems, 2), dtype=np.float32)
    #     self.display_pbs = np.zeros((2 * n_elems, 2), dtype=np.float32)
    #     self.display_pcs = np.zeros((2 * n_elems, 2), dtype=np.float32)
    #     self.display_colors = np.zeros((2 * n_elems), dtype=np.float32)

    #     index = 0
    #     for i in range(*self.range_elems[0]):
    #         for j in range(*self.range_elems[1]):
    #             pa = self.scale_to_screen(
    #                 self.util_vec2_to_tuple(self.x[i - 1, j - 1]))
    #             pb = self.scale_to_screen(
    #                 self.util_vec2_to_tuple(self.x[i, j - 1]))
    #             pc = self.scale_to_screen(
    #                 self.util_vec2_to_tuple(self.x[i - 1, j]))
    #             pd = self.scale_to_screen(self.util_vec2_to_tuple(self.x[i,
    #                                                                      j]))
    #             index = 2 * ((i - 1) * n_elems_y + j - 1)
    #             self.display_pas[index] = [pa[0], pa[1]]
    #             self.display_pbs[index] = [pb[0], pb[1]]
    #             self.display_pcs[index] = [pc[0], pc[1]]
    #             # index += 1
    #             index = 2 * ((i - 1) * n_elems_y + j - 1) + 1
    #             self.display_pas[index] = [pb[0], pb[1]]
    #             self.display_pbs[index] = [pc[0], pc[1]]
    #             self.display_pcs[index] = [pd[0], pd[1]]
    #             # index += 1

    # def display_elem_q_raw(self, q_index):
    #     n_elems_x = self.range_elems[0][1] - self.range_elems[0][0]
    #     n_elems_y = self.range_elems[1][1] - self.range_elems[1][0]
    #     n_elems = n_elems_x * n_elems_y

    #     index = 0
    #     for i in range(*self.range_elems[0]):
    #         for j in range(*self.range_elems[1]):
    #             c = self.scale_value_to_color(self.q[i, j][q_index],
    #                                           min_value=self.display_value_min,
    #                                           max_value=self.display_value_max)
    #             self.display_colors[index] = c
    #             index += 1
    #             self.display_colors[index] = c
    #             index += 1

    #     self.gui.triangles(self.display_pas,
    #                        self.display_pbs,
    #                        self.display_pcs,
    #                        color=self.display_colors)

    @ti.func
    def util_ti_scale_to_screen(self, p: ti.template()) -> ti.template():
        return p * ti.Vector([1.0 / self.width, 1.0 / self.height])

    @ti.func
    def util_color_map_value(self, c0, c1, v) -> ti.i32:
        vl = ti.max(0.0, ti.min(1.0, v))
        r = c1 - c0
        d = ti.cast(c0 + r * vl, ti.i32)
        return d

    @ti.func
    def util_ti_scale_value_to_color(self, c) -> ti.i32:
        max_value = self.display_value_max
        min_value = self.display_value_min
        v = (c - min_value) / (max_value - min_value)
        
        value = 0
        if ti.static(self.display_color_map == 0):
            # map [0,1] ~ gray color 0~255
            color_gray = ti.min(255, ti.max(0, v * 256.0))
            value = ti.cast(color_gray, ti.i32) * 0x010101
        else:
            r = self.util_color_map_value(0x01, 0xff, v)
            g = self.util_color_map_value(0x20, 0xbb, v)
            b = self.util_color_map_value(0xff, 0x00, v)
            value = r * 0x010000 + g * 0x0100 + b
        return value

    @ti.kernel
    def display_elem_q_writeelems_block(self,
                        block: ti.template(), ### THIS MUST BE template not ti.32 to allow array index to function
                        x: ti.template(), ### TODO: maybe block alone is enough
                        i0: ti.i32,
                        i1: ti.i32,
                        j0: ti.i32,
                        j1: ti.i32,
                        ni: ti.i32,
                        nj: ti.i32):
        index = 0
        for I in ti.grouped(ti.ndrange((i0, i1), (j0, j1))):
            Ia = I + ti.Vector([-1, -1])
            Ib = I + ti.Vector([0, -1])
            Ic = I + ti.Vector([-1, 0])
            Id = I
            index = 2 * ((I[0] - 1) * nj + (I[1] - 1))
            index2 = index + 1
            self.display_elems_triangle_a[block][
                index] = self.util_ti_scale_to_screen(x[Ia])
            self.display_elems_triangle_b[block][
                index] = self.util_ti_scale_to_screen(x[Ib])
            self.display_elems_triangle_c[block][
                index] = self.util_ti_scale_to_screen(x[Ic])
            self.display_elems_triangle_a[block][
                index2] = self.util_ti_scale_to_screen(x[Ib])
            self.display_elems_triangle_b[block][
                index2] = self.util_ti_scale_to_screen(x[Ic])
            self.display_elems_triangle_c[block][
                index2] = self.util_ti_scale_to_screen(x[Id])

    def display_elem_q_writeelems(self):
        for block in range(self.n_blocks):
            solver = self.solvers[block]
            ((i0, i1), (j0, j1)) = solver.range_elems
            self.display_elem_q_writeelems_block(block, solver.x, i0, i1, j0, j1, solver.ni, solver.nj)


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
    def display_elem_q_writeq_block(self, block: ti.template()):  ### block should be template to work
        index = 0
        for I in ti.grouped(ti.ndrange(*self.solvers[block].range_elems)):
            index = 2 * ((I[0] - 1) * self.solvers[block].nj + (I[1] - 1))
            index2 = index + 1
            value = self.util_output_line_getvalue(self.solvers[block].q, I)
            self.display_elems_q[block][index] = self.util_ti_scale_value_to_color(
                value)
            self.display_elems_q[block][index2] = self.util_ti_scale_value_to_color(
                value)

    def display_elem_q(self):
        for block in range(self.n_blocks):
            self.display_elem_q_writeq_block(block)
            self.gui.triangles(self.display_elems_triangle_a[block].to_numpy(),
                           self.display_elems_triangle_b[block].to_numpy(),
                           self.display_elems_triangle_c[block].to_numpy(),
                           color=self.display_elems_q[block].to_numpy())

    def display_xc(self):
        for solver in self.solvers:
            np_xc = solver.xc.to_numpy()[1:1 + solver.ni, 1:1 + solver.nj]
            np_xc = np.reshape(np_xc, (solver.ni * solver.nj, 2))

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
        p1 = self.scale_to_screen(self.line_arrow(xc, uv, coef))
        su1 = self.util_rotate_vector(uv, 140)
        su2 = self.util_rotate_vector(uv, -140)
        p2 = self.line_arrow(p1, su1, coef * 0.1)
        p3 = self.line_arrow(p1, su2, coef * 0.1)
        self.gui.line(p0, p1, radius=1, color=0xcc4400)
        self.gui.line(p1, p2, radius=1, color=0xcc4400)
        self.gui.line(p1, p3, radius=1, color=0xcc4400)

    def display_v(self):
        for solver in self.solvers:
            for i in range(1, solver.ni + 1, self.display_show_velocity_skip[0]):
                for j in range(1, solver.nj + 1, self.display_show_velocity_skip[1]):
                    # elem v
                    u = solver.q[i, j][1] / solver.q[i, j][0]
                    v = solver.q[i, j][2] / solver.q[i, j][0]
                    xc = (solver.xc[i, j][0], solver.xc[i, j][1])
                    self.display_draw_arrow(xc, (u, v), 0.05)

    @ti.pyfunc
    def util_vector_to_plain_array(self, v: ti.template()) -> ti.template():
        ### used .mat of Proxy object, PROBLEM?
        return list([v[i] for i in ti.static(range(v.mat.n))])

    def util_matrix_to_plain_array(self, v: ti.template()) -> ti.template():
        ### used .mat of Proxy object, PROBLEM?
        return list([[v[n, m] for m in ti.static(range(v.mat.m))] for n in ti.static(range(v.mat.n))])

    ### point: (block, i, j)
    def print_output_monitor_point(self, point):
        block, i, j = point
        solver = self.solvers[block]

        print("===========================")
        print("Monitor point:", (i, j))

        print(" GEOM ",
            self.util_vector_to_plain_array(solver.xc[i, j]),
            solver.elem_area[i, j],
            self.util_vector_to_plain_array(solver.elem_width[i, j]),
        )

        print(" Q ", self.util_vector_to_plain_array(solver.q[i, j]))
        print(" flux ", self.util_vector_to_plain_array(solver.flux[i, j]))
        if solver.is_viscous:
            print(" v_c ", self.util_vector_to_plain_array(solver.v_c[i, j]))
            print(" temp_c ", solver.temp_c[i, j])
            print(" gradient_v_c ", self.util_matrix_to_plain_array(solver.gradient_v_c[i, j]))
            print(" gradient_temp_c ", self.util_vector_to_plain_array(solver.gradient_temp_c[i, j]))

        if solver.is_viscous:
            surfs = [
                ('LEFT', i - 1, j, 0),
                ('RIGHT', i, j, 0),
                ('DOWN', i, j - 1, 1),
                ('UP', i, j, 1),
            ]
            for tag, si, sj, sdir in surfs:
                print(" ", tag)
                print("    normal", self.util_vector_to_plain_array(solver.vec_surf[si, sj, sdir]))
                print("    v_surf", self.util_vector_to_plain_array(solver.v_surf[si, sj, sdir]))
                print("    temp_surf", solver.temp_surf[si, sj, sdir])
                print("    gradient_v_surf", self.util_matrix_to_plain_array(solver.gradient_v_surf[si, sj, sdir]))
                print("    gradient_temp_surf", self.util_vector_to_plain_array(solver.gradient_temp_surf[si, sj, sdir]))

    def init_display(self):
        # ## init for field display
        if self.display_field:
            # self.display_elem_q_raw_writeelems()
            self.display_elem_q_writeelems()

        # ## init for output line
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

    def display(self, step_index):
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

        ## output quantities on monitor points to console
        for point in self.output_monitor_points:
            self.print_output_monitor_point(point)

        ## output gif
        if self.display_gif_files:
            gif_name = f'img_{step_index:03}.png'
            ti.imwrite(self.gui.get_image(), gif_name)

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
    def display_output_line_init_points(self) -> ti.i32:
        ti.static_print("init output line search")

        ## calc output point coords, evenly spaced on the line
        point_start = ti.Vector(
            [self.output_line_ends[0][0], self.output_line_ends[0][1]])
        point_end = ti.Vector(
            [self.output_line_ends[1][0], self.output_line_ends[1][1]])
        dx = (point_end - point_start) / (self.output_line_num_points - 1)
        for i in range(self.output_line_num_points):
            self.output_line_points[i] = point_start + dx * i

    # @ti.func
    @ti.kernel
    def display_output_line_init_block(self, i: ti.i32, block: ti.template(), point: ti.template()) -> ti.i32:
        solver = self.solvers[block]
        notFound = 1
        for I in ti.grouped(ti.ndrange((1, solver.ni), (1, solver.nj))):
            pa = solver.xc[I]
            pb = solver.xc[I + ti.Vector([1, 0])]
            pc = solver.xc[I + ti.Vector([0, 1])]
            pd = solver.xc[I + ti.Vector([1, 1])]
            # coefs = ti.Vector([0.0, 0.0, 0.0])
            coefs = self.triangle_interp_coef(point, pa, pb, pd)
            if self.util_coefs_in_triangle(coefs):
                self.output_line_block_index[i] = block
                self.output_line_interp_index[i] = I
                self.output_line_interp_coef[i] = ti.Vector(
                    [coefs[0], coefs[1], 0.0, coefs[2]])
                notFound *= 0
                # print('found', notFound, block, i, I)
                continue

        for I in ti.grouped(ti.ndrange((1, solver.ni), (1, solver.nj))):
            pa = solver.xc[I]
            pb = solver.xc[I + ti.Vector([1, 0])]
            pc = solver.xc[I + ti.Vector([0, 1])]
            pd = solver.xc[I + ti.Vector([1, 1])]
            coefs = self.triangle_interp_coef(point, pa, pc, pd)
            if self.util_coefs_in_triangle(coefs):
                self.output_line_block_index[i] = block
                self.output_line_interp_index[i] = I
                self.output_line_interp_coef[i] = ti.Vector(
                    [coefs[0], 0.0, coefs[1], coefs[2]])
                notFound *= 0
                # print('found', notFound, block, i, I)
                continue

        return notFound

    # @ti.kernel
    def display_output_line_init(self) -> ti.i32:
        self.display_output_line_init_points()

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
            # point = self.output_line_points[i]
            point = ti.Vector([self.output_line_points[i][0], self.output_line_points[i][1]])
            for block in range(self.n_blocks):
                notFound = self.display_output_line_init_block(i, block, point)
                found = (notFound == 0)
                if found:
                    break
            if not found:
                # cannot print in kernel func
                # print(f'Output line point ({I[0]}, {I[1]}): ({point[0]}, {point[1]}) not in region.')
                # dd[None] = i
                # dd3[None] = point
                all_found *= 0  # reduction style
                break
            print('point', i, point, found)
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
    def util_output_line_getvalue(self, q_var: ti.template(), I: ti.template()) -> real:  ## block should be template to function
        # get value on point xc[I]
        q = q_var[I]
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
    def display_output_line_save_block(self, i: ti.i32, solver: ti.template()):
        # va, vb, vc, vd
        Ia = self.output_line_interp_index[i]
        Ib = Ia + ti.Vector([1, 0])
        Ic = Ia + ti.Vector([0, 1])
        Id = Ia + ti.Vector([1, 1])
        coefs = self.output_line_interp_coef[i]  # vector 4
        values = ti.Vector([
            self.util_output_line_getvalue(solver.q, Ia),
            self.util_output_line_getvalue(solver.q, Ib),
            self.util_output_line_getvalue(solver.q, Ic),
            self.util_output_line_getvalue(solver.q, Id)
        ])
        self.output_line_q[i] = values.dot(coefs)

    # @ti.kernel
    def display_output_line_save(self):
        ## calc and save output var on the line
        for i in range(self.output_line_num_points):
            # va, vb, vc, vd
            block = int(self.output_line_block_index[i])
            self.display_output_line_save_block(i, self.solvers[block])

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
        ax.plot(np_x, np_var, label=self.output_line_label)
        ax.legend()
        ax.grid(True)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        plt.show()
        plt.pause(0.001)
