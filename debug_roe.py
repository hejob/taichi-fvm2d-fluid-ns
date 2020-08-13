import taichi as ti

from solver.Solver import Solver


width=400
height=200
ni=200
nj=100

gamma = 1.4
ma0 = 2.0
re0 = 1.225 * (ma0 * 343) * 1.0 / (1.81e-5)
p0 = 1.0 / gamma / ma0 / ma0
e0 = p0 / (gamma - 1.0) + 0.5 * 1.0

solver = Solver(
    width=width,
    height=height,
    ni=ni,
    nj=nj,
    ma0=ma0,
    dt=1e-5,
    is_viscous=True,
    temp0_raw=273,
    re0=re0,
    gui_size=(400, 320),
    display_field=True,
    display_value_min=0.0,
    display_value_max=3.0,
    output_line=True,
    output_line_ends=((0.1, 0.4), (0.9, 0.4)),
    output_line_num_points=200,
    output_line_var=7,  # Mach number. 0~7: rho/u/v/et/uu/p/a/ma
    output_line_plot_var=0)  # output along x-axis on plot

solver.is_debug = True

@ti.kernel
def debug_roe(ql: ti.template(), qr: ti.template()):
	vec = ti.Vector([1.0, 0.0])
	print('ql', ql)
	print('qr', qr)
	print('ruvpahl', solver.q_to_primitive_ruvpah(ql))
	print('ruvpahr', solver.q_to_primitive_ruvpah(qr))
	print('')
	qroe = solver.calc_roe_flux(ql, qr, vec)
	print('')
	print('flux', qroe)

@ti.kernel
def debug_vanleer(ql: ti.template(), qr: ti.template()):
	vec = ti.Vector([1.0, 0.0])
	print('.......van leer......')
	q = solver.calc_van_leer_flux(ql, qr, vec)
	print('flux', q)
	print('.......van leer ends......')
print('')

print('------x even------')
ql = ti.Vector([1.0, 1.0, 0.0, 1.0 * e0])
qr = ti.Vector([1.0, 1.0, 0.0, 1.0 * e0])
qroe = debug_roe(ql, qr)
qroe = debug_vanleer(ql, qr)
print('')

print('------x reverse------')
ql = ti.Vector([1.0, -1.0, 0.0, 1.0 * e0])
qr = ti.Vector([1.0, -1.0, 0.0, 1.0 * e0])
qroe = debug_roe(ql, qr)
qroe = debug_vanleer(ql, qr)
print('')

print('------y 0------')
ql = ti.Vector([1.0, 0.0, 1.0, 1.0 * e0])
qr = ti.Vector([1.0, 0.0, 1.0, 1.0 * e0])
qroe = debug_roe(ql, qr)
qroe = debug_vanleer(ql, qr)
print('')

print('------y reverse------')
ql = ti.Vector([1.0, 0.0, -1.0, 1.0 * e0])
qr = ti.Vector([1.0, 0.0, -1.0, 1.0 * e0])
qroe = debug_roe(ql, qr)
qroe = debug_vanleer(ql, qr)
print('')

print('------xy 45deg------')
ql = ti.Vector([1.0, 0.5**0.5, 0.5**0.5, 1.0 * e0])
qr = ti.Vector([1.0, 0.5**0.5, 0.5**0.5, 1.0 * e0])
qroe = debug_roe(ql, qr)
qroe = debug_vanleer(ql, qr)
print('')

print('------x shock------')
ql = ti.Vector([1.0, 3.0, 0.0, 1.0 * e0 + 0.5 * 1.0 * (3.0**2 - 1.0**2)])
qr = ti.Vector([1.0, 2.0, 0.0, 1.0 * e0 + 0.5 * 1.0 * (2.0**2 - 1.0**2)])
qroe = debug_roe(ql, qr)
qroe = debug_vanleer(ql, qr)
print('')


print('------y shear------')
ql = ti.Vector([1.0, 0.0, 3.0, 1.0 * e0 + 0.5 * 1.0 * (3.0**2 - 1.0**2)])
qr = ti.Vector([1.0, 0.0, 2.0, 1.0 * e0 + 0.5 * 1.0 * (2.0**2 - 1.0**2)])
qroe = debug_roe(ql, qr)
qroe = debug_vanleer(ql, qr)
print('')

print('ends')

