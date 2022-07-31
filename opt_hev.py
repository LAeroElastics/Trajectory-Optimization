import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from gekko import GEKKO

d2r = 3.14 / 180.0
r2d = 1. / d2r

# Model setup
m = GEKKO(remote=True)
m._path = './'

# Initialize time array
num_nodes = 21
time_array = np.linspace(0, 1, num_nodes)
m.time = time_array

# Set final point
index_final = np.zeros(num_nodes)
index_final[-1] = 1.0
final = m.Param(value=index_final)

# Set solver options
m.options.SENSITIVITY = 0
m.options.CSV_WRITE = 2
m.options.CSV_READ = 2
m.options.SCALING = 1
m.options.NODES = 50
m.options.SOLVER = 3
m.options.IMODE = 6
m.options.MAX_ITER = 5000
m.options.MV_TYPE = 0
m.options.DIAGLEVEL = 0
m.options.REDUCE = 0

# Initial conditions
estimate_terminate_time = 1000.
max_time = 10000.
min_time = 0.
max_range = 100. * 5280.
min_range = 0. * 5280.
max_altitude = 60000.
min_altitude = 0.
max_velocity = 430.
min_velocity = 120.
max_alpha = 20. * d2r
min_alpha = -20. * d2r
max_gamma = 45. * d2r
min_gamma = -45. * d2r

# Set constants
g0 = 32.174
s_ref = 179.
m_empty = 4000.
m_fuel_init = 500 * 2.2046
m_pgs = 200 * 2.2046
m_bat = 200 * 2.2046
pgs_sfc = 0.2
bat_spec_energy = 0.4
bat_spec_power = 200.0
eta = 0.9
cd0 = 0.02
cla = 4.0
d2r = np.pi / 180.
r2d = 180. / np.pi

# Control variables
alpha = m.MV(value=0., lb=min_alpha, ub=max_alpha)
pw_pgs = m.MV(value=200., lb=0., ub=500. * 1.341)
pw_bat = m.MV(value=200., lb=0., ub=bat_spec_power * m_bat / 2.2046)
tf = m.FV(value=estimate_terminate_time, lb=min_time, ub=max_time)
alpha.STATUS = 1
pw_pgs.STATUS = 1
pw_bat.STATUS = 1
tf.STATUS = 1

# State variables
xa = m.Var(value=0., lb=min_range, ub=max_range)
ya = m.Var(value=0., lb=min_altitude, ub=max_altitude)
vm = m.Var(value=200., lb=min_velocity, ub=max_velocity)
gamma = m.Var(value=0., lb=min_gamma, ub=max_gamma)
m_fuel_used = m.Var(value=0., lb=0., ub=m_fuel_init)
e_bat_used = m.Var(value=0., lb=0., ub=bat_spec_energy * m_bat / 2.2046)

# Intermediate
m_total = m.Intermediate(m_empty + m_pgs + m_bat + (m_fuel_init - m_fuel_used))
e_bat_init = bat_spec_energy * m_bat / 2.2046
soc_bat = m.Intermediate((e_bat_init - e_bat_used) / e_bat_init * 100.)

# Aerodynamics values
cl = m.Intermediate(cla * alpha * r2d)
cd = m.Intermediate(cd0 + 0.04 * cl ** 2.)
rho = m.Intermediate(1.22 * m.exp(-0.000091 * ya + -1.88e-9 * ya ** 2) * 0.062428)
q = m.Intermediate(0.5 * rho * vm ** 2. / g0)

# Forces
f_thrust = m.Intermediate((pw_pgs + pw_bat) * 1.341 * 550. * eta / vm * 1000.)
f_drag = m.Intermediate(q * cl * s_ref)
f_lift = m.Intermediate(q * cd * s_ref)

# Aircraft Dynamics
m.Equation(xa.dt() == tf * vm * m.cos(gamma))
m.Equation(ya.dt() == tf * vm * m.sin(gamma))
m.Equation(vm.dt() * m_total == tf * (f_thrust - f_drag - m_total * g0 * m.sin(gamma)))
m.Equation(gamma.dt() * m_total * vm == tf * (f_lift - m_total * g0 * m.cos(gamma)))

# PGS & BAT dynamics
m.Equation(m_fuel_used.dt() == tf * pgs_sfc * pw_pgs * 1.341 / 3600.)
m.Equation(e_bat_used.dt() == tf * pw_bat / 3600.)

# General  constraints
m.Equation(e_bat_init - e_bat_used >= 0.)
m.Equation(m_fuel_init - m_fuel_used >= 0.)

# Initial  conditions
m.fix_initial(xa, val=0.)
m.fix_initial(ya, val=50.)
# m.fix_initial(gamma, val=0.)

# Final conditions
# m.fix_final(xa, val=100.)
m.fix_final(ya, val=50.)

# Object functions
m.Minimize(tf)
# m.Maximize(final * xa)

# Solve(Pre)
#m.options.COLDSTART = 2
#m.solve(disp=True)

# Solve
m.options.COLDSTART = 0
m.options.TIME_SHIFT = 0

m.solve(disp=True)
m.open_folder(inifeasibility.txt)

tm = m.time * tf.value[0]

i = 1
plt.figure(i)
plt.plot(xa.value, ya.value, 'b-', label=r'$y$')
plt.xlabel("x, ft")
plt.ylabel("y, ft")
plt.grid()
plt.show()

plt.figure(i + 1)
plt.plot(tm, vm.value, 'b-', label=r'$v$')
plt.xlabel("t, s")
plt.ylabel("v, ft/s")
plt.grid()
plt.show()

plt.figure(i + 1)
plt.plot(tm, f_thrust.value, 'b-', label=r'$v$')
plt.xlabel("t, s")
plt.ylabel("ft, lb")
plt.grid()
plt.show()

plt.figure(i + 1)
plt.plot(tm, gamma.value, 'b-', label=r'$gam$')
plt.xlabel("t, s")
plt.ylabel("gamma, rad")
plt.grid()
plt.show()

plt.figure(i + 1)
plt.plot(tm, alpha.value, 'b-', label=r'$gam$')
plt.xlabel("t, s")
plt.ylabel("alpha, rad")
plt.grid()
plt.show()
