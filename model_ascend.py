import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from gekko import GEKKO

d2r = 3.14 / 180.0
r2d = 1. / d2r

m = GEKKO(remote=True)
m._path = './'

nt = 51
t_array = np.linspace(0, 1, nt)
m.time = t_array

m.options.SENSITIVITY = 0

m.options.CSV_WRITE = 2
m.options.CSV_READ = 2

print(m._path)

m.options.SCALING = 0
m.options.NODES = 50
m.options.SOLVER = 3
m.options.IMODE = 6
m.options.MAX_ITER = 5000
m.options.MV_TYPE = 0
m.options.DIAGLEVEL = 0
m.options.REDUCE = 0

g = 9.8
S = 0.05
m0 = 600
mp0 = 550
tsp = 5
Isp = 300

tf = m.CV(value=200, lb=0.)
tf.STATUS = 1

alpha = m.MV(value=0.0, lb=-20 * d2r, ub=20 * d2r)
alpha.STATUS = 1

x = m.Var(value=0., lb=0.)
y = m.Var(value=0., lb=0.)
v = m.Var(value=0., lb=0.)  # , ub=1000.)

gam = m.Var(value=89.9 * d2r, lb=-180. * d2r, ub=180. * d2r)
mass = m.Var(value=m0)#, lb=m0-mp0, ub=m0)

tave = mp0 * Isp * g / tsp
step = [0 if z > tsp else tave for z in t_array * tf.value]
ft = m.Param(value=step)

rho = m.Intermediate( 1.22 * m.exp(-0.000091 * y + -1.88e-9 * y**2 ))
q = m.Intermediate(0.5 * rho * v ** 2.)

cd0 = m.Var(value=0.02)
cdf = m.Var(value=0.5)
cna = m.Var(value=0.2)

cs = m.Intermediate(-0.00117 * y + 340.288)
mn = m.Var(value=0.0)
m.Equation(mn == v / cs)

t_mn = [0.3, 0.5, 0.9, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
t_cd0 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
t_cdf = [0.5, 0.6, 0.7, 0.8, 0.65, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
t_cna = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

m.cspline(mn, cd0, t_mn, t_cd0, True)
m.cspline(mn, cdf, t_mn, t_cdf, True)
m.cspline(mn, cna, t_mn, t_cdf, True)

ca = m.Intermediate(cd0 + cdf * 0.1)
cn = m.Intermediate(cna * 10. * alpha)
fa = m.Intermediate(q * ca * S)
fn = m.Intermediate(q * cn * S)

m.Equation(x.dt() == tf * v * m.cos(gam))
m.Equation(y.dt() == tf * v * m.sin(gam))

fg_x = m.Intermediate(g * m.cos(gam))
fg_y = m.Intermediate(g * m.sin(gam))

m.Equation(v.dt() * mass == tf * ft - tf * fa - tf * mass * fg_y)
m.Equation(gam.dt() * mass * v == tf * fn - tf * mass * fg_x)
m.Equation(mass.dt() * g * Isp == -tf * ft)

m.fix_initial(mass, val=m0)
m.fix_initial(x, val=0.0)
m.fix_initial(y, val=0.0)

#m.fix_final(x, val=300000.)
m.fix_final(y, val=10000)

m.fix_initial(v, val=0)
#m.fix_final(v,val=295.)

m.fix_initial(gam, val=90.0 * d2r)
#m.fix_initial(alpha, val=0.)

#m.fix(gam, pos=len(m.time)-1, val=45. * d2r)
# m.fix(m.alpha, pos=len(m.time)-1, val=0.)

m.Minimize(tf)
#m.Obj(-y)
"""
m.options.COLDSTART = 2
m.solve(disp=True)

m.options.COLDSTART = 0
m.options.TIME_SHIFT = 0
"""
m.solve(disp=True)

tm = m.time * tf.value[0]

i = 1
plt.figure(i)
plt.plot(x.value, y.value, 'b-', label=r'$y$')
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.grid()
plt.show()

plt.figure(i + 1)
plt.plot(tm, v.value, 'b-', label=r'$v$')
plt.xlabel("t[s]")
plt.ylabel("v[m/s]")
plt.grid()
plt.show()

plt.figure(i + 1)
plt.plot(tm, ft.value, 'b-', label=r'$v$')
plt.xlabel("t[s]")
plt.ylabel("ft[N]")
plt.grid()
plt.show()

plt.figure(i + 1)
plt.plot(tm, gam.value, 'b-', label=r'$gam$')
plt.xlabel("t[s]")
plt.ylabel("gam[rad]")
plt.grid()
plt.show()

plt.figure(i + 1)
plt.plot(tm, alpha.value, 'b-', label=r'$gam$')
plt.xlabel("t[s]")
plt.ylabel("alpha[rad]")
plt.grid()
plt.show()

plt.figure(i + 1)
plt.plot(v.value, y.value, 'b-', label=r'$v$')
plt.xlabel("v[m/s]")
plt.ylabel("altitude[m]")
plt.grid()
plt.show()

fig = plt.figure(i + 1)
ax1 = fig.add_subplot(611)
ax1.plot(tm, x.value, 'b-', label=r'x')
ax1.set_xlabel("time[s]")
ax1.set_ylabel("pos_x[m]")

ax2 = fig.add_subplot(612)
ax2.plot(tm, y.value, 'b-', label=r'y')
ax2.set_xlabel("time[s]")
ax2.set_ylabel("pos_y[m]")

ax3 = fig.add_subplot(613)
ax3.plot(tm, v.value, 'b-', label=r'gam')
ax3.set_xlabel("time[s]")
ax3.set_ylabel("velocity[m/s]")

ax4 = fig.add_subplot(614)
ax4.plot(tm, gam.value, 'b-', label=r'gam')
ax4.set_xlabel("time[s]")
ax4.set_ylabel("gam[rad]")

ax5 = fig.add_subplot(615)
ax5.plot(tm, alpha.value, 'b-', label=r'alpha')
ax5.set_xlabel("time[s]")
ax5.set_ylabel("aoa[rad]")

ax6 = fig.add_subplot(616)
ax6.plot(tm, ft.value, 'b-', label=r'alpha')
ax6.set_xlabel("time[s]")
ax6.set_ylabel("ft[N]")

#fig.tight_layout()
fig.show()
"""
plt.plot(tm, x.value, 'k-', label=r'$x$')
plt.plot(tm, y.value, 'b-', label=r'$y$')
plt.plot(tm, v.value, 'g--', label=r'$v$')
plt.plot(tm, gam.value, 'r--', label=r'$gam$')
plt.plot(tm, alpha.value, 'r--', label=r'$alpha$')
plt.legend(loc='best')
plt.xlabel('Time')

plt.tight_layout()
plt.show()
"""
#fig.tight_layout()
plt.show()