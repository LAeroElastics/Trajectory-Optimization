import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from gekko import GEKKO

d2r = 3.14 / 180.0
r2d = 1. / d2r

m = GEKKO(remote=True)
nt = 51
t_array = np.linspace(0, 1, nt)
m.time = t_array

m.options.SENSITIVITY = 0

m.options.CSV_WRITE = 1
m.options.CSV_READ = 1

m.options.NODES = 50
m.options.SOLVER = 3
m.options.IMODE = 6
m.options.MAX_ITER = 5000
m.options.MV_TYPE = 0
m.options.DIAGLEVEL = 0
m.options.REDUCE = 0

g = 9.8
S = 0.5
m0 = 600
mp0 = 550
tsp = 20
Isp = 300

tf = m.CV(value=200, lb=0.)
tf.STATUS = 1

alpha = m.MV(value=0, lb=-20 * d2r, ub=20 * d2r)
alpha.STATUS = 1

x = m.Var(value=0.0, lb=0.)
y = m.Var(value=0.0, lb=0.)
v = m.Var(value=0.0, lb=0.)  # , ub=1000.)

gam = m.Var(value=90., lb=-180. * d2r, ub=180. * d2r)
mass = m.Var(value=m0, lb=m0-mp0, ub=m0)

tave = mp0 * Isp * g / tsp
step = [0 if z > tsp else tave for z in t_array * tf.value]
#step = [0 if z > tsp/nt else tave for z in t_array]
ft = m.Param(value=step)

rho = m.Intermediate( 1.2205611857638659 * m.exp(-0.00009107790874911096 * y + -1.8783521651107734e-9 * y**2 ))
q = m.Intermediate(0.5 * rho * v * v)

cd0 = m.Var(value=0.01)
cdf = m.Var(value=0.5)
cna = m.Var(value=0.2)

cs = m.Intermediate(-0.00117 * y + 340.288)
mn = m.Var()
m.Equation(mn == v / cs)

t_mn = [0.3, 0.5, 0.9, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]
t_cd0 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
t_cdf = [0.5, 0.6, 0.7, 0.8, 0.65, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
t_cna = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

m.cspline(mn, cd0, t_mn, t_cd0, True)
m.cspline(mn, cdf, t_mn, t_cdf, True)
m.cspline(mn, cna, t_mn, t_cdf, True)

ca = m.Intermediate(cd0 + cdf)
cn = m.Intermediate(cna * alpha)
fa = m.Intermediate(q * ca * S)
fn = m.Intermediate(q * cn * S)

m.Equation(x.dt() == tf * v * m.cos(gam))
m.Equation(y.dt() == tf * v * m.sin(gam))

fg_x = m.Intermediate(g * m.cos(gam))
fg_y = m.Intermediate(g * m.sin(gam))

ft_x = m.Intermediate(ft * m.cos(alpha))
ft_y = m.Intermediate(ft * m.sin(alpha))

m.Equation(v.dt() * mass == tf * ft_x - tf * fa)
m.Equation(gam.dt() * mass * v == tf * ft_y + tf * fn - tf * mass * fg_x)
m.Equation(mass.dt() * g * Isp == -tf * ft)

m.fix_initial(mass, val=m0)
m.fix_initial(x, val=0.0)
m.fix_initial(y, val=0.0)

#m.fix_final(x, val=10000.)
m.fix_final(y, val=10000)

m.fix_initial(v, val=0)
#m.fix_final(v,val=295.)

m.fix_initial(gam, val=90. * d2r)
m.fix_initial(alpha, val=0.)

#m.fix(gam, pos=len(m.time)-1, val=45. * d2r)
# m.fix(m.alpha, pos=len(m.time)-1, val=0.)

m.Obj(tf)
#m.Obj(-y)

m.options.COLDSTART = 2
m.solve(disp=True)

m.options.COLDSTART = 0
m.options.TIME_SHIFT = 0
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
ax1 = fig.add_subplot(4, 1, 1)
ax1.plot(tm, x.value, 'b-', label=r'x')
ax1.set_xlabel("time[s]")
ax1.set_ylabel("pos_x[m]")

ax2 = fig.add_subplot(4, 1, 2)
ax2.plot(tm, y.value, 'b-', label=r'y')
ax2.set_xlabel("time[s]")
ax2.set_ylabel("pos_y[m]")

ax3 = fig.add_subplot(4, 1, 3)
ax3.plot(tm, gam.value, 'b-', label=r'gam')
ax3.set_xlabel("time[s]")
ax3.set_ylabel("gam[rad]")

ax4 = fig.add_subplot(4, 1, 4)
ax4.plot(tm, alpha.value, 'b-', label=r'alpha')
ax4.set_xlabel("time[s]")
ax4.set_ylabel("aoa[rad]")

fig.tight_layout()
fig.show()

"""
plt.plot(tm, m.x.value, 'k-', label=r'$x$')
plt.plot(tm, m.y.value, 'b-', label=r'$y$')
plt.plot(tm, m.v.value, 'g--', label=r'$v$')
plt.plot(tm, m.gam.value, 'r--', label=r'$gam$')
plt.plot(tm, m.alpha.value, 'r--', label=r'$alpha$')
plt.legend(loc='best')
plt.xlabel('Time')
plt.show()
"""