from pyomo.environ import *
from pyomo.dae import *

import matplotlib.pyplot as plt

"""
    2D projectile flight 
"""

# Simulation Params
div_size = 100  # Number of divide points

# Model Definitions
md = ConcreteModel()  # Concrete model for ODE
md.N = Param(initialize=div_size)  # Number of divide points
md.t_final = Var(initialize=20.0, within=PositiveReals)  # Time of Flight
md.t_sp = Param(initialize=5.0)  #
md.dt = Var(initialize=0.1, within=PositiveReals)  # Delta T


# Defining relations of time and divide points num.
def dt_rule(m):
    return m.dt == m.t_final / m.N


md.dtcon = Constraint(rule=dt_rule)


def i_init(m):
    result = []
    j = 0
    while j <= m.N:
        result.append(j)
        j = j + 1
    return result


md.i = Set(initialize=i_init, ordered=True)

md.x = Var(md.i, initialize=0.0)
md.y = Var(md.i, initialize=0.0)
md.beta = Var(md.i, initialize=90.0 * 3.14 / 180.0)
md.u = Var(md.i, initialize=10.0, bounds=(0.0, 50.0))
md.v = Var(md.i, initialize=10.0, bounds=(-1000.0, 1000.0))
md.Ft = Var(md.i, initialize=0.0)
md.Isp = Var(md.i, initialize=100.0, bounds=(100.0, 100.0))
md.g0 = Var(md.i, initialize=9.8, bounds=(9.8, 9.8))
md.rho = Var(md.i, initialize=1.2, bounds=(1.2, 1.2))
md.mass = Var(md.i, initialize=100.0, bounds=(0.0, 1000.0))
md.delta_mass = Var(md.i, initialize=0.1, bounds=(0.0, 2000.0))  # mass fluctuation
md.ax = Var(md.i, initialize=0.0)
md.ay = Var(md.i, initialize=0.0)
md.alpha = Var(md.i, initialize=0.0, bounds=(-90.0 * 3.14 / 180.0, 90.0 * 3.14 / 180.0))

# Objective Function
md.obj = Objective(expr=md.t_final, sense=minimize)

md.CA = Var(md.i, initialize=0.3, within=PositiveReals)
md.CN = Var(md.i, initialize=0.3, within=PositiveReals)
md.Vm = Var(md.i, initialize=0.0)

md.CA_base = Param(initialize=0.3)
md.CA_alpha = Param(initialize=0.25)
md.CN_base = Param(initialize=0.3)
md.CN_alpha = Param(initialize=1.75)
md.alpha_max = Param(initialize=10.0)
md.delta_mass_max = Param(initialize=10.0)

md.c0 = Param(initialize=0.5)
md.zero = Param(initialize=0.0)


def delta_mass_rule(m, i):
    if value(i * m.dt) <= value(m.t_sp):
        return m.delta_mass[i] == m.delta_mass_max
    else:
        return m.delta_mass[i] == m.zero


md.delta_masscon = Constraint(md.i, rule=delta_mass_rule)


def mass_rule(m, i):
    if i == 0: return Constraint.Skip
    if value(m.delta_mass[i]) == md.zero:
        return m.mass[i] == m.mass[i - 1]
    else:
        return m.mass[i] == m.mass[i - 1] - m.delta_mass[i] * md.dt


md.masscon = Constraint(md.i, rule=mass_rule)


def Ft_rule(m, i):
    return m.Ft[i] == m.delta_mass[i] * m.Isp[i] * m.g0[i]


md.Ftcon = Constraint(md.i, rule=Ft_rule)

def CA_rule(m, i):
    return m.CA[i] == (m.CA_base + m.CA_alpha * m.alpha[i] / m.alpha_max)


md.CAcon = Constraint(md.i, rule=CA_rule)


def CN_rule(m, i):
    return m.CN[i] == (m.CN_base + m.CN_alpha * m.alpha[i] / m.alpha_max)


md.CNcon = Constraint(md.i, rule=CN_rule)


def Vm_rule(m, i):
    return m.Vm[i] == (sqrt(m.u[i] ** 2.0 + m.v[i] ** 2.0))


md.Vmcon = Constraint(md.i, rule=Vm_rule)


# Dynamics
def ax_rule(m, i):
    if i == 0: return Constraint.Skip
    return m.ax[i] == (
            m.Ft[i] - m.c0 * m.rho[i] * m.Vm[i] * m.CA[i] - m.mass[i] * m.g0[i] * sin(m.beta[i])) / \
           m.mass[i]


md._ax = Constraint(md.i, rule=ax_rule)


def ay_rule(m, i):
    if i == 0: return Constraint.Skip
    return m.ay[i] == (m.c0 * m.rho[i] * m.Vm[i] * m.CN[i] - m.mass[i] * m.g0[i] * cos(m.beta[i])) / \
           m.mass[i]


md._ay = Constraint(md.i, rule=ay_rule)


def u_rule(m, i):
    if i == 0: return Constraint.Skip
    return m.u[i] == m.u[i - 1] + m.ax[i] * m.dt


md._u = Constraint(md.i, rule=u_rule)


def v_rule(m, i):
    if i == 0: return Constraint.Skip
    return m.v[i] == m.v[i - 1] + m.ay[i] * m.dt


md._v = Constraint(md.i, rule=v_rule)


def x_rule(m, i):
    if i == 0: return Constraint.Skip
    return m.x[i] == m.x[i - 1] + m.u[i] * m.dt


md._x = Constraint(md.i, rule=x_rule)


def y_rule(m, i):
    if i == 0: return Constraint.Skip
    return m.y[i] == m.y[i - 1] + m.v[i] * m.dt


md._y = Constraint(md.i, rule=y_rule)


# Constraints
def conlist_rule(m):
    yield m.x[0] == 0
    yield m.x[100] == 10000
    yield m.y[0] == 0
    yield m.y[100] == 10000
    yield m.u[0] == 0.0
    # yield m.u[100] == 20
    yield m.v[0] == 0.0
    # yield m.v[100] == 30
    # yield m.beta[0] == 3.14 / 180.0 * 45.0
    yield ConstraintList.End


md.conlist = ConstraintList(rule=conlist_rule(md))

# Solver Controls
disc = TransformationFactory("dae.collocation")
disc.apply_to(md, scheme="LAGRANGE-LEGENDRE")

solver = SolverFactory("ipopt")
md.display()
results = solver.solve(md)

# Display Results
x = [md.x[i].value for i in md.i]
y = [md.y[i].value for i in md.i]
u = [md.u[i].value for i in md.i]
v = [md.v[i].value for i in md.i]
beta = [md.beta[i].value for i in md.i]
m = [md.mass[i].value for i in md.i]
dm = [md.delta_mass[i].value for i in md.i]
g = [md.g0[i].value for i in md.i]

print(beta[0] * 180.0 / 3.14)

plt.plot(x, y)
plt.show()

plt.plot(m)
plt.show()
