from pyomo.environ import *
from pyomo.dae import *

import matplotlib.pyplot as plt

"""
    2D ball throwing under gravity 
"""

# Simulation Params
div_size = 100  # Number of divide points

# Model Definitions
md = ConcreteModel()  # Concrete model for ODE
md.N = Param(initialize=div_size)  # Number of divide points
md.t_final = Var(initialize=20.0, within=PositiveReals)  # Time of Flight
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
md.beta = Var(md.i, initialize=45.0 * 3.14 / 180.0)
md.u = Var(md.i, initialize=10.0, bounds=(0.0, 50.0))
md.v = Var(md.i, initialize=10.0, bounds=(-1000.0, 1000.0))
md.F = Var(md.i, initialize=0.0)
md.g0 = Var(md.i, initialize=9.8)
md.ax = Var(md.i, initialize=0.0)
md.ay = Var(md.i, initialize=0.0)

# Objective Function
md.obj = Objective(expr=md.t_final, sense=minimize)


# Dynamics
def ax_rule(m, i):
    if i == 0: return Constraint.Skip
    return m.ax[i] == 0.0


md._ax = Constraint(md.i, rule=ax_rule)


def ay_rule(m, i):
    if i == 0: return Constraint.Skip
    return m.ay[i] == -m.g0[i]


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
    yield m.x[100] == 100
    yield m.y[0] == 0
    yield m.y[100] == 0
    yield m.u[0] == 100
    # yield m.u[100] == 20
    yield m.v[0] == 100
    # yield m.v[100] == 30
    yield m.beta[0] == 3.14 / 180.0 * 45.0
    yield ConstraintList.End


md.conlist = ConstraintList(rule=conlist_rule(md))


# Solver Controls
disc = TransformationFactory("dae.collocation")
disc.apply_to(md, scheme="LAGRANGE-LEGENDRE")

solver = SolverFactory("ipopt")
results = solver.solve(md)
md.display()

# Display Results
x = [md.x[i].value for i in md.i]
y = [md.y[i].value for i in md.i]
u = [md.u[i].value for i in md.i]
v = [md.v[i].value for i in md.i]
beta = [md.beta[i].value for i in md.i]

print(beta[0] * 180.0 / 3.14)

plt.plot(x, y)
plt.show()
