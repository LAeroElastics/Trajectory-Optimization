import numpy as np
import casadi as ca

g = 32.174

kappa = 0.04
s_ref = 179.
eta = 0.9

k_ap = 1.
tau_ap = 0.1

sfc = 0.4

t_alt = [0, 2000., 4000., 6000., 8000., 10000.]
t_rho = ca.SX([1.22500, 1.00649, 0.819129, 0.659697, 0.525168, 0.412707])
t_mach = ca.SX([0., 0.3, 0.5, 0.7, 0.9])
t_cla = ca.SX([0.1, 0.1, 0.1, 0.1, 0.1])
t_cd0 = ca.SX([0.01, 0.01, 0.01, 0.01, 0.01])

rho = lambda y: ca.interp1d(t_alt, t_rho, y)
cla = lambda m: ca.interp1d(t_mach, t_cla, m)
cd0 = lambda m: ca.interp1d(t_mach, t_cd0, m)


def get_aero(y, vm, a):
    _rho = 1.22
    _q = 0.5 * _rho * vm ** 2. / g
    _cl = 0.1 * a
    _cd = 0.01 + kappa * _cl ** 2.
    fl = _cl * _q * s_ref
    fd = _cd * _q * s_ref
    return fl, fd


def get_thrust(p, vm):
    return p * 550. * eta / vm


def evalDynamics(x, u):
    fl, fd = get_aero(x[1, :], x[2, :], x[4, :])
    ft = get_thrust(u[0, :] + u[1, :], x[2, :])

    dx = []
    dx.append(x[2, :] * ca.cos(x[3, :]))
    dx.append(x[2, :] * ca.sin(x[3, :]))
    dx.append((ft - fd) / x[5, :] - g * ca.sin(x[2, :]))
    dx.append(fl / (x[5, :] * x[2, :]) - g * ca.cos(x[2, :]))
    dx.append((u[2, :] * k_ap - x[4, :]) / tau_ap)
    dx.append(u[0, :] * sfc)
    return ca.vertcat(*dx)


def evalCoeffsLagrange(n):
    beta = np.array([0.5 / np.sqrt(1 - (2 * i) ** (-2)) for i in range(1, n)])
    T = np.diag(beta, 1) + np.diag(beta, -1)
    D_, V = np.linalg.eig(T)
    tau = np.sort(D_)
    i = np.argsort(D_)
    w = 2 * (V[0, i] ** 2)
    tau = np.hstack((-1, tau, 1))
    D = np.zeros([n, n + 1])
    for k in range(1, n + 1):
        for l in range(0, n + 1):
            if k == l:
                D[k - 1, l] = 0
                for m in range(0, n + 1):
                    if m != k:
                        D[k - 1, l] += 1.0 / (tau[k] - tau[m])
            else:
                D[k - 1, l] = 1.0 / (tau[l] - tau[k])
                for m in range(0, n + 1):
                    if m != k and m != l:
                        D[k - 1, l] *= (tau[k] - tau[m]) / (tau[l] - tau[m])
    return tau, w, D


def main():
    opti = ca.Opti()
    dimLGPolynomial = 3
    timeDiv = 100

    x = opti.variable(6, timeDiv)
    u = opti.variable(3, timeDiv)

    tf = opti.variable()

    # objective
    opti.minimize(tf)

    t0 = 0.
    opti.set_initial(x[0, :], 0.)
    opti.set_initial(x[1, :], 0.)
    opti.set_initial(x[2, :], 100.)
    opti.set_initial(x[3, :], 0.)
    opti.set_initial(x[4, :], 0.)
    opti.set_initial(x[5, :], 1000.)

    opti.subject_to(tf > 0.)
    opti.subject_to(opti.bounded(-10., u[0, :], 10.))
    opti.subject_to(opti.bounded(0., u[1, :], ca.inf))
    opti.subject_to(opti.bounded(0., u[2, :], ca.inf))

    tau = np.append(0, ca.collocation_points(dimLGPolynomial, "legendre"))

    C = np.zeros((dimLGPolynomial + 1, dimLGPolynomial + 1))

    D = np.zeros(dimLGPolynomial + 1)

    B = np.zeros(dimLGPolynomial + 1)

    for j in range(dimLGPolynomial + 1):
        p = np.poly1d([1])
        for r in range(dimLGPolynomial + 1):
            if r != j:
                p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])

        D[j] = p(1.0)

        pder = np.polyder(p)
        for r in range(dimLGPolynomial + 1):
            C[j, r] = pder(tau[r])

        pint = np.polyint(p)
        B[j] = pint(1.0)

    h = tf / timeDiv
    J = 0

    # state at the time
    for k in range(timeDiv):
        # state at the collocation point
        Xc = opti.variable(6, dimLGPolynomial)
        # set initial value to xk_0
        Xk_end = D[0] * x[:, k]
        for j in range(1, dimLGPolynomial + 1):
            # set initial value to xk_0
            xp = C[0, j] * x[:, k]
            for r in range(dimLGPolynomial):
                # integration
                xp += C[r + 1, j] * Xc[r]
            fj = evalDynamics(Xc[j - 1], u[:, k])
            opti.subject_to(h * fj - xp)

            # Integrate jth collocation pt to fill up kth period
            Xk_end += D[j] * Xc[j - 1]

        opti.subject_to(Xk_end - x[:, timeDiv])

        opti.solver("ipopt")
        sol = opti.solve()


if __name__ == '__main__':
    main()
