import csv
import numpy as np
import casadi as ca

initial_guess = 0
input_path = "./out.csv"

g0 = 9.8
sRef = 49.2386
thrust_type = "TF"

wEmpty = 19050.864
wPayload = 0.
wFuelInit = 0.
wPowerUnit = 0.
isp = 1600.

w0 = (wEmpty + wPowerUnit + wFuelInit + wPayload)

deg2rad = np.pi / 180.
rad2deg = 180. / np.pi

tAltitude = [-2000, 0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000,
             30000, 32000, 34000, 36000, 38000, 40000, 42000, 44000, 46000, 48000, 50000, 52000, 54000, 56000, 58000,
             60000, 62000, 64000, 66000, 68000, 70000, 72000, 74000, 76000, 78000, 80000, 82000, 84000, 86000]

tRho = [1.478e+00, 1.225e+00, 1.007e+00, 8.193e-01, 6.601e-01, 5.258e-01, 4.135e-01, 3.119e-01, 2.279e-01, 1.665e-01,
        1.216e-01, 8.891e-02, 6.451e-02, 4.694e-02, 3.426e-02, 2.508e-02, 1.841e-02, 1.355e-02, 9.887e-03, 7.257e-03,
        5.366e-03, 3.995e-03, 2.995e-03, 2.259e-03, 1.714e-03, 1.317e-03, 1.027e-03, 8.055e-04, 6.389e-04, 5.044e-04,
        3.962e-04, 3.096e-04, 2.407e-04, 1.860e-04, 1.429e-04, 1.091e-04, 8.281e-05, 6.236e-05, 4.637e-05, 3.430e-05,
        2.523e-05, 1.845e-05, 1.341e-05, 9.690e-06, 6.955e-06]

tCs = [3.479e+02, 3.403e+02, 3.325e+02, 3.246e+02, 3.165e+02, 3.081e+02, 2.995e+02, 2.951e+02, 2.951e+02, 2.951e+02,
       2.951e+02, 2.951e+02, 2.964e+02, 2.977e+02, 2.991e+02, 3.004e+02, 3.017e+02, 3.030e+02, 3.065e+02, 3.101e+02,
       3.137e+02, 3.172e+02, 3.207e+02, 3.241e+02, 3.275e+02, 3.298e+02, 3.298e+02, 3.288e+02, 3.254e+02, 3.220e+02,
       3.186e+02, 3.151e+02, 3.115e+02, 3.080e+02, 3.044e+02, 3.007e+02, 2.971e+02, 2.934e+02, 2.907e+02, 2.880e+02,
       2.853e+02, 2.825e+02, 2.797e+02, 2.769e+02, 2.741e+02]

tMach = [-10, 0, 0.4, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8]
tCla = [3.44, 3.44, 3.44, 3.44, 3.58, 4.44, 3.44, 3.01, 2.86, 2.44]
tCd0 = [0.013, 0.013, 0.013, 0.013, 0.014, 0.031, 0.041, 0.039, 0.036, 0.035]
tEta = [0.54, 0.54, 0.54, 0.54, 0.75, 0.79, 0.78, 0.89, 0.93, 0.93]

tMachTh = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]
tAltTh = [0 * 304.8, 5 * 304.8, 10 * 304.8, 15 * 304.8, 20 * 304.8, 25 * 304.8, 30 * 304.8, 40 * 304.8, 50 * 304.8,
          70 * 304.8]
tThrust = [24.2, 24.0, 20.3, 17.3, 14.5, 12.2, 10.2, 5.7, 3.4, 0.1,
           28.0, 24.6, 21.1, 18.1, 15.2, 12.8, 10.7, 6.5, 3.9, 0.2,
           28.3, 25.2, 21.9, 18.7, 15.9, 13.4, 11.2, 7.3, 4.4, 0.4,
           30.8, 27.2, 23.8, 20.5, 17.3, 14.7, 12.3, 8.1, 4.9, 0.8,
           34.5, 30.3, 26.6, 23.2, 19.8, 16.8, 14.1, 9.4, 5.6, 1.1,
           37.9, 34.3, 30.4, 26.8, 23.3, 19.8, 16.8, 11.2, 6.8, 1.4,
           36.1, 38.0, 34.9, 31.3, 27.3, 23.6, 20.1, 13.4, 8.3, 1.7,
           36.1, 36.6, 38.5, 36.1, 31.6, 28.1, 24.2, 16.2, 10.0, 2.2,
           36.1, 35.2, 42.1, 38.7, 35.7, 32.0, 28.1, 19.3, 11.9, 2.9,
           36.1, 33.8, 45.7, 41.3, 39.8, 34.6, 31.1, 21.7, 13.3, 3.1]

rho = ca.interpolant("rho", "linear", [tAltitude], tRho)
cs = ca.interpolant("cs", "linear", [tAltitude], tCs)
cla = ca.interpolant("cla", "linear", [tMach], tCla)
cd0 = ca.interpolant("cd0", "linear", [tMach], tCd0)
eta = ca.interpolant("eta", "linear", [tMach], tEta)

grid = [tAltTh, tMachTh]
coeffTh = ca.interpolant("coeffTh", "linear", grid, tThrust)


def thrustTP(hp, v):
    return hp * 550. * etaProp / v


def thrustTF():
    return 1.


def getThrust(func, *args):
    return func(*args)


def getDP(tas, h):
    return 0.5 * rho(h) * tas ** 2.


def tas2mach(tas, altitude):
    return tas / cs(altitude)


def mach2tas(mach, altitude):
    return mach * cs(altitude)


def getCl(mach, alpha):
    return cla(mach) * alpha


def getCd0(mach):
    return cd0(mach)


def getLift(mach, alpha, q):
    return q * getCl(mach, alpha) * sRef


def getDrag(mach, alpha, q):
    return q * (getCd0(mach) + eta(mach) * getCl(mach, alpha ** 2.)) * sRef


def evalDynamics(xk, uk):
    mach = tas2mach(xk[1], xk[0])
    alpha = uk[0]
    q = getDP(xk[1], xk[0])
    lift = getLift(mach, alpha, q)
    drag = getDrag(mach, alpha, q)
    Re = 6378145
    mu = 3.986e14
    r = Re + xk[0]

    thrustCallback = None
    if thrust_type == "TF":
        thrustCallback = thrustTF
    elif thrust_type == "TP":
        thrustCallback = thrustTP

    # thrust = getThrust(thrustCallback, uk[0])
    thrust = thrustTF()
    thrust *= coeffTh(ca.horzcat(xk[0], mach))
    thrust *= 4448.2

    dx = []
    dx.append(xk[1] * ca.sin(xk[2]))
    dx.append((thrust * ca.cos(alpha) - drag) / xk[3] - mu * ca.sin(xk[2]) / r ** 2.)
    dx.append((thrust * ca.sin(alpha) + lift) / (xk[3] * xk[1]) + ca.cos(xk[2]) * (
            xk[1] / r - mu / (xk[1] * r ** 2.)))
    dx.append(-thrust / (g0 * isp))

    return ca.vertcat(*dx)


def generateCtrlMesh():
    h = []


def main():
    prob = ca.Opti()

    dimLGPolynomial = 9
    ctrlDiv = 50

    # state variables
    x = prob.variable(4, ctrlDiv + 1)  # x,y,vm,gamma,alpha,m

    # control variable
    u = prob.variable(1, ctrlDiv + 1)  # alpha_c,powerHp

    tf = prob.variable()

    # objective
    prob.minimize(tf)
    # prob.minimize(-x[5, -1])

    # initial guess
    t0 = 0.
    prob.set_initial(tf, 324.)
    """
    prob.set_initial(x[0, :], 0.)
    prob.set_initial(x[1, :], 0.)
    prob.set_initial(x[2, :], 200. * kt2fps)
    prob.set_initial(x[3, :], 0.)
    prob.set_initial(x[4, :], 0.)
    """
    prob.set_initial(x[3, :], w0)

    if initial_guess:
        f = np.loadtxt(input_path, delimiter=',', dtype='float')
        guess = f
        prob.set_initial(tf, guess[-1, 0])
        prob.set_initial(x[0, :], guess[:, 0])
        prob.set_initial(x[1, :], guess[:, 1])
        prob.set_initial(x[2, :], guess[:, 2])
        prob.set_initial(x[3, :], guess[:, 3])

    # initial condition
    prob.subject_to(x[0, 0] - 0. == 0.)
    prob.subject_to(x[1, 0] - 129.314 == 0.)
    prob.subject_to(x[2, 0] - 0. == 0.)
    prob.subject_to(x[3, 0] - w0 == 0.)

    # boundary condition
    prob.subject_to(prob.bounded(0., tf, 400.))
    prob.subject_to(prob.bounded(0., x[0, :], 21031.2))
    prob.subject_to(prob.bounded(5., x[1, :], 600.))
    prob.subject_to(prob.bounded(-45.0 * deg2rad, x[2, :], 45.0 * deg2rad))
    # prob.subject_to(prob.bounded(16500, x[3, :], 20410))

    # final condition
    prob.subject_to(x[0, -1] - 19995. == 0.)
    prob.subject_to(x[1, -1] - 330.092 == 0.)
    # prob.subject_to(x[2, -1] >= 0.)

    for i in range(1):
        prob.set_initial(u[i, :], 0 * deg2rad)

    # boundary condition(ctrl)
    # prob.subject_to(tf >= 0.)
    prob.subject_to(prob.bounded(-45. * deg2rad, u[0, :], 45. * deg2rad))

    # LG coefficients
    tau = np.append(0, ca.collocation_points(dimLGPolynomial, "legendre"))

    c = np.zeros((dimLGPolynomial + 1, dimLGPolynomial + 1))
    d = np.zeros(dimLGPolynomial + 1)
    b = np.zeros(dimLGPolynomial + 1)

    for j in range(dimLGPolynomial + 1):
        p = np.poly1d([1])
        for r in range(dimLGPolynomial + 1):
            if r != j:
                p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])

        d[j] = p(1.0)

        pder = np.polyder(p)
        for r in range(dimLGPolynomial + 1):
            c[j, r] = pder(tau[r])

        pint = np.polyint(p)
        b[j] = pint(1.0)
    h = tf / ctrlDiv

    # state at the time
    for k in range(ctrlDiv):
        # state at the collocation point
        xc = prob.variable(4, dimLGPolynomial)
        # initial condition
        prob.set_initial(xc[1, :], 180.)
        prob.set_initial(xc[3, :], 10000)

        # set initial value to xk_0
        xk_end = d[0] * x[:, k]
        for j in range(1, dimLGPolynomial + 1):
            # set initial value to xk_0
            xp = c[0, j] * x[:, k]
            for r in range(dimLGPolynomial):
                # integration
                xp += c[r + 1, j] * xc[:, r]
            fj = evalDynamics(xc[:, j - 1], u[:, k])
            # boundary condition at the collocation point
            prob.subject_to(fj * h - xp == 0.)

            # integrate jth collocation point to fill up kth period
            xk_end += d[j] * xc[:, j - 1]
        # close the gap
        prob.subject_to(xk_end - x[:, k + 1] == 0.)

    solver_options = {"ipopt.max_iter": 5000,
                      "ipopt.mu_init": 0.01,
                      "ipopt.tol": 1e-3,
                      "ipopt.mu_strategy": 'adaptive',
                      "ipopt.hessian_approximation": 'exact',
                      "ipopt.limited_memory_max_history": 6,
                      "ipopt.limited_memory_max_skipping": 1,
                      "ipopt.nlp_scaling_method": 'gradient-based'}

    prob.solver("ipopt", solver_options)
    sol = prob.solve()

    reconTime = 0.5 * (tf - t0) * tau + 0.5 * (tf - t0)

    tgrid = np.linspace(0, sol.value(tf), ctrlDiv + 1)
    qq = getDP(sol.value(x[2, :]), sol.value(x[1, :]))
    rro = rho(sol.value(x[1, :]))
    mn = tas2mach(sol.value(x[1, :]), sol.value(x[0, :]))

    thrust = thrustTF()
    thrust_c = sol.value(coeffTh(ca.vertcat(x[1, :], tas2mach(x[2, :], x[1, :]))))

    tgrid_out = tgrid.transpose()
    y_i = np.array(sol.value(x[0, :]))
    vm = np.array(sol.value(x[1, :]))
    qq_out = np.array(qq)[:, 0]
    mn_out = np.array(mn)[:, 0]
    gam = np.array(sol.value(x[2, :]))
    alpha_c = np.array(sol.value(u[0, :]))
    weight = np.array(sol.value(x[3, :]))

    outArray = [
        tgrid_out,
        y_i,
        vm,
        gam,
        weight
    ]

    block = np.vstack(outArray)

    with open("./out.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(block.transpose())

    from pylab import plot, step, figure, legend, show, spy
    a_airspd = np.linspace(0, 700, 100)
    a_altitude = np.linspace(0, 20000, 2000)

    def a_0(v, hh, w):
        return coeffTh(ca.horzcat(hh, tas2mach(v, hh))) * 4448.2 / w

    def a_1(v, hh, w):
        q = getDP(v, hh)
        mach = tas2mach(v, hh)
        cd0 = getCd0(mach)
        return q * cd0 / (w / sRef)

    def a_2(v, hh, w, n=1):
        q = getDP(v, hh)
        mach = tas2mach(v, hh)
        et = eta(mach)
        return n ** 2. * et * w / (q * sRef)

    def a_3(v, hh):
        return hh + v ** 2. / g0 * 0.5

    _v, _h = np.meshgrid(a_airspd, a_altitude)
    ps = [[v * (a_0(v, h, w=w0) - a_1(v, h, w=w0) - a_2(v, h, w=w0)) for v in a_airspd] for h in a_altitude]
    eh = [[a_3(v, h) for v in a_airspd] for h in a_altitude]
    ps_levels = [-400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    eh_levels = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]

    fig = figure(tight_layout=True)
    ax = fig.add_subplot(111, xlabel="airspeed, m/s", ylabel="altitude, m")
    ctps = ax.contour(_v, _h, ps, levels=ps_levels, colors="k", linewidths=1)
    cteh = ax.contour(_v, _h, eh, levels=eh_levels, colors="k", linewidths=1)
    plot(sol.value(x[1, :]), sol.value(x[0, :]))
    ax.clabel(ctps)
    ax.clabel(cteh)
    # ax.set_aspect('equal')
    show()

    figure(1)
    plot(tgrid, sol.value(x[0, :]), label="altitude")
    legend(loc="upper left")
    figure(2)
    plot(tgrid, sol.value(x[1, :]), label="vm")
    legend(loc="upper left")
    figure(3)
    plot(tgrid, sol.value(x[2, :]), label="gamma")
    plot(tgrid, sol.value(u[0, :]), label="alpha")
    legend(loc="upper left")
    figure(4)
    plot(tgrid, sol.value(x[3, :]), label="mass")
    legend(loc="upper left")
    figure(5)
    plot(tgrid, qq, label="Dynamic Pressure")
    legend(loc="upper left")
    figure(6)
    plot(tgrid, mn, label="Mach number")
    legend(loc="upper left")
    figure(7)
    plot(sol.value(x[1, :]), sol.value(x[0, :]), label="VM")
    legend(loc="upper left")
    show()


if __name__ == '__main__':
    main()
