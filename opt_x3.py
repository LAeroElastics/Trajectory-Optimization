import csv
import numpy as np
import casadi as ca

initial_guess = 1
input_path = "./out.csv"

g0 = 32.174
sRef = 215.
thrust_type = "TF"

wEmpty = 16480.
wPayload = 0.
wFuelInit = 4220.
wPowerUnit = 0.
sfcPower = 0.7
etaProp = 0.9
kappa = 0.09
mile2ft = 5280.
ft2mile = 1. / mile2ft

kAp = 1.
tauAp = 0.1

w0 = (wEmpty + wPowerUnit + wFuelInit + wPayload)

fps2kt = 0.592484
kt2fps = 1. / fps2kt
deg2rad = np.pi / 180.
rad2deg = 180. / np.pi

tAltitude = [0.00000, 1000.00, 2000.00, 3000.00, 4000.00, 5000.00, 6000.00, 7000.00, 8000.00, 9000.00,
             10000.0, 11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0,
             20000.0, 21000.0, 22000.0, 23000.0, 24000.0, 25000.0, 26000.0, 27000.0, 28000.0, 29000.0,
             30000.0, 31000.0, 32000.0, 33000.0, 34000.0, 35000.0, 36000.0, 37000.0, 38000.0, 39000.0,
             40000.0, 41000.0, 42000.0, 43000.0, 44000.0, 45000.0, 46000.0, 47000.0, 48000.0, 49000.0,
             50000.0, 51000.0, 52000.0, 53000.0, 54000.0, 55000.0, 56000.0, 57000.0, 58000.0, 59000.0,
             60000.0, 61000.0, 62000.0, 63000.0, 64000.0, 65000.0, 66000.0, 67000.0, 68000.0, 69000.0, 70000.0]

tTemp = [288.150, 286.169, 284.188, 282.206, 280.225, 278.244, 276.263, 274.282, 272.300, 270.319,
         268.338, 266.357, 264.376, 262.394, 260.413, 258.432, 256.451, 254.470, 252.488, 250.507,
         248.526, 246.545, 244.564, 242.582, 240.601, 238.620, 236.639, 234.658, 232.676, 230.695,
         228.714, 226.733, 224.752, 222.770, 220.789, 218.808, 216.827, 216.650, 216.650, 216.650,
         216.650, 216.650, 216.650, 216.650, 216.650, 216.650, 216.650, 216.650, 216.650, 216.650,
         216.650, 216.650, 216.650, 216.650, 216.650, 216.650, 216.650, 216.650, 216.650, 216.650,
         216.650, 216.650, 216.650, 216.650, 216.650, 216.650, 216.767, 217.072, 217.376, 217.681, 217.986]

tPres = [101325, 97716.6, 94212.9, 90811.7, 87510.5, 84307.3, 81199.6, 78185.4, 75262.4, 72428.5,
         69681.7, 67019.8, 64440.9, 61942.9, 59523.9, 57182.0, 54915.2, 52721.8, 50599.8, 48547.6,
         46563.3, 44645.1, 42791.5, 41000.7, 39271.0, 37600.9, 35988.8, 34433.1, 32932.4, 31485.0,
         30089.6, 28744.7, 27448.9, 26200.8, 24999.0, 23842.3, 22729.3, 21662.7, 20646.2, 19677.3,
         18753.9, 17873.9, 17035.1, 16235.7, 15473.8, 14747.7, 14055.6, 13396.0, 12767.4, 12168.3,
         11597.3, 11053.0, 10534.4, 10040.0, 9568.87, 9119.83, 8691.87, 8283.99, 7895.25, 7524.75,
         7171.64, 6835.10, 6514.35, 6208.65, 5917.30, 5639.62, 5375.00, 5123.08, 4883.29, 4655.03, 4437.75]

tRho = [1.22500, 1.18955, 1.15490, 1.12102, 1.08791, 1.05555, 1.02393, 0.993040, 0.962870, 0.933406,
        0.904637, 0.876551, 0.849137, 0.822384, 0.796281, 0.770816, 0.745979, 0.721759, 0.698145,
        0.675127, 0.652694, 0.630836, 0.609542, 0.588802, 0.568607, 0.548946, 0.529809, 0.511187,
        0.493070, 0.475448, 0.458312, 0.441653, 0.425461, 0.409727, 0.394442, 0.379597, 0.365184,
        0.348331, 0.331985, 0.316406, 0.301559, 0.287407, 0.273920, 0.261066, 0.248815, 0.237139,
        0.226011, 0.215405, 0.205297, 0.195663, 0.186481, 0.177730, 0.169390, 0.161441, 0.153865,
        0.146645, 0.139763, 0.133205, 0.126954, 0.120996, 0.115318, 0.109907, 0.104749, 0.0998336,
        0.0951488, 0.0906838, 0.0863821, 0.0822178, 0.0782597, 0.0744972, 0.0709206]

tCs = [1116.45, 1112.61, 1108.75, 1104.88, 1100.99, 1097.09, 1093.18, 1089.25, 1085.31, 1081.36,
       1077.39, 1073.40, 1069.40, 1065.39, 1061.36, 1057.31, 1053.25, 1049.18, 1045.08, 1040.97,
       1036.85, 1032.71, 1028.55, 1024.38, 1020.19, 1015.98, 1011.75, 1007.51, 1003.24, 998.963,
       994.664, 990.347, 986.010, 981.655, 977.280, 972.885, 968.471, 968.076, 968.076, 968.076,
       968.076, 968.076, 968.076, 968.076, 968.076, 968.076, 968.076, 968.076, 968.076, 968.076,
       968.076, 968.076, 968.076, 968.076, 968.076, 968.076, 968.076, 968.076, 968.076, 968.076,
       968.076, 968.076, 968.076, 968.076, 968.076, 968.076, 968.337, 969.017, 969.698, 970.377, 971.056]

tMu = [0.0000181206, 0.0000180215, 0.0000179221, 0.0000178223, 0.0000177223, 0.0000176219, 0.0000175212,
       0.0000174202, 0.0000173188, 0.0000172171, 0.0000171150, 0.0000170126, 0.0000169099, 0.0000168068,
       0.0000167033, 0.0000165995, 0.0000164953, 0.0000163908, 0.0000162859, 0.0000161807, 0.0000160751,
       0.0000159691, 0.0000158627, 0.0000157560, 0.0000156489, 0.0000155414, 0.0000154335, 0.0000153252,
       0.0000152165, 0.0000151075, 0.0000149980, 0.0000148881, 0.0000147779, 0.0000146672, 0.0000145561,
       0.0000144446, 0.0000143326, 0.0000143226, 0.0000143226, 0.0000143226, 0.0000143226, 0.0000143226,
       0.0000143226, 0.0000143226, 0.0000143226, 0.0000143226, 0.0000143226, 0.0000143226, 0.0000143226,
       0.0000143226, 0.0000143226, 0.0000143226, 0.0000143226, 0.0000143226, 0.0000143226, 0.0000143226,
       0.0000143226, 0.0000143226, 0.0000143226, 0.0000143226, 0.0000143226, 0.0000143226, 0.0000143226,
       0.0000143226, 0.0000143226, 0.0000143226, 0.0000143293, 0.0000143465, 0.0000143637, 0.0000143810, 0.0000143982]

tMach = [0., 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 2.2]
tCd0 = [0.01, 0.01, 0.01, 0.02, 0.02, 0.025, 0.025, 0.02, 0.02]
tCla = [1., 3.7, 3.9, 4.2, 4.7, 5.1, 3.4, 2.4, 2.2]

tMachTh = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
tAltTh = [0., 5000., 10000., 15000., 20000., 25000., 30000., 40000., 50000., 70000]

tCoeffTh = [0.53, 0.53, 0.44, 0.38, 0.32, 0.27, 0.22, 0.12, 0.07, 0.00,
            0.61, 0.54, 0.46, 0.40, 0.33, 0.28, 0.23, 0.14, 0.09, 0.00,
            0.62, 0.55, 0.48, 0.41, 0.35, 0.29, 0.25, 0.16, 0.10, 0.01,
            0.67, 0.60, 0.52, 0.45, 0.38, 0.32, 0.27, 0.18, 0.11, 0.02,
            0.75, 0.66, 0.58, 0.51, 0.43, 0.37, 0.31, 0.21, 0.12, 0.02,
            0.83, 0.75, 0.67, 0.59, 0.51, 0.43, 0.37, 0.25, 0.15, 0.03,
            0.79, 0.83, 0.76, 0.68, 0.60, 0.52, 0.44, 0.29, 0.18, 0.04,
            0.79, 0.80, 0.84, 0.79, 0.69, 0.61, 0.53, 0.35, 0.22, 0.05,
            0.79, 0.77, 0.92, 0.85, 0.78, 0.70, 0.61, 0.42, 0.26, 0.06,
            0.79, 0.74, 1.00, 0.90, 0.87, 0.76, 0.68, 0.47, 0.29, 0.07]

rho = ca.interpolant("rho", "linear", [tAltitude], tRho)
cla = ca.interpolant("cla", "linear", [tMach], tCla)
cd0 = ca.interpolant("cd0", "linear", [tMach], tCd0)
cs = ca.interpolant("cs", "linear", [tAltitude], tCs)

coeffTh = ca.interpolant("coeffTh", "linear", [tMachTh, tAltTh], tCoeffTh)


def thrustTP(hp, v):
    return hp * 550. * etaProp / v


def thrustTF(p):
    return 10000. * p


def getThrust(func, *args):
    return func(*args)


def getDP(tas, h):
    return 0.5 * rho(h) * 0.062428 * tas ** 2. / g0


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
    return q * (getCd0(mach) + kappa * getCl(mach, alpha) ** 2.0) * sRef


def evalDynamics(xk, uk):
    mach = tas2mach(xk[2], xk[1])
    alpha = xk[4]
    q = getDP(xk[2], xk[1])
    lift = getLift(mach, alpha, q)
    drag = getDrag(mach, alpha, q)

    thrustCallback = None
    if thrust_type == "TF":
        thrustCallback = thrustTF
    elif thrust_type == "TP":
        thrustCallback = thrustTP

    thrust = getThrust(thrustCallback, uk[0])
    thrust *= coeffTh([mach, xk[1]])
    dx = []
    dx.append(xk[2] * ca.cos(xk[3]))
    dx.append(xk[2] * ca.sin(xk[3]))
    dx.append((thrust - drag - xk[5] * ca.sin(xk[3])) / (xk[5] / g0))
    dx.append((lift - xk[5] * ca.cos(xk[3])) / (xk[2] * (xk[5] / g0)))
    dx.append((kAp * uk[1] - xk[4]) / tauAp)
    dx.append(-sfcPower * thrust / 3600.)
    return ca.vertcat(*dx)


def generateCtrlMesh():
    h = []


def main():
    prob = ca.Opti()

    dimLGPolynomial = 5
    ctrlDiv = 50

    # state variables
    x = prob.variable(6, ctrlDiv + 1)  # x,y,vm,gamma,alpha,m

    # control variable
    u = prob.variable(2, ctrlDiv + 1)  # alpha_c,powerHp

    tf = prob.variable()

    # objective
    prob.minimize(tf)
    # prob.minimize(-x[5, -1])

    # initial guess
    t0 = 0.
    prob.set_initial(tf, 200.)
    """
    prob.set_initial(x[0, :], 0.)
    prob.set_initial(x[1, :], 0.)
    prob.set_initial(x[2, :], 200. * kt2fps)
    prob.set_initial(x[3, :], 0.)
    prob.set_initial(x[4, :], 0.)
    """
    prob.set_initial(x[5, :], w0)

    if initial_guess:
        f = np.loadtxt(input_path, delimiter=',', dtype='float')
        guess = f
        prob.set_initial(tf, guess[-1, 0])
        prob.set_initial(x[0, :], guess[:, 1])
        prob.set_initial(x[1, :], guess[:, 2])
        prob.set_initial(x[2, :], guess[:, 3])
        prob.set_initial(x[3, :], guess[:, 4])
        prob.set_initial(x[4, :], guess[:, 6])
        prob.set_initial(x[5, :], guess[:, 8])

    # initial condition
    # prob.subject_to(x[0, 0] == 0.)
    prob.subject_to(x[1, 0] == 0.)
    prob.subject_to(x[2, 0] == 120. * kt2fps)
    # prob.subject_to(x[3, 0] == 0.)
    # prob.subject_to(x[4, 0] == 0.)
    prob.subject_to(x[5, 0] == w0)

    # boundary condition
    prob.subject_to(prob.bounded(0., x[0, :], ca.inf))
    prob.subject_to(prob.bounded(0., x[1, :], 35000.))
    prob.subject_to(prob.bounded(80. * kt2fps, x[2, :], ca.inf))
    prob.subject_to(prob.bounded(-40. * deg2rad, x[3, :], 40. * deg2rad))
    prob.subject_to(prob.bounded(-45. * deg2rad, x[4, :], 45. * deg2rad))
    # prob.subject_to(prob.bounded(w0 - wFuelInit, x[5, :], ca.inf))

    # final condition
    # prob.subject_to(x[0, -1] == 100.)
    prob.subject_to(x[1, -1] == 35000.)
    # prob.subject_to(x[2, -1] == 1000.)
    prob.subject_to(x[3, -1] == 0.)
    # prob.subject_to(x[4, -1] == 0.)
    # prob.subject_to(x[5, -1] == 1000.)

    for i in range(2):
        prob.set_initial(u[i, :], 0)

    # boundary condition(ctrl)
    prob.subject_to(tf >= 0.)
    prob.subject_to(prob.bounded(0.1, u[0, :], 1.))
    prob.subject_to(prob.bounded(-20. * deg2rad, u[1, :], 20. * deg2rad))

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
        xc = prob.variable(6, dimLGPolynomial)
        # initial condition
        # prob.set_initial(xc[0, :], 0.)
        # prob.set_initial(xc[1, :], 0.)
        prob.set_initial(xc[2, :], 200. * kt2fps)
        # prob.set_initial(xc[3, :], 0.)
        # prob.set_initial(xc[4, :], 0.)
        prob.set_initial(xc[5, :], w0)

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

    solver_options = {"ipopt.max_iter": 5000}

    prob.solver("ipopt", solver_options)
    sol = prob.solve()

    reconTime = 0.5 * (tf - t0) * tau + 0.5 * (tf - t0)

    tgrid = np.linspace(0, sol.value(tf), ctrlDiv + 1)
    qq = getDP(sol.value(x[2, :]), sol.value(x[1, :]))
    mn = tas2mach(sol.value(x[2, :]), sol.value(x[1, :]))

    tgrid_out = tgrid.transpose()
    x_i = np.array(sol.value(x[0, :]))
    y_i = np.array(sol.value(x[1, :]))
    vm = np.array(sol.value(x[2, :]))
    qq_out = np.array(qq)[:, 0]
    mn_out = np.array(mn)[:, 0]
    gam = np.array(sol.value(x[3, :]))
    alpha = np.array(sol.value(x[4, :]))
    alpha_c = np.array(sol.value(u[1, :]))
    thrust = np.array(sol.value(u[0, :]))
    weight = np.array(sol.value(x[5, :]))

    outArray = [
        tgrid_out,
        x_i,
        y_i,
        vm,
        gam,
        alpha_c,
        alpha,
        thrust,
        weight,
        qq_out,
        mn_out
    ]

    block = np.vstack(outArray)
    print(np.shape(block))

    with open("./out.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(block.transpose())

    from pylab import plot, step, figure, legend, show, spy
    figure(1)
    plot(sol.value(x[0, :]), sol.value(x[1, :]), label="pos")
    legend(loc="upper left")
    figure(2)
    plot(tgrid, sol.value(x[2, :]), label="vm")
    legend(loc="upper left")
    figure(3)
    plot(tgrid, sol.value(x[3, :]), label="gamma")
    plot(tgrid, sol.value(x[4, :]), label="alpha")
    plot(tgrid, sol.value(u[1, :]), label="alpha_c")
    legend(loc="upper left")
    figure(4)
    plot(tgrid, sol.value(x[5, :]), label="mass")
    legend(loc="upper left")
    figure(5)
    plot(tgrid, sol.value(u[0, :]), label="thrust")
    legend(loc="upper left")
    figure(6)
    plot(tgrid, qq, label="Dynamic Pressure")
    legend(loc="upper left")
    figure(7)
    plot(tgrid, mn, label="Mach number")
    legend(loc="upper left")
    figure(8)
    plot(sol.value(x[2, :]), sol.value(x[1, :]), label="VM")
    legend(loc="upper left")

    show()


if __name__ == '__main__':
    main()
