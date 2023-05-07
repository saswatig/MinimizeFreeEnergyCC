import sys
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import pylab


kb = 1.0
eps = 1.0
sigma = 1.0

# Reading in the lattice vectors (In this case for fcc with lp=1, \rho=4)
Gx_lp1 = np.loadtxt(fname='R_FCC_x')
Gy_lp1 = np.loadtxt(fname='R_FCC_y')
Gz_lp1 = np.loadtxt(fname='R_FCC_z')

# GEM 4 potential


def phi_r(r):
    V_r = eps * np.exp(-(r / sigma)**4)
    return V_r


def integrand_1(r, alpha, R_i):
    func_1 = (r / R_i) * (np.exp(-(alpha * (r - R_i)**2) / 2.0) -
                          np.exp(-(alpha * (r + R_i)**2) / 2.0)) * phi_r(r)
    return func_1


def integrand_2(r, alpha):
    func_2 = r * r * (np.exp(-alpha * r * r / 2.0)) * phi_r(r)
    return func_2

# The function to be minimized


def the_function(p_in, n0, T):
    alpha, nc = p_in
    param_1 = kb * T * (np.log(nc) + 1.5 * np.log(alpha *
                                                  sigma**2 / np.pi)) - 2.5  # + 3.0 * np.log(((1.0 / n0)**(1.0 / 3)) / sigma)

    sum1 = 0.0
    for i1 in range(len(Gx_lp1)):
        Gx = Gx_lp1[i1] * ((nc * 4.0 / n0)**(1.0 / 3.0))
        Gy = Gy_lp1[i1] * ((nc * 4.0 / n0)**(1.0 / 3.0))
        Gz = Gz_lp1[i1] * ((nc * 4.0 / n0)**(1.0 / 3.0))
        R_i = ((Gx**2 + Gy**2 + Gz**2)**0.5)
        I = quad(integrand_1, 0, np.inf, args=(alpha, R_i))
        sum1 = sum1 + I[0]

    param_2 = nc * ((alpha / (8.0 * np.pi))**0.5) * sum1

    I = quad(integrand_2, 0, np.inf, args=(alpha))
    param_3 = (nc - 1) * ((alpha**3 / (2.0 * np.pi))**0.5) * I[0]

    obj = param_1 + param_2 + param_3

    return obj


# Initial guesses
alpha_init = 40
nc_init = 15.0
p_in0 = [alpha_init, nc_init]
bnds = ((0.0, None), (0.0, None),)


# bounds set to minimize the functions, the lower bound must >=0 so that nc, \alpha does not take up -ve values


# output the nc, alpha for a given value of temperature T for 7.5<=n0<=9.0
def minimization(n0, T):
    # n0 = Input the desired range of average density
    # T  = Input the desired temperature
    res = minimize(the_function, p_in0, args=(n0, T), bounds=bnds)
    alpha_fin = (res.x[0])
    nc_fin = (res.x[1])
    p_in_fin = [alpha_fin, nc_fin]
    f_fin = the_function(p_in_fin, n0, T)
    output_func = [nc_fin, alpha_fin, f_fin]

    return output_func


npoints = 11
change = 0.25
T = 1.0

f = open("n0_alpha_nc_F_T_1p0.txt", "w")
for i in range(npoints):
    n0 = 7.0 + i * change
    #nc_alpha_fen = F_Minimize_sum_RLV.minimization(n0, T)
    nc_alpha_fen = minimization(n0, T)
    nc = nc_alpha_fen[0]
    alpha = nc_alpha_fen[1]
    free_en = nc_alpha_fen[2]
    print(n0, alpha, nc, free_en, file=f)
    #nc = 16.882656216953126
    #alpha = 34.56376933616536

    #q_0_elastic_const.cal_q0_elastic_const(n0, T, nc, alpha)
f.close()
