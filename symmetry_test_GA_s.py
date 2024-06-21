from numpy import *
from matplotlib.pyplot import *
from Cosine_Terms import generating_matrices_of_cosine_combinations
from brillouin_zone import *
from scipy import optimize


D_test = 0.03
n_el_test = 0.1

U = 4.0

kBT = 1.0e-4  # temperature
grid_points = 400  # dimension of grid, has to be an even integer!
V1 = -2.0   # (constant) potential energy for next neighbour interaction
N = (grid_points-1)*(grid_points-1)  # total number of grid points
c = V1/(8*N)  # a constant which is often needed in the calculations
da = 1.0e-6  # the accuracy of the derivatives for the Newton method, which is used for solving the equation
a = 1.0  # grid constant
t = 1.0  # hopping parameter for horizontal and vertical neighbours
t_diag = -0.4  # hopping parameter for diagonal neighbours, should take values like 0.0, -0.2, -0.4
# x = [W, Delta_s, Delta_d, mu, cidciu, lamda]
x_test = array([-0.1, -0.80, -2.02, 0.5, 0.3])

def GA_CRI_ZZ(D, n, cidciu):
    """
    GA_CRI_ZZ is the <zz> coefficient inside the epsilon_k from the charge rotational invariant Gutzwiller approximation
    :param D: probability for double occupation per site
    :param n: electron density
    :param cidciu: self correlation <c_{i, downarrow} c_{i, uparrow}>
    :return: the charge rotational invariant Gutzwiller coefficient
    """
    Jx = cidciu
    Jz = 0.5 * (n - 1.0)
    J = sqrt(Jx ** 2 + Jz ** 2)
    annoying_root_term = D - Jz - J
    if allclose(annoying_root_term, 0.0):
        annoying_root_term = 0.0
    if annoying_root_term < 0.0:
        # print('D = ', D, ', Jz = ', Jz, ', J = ', J, ', D - Jz - J =', D-Jz-J, 'Jx = ', Jx)
        # raise ValueError("sqrt(D - Jz - J) is complex!")
        annoying_root_term = 0.0
        return 2.0 * (0.5 + Jz - D) / (0.25 - J ** 2) * (D - Jz + sqrt(annoying_root_term) * sqrt(D - Jz + J))
    else:
        return 2.0 * (0.5 + Jz - D) / (0.25 - J ** 2) * (D - Jz + sqrt(annoying_root_term) * sqrt(D - Jz + J))


@vectorize
def eps_k(gks, G, Term1, Term2):
    return Term1 * gks + Term2 * G


@vectorize
def E_k(ek, gks, Delta_s, mu, lamda):
    return sqrt((ek - mu) ** 2 + (Delta_s * gks + lamda) ** 2)



@vectorize
def gamma_ks(kx, ky, a=1.0):
    return 2.0 * (cos(a * kx) + cos(a * ky))
@vectorize
def gamma_kd(kx, ky, a=1.0):
    return 2.0 * (cos(a * kx) - cos(a * ky))

@vectorize
def cos_term(kx, ky, a=1.0):
    return 4.0 * cos(kx * a) * cos(ky * a)

k_array_complete, area_list_8S, area_list_4S = create_grids(grid_points)
matrix_list_8S = generating_matrices_of_cosine_combinations(area_list_8S)  # generating the cosine terms for the different areas
matrix_list_4S = generating_matrices_of_cosine_combinations(area_list_4S)  # generating the cosine terms for the different areas

kx = create_whole_grid(grid_points)[0:-1]
ky = create_whole_grid(grid_points)[0:-1]
X, Y = meshgrid(kx, ky)



######################## Calculation of the Energy #######################

def sum_over_all_k_for_E(x, n_el, D):
    Term1 = -(t * GA_CRI_ZZ(D, n_el, x[3]) + x[0])
    Term2 = - t_diag * GA_CRI_ZZ(D, n_el, x[3])
    y0s = gamma_ks(X, Y)
    G = cos_term(X, Y)
    y1 = eps_k(y0s, G, Term1, Term2)
    y2 = E_k(y1, y0s, x[1], x[2], x[4])
    y5 = tanh(y2 / (2.0 * kBT))
    Z = y1 - y2 * y5
    contourf(X, Y, Z, 100)
    colorbar()
    title('E')
    show()
    res_vec = sum(Z)
    return res_vec


def calc_E(D, n_el, x):
    sum_res = sum_over_all_k_for_E(x, n_el, D)
    return sum_res / N - 4.0 / V1 * (x[1] ** 2 - x[0] ** 2) + x[2] * (n_el - 1.0) + U * D - 2.0 * x[3] * x[4]

print('E = ', calc_E(D_test, n_el_test, x_test))
# we see: E is 8S-symmetric

def calc_y_for_symmetry_test(x, n_el, D):
    y0s = gamma_ks(X, Y)
    G = cos_term(X, Y)
    Term1 = -(t * GA_CRI_ZZ(D, n_el, x[3]) + x[0])
    Term2 = - t_diag * GA_CRI_ZZ(D, n_el, x[3])
    y1 = eps_k(y0s, G, Term1, Term2)
    y2 = E_k(y1, y0s, x[1], x[2], x[4])
    y3 = (y1 - x[2]) / y2
    y4 = y0s * (x[1] * y0s + x[4]) / y2
    y5 = tanh(y2 / (2.0 * kBT))
    y6 = (x[1] * y0s + x[4]) / y2 * y5
    eps0_k = -(t * y0s + t_diag * G)
    y7 = eps0_k * (y1 - x[2]) / y2 * y5
    return array([y0s, y1, y2, y3, y4, y5, y6, y7])

def f0_sum_for_newton_for_symmetry_test(y):
    Z = y[1]   # y[0] * y[3] * y[5]
    contourf(X, Y, Z, 100)
    colorbar()
    title('W')
    show()
    return sum(Z)


# Delta_s
def f1_sum_for_newton_for_symmetry_test(y):
    Z = y[4] * y[5]
    contourf(X, Y, Z, 100)
    colorbar()
    title('Delta_s')
    show()
    return


# Delta_d
def f2_sum_for_newton_for_symmetry_test(y):
    Z = y[3] * y[5]
    contourf(X, Y, Z, 100)
    colorbar()
    title('mu')
    show()
    return sum(Z)


# mu
def f3_sum_for_newton_for_symmetry_test(y):
    Z = y[6]
    contourf(X, Y, Z, 100)
    colorbar()
    title('cidciu')
    show()
    return


# cidciu
def f4_sum_for_newton_for_symmetry_test(y):
    Z = y[7]
    contourf(X, Y, Z, 100)
    colorbar()
    title('lamda')
    show()
    return




y_array = calc_y_for_symmetry_test(x_test, n_el_test, D_test)

print(f0_sum_for_newton_for_symmetry_test(y_array))
# we see: f0 is 8S-symmetric

print(f1_sum_for_newton_for_symmetry_test(y_array))
# we see: f1 is 8S-symmetric

print(f2_sum_for_newton_for_symmetry_test(y_array))
# we see: f2 is 8S-symmetric

print(f3_sum_for_newton_for_symmetry_test(y_array))
# we see: f3 is 8S-symmetric

print(f4_sum_for_newton_for_symmetry_test(y_array))
# we see: f4 is 8S-symmetric

