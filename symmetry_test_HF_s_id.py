from numpy import *
from matplotlib.pyplot import *
from Cosine_Terms import generating_matrices_of_cosine_combinations
from brillouin_zone import *
from scipy import optimize


n_el_test = 0.2

U = 4.0

kBT = 1.0e-2  # temperature
grid_points = 400  # dimension of grid, has to be an even integer!
V1 = -2.0   # (constant) potential energy for next neighbour interaction
N = (grid_points-1)*(grid_points-1)  # total number of grid points
c = V1/(8.0*N)  # a constant which is often needed in the calculations
da = 1.0e-6  # the accuracy of the derivatives for the Newton method, which is used for solving the equation
a = 1.0  # grid constant
t = 1.0  # hopping parameter for horizontal and vertical neighbours
t_diag = -0.2  # hopping parameter for diagonal neighbours, should take values like 0.0, -0.2, -0.4

# x = [W, Delta_s, Delta_d, mu, L]
x_test = array([-0.1435, -1.1675235, -0.232, -2.65, -1.521])


@vectorize
def eps_k(gks, G, Term1, Term2):
    return Term1 * gks + Term2 * G


@vectorize
def E_k(ek, gks, gkd, Delta_s, Delta_d, mu, L):
    return sqrt((ek - mu) ** 2 + (Delta_s * gks + L) ** 2 + Delta_d ** 2 * gkd ** 2)

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

def sum_over_all_k_for_E(x, sym):
    Term1 = -(t + x[0])
    Term2 = - t_diag
    if sym == 8:
        res = zeros(4, dtype=float)
        for i in range(4):
            y0s = array(matrix_list_8S[0][i][:])
            y0d = array(matrix_list_8S[1][i][:])
            G = array(matrix_list_8S[2][i][:])
            y1 = eps_k(y0s, G, Term1, Term2)
            y2 = E_k(y1, y0s, y0d, x[1], x[2], x[3], x[4])
            y5 = tanh(y2 / (2.0 * kBT))
            Z = y1 - y2 * y5
            res[i] = sum(Z)
        return 8.0*res[0] + 4.0*res[1] + 4.0*res[2] + res[3]
    elif sym == 4:
        res = zeros(4, dtype=float)
        for i in range(4):
            y0s = array(matrix_list_4S[0][i][:])
            y0d = array(matrix_list_4S[1][i][:])
            G = array(matrix_list_4S[2][i][:])
            y1 = eps_k(y0s, G, Term1, Term2)
            y2 = E_k(y1, y0s, y0d, x[1], x[2], x[3], x[4])
            y5 = tanh(y2 / (2.0 * kBT))
            Z = y1 - y2 * y5
            res[i] = sum(Z)
        return 4.0*res[0] + 2.0*res[1] + 2.0*res[2] + res[3]
    elif sym == 0:
        y0s = gamma_ks(X, Y)
        y0d = gamma_kd(X, Y)
        G = cos_term(X, Y)
        y1 = eps_k(y0s, G, Term1, Term2)
        y2 = E_k(y1, y0s, y0d, x[1], x[2], x[3], x[4])
        y5 = tanh(y2 / (2.0 * kBT))
        Z = y1 - y2 * y5
        return sum(Z)
    else:
        print('symmetry not implemented')
        return

print('E_0 = ', sum_over_all_k_for_E(x_test, 0))
print('E_4 = ', sum_over_all_k_for_E(x_test, 4))
print('E_8 = ', sum_over_all_k_for_E(x_test, 8))



def sum_over_all_k_for_f(x, sym):
    Term1 = -(t + x[0])
    Term2 = - t_diag
    if sym == 8:
        res = zeros((4, 5), dtype=complex)
        res_vec = zeros(5, dtype=complex)
        for i in range(4):
            y0s = array(matrix_list_8S[0][i][:])
            y0d = array(matrix_list_8S[1][i][:])
            G = array(matrix_list_8S[2][i][:])
            y1 = eps_k(y0s, G, Term1, Term2)
            y2 = E_k(y1, y0s, y0d, x[1], x[2], x[3], x[4])
            y5 = tanh(y2 / (2.0 * kBT))
            y3 = (y1 - x[3]) / y2 * y5
            y6 = (x[1] * y0s + x[4] + 1j*x[2]*y0d) / y2 * y5
            y7 = (-1j*(x[1] * y0s + x[4]) + x[2]*y0d) / y2 * y5
            y4s = y0s * y6
            y4d = y0d * y7
            Z0 = y0s * y3
            Z1 = y4s
            Z2 = y4d
            Z3 = y3
            Z4 = y6
            res[i, 0] = sum(Z0)
            res[i, 1] = sum(Z1)
            res[i, 2] = sum(Z2)
            res[i, 3] = sum(Z3)
            res[i, 4] = sum(Z4)
        for i in range(5):
            res_vec[i] = 8.0*res[0, i] + 4.0*res[1, i] + 4.0*res[2, i] + res[3, i]
        return res_vec
    elif sym == 4:
        res = zeros((4, 5), dtype=complex)
        res_vec = zeros(5, dtype=complex)
        for i in range(4):
            y0s = array(matrix_list_4S[0][i][:])
            y0d = array(matrix_list_4S[1][i][:])
            G = array(matrix_list_4S[2][i][:])
            y1 = eps_k(y0s, G, Term1, Term2)
            y2 = E_k(y1, y0s, y0d, x[1], x[2], x[3], x[4])
            y5 = tanh(y2 / (2.0 * kBT))
            y3 = (y1 - x[2]) / y2 * y5
            y6 = (x[1] * y0s + x[4] + 1j * x[2] * y0d) / y2 * y5
            y7 = (-1j * (x[1] * y0s + x[4]) + x[2] * y0d) / y2 * y5
            y4s = y0s * y6
            y4d = y0d * y7
            Z0 = y0s * y3
            Z1 = y4s
            Z2 = y4d
            Z3 = y3
            Z4 = y6
            res[i, 0] = sum(Z0)
            res[i, 1] = sum(Z1)
            res[i, 2] = sum(Z2)
            res[i, 3] = sum(Z3)
            res[i, 4] = sum(Z4)
        for i in range(5):
            res_vec[i] = 4.0 * res[0, i] + 2.0 * res[1, i] + 2.0 * res[2, i] + res[3, i]
        return res_vec
    elif sym == 0:
        y0s = gamma_ks(X, Y)
        y0d = gamma_kd(X, Y)
        G = cos_term(X, Y)
        y1 = eps_k(y0s, G, Term1, Term2)
        y2 = E_k(y1, y0s, y0d, x[1], x[2], x[3], x[4])
        y5 = tanh(y2 / (2.0 * kBT))
        y3 = (y1 - x[2]) / y2 * y5
        y6 = (x[1] * y0s + x[4] + 1j * x[2] * y0d) / y2 * y5
        y7 = (-1j * (x[1] * y0s + x[4]) + x[2] * y0d) / y2 * y5
        y4s = y0s * y6
        y4d = y0d * y7
        Z0 = y0s * y3
        Z1 = y4s
        Z2 = y4d
        Z3 = y3
        Z4 = y6
        return array([sum(Z0), sum(Z1), sum(Z2), sum(Z3), sum(Z4)])
    else:
        print('symmetry not implemented')
        return


print('f_0 = ', sum_over_all_k_for_f(x_test, 0))
print('f_4 = ', sum_over_all_k_for_f(x_test, 4))
print('f_8 = ', sum_over_all_k_for_f(x_test, 8))




