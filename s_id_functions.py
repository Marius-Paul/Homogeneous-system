from numpy import vectorize, array, zeros, insert, savetxt, sqrt, cos, exp, sum, tanh
from scipy import optimize


def create_functions(V1, c, N, t, t_diag, matrix_list_8S, matrix_list_4S, kBT, U):

    @vectorize
    def eps_k(gks, G, Term1, Term2):
        return Term1 * gks + Term2 * G

    @vectorize
    def E_k(ek, gks, gkd, Delta_s, Delta_d, mu, L):
        return sqrt((ek - mu) ** 2 + (Delta_s * gks + L) ** 2 + Delta_d ** 2 * gkd ** 2)


    ######################## Calculation of W, Delta and mu with scipy #######################

    def sum_over_all_k_for_scipy(x):
        Term1 = -(t + x[0])
        Term2 = - t_diag
        res = zeros((4, 5), dtype=float)
        res_vec = zeros(5, dtype=float)
        for i in range(4):
            y0s = array(matrix_list_4S[0][i][:])
            y0d = array(matrix_list_4S[1][i][:])
            G = array(matrix_list_4S[2][i][:])
            y1 = eps_k(y0s, G, Term1, Term2)
            y2 = E_k(y1, y0s, y0d, x[1], x[2], x[3], U*x[4])
            y5 = tanh(y2 / (2.0 * kBT))
            y3 = (y1 - x[3]) / y2 * y5
            y6 = (x[1] * y0s + U*x[4]) / y2 * y5
            y7 = (x[2] * y0d) / y2 * y5
            y4s = y0s * y6
            y4d = y0d * y7
            res[i, 0] = sum(y0s * y3)
            res[i, 1] = sum(y4s)
            res[i, 2] = sum(y4d)
            res[i, 3] = sum(y3)
            res[i, 4] = sum(y6)
        for i in range(5):
            res_vec[i] = 4.0 * res[0, i] + 2.0 * res[1, i] + 2.0 * res[2, i] + res[3, i]
        return res_vec

    def calc_funcs(x, n_el):
        sum_res = sum_over_all_k_for_scipy(x)
        f0_func = c * sum_res[0] + x[0]
        f1_func = c * sum_res[1] + x[1]
        f2_func = c * sum_res[2] + x[2]
        f3_func = n_el - 1.0 + 1.0 / N * sum_res[3]
        f4_func = 1/(2.0*N) * sum_res[4] + x[4]
        return array([f0_func, f1_func, f2_func, f3_func, f4_func])

    ######################## Calculation of the Energy #######################

    def sum_over_all_k_for_E(x):
        Term1 = -(t + x[0])
        Term2 = - t_diag
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
        return 8.0 * res[0] + 4.0 * res[1] + 4.0 * res[2] + res[3]

    def calc_E(n_el, x):
        sum_res = sum_over_all_k_for_E(x)
        return sum_res / N - 4.0 / V1 * (x[1] ** 2 + x[2] ** 2 - x[0] ** 2) + x[3] * (n_el - 1.0) - x[4]**2*U


    def solve_by_scipy(n_el, x_start):
        sol = optimize.root(calc_funcs, x_start, args=(n_el))
        return sol.x



    return solve_by_scipy, calc_E
