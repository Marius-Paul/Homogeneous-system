from numpy import vectorize, array, zeros, insert, savetxt, sqrt, cos, exp, sum, tanh
from scipy import optimize


def create_functions(V1, c, N, t, t_diag, matrix_list, kBT, U):
    @vectorize
    def eps_k(gks, G, Term1, Term2):
        return Term1 * gks + Term2 * G

    @vectorize
    def E_k(ek, gks, Delta_s, mu, L):
        return sqrt((ek - mu) ** 2 + (Delta_s * gks + L) ** 2)

    def f0_sum_for_newton(y):
        return y[0] * y[3]

    def f1_sum_for_newton(y):
        return y[4]

    def f2_sum_for_newton(y):
        return y[3]

    def f3_sum_for_newton(y):
        return y[5]

    ######################## Calculation of W, Delta and mu with scipy #######################
    def calc_y(x, i):
        y0s = array(matrix_list[0][i][:])
        G = array(matrix_list[2][i][:])
        Term1 = -(t + x[0])
        Term2 = - t_diag
        y1 = eps_k(y0s, G, Term1, Term2)
        y2 = E_k(y1, y0s, x[1], x[2], U*x[3])
        y5 = tanh(y2 / (2.0 * kBT))
        y3 = (y1 - x[2]) / y2 * y5
        y6 = (x[1] * y0s + U*x[3]) / y2 * y5
        y4s = y0s * y6
        return array([y0s, y1, y2, y3, y4s, y6])

    def sum_over_all_k_for_scipy(x):
        Function_list = [f0_sum_for_newton, f1_sum_for_newton, f2_sum_for_newton, f3_sum_for_newton]
        res_matrix = zeros((4, 4), dtype=float)
        res_vec = zeros(4, dtype=float)
        for i in range(4):
            y = calc_y(x, i)
            for j in range(4):
                res_matrix[i, j] = sum(Function_list[j](y))
        for i in range(4):
            res_vec[i] = 8 * res_matrix[0, i] + 4 * res_matrix[1, i] + 4 * res_matrix[2, i] + res_matrix[3, i]
        return res_vec

    def calc_funcs(x, n_el):
        sum_res = sum_over_all_k_for_scipy(x)
        f0_func = c * sum_res[0] + x[0]
        f1_func = c * sum_res[1] + x[1]
        f2_func = n_el - 1.0 + 1.0 / N * sum_res[2]
        f3_func = 1.0/(2.0*N) * sum_res[3] + x[3]
        return array([f0_func, f1_func, f2_func, f3_func])

    ######################## Calculation of the Energy #######################
    def sum_over_all_k_for_E(x):
        res_vec = zeros(4, dtype = float)
        for i in range(4):
            y0s = array(matrix_list[0][i][:])
            G = array(matrix_list[2][i][:])
            Term1 = -(t + x[0])
            Term2 = - t_diag
            y1 = eps_k(y0s, G, Term1, Term2)
            y2 = E_k(y1, y0s, x[1], x[2], U*x[3])
            y5 = tanh(y2 / (2.0 * kBT))
            res_vec[i] += sum(y1 - y2 * y5)
        return 8.0*res_vec[0] + 4.0*res_vec[1] + 4.0*res_vec[2] + res_vec[3]

    def calc_E(n_el, x):
        sum_res = sum_over_all_k_for_E(x)
        return sum_res / N - 4.0 / V1 * (x[1] ** 2 - x[0] ** 2) + x[2] * (n_el - 1.0) - x[3]**2*U

    ######################## Calculation of the cidciu #######################
    @vectorize
    def gamma_ks(k_x, k_y, a=1.0):
        return 2.0 * (cos(a * k_x) + cos(a * k_y))

    @vectorize
    def cos_term(kx, ky, a=1.0):
        return 4.0 * cos(kx * a) * cos(ky * a)



    def solve_by_scipy(n_el, x_start):
        sol = optimize.root(calc_funcs, x_start, args=(n_el))
        x_start = sol.x
        if x_start[1] > 0.0:
            x_start[1] *= (-1.0)
            sol = optimize.root(calc_funcs, x_start, args=(n_el))
        return sol.x



    return solve_by_scipy, calc_E
