from numpy import *
from scipy import optimize

def create_functions(V1, c, N, t, t_diag, matrix_list_8S, da, U, kBT, matrix_list_4S):
    def GA_CRI_ZZ(D, n):
        """
        GA_CRI_ZZ is the <zz> coefficient inside the epsilon_k from the charge rotational invariant Gutzwiller approximation
        :param D: probability for double occupation per site
        :param n: electron density
        :param cidciu: self correlation <c_{i, downarrow} c_{i, uparrow}>
        :return: the charge rotational invariant Gutzwiller coefficient
        """
        Jz = 0.5 * (n - 1.0)
        J = sqrt(Jz ** 2)
        annoying_root_term = D - Jz - J
        #if allclose(annoying_root_term, 0.0):
        #   annoying_root_term = 0.0
        if annoying_root_term < 0.0:
            annoying_root_term = 0.0
            #print('D = ', D, ', Jz = ', Jz, ', J = ', J, ', D - Jz - J =', D - Jz - J, 'Jx = ', 0.0)
            #raise ValueError("sqrt(D - Jz - J) is complex!")
        return 2.0*(0.5 + Jz - D) / (0.25 - J ** 2) * (D - Jz + sqrt(annoying_root_term) * sqrt(D - Jz + J))

    @vectorize
    def eps_k(gks, costerm, Term1, Term2):
        return Term1 * gks + Term2 * costerm

    @vectorize
    def E_k(ek, gk, Delta, mu):
        return sqrt((ek - mu) ** 2 + (Delta * gk) ** 2)


    def f0_sum_for_newton(y):
        return y[0]

    def f1_sum_for_newton(y):
        return y[1]

    def f2_sum_for_newton(y):
        return y[2]


    ######################## Calculation of the functions for scipy #######################
    def calc_y(x, i, n_el, D):
        y0s_8S = array(matrix_list_8S[0][i][:])
        y0d_8S = array(matrix_list_8S[1][i][:])
        G_8S = array(matrix_list_8S[2][i][:])
        Term1 = -(t * GA_CRI_ZZ(D, n_el) + x[0])
        Term2 = - t_diag * GA_CRI_ZZ(D, n_el)
        y1_8S = eps_k(y0s_8S, G_8S, Term1, Term2)
        y2_8S = E_k(y1_8S, y0d_8S, x[1], x[2])
        y3_8S = (y1_8S - x[2]) / y2_8S
        y5_8S = tanh(y2_8S / (2.0 * kBT))

        W = y0s_8S * y3_8S * y5_8S
        Delta = (x[1] * y0d_8S * y0d_8S) / y2_8S * y5_8S
        mu = y3_8S * y5_8S

        return array([W, Delta, mu])

    def sum_over_all_k_for_scipy(x, n_el, D):
        Function_list = [f0_sum_for_newton, f1_sum_for_newton, f2_sum_for_newton]
        res_matrix = zeros((4, 3), dtype=float)
        res_vec = zeros(3, dtype=float)
        for i in range(4):
            y = calc_y(x, i, n_el, D)
            for j in range(3):
                res_matrix[i, j] = sum(Function_list[j](y))
        for i in range(3):
            res_vec[i] = 8.0 * res_matrix[0, i] + 4.0 * res_matrix[1, i] + 4.0 * res_matrix[2, i] + res_matrix[3, i]
        return res_vec

    def calc_funcs(x, n_el, D):
        sum_res = sum_over_all_k_for_scipy(x, n_el, D)
        f0_func = c * sum_res[0] + x[0]  # W
        f1_func = c * sum_res[1] + x[1]  # Delta
        f2_func = n_el - 1.0 + 1.0 / N * sum_res[2]  # mu
        return array([f0_func, f1_func, f2_func])


    def calc_funcs_with_E(x, n_el):
        sum_res = sum_over_all_k_for_scipy(x, n_el, x[3])
        f0_func = c * sum_res[0] + x[0]  # W
        f1_func = c * sum_res[1] + x[1]  # Delta
        f2_func = n_el - 1.0 + 1.0 / N * sum_res[2]  # mu
        f3_func = (calc_E(x[3] + da, n_el, x) - calc_E(x[3] - da, n_el, x)) / (2.0 * da)  # dE/dD

        #Jz = 0.5 * (n_el - 1.0)
        #J = abs(Jz)
        #sXY = sqrt(x[3] - Jz - J) * sqrt(x[3] - Jz + J)
        #v = n_el / 2.0 - x[3]
        #z2 = GA_CRI_ZZ(x[3], n_el)
        #A = z2 * (0.25 - J ** 2) / (2.0 * v)
        #if U == 0.0:
        #    f3_func = sXY - v
        #else:
        #    f3_func = -2.0 * A * (v - sXY) + U*(v - z2 * sXY)   # dE/dD
        return array([f0_func, f1_func, f2_func, f3_func])


    ######################## Calculation of the Energy #######################


    def sum_over_all_k_for_E(x, n_el, D):
        res_vec = zeros(4, dtype=float)
        for i in range(4):
            y0s = array(matrix_list_8S[0][i][:])
            y0d = array(matrix_list_8S[1][i][:])
            G = array(matrix_list_8S[2][i][:])
            Term1 = -(t * GA_CRI_ZZ(D, n_el) + x[0])
            Term2 = - t_diag * GA_CRI_ZZ(D, n_el)
            y1 = eps_k(y0s, G, Term1, Term2)
            y2 = E_k(y1, y0d, x[1], x[2])
            y5 = tanh(y2 / (2.0 * kBT))
            res_vec[i] = sum(y1 - y2 * y5)
        return 8.0 * res_vec[0] + 4.0 * res_vec[1] + 4.0 * res_vec[2] + res_vec[3]

    def calc_E(D, n_el, x):
        sum_res = sum_over_all_k_for_E(x, n_el, D)
        return sum_res / N - 4.0 / V1 * (x[1] ** 2 - x[0] ** 2) + x[2] * (n_el - 1.0) + U * D

    ######################## Calculation of W, Delta, mu #######################

    accuracy = 1.0e-7  # accuracy for the iteration
    algo_switch = 0

    def solve_by_scipy(n_el, x_start):
        delta_x = 1.0
        delta_D = 1.0
        cnt = 0
        cidciu_start = 0.0  # in d-wave symmetry the self correlation is always zero
        lamda_start = 0.0  # ... therefore also the lagrange multiplier
        D_start = x_start[4]/1.0
        Jz = 0.5 * (n_el - 1.0)

        if algo_switch == 0:
            x_all_start = array([x_start[0], x_start[1], x_start[2]])
        elif algo_switch == 1:
            x_all_start = array([x_start[0], x_start[1]*1.0, x_start[2], D_start])

        J = sqrt(Jz ** 2)
        left_bound = Jz + J

        if t_diag < 0.0:
            if n_el < 0.5:
                alpha = 0.5
            else:
                alpha = 0.01
        else:
            alpha = 0.5

        while delta_x > accuracy:

            if algo_switch == 0:
                x_all_new = (optimize.root(calc_funcs, x_all_start, args=(n_el, D_start), tol=1.0e-12)).x

                D_new = (optimize.minimize_scalar(calc_E, bounds=(left_bound, n_el ** 2 / 4.0),
                                                  args=(n_el, x_all_start),
                                                  method='bounded', options={'xatol': 1.0e-12})).x

                D_new = alpha * D_new + (1.0 - alpha) * D_start
                delta_D = abs(D_new - D_start)
                D_start = array(D_new)
            elif algo_switch == 1:
                x_all_new = (optimize.root(calc_funcs_with_E, x_all_start, args=(n_el), tol=1.0e-12)).x
                if x_all_new[0] > 0.0:
                    print('Problem: x_all_new[0] = ', x_all_new[0])
                    x_all_new[0] = array(x_start[0])
                if x_all_new[1] > 0.0:
                    print('Problem: x_all_new[1] = ', x_all_new[1])
                    x_all_new[1] = array(x_start[1])
                if x_all_new[3] < 0.0:
                    print('Problem: x_all_new[3] = ', x_all_new[3])
                    x_all_new[3] = array(D_start)
                #x_all_new[3] = alpha*(optimize.minimize_scalar(calc_E, bounds=(left_bound, n_el ** 2 / 4.0),
                #                                  args=(n_el, x_all_new),
                #                                  method='bounded', options={'xatol': 1.0e-12})).x + (1.0 - alpha) * D_start
                #D_start = array(x_all_new[3])

            delta_x = abs(x_all_new - x_all_start).max()
            x_all_start[:] = x_all_new[:]


            cnt += 1
            if cnt > 200:
                break
            if algo_switch == 0:
                print('cnt = ', cnt, ', delta_x = ', delta_x, 'delta_D = ', delta_D, 'x_all = ', x_all_start, 'D = ', D_start)
            elif algo_switch == 1:
                print('cnt = ', cnt, ', delta_x = ', delta_x, 'x_all = ', x_all_start)
        if algo_switch == 0:
            result = array([x_all_start[0], x_all_start[1], x_all_start[2], D_start])
        elif algo_switch == 1:
            result = array([x_all_start[0], x_all_start[1], x_all_start[2], x_all_start[3]])
        return array(result), cidciu_start, lamda_start


    return solve_by_scipy, calc_E, GA_CRI_ZZ
