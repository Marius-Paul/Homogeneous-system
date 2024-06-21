from numpy import vectorize, array, zeros, sqrt, sum, allclose, tanh, shape
from scipy import optimize

def create_functions(V1, c, N, t, t_diag, matrix_list_8S, da, U, kBT, matrix_list_4S):

    def GA_CRI_ZZ(D, n, cidciu):
        Jx = cidciu
        Jz = 0.5 * (n - 1.0)
        J = sqrt(Jx ** 2 + Jz ** 2)
        annoying_root_term = D - Jz - J
        #if allclose(annoying_root_term, 0.0):
        #    annoying_root_term = 0.0
        if annoying_root_term < 0.0:
            #print('D = ', D, ', Jz = ', Jz, ', J = ', J, ', D - Jz - J =', D-Jz-J, 'Jx = ', Jx)
            #raise ValueError("sqrt(D - Jz - J) is complex!")
            annoying_root_term = 0.0
            return 2.0 * (0.5 + Jz - D) / (0.25 - J ** 2) * (D - Jz + sqrt(annoying_root_term) * sqrt(D - Jz + J))
        else:
            return 2.0 * (0.5 + Jz - D) / (0.25 - J ** 2) * (D - Jz + sqrt(annoying_root_term) * sqrt(D - Jz + J))

    def d_GA_CRI_ZZ_dJx(D, n, cidciu):
        Jx = cidciu
        Jz = 0.5 * (n - 1.0)
        J = sqrt(Jx ** 2 + Jz ** 2)
        return 2.0 * Jx / (0.25 - J ** 2) * (GA_CRI_ZZ(D, n, cidciu) - (n / 2.0 - D) / sqrt(D ** 2 - 2.0 * D * Jz - Jx ** 2))

    @vectorize
    def eps_k(gks, G, Term1, Term2):
        return Term1 * gks + Term2 * G

    @vectorize
    def E_k(ek, gks, gkd, Delta_s, Delta_d, mu, lamda):
        return sqrt((ek - mu) ** 2 + (Delta_s * gks + lamda) ** 2 + (Delta_d * gkd) ** 2)

    # W
    def f0_sum(y):
        return y[0] * y[3]

    # Delta_s
    def f1_sum(y):
        return y[4]

    # Delta_d
    def f2_sum(y):
        return y[5]

    # mu
    def f3_sum(y):
        return y[3]

    # cidciu
    def f4_sum(y):
        return y[7]

    # lamda
    def f5_sum(y):
        return y[8]

    # x = [W, Delta_s, Delta_d, mu, cidciu, lamda]
    ######################## Calculation of the functions for scipy #######################
    def calc_y(x, i, n_el, D):
        y0s_8S = array(matrix_list_8S[0][i][:])
        y0d_8S = array(matrix_list_8S[1][i][:])
        G_8S = array(matrix_list_8S[2][i][:])
        Term1 = -(t * GA_CRI_ZZ(D, n_el, x[4]) + x[0])
        Term2 = - t_diag * GA_CRI_ZZ(D, n_el, x[4])
        y1 = eps_k(y0s_8S, G_8S, Term1, Term2)
        y2 = E_k(y1, y0s_8S, y0d_8S, x[1], x[2], x[3], x[5])
        y5 = tanh(y2 / (2.0 * kBT))
        y3 = (y1 - x[3]) / y2 * y5
        y6 = (x[1] * y0s_8S + x[5]) / y2 * y5
        y4s = y0s_8S*y6
        y0s_4S = array(matrix_list_4S[0][i][:])
        y0d_4S = array(matrix_list_4S[1][i][:])
        G_4S = array(matrix_list_4S[2][i][:])
        y2_4S = E_k(eps_k(y0s_4S, G_4S, Term1, Term2), y0s_4S, y0d_4S, x[1], x[2], x[3], x[5])
        y5_4S = tanh(y2_4S / (2.0 * kBT))
        y4d = y0d_4S*(x[2] * y0d_4S + x[5]) / y2_4S * y5_4S
        eps0_k = -(t * y0s_8S + t_diag * G_8S)
        y7 = eps0_k * y3
        return [y0s_8S, y1, y2, y3, y4s, y4d, y5, y6, y7]

    def sum_over_all_k_for_scipy(x, n_el, D):
        Function_list = [f0_sum, f1_sum, f2_sum, f3_sum, f4_sum, f5_sum]
        res_matrix = zeros((4, 6), dtype=float)
        res_vec = zeros(6, dtype=float)
        for i in range(4):
            y = calc_y(x, i, n_el, D)
            for j in range(6):
                res_matrix[i, j] = sum(Function_list[j](y))
        for i in range(6):
            if i == 2:
                res_vec[i] = 4.0 * res_matrix[0, i] + 2.0 * res_matrix[1, i] + 2.0 * res_matrix[2, i] + res_matrix[3, i]
            else:
                res_vec[i] = 8.0 * res_matrix[0, i] + 4.0 * res_matrix[1, i] + 4.0 * res_matrix[2, i] + res_matrix[3, i]
        return res_vec

    def calc_funcs(x, n_el, D):
        sum_res = sum_over_all_k_for_scipy(x, n_el, D)
        f0_func = c * sum_res[0] + x[0]     # W
        f1_func = c * sum_res[1] + x[1]     # Delta_s
        f2_func = c * sum_res[2] + x[2]     # Delta_d
        f3_func = n_el - 1.0 + 1.0 / N * sum_res[3]     # mu
        f4_func = 1.0 / (2.0 * N) * sum_res[4] + x[4]   # cidciu
        f5_func = x[5] + (GA_CRI_ZZ(D, n_el, x[4] + da) - GA_CRI_ZZ(D, n_el, x[4] - da)) / (2.0 * da) * sum_res[5] / (2.0*N)    # lamda
        return array([f0_func, f1_func, f2_func, f3_func, f4_func, f5_func])

    def calc_funcs_with_E(x, n_el):
        sum_res = sum_over_all_k_for_scipy(x, n_el, x[6])
        f0_func = c * sum_res[0] + x[0]     # W
        f1_func = c * sum_res[1] + x[1]     # Delta_s
        f2_func = c * sum_res[2] + x[2]     # Delta_d
        f3_func = n_el - 1.0 + 1.0 / N * sum_res[3]     # mu
        f4_func = 1.0 / (2.0 * N) * sum_res[4] + x[4]   # cidciu
        f5_func = x[5] + (GA_CRI_ZZ(x[6], n_el, x[4] + da) - GA_CRI_ZZ(x[6], n_el, x[4] - da)) / (2.0 * da) * sum_res[5] / (2.0*N)    # lamda
        #f5_func = x[5] + d_GA_CRI_ZZ_dJx(x[6], n_el, x[4]) * sum_res[5] / (2.0 * N)    # lamda

        #Jz = 0.5 * (n_el - 1.0)
        #J = sqrt(x[4] ** 2 + Jz ** 2)
        #sXY = sqrt(x[6] - Jz - J) * sqrt(x[6] - Jz + J)
        #v = n_el / 2.0 - x[6]
        #z2 = GA_CRI_ZZ(x[6], n_el, x[4])
        #A = z2 * (0.25 - J ** 2) / (2.0 * v)
        #if U == 0.0:
        #    f6_func = (sXY - v)
        #else:
        #    f6_func = -2.0 * x[5] * A * (sXY - v) / (v - z2 * sXY) + U * x[4]  # dE/dD
        f6_func = (calc_E(x[6]+da, n_el, x) - calc_E(x[6]-da, n_el, x)) / (2.0 * da)   # dE/dD
        return array([f0_func, f1_func, f2_func, f3_func, f4_func, f5_func, f6_func])



    ######################## Calculation of the Energy #######################

    def sum_over_all_k_for_E(x, n_el, D):
        res_vec = zeros(4, dtype=float)
        Term1 = -(t * GA_CRI_ZZ(D, n_el, x[4]) + x[0])
        Term2 = - t_diag * GA_CRI_ZZ(D, n_el, x[4])
        for i in range(4):
            y0s = array(matrix_list_8S[0][i][:])
            y0d = array(matrix_list_8S[1][i][:])
            G = array(matrix_list_8S[2][i][:])
            y1 = eps_k(y0s, G, Term1, Term2)
            y2 = E_k(y1, y0s, y0d, x[1], x[2], x[3], x[5])
            y5 = tanh(y2 / (2.0 * kBT))
            res_vec[i] = sum(y1 - y2 * y5)
        return 8.0 * res_vec[0] + 4.0 * res_vec[1] + 4.0 * res_vec[2] + res_vec[3]

    def calc_E(D, n_el, x):
        sum_res = sum_over_all_k_for_E(x, n_el, D)
        return sum_res / N - 4.0 / V1 * (x[1] ** 2 + x[2] ** 2 - x[0] ** 2) + x[3] * (n_el - 1.0) + U * D - 2.0 * x[4] * x[5]




    ######################## Calculation of W, Delta, mu #######################

    accuracy = 1.0e-8  # accuracy for the iteration
    alpha = 0.5  # relaxation parameter
    algo_switch = 0

    def solve_by_scipy(n_el, x_start):
        delta_x = 1.0
        delta_D = 1.0
        cnt = 0
        D_start = array(x_start[5])/2.0
        cidciu_start = array(x_start[4])/1.0
        lamda_start = array(x_start[4]*U)/1.0

        Jz = 0.5 * (n_el - 1.0)

        if algo_switch == 0:
            x_all_start = array([x_start[0], x_start[1]*4.0, x_start[3]*1.0, x_start[2], cidciu_start, lamda_start])
        elif algo_switch == 1:
            x_all_start = array([x_start[0], x_start[1]*2.0, x_start[3]*2.0, x_start[2]*1.0, cidciu_start, lamda_start, D_start])

        while delta_x>accuracy:   #  or delta_D>accuracy:

            # print('left_bound for D = ', left_bound, 'right_bound for D = ', n_el_test ** 2 / 4.0)


            if algo_switch == 0:
                x_all_new = (optimize.root(calc_funcs, x_all_start, args=(n_el, D_start), tol=1.0e-12)).x


                if x_all_new[1] > 0.0:
                    x_all_new[1] *= -1.0
                if x_all_new[2] > 0.0:
                    x_all_new[2] *= -1.0
                if x_all_new[4] < 0.0:
                    x_all_new[4] *= -1.0

                J = sqrt(x_all_start[4] ** 2 + Jz ** 2)
                left_bound = Jz + J
                D_new = (optimize.minimize_scalar(calc_E, bounds=(left_bound, n_el ** 2 / 4.0),
                                                  args=(n_el, x_all_new),
                                                  method='bounded', options={'xatol': 1.0e-12})).x

                delta_D = abs(D_new - D_start)
                D_start = alpha * D_new + (1.0 - alpha) * D_start


            elif algo_switch == 1:
                x_all_new = (optimize.root(calc_funcs_with_E, x_all_start, args=(n_el), tol=1.0e-12)).x
                if x_all_new[1] > 0.0:
                    x_all_new[1] *= -1.0
                if x_all_new[2] > 0.0:
                    x_all_new[2] *= -1.0
                #if x_all_new[4] < 0.0:
                #    x_all_new[4] = cidciu_start
                if x_all_new[6] < 0.0:
                    x_all_new[6] = D_start

            delta_x = abs(x_all_new - x_all_start).max()
            x_all_start[:] = x_all_new[:]

            cnt += 1
            if cnt > 120:
                break
            if algo_switch == 0:
                print('cnt = ', cnt, ', delta_x = ', delta_x, 'delta_D = ', delta_D, ', D = ', D_start, ', x = ', x_all_start)
            elif algo_switch == 1:
                print('cnt = ', cnt, ', delta_x = ', delta_x, ', x = ', x_all_start)

        if algo_switch == 0:
            result = array([x_all_start[0], x_all_start[1], x_all_start[2], x_all_start[3], D_start])
        elif algo_switch == 1:
            result = array([x_all_start[0], x_all_start[1], x_all_start[2], x_all_start[3], x_all_start[6]])
        return array(result), x_all_start[4], x_all_start[5]


    return solve_by_scipy, calc_E, GA_CRI_ZZ
