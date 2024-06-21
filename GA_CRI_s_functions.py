from numpy import vectorize, array, zeros, sqrt, sum, allclose, tanh, linspace
from scipy import optimize
from matplotlib.pyplot import plot, grid, show

def create_functions(V1, c, N, t, t_diag, matrix_list, U, kBT, da):

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
        #    return 2.0 * (0.5 + Jz - D) / (0.25 - J ** 2) * (D - Jz + sqrt(annoying_root_term) * sqrt(D - Jz + J))
        #else:
        return 2.0 * (0.5 + Jz - D) / (0.25 - J ** 2) * (D - Jz + sqrt(annoying_root_term) * sqrt(D - Jz + J))

    def d_GA_CRI_ZZ_dJx(D, n, cidciu):
        Jx = cidciu
        Jz = 0.5 * (n - 1.0)
        J = sqrt(Jx ** 2 + Jz ** 2)
        u = 0.25 - J ** 2
        v = n / 2.0 - D
        z2 = GA_CRI_ZZ(D, n, cidciu)
        return 2.0 * Jx / (0.25 - J ** 2) * (z2 - (n / 2.0 - D) / (z2 * u / (2.0 * v) - 0.5 + v))


    @vectorize
    def eps_k(gks, G, Term1, Term2):
        return Term1 * gks + Term2 * G

    @vectorize
    def E_k(ek, gk, Delta, mu, lamda):
        return sqrt((ek - mu) ** 2 + (Delta * gk + lamda) ** 2)

    # W
    def f0_sum_for_newton(y):
        return y[0] * y[3] * y[5]

    # Delta
    def f1_sum_for_newton(y):
        return y[4] * y[5]

    # mu
    def f2_sum_for_newton(y):
        return y[3] * y[5]

    # cidciu
    def f3_sum_for_newton(y):
        return y[6]

    # lamda
    def f4_sum_for_newton(y):
        return y[7]


    ######################## Calculation of the functions for scipy #######################
    def calc_y(x, i, n_el, D):
        y0s = array(matrix_list[0][i][:])
        G = array(matrix_list[2][i][:])
        Term1 = -(t * GA_CRI_ZZ(D, n_el, x[3]) + x[0])
        Term2 = - t_diag * GA_CRI_ZZ(D, n_el, x[3])
        y1 = eps_k(y0s, G, Term1, Term2)
        y2 = E_k(y1, y0s, x[1], x[2], x[4])
        y3 = (y1 - x[2]) / y2
        y4 = y0s*(x[1] * y0s + x[4]) / y2
        y5 = tanh(y2 / (2.0 * kBT))
        y6 = (x[1] * y0s + x[4]) / y2 * y5
        eps0_k = -(t * y0s + t_diag * G)
        y7 = eps0_k * y3 * y5
        return array([y0s, y1, y2, y3, y4, y5, y6, y7])

    def sum_over_all_k_for_scipy(x, n_el, D):
        Function_list = [f0_sum_for_newton, f1_sum_for_newton, f2_sum_for_newton, f3_sum_for_newton, f4_sum_for_newton]
        res_matrix = zeros((4, 5), dtype=float)
        res_vec = zeros(5, dtype=float)
        for i in range(4):
            y = calc_y(x, i, n_el, D)
            for j in range(5):
                res_matrix[i, j] = sum(Function_list[j](y))
        for i in range(5):
            res_vec[i] = 8.0 * res_matrix[0, i] + 4.0 * res_matrix[1, i] + 4.0 * res_matrix[2, i] + res_matrix[3, i]
        return res_vec


    def calc_funcs(x, n_el, D):
        sum_res = sum_over_all_k_for_scipy(x, n_el, D)
        f0_func = c * sum_res[0] + x[0]  # W
        f1_func = c * sum_res[1] + x[1]  # Delta
        f2_func = n_el - 1.0 + 1.0 / N * sum_res[2]     # mu
        f3_func = 1.0 / (2.0 * N) * sum_res[3] + x[3]   # cidciu
        f4_func = x[4] + d_GA_CRI_ZZ_dJx(D, n_el, x[3]) * sum_res[4] / (2.0 * N)  # lamda
        return array([f0_func, f1_func, f2_func, f3_func, f4_func])


    def dE_dD(D, n, cidciu, lamda):
        Jz = 0.5 * (n - 1.0)
        J = sqrt(cidciu ** 2 + Jz ** 2)
        u = 0.25 - J ** 2
        v = n / 2.0 - D
        z2 = GA_CRI_ZZ(D, n, cidciu)
        A = z2 * (u / (2.0 * v))
        sXY = z2*u/(2.0*v) - 0.5 + v
        return 2.0 *A*lamda / cidciu * (sXY - v) / (v - z2 * sXY) + U
        # return -lamda*u/(cidciu*v) * (1.0/(1 + (v*(1-z2))/(0.5*(z2 - z2**2*u/v)))) + U

    def calc_funcs_with_E(x, n_el):
        sum_res = sum_over_all_k_for_scipy(x, n_el, x[5])
        f0_func = c * sum_res[0] + x[0]  # W
        f1_func = c * sum_res[1] + x[1]  # Delta
        f2_func = n_el - 1.0 + 1.0 / N * sum_res[2]  # mu
        f3_func = 1.0 / (2.0 * N) * sum_res[3] + x[3]  # cidciu
        f4_func = (x[4] + (GA_CRI_ZZ(x[5], n_el, x[3]+da) - GA_CRI_ZZ(x[5], n_el, x[3]-da)) / (2.0 * da) * sum_res[4] / (2.0 * N))  # lamda
        #f4_func = (x[4] + d_GA_CRI_ZZ_dJx(x[5], n_el, x[3]) * sum_res[4] / (2.0 * N))  # lamda
        #f5_func = dE_dD(x[5], n_el, x[3], x[4]-f4_func)
        f5_func = (calc_E(x[5] + da, n_el, x) - calc_E(x[5] - da, n_el, x)) / (2.0 * da)  # dE/dD
        #print(abs(f5_func-f5_func_num)/f5_func, x)
        return array([f0_func, f1_func, f2_func, f3_func, f4_func, f5_func])



    ######################## Calculation of the Energy #######################

    def sum_over_all_k_for_E(x, n_el, D):
        res_vec = zeros(4, dtype=float)
        Term1 = -(t * GA_CRI_ZZ(D, n_el, x[3]) + x[0])
        Term2 = - t_diag * GA_CRI_ZZ(D, n_el, x[3])
        for i in range(4):
            y0s = array(matrix_list[0][i][:])
            G = array(matrix_list[2][i][:])
            y1 = eps_k(y0s, G, Term1, Term2)
            y2 = E_k(y1, y0s, x[1], x[2], x[4])
            y5 = tanh(y2 / (2.0 * kBT))
            res_vec[i] = sum(y1 - y2 * y5)
        return 8.0 * res_vec[0] + 4.0 * res_vec[1] + 4.0 * res_vec[2] + res_vec[3]

    def calc_E(D, n_el, x):
        sum_res = sum_over_all_k_for_E(x, n_el, D)
        return sum_res / N - 4.0 / V1 * (x[1] ** 2 - x[0] ** 2) + x[2] * (n_el - 1.0) + U * D - 2.0 * x[3] * x[4]



    ######################## Calculation of W, Delta, mu #######################

    accuracy = 1.0e-8  # accuracy for the iteration
    alpha = 0.5  # relaxation parameter
    algo_switch = 1  # 0 for the old algorithm, 1 for the new algorithm



    def solve_by_scipy(n_el, x_start):
        delta_x = 1.0
        delta_D = 1.0
        cnt = 0
        f = 2.0
        Jz = 0.5 * (n_el - 1.0)


        D_start = array(x_start[4])/f
        #right_bound = array(x_start[4])*1.1
        #cidciu_start = sqrt(D_start ** 2 - D_start * (n_el - 1.0))*0.5
        #print('cidciu_start = ', cidciu_start)
        lamda_start = array(x_start[3])*U
        cidciu_start = array(x_start[3])

        J = sqrt(cidciu_start** 2 + Jz ** 2)
        left_bound = Jz + J

        #cidciu_start = sqrt((left_bound/f)**2 - 2.0*Jz* (left_bound/f))
        print('cidciu_start = ', cidciu_start)
        print('D_start = ', D_start)
        #D_vec = linspace(0.0, right_bound, 1000)
        #plot(D_vec, dE_dD(D_vec, n_el, cidciu_start, lamda_start))
        #grid()
        #show()


        if algo_switch == 0:
            x_all_start = array([x_start[0], x_start[1], x_start[2], cidciu_start, lamda_start])
        elif algo_switch == 1:
            delta_D = 0.0
            x_all_start = array([x_start[0], x_start[1]*2.0, x_start[2], cidciu_start, lamda_start, D_start])


        while delta_x>accuracy or delta_D>accuracy:


            # print('left_bound for D = ', left_bound, 'right_bound for D = ', n_el_test ** 2 / 4.0)
            if algo_switch == 1:
                x_all_new = (optimize.root(calc_funcs_with_E, x_all_start, args=(n_el))).x
                if x_all_new[0] > 0.0:
                    print('x_all_new[0] = ', x_all_new[0])
                    x_all_new[0] = array(x_start[0])
                if x_all_new[1] > 0.0:
                    print('x_all_new[1] = ', x_all_new[1])
                    x_all_new[1] = array(x_start[1])
                if x_all_new[2] > 0.0:
                    print('x_all_new[2] = ', x_all_new[2])
                    x_all_new[2] = array(x_start[2])

                if x_all_new[5] < 0.0:
                    print('x_all_new[5] = ', x_all_new[5])
                    x_all_new[5] = array(D_start)


            else:
                x_all_new = (optimize.root(calc_funcs, x_all_start, args=(n_el, D_start))).x
                if x_all_new[0] > 0.0:
                    x_all_new[0] = array(x_start[0])
                if x_all_new[1] > 0.0:
                    x_all_new[1] = array(x_start[1])
                if x_all_new[3] < 0.0:
                    x_all_new[3] = array(cidciu_start)
                if x_all_new[4] < 0.0:
                    x_all_new[4] = array(lamda_start)
                J = sqrt(x_all_new[3] ** 2 + Jz ** 2)

            if algo_switch == 0:
                left_bound = Jz + J
                #if left_bound > right_bound:
                #    print('left_bound = ', left_bound, 'right_bound = ', right_bound)
                #    print('cidciu_start = ', x_all_new[3], 'Jz = ', Jz, 'J = ', J)
                #    raise ValueError('left_bound > right_bound')
                #D_new = (optimize.minimize_scalar(calc_E, bounds=(left_bound, right_bound),
                #                                  args=(n_el, x_all_new),
                #                                  method='bounded', options={'xatol': 1.0e-12})).x
                solution = optimize.fsolve(dE_dD, D_start, args=(n_el, x_all_new[3], x_all_new[4]))
                #print('solution = ', solution)
                D_new = solution[0]
                delta_D = abs(D_new - D_start)
                D_start = alpha * D_new + (1.0 - alpha) * D_start
                if D_start <= left_bound:
                    raise ValueError('D_start < left_bound')
                    #D_start = array(left_bound)


            delta_x = abs(x_all_new - x_all_start).max()

            x_all_start[:] = x_all_new[:]



            cnt += 1
            if cnt > 30:
                break

            if algo_switch == 0:
                print('cnt = ', cnt, ', delta_x = ', delta_x, 'delta_D = ', delta_D, ', D = ', D_start, ', x = ', x_all_start)
            elif algo_switch == 1:
                print('cnt = ', cnt, ', delta_x = ', delta_x, ', x = ', x_all_start)
        if algo_switch == 1:
            result = array([x_all_start[0], x_all_start[1], x_all_start[2], x_all_start[5]])
        else:
            result = array([x_all_start[0], x_all_start[1], x_all_start[2], D_start])
        return array(result), x_all_start[3], x_all_start[4]


    return solve_by_scipy, calc_E, GA_CRI_ZZ
