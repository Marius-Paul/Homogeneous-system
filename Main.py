from Cosine_Terms import generating_matrices_of_cosine_combinations
from plot_results import *

from brillouin_zone import *
from split_function import *
from numpy import *

# if parallel computing is activated, s- and d-symmetry are calculated simultaneously
parallel_computing = False
if parallel_computing:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    rank = 0

############################## Options of the simulation ############################

# consider also Gutzwiller approximation
GA = False  # make sure to run the simulation without GA first, to get the initial values for GA simulation

# solves the system of equations for W, Delta, mu (and D if GA is activated) for several electron densities and both s- and d-symmetry
solve_equations = False

# plot the results (if the results are already there and don't need to be calculated again: switch off solve_equations)
plot_results = True
plot_s_id = True  # if True, the results for s_id-symmetry are plotted
plot_for_poster = True     # plot both, HF and GA results in one plot for comparison

# If GA is not activated, this option is for calculation of the Energy E for different D from the Gutzwiller Approximation for ONE given electron
# density and for s- and d-symmetry. solve_equations need to be activated for this.
solve_for_one_density_only_and_check_D = False   # only for no GA!
if not solve_equations and solve_for_one_density_only_and_check_D:
    raise ValueError("solve_for_one_density_only_and_check_D only makes sense if solve_equations is activated")


############################## Parameters of the model ############################

kBT = 1.0e-5  # temperature
grid_points = 400  # dimension of grid, has to be an even integer!
V1 = -2.0   # (constant) potential energy for next neighbour interaction
N = (grid_points-1)*(grid_points-1)  # total number of grid points
c = V1/(8.0*N)  # a constant which is often needed in the calculations
da = 1.0e-6  # the accuracy of derivatives
a = 1.0  # grid constant
t = 1.0  # hopping parameter for horizontal and vertical neighbours
t_diag = 0.0  # hopping parameter for diagonal neighbours, should take values like 0.0, -0.2, -0.4

# U is the potential energy of the interaction of an electron at a site with another electron on the same site
U = 4.0 # should take values like 4, 8, 12


############################## Printing the parameters of the model ############################
if rank == 0 or not parallel_computing:
    # calculate the electron densities
    print('################################ parameters ###################################')
    print('grid: ', grid_points, 'x', grid_points)
    print('kBT =', kBT)
    print('Number of gridpoints N = ', N)
    # potential energy
    print('V1 =', V1)
    print('t\' =', t_diag)
    if GA:
        print('Use Gutzwiller approximation')
    else:
        print('No Gutzwiller approximation\n')
    print('U = ', U)

############################## Parameters of the simulation ############################

n_el_start, n_el_end = 0.27, 0.33  # range of the densities for which the equations are solved
nod = 40  # number of densities within that range
symmetries = ['s_id']  # symmetries for which the equations are solved

# In case that GA is not used, initial values for W, Delta and mu for the FIRST density are required for the Newton method
if not GA:
    # initial values for s-symmetry
    W_start_s = -0.2
    Delta_start_s = -0.2
    mu_start_s = -1.8
    L_start_s = 0.1

    # initial values for d-symmetry
    W_start_d = -0.22
    Delta_start_d = -0.004
    mu_start_d = -2.0
    #L_start_d = 0.0

############################## Printing the parameters of the simulation ############################
if rank == 0 or not parallel_computing:
    print('################################ simulation ###################################')
    print('Solve the equations for ', nod, ' electron densities between ', n_el_start, ' and ', n_el_end)
    print('################################################################################')



############################## Initializing all arrays and matrices for the simulation ############################
electron_density_array = linspace(n_el_start, n_el_end, nod)  # array which contains all densities
set_initial_values_manually_for_specific_densities = False

if set_initial_values_manually_for_specific_densities:
    el_dens_number1 = 22
    el_dens_number2 = 24
    electron_density_array = electron_density_array[el_dens_number1:el_dens_number2]  # array which contains some specific densities
    n_el_start, n_el_end = electron_density_array[0], electron_density_array[-1]  # range of the densities for which the equations are solved
    nod = len(electron_density_array)  # number of densities within that range


# array with the initial values for W, Delta, mu for the first density (for the no GA case)
if not GA:
    start_values_for_very_first_density_s = array([W_start_s, Delta_start_s, mu_start_s, L_start_s])
    start_values_for_very_first_density_d = array([W_start_d, Delta_start_d, mu_start_d])



k_array_complete, area_list_8S, area_list_4S = create_grids(grid_points)  # k_array_complete contains ALL points withing the 1. Brillouin-Zone
# (needed for summations of functions which are not symmetric inside the zone); area_list contains 4 areas which together form
# one 8th part of the Brillouin-Zone. The areas are weighted differently, that's why they need to be conidered separately
#global matrix_list
matrix_list_8S = generating_matrices_of_cosine_combinations(area_list_8S)  # generating the cosine terms for the different areas
matrix_list_4S = generating_matrices_of_cosine_combinations(area_list_4S)  # generating the cosine terms for the different areas





if solve_equations:
    # Creating the folder for the results, in case it doesn't exist yet
    if rank == 0 or not parallel_computing:
        if GA:
            folder_name = 'results_with_GA' + '_U_' + str(U) + '_t_diag_' + str(t_diag) + '_V1_' + str(V1)
        else:
            folder_name = 'results_no_GA' + '_U_' + str(U) + '_t_diag_' + str(t_diag) + '_V1_' + str(V1)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        path = './' + folder_name + '/'

    for symmetry in symmetries:
        if rank == 0 or not parallel_computing:
            print('symmetry = ', symmetry)

            if symmetry == 's_id' and not GA:
                start_values_for_s_id = zeros((nod, 5), dtype=float)
                folder_name_no_GA = 'results_no_GA' + '_U_' + str(U) + '_t_diag_' + str(t_diag) + '_V1_' + str(V1)
                path_no_GA = './' + folder_name_no_GA + '/'
                values_s = 'res_array_' + 's' + '_no_GA'
                values_d = 'res_array_' + 'd' + '_no_GA'
                s_solution = loadtxt(os.path.join(path_no_GA, values_s))
                d_solution = loadtxt(os.path.join(path_no_GA, values_d))

                start_values_for_s_id[0:int(nod / 2), 0] = s_solution[0:int(nod / 2), 0]
                start_values_for_s_id[0:int(nod / 2), 1] = s_solution[0:int(nod / 2), 1]
                start_values_for_s_id[0:int(nod / 2), 2] = d_solution[0:int(nod / 2), 1]
                start_values_for_s_id[0:int(nod / 2), 3] = s_solution[0:int(nod / 2), 2]
                start_values_for_s_id[0:int(nod / 2), 4] = s_solution[0:int(nod / 2), 3]

                start_values_for_s_id[int(nod / 2):, 0] = d_solution[int(nod / 2):, 0]
                start_values_for_s_id[int(nod / 2):, 1] = s_solution[int(nod / 2):, 1]
                start_values_for_s_id[int(nod / 2):, 2] = d_solution[int(nod / 2):, 1]
                start_values_for_s_id[int(nod / 2):, 3] = d_solution[int(nod / 2):, 2]
                start_values_for_s_id[int(nod / 2):, 4] = d_solution[int(nod / 2):, 3]
                # print('start_values_for_s_id = ', start_values_for_s_id)


        # for GA the initial values for the iteration are taken from the Hartree-Fock result
        if GA:
            folder_name_no_GA = 'results_no_GA' + '_U_' + str(U) + '_t_diag_' + str(t_diag) + '_V1_' + str(V1)
            path_no_GA = './' + folder_name_no_GA + '/'
            name_for_start_values = 'res_array_' + symmetry + '_no_GA'
            GA_start_values = loadtxt(os.path.join(path_no_GA, name_for_start_values))
            if set_initial_values_manually_for_specific_densities:
                factor = 4.0
                GA_start_values = GA_start_values[el_dens_number1:el_dens_number2, :]



        # create the functions for the calculation of the densities
        if not GA and symmetry == 's':
            from s_functions import *
            solve_fct, calc_E = create_functions(V1, c, N, t, t_diag, matrix_list_8S, kBT, U)
        elif not GA and symmetry == 'd':
            from d_functions import *
            solve_fct, calc_E = create_functions(V1, c, N, t, t_diag, matrix_list_4S, kBT, U)
        elif not GA and symmetry == 's_id':
            from s_id_functions import *
            solve_fct, calc_E = create_functions(V1, c, N, t, t_diag, matrix_list_8S, matrix_list_4S, kBT, U)
        elif GA and symmetry == 's':
            from GA_CRI_s_functions import *
            solve_fct, calc_E, calc_ZZ = create_functions(V1, c, N, t, t_diag, matrix_list_8S, U, kBT, da)
        elif GA and symmetry == 'd':
            from GA_CRI_d_functions import *
            solve_fct, calc_E, calc_ZZ = create_functions(V1, c, N, t, t_diag, matrix_list_8S, da, U, kBT, matrix_list_4S)
        elif GA and symmetry == 's_id':
            from GA_CRI_s_id_functions import *
            solve_fct, calc_E, calc_ZZ = create_functions(V1, c, N, t, t_diag, matrix_list_8S, da, U, kBT, matrix_list_4S)
        else:
            raise ValueError("no valid choice for simulation options")

        # create arrays for the results
        if rank == 0 or not parallel_computing:
            if GA:
                if symmetry == 's' or symmetry == 'd':
                    Solution_arr = zeros((nod, 4), dtype=float)
                elif symmetry == 's_id':
                    Solution_arr = zeros((nod, 5), dtype=float)
            else:
                if symmetry == 's' or symmetry == 'd':
                    Solution_arr = zeros((nod, 4), dtype=float)
                elif symmetry == 's_id':
                    Solution_arr = zeros((nod, 5), dtype=float)

            E_res_arr = zeros(nod, dtype=float)
            if symmetry == 's' or symmetry == 'd':
               cidciu_res_arr = zeros(nod, dtype=float)
            elif symmetry == 's_id':
                cidciu_res_arr = zeros(nod, dtype = complex)
            density_array = zeros(nod, dtype=float)
            Delta_res_arr = zeros(nod, dtype=float)
            if symmetry == 's_id':
                Delta_d_res_arr = zeros(nod, dtype=float)
            W_res_arr = zeros(nod, dtype=float)
            mu_res_arr = zeros(nod, dtype=float)
            if GA:
                D_res_arr = zeros(nod, dtype=float)
                ZZ_res_arr = zeros(nod, dtype=float)
                lamda_res_arr = zeros(nod, dtype=float)
            if parallel_computing:
                n_el_array_split = split_array(electron_density_array, size)
                n_el_index_split = split_array(range(nod), size)
        elif parallel_computing:
            n_el_array_split = None
            n_el_index_split = None

        # Here the equations are solved in parallel for each density (if parallel computing is activated)
        if parallel_computing:
            n_el = comm.scatter(n_el_array_split, root=0)
            n_el_index_list = comm.scatter(n_el_index_split, root=0)
            if rank == 0:
                print('\t \t \t\t \t\t\t', ' W \t\t', ' Delta \t', 'mu \t\t', ' d')
            # create arrays for the results for each rank
            solution_array_rank = []
            W_array_rank = []
            Delta_array_rank = []
            if symmetry == 's_id':
                Delta_d_array_rank = []
            mu_array_rank = []
            E_array_rank = []
            cidciu_array_rank = []
            if GA:
                D_array_rank = []
                GA_ZZ_array_rank = []
            # now each rank solves the equations for a part of the densities
            cnt = 0
            for n_el_index in n_el_index_list:
                if GA:
                    start_values = list(GA_start_values[n_el_index, :])
                    start_values.append(n_el[cnt] ** 2 / 4.0)
                else:
                    if symmetry == 's':
                        start_values = list(start_values_for_very_first_density_s)
                    elif symmetry == 'd':
                        start_values = list(start_values_for_very_first_density_d)
                    elif symmetry == 's_id':
                        start_values = list(start_values_for_s_id)

                # Here the equations are solved!!!
                if GA:
                    solution, cidciu, lamda = solve_fct(n_el[cnt], array(start_values))
                else:
                    solution = solve_fct(n_el[cnt], array(start_values))

                # save the results in the arrays
                solution_array_rank.append(solution)
                W_array_rank.append(solution[0])
                Delta_array_rank.append(solution[1])
                mu_array_rank.append(solution[2])
                if symmetry == 's_id':
                    Delta_d_array_rank.append(solution[3])
                if GA:
                    D_array_rank.append(solution[3])
                    if symmetry == 's':
                        cidciu_array_rank.append(cidciu)
                        E_array_rank.append(calc_E(solution[3], n_el[cnt], array(list(solution[0:3]) + [cidciu] + [lamda])))
                    elif symmetry == 'd':
                        cidciu_array_rank.append(cidciu)
                        E_array_rank.append(calc_E(solution[3], n_el[cnt], solution[0:3], cidciu, lamda))
                else:
                    if symmetry == 's' or symmetry == 'd':
                        cidciu_array_rank.append(solution[3])
                    elif symmetry == 's_id':
                        cidciu_array_rank.append(solution[4])

                    E_array_rank.append(calc_E(n_el[cnt], solution))
                if GA:
                    if symmetry == 's':
                        GA_ZZ_array_rank.append(calc_ZZ(solution[3], n_el[cnt], cidciu))
                    elif symmetry == 'd':
                        GA_ZZ_array_rank.append(calc_ZZ(solution[3], n_el[cnt], cidciu))
                print('n_el = ', n_el[cnt], ', solution = ', solution, '\n')
                cnt += 1
            result_array = comm.gather(solution_array_rank, root=0)
            W_array = comm.gather(W_array_rank, root=0)
            Delta_array = comm.gather(Delta_array_rank, root=0)
            if symmetry == 's_id':
                Delta_d_array = comm.gather(Delta_d_array_rank, root=0)
            mu_array = comm.gather(mu_array_rank, root=0)
            E_array = comm.gather(E_array_rank, root=0)
            cidciu_array = comm.gather(cidciu_array_rank, root=0)
            if GA:
                D_array = comm.gather(D_array_rank, root=0)
                GA_ZZ_array = comm.gather(GA_ZZ_array_rank, root=0)

            if rank == 0:
                for i in range(size):
                    Solution_arr[i::size] = result_array[i]
                    W_res_arr[i::size] = W_array[i]
                    Delta_res_arr[i::size] = Delta_array[i]
                    if symmetry == 's_id':
                        Delta_d_res_arr[i::size] = Delta_d_array[i]
                    mu_res_arr[i::size] = mu_array[i]
                    E_res_arr[i::size] = E_array[i]
                    cidciu_res_arr[i::size] = cidciu_array[i]
                    if GA:
                        D_res_arr[i::size] = D_array[i]
                        ZZ_res_arr[i::size] = GA_ZZ_array[i]
                    density_array[i::size] = n_el_array_split[i]
            comm.Barrier()



        # Here the equations are solved in serial for each density (if parallel computing is not activated)
        else:
            for n_el_index, n_el in enumerate(electron_density_array):
                if GA:
                    if symmetry != 's_id':
                        start_values = list(GA_start_values[n_el_index, :])
                        if symmetry == 's':
                            start_values.append(n_el ** 2 / 4.0 + start_values[3]**2)
                        elif symmetry == 'd':
                            start_values.append(n_el ** 2 / 4.0)
                    else:
                        start_values = list(GA_start_values[n_el_index, :])
                        start_values = list([start_values[0]] + [start_values[1]] + [start_values[3]] + [start_values[2]] + [start_values[4]])
                        start_values.append(n_el ** 2 / 4.0)
                    print('initial guess: ', start_values)
                else:
                    if symmetry == 's':
                        start_values = list(start_values_for_very_first_density_s)
                    elif symmetry == 'd':
                        start_values = list(start_values_for_very_first_density_d)
                    elif symmetry == 's_id':
                        start_values = list(start_values_for_s_id[n_el_index, :])

                # Here the equations are solved!!!
                if GA:
                    solution, cidciu, lamda = solve_fct(n_el, array(start_values))
                else:
                    solution = solve_fct(n_el, array(start_values))

                # save the results in the arrays
                Solution_arr[n_el_index, :] = solution[:]
                W_res_arr[n_el_index] = solution[0]
                Delta_res_arr[n_el_index] = solution[1]
                if symmetry == 's_id':
                    mu_res_arr[n_el_index] = solution[3]
                    Delta_d_res_arr[n_el_index] = solution[2]
                else:
                    mu_res_arr[n_el_index] = solution[2]


                print('n_el = ', n_el, ', solution = ', solution, '\n')
                if GA:
                    if symmetry == 's' or symmetry == 'd':
                        E_res_arr[n_el_index] = calc_E(solution[3], n_el, array([solution[0], solution[1], solution[2], cidciu, lamda]))
                    elif symmetry == 's_id':
                        E_res_arr[n_el_index] = calc_E(solution[4], n_el, array([solution[0], solution[1], solution[2], solution[3], cidciu, lamda]))
                    cidciu_res_arr[n_el_index] = array(cidciu)
                    lamda_res_arr[n_el_index] = lamda
                else:
                    E_res_arr[n_el_index] = calc_E(n_el, solution)
                    if symmetry == 's' or symmetry == 'd':
                        cidciu_res_arr[n_el_index] = solution[3]
                    elif symmetry == 's_id':
                        cidciu_res_arr[n_el_index] = solution[4]

                density_array[n_el_index] = n_el
                if GA:
                    if symmetry == 's':
                        D_res_arr[n_el_index] = solution[3]
                        ZZ_res_arr[n_el_index] = calc_ZZ(solution[3], n_el, cidciu)
                    elif symmetry == 'd':
                        D_res_arr[n_el_index] = solution[3]
                        ZZ_res_arr[n_el_index] = calc_ZZ(solution[3], n_el)
                    elif symmetry == 's_id':
                        D_res_arr[n_el_index] = solution[4]
                        ZZ_res_arr[n_el_index] = calc_ZZ(solution[4], n_el, cidciu)









        # now the simulation is done! -> save the results in txt files
        if set_initial_values_manually_for_specific_densities:
            if rank == 0 or not parallel_computing:
                str_txt = '_with_GA' if GA else '_no_GA'
                name_solutions = 'res_array_' + symmetry + str_txt
                name_E = 'res_E_' + symmetry + str_txt
                name_cidciu = 'res_cidciu_' + symmetry + str_txt
                name_density = 'res_density_' + symmetry + str_txt
                name_W = 'res_W_' + symmetry + str_txt
                if symmetry == 's' or symmetry == 'd':
                    name_Delta = 'res_Delta_' + symmetry + str_txt
                elif symmetry == 's_id':
                    name_Delta_s = 'res_Delta_s_' + symmetry + str_txt
                    name_Delta_d = 'res_Delta_d_' + symmetry + str_txt
                name_mu = 'res_mu_' + symmetry + str_txt
                if GA:
                    name_D = 'res_D_' + symmetry + str_txt
                    name_ZZ = 'res_ZZ_' + symmetry + str_txt
                    name_lamda = 'res_lamda_' + symmetry + str_txt

                Solution_arr_ = loadtxt(os.path.join(path, name_solutions))
                Solution_arr_[el_dens_number1:el_dens_number2, :] = Solution_arr
                savetxt(os.path.join(path, name_solutions), Solution_arr_)

                E_res_arr_ = loadtxt(os.path.join(path, name_E))
                E_res_arr_[el_dens_number1:el_dens_number2] = E_res_arr
                savetxt(os.path.join(path, name_E), E_res_arr_)

                if symmetry == 's' or symmetry == 'd':
                    cidciu_res_arr_ = loadtxt(os.path.join(path, name_cidciu))
                    cidciu_res_arr_[el_dens_number1:el_dens_number2] = cidciu_res_arr
                    savetxt(os.path.join(path, name_cidciu), cidciu_res_arr_)

                elif symmetry == 's_id':
                    cidciu_res_arr_ = loadtxt(os.path.join(path, name_cidciu))
                    cidciu_res_arr_[el_dens_number1:el_dens_number2] = abs(cidciu_res_arr)
                    savetxt(os.path.join(path, name_cidciu), cidciu_res_arr_)

                W_res_arr_ = loadtxt(os.path.join(path, name_W))
                W_res_arr_[el_dens_number1:el_dens_number2] = W_res_arr
                savetxt(os.path.join(path, name_W), W_res_arr_)

                if symmetry == 's' or symmetry == 'd':
                    Delta_res_arr_ = loadtxt(os.path.join(path, name_Delta))
                    Delta_res_arr_[el_dens_number1:el_dens_number2] = Delta_res_arr
                    savetxt(os.path.join(path, name_Delta), Delta_res_arr_)
                if symmetry == 's_id':
                    Delta_res_arr_ = loadtxt(os.path.join(path, name_Delta_s))
                    Delta_res_arr_[el_dens_number1:el_dens_number2] = Delta_res_arr
                    savetxt(os.path.join(path, name_Delta_s), Delta_res_arr_)
                    Delta_d_res_arr_ = loadtxt(os.path.join(path, name_Delta_d))
                    Delta_d_res_arr_[el_dens_number1:el_dens_number2] = Delta_d_res_arr
                    savetxt(os.path.join(path, name_Delta_d), Delta_d_res_arr_)

                mu_res_arr_ = loadtxt(os.path.join(path, name_mu))
                mu_res_arr_[el_dens_number1:el_dens_number2] = mu_res_arr
                savetxt(os.path.join(path, name_mu), mu_res_arr_)
                if GA:
                    D_res_arr_ = loadtxt(os.path.join(path, name_D))
                    D_res_arr_[el_dens_number1:el_dens_number2] = D_res_arr
                    savetxt(os.path.join(path, name_D), D_res_arr_)

                    ZZ_res_arr_ = loadtxt(os.path.join(path, name_ZZ))
                    ZZ_res_arr_[el_dens_number1:el_dens_number2] = ZZ_res_arr
                    savetxt(os.path.join(path, name_ZZ), ZZ_res_arr_)

                    lamda_res_arr_ = loadtxt(os.path.join(path, name_lamda))
                    lamda_res_arr_[el_dens_number1:el_dens_number2] = lamda_res_arr
                    savetxt(os.path.join(path, name_lamda), lamda_res_arr_)

                density_array_ = loadtxt(os.path.join(path, name_density))
                density_array_[el_dens_number1:el_dens_number2] = density_array
                savetxt(os.path.join(path, name_density), density_array_)





        else:
            if rank == 0 or not parallel_computing:
                str_txt = '_with_GA' if GA else '_no_GA'
                name_solutions = 'res_array_' + symmetry + str_txt
                name_E = 'res_E_' + symmetry + str_txt
                name_cidciu = 'res_cidciu_' + symmetry + str_txt
                name_density = 'res_density_' + symmetry + str_txt
                name_W = 'res_W_' + symmetry + str_txt
                if symmetry == 's' or symmetry == 'd':
                    name_Delta = 'res_Delta_' + symmetry + str_txt
                elif symmetry == 's_id':
                    name_Delta_s = 'res_Delta_s_' + symmetry + str_txt
                    name_Delta_d = 'res_Delta_d_' + symmetry + str_txt
                name_mu = 'res_mu_' + symmetry + str_txt
                if GA:
                    name_D = 'res_D_' + symmetry + str_txt
                    name_ZZ = 'res_ZZ_' + symmetry + str_txt
                    name_lamda = 'res_lamda_' + symmetry + str_txt

                savetxt(os.path.join(path, name_solutions), Solution_arr)
                savetxt(os.path.join(path, name_E), E_res_arr)

                if symmetry == 's' or symmetry == 'd':
                    savetxt(os.path.join(path, name_cidciu), cidciu_res_arr)
                elif symmetry == 's_id':
                    savetxt(os.path.join(path, name_cidciu), abs(cidciu_res_arr))
                savetxt(os.path.join(path, name_W), W_res_arr)
                if symmetry == 's' or symmetry == 'd':
                    savetxt(os.path.join(path, name_Delta), Delta_res_arr)
                if symmetry == 's_id':
                    savetxt(os.path.join(path, name_Delta_s), Delta_res_arr)
                    savetxt(os.path.join(path, name_Delta_d), Delta_d_res_arr)
                savetxt(os.path.join(path, name_mu), mu_res_arr)
                if GA:
                    savetxt(os.path.join(path, name_D), D_res_arr)
                    savetxt(os.path.join(path, name_ZZ), ZZ_res_arr)
                    savetxt(os.path.join(path, name_lamda), lamda_res_arr)
                savetxt(os.path.join(path, name_density), density_array)
if plot_results:
    if rank == 0 or not parallel_computing:
        plot_result(U, t_diag, V1, GA, kBT, plot_s_id, plot_for_poster)
        print('done')
