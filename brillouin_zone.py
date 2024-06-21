from numpy import linspace, pi, meshgrid
from matplotlib.pyplot import *

############################## Construction of the areas in s-symmetry (same in d-symmetry!!) ############################
def create_whole_grid(grid_points, a=1.0):
    return linspace(-pi/a, pi/a, grid_points)

def create_brillouin_zone_8S(grid_points, a=1.0):
    # values of the wavenumbers within the 1. Brillouin zone
    k_array_complete = create_whole_grid(grid_points, a)
    k_array_southwest = k_array_complete[0:int(grid_points/2)]

    # plot the grid to check which areas you selected in order to consider the symmentry
    plot_grid = False    # careful: only plot for very less grid points, like e.g. 20
    if plot_grid == True:
        for i in range(grid_points):
            for j in range(grid_points):
                plot(k_array_complete[i], k_array_complete[j], 'b.', markersize=3)
        xlabel(r'$k_x$', fontsize=16)
        ylabel(r'$k_y$', fontsize=16)

    area_A = []
    A_vec = k_array_southwest[1:-1]
    A_len = len(A_vec)
    for i in range(A_len):
        for j in range(A_len-i):
            area_A.append([A_vec[i], k_array_southwest[-1-j]])

    if plot_grid == True:
        for el in area_A:
            plot(el[0], el[1], 'r*')

    area_B = []
    B_vec = k_array_southwest[1::]
    B_len = len(B_vec)
    for i in range(B_len):
        area_B.append([B_vec[i], B_vec[i]])

    if plot_grid == True:
        for el in area_B:
            plot(el[0], el[1], 'y+')

    area_C = []
    for i in range(B_len):
        area_C.append([k_array_southwest[0], B_vec[i]])

    if plot_grid == True:
        for el in area_C:
            plot(el[0], el[1], 'gs')

    area_D = [[k_array_southwest[0], k_array_southwest[0]]]

    if plot_grid == True:
        plot(area_D[0][0], area_D[0][1], 'ko')
        show()

    return [area_A, area_B, area_C, area_D]



def create_brillouin_zone_4S(grid_points, a=1.0):
    # values of the wavenumbers within the 1. Brillouin zone
    k_array_complete = create_whole_grid(grid_points, a)
    k_array_southwest = k_array_complete[0:int(grid_points/2)]

    # plot the grid to check which areas you selected in order to consider the symmentry
    plot_grid = False    # careful: only plot for very less grid points, like e.g. 20
    if plot_grid == True:
        for i in range(grid_points):
            for j in range(grid_points):
                plot(k_array_complete[i], k_array_complete[j], 'b.', markersize=3)
        xlabel(r'$k_x$', fontsize=16)
        ylabel(r'$k_y$', fontsize=16)

    area_A = []
    A_vec = k_array_southwest[1::]
    A_len = len(A_vec)
    for i in range(A_len):
        for j in range(A_len):
            area_A.append([A_vec[i], A_vec[j]])

    if plot_grid == True:
        for el in area_A:
            plot(el[0], el[1], 'r*')

    area_B = []
    B_vec = k_array_southwest[1::]
    B_len = len(B_vec)
    for i in range(B_len):
        area_B.append([B_vec[i], k_array_southwest[0]])

    if plot_grid == True:
        for el in area_B:
            plot(el[0], el[1], 'y+')

    area_C = []
    for i in range(B_len):
        area_C.append([k_array_southwest[0], B_vec[i]])

    if plot_grid == True:
        for el in area_C:
            plot(el[0], el[1], 'gs')

    area_D = [[k_array_southwest[0], k_array_southwest[0]]]

    if plot_grid == True:
        plot(area_D[0][0], area_D[0][1], 'ko')
        show()

    return [area_A, area_B, area_C, area_D]




def create_grids(grid_points, a=1.0):
    # create grid
    all_k = create_whole_grid(grid_points, a)
    k_array_complete = []
    for i in range(grid_points-1):
        for j in range(grid_points-1):
            k_array_complete.append([all_k[i], all_k[j]])
    area_list_8S = create_brillouin_zone_8S(grid_points, a)
    area_list_4S = create_brillouin_zone_4S(grid_points, a)
    return k_array_complete, area_list_8S, area_list_4S
