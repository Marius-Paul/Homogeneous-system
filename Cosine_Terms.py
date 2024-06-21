from numpy import cos
def gamma_ks(k_x, k_y, a=1.0):
    return 2.0*(cos(a*k_x) + cos(a*k_y))

def gamma_kd(k_x, k_y, a=1.0):
    return 2.0*(cos(a*k_x) - cos(a*k_y))

def cos_term(kx, ky, a=1.0):
    return 4.0*cos(kx*a)*cos(ky*a)
def generating_matrices_of_cosine_combinations(area_list):
    gamma_ks_matrices = []
    gamma_kd_matrices = []
    cos_term_matrices = []
    for area in area_list:
        gamma_ks_matrix_area = []
        gamma_kd_matrix_area = []
        cos_term_matrix_area = []
        for k in area:
            gamma_ks_matrix_area.append(gamma_ks(k[0], k[1]))
            gamma_kd_matrix_area.append(gamma_kd(k[0], k[1]))
            cos_term_matrix_area.append(cos_term(k[0], k[1]))
        gamma_ks_matrices.append(gamma_ks_matrix_area)
        gamma_kd_matrices.append(gamma_kd_matrix_area)
        cos_term_matrices.append(cos_term_matrix_area)
    matrix_list = [gamma_ks_matrices, gamma_kd_matrices, cos_term_matrices]
    return matrix_list
