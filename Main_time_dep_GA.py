from numpy import *
from matplotlib.pyplot import *
from scipy import optimize


sqrt_N = 100
kBT = 1.0e-5  # temperature
N = sqrt_N**2  # dimension of grid, has to be an even integer!
V1 = -2.0   # (constant) potential energy for next neighbour interaction
c = 1.0/(8.0*N)  # a constant which is often needed in the calculations
a = 1.0  # grid constant
t = 1.0  # hopping parameter for horizontal and vertical neighbours
#t_diag = 0.0  # hopping parameter for diagonal neighbours, should take values like 0.0, -0.2, -0.4

rel_accuracy_delta_x = 1e-5
min_noi = 0

# U is the potential energy of the interaction of an electron at a site with another electron on the same site
#U = 0.0 # should take values like 4, 8, 12

#n_el = 0.6
U_array = linspace(0.0, 4.0, 17)  # only do it for ONE t_diag (len(t_prime_array)=1)
t_prime_array = linspace(0.0, -1.0, 1)      # only do it for ONE U (len(U_array)=1)
n_el_array = linspace(0.51, 0.98, 21)

phase_diagramm_HF = zeros((len(n_el_array), len(U_array)))   # contains the symmetry for each n_el and U
phase_diagramm_time_dependent_GA = zeros((len(n_el_array), len(U_array)))   # contains the symmetry for each n_el and U

t_prime_phase_diagramm_HF = zeros((len(n_el_array), len(t_prime_array)))
t_prime_phase_diagramm_time_dependent_GA = zeros((len(n_el_array), len(t_prime_array)))


if len(U_array) > 1:
    all_Energies_HF = zeros((len(n_el_array), len(U_array), 1))
    all_Energies_GA = zeros((len(n_el_array), len(U_array), 1))
elif len(t_prime_array) > 1:
    all_Energies_HF = zeros((len(n_el_array), len(t_prime_array), 1))
    all_Energies_GA = zeros((len(n_el_array), len(t_prime_array), 1))

#phase_diagramm_td_GA_eta = zeros((len(n_el_array), len(U_array)))   # contains the eta for each n_el and U

only_HF = True
calc_stuff = True

dk = 2.0*pi/sqrt_N
k_space_1D = arange(-pi, pi, dk)
k_space_2D = array([[k_space_1D[i], k_space_1D[j]] for i in range(sqrt_N) for j in range(sqrt_N)])
def gamma_ks(k):
    return 2*(cos(k[0]*a) + cos(k[1]*a))

def gamma_kd(k):
    return 2*(cos(k[0]*a) - cos(k[1]*a))

def Gamma_k(k):
    return 4*(cos(k[0]*a) * cos(k[1]*a))

gamma_ks_matrix = array([gamma_ks(k) for k in k_space_2D])
gamma_kd_matrix = array([gamma_kd(k) for k in k_space_2D])
Gamma_k_matrix = array([Gamma_k(k) for k in k_space_2D])

# for K and DO
def calc_R(E, P, D):
    return (sqrt(E*P) + sqrt(P*D))/(sqrt(E+P)*sqrt(P+D))

def calc_J(Delta, Jz):
    return sqrt(Jz**2 + abs(Delta)**2)

def calc_K(R):
    return R**2
# double occupancy
def calc_DO(D, Jz, J):
    return D + Jz - sign(Jz)*J


# for the expectation values
def calc_eps_k_matrix(K, W, t_diag):
    return -gamma_ks_matrix*(K*t + V1*W) - K*t_diag*Gamma_k_matrix
def calc_Ak(K, W, mu,t_diag):
    return calc_eps_k_matrix(K, W, t_diag) - mu
def calc_Bk_s_id(lamda3, Delta_0, eta_k_alpha):
    return lamda3 + V1*Delta_0*eta_k_alpha
def calc_Ek(Ak, Bk):
    return sqrt(Ak**2 + abs(Bk)**2)
def calc_Tk(Ek):
    return 1.0  #tanh(Ek/(2.0*kBT))


def calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W, mu, lamda3, Delta_0, t_diag, eta_k_alpha):
    ak = calc_Ak(K, W, mu, t_diag)
    bk = calc_Bk_s_id(lamda3, Delta_0, eta_k_alpha)
    ek = calc_Ek(ak, bk)
    return ak, bk, ek, 1.0


# expectation values
def calc_W(Ak, Ek, Tk):
    return -c * sum(gamma_ks_matrix*Ak/Ek * Tk)
def calc_n(Ak, Ek, Tk):
    return 1.0 - 1.0/N * sum(Ak/Ek*Tk)


def calc_Delta_0(Bk, Ek, Tk, eta_k_alpha):
    return -c*sum(eta_k_alpha.conjugate()*Bk/Ek*Tk)

def calc_Delta(Bk, Ek, Tk):
    return -1.0/(2.0*N) * sum(Bk/Ek*Tk)

def calc_W_n_Delta_0_Delta(Ak, Bk, Tk, Ek, eta_k_alpha):

    return calc_W(Ak, Ek, Tk), calc_n(Ak, Ek, Tk), calc_Delta_0(Bk, Ek, Tk, eta_k_alpha), calc_Delta(Bk, Ek, Tk)



def exp_fct_minus_exp_vals_sid(W_Delta_0_Delta_mu, n_el, t_diag, K, U, eta_k_alpha):
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W_Delta_0_Delta_mu[0], W_Delta_0_Delta_mu[3]  - n_el/2 * U, W_Delta_0_Delta_mu[2]*U, W_Delta_0_Delta_mu[1], t_diag, eta_k_alpha)
    return real(calc_W_n_Delta_0_Delta(ak, bk, tk, ek, eta_k_alpha)) - array([W_Delta_0_Delta_mu[0], n_el, W_Delta_0_Delta_mu[1], W_Delta_0_Delta_mu[2]])

def calc_eta_k_alpha(alpha):
    return gamma_ks_matrix*cos(pi*alpha) + 1j*gamma_kd_matrix*sin(pi*alpha)




def GA_exp_fct_minus_exp_vals_sid(W_Delta_0_Delta_mu, n_el, t_diag, K, U, Jz, eta_k_alpha):
    lamda_3 = calc_lamda3(Jz, W_Delta_0_Delta_mu[2], U, n_el)
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W_Delta_0_Delta_mu[0], W_Delta_0_Delta_mu[3], lamda_3, W_Delta_0_Delta_mu[1], t_diag, eta_k_alpha)
    return real(calc_W_n_Delta_0_Delta(ak, bk, tk, ek, eta_k_alpha)) - array([W_Delta_0_Delta_mu[0], n_el, W_Delta_0_Delta_mu[1], W_Delta_0_Delta_mu[2]])



# conditions
def cond1(EPD):
    return EPD[2] + 2.0*EPD[1] + EPD[0] - 1.0
def cond2(EPD, n):
    return EPD[1] + EPD[2] - n/2.0

# energies



def calc_lamda3(Jz, Delta, U, n):
    return - (Delta*sign(n-1.0)*U)/sqrt(Jz**2 + abs(Delta)**2)/2.0




def dR_dE(E, P, D):
    return (sqrt(P) ** 3 - sqrt(E * P * D)) / (2.0 * sqrt(E) * sqrt(E + P) ** 3 * sqrt(D + P))
def dR_dP(E, P, D):
    return (sqrt(E) ** 3 * D + E * sqrt(D) ** 3 - sqrt(E) * P ** 2 - sqrt(D) * P ** 2) / (
                2.0 * sqrt(P) * sqrt(E + P) ** 3 * sqrt(D + P) ** 3)
def dR_dD(E, P, D):
    return (sqrt(P) * (P - sqrt(E * D))) / (2.0 * sqrt(D) * sqrt(E + P) * sqrt(D + P) ** 3)




def dK_dE(EPD, R):
    return 2.0*R*dR_dE(*EPD)
def dK_dP(EPD, R):
    return 2.0*R*dR_dP(*EPD)
def dK_dD(EPD, R):
    return 2.0*R*dR_dD(*EPD)


def dAk_dE(EPD, R, t_diag):
    return - dK_dE(EPD, R)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
def dAk_dP(EPD, R, t_diag):
    return - dK_dP(EPD, R)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
def dAk_dD(EPD, R, t_diag):
    return - dK_dD(EPD, R)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)

def dEk_dE(Ek, Ak, EPD, R, t_diag):
    return Ak/Ek * dAk_dE(EPD, R, t_diag)
def dEk_dP(Ek, Ak, EPD, R, t_diag):
    return Ak/Ek * dAk_dP(EPD, R, t_diag)
def dEk_dD(Ek, Ak, EPD, R, t_diag):
    return Ak/Ek * dAk_dD(EPD, R, t_diag)

def dEnergy_dE(Ek, Ak, EPD, R, lamda, t_diag):
    return 1.0/N * sum(- dEk_dE(Ek, Ak, EPD, R, t_diag)
                        *1.0) + lamda* 2.0 * cond1(EPD)
# + Ek/(cosh(Ek/(2.0*kBT))**2 * 2.0*kBT))
def dEnergy_dP(Ek, Ak, EPD, R, lamda, n, t_diag):
    return 1.0/N * sum(- dEk_dP(Ek, Ak, EPD, R, t_diag)
                        *1.0) + lamda* 4.0 * cond1(EPD) + 2.0*lamda*cond2(EPD, n)

def dEnergy_dD(Ek, Ak, EPD, R, lamda, U, n, t_diag):
    return 1.0/N * sum(- dEk_dD(Ek, Ak, EPD, R, t_diag)
                        *1.0 ) + U + lamda* 2.0 * cond1(EPD) + 2.0*lamda*cond2(EPD, n)


def calc_Energy_sid(EPD, W, Delta_0, mu, lamda3, n, Delta, lamda, U, Jz, J, t_diag, eta_k_alpha):
    #R = calc_R(*EPD)
    #k = R**2
    DO = calc_DO(EPD[2], Jz, J)
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(calc_R(*EPD)**2, W, mu, lamda3, Delta_0, t_diag, eta_k_alpha)
    return real((n-1.0)*mu + 1.0/N*sum(- ek*tk) - 4.0*V1*(abs(Delta_0)**2 - abs(W)**2) - 2.0*real(lamda3.conjugate()*Delta) + U*DO + lamda*cond1(EPD)**2 + lamda*cond2(EPD,n)**2)


def gradient_of_Energy_sid(EPD, W, Delta_0, mu, lamda3, n, Delta, lamda, U, Jz, J, t_diag, eta_k_alpha):
    Ak, Bk, Ek, Tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(calc_R(*EPD)**2, W, mu, lamda3, Delta_0, t_diag, eta_k_alpha)
    return array([dEnergy_dE(Ek, Ak, EPD, R, lamda, t_diag), dEnergy_dP(Ek, Ak, EPD, R, lamda, n, t_diag), dEnergy_dD(Ek, Ak, EPD, R, lamda, U, n, t_diag)]).real




def HF_calc_energy(alpha, n_el, t_diag, U, W_start, Delta_0, mu_start, Delta_start):
    eta_k_alpha = calc_eta_k_alpha(alpha)
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(1.0, W_start, mu_start - n_el / 2 * U,
                                                U * Delta_start,
                                                Delta_0, t_diag, eta_k_alpha)
    return real((n_el - 1.0) * mu_start + 1.0 / N * sum(- ek * tk) - 4.0 * V1 * (
                                abs(Delta_0) ** 2 - abs(
                            W_start) ** 2) - U * Delta_start ** 2 + n_el / 2 * U - U * n_el ** 2 / 4)

def GA_calc_energy(alpha, n_el, t_diag, U, W_start, Delta_0, mu_start, Delta_start, EPD, DO, K, lamda3):
    eta_k_alpha = calc_eta_k_alpha(alpha)
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W_start, mu_start,
                                                lamda3,
                                                Delta_0, t_diag, eta_k_alpha)
    return real((n_el-1.0)*mu_start + 1.0/N*sum(- ek*tk) - 4.0*V1*(abs(Delta_0)**2 - abs(W_start)**2) - 2.0*real(lamda3.conjugate()*Delta_start) + U*DO + lamda*cond1(EPD)**2 + lamda*cond2(EPD,n)**2)





def dBk_dalpha(Delta_0, alpha):
    return V1*pi*Delta_0*(-gamma_ks_matrix*sin(pi*alpha) + 1j*gamma_kd_matrix*cos(pi*alpha))

def d_HF_energy_d_alpha(alpha, n_el, t_diag, U, W_start, Delta_0, mu_start, Delta_start):
    eta_k_alpha = calc_eta_k_alpha(alpha)
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(1.0, W_start, mu_start - n_el / 2 * U,
                                                U * Delta_start,
                                                Delta_0, t_diag, eta_k_alpha)
    return -1.0/N * sum(1.0/ek * real(dBk_dalpha(Delta_0, alpha) * bk.conjugate()) * tk)



def d_GA_energy_d_alpha(alpha, n_el, t_diag, U, W_start, Delta_0, mu_start, Delta_start, EPD, DO, K, lamda3):
    eta_k_alpha = calc_eta_k_alpha(alpha)
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W_start, mu_start,
                                                lamda3,
                                                Delta_0, t_diag, eta_k_alpha)
    return -1.0/N * sum(1.0/ek * real(dBk_dalpha(Delta_0, alpha) * bk.conjugate()) * tk)




HF_result_sid = zeros(3, dtype = float)

print('U_array = ', U_array)
print('n_el_array = ', n_el_array)

symmetries = ['sid']


if calc_stuff:

    for n_el_array_index, n_el in enumerate(n_el_array):
        Jz = 0.5 * (n_el - 1.0)
        for t_diag_index, t_diag in enumerate(t_prime_array):
            # print('t_diag = ', t_diag)

            for U_index, U in enumerate(U_array):
                # print('\n')
                # print('################################################################### Hartree-Fock ###################################################################')

                # Hartree-Fock
                # set initial values

                if U==U_array[0] and t_diag==t_prime_array[0]:
                    mu_start = -2.0
                    W_start = 0.15
                    Delta_0_start = 0.15
                    Delta_start = 0.15

                    if n_el < 0.6:
                        x_start = array([W_start, Delta_0_start, Delta_start, mu_start])
                        #print('x_start = ', x_start)
                        alpha_vec = linspace(0.0, 0.5, 200)
                    else:
                        x_start = array([0.1, 0.35, 0.0, -0.5])
                        alpha_vec = linspace(0.5, 0.0, 200)
                        #print(alpha_vec)

                elif (U != U_array[0] or t_diag != t_prime_array[0]):
                    mu_start = x_result_vec[0][3]
                    W_start = x_result_vec[0][0]
                    Delta_0_start = x_result_vec[0][1]
                    Delta_start = x_result_vec[0][2]
                    x_start = array([W_start, Delta_0_start, Delta_start, mu_start])
                    #print('x_start = ', x_start)

                n = array(n_el)
                K = 1.0



                energy_vec = zeros(len(alpha_vec))
                x_result_vec = []
                for indc in range(len(alpha_vec)):
                    eta_k_alpha = calc_eta_k_alpha(alpha_vec[indc])
                    if indc > 1:
                        x_start = x_result.x
                    #x_result = optimize.least_squares(exp_fct_minus_exp_vals_sid, x_start,
                    #                                  args=(n_el, t_diag, K, U, eta_k_alpha))
                    x_result = optimize.root(exp_fct_minus_exp_vals_sid, x_start,
                                                      args=(n_el, t_diag, K, U, eta_k_alpha), method='hybr',  tol=1.0e-12)
                    x_result_vec.append(x_result.x)
                    energy_vec[indc] = HF_calc_energy(alpha_vec[indc], n_el, t_diag, U, x_result.x[0], x_result.x[1],
                                                      x_result.x[3], x_result.x[2])

                HF_alpha_min = alpha_vec[argmin(energy_vec)]
                HF_results = x_result_vec[argmin(energy_vec)]
                print('t_diag = ', round(t_diag, 2), ', n_el = ', round(n_el, 2), ', U = ', round(U, 2), ':   HF symmetry: ', HF_alpha_min)
                phase_diagramm_HF[n_el_array_index, U_index] = HF_alpha_min
                t_prime_phase_diagramm_HF[n_el_array_index, t_diag_index] = HF_alpha_min

                #plot(alpha_vec, energy_vec)
                #show()




                if not only_HF:
                    #print('\n')
                    #print('################################################################### Time-dep. Gutzwiller ###################################################################')
                    # time dependent GA

                    W_start = HF_results[0]
                    Delta_0_start = HF_results[1]
                    Delta_start = HF_results[2]
                    mu_start = HF_results[3]
                    alpha = array(HF_alpha_new)


                    D_start = n_el ** 2 / 4 - 0.5 * (n_el - 1.0) + sign(n_el - 1.0) * sqrt((0.5 * (n_el - 1.0)) ** 2 + Delta_start ** 2) + Delta_start**2
                    P_start = n_el / 2.0 - D_start
                    E_start = 1.0 - 2.0 * P_start - D_start
                    #print('P_start = ', P_start, ', D_start = ', D_start, ', E_start = ', E_start)
                    #eta_start = pi/6

                    EPD_start = array([E_start, P_start, D_start])
                    lamda3 = U*Delta_start
                    lamda = 1.0e4
                    #n = array(n_el)

                    accuracy_conditions = 1.0e-8
                    #delta_x = 1.0, 1.0
                    #delta_mu = 1.0
                    delta_EPD = 1.0
                    conditions_1 = 1.0
                    conditions_2 = 1.0
                    # parameters for minimization of the slave boson conditions
                    lamda_1_2 = array([1.0e4, 1.0e4])

                    leftbound = 1.0e-7  # for low U a higher leftbound works better! (e.g. for U=4.0 use leftbound=1.0e-6)
                    cnt_GA = 0
                    max_noi = 200
                    ema = 1.0e-12  # accuracy for the energy minimization
                    damping_fac = 0.5/(U+1.0)
                    acc_alpha = 1.0e-9
                    rel_Delta_x = 1.0

                    bounds_array = zeros((3, 2), dtype=float)
                    for ii in range(3):
                        bounds_array[ii] = (leftbound, 1.0)

                    #bounds_array_exp = ([0.0, 0.0, 0.0, -4.0], [1.0, 1.0, 1.0, 4.0])

                    while delta_EPD > 1.0e-7 or conditions_1 > accuracy_conditions or conditions_2 > accuracy_conditions or cnt_GA < min_noi:

                        R = calc_R(*EPD_start)
                        K = R**2


                        Delta_alpha = 1.0
                        GA_HF_cnt = 0

                        while Delta_alpha > acc_alpha or rel_Delta_x > rel_accuracy_delta_x:

                            eta_k_alpha = calc_eta_k_alpha(alpha)

                            x_start = array([W_start, Delta_0_start, Delta_start, mu_start])
                            x_result = optimize.least_squares(GA_exp_fct_minus_exp_vals_sid, x_start,
                                                              args=(n_el, t_diag, K, U, Jz, eta_k_alpha))

                            # x_result = optimize.root(exp_fct_minus_exp_vals_sid, x_start, args=(n_el, t_diag, K, U), method = 'hybr')
                            rel_Delta_x = linalg.norm(x_start - x_result.x)
                            W_start = damping_fac*x_result.x[0] + (1.0 - damping_fac)*W_start
                            Delta_0_start = damping_fac*x_result.x[1] + (1.0 - damping_fac)*Delta_0_start
                            Delta_start = damping_fac*x_result.x[2] + (1.0 - damping_fac)*Delta_start
                            mu_start = x_result.x[3]

                            J = calc_J(Delta_start, Jz)
                            lamda3 = calc_lamda3(Jz, Delta_start, U, n_el)
                            DO = calc_DO(EPD_start[2], Jz, J)

                            #energy_minimization2 = optimize.minimize_scalar(GA_calc_energy, alpha, args=(
                            #    n_el, t_diag, U, W_start, Delta_0_start, mu_start, Delta_start, EPD_start, DO, K, lamda3),
                            #                                                bounds=(0.0, 0.5))
                            energy_minimization2 = optimize.minimize(GA_calc_energy, HF_alpha, args=(
                                n_el, t_diag, U, W_start, Delta_0_start, mu_start, Delta_start, EPD_start, DO, K, lamda3), bounds=[(0.0, 0.5)],
                                                                    jac=d_GA_energy_d_alpha, method='L-BFGS-B', tol=1.0e-12)

                            alpha_new = energy_minimization2.x[0]
                            Delta_alpha = abs(alpha_new - alpha)
                            alpha = array(alpha_new)
                            # alpha = array(alpha_new) * damping_fac + (1.0 - damping_fac) * alpha

                            GA_HF_cnt += 1
                            # if HF_cnt == 50:
                            #     W_start = HF_results[0]
                            #     Delta_0_start = HF_results[1]
                            #     Delta_start = HF_results[2]
                            #     mu_start = HF_results[3]
                            #     alpha = 0.25
                            #     #print('GA: HF_cnt: ', HF_cnt)
                            #     #print('alpha = ', alpha, ', Delta alpha: ', Delta_alpha, ', Delta x = ', rel_Delta_x)
                            #     Delta_alpha = 1.0
                            #     rel_Delta_x = 1.0
                            #
                            # if HF_cnt == 100:
                            #     W_start = HF_results[0]
                            #     Delta_0_start = HF_results[1]
                            #     Delta_start = 0.0
                            #     mu_start = HF_results[3]
                            #     alpha = 0.5
                            #     #print('GA: HF_cnt: ', HF_cnt)
                            #     #print('alpha = ', alpha, ', Delta alpha: ', Delta_alpha, ', Delta x = ', rel_Delta_x)
                            #     Delta_alpha = 1.0
                            #     rel_Delta_x = 1.0

                            if GA_HF_cnt == 400:
                                #print('GA: HF_cnt: ', HF_cnt, ', alpha = ', alpha, ', Delta alpha: ', Delta_alpha,
                                #     ', mu = ', mu_start, ', W = ', W_start, ', Delta_0 = ', Delta_0_start,
                                #     ', Delta = ', Delta_start, ', Delta x = ', rel_Delta_x, ', Delta EPD = ', delta_EPD)
                                # raise ValueError
                                break


                        eta_k_alpha = calc_eta_k_alpha(alpha)
                        energy_minimization = optimize.minimize(calc_Energy_sid, EPD_start, args=(
                            W_start, Delta_0_start, mu_start, lamda3, n_el, Delta_start, lamda, U, Jz, J, t_diag, eta_k_alpha), method='L-BFGS-B', jac = gradient_of_Energy_sid,
                                                                options={'disp': False, 'maxiter': 1000},
                                                                bounds=bounds_array, tol=ema)

                        delta_EPD = linalg.norm(energy_minimization.x - EPD_start)
                        #print('delta_EPD = ', delta_EPD)
                        EPD_start = (energy_minimization.x).reshape(3)



                        if cnt_GA > 3:
                            lamda = 5.0e4
                        if cnt_GA > 6:
                            lamda = 1.0e5
                        if cnt_GA > 9:
                            lamda = 1.0e6
                        if cnt_GA > 12:
                            lamda = 1.0e7
                        if cnt_GA > 15:
                            lamda = 1.0e8


                        cnt_GA += 1
                        if cnt_GA > max_noi:
                            #raise ValueError('No convergence in time dependent GA')
                            #print('HF energy = ', HF_energy, ', GA energy = ', energy_minimization.fun)
                            break

                        conditions_1 = linalg.norm(cond1(EPD_start))
                        conditions_2 = linalg.norm(cond1(EPD_start))



                    #J = calc_J(Delta_start, Jz)
                    #lamda3 = calc_lamda3(Jz, Delta_start, U, lamda3)

                    GA_result_sid = array([W_start, Delta_0_start, Delta_start, mu_start])
                    # print('cnt_GA =', cnt_GA, ', s+id-symmetry results:')
                    # print('GA_energy_sid = ', GA_energy_sid)
                    # print('W = ', W_start, ', Delta_0 = ', Delta_0_start, ', Delta = ', Delta_start, ', mu = ', mu_start)
                    # print('EPD: ', EPD_start, '\n')

                    GA_energy = energy_minimization.fun

                    if len(U_array) > 1:
                        all_Energies_GA[n_el_array_index, U_index] = GA_energy
                    elif len(t_prime_array) > 1:
                        all_Energies_GA[n_el_array_index, t_diag_index] = GA_energy

                    phase_diagramm_time_dependent_GA[n_el_array_index, U_index] = alpha
                    t_prime_phase_diagramm_time_dependent_GA[n_el_array_index, t_diag_index] = alpha

                    # if phase_diagramm_time_dependent_GA[n_el_array_index, U_index] == 0:
                    #     print('s-symmetry is the lowest energy solution')
                    # elif phase_diagramm_time_dependent_GA[n_el_array_index, U_index] == 1:
                    #     print('d-symmetry is the lowest energy solution')
                    # else:
                    #     print('s+id-symmetry is the lowest energy solution')

                    print('HF_cnt = ', HF_cnt, ', t_diag = ', round(t_diag, 2), ', n_el = ', round(n_el, 2), ', U = ', round(U, 2), ':   HF symmetry: ', phase_diagramm_HF[n_el_array_index, U_index], ', GA symmetry: ', phase_diagramm_time_dependent_GA[n_el_array_index, U_index], ', cnt_GA',cnt_GA, ', GA_HF_cnt = ', GA_HF_cnt)
                    # print('HF energy: ', HF_energy, 'GA energy: ', GA_energy)
                    # print('HF results: ', HF_results, ', lamda3 = ', U*HF_results[3])
                    # print('GA results: ', GA_result_sid, ', lamda3 = ', calc_lamda3(Jz, GA_result_sid[3], U, 0), ', EPD: ', EPD_start)
                    # print('\n')
                #else:
                    #print('HF_cnt = ', HF_cnt, ', t_diag = ', round(t_diag, 2), ', n_el = ', round(n_el, 2), ', U = ', round(U, 2), ':   HF symmetry: ', phase_diagramm_HF[n_el_array_index, U_index], ', HF results: ', HF_results)
                    #print('HF energy: ', HF_energy)
                    # print('HF results: ', HF_results)
                    # print('\n')
                    #








    save('Energies_HF', all_Energies_HF)
    save('Energies_GA', all_Energies_GA)
    savetxt('phase_diagramm_HF.txt', phase_diagramm_HF)
    savetxt('phase_diagramm_time_dependent_GA.txt', phase_diagramm_time_dependent_GA)
else:
    all_Energies_HF = load('Energies_HF.npy')
    all_Energies_GA = load('Energies_GA.npy')
    phase_diagramm_HF = loadtxt('phase_diagramm_HF.txt')
    phase_diagramm_time_dependent_GA = loadtxt('phase_diagramm_time_dependent_GA.txt')


# create phase diagram


# 0: s-symmetry, 1: d-symmetry, 2: s+id-symmetry




fs = 24
fs2 = 27
#
#print('Phase diagramm HF:')
#print(phase_diagramm_HF)
#print('Phase diagramm GA:')
# print(phase_diagramm_GA)
# print('Phase diagramm time-dependent GA:')
# print(phase_diagramm_time_dependent_GA)

if len(U_array) > 1:
    indx = 0
    n_el_of_interest = n_el_array[indx]

    fig, ax1 = subplots(1)

    ax1.plot(U_array, all_Energies_HF[indx, :, 0], 'r*-', label='HF')
    ax1.plot(U_array, all_Energies_GA[indx, :, 0], 'b*-', label='GA')

    ax1.set_title('Energies, n = ' + str(n_el_array[indx]), fontsize=fs)
    ax1.set_xlabel(r'$U/t$', fontsize=fs)
    ax1.set_ylabel(r'energy $E$', fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs)
    ax1.legend(loc='upper right', fontsize=fs)
    ax1.grid()

    show()

if len(t_prime_array) > 1:
    indx = 0
    n_el_of_interest = n_el_array[indx]

    fig, (ax1, ax2) = subplots(2)

    ax1.plot(t_prime_array, all_Energies_HF[indx, :], 'r*-', label='s-symmetry')

    ax1.set_title('Hartree-Fock, n = ' + str(n_el_array[indx]), fontsize=fs)
    ax1.set_xlabel(r'$t \prime/t$', fontsize=fs)
    ax1.set_ylabel(r'energy $E$', fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs)
    ax1.legend(loc='upper right', fontsize=fs)
    ax1.grid()

    ax2.plot(t_prime_array, all_Energies_GA[indx, :], 'r*-', label='s-symmetry')
    ax2.set_title('Gutzwiller, n = ' + str(n_el_array[indx]), fontsize=fs)
    ax2.set_xlabel(r'$t \prime/t$', fontsize=fs)
    ax2.set_ylabel(r'energy $E$', fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs)
    ax2.legend(loc='upper right', fontsize=fs)
    ax2.grid()

    show()





if len(t_prime_array)>1 and len(n_el_array)>1:
    d_nel = n_el_array[1] - n_el_array[0]
    dt_diag = t_prime_array[1] - t_prime_array[0]

    figure()

    subplot(211)
    #print(t_prime_phase_diagramm_HF)
    #ylim(t_prime_array[0], t_prime_array[-1])
    contourf(n_el_array, t_prime_array + dt_diag / 2, t_prime_phase_diagramm_HF.T, levels=3)
    title('Hartree-Fock', fontsize=fs)
    xlabel(r'electron density $n$', fontsize=fs, loc = 'right')
    ylabel(r'$t\prime /t$', fontsize=fs)
    tick_params(axis='both', which='major', labelsize=fs)
    ylim(t_prime_array[0], t_prime_array[-1])
    #text(0.21, 0.5, 's-symmetry', color='w', fontsize=fs2, transform=gca().transAxes, ha='center', va='center')
    #text(0.71, 0.5, 'd-symmetry', color='w', fontsize=fs2, transform=gca().transAxes, ha='center', va='center')
    #text(0.30, 0.91, 's+id-symmetry', color='k', fontsize=fs2, transform=gca().transAxes, ha='center', va='center')
    #grid()

    # subplot(312)
    # ylim(U_array[0], U_array[-1])
    # pcolormesh(n_el_array, U_array+dU/2, phase_diagramm_GA.T)
    # title('GA')
    # xlabel(r'$n$', fontsize=fs)
    # ylabel(r'$U$', fontsize=fs)
    # grid()

    subplot(212)
    ylim(t_prime_array[0], t_prime_array[-1])
    contourf(n_el_array, t_prime_array + dt_diag / 2, t_prime_phase_diagramm_time_dependent_GA.T, levels=3)
    title('Gutzwiller', fontsize=fs)
    xlabel(r'electron density $n$', fontsize=fs, loc = 'right')
    ylabel(r'$t\prime /t$', fontsize=fs)
    tick_params(axis='both', which='major', labelsize=fs)
    #text(0.21, 0.5, 's-symmetry', color='w', fontsize=fs2, transform=gca().transAxes, ha='center', va='center')
    #text(0.61, 0.5, 'd-symmetry', color='w', fontsize=fs2, transform=gca().transAxes, ha='center', va='center')
    #text(0.34, 0.9, 's+id-symmetry', color='k', fontsize=fs2, transform=gca().transAxes, ha='center', va='center')
    #grid()

    # subplot(313)
    # ylim(U_array[0], U_array[-1])
    # pcolormesh(n_el_array, U_array+dU/2, phase_diagramm_td_GA_eta.T)
    # title(r'Gutzwiller $\eta$', fontsize=fs)
    # xlabel(r'electron density $n$', fontsize=fs, loc = 'right')
    # ylabel(r'$U/t$', fontsize=fs)
    # tick_params(axis='both', which='major', labelsize=fs)


    show()








if len(U_array)>1 and len(n_el_array)>1:


    d_nel = n_el_array[1] - n_el_array[0]
    dU = U_array[1] - U_array[0]

    figure()

    subplot(211)
    ylim(U_array[0], U_array[-1])
    xlim(n_el_array[0], n_el_array[-1])
    contourf(n_el_array, U_array+dU/2, phase_diagramm_HF.T, levels=3)
    title('Hartree-Fock', fontsize=fs)
    xlabel(r'electron density $n$', fontsize=fs, loc = 'right')
    ylabel(r'$U/t$', fontsize=fs)
    tick_params(axis='both', which='major', labelsize=fs)
    #text(0.21, 0.5, 's-symmetry', color='w', fontsize=fs2, transform=gca().transAxes, ha='center', va='center')
    #text(0.71, 0.5, 'd-symmetry', color='w', fontsize=fs2, transform=gca().transAxes, ha='center', va='center')
    #text(0.30, 0.91, 's+id-symmetry', color='k', fontsize=fs2, transform=gca().transAxes, ha='center', va='center')
    #grid()

    # subplot(312)
    # ylim(U_array[0], U_array[-1])
    # pcolormesh(n_el_array, U_array+dU/2, phase_diagramm_GA.T)
    # title('GA')
    # xlabel(r'$n$', fontsize=fs)
    # ylabel(r'$U$', fontsize=fs)
    # grid()

    subplot(212)
    ylim(U_array[0], U_array[-1])
    contourf(n_el_array, U_array + dU / 2, phase_diagramm_time_dependent_GA.T, levels=3)
    title('Gutzwiller', fontsize=fs)
    xlabel(r'electron density $n$', fontsize=fs, loc = 'right')
    ylabel(r'$U/t$', fontsize=fs)
    tick_params(axis='both', which='major', labelsize=fs)
    #text(0.21, 0.5, 's-symmetry', color='w', fontsize=fs2, transform=gca().transAxes, ha='center', va='center')
    #text(0.61, 0.5, 'd-symmetry', color='w', fontsize=fs2, transform=gca().transAxes, ha='center', va='center')
    #text(0.34, 0.9, 's+id-symmetry', color='k', fontsize=fs2, transform=gca().transAxes, ha='center', va='center')
    #grid()

    # subplot(313)
    # ylim(U_array[0], U_array[-1])
    # pcolormesh(n_el_array, U_array+dU/2, phase_diagramm_td_GA_eta.T)
    # title(r'Gutzwiller $\eta$', fontsize=fs)
    # xlabel(r'electron density $n$', fontsize=fs, loc = 'right')
    # ylabel(r'$U/t$', fontsize=fs)
    # tick_params(axis='both', which='major', labelsize=fs)


    show()


