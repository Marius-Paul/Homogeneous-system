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

s_d_sid_transition_accuracy = 1.0e-5
rel_accuracy_delta_mu = 1e-7
rel_accuracy_delta_x = 1e-6
min_noi = 0

# U is the potential energy of the interaction of an electron at a site with another electron on the same site
#U = 0.0 # should take values like 4, 8, 12

#n_el = 0.6
U_array = linspace(0.0, 8.0, 130)
t_prime_array = linspace(0.0, 1.0, 1)      # only do it for ONE U (len(U_array)=1)
n_el_array = linspace(0.05, 1.0, 81)

phase_diagramm_HF = zeros((len(n_el_array), len(U_array)))   # contains the symmetry for each n_el and U
phase_diagramm_time_dependent_GA = zeros((len(n_el_array), len(U_array)))   # contains the symmetry for each n_el and U

t_prime_phase_diagramm_HF = zeros((len(n_el_array), len(t_prime_array)))
t_prime_phase_diagramm_time_dependent_GA = zeros((len(n_el_array), len(t_prime_array)))


if len(U_array) > 1:
    all_Energies_HF = zeros((len(n_el_array), len(U_array), 3))
    all_Energies_GA = zeros((len(n_el_array), len(U_array), 3))
elif len(t_prime_array) > 1:
    all_Energies_HF = zeros((len(n_el_array), len(t_prime_array), 3))
    all_Energies_GA = zeros((len(n_el_array), len(t_prime_array), 3))

#phase_diagramm_td_GA_eta = zeros((len(n_el_array), len(U_array)))   # contains the eta for each n_el and U

only_HF = False
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
# def calc_I(E, P, D, eta):
#     return (sqrt(P*D)*sin(eta))/(sqrt(E+P)*sqrt(P+D))

def calc_J(Delta, Jz):
    return sqrt(Jz**2 + abs(Delta)**2)
# def calc_red_Jz(Jz, J):
#     return sign(Jz)*Jz/J
# def calc_red_J_plus(Jz, J, Delta):
#     return sign(Jz)*Delta.conjugate()/J



#def calc_K(R):
#    return R**2 #+ I**2*(red_Jz**2 - abs(red_J_plus)**2)
# def calc_F(R, I, red_Jz, red_J_plus):
#     return 2.0*I*red_J_plus.conjugate()*(1j *R + I*red_Jz)

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
def calc_Bk_s(lamda3, Delta_s):
    return lamda3 + V1*Delta_s*gamma_ks_matrix
def calc_Bk_d(lamda3, Delta_d):
    return lamda3 + V1*Delta_d*gamma_kd_matrix
def calc_Bk_s_id(lamda3, Delta_s, Delta_d):
    return lamda3 + V1*Delta_s*gamma_ks_matrix + 1j*V1*Delta_d*gamma_kd_matrix
def calc_Ek(Ak, Bk):
    return sqrt(Ak**2 + abs(Bk)**2)
def calc_Tk(Ek):
    return 1.0  #tanh(Ek/(2.0*kBT))

def calc_epsk_Ak_Bk_Ek_Tk_s(K, W, mu, lamda3, Delta_s, t_diag):
    ak = calc_Ak(K, W, mu, t_diag)
    bk = calc_Bk_s(lamda3, Delta_s)
    ek = calc_Ek(ak, bk)
    return ak, bk, ek, calc_Tk(ek)
def calc_epsk_Ak_Bk_Ek_Tk_d(K, W, mu, lamda3, Delta_d, t_diag):
    ak = calc_Ak(K, W, mu, t_diag)
    bk = calc_Bk_d(lamda3, Delta_d)
    ek = calc_Ek(ak, bk)
    return ak, bk, ek, calc_Tk(ek)
def calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W, mu, lamda3, Delta_s, Delta_d, t_diag):
    ak = calc_Ak(K, W, mu, t_diag)
    bk = calc_Bk_s_id(lamda3, Delta_s, Delta_d)
    ek = calc_Ek(ak, bk)
    return ak, bk, ek, calc_Tk(ek)


# expectation values
def calc_W(Ak, Ek, Tk):
    return -c * sum(gamma_ks_matrix*Ak/Ek * Tk)
def calc_n(Ak, Ek, Tk):
    return 1.0 - 1.0/N * sum(Ak/Ek*Tk)


def calc_Delta_s(Bk, Ek, Tk):
    return -c*sum(gamma_ks_matrix*Bk/Ek*Tk)
def calc_Delta_d(Bk, Ek, Tk):
    return -c*sum(gamma_kd_matrix*Bk/Ek*Tk)
def calc_Delta(Bk, Ek, Tk):
    return -1.0/(2.0*N) * sum(Bk/Ek*Tk)
def calc_W_n_Delta_s_Delta(Ak, Bk, Tk, Ek):
    return calc_W(Ak, Ek, Tk), calc_n(Ak, Ek, Tk), calc_Delta_s(Bk, Ek, Tk), calc_Delta(Bk, Ek, Tk)
def calc_W_n_Delta_d(Ak, Bk, Tk, Ek):
    return calc_W(Ak, Ek, Tk), calc_n(Ak, Ek, Tk), calc_Delta_d(Bk, Ek, Tk)     #, calc_Delta(Bk, Ek, Tk)
def calc_W_n_Delta_s_Delta_d_Delta(Ak, Bk, Tk, Ek):
    return calc_W(Ak, Ek, Tk), calc_n(Ak, Ek, Tk), calc_Delta_s(Bk, Ek, Tk), -1j*calc_Delta_d(Bk, Ek, Tk), calc_Delta(Bk, Ek, Tk)




def exp_fct_minus_exp_vals_s(W_Delta_s_Delta_mu, n_el, t_diag, K, U):
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s(K, W_Delta_s_Delta_mu[0], W_Delta_s_Delta_mu[-1]  - n_el/2 * U, W_Delta_s_Delta_mu[2]*U, W_Delta_s_Delta_mu[1], t_diag)
    return calc_W_n_Delta_s_Delta(ak, bk, tk, ek) - array([W_Delta_s_Delta_mu[0], n_el, W_Delta_s_Delta_mu[1], W_Delta_s_Delta_mu[2]])

def exp_fct_minus_exp_vals_d(W_Delta_d_mu, n_el, t_diag, K, U):
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_d(K, W_Delta_d_mu[0], W_Delta_d_mu[-1]  - n_el/2 * U, 0.0, W_Delta_d_mu[1], t_diag)
    res = array(calc_W_n_Delta_d(ak, bk, tk, ek))
    #print('res = ', res)
    return  res - array([W_Delta_d_mu[0], n_el, W_Delta_d_mu[1]])

def exp_fct_minus_exp_vals_sid(W_Delta_s_Delta_d_Delta_mu, n_el, t_diag, K, U):
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W_Delta_s_Delta_d_Delta_mu[0], W_Delta_s_Delta_d_Delta_mu[4]  - n_el/2 * U, W_Delta_s_Delta_d_Delta_mu[3]*U, W_Delta_s_Delta_d_Delta_mu[1], W_Delta_s_Delta_d_Delta_mu[2], t_diag)
    return real(calc_W_n_Delta_s_Delta_d_Delta(ak, bk, tk, ek)) - array([W_Delta_s_Delta_d_Delta_mu[0], n_el, W_Delta_s_Delta_d_Delta_mu[1], W_Delta_s_Delta_d_Delta_mu[2], W_Delta_s_Delta_d_Delta_mu[3]])




# W_Delta_s_Delta_d_Delta_mu_test = array([0.1, 0.1, 0.0742, 0.04, -2.1])
# U_test = 0.0
# n_el_test = 0.3
# t_diag_test = -0.5
#
# ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(1.0, W_Delta_s_Delta_d_Delta_mu_test[0], W_Delta_s_Delta_d_Delta_mu_test[4]  - n_el_test/2 * U_test, W_Delta_s_Delta_d_Delta_mu_test[3]*U_test, W_Delta_s_Delta_d_Delta_mu_test[1], W_Delta_s_Delta_d_Delta_mu_test[2], t_diag_test)
#     ##### hier weitermachen!!! s-id funktioniert noch nicht!!!!  calc_W_n_Delta_s_Delta_d_Delta(ak, bk, tk, ek)
# print(calc_W_n_Delta_s_Delta_d_Delta(ak, bk, tk, ek))
# exit()




def GA_exp_fct_minus_exp_vals_s(W_Delta_s_Delta_mu, n_el, t_diag, K, U, Jz):
    lamda_3 = calc_lamda3(Jz, W_Delta_s_Delta_mu[2], U, n_el)
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s(K, W_Delta_s_Delta_mu[0], W_Delta_s_Delta_mu[-1], lamda_3, W_Delta_s_Delta_mu[1], t_diag)
    return calc_W_n_Delta_s_Delta(ak, bk, tk, ek) - array([W_Delta_s_Delta_mu[0], n_el, W_Delta_s_Delta_mu[1], W_Delta_s_Delta_mu[2]])

def GA_exp_fct_minus_exp_vals_d(W_Delta_d_mu, n_el, t_diag, K, U, Jz):
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_d(K, W_Delta_d_mu[0], W_Delta_d_mu[-1], 0.0, W_Delta_d_mu[1], t_diag)
    return calc_W_n_Delta_d(ak, bk, tk, ek) - array([W_Delta_d_mu[0], n_el, W_Delta_d_mu[1]])

def GA_exp_fct_minus_exp_vals_sid(W_Delta_s_Delta_d_Delta_mu, n_el, t_diag, K, U, Jz):
    lamda_3 = calc_lamda3(Jz, W_Delta_s_Delta_d_Delta_mu[3], U, n_el)
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W_Delta_s_Delta_d_Delta_mu[0], W_Delta_s_Delta_d_Delta_mu[4], lamda_3, W_Delta_s_Delta_d_Delta_mu[1], W_Delta_s_Delta_d_Delta_mu[2], t_diag)
    return real(calc_W_n_Delta_s_Delta_d_Delta(ak, bk, tk, ek)) - array([W_Delta_s_Delta_d_Delta_mu[0], n_el, W_Delta_s_Delta_d_Delta_mu[1], W_Delta_s_Delta_d_Delta_mu[2], W_Delta_s_Delta_d_Delta_mu[3]])



# conditions
def cond1(EPD):
    return EPD[2] + 2.0*EPD[1] + EPD[0] - 1.0
def cond2(EPD, n):
    return EPD[1] + EPD[2]- n/2.0

# energies

def calc_Energy_s(EPD, W, Delta_s, mu, lamda3, n, Delta, lamda, U, Jz, J, t_diag):
    R = calc_R(*EPD)
    k = R**2
    DO = calc_DO(EPD[2], Jz, J)
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s(k, W, mu, lamda3, Delta_s, t_diag)
    return real((n-1.0)*mu + 1.0/N*sum(- ek*tk) - 4.0*V1*(abs(Delta_s)**2 - abs(W)**2) - 2.0*real(lamda3.conjugate()*Delta) + U*DO + lamda*cond1(EPD)**2 + lamda*cond2(EPD,n)**2)

def calc_Energy_d(EPD, W, Delta_d, mu, lamda3, n, Delta, lamda, U, Jz, J, t_diag):
    R = calc_R(*EPD)
    k = R**2
    DO = calc_DO(EPD[2], Jz, J)
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_d(k, W, mu, lamda3, Delta_d, t_diag)
    return real((n-1.0)*mu + 1.0/N*sum(- ek*tk) - 4.0*V1*(abs(Delta_d)**2 - abs(W)**2) - 2.0*real(lamda3.conjugate()*Delta) + U*DO + lamda*cond1(EPD)**2 + lamda*cond2(EPD,n)**2)

def calc_Energy_sid(EPD, W, Delta_s, Delta_d, mu, lamda3, n, Delta, lamda, U, Jz, J, t_diag):
    R = calc_R(*EPD)
    k = R**2
    DO = calc_DO(EPD[2], Jz, J)
    ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(k, W, mu, lamda3, Delta_s, Delta_d, t_diag)
    return real((n-1.0)*mu + 1.0/N*sum(- ek*tk) - 4.0*V1*(abs(Delta_s)**2 + abs(Delta_d)**2 - abs(W)**2) - 2.0*real(lamda3.conjugate()*Delta) + U*DO + lamda*cond1(EPD)**2 + lamda*cond2(EPD,n)**2)


def calc_lamda3(Jz, Delta, U, n):
    return - (Delta*sign(n-1.0)*U)/sqrt(Jz**2 + abs(Delta)**2)/2.0
    #return - l3 - (Delta*sign(n-1.0)*U)/sqrt(Jz**2 + abs(Delta)**2)




def dR_dE(E, P, D):
    return (sqrt(P) ** 3 - sqrt(E * P * D)) / (2.0 * sqrt(E) * sqrt(E + P) ** 3 * sqrt(D + P))
def dR_dP(E, P, D):
    return (sqrt(E) ** 3 * D + E * sqrt(D) ** 3 - sqrt(E) * P ** 2 - sqrt(D) * P ** 2) / (
                2.0 * sqrt(P) * sqrt(E + P) ** 3 * sqrt(D + P) ** 3)
def dR_dD(E, P, D):
    return (sqrt(P) * (P - sqrt(E * D))) / (2.0 * sqrt(D) * sqrt(E + P) * sqrt(D + P) ** 3)





# def dR_deta(E, P, D):
#     return -sqrt(P * D) * sin(eta) / (sqrt(E + P) * sqrt(P + D))

# def dI_dE(E, P, D, eta):
#     return -(sqrt(P * D) * sin(eta)) / (2.0 * sqrt(E + P) ** 3 * sqrt(D + P))
# def dI_dP(E, P, D, eta):
#     return (sqrt(D) * sin(eta) * (D * E - P ** 2)) / (2.0 * sqrt(P) * sqrt(E + P) ** 3 * sqrt(D + P) ** 3)
# def dI_dD(E, P, D, eta):
#     return (sqrt(P) ** 3 * sin(eta)) / (2.0 * sqrt(D) * sqrt(E + P) * sqrt(D + P) ** 3)
# def dI_deta(E, P, D, eta):
#     return (sqrt(P * D) * cos(eta)) / (sqrt(E + P) * sqrt(P + D))



def dK_dE(EPD, R):
    return 2.0*R*dR_dE(*EPD)
def dK_dP(EPD, R):
    return 2.0*R*dR_dP(*EPD)
def dK_dD(EPD, R):
    return 2.0*R*dR_dD(*EPD)
# def dK_deta(EPD, R):
#     return 2.0*R*dR_deta(*EPD)

# def dF_dE(EPD, R, I, redJz, redJplus):
#     return 2.0*dI_dE(*EPD)*redJplus.conjugate() * (R*1j + 2.0*I*redJz) + 2.0*I*redJplus*(1j*dR_dE(*EPD))
# def dF_dP(EPD, R, I, redJz, redJplus):
#     return 2.0*dI_dP(*EPD)*redJplus.conjugate() * (R*1j + 2.0*I*redJz) + 2.0*I*redJplus*(1j*dR_dP(*EPD))
# def dF_dD(EPD, R, I, redJz, redJplus):
#     return 2.0*dI_dD(*EPD)*redJplus.conjugate() * (R*1j + 2.0*I*redJz) + 2.0*I*redJplus*(1j*dR_dD(*EPD))
# def dF_deta(EPD, R, I, redJz, redJplus):
#     return 2.0*dI_deta(*EPD)*redJplus.conjugate() * (R*1j + 2.0*I*redJz) + 2.0*I*redJplus*(1j*dR_deta(*EPD))



def dAk_dE(EPD, R, t_diag):
    return - dK_dE(EPD, R)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
def dAk_dP(EPD, R, t_diag):
    return - dK_dP(EPD, R)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
def dAk_dD(EPD, R, t_diag):
    return - dK_dD(EPD, R)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
# def dAk_deta(EPD, R):
#     return - dK_deta(EPD, R)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)

# def dBk_dE(EPD, R):
#     return - dF_dE(EPD, R)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
# def dBk_dP(EPD, R):
#     return - dF_dP(EPD, R)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
# def dBk_dD(EPD, R):
#     return - dF_dD(EPD, R)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
# def dBk_deta(EPD, R):
#     return - dF_deta(EPD, R)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)

def dEk_dE(Ek, Ak, EPD, R, t_diag):
    return Ak/Ek * dAk_dE(EPD, R, t_diag)
def dEk_dP(Ek, Ak, EPD, R, t_diag):
    return Ak/Ek * dAk_dP(EPD, R, t_diag)
def dEk_dD(Ek, Ak, EPD, R, t_diag):
    return Ak/Ek * dAk_dD(EPD, R, t_diag)
# def dEk_deta(Ek, Ak, EPD, R):
#     return 1.0/(2.0*Ek) * (2.0*Ak * dAk_deta(EPD, R))
#

def dEnergy_dE(Ek, Ak, EPD, R, lamda, t_diag):
    return 1.0/N * sum(- dEk_dE(Ek, Ak, EPD, R, t_diag)
                        *(tanh(Ek/(2.0*kBT)))) + lamda* 2.0 * cond1(EPD)
# + Ek/(cosh(Ek/(2.0*kBT))**2 * 2.0*kBT))
def dEnergy_dP(Ek, Ak, EPD, R, lamda, n, t_diag):
    return 1.0/N * sum(- dEk_dP(Ek, Ak, EPD, R, t_diag)
                        *(tanh(Ek/(2.0*kBT)))) + lamda* 4.0 * cond1(EPD) + 2.0*lamda*cond2(EPD, n)

def dEnergy_dD(Ek, Ak, EPD, R, lamda, U, n, t_diag):
    return 1.0/N * sum(- dEk_dD(Ek, Ak, EPD, R, t_diag)
                        *(tanh(Ek/(2.0*kBT))) ) + U + lamda* 2.0 * cond1(EPD) + 2.0*lamda*cond2(EPD, n)

# def dEnergy_deta(Ek, Ak, EPD, R):
#     return 1.0/N * sum( dAk_deta(EPD, R) - dEk_deta(Ek, Ak, EPD, R)
#                         *(tanh(Ek/(2.0*kBT))))

def gradient_of_Energy_s(EPD, W, Delta_s, mu, lamda3, n, Delta, lamda, U, Jz, J, t_diag):
    R = calc_R(*EPD)
    k = R**2
    Ak, Bk, Ek, Tk = calc_epsk_Ak_Bk_Ek_Tk_s(k, W, mu, lamda3, Delta_s, t_diag)
    return array([dEnergy_dE(Ek, Ak, EPD, R, lamda, t_diag), dEnergy_dP(Ek, Ak, EPD, R, lamda, n, t_diag), dEnergy_dD(Ek, Ak, EPD, R, lamda, U, n, t_diag)]).real

def gradient_of_Energy_d(EPD, W, Delta_d, mu, lamda3, n, Delta, lamda, U, Jz, J, t_diag):
    R = calc_R(*EPD)
    k = R**2
    Ak, Bk, Ek, Tk = calc_epsk_Ak_Bk_Ek_Tk_d(k, W, mu, lamda3, Delta_d, t_diag)
    return array([dEnergy_dE(Ek, Ak, EPD, R, lamda, t_diag), dEnergy_dP(Ek, Ak, EPD, R, lamda, n, t_diag), dEnergy_dD(Ek, Ak, EPD, R, lamda, U, n, t_diag)]).real

def gradient_of_Energy_sid(EPD, W, Delta_s, Delta_d, mu, lamda3, n, Delta, lamda, U, Jz, J, t_diag):
    R = calc_R(*EPD)
    k = R**2
    Ak, Bk, Ek, Tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(k, W, mu, lamda3, Delta_s, Delta_d, t_diag)
    return array([dEnergy_dE(Ek, Ak, EPD, R, lamda, t_diag), dEnergy_dP(Ek, Ak, EPD, R, lamda, n, t_diag), dEnergy_dD(Ek, Ak, EPD, R, lamda, U, n, t_diag)]).real







#
#
#
# # gradient test
#
#
#
# EPD_start =  [0.80962301, 0.09037699, 0.00962301]
# W_start =  0.08671810163881492
# Delta_s_start =  0.047296126586444186
# mu_start =  -2.2957433627913906
# lamda3 =  0.012044197447360891
# n_el =  0.4
# Delta_start =  0.05211418879043883
# lamda =  10.0
# U =  8.263157894736842
# Jz =  -0.4
# J =  0.40338057547840045
#
# # compare the energy gradient with the numerical gradient
# print('EPD_start = ', EPD_start)
# print('W_start = ', W_start)
# print('Delta_s_start = ', Delta_s_start)
# print('mu_start = ', mu_start)
# print('lamda3 = ', lamda3)
# print('n_el = ', n_el)
# print('Delta_start = ', Delta_start)
# print('lamda = ', lamda)
# print('U = ', U)
# print('Jz = ', Jz)
# print('J = ', J)
#
# print('gradient = ',
#       gradient_of_Energy_s(EPD_start, W_start, Delta_s_start, mu_start, lamda3, n_el, Delta_start, lamda, U, Jz, J))
# h = 1.0e-6
# E_s_left = calc_Energy_s(EPD_start + array([h, 0.0, 0.0]), W_start, Delta_s_start, mu_start, lamda3, n_el,
#                          Delta_start, lamda, U, Jz, J)
# E_s_right = calc_Energy_s(EPD_start - array([h, 0.0, 0.0]), W_start, Delta_s_start, mu_start, lamda3, n_el,
#                           Delta_start, lamda, U, Jz, J)
# E_s_up = calc_Energy_s(EPD_start + array([0.0, h, 0.0]), W_start, Delta_s_start, mu_start, lamda3, n_el,
#                        Delta_start, lamda, U, Jz, J)
# E_s_down = calc_Energy_s(EPD_start - array([0.0, h, 0.0]), W_start, Delta_s_start, mu_start, lamda3, n_el,
#                          Delta_start, lamda, U, Jz, J)
# E_s_up_down = calc_Energy_s(EPD_start + array([0.0, 0.0, h]), W_start, Delta_s_start, mu_start, lamda3, n_el,
#                             Delta_start, lamda, U, Jz, J)
# E_s_down_up = calc_Energy_s(EPD_start - array([0.0, 0.0, h]), W_start, Delta_s_start, mu_start, lamda3, n_el,
#                             Delta_start, lamda, U, Jz, J)
#
# numeric_grad_E = array(
#     [(E_s_left - E_s_right) / (2.0 * h), (E_s_up - E_s_down) / (2.0 * h), (E_s_up_down - E_s_down_up) / (2.0 * h)])
# print('numeric_grad_E = ', numeric_grad_E)
#
# exit(0)
#
#
























HF_result_s = zeros(4, dtype = float)
HF_result_d = zeros(4, dtype = float)
HF_result_sid = zeros(5, dtype = float)

print('U_array = ', U_array)
print('n_el_array = ', n_el_array)

symmetries = ['s', 'd', 'sid']


if calc_stuff:

    for n_el_array_index, n_el in enumerate(n_el_array):
        Jz = 0.5 * (n_el - 1.0)

        for t_diag_index, t_diag in enumerate(t_prime_array):
            # print('t_diag = ', t_diag)

            for U_index, U in enumerate(U_array):
                # print('\n')
                # print('################################################################### Hartree-Fock ###################################################################')

                # Hartree-Fock
                for symmetry in symmetries:
                    # set initial values
                    if U_index == 0:
                        if symmetry == 's':
                            mu_start = -2.0
                            W_start = 0.05
                            Delta_s_start = 0.05
                            Delta_start = 0.05
                        elif symmetry == 'd':
                            mu_start = -2.0
                            W_start = 0.05
                            Delta_d_start = 0.05
                            Delta_start = 0.0
                        elif symmetry == 'sid':
                            mu_start = -2.0
                            W_start = 0.05
                            Delta_s_start = 0.05
                            Delta_d_start = 0.05
                            Delta_start = 0.05

                    else:
                        if symmetry == 's':
                            mu_start = HF_result_s[3]
                            W_start = HF_result_s[0]
                            Delta_s_start = HF_result_s[1]
                            Delta_start = HF_result_s[2]
                        elif symmetry == 'd':
                            mu_start = HF_result_d[3]
                            W_start = HF_result_d[0]
                            Delta_d_start = HF_result_d[1]
                            Delta_start = 0.0
                        elif symmetry == 'sid':
                            mu_start = HF_result_sid[4]
                            W_start = HF_result_sid[0]
                            Delta_s_start = HF_result_sid[1]
                            Delta_d_start = HF_result_sid[2]
                            Delta_start = HF_result_sid[3]

                    # if symmetry == 's':
                    #     print('start values: W = ', W_start, ', Delta_s = ', Delta_s_start, ' Delta = ', Delta_start, ', mu = ', mu_start)
                    # elif symmetry == 'd':
                    #     print('start values: W = ', W_start, ', Delta_d = ', Delta_d_start, ' Delta = ', Delta_start, ', mu = ', mu_start)
                    # elif symmetry == 'sid':
                    #     print('start values: W = ', W_start, ', Delta_s = ', Delta_s_start, ' Delta_d = ', Delta_d_start, ' Delta = ', Delta_start, ', mu = ', mu_start)

                    n = array(n_el)

                    K = 1.0


                    if symmetry == 's':
                        #bounds_array = ([-1.0, -1.0, -1.0, -4.0], [1.0, 1.0, 1.0, 4.0])
                        x_start = array([W_start, Delta_s_start, Delta_start, mu_start])
                        x_result = optimize.least_squares(exp_fct_minus_exp_vals_s, x_start, args=(n_el, t_diag, K, U))#,
                                                          #bounds = bounds_array)
                        #x_result = optimize.root(exp_fct_minus_exp_vals_s, x_start, args=(n_el, t_diag, K, U), method = 'hybr')


                        W_start = x_result.x[0]
                        Delta_s_start = x_result.x[1]
                        Delta_start = x_result.x[2]
                        mu_start = x_result.x[3]

                        ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s(1.0, W_start, mu_start - n_el / 2 * U, U * Delta_start,
                                                                 Delta_s_start, t_diag)
                        HF_result_s = array([W_start, Delta_s_start, Delta_start, mu_start])
                        HF_energy_s = real((n_el - 1.0) * mu_start + 1.0 / N * sum(- ek * tk) - 4.0 * V1 * (
                                abs(Delta_s_start) ** 2 - abs(
                            W_start) ** 2) - U * Delta_start ** 2 + n_el / 2 * U - U * n_el ** 2 / 4)

                        if U_index == 0:
                            last_n_el_solution_s = HF_result_s.copy()


                    elif symmetry == 'd':
                        #bounds_array = ([-1.0, -1.0, -6.0], [1.0, 1.0, 4.0])
                        x_start = array([W_start, Delta_d_start, mu_start])
                        x_result = optimize.least_squares(exp_fct_minus_exp_vals_d, x_start, args=(n_el, t_diag, K, U)) #, bounds = bounds_array)

                        # x_result = optimize.root(exp_fct_minus_exp_vals_d, x_start, args=(n_el, t_diag, K, U), method = 'hybr')

                        W_start = x_result.x[0]
                        Delta_d_start = x_result.x[1]
                        mu_start = x_result.x[2]
                        #print('W_start = ', W_start, 'Delta_d_start = ', Delta_d_start, 'mu_start = ', mu_start, 'Delta_start = ', Delta_start)

                        ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_d(1.0, W_start, mu_start - n_el / 2 * U, 0.0,
                                                                 Delta_d_start, t_diag)
                        HF_result_d = array([W_start, Delta_d_start, Delta_start, mu_start])
                        HF_energy_d = (n_el - 1.0) * mu_start + 1.0 / N * sum(- ek * tk) - 4.0 * V1 * (
                                abs(Delta_d_start) ** 2 - abs(
                            W_start) ** 2) + n_el / 2 * U - U * n_el ** 2 / 4

                        if U_index == 0:
                            last_n_el_solution_d = HF_result_d.copy()



                    elif symmetry == 'sid':
                        #bounds_array = ([-1.0, -1.0, -1.0, -1.0, -4.0], [1.0, 1.0, 1.0, 1.0, 4.0])
                        x_start = array([W_start, Delta_s_start, Delta_d_start, Delta_start, mu_start])
                        x_result = optimize.least_squares(exp_fct_minus_exp_vals_sid, x_start,
                                                          args=(n_el, t_diag, K, U))
                        # x_result = optimize.root(exp_fct_minus_exp_vals_sid, x_start, args=(n_el, t_diag, K, U), method = 'hybr')

                        W_start = x_result.x[0]
                        Delta_s_start = x_result.x[1]
                        Delta_d_start = x_result.x[2]
                        Delta_start = x_result.x[3]
                        mu_start = x_result.x[4]
                        # print('W_start = ', W_start, 'Delta_s_start = ', Delta_s_start, 'Delta_d_start = ', Delta_d_start, 'mu_start = ', mu_start, 'Delta_start = ', Delta_start)

                        ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(1.0, W_start, mu_start - n_el / 2 * U,
                                                                    U * Delta_start,
                                                                    Delta_s_start, Delta_d_start, t_diag)
                        HF_result_sid = array([W_start, Delta_s_start, Delta_d_start, Delta_start, mu_start])
                        HF_energy_sid = real((n_el - 1.0) * mu_start + 1.0 / N * sum(- ek * tk) - 4.0 * V1 * (
                                abs(Delta_s_start) ** 2 + abs(Delta_d_start) ** 2 - abs(
                            W_start) ** 2) - U * Delta_start ** 2 + n_el / 2 * U - U * n_el ** 2 / 4)

                        if U_index == 0:
                            last_n_el_solution_sid = HF_result_sid.copy()


                    else:
                        raise ValueError


                if len(U_array) > 1:
                    all_Energies_HF[n_el_array_index, U_index, 0] = HF_energy_s
                    all_Energies_HF[n_el_array_index, U_index, 1] = HF_energy_d
                    all_Energies_HF[n_el_array_index, U_index, 2] = HF_energy_sid
                elif len(t_prime_array) > 1:
                    all_Energies_HF[n_el_array_index, t_diag_index, 0] = HF_energy_s
                    all_Energies_HF[n_el_array_index, t_diag_index, 1] = HF_energy_d
                    all_Energies_HF[n_el_array_index, t_diag_index, 2] = HF_energy_sid

                smallest_HF_energy = argmin(array([HF_energy_s, HF_energy_d, HF_energy_sid]))

                if abs((HF_energy_sid - HF_energy_d)/HF_energy_d) < s_d_sid_transition_accuracy and HF_energy_d < HF_energy_s:
                    phase_diagramm_HF[n_el_array_index, U_index] = 1
                    t_prime_phase_diagramm_HF[n_el_array_index, t_diag_index] = 1
                elif abs((HF_energy_sid - HF_energy_s)/HF_energy_s) < s_d_sid_transition_accuracy and HF_energy_s < HF_energy_d:
                    phase_diagramm_HF[n_el_array_index, U_index] = 0
                    t_prime_phase_diagramm_HF[n_el_array_index, t_diag_index] = 0
                else:
                    phase_diagramm_HF[n_el_array_index, U_index] = argmin(array([HF_energy_s, HF_energy_d, HF_energy_sid]))
                    t_prime_phase_diagramm_HF[n_el_array_index, t_diag_index] = argmin(array([HF_energy_s, HF_energy_d, HF_energy_sid]))


                # 0: s-symmetry, 1: d-symmetry, 2: s+id-symmetry













                if not only_HF:
                    #print('\n')
                    #print('################################################################### Time-dep. Gutzwiller ###################################################################')
                    # time dependent GA
                    for symmetry in symmetries:

                        if symmetry=='s':
                            mu_start = HF_result_s[3]
                            W_start = HF_result_s[0]
                            Delta_s_start = HF_result_s[1]
                            Delta_start = HF_result_s[2]
                        elif symmetry=='d':
                            mu_start = HF_result_d[3]
                            W_start = HF_result_d[0]
                            Delta_d_start = HF_result_d[1]
                            Delta_start = 0.0
                            J = calc_J(Delta_start, Jz)
                            #lamda3 = 0.0  # calc_lamda3(n, Delta_start, U)
                        elif symmetry=='sid':
                            mu_start = HF_result_sid[4]
                            W_start = HF_result_sid[0]
                            Delta_s_start = HF_result_sid[1]
                            Delta_d_start = HF_result_sid[2]
                            Delta_start = HF_result_sid[3]


                        D_start = n_el ** 2 / 4 - 0.5 * (n_el - 1.0) + sign(n_el - 1.0) * sqrt((0.5 * (n_el - 1.0)) ** 2 + Delta_start ** 2) + Delta_start**2
                        P_start = n_el / 2.0 - D_start
                        E_start = 1.0 - 2.0 * P_start - D_start
                        #print('P_start = ', P_start, ', D_start = ', D_start, ', E_start = ', E_start)
                        #eta_start = pi/6

                        EPD_start = array([E_start, P_start, D_start])
                        lamda3 = U*Delta_start
                        lamda = 1.0e4
                        n = array(n_el)

                        accuracy_conditions = 1.0e-5
                        #delta_x = 1.0, 1.0
                        #delta_mu = 1.0
                        delta_EPD = 1.0
                        conditions_1 = 1.0
                        conditions_2 = 1.0
                        # parameters for minimization of the slave boson conditions
                        lamda_1_2 = array([1.0e4, 1.0e4])
                        alpha_GA = 0.5/(U+1.0)
                        if len(t_prime_array) > 1:
                            alpha_GA = 0.5 / (-t_diag + 1.0)
                        mu_fac = 0.5

                        leftbound = 1.0e-6  # for low U a higher leftbound works better! (e.g. for U=4.0 use leftbound=1.0e-6)
                        cnt_GA = 0
                        max_noi = 2000
                        ema = 1.0e-8  # accuracy for the energy minimization

                        bounds_array = zeros((3, 2), dtype=float)
                        for ii in range(3):
                            bounds_array[ii] = (leftbound, 1.0)

                        while delta_EPD > 1.0e-8 or conditions_1 > accuracy_conditions or conditions_2 > accuracy_conditions or cnt_GA < min_noi:
                            if symmetry == 's':
                                R = calc_R(*EPD_start)
                                K = R ** 2

                                #lamda3 = calc_lamda3(Jz, Delta_start, U, lamda3)

                                # bounds_array_exp = ([0.0, 0.0, 0.0, -4.0], [1.0, 1.0, 1.0, 4.0])
                                x_start = array([W_start, Delta_s_start, Delta_start, mu_start])
                                x_result = optimize.least_squares(GA_exp_fct_minus_exp_vals_s, x_start,
                                                                  args=(n_el, t_diag, K, U, Jz))



                                # x_result = optimize.root(exp_fct_minus_exp_vals_s, x_start, args=(n_el, t_diag, K, U), method = 'hybr')

                                W_start = x_result.x[0]
                                Delta_s_start = x_result.x[1]
                                Delta_start = x_result.x[2]
                                mu_start = x_result.x[3]



                                # for cdcnt in range(1):
                                #     ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s(K, W_start, mu_start, lamda3, Delta_s_start, t_diag)
                                #
                                #     W, n, Delta_s, Delta = calc_W_n_Delta_s_Delta(ak, bk, tk, ek)
                                #     mu = mu_start + mu_fac * (n_el - n)
                                #
                                #     delta_mu = abs((mu - mu_start) / mu)
                                #     mu_start = array(mu)
                                #     delta_x = linalg.norm(array([(W - W_start) / W]))
                                #
                                #     #if cnt_GA%10 == 0:
                                #     #    print('cnt_GA = ', cnt_GA, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                                #     #    print('W = ', W, ', Delta_s = ', Delta_s, ', Delta = ', Delta, ', mu = ', mu_start)
                                #
                                #     W_start = alpha_GA * W + (1.0 - alpha_GA) * W_start
                                #     Delta_s_start = alpha_GA * Delta_s + (1.0 - alpha_GA) * Delta_s_start
                                #     Delta_start = alpha_GA * Delta + (1.0 - alpha_GA) * Delta_start

                                J = calc_J(Delta_start, Jz)






                                lamda3 = calc_lamda3(Jz, Delta_start, U, n_el)
                                #print('lamda3 = ', lamda3, U*Delta_start, ', K = ', K, ', Delta_start = ', Delta_start)
                                energy_minimization = optimize.minimize(calc_Energy_s, EPD_start, args=(
                                    W_start, Delta_s_start, mu_start, lamda3, n_el, Delta_start, lamda, U, Jz, J, t_diag), method='L-BFGS-B', options={'disp': False, 'maxiter': 1000}, jac = gradient_of_Energy_s,
                                                                        bounds=bounds_array, tol=ema)


                                # Jz, J = calc_Jz_J(n, Delta_start)
                                # lamda3 = calc_lamda3(n, Delta_start, U)
                                # # print('lamda3 = ', lamda3, U*Delta_start)
                                # energy_minimization = optimize.minimize(calc_Energy_s, EPD_start, args=(
                                #     W, Delta_s, mu, lamda3, n, Delta, lamda, U, Jz, J), method='L-BFGS-B',
                                #                                         options={'disp': False, 'maxiter': 1000},
                                #                                         jac=gradient_of_Energy_s,
                                #                                         bounds=bounds_array, tol=ema)


                            elif symmetry == 'd':
                                R = calc_R(*EPD_start)
                                K = R**2
                                # ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_d(K, W_start, mu_start, lamda3, Delta_d_start, t_diag)
                                # W, n, Delta_d, Delta = calc_W_n_Delta_d_Delta(ak, bk, tk, ek)
                                # mu = mu_start + mu_fac * (n_el - n)
                                # delta_mu = abs((mu - mu_start) / mu)
                                # mu_start = array(mu)
                                # delta_x = linalg.norm(array([(W - W_start) / W]))
                                # #if cnt_GA % 10 == 0:
                                # #    print('cnt_GA = ', cnt_GA, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                                # #    print('W = ', W, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)
                                #
                                # W_start = alpha_GA * W + (1.0 - alpha_GA) * W_start
                                # Delta_d_start = alpha_GA * Delta_d + (1.0 - alpha_GA) * Delta_d_start
                                # Delta_start = alpha_GA * Delta + (1.0 - alpha_GA) * Delta_start

                                #bounds_array_exp = ([0.0, 0.0, -6.0], [1.0, 1.0, 4.0])
                                x_start = array([W_start, Delta_d_start, mu_start])
                                x_result = optimize.least_squares(GA_exp_fct_minus_exp_vals_d, x_start,
                                                                  args=(n_el, t_diag, K, U, Jz))
                                                                  # bounds=bounds_array_exp)
                                # x_result = optimize.root(exp_fct_minus_exp_vals_d, x_start, args=(n_el, t_diag, K, U), method = 'hybr')

                                W_start = x_result.x[0]
                                Delta_d_start = x_result.x[1]
                                mu_start = x_result.x[2]

                                energy_minimization = optimize.minimize(calc_Energy_d, EPD_start, args=(
                                    W_start, Delta_d_start, mu_start, lamda3, n_el, Delta_start, lamda, U, Jz, J, t_diag), method='L-BFGS-B', jac = gradient_of_Energy_d,
                                                                        options={'disp': False, 'maxiter': 1000},
                                                                        bounds=bounds_array, tol=ema)



                            elif symmetry == 'sid':
                                R = calc_R(*EPD_start)

                                #lamda3 = calc_lamda3(Jz, Delta_start, U, lamda3)
                                K = R**2

                                # for cdcnt in range(1):
                                #     ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W_start, mu_start, lamda3, Delta_s_start, Delta_d_start, t_diag)
                                #     W, n, Delta_s, Delta_d, Delta = calc_W_n_Delta_s_Delta_d_Delta(ak, bk, tk, ek)
                                #
                                #     mu = mu_start + mu_fac * (n_el - n)
                                #     delta_mu = abs((mu - mu_start) / mu)
                                #     mu_start = array(mu)
                                #     delta_x = linalg.norm(array([(W - W_start) / W]))
                                #     #if cnt_GA % 10 == 0:
                                #     #    print('cnt_GA = ', cnt_GA, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                                #     #    print('W = ', W,', Delta_s = ', Delta_s, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)
                                #
                                #     W_start = alpha_GA * W + (1.0 - alpha_GA) * W_start
                                #     Delta_s_start = alpha_GA * Delta_s + (1.0 - alpha_GA) * Delta_s_start
                                #     Delta_d_start = alpha_GA * Delta_d + (1.0 - alpha_GA) * Delta_d_start
                                #     Delta_start = alpha_GA * Delta + (1.0 - alpha_GA) * Delta_start

                                #bounds_array_exp = ([0.0, 0.0, 0.0, 0.0, -4.0], [1.0, 1.0, 1.0, 1.0, 4.0])
                                x_start = array([W_start, Delta_s_start, Delta_d_start, Delta_start, mu_start])
                                x_result = optimize.least_squares(GA_exp_fct_minus_exp_vals_sid, x_start,
                                                                  args=(n_el, t_diag, K, U, Jz))
                                                                #  bounds=bounds_array_exp)
                                # x_result = optimize.root(exp_fct_minus_exp_vals_sid, x_start, args=(n_el, t_diag, K, U), method = 'hybr')

                                W_start = x_result.x[0]
                                Delta_s_start = x_result.x[1]
                                Delta_d_start = x_result.x[2]
                                Delta_start = x_result.x[3]
                                mu_start = x_result.x[4]

                                J = calc_J(Delta_start, Jz)
                                lamda3 = calc_lamda3(Jz, Delta_start, U, n_el)
                                energy_minimization = optimize.minimize(calc_Energy_sid, EPD_start, args=(
                                    W_start, Delta_s_start, Delta_d_start, mu_start, lamda3, n_el, Delta_start, lamda, U, Jz, J, t_diag), method='L-BFGS-B', jac = gradient_of_Energy_sid,
                                                                        options={'disp': False, 'maxiter': 1000},
                                                                        bounds=bounds_array, tol=ema)

                            else:
                                raise ValueError


                            delta_EPD = linalg.norm(energy_minimization.x - EPD_start)
                            EPD_start = (energy_minimization.x).reshape(3)


                            if cnt_GA > 10:
                                lamda = 5.0e4
                            if cnt_GA > 20:
                                lamda = 1.0e5
                            if cnt_GA > 30:
                                lamda = 1.0e6
                            if cnt_GA > 40:
                                lamda = 1.0e7
                            if cnt_GA > 50:
                                lamda = 1.0e8

                            if cnt_GA > 70:
                                lamda = 1.0e9
                            if cnt_GA > 80:
                                lamda = 1.0e10


                            cnt_GA += 1
                            if cnt_GA > max_noi:
                                #raise ValueError('No convergence in time dependent GA')
                                break

                            conditions_1 = linalg.norm(cond1(EPD_start))
                            conditions_2 = linalg.norm(cond1(EPD_start))

                        # print('After', cnt_GA, 'iterations: ', ' delta_mu =', delta_mu, 'delta_x =', delta_x, ', n_el_rel_deviation =',
                        #       (n - n_el) / n_el, ', energy =', energy_minimization.fun, ', conditions:',
                        #       conditions_1, conditions_2, '\n')

                        if symmetry == 's':
                            #J = calc_J(Delta_start, Jz)
                            #lamda3 = calc_lamda3(Jz, Delta_start, U, lamda3)

                            GA_energy_s = calc_Energy_s(EPD_start, W_start, Delta_s_start, mu_start, lamda3, n, Delta_start, 0.0, U, Jz, J, t_diag)
                            GA_result_s = array([W_start, Delta_s_start, Delta_start, mu_start])

                            # print('cnt_GA =', cnt_GA, ', s-symmetry results:')
                            # print('GA_energy_s = ', GA_energy_s)
                            # print('W = ', W, ', Delta_s = ', Delta_s, ', Delta = ', Delta, ', mu = ', mu_start)
                            # print('EPD: ', EPD_start, '\n')

                        elif symmetry == 'd':
                            #J = calc_J(Delta_start, Jz)

                            GA_energy_d = calc_Energy_d(EPD_start, W_start, Delta_d_start, mu_start, 0.0, n, Delta_start, 0.0, U, Jz, J, t_diag)
                            GA_result_d = array([W_start, Delta_d_start, Delta_start, mu_start])

                            # print('cnt_GA =', cnt_GA, ', d-symmetry results:')
                            # print('GA_energy_d = ', GA_energy_d)
                            # print('W = ', W, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)
                            # print('EPD: ', EPD_start, '\n')

                        elif symmetry == 'sid':
                            #J = calc_J(Delta_start, Jz)
                            #lamda3 = calc_lamda3(Jz, Delta_start, U, lamda3)

                            GA_energy_sid = calc_Energy_sid(EPD_start, W_start, Delta_s_start, Delta_d_start, mu_start, lamda3, n, Delta_start, 0.0, U, Jz, J, t_diag)

                            GA_result_sid = array([W_start, Delta_s_start, Delta_d_start, Delta_start, mu_start])
                            # print('cnt_GA =', cnt_GA, ', s+id-symmetry results:')
                            # print('GA_energy_sid = ', GA_energy_sid)
                            # print('W = ', W, ', Delta_s = ', Delta_s, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)
                            # print('EPD: ', EPD_start, '\n')

                    smallest_GA_energy = argmin(array([GA_energy_s, GA_energy_d, GA_energy_sid]))

                    if len(U_array) > 1:
                        all_Energies_GA[n_el_array_index, U_index, 0] = GA_energy_s
                        all_Energies_GA[n_el_array_index, U_index, 1] = GA_energy_d
                        all_Energies_GA[n_el_array_index, U_index, 2] = GA_energy_sid
                    elif len(t_prime_array) > 1:
                        all_Energies_GA[n_el_array_index, t_diag_index, 0] = GA_energy_s
                        all_Energies_GA[n_el_array_index, t_diag_index, 1] = GA_energy_d
                        all_Energies_GA[n_el_array_index, t_diag_index, 2] = GA_energy_sid

                    if abs((GA_energy_sid - GA_energy_d)/GA_energy_d) < s_d_sid_transition_accuracy and GA_energy_d < GA_energy_s:
                        phase_diagramm_time_dependent_GA[n_el_array_index, U_index] = 1
                        t_prime_phase_diagramm_time_dependent_GA[n_el_array_index, t_diag_index] = 1

                    elif abs((GA_energy_sid - GA_energy_s)/GA_energy_s) < s_d_sid_transition_accuracy and GA_energy_s < GA_energy_d:
                        phase_diagramm_time_dependent_GA[n_el_array_index, U_index] = 0
                        t_prime_phase_diagramm_time_dependent_GA[n_el_array_index, t_diag_index] = 0

                    else:
                        phase_diagramm_time_dependent_GA[n_el_array_index, U_index] = argmin(array([GA_energy_s, GA_energy_d, GA_energy_sid]))
                        t_prime_phase_diagramm_time_dependent_GA[n_el_array_index, t_diag_index] = argmin(array([GA_energy_s, GA_energy_d, GA_energy_sid]))

                    # if phase_diagramm_time_dependent_GA[n_el_array_index, U_index] == 0:
                    #     print('s-symmetry is the lowest energy solution')
                    # elif phase_diagramm_time_dependent_GA[n_el_array_index, U_index] == 1:
                    #     print('d-symmetry is the lowest energy solution')
                    # else:
                    #     print('s+id-symmetry is the lowest energy solution')

                    print('t_diag = ', round(t_diag, 2), ', n_el = ', round(n_el, 2), ', U = ', round(U, 2), ':   HF symmetry: ', phase_diagramm_HF[n_el_array_index, U_index], ', GA symmetry: ', phase_diagramm_time_dependent_GA[n_el_array_index, U_index], ', number of iterations: ', cnt_GA)
                    print('HF energies: ', HF_energy_s, HF_energy_d, HF_energy_sid, 'GA energies: ', GA_energy_s, GA_energy_d, GA_energy_sid)
                    print('HF results: ', HF_result_s, HF_result_d, HF_result_sid, ', lamda3 = ', U*HF_result_s[2], U*HF_result_d[2], U*HF_result_sid[3])
                    print('GA results: ', GA_result_s, GA_result_d, GA_result_sid, ', lamda3 = ', calc_lamda3(Jz, GA_result_s[2], U, 0), calc_lamda3(Jz, GA_result_d[2], U, 0), calc_lamda3(Jz, GA_result_sid[3], U, 0), ', EPD: ', EPD_start)
                    print('\n')
                else:
                    print('t_diag = ', round(t_diag, 2), ', n_el = ', round(n_el, 2), ', U = ', round(U, 2), ':   HF symmetry: ', phase_diagramm_HF[n_el_array_index, U_index])
                    print('HF energies: ', HF_energy_s, HF_energy_d, HF_energy_sid, ', dev: ', abs((HF_energy_sid - HF_energy_d)/HF_energy_d), abs((HF_energy_sid - HF_energy_s)/HF_energy_s))
                    print('HF results: ', HF_result_s, HF_result_d, HF_result_sid)
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

    fig, (ax1, ax2) = subplots(2)

    ax1.plot(U_array, all_Energies_HF[indx, :, 0], 'r*-', label='s-symmetry')
    ax1.plot(U_array, all_Energies_HF[indx, :, 1], 'g*-', label='d-symmetry')
    ax1.plot(U_array, all_Energies_HF[indx, :, 2], 'b*-', label='s+id-symmetry')

    ax1.set_title('Hartree-Fock, n = ' + str(n_el_array[indx]), fontsize=fs)
    ax1.set_xlabel(r'$U/t$', fontsize=fs)
    ax1.set_ylabel(r'energy $E$', fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs)
    ax1.legend(loc='upper right', fontsize=fs)
    ax1.grid()

    ax2.plot(U_array, all_Energies_GA[indx, :, 0], 'r*-', label='s-symmetry')
    ax2.plot(U_array, all_Energies_GA[indx, :, 1], 'g*-', label='d-symmetry')
    ax2.plot(U_array, all_Energies_GA[indx, :, 2], 'b*-', label='s+id-symmetry')
    ax2.set_title('Gutzwiller, n = ' + str(n_el_array[indx]), fontsize=fs)
    ax2.set_xlabel(r'$U/t$', fontsize=fs)
    ax2.set_ylabel(r'energy $E$', fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs)
    ax2.legend(loc='upper right', fontsize=fs)
    ax2.grid()

    show()

if len(t_prime_array) > 1:
    indx = 0
    n_el_of_interest = n_el_array[indx]

    fig, (ax1, ax2) = subplots(2)

    ax1.plot(t_prime_array, all_Energies_HF[indx, :, 0], 'r*-', label='s-symmetry')
    ax1.plot(t_prime_array, all_Energies_HF[indx, :, 1], 'g*-', label='d-symmetry')
    ax1.plot(t_prime_array, all_Energies_HF[indx, :, 2], 'b*-', label='s+id-symmetry')

    ax1.set_title('Hartree-Fock, n = ' + str(n_el_array[indx]), fontsize=fs)
    ax1.set_xlabel(r'$t \prime/t$', fontsize=fs)
    ax1.set_ylabel(r'energy $E$', fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs)
    ax1.legend(loc='upper right', fontsize=fs)
    ax1.grid()

    ax2.plot(t_prime_array, all_Energies_GA[indx, :, 0], 'r*-', label='s-symmetry')
    ax2.plot(t_prime_array, all_Energies_GA[indx, :, 1], 'g*-', label='d-symmetry')
    ax2.plot(t_prime_array, all_Energies_GA[indx, :, 2], 'b*-', label='s+id-symmetry')
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
    print(t_prime_phase_diagramm_HF)
    ylim(t_prime_array[0], t_prime_array[-1])
    pcolormesh(n_el_array, t_prime_array+dt_diag/2, t_prime_phase_diagramm_HF.T)
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
    pcolormesh(n_el_array, t_prime_array+dt_diag/2, t_prime_phase_diagramm_time_dependent_GA.T)
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
    pcolormesh(n_el_array, U_array+dU/2, phase_diagramm_HF.T)
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
    pcolormesh(n_el_array, U_array+dU/2, phase_diagramm_time_dependent_GA.T)
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