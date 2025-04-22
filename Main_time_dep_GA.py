from numpy import *
from matplotlib.pyplot import *
from scipy import optimize


sqrt_N = 100
kBT = 1.0e-3  # temperature
N = sqrt_N**2  # dimension of grid, has to be an even integer!
V1 = -2.0   # (constant) potential energy for next neighbour interaction
c = 1.0/(8.0*N)  # a constant which is often needed in the calculations
a = 1.0  # grid constant
t = 1.0  # hopping parameter for horizontal and vertical neighbours
t_diag = 0.0  # hopping parameter for diagonal neighbours, should take values like 0.0, -0.2, -0.4

s_d_sid_transition_accuracy = 1.0e-6
min_noi = 5

# U is the potential energy of the interaction of an electron at a site with another electron on the same site
#U = 0.0 # should take values like 4, 8, 12

#n_el = 0.6
U_array = linspace(0.0, 8.0, 20)
n_el_array = linspace(0.20, 0.76, 1)

phase_diagramm_HF = zeros((len(n_el_array), len(U_array)))   # contains the symmetry for each n_el and U
phase_diagramm_time_dependent_GA = zeros((len(n_el_array), len(U_array)))   # contains the symmetry for each n_el and U

all_Energies_HF = zeros((len(n_el_array), len(U_array), 3))
all_Energies_GA = zeros((len(n_el_array), len(U_array), 3))

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
def calc_eps_k_matrix(K, W):
    return -gamma_ks_matrix*(K*t + V1*W) - K*t_diag*Gamma_k_matrix
def calc_Ak(K, W, mu):
    return calc_eps_k_matrix(K, W) - mu
def calc_Bk_s(lamda3, Delta_s):
    return lamda3 + V1*Delta_s*gamma_ks_matrix
def calc_Bk_d(lamda3, Delta_d):
    return lamda3 + V1*Delta_d*gamma_kd_matrix
def calc_Bk_s_id(lamda3, Delta_s, Delta_d):
    return lamda3 + V1*Delta_s*gamma_ks_matrix + 1j*V1*Delta_d*gamma_kd_matrix
def calc_Ek(Ak, Bk):
    return sqrt(Ak**2 + abs(Bk)**2)
def calc_Tk(Ek):
    return tanh(Ek/(2.0*kBT))

def calc_epsk_Ak_Bk_Ek_Tk_s(K, W, mu, lamda3, Delta_s):
    ak = calc_Ak(K, W, mu)
    bk = calc_Bk_s(lamda3, Delta_s)
    ek = calc_Ek(ak, bk)
    return calc_eps_k_matrix(K, W), ak, bk, ek, calc_Tk(ek)
def calc_epsk_Ak_Bk_Ek_Tk_d(K, W, mu, lamda3, Delta_d):
    ak = calc_Ak(K, W, mu)
    bk = calc_Bk_d(lamda3, Delta_d)
    ek = calc_Ek(ak, bk)
    return calc_eps_k_matrix(K, W), ak, bk, ek, calc_Tk(ek)
def calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W, mu, lamda3, Delta_s, Delta_d):
    ak = calc_Ak(K, W, mu)
    bk = calc_Bk_s_id(lamda3, Delta_s, Delta_d)
    ek = calc_Ek(ak, bk)
    return calc_eps_k_matrix(K, W), ak, bk, ek, calc_Tk(ek)


# expectation values
def calc_W(Ak, Ek, Tk):
    return -c * sum(gamma_ks_matrix*Ak/Ek * Tk)
def calc_n(Ak, Ek, Tk):
    return real(1.0 - 1.0/N * sum(Ak/Ek*Tk))


def calc_Delta_s(Bk, Ek, Tk):
    return -c*sum(gamma_ks_matrix*Bk/Ek*Tk)
def calc_Delta_d(Bk, Ek, Tk):
    return -c*sum(gamma_kd_matrix*Bk/Ek*Tk)
def calc_Delta(Bk, Ek, Tk):
    return -1.0/(2.0*N) * sum(Bk/Ek*Tk)
def calc_W_n_Delta_s_Delta(Ak, Bk, Tk, Ek):
    return calc_W(Ak, Ek, Tk), calc_n(Ak, Ek, Tk), calc_Delta_s(Bk, Ek, Tk), calc_Delta(Bk, Ek, Tk)
def calc_W_n_Delta_d_Delta(Ak, Bk, Tk, Ek):
    return calc_W(Ak, Ek, Tk), calc_n(Ak, Ek, Tk), calc_Delta_d(Bk, Ek, Tk), 0.0     #, calc_Delta(Bk, Ek, Tk)
def calc_W_n_Delta_s_Delta_d_Delta(Ak, Bk, Tk, Ek):
    return calc_W(Ak, Ek, Tk), calc_n(Ak, Ek, Tk), calc_Delta_s(Bk, Ek, Tk), -1j*calc_Delta_d(Bk, Ek, Tk), calc_Delta(Bk, Ek, Tk)


# conditions
def cond1(EPD):
    return EPD[2] + 2.0*EPD[1] + EPD[0] - 1.0
def cond2(EPD,n):
    return EPD[1] + EPD[2] - n/2.0

# energies

def calc_Energy_s(EPD, W, Delta_s, mu, lamda3, n, Delta, lamda, U, Jz, J):
    R = calc_R(*EPD)
    k = R**2
    DO = calc_DO(EPD[2], Jz, J)
    epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s(k, W, mu, lamda3, Delta_s)
    return real((n-1.0)*mu + 1.0/N*sum(epsk - ek*tk) - 4.0*V1*(abs(Delta_s)**2 - abs(W)**2) - 2.0*real(lamda3.conjugate()*Delta) + U*DO + lamda*cond1(EPD)**2 + lamda*cond2(EPD,n)**2)

def calc_Energy_d(EPD, W, Delta_d, mu, lamda3, n, Delta, lamda, U, Jz, J):
    R = calc_R(*EPD)
    k = R**2
    DO = calc_DO(EPD[2], Jz, J)
    epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_d(k, W, mu, lamda3, Delta_d)
    return real((n-1.0)*mu + 1.0/N*sum(epsk - ek*tk) - 4.0*V1*(abs(Delta_d)**2 - abs(W)**2) - 2.0*real(lamda3.conjugate()*Delta) + U*DO + lamda*cond1(EPD)**2 + lamda*cond2(EPD,n)**2)

def calc_Energy_sid(EPD, W, Delta_s, Delta_d, mu, lamda3, n, Delta, lamda, U, Jz, J):
    R = calc_R(*EPD)
    k = R**2
    DO = calc_DO(EPD[2], Jz, J)
    epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(k, W, mu, lamda3, Delta_s, Delta_d)
    return real((n-1.0)*mu + 1.0/N*sum(epsk - ek*tk) - 4.0*V1*(abs(Delta_s)**2 + abs(Delta_d)**2 - abs(W)**2) - 2.0*real(lamda3.conjugate()*Delta) + U*DO + lamda*cond1(EPD)**2 + lamda*cond2(EPD,n)**2)


def calc_lamda3(Jz, Delta, U, l3):
    return - l3 - (Delta*sign(n-1.0)*U)/sqrt(Jz**2 + abs(Delta)**2)/2.0





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



def dAk_dE(EPD, R):
    return - dK_dE(EPD, R)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
def dAk_dP(EPD, R):
    return - dK_dP(EPD, R)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
def dAk_dD(EPD, R):
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

def dEk_dE(Ek, Ak, EPD, R):
    return Ak/Ek * dAk_dE(EPD, R)
def dEk_dP(Ek, Ak, EPD, R):
    return Ak/Ek * dAk_dP(EPD, R)
def dEk_dD(Ek, Ak, EPD, R):
    return Ak/Ek * dAk_dD(EPD, R)
# def dEk_deta(Ek, Ak, EPD, R):
#     return 1.0/(2.0*Ek) * (2.0*Ak * dAk_deta(EPD, R))
#

def dEnergy_dE(Ek, Ak, EPD, R, lamda):
    return 1.0/N * sum( dAk_dE(EPD, R) - dEk_dE(Ek, Ak, EPD, R)
                        *(tanh(Ek/(2.0*kBT)))) + lamda* 2.0 * cond1(EPD)
# + Ek/(cosh(Ek/(2.0*kBT))**2 * 2.0*kBT))
def dEnergy_dP(Ek, Ak, EPD, R, lamda, n):
    return 1.0/N * sum( dAk_dP(EPD, R) - dEk_dP(Ek, Ak, EPD, R)
                        *(tanh(Ek/(2.0*kBT)))) + lamda* 4.0 * cond1(EPD) + 2.0*lamda*cond2(EPD, n)

def dEnergy_dD(Ek, Ak, EPD, R, lamda, U, n):
    return 1.0/N * sum( dAk_dD(EPD, R) - dEk_dD(Ek, Ak, EPD, R)
                        *(tanh(Ek/(2.0*kBT))) ) + U + lamda* 2.0 * cond1(EPD) + 2.0*lamda*cond2(EPD, n)

# def dEnergy_deta(Ek, Ak, EPD, R):
#     return 1.0/N * sum( dAk_deta(EPD, R) - dEk_deta(Ek, Ak, EPD, R)
#                         *(tanh(Ek/(2.0*kBT))))

def gradient_of_Energy_s(EPD, W, Delta_s, mu, lamda3, n, Delta, lamda, U, Jz, J):
    R = calc_R(*EPD)
    k = R**2
    Epsk, Ak, Bk, Ek, Tk = calc_epsk_Ak_Bk_Ek_Tk_s(k, W, mu, lamda3, Delta_s)
    return array([dEnergy_dE(Ek, Ak, EPD, R, lamda), dEnergy_dP(Ek, Ak, EPD, R, lamda, n), dEnergy_dD(Ek, Ak, EPD, R, lamda, U, n)]).real

def gradient_of_Energy_d(EPD, W, Delta_d, mu, lamda3, n, Delta, lamda, U, Jz, J):
    R = calc_R(*EPD)
    k = R**2
    Epsk, Ak, Bk, Ek, Tk = calc_epsk_Ak_Bk_Ek_Tk_d(k, W, mu, lamda3, Delta_d)
    return array([dEnergy_dE(Ek, Ak, EPD, R, lamda), dEnergy_dP(Ek, Ak, EPD, R, lamda, n), dEnergy_dD(Ek, Ak, EPD, R, lamda, U, n)]).real

def gradient_of_Energy_sid(EPD, W, Delta_s, Delta_d, mu, lamda3, n, Delta, lamda, U, Jz, J):
    R = calc_R(*EPD)
    k = R**2
    Epsk, Ak, Bk, Ek, Tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(k, W, mu, lamda3, Delta_s, Delta_d)
    return array([dEnergy_dE(Ek, Ak, EPD, R, lamda), dEnergy_dP(Ek, Ak, EPD, R, lamda, n), dEnergy_dD(Ek, Ak, EPD, R, lamda, U, n)]).real








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
# n_el =  0.2
# Delta_start =  0.05211418879043883
# lamda =  10000.0
# U =  1.263157894736842
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

























HF_result_s = zeros(3, dtype = float)
HF_result_d = zeros(3, dtype = float)
HF_result_sid = zeros(3, dtype = float)

print('U_array = ', U_array)
print('n_el_array = ', n_el_array)

symmetries = ['s', 'd', 'sid']


if calc_stuff:

    for n_el_array_index, n_el in enumerate(n_el_array):
        Jz = 0.5 * (n_el - 1.0)


        for U_index, U in enumerate(U_array):
            # print('\n')
            # print('################################################################### Hartree-Fock ###################################################################')

            # Hartree-Fock
            for symmetry in symmetries:
                # set initial values
                if U==0.0:
                    mu_start = -2.0
                    W_start = 0.1
                    Delta_s_start = 0.1
                    Delta_d_start = 0.1
                    Delta_start = 0.1
                elif U > 0.0 and symmetry=='s':
                    mu_start = HF_result_s[3]
                    W_start = HF_result_s[0]
                    Delta_s_start = HF_result_s[1]
                    Delta_start = HF_result_s[2]
                elif U > 0.0 and symmetry=='d':
                    mu_start = HF_result_d[3]
                    W_start = HF_result_d[0]
                    Delta_d_start = HF_result_d[1]
                    Delta_start = 0.0
                elif U > 0.0 and symmetry=='sid':
                    mu_start = HF_result_sid[4]
                    W_start = HF_result_sid[0]
                    Delta_s_start = HF_result_sid[1]
                    if Delta_d_start < 1.0e-10:
                        Delta_d_start = 0.0
                    else:
                        Delta_d_start = HF_result_sid[2]
                    Delta_start = HF_result_sid[3]

                n = array(n_el)

                rel_accuracy_delta_mu = 1e-5
                rel_accuracy_delta_x = 1e-4
                delta_x = 1.0
                delta_mu = 1.0
                alpha_HF = 0.5/(U+1.0)

                cnt_HF = 0
                max_noi = 2000

                mu_fac = 0.5
                K = 1.0

                while delta_mu > rel_accuracy_delta_mu or delta_x > rel_accuracy_delta_x or cnt_HF < min_noi:

                    if symmetry == 's':
                        lamda3 = U*Delta_start

                        epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s(K, W_start, mu_start  - n_el/2 * U, lamda3, Delta_s_start)
                        W, n, Delta_s, Delta = calc_W_n_Delta_s_Delta(ak, bk, tk, ek)
                        #n = alpha_HF*n + (1.0 - alpha_HF) * n
                        mu = mu_start + mu_fac * (n_el - n)
                        #mu = optimize.minimize(calc_mu_s, x0=mu_start, args=(W_start, n_el, bk, U), method='Nelder-Mead', options={'xatol': 1e-8, 'disp': False}).x[0]

                        delta_mu = abs((mu - mu_start)/mu)
                        mu_start = array(mu)
                        #print('mu_start = ', mu_start)
                        delta_x = linalg.norm(array([(W - W_start)/W]))

                        #if cnt_HF%10 == 0:
                        #    print('cnt_HF = ', cnt_HF, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                        #    print('W = ', W, ', Delta_s = ', Delta_s, ', Delta = ', Delta, ', mu = ', mu_start)

                        W_start = alpha_HF * W + (1.0 - alpha_HF) * W_start
                        Delta_s_start = alpha_HF * Delta_s + (1.0 - alpha_HF) * Delta_s_start
                        Delta_start = alpha_HF * Delta + (1.0 - alpha_HF) * Delta_start

                    elif symmetry == 'd':
                        lamda3 = 0.0

                        epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_d(K, W_start, mu_start  - n_el/2 * U, 0.0, Delta_d_start)
                        W, n, Delta_d, Delta = calc_W_n_Delta_d_Delta(ak, bk, tk, ek)
                        mu = mu_start + mu_fac * (n_el - n)
                        delta_mu = abs((mu - mu_start)/mu)
                        mu_start = array(mu)
                        delta_x = linalg.norm(array([(W - W_start)/W]))
                        #if cnt_HF % 10 == 0:
                        #    print('cnt_HF = ', cnt_HF, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                        #    print('W = ', W, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)

                        W_start = alpha_HF * W + (1.0 - alpha_HF) * W_start
                        Delta_d_start = alpha_HF * Delta_d + (1.0 - alpha_HF) * Delta_d_start
                        Delta_start = alpha_HF * Delta + (1.0 - alpha_HF) * Delta_start


                    elif symmetry == 'sid':
                        lamda3 = U*Delta_start

                        epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W_start, mu_start  - n_el/2 * U, lamda3, Delta_s_start, Delta_d_start)
                        W, n, Delta_s, Delta_d, Delta = calc_W_n_Delta_s_Delta_d_Delta(ak, bk, tk, ek)
                        mu = mu_start + mu_fac * (n_el - n)
                        delta_mu = abs((mu - mu_start)/mu)
                        mu_start = array(mu)
                        delta_x = linalg.norm(array([(W - W_start)/W]))
                        #if cnt_HF % 10 == 0:
                        #    print('cnt_HF = ', cnt_HF, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                        #    print('W = ', W,', Delta_s = ', Delta_s, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)

                        W_start = alpha_HF * W + (1.0 - alpha_HF) * W_start
                        Delta_s_start = alpha_HF * Delta_s + (1.0 - alpha_HF) * Delta_s_start
                        Delta_d_start = alpha_HF * Delta_d + (1.0 - alpha_HF) * Delta_d_start
                        Delta_start = alpha_HF * Delta + (1.0 - alpha_HF) * Delta_start

                    else:
                        raise ValueError



                    if cnt_HF == 300:
                        alpha_HF = 0.2
                        #print('reduce alpha to', alpha_HF)
                    if cnt_HF == 400:
                        alpha_HF = 0.1
                        #print('reduce alpha to', alpha_HF)
                    if cnt_HF == 500:
                        alpha_HF = 0.05
                        #print('reduce alpha to', alpha_HF)
                    if cnt_HF == 600:
                        alpha_HF = 0.01
                        #print('reduce alpha to', alpha_HF)
                    if cnt_HF == 700:
                        alpha_HF = 0.005
                        #print('reduce alpha to', alpha_HF)
                    if cnt_HF == 800:
                        alpha_HF = 0.001
                        #print('reduce alpha to', alpha_HF)


                    cnt_HF += 1
                    if cnt_HF > max_noi:
                        print('After', cnt_HF, 'iterations: ', ' delta_mu =', delta_mu, 'delta_x =', delta_x,
                              ', n_el_rel_deviation =',
                              (n - n_el) / n_el)
                        #raise ValueError('No convergence in Hartree-Fock', symmetry)
                        break

                #print('After', cnt_HF, 'iterations: ', ' delta_mu =', delta_mu, 'delta_x =', delta_x, ', n_el_rel_deviation =',
                #      (n - n_el) / n_el)

                if symmetry == 's':
                    epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s(1.0, W_start, mu_start - n_el/2*U, U * Delta_start,
                                                                   Delta_s_start)
                    HF_result_s = array([W_start, Delta_s_start, Delta_start, mu_start])
                    HF_energy_s = real((n_el - 1.0) * mu_start + 1.0 / N * sum(epsk - ek * tk) - 4.0 * V1 * (
                            abs(Delta_s_start) ** 2 - abs(W_start) ** 2) - U * Delta_start ** 2 + n_el/2*U - U*n_el**2/4)
                    # print('cnt_HF =', cnt_HF, ', s-symmetry results:')
                    # print('HF_energy_s = ', HF_energy_s)
                    # print('W = ', W_start, ', Delta_s = ', Delta_s_start, ', Delta = ', Delta_start, ', mu = ', mu_start, ', n = ', n)
                elif symmetry == 'd':
                    epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_d(1.0, W_start, mu_start - n_el/2*U, U * 0.0,
                                                                   Delta_d_start)
                    HF_result_d = array([W_start, Delta_d_start, Delta_start, mu_start])
                    HF_energy_d = real((n_el - 1.0) * mu_start + 1.0 / N * sum(epsk - ek * tk) - 4.0 * V1 * (
                            abs(Delta_d_start) ** 2 - abs(W_start) ** 2) - U * Delta_start ** 2 + n_el/2*U- U*n_el**2/4)
                    # print('cnt_HF =', cnt_HF, ', d-symmetry results:')
                    # print('HF_energy_d = ', HF_energy_d)
                    # print('W = ', W_start, ', Delta_d = ', Delta_d_start, ', Delta = ', Delta_start, ', mu = ', mu_start, ', n = ', n)
                elif symmetry == 'sid':
                    epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(1.0, W_start, mu_start - n_el/2*U, U * Delta_start,
                                                                   Delta_s_start, Delta_d_start)
                    HF_result_sid = array([W_start, Delta_s_start, Delta_d_start, Delta_start, mu_start])
                    HF_energy_sid = real((n_el - 1.0) * mu_start + 1.0 / N * sum(epsk - ek * tk) - 4.0 * V1 * (
                            abs(Delta_s_start) ** 2 + abs(Delta_d_start) ** 2 - abs(W_start) ** 2) - U * Delta_start ** 2 + n_el/2*U - U*n_el**2/4)
                #     print('cnt_HF =', cnt_HF, ', s+id-symmetry results:')
                #     print('HF_energy_sid = ', HF_energy_sid)
                #     print('W = ', W, ', Delta_s = ', Delta_s_start, ', Delta_d = ', Delta_d_start*1j, ', Delta = ', Delta_start, ', mu = ', mu_start, ', n = ', n)
                # print('\n')

            all_Energies_HF[n_el_array_index, U_index, 0] = HF_energy_s
            all_Energies_HF[n_el_array_index, U_index, 1] = HF_energy_d
            all_Energies_HF[n_el_array_index, U_index, 2] = HF_energy_sid

            smallest_HF_energy = argmin(array([HF_energy_s, HF_energy_d, HF_energy_sid]))

            if abs(HF_energy_sid - HF_energy_d) < s_d_sid_transition_accuracy and HF_energy_d < HF_energy_s:
                phase_diagramm_HF[n_el_array_index, U_index] = 1
            elif abs(HF_energy_sid - HF_energy_s) < s_d_sid_transition_accuracy and HF_energy_s < HF_energy_d:
                phase_diagramm_HF[n_el_array_index, U_index] = 0
            else:
                phase_diagramm_HF[n_el_array_index, U_index] = argmin(array([HF_energy_s, HF_energy_d, HF_energy_sid]))

            # if phase_diagramm_HF[n_el_array_index, U_index] == 0:
            #     print('s-symmetry is the lowest energy solution')
            # elif phase_diagramm_HF[n_el_array_index, U_index] == 1:
            #     print('d-symmetry is the lowest energy solution')
            # else:
            #     print('s+id-symmetry is the lowest energy solution')


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

                    rel_accuracy_delta_mu = 1e-5
                    rel_accuracy_delta_x = 1e-4
                    accuracy_conditions = 1.0e-5
                    delta_x = 1.0, 1.0
                    delta_mu = 1.0
                    conditions_1 = 1.0
                    conditions_2 = 1.0
                    # parameters for minimization of the slave boson conditions
                    lamda_1_2 = array([1.0e4, 1.0e4])
                    alpha_GA = 0.5/(U+1.0)
                    mu_fac = 0.5

                    leftbound = 1.0e-12  # for low U a higher leftbound works better! (e.g. for U=4.0 use leftbound=1.0e-6)
                    cnt_GA = 0
                    max_noi = 2000
                    ema = 1.0e-12  # accuracy for the energy minimization

                    bounds_array = zeros((3, 2), dtype=float)
                    for ii in range(3):
                        bounds_array[ii] = (leftbound, 1.0)

                    while delta_mu > rel_accuracy_delta_mu or delta_x > rel_accuracy_delta_x or conditions_1 > accuracy_conditions or conditions_2 > accuracy_conditions or cnt_GA < min_noi:
                        if symmetry == 's':
                            R = calc_R(*EPD_start)
                            K = R ** 2

                            lamda3 = calc_lamda3(Jz, Delta_start, U, lamda3)

                            for cdcnt in range(10):
                                epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s(K, W_start, mu_start, lamda3, Delta_s_start)

                                W, n, Delta_s, Delta = calc_W_n_Delta_s_Delta(ak, bk, tk, ek)
                                mu = mu_start + mu_fac * (n_el - n)

                                delta_mu = abs((mu - mu_start) / mu)
                                mu_start = array(mu)
                                delta_x = linalg.norm(array([(W - W_start) / W]))

                                #if cnt_GA%10 == 0:
                                #    print('cnt_GA = ', cnt_GA, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                                #    print('W = ', W, ', Delta_s = ', Delta_s, ', Delta = ', Delta, ', mu = ', mu_start)

                                W_start = alpha_GA * W + (1.0 - alpha_GA) * W_start
                                Delta_s_start = alpha_GA * Delta_s + (1.0 - alpha_GA) * Delta_s_start
                                Delta_start = alpha_GA * Delta + (1.0 - alpha_GA) * Delta_start

                            J = calc_J(Delta_start, Jz)






                            lamda3 = calc_lamda3(Jz, Delta_start, U, lamda3)
                            #print('lamda3 = ', lamda3, U*Delta_start, ', K = ', K, ', Delta_start = ', Delta_start)
                            energy_minimization = optimize.minimize(calc_Energy_s, EPD_start, args=(
                                W_start, Delta_s_start, mu_start, lamda3, n_el, Delta_start, lamda, U, Jz, J), method='L-BFGS-B', options={'disp': False, 'maxiter': 1000}, jac = gradient_of_Energy_s,
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
                            epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_d(K, W_start, mu_start, lamda3, Delta_d_start)
                            W, n, Delta_d, Delta = calc_W_n_Delta_d_Delta(ak, bk, tk, ek)
                            mu = mu_start + mu_fac * (n_el - n)
                            delta_mu = abs((mu - mu_start) / mu)
                            mu_start = array(mu)
                            delta_x = linalg.norm(array([(W - W_start) / W]))
                            #if cnt_GA % 10 == 0:
                            #    print('cnt_GA = ', cnt_GA, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                            #    print('W = ', W, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)

                            W_start = alpha_GA * W + (1.0 - alpha_GA) * W_start
                            Delta_d_start = alpha_GA * Delta_d + (1.0 - alpha_GA) * Delta_d_start
                            Delta_start = alpha_GA * Delta + (1.0 - alpha_GA) * Delta_start

                            energy_minimization = optimize.minimize(calc_Energy_d, EPD_start, args=(
                                W_start, Delta_d_start, mu_start, lamda3, n_el, Delta_start, lamda, U, Jz, J), method='L-BFGS-B', jac = gradient_of_Energy_d,
                                                                    options={'disp': False, 'maxiter': 1000},
                                                                    bounds=bounds_array, tol=ema)



                        elif symmetry == 'sid':
                            R = calc_R(*EPD_start)

                            lamda3 = calc_lamda3(Jz, Delta_start, U, lamda3)
                            K = R**2

                            for cdcnt in range(10):
                                epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W_start, mu_start, lamda3, Delta_s_start, Delta_d_start)
                                W, n, Delta_s, Delta_d, Delta = calc_W_n_Delta_s_Delta_d_Delta(ak, bk, tk, ek)

                                mu = mu_start + mu_fac * (n_el - n)
                                delta_mu = abs((mu - mu_start) / mu)
                                mu_start = array(mu)
                                delta_x = linalg.norm(array([(W - W_start) / W]))
                                #if cnt_GA % 10 == 0:
                                #    print('cnt_GA = ', cnt_GA, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                                #    print('W = ', W,', Delta_s = ', Delta_s, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)

                                W_start = alpha_GA * W + (1.0 - alpha_GA) * W_start
                                Delta_s_start = alpha_GA * Delta_s + (1.0 - alpha_GA) * Delta_s_start
                                Delta_d_start = alpha_GA * Delta_d + (1.0 - alpha_GA) * Delta_d_start
                                Delta_start = alpha_GA * Delta + (1.0 - alpha_GA) * Delta_start

                            J = calc_J(Delta_start, Jz)
                            lamda3 = calc_lamda3(Jz, Delta_start, U, lamda3)
                            energy_minimization = optimize.minimize(calc_Energy_sid, EPD_start, args=(
                                W_start, Delta_s_start, Delta_d_start, mu_start, lamda3, n_el, Delta_start, lamda, U, Jz, J), method='L-BFGS-B', jac = gradient_of_Energy_sid,
                                                                    options={'disp': False, 'maxiter': 1000},
                                                                    bounds=bounds_array, tol=ema)

                        else:
                            raise ValueError


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

                        if cnt_GA == 300:
                            alpha_GA = 0.2
                        if cnt_GA == 400:
                            alpha_GA = 0.1
                        if cnt_GA == 500:
                            alpha_GA = 0.05
                        if cnt_GA == 600:
                            alpha_GA = 0.01
                        if cnt_GA == 700:
                            alpha_GA = 0.005
                        if cnt_GA == 800:
                            alpha_GA = 0.001


                        cnt_GA += 1
                        if cnt_GA > max_noi:
                            print('After', cnt_GA, 'iterations: ', ' delta_mu =', delta_mu, 'delta_x =', delta_x,
                                  ', n_el_rel_deviation =',
                                  (n - n_el) / n_el, ', energy =', energy_minimization.fun, ', conditions:',
                                  conditions_1, conditions_2)
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

                        GA_energy_s = calc_Energy_s(EPD_start, W_start, Delta_s_start, mu_start, lamda3, n, Delta_start, 0.0, U, Jz, J)
                        GA_result_s = array([W_start, Delta_s_start, Delta_start, mu_start])

                        # print('cnt_GA =', cnt_GA, ', s-symmetry results:')
                        # print('GA_energy_s = ', GA_energy_s)
                        # print('W = ', W, ', Delta_s = ', Delta_s, ', Delta = ', Delta, ', mu = ', mu_start)
                        # print('EPD: ', EPD_start, '\n')

                    elif symmetry == 'd':
                        #J = calc_J(Delta_start, Jz)

                        GA_energy_d = calc_Energy_d(EPD_start, W_start, Delta_d_start, mu_start, 0.0, n, Delta_start, 0.0, U, Jz, J)
                        GA_result_d = array([W_start, Delta_d_start, Delta_start, mu_start])

                        # print('cnt_GA =', cnt_GA, ', d-symmetry results:')
                        # print('GA_energy_d = ', GA_energy_d)
                        # print('W = ', W, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)
                        # print('EPD: ', EPD_start, '\n')

                    elif symmetry == 'sid':
                        #J = calc_J(Delta_start, Jz)
                        #lamda3 = calc_lamda3(Jz, Delta_start, U, lamda3)

                        GA_energy_sid = calc_Energy_sid(EPD_start, W_start, Delta_s_start, Delta_d_start, mu_start, lamda3, n, Delta_start, 0.0, U, Jz, J)

                        GA_result_sid = array([W_start, Delta_s_start, Delta_d_start, Delta_start, mu_start])
                        # print('cnt_GA =', cnt_GA, ', s+id-symmetry results:')
                        # print('GA_energy_sid = ', GA_energy_sid)
                        # print('W = ', W, ', Delta_s = ', Delta_s, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)
                        # print('EPD: ', EPD_start, '\n')

                smallest_GA_energy = argmin(array([GA_energy_s, GA_energy_d, GA_energy_sid]))

                all_Energies_GA[n_el_array_index, U_index, 0] = GA_energy_s
                all_Energies_GA[n_el_array_index, U_index, 1] = GA_energy_d
                all_Energies_GA[n_el_array_index, U_index, 2] = GA_energy_sid

                if abs(GA_energy_sid - GA_energy_d) < s_d_sid_transition_accuracy and GA_energy_d < GA_energy_s:
                    phase_diagramm_time_dependent_GA[n_el_array_index, U_index] = 1

                elif abs(GA_energy_sid - GA_energy_s) < s_d_sid_transition_accuracy and GA_energy_s < GA_energy_d:
                    phase_diagramm_time_dependent_GA[n_el_array_index, U_index] = 0

                else:
                    phase_diagramm_time_dependent_GA[n_el_array_index, U_index] = argmin(array([GA_energy_s, GA_energy_d, GA_energy_sid]))

                # if phase_diagramm_time_dependent_GA[n_el_array_index, U_index] == 0:
                #     print('s-symmetry is the lowest energy solution')
                # elif phase_diagramm_time_dependent_GA[n_el_array_index, U_index] == 1:
                #     print('d-symmetry is the lowest energy solution')
                # else:
                #     print('s+id-symmetry is the lowest energy solution')

                print('n_el = ', round(n_el, 2), ', U = ', round(U, 2), ':   HF symmetry: ', phase_diagramm_HF[n_el_array_index, U_index], ', GA symmetry: ', phase_diagramm_time_dependent_GA[n_el_array_index, U_index], ', number of iterations: ', cnt_GA)
                print('HF energies: ', HF_energy_s, HF_energy_d, HF_energy_sid, 'GA energies: ', GA_energy_s, GA_energy_d, GA_energy_sid)
                print('HF results: ', HF_result_s, HF_result_d, HF_result_sid)
                print('GA results: ', GA_result_s, GA_result_d, GA_result_sid)
                print('\n')
            else:
                print('n_el = ', round(n_el, 2), ', U = ', round(U, 2), ':   HF symmetry: ', phase_diagramm_HF[n_el_array_index, U_index], ', number of iterations: ', cnt_HF)
                # print('HF energies: ', HF_energy_s, HF_energy_d, HF_energy_sid)
                # print('HF results: ', HF_result_s, HF_result_d, HF_result_sid)
                # print('\n')







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


