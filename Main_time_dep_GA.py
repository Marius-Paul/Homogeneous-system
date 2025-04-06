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

# U is the potential energy of the interaction of an electron at a site with another electron on the same site
#U = 0.0 # should take values like 4, 8, 12

#n_el = 0.6


dk = 2*pi/sqrt_N
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

# for K, F and DO
def calc_R(E, P, D, eta):
    return (sqrt(E*P) + sqrt(P*D)*cos(eta))/(sqrt(E+P)*sqrt(P+D))
def calc_I(E, P, D, eta):
    return (sqrt(P*D)*sin(eta))/(sqrt(E+P)*sqrt(P+D))
def calc_Jz(n):
    return 0.5*(n-1.0)
def calc_J(Delta, Jz):
    return sqrt(Jz**2 + abs(Delta)**2)
def calc_red_Jz(Jz, J):
    return sign(Jz)*Jz/J
def calc_red_J_plus(Jz, J, Delta):
    return sign(Jz)*Delta.conjugate()/J

def calc_RI(EPDeta):
    return calc_R(*EPDeta), calc_I(*EPDeta)
def calc_Jz_J_redJz_redJplus(n, Delta):
    return calc_Jz(n), calc_J(Delta, calc_Jz(n)), calc_red_Jz(calc_Jz(n), calc_J(Delta, calc_Jz(n))), calc_red_J_plus(calc_Jz(n), calc_J(Delta, calc_Jz(n)), Delta)



def calc_K(R, I, red_Jz, red_J_plus):
    return R**2 + I**2*(red_Jz**2 - abs(red_J_plus)**2)
def calc_F(R, I, red_Jz, red_J_plus):
    return 2.0*I*red_J_plus.conjugate()*(1j *R + I*red_Jz)
def calc_KF(R, I, red_Jz, red_J_plus):
    return calc_K(R, I, red_Jz, red_J_plus), calc_F(R, I, red_Jz, red_J_plus)
# double occupancy
def calc_DO(D, Jz, J):
    return D + Jz - sign(Jz)*J


# for the expectation values
def calc_eps_k_matrix(K, W):
    return -gamma_ks_matrix*(K*t + V1*W) - K*t_diag*Gamma_k_matrix
def calc_Ak(K, W, mu):
    return calc_eps_k_matrix(K, W) - mu
def calc_Bk_s(F, lamda3, Delta_s):
    return -t_diag*F*Gamma_k_matrix + lamda3 + V1*Delta_s*gamma_ks_matrix - t*F*gamma_ks_matrix
def calc_Bk_d(F, lamda3, Delta_d):
    return -t_diag*F*Gamma_k_matrix + lamda3 + V1*Delta_d*gamma_kd_matrix - t*F*gamma_ks_matrix
def calc_Bk_s_id(F, lamda3, Delta_s, Delta_d):
    return -t_diag*F*Gamma_k_matrix + lamda3 + V1*Delta_s*gamma_ks_matrix + 1j*V1*Delta_d*gamma_kd_matrix - t*F*gamma_ks_matrix
def calc_Ek(Ak, Bk):
    return sqrt(abs(Ak)**2 + abs(Bk)**2)
def calc_Tk(Ek):
    if (Ek.imag != 0.0).any():
        print(tanh(Ek/(2.0*kBT)), Ek, kBT)
        exit()
    return tanh(Ek/(2.0*kBT))

def calc_epsk_Ak_Bk_Ek_Tk_s(K, W, mu, F, lamda3, Delta_s):
    ak = calc_Ak(K, W, mu)
    bk = calc_Bk_s(F, lamda3, Delta_s)
    ek = calc_Ek(ak, bk)
    return calc_eps_k_matrix(K, W), ak, bk, ek, calc_Tk(ek)
def calc_epsk_Ak_Bk_Ek_Tk_d(K, W, mu, F, lamda3, Delta_d):
    ak = calc_Ak(K, W, mu)
    bk = calc_Bk_d(F, lamda3, Delta_d)
    ek = calc_Ek(ak, bk)
    return calc_eps_k_matrix(K, W), ak, bk, ek, calc_Tk(ek)
def calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W, mu, F, lamda3, Delta_s, Delta_d):
    ak = calc_Ak(K, W, mu)
    bk = calc_Bk_s_id(F, lamda3, Delta_s, Delta_d)
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
    return -1/(2*N) * sum(Bk.conjugate()/Ek*Tk)
def calc_W_n_Delta_s_Delta(Ak, Bk, Tk, Ek):
    return calc_W(Ak, Ek, Tk), calc_n(Ak, Ek, Tk), calc_Delta_s(Bk, Ek, Tk), calc_Delta(Bk, Ek, Tk)
def calc_W_n_Delta_d_Delta(Ak, Bk, Tk, Ek):
    return calc_W(Ak, Ek, Tk), calc_n(Ak, Ek, Tk), calc_Delta_d(Bk, Ek, Tk), calc_Delta(Bk, Ek, Tk)
def calc_W_n_Delta_s_Delta_d_Delta(Ak, Bk, Tk, Ek):
    return calc_W(Ak, Ek, Tk), calc_n(Ak, Ek, Tk), calc_Delta_s(Bk, Ek, Tk), 1j*c*sum(gamma_kd_matrix*Bk/Ek*Tk), calc_Delta(Bk, Ek, Tk)

# conditions
def cond1(EPDeta):
    return EPDeta[2] + 2*EPDeta[1] + EPDeta[0] - 1
def cond2(EPDeta,n):
    return EPDeta[1] + EPDeta[2] - n/2

# energies

def calc_Energy_s(EPDeta, Jz, J, redJz, redJplus, W, Delta_s, mu, lamda3, n, Delta, lamda):
    R, I = calc_RI(EPDeta)
    k, f = calc_KF(R, I, redJz, redJplus)
    epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s(k, W, mu, f, lamda3, Delta_s)
    DO = calc_DO(EPDeta[2], Jz, J)
    return real((n-1)*mu + 1/N*sum(epsk - ek*tk) - 4*V1*(abs(Delta_s)**2 - abs(W)**2) - lamda3*Delta - lamda3*Delta.conjugate() + U*DO + lamda*cond1(EPDeta)**2 + lamda*cond2(EPDeta,n)**2)

def calc_Energy_d(EPDeta, Jz, J, redJz, redJplus, W, Delta_d, mu, lamda3, n, Delta, lamda):
    R, I = calc_RI(EPDeta)
    k, f = calc_KF(R, I, redJz, redJplus)
    epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_d(k, W, mu, f, lamda3, Delta_d)
    DO = calc_DO(EPDeta[2], Jz, J)
    return real((n-1)*mu + 1/N*sum(epsk - ek*tk) - 4*V1*(abs(Delta_d)**2 - abs(W)**2) - lamda3*Delta.conjugate() - lamda3*Delta + U*DO + lamda*cond1(EPDeta)**2 + lamda*cond2(EPDeta,n)**2)

def calc_Energy_sid(EPDeta, Jz, J, redJz, redJplus, W, Delta_s, Delta_d, mu, lamda3, n, Delta, lamda):
    R, I = calc_RI(EPDeta)
    k, f = calc_KF(R, I, redJz, redJplus)
    epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(k, W, mu, f, lamda3, Delta_s, Delta_d)
    DO = calc_DO(EPDeta[2], Jz, J)
    return real((n-1)*mu + 1/N*sum(epsk - ek*tk) - 4*V1*(abs(Delta_s)**2 + abs(Delta_d)**2 - abs(W)**2) - lamda3*Delta - lamda3*Delta.conjugate() + U*DO + lamda*cond1(EPDeta)**2 + lamda*cond2(EPDeta,n)**2)


h = 1.0e-6
def calc_lamda3_s(EPDeta, W, Delta_s, mu, lamda3, n, lamda, Delta):
    Jz_left, J_left, redJz_left, redJplus_left = calc_Jz_J_redJz_redJplus(n, Delta-h)
    Energy_left = calc_Energy_s(EPDeta, Jz_left, J_left, redJz_left, redJplus_left, W, Delta_s, mu, lamda3, n, Delta-h, lamda)
    Jz_right, J_right, redJz_right, redJplus_right = calc_Jz_J_redJz_redJplus(n, Delta + h)
    Energy_right = calc_Energy_s(EPDeta, Jz_right, J_right, redJz_right, redJplus_right, W, Delta_s, mu, lamda3, n, Delta + h, lamda)
    return real((Energy_right-Energy_left)/(2*h))

def calc_lamda3_d(EPDeta, W, Delta_d, mu, lamda3, n, lamda, Delta):
    Jz_left, J_left, redJz_left, redJplus_left = calc_Jz_J_redJz_redJplus(n, Delta-h)
    Energy_left = calc_Energy_d(EPDeta, Jz_left, J_left, redJz_left, redJplus_left, W, Delta_d, mu, lamda3, n, Delta-h, lamda)
    Jz_right, J_right, redJz_right, redJplus_right = calc_Jz_J_redJz_redJplus(n, Delta + h)
    Energy_right = calc_Energy_d(EPDeta, Jz_right, J_right, redJz_right, redJplus_right, W, Delta_d, mu, lamda3, n, Delta + h, lamda)
    return real((Energy_right-Energy_left)/(2*h))

def calc_lamda3_sid(EPDeta, W, Delta_s, Delta_d, mu, lamda3, n, lamda, Delta):
    Jz_left, J_left, redJz_left, redJplus_left = calc_Jz_J_redJz_redJplus(n, Delta-h)
    Energy_left = calc_Energy_sid(EPDeta, Jz_left, J_left, redJz_left, redJplus_left, W, Delta_s, Delta_d, mu, lamda3, n, Delta-h, lamda)
    Jz_right, J_right, redJz_right, redJplus_right = calc_Jz_J_redJz_redJplus(n, Delta + h)
    Energy_right = calc_Energy_sid(EPDeta, Jz_right, J_right, redJz_right, redJplus_right, W, Delta_s, Delta_d, mu, lamda3, n, Delta + h, lamda)
    return real((Energy_right-Energy_left)/(2*h))





def dR_dE(E, P, D, eta):
    return (sqrt(P) ** 3 - sqrt(E * P * D) * cos(eta)) / (2.0 * sqrt(E) * sqrt(E + P) ** 3 * sqrt(D + P))
def dR_dP(E, P, D, eta):
    return (sqrt(E) ** 3 * D + E * sqrt(D) ** 3 * cos(eta) - sqrt(E) * P ** 2 - sqrt(D) * P ** 2 * cos(eta)) / (
                2.0 * sqrt(P) * sqrt(E + P) ** 3 * sqrt(D + P) ** 3)
def dR_dD(E, P, D, eta):
    return (sqrt(P) * (P * cos(eta) - sqrt(E * D))) / (2.0 * sqrt(D) * sqrt(E + P) * sqrt(D + P) ** 3)
def dR_deta(E, P, D, eta):
    return -sqrt(P * D) * sin(eta) / (sqrt(E + P) * sqrt(P + D))

def dI_dE(E, P, D, eta):
    return -(sqrt(P * D) * sin(eta)) / (2.0 * sqrt(E + P) ** 3 * sqrt(D + P))
def dI_dP(E, P, D, eta):
    return (sqrt(D) * sin(eta) * (D * E - P ** 2)) / (2.0 * sqrt(P) * sqrt(E + P) ** 3 * sqrt(D + P) ** 3)
def dI_dD(E, P, D, eta):
    return (sqrt(P) ** 3 * sin(eta)) / (2.0 * sqrt(D) * sqrt(E + P) * sqrt(D + P) ** 3)
def dI_deta(E, P, D, eta):
    return (sqrt(P * D) * cos(eta)) / (sqrt(E + P) * sqrt(P + D))



def dK_dE(EPDeta, R, I, redJz, redJplus):
    return 2.0*R*dR_dE(*EPDeta) + 2.0*I*dI_dE(*EPDeta)*(redJz**2 - abs(redJplus)**2)
def dK_dP(EPDeta, R, I, redJz, redJplus):
    return 2.0*R*dR_dP(*EPDeta) + 2.0*I*dI_dP(*EPDeta)*(redJz**2 - abs(redJplus)**2)
def dK_dD(EPDeta, R, I, redJz, redJplus):
    return 2.0*R*dR_dD(*EPDeta) + 2.0*I*dI_dD(*EPDeta)*(redJz**2 - abs(redJplus)**2)
def dK_deta(EPDeta, R, I, redJz, redJplus):
    return 2.0*R*dR_deta(*EPDeta) + 2.0*I*dI_deta(*EPDeta)*(redJz**2 - abs(redJplus)**2)

def dF_dE(EPDeta, R, I, redJz, redJplus):
    return 2.0*dI_dE(*EPDeta)*redJplus.conjugate() * (R*1j + 2.0*I*redJz) + 2.0*I*redJplus*(1j*dR_dE(*EPDeta))
def dF_dP(EPDeta, R, I, redJz, redJplus):
    return 2.0*dI_dP(*EPDeta)*redJplus.conjugate() * (R*1j + 2.0*I*redJz) + 2.0*I*redJplus*(1j*dR_dP(*EPDeta))
def dF_dD(EPDeta, R, I, redJz, redJplus):
    return 2.0*dI_dD(*EPDeta)*redJplus.conjugate() * (R*1j + 2.0*I*redJz) + 2.0*I*redJplus*(1j*dR_dD(*EPDeta))
def dF_deta(EPDeta, R, I, redJz, redJplus):
    return 2.0*dI_deta(*EPDeta)*redJplus.conjugate() * (R*1j + 2.0*I*redJz) + 2.0*I*redJplus*(1j*dR_deta(*EPDeta))



def dAk_dE(EPDeta, R, I, redJz, redJplus):
    return - dK_dE(EPDeta, R, I, redJz, redJplus)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
def dAk_dP(EPDeta, R, I, redJz, redJplus):
    return - dK_dP(EPDeta, R, I, redJz, redJplus)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
def dAk_dD(EPDeta, R, I, redJz, redJplus):
    return - dK_dD(EPDeta, R, I, redJz, redJplus)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
def dAk_deta(EPDeta, R, I, redJz, redJplus):
    return - dK_deta(EPDeta, R, I, redJz, redJplus)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)

def dBk_dE(EPDeta, R, I, redJz, redJplus):
    return - dF_dE(EPDeta, R, I, redJz, redJplus)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
def dBk_dP(EPDeta, R, I, redJz, redJplus):
    return - dF_dP(EPDeta, R, I, redJz, redJplus)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
def dBk_dD(EPDeta, R, I, redJz, redJplus):
    return - dF_dD(EPDeta, R, I, redJz, redJplus)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)
def dBk_deta(EPDeta, R, I, redJz, redJplus):
    return - dF_deta(EPDeta, R, I, redJz, redJplus)*(t*gamma_ks_matrix + t_diag*Gamma_k_matrix)

def dEk_dE(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus):
    return 1.0/(2.0*Ek) * (2.0*Ak * dAk_dE(EPDeta, R, I, redJz, redJplus) + 2.0*real(Bk.conjugate() * dBk_dE(EPDeta, R, I, redJz, redJplus)))
def dEk_dP(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus):
    return 1.0/(2.0*Ek) * (2.0*Ak * dAk_dP(EPDeta, R, I, redJz, redJplus) + 2.0*real(Bk.conjugate() * dBk_dP(EPDeta, R, I, redJz, redJplus)))
def dEk_dD(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus):
    return 1.0/(2.0*Ek) * (2.0*Ak * dAk_dD(EPDeta, R, I, redJz, redJplus) + 2.0*real(Bk.conjugate() * dBk_dD(EPDeta, R, I, redJz, redJplus)))
def dEk_deta(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus):
    return 1.0/(2.0*Ek) * (2.0*Ak * dAk_deta(EPDeta, R, I, redJz, redJplus) + 2.0*real(Bk.conjugate() * dBk_deta(EPDeta, R, I, redJz, redJplus)))


def dEnergy_dE(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus, lamda):
    return 1.0/N * sum( dAk_dE(EPDeta, R, I, redJz, redJplus) - dEk_dE(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus)
                        *(tanh(Ek/(2.0*kBT)))) + lamda* 2.0 * cond1(EPDeta)
# + Ek/(cosh(Ek/(2.0*kBT))**2 * 2.0*kBT))
def dEnergy_dP(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus, lamda, n):
    return 1.0/N * sum( dAk_dP(EPDeta, R, I, redJz, redJplus) - dEk_dP(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus)
                        *(tanh(Ek/(2.0*kBT)))) + lamda* 4.0 * cond1(EPDeta) + 2.0*lamda*cond2(EPDeta, n)

def dEnergy_dD(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus, lamda):
    return 1.0/N * sum( dAk_dD(EPDeta, R, I, redJz, redJplus) - dEk_dD(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus)
                        *(tanh(Ek/(2.0*kBT))) ) + U + lamda* 2.0 * cond1(EPDeta) + 2.0*lamda*cond2(EPDeta, n)

def dEnergy_deta(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus):
    return 1.0/N * sum( dAk_deta(EPDeta, R, I, redJz, redJplus) - dEk_deta(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus)
                        *(tanh(Ek/(2.0*kBT))))

def gradient_of_Energy_s(EPDeta, Jz, J, redJz, redJplus, W, Delta_s, mu, lamda3, n, Delta, lamda):
    R, I = calc_RI(EPDeta)
    k, f = calc_KF(R, I, redJz, redJplus)
    Epsk, Ak, Bk, Ek, Tk = calc_epsk_Ak_Bk_Ek_Tk_s(k, W, mu, f, lamda3, Delta_s)
    return array([dEnergy_dE(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus, lamda), dEnergy_dP(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus, lamda, n), dEnergy_dD(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus, lamda), dEnergy_deta(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus)]).real

def gradient_of_Energy_d(EPDeta, Jz, J, redJz, redJplus, W, Delta_d, mu, lamda3, n, Delta, lamda):
    R, I = calc_RI(EPDeta)
    k, f = calc_KF(R, I, redJz, redJplus)
    Epsk, Ak, Bk, Ek, Tk = calc_epsk_Ak_Bk_Ek_Tk_d(k, W, mu, f, lamda3, Delta_d)
    return array([dEnergy_dE(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus, lamda), dEnergy_dP(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus, lamda, n), dEnergy_dD(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus, lamda), dEnergy_deta(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus)]).real

def gradient_of_Energy_sid(EPDeta, Jz, J, redJz, redJplus, W, Delta_s, Delta_d, mu, lamda3, n, Delta, lamda):
    R, I = calc_RI(EPDeta)
    k, f = calc_KF(R, I, redJz, redJplus)
    Epsk, Ak, Bk, Ek, Tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(k, W, mu, f, lamda3, Delta_s, Delta_d)
    return array([dEnergy_dE(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus, lamda), dEnergy_dP(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus, lamda, n), dEnergy_dD(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus, lamda), dEnergy_deta(Ek, Ak, Bk, EPDeta, R, I, redJz, redJplus)]).real





U_array = linspace(0.0, 4.0, 4)
n_el_array = linspace(0.3, 0.6, 4)

only_HF = False

phase_diagramm_HF = zeros((len(n_el_array), len(U_array)))   # contains the symmetry for each n_el and U
phase_diagramm_time_dependent_GA = zeros((len(n_el_array), len(U_array)))   # contains the symmetry for each n_el and U


print('U_array = ', U_array)
print('n_el_array = ', n_el_array)

symmetries = ['s', 'd', 'sid']
for n_el_array_index, n_el in enumerate(n_el_array):
    for U_index, U in enumerate(U_array):
        print('WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW  n_el = ', n_el, ', U = ', U, ' WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW')
        print('\n')
        print('################################################################### Hartree-Fock ###################################################################')
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
                Delta_start = HF_result_d[2]
            elif U > 0.0 and symmetry=='sid':
                mu_start = HF_result_sid[4]
                W_start = HF_result_sid[0]
                Delta_s_start = HF_result_sid[1]
                Delta_d_start = HF_result_sid[2]
                Delta_start = HF_result_sid[3]

            n = array(n_el)

            accuracy_delta_mu = 1e-2
            accuracy_delta_x = 1e-2
            delta_x = 1.0
            delta_mu = 1.0
            alpha_HF = 0.5

            cnt = 0
            max_noi = 1000
            min_noi = 50 + int(100 * U / 2)

            while delta_mu > accuracy_delta_mu or delta_x > accuracy_delta_x or cnt < min_noi:

                if symmetry == 's':
                    lamda3 = U*Delta_start
                    K, F = 1.0, 0.0
                    epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s(K, W_start, mu_start, F, lamda3, Delta_s_start)
                    W, n, Delta_s, Delta = calc_W_n_Delta_s_Delta(ak, bk, tk, ek)
                    mu = mu_start + 0.5 * (n_el - n)
                    delta_mu = abs(mu - mu_start)
                    mu_start = array(mu)
                    delta_x = linalg.norm(array([W - W_start, Delta_s - Delta_s_start, Delta - Delta_start]))
                    #if cnt%10 == 0:
                    #    print('cnt = ', cnt, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                    #    print('W = ', W, ', Delta_s = ', Delta_s, ', Delta = ', Delta, ', mu = ', mu_start)

                    W_start = alpha_HF * W + (1.0 - alpha_HF) * W_start
                    Delta_s_start = alpha_HF * Delta_s + (1.0 - alpha_HF) * Delta_s_start
                    Delta_start = alpha_HF * Delta + (1.0 - alpha_HF) * Delta_start


                elif symmetry == 'd':
                    lamda3 = U*Delta_start
                    K, F = 1.0, 0.0
                    epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_d(K, W_start, mu_start, F, lamda3, Delta_d_start)
                    W, n, Delta_d, Delta = calc_W_n_Delta_d_Delta(ak, bk, tk, ek)
                    mu = mu_start + 0.5 * (n_el - n)
                    delta_mu = abs(mu - mu_start)
                    mu_start = array(mu)
                    delta_x = linalg.norm(array([W - W_start, Delta_d - Delta_d_start, Delta - Delta_start]))
                    #if cnt % 10 == 0:
                    #    print('cnt = ', cnt, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                    #    print('W = ', W, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)

                    W_start = alpha_HF * W + (1.0 - alpha_HF) * W_start
                    Delta_d_start = alpha_HF * Delta_d + (1.0 - alpha_HF) * Delta_d_start
                    Delta_start = alpha_HF * Delta + (1.0 - alpha_HF) * Delta_start


                elif symmetry == 'sid':
                    lamda3 = U*Delta_start
                    K, F = 1.0, 0.0
                    epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W_start, mu_start, F, lamda3, Delta_s_start, Delta_d_start)
                    W, n, Delta_s, Delta_d, Delta = calc_W_n_Delta_s_Delta_d_Delta(ak, bk, tk, ek)
                    mu = mu_start + 0.5 * (n_el - n)
                    delta_mu = abs(mu - mu_start)
                    mu_start = array(mu)
                    delta_x = linalg.norm(array([W - W_start, Delta_s - Delta_s_start, Delta_d - Delta_d_start, Delta - Delta_start]))
                    #if cnt % 10 == 0:
                    #    print('cnt = ', cnt, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                    #    print('W = ', W,', Delta_s = ', Delta_s, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)

                    W_start = alpha_HF * W + (1.0 - alpha_HF) * W_start
                    Delta_s_start = alpha_HF * Delta_s + (1.0 - alpha_HF) * Delta_s_start
                    Delta_d_start = alpha_HF * Delta_d + (1.0 - alpha_HF) * Delta_d_start
                    Delta_start = alpha_HF * Delta + (1.0 - alpha_HF) * Delta_start

                else:
                    raise ValueError



                if cnt == 100:
                    alpha_HF = 0.2
                    #print('reduce alpha to', alpha_HF)
                if cnt == 200:
                    alpha_HF = 0.1
                    #print('reduce alpha to', alpha_HF)
                if cnt == 300:
                    alpha_HF = 0.05
                    #print('reduce alpha to', alpha_HF)
                if U > 6.0:
                    if cnt == 400:
                        alpha_HF = 0.01
                        #print('reduce alpha to', alpha_HF)
                    if cnt == 500:
                        alpha_HF = 0.005
                        #print('reduce alpha to', alpha_HF)
                    if cnt == 600:
                        alpha_HF = 0.001
                        #print('reduce alpha to', alpha_HF)
                if U > 6.5:
                    accuracy_delta_x = 1.0e-1

                cnt += 1
                if cnt > max_noi:
                    print('After', cnt, 'iterations: ', ' delta_mu =', delta_mu, 'delta_x =', delta_x,
                          ', n_el_rel_deviation =',
                          (n - n_el) / n_el)
                    raise ValueError('No convergence in Hartree-Fock', symmetry)

            print('After', cnt, 'iterations: ', ' delta_mu =', delta_mu, 'delta_x =', delta_x, ', n_el_rel_deviation =',
                  (n - n_el) / n_el)

            if symmetry == 's':
                epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s(1.0, W_start, mu_start, 0.0, U * Delta_start,
                                                               Delta_s_start)
                HF_result_s = array([W_start, Delta_s_start, Delta_start, mu_start])
                HF_energy_s = real((n - 1) * mu_start + 1 / N * sum(epsk - ek * tk) - 4 * V1 * (
                        abs(Delta_s_start) ** 2 - abs(W_start) ** 2) - U * Delta_start ** 2)
                print('s-symmetry results:')
                print('HF_energy_s = ', HF_energy_s)
                print('W = ', W_start, ', Delta_s = ', Delta_s_start, ', Delta = ', Delta_start, ', mu = ', mu_start, ', n = ', n)
            elif symmetry == 'd':
                epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_d(1.0, W_start, mu_start, 0.0, U * Delta_start,
                                                               Delta_d_start)
                HF_result_d = array([W_start, Delta_d_start, Delta_start, mu_start])
                HF_energy_d = real((n - 1) * mu_start + 1 / N * sum(epsk - ek * tk) - 4 * V1 * (
                        abs(Delta_d_start) ** 2 - abs(W_start) ** 2) - U * Delta_start ** 2)
                print('d-symmetry results:')
                print('HF_energy_d = ', HF_energy_d)
                print('W = ', W_start, ', Delta_d = ', Delta_d_start, ', Delta = ', Delta_start, ', mu = ', mu_start, ', n = ', n)
            elif symmetry == 'sid':
                epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(1.0, W_start, mu_start, 0.0, U * Delta_start,
                                                               Delta_s_start, Delta_d_start)
                HF_result_sid = array([W_start, Delta_s_start, Delta_d_start, Delta_start, mu_start])
                HF_energy_sid = real((n - 1) * mu_start + 1 / N * sum(epsk - ek * tk) - 4 * V1 * (
                        abs(Delta_s_start) ** 2 + abs(Delta_d_start) ** 2 - abs(W_start) ** 2) - U * Delta_start ** 2)
                print('s+id-symmetry results:')
                print('HF_energy_sid = ', HF_energy_sid)
                print('W = ', W, ', Delta_s = ', Delta_s_start, ', Delta_d = ', Delta_d_start, ', Delta = ', Delta_start, ', mu = ', mu_start, ', n = ', n)
            print('\n')

        phase_diagramm_HF[n_el_array_index, U_index] = argmin(array([HF_energy_s, HF_energy_d, HF_energy_sid]))
        #exit()













        if not only_HF:
            print('\n')
            print('################################################################### Time-dep. Gutzwiller ###################################################################')
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
                    Delta_start = HF_result_d[2]
                elif symmetry=='sid':
                    mu_start = HF_result_sid[4]
                    W_start = HF_result_sid[0]
                    Delta_s_start = HF_result_sid[1]
                    Delta_d_start = HF_result_sid[2]
                    Delta_start = HF_result_sid[3]


                D_start = n_el ** 2 / 4 - 0.5 * (n_el - 1.0) + sign(n_el - 1.0) * sqrt((0.5 * (n_el - 1.0)) ** 2 + Delta_start ** 2)
                P_start = n_el / 2.0 - D_start
                E_start = 1.0 - 2.0 * P_start - D_start
                eta_start = 0.0

                EPDeta_start = array([E_start, P_start, D_start, eta_start])
                lamda3_start = 0.0
                lamda = 1.0e4
                n = array(n_el)

                accuracy_delta_mu = 1e-2
                accuracy_delta_x = 1e-1
                accuracy_conditions = 1.0e-2
                delta_EPDeta, delta_x = 1.0, 1.0
                delta_mu = 1.0
                conditions_1 = 1.0
                conditions_2 = 1.0
                # parameters for minimization of the slave boson conditions
                lamda_1_2 = array([1.0e4, 1.0e4])
                alpha_GA = 0.5

                leftbound = 1.0e-8  # for low U a higher leftbound works better! (e.g. for U=4.0 use leftbound=1.0e-6)
                cnt = 0
                max_noi = 1000
                min_noi = 50 + int(100 * U / 2)
                ema = 1.0e-12  # accuracy for the energy minimization

                bounds_array = zeros((4, 2), dtype=float)
                for ii in range(3):
                    bounds_array[ii] = (leftbound, 1.0)
                bounds_array[3] = (-pi, pi)

                while delta_mu > accuracy_delta_mu or delta_x > accuracy_delta_x or conditions_1 > accuracy_conditions or conditions_2 > accuracy_conditions or cnt < min_noi:

                    if symmetry == 's':
                        R, I = calc_RI(EPDeta_start)
                        Jz, J, redJz, redJplus = calc_Jz_J_redJz_redJplus(n, Delta_start)

                        lamda3 = calc_lamda3_s(EPDeta_start, W_start, Delta_s_start, mu_start, lamda3_start, n, lamda, Delta_start)
                        K, F = calc_KF(R, I, redJz, redJplus)
                        epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s(K, W_start, mu_start, F, lamda3, Delta_s_start)

                        W, n, Delta_s, Delta = calc_W_n_Delta_s_Delta(ak, bk, tk, ek)
                        mu = mu_start + 0.5 * (n_el - n)
                        delta_mu = abs(mu - mu_start)
                        mu_start = array(mu)
                        delta_x = linalg.norm(array([W - W_start, Delta_s - Delta_s_start, Delta - Delta_start]))
                        #if cnt%10 == 0:
                        #    print('cnt = ', cnt, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                        #    print('W = ', W, ', Delta_s = ', Delta_s, ', Delta = ', Delta, ', mu = ', mu_start)

                        W_start = alpha_GA * W + (1.0 - alpha_GA) * W_start
                        Delta_s_start = alpha_GA * Delta_s + (1.0 - alpha_GA) * Delta_s_start
                        Delta_start = alpha_GA * Delta + (1.0 - alpha_GA) * Delta_start

                        Jz, J, redJz, redJplus = calc_Jz_J_redJz_redJplus(n, Delta_start)
                        lamda3 = calc_lamda3_s(EPDeta_start, W_start, Delta_s_start, mu_start, lamda3_start, n, lamda, Delta_start)
                        energy_minimization = optimize.minimize(calc_Energy_s, EPDeta_start, args=(
                            Jz, J, redJz, redJplus, W, Delta_s, mu, lamda3, n, Delta, lamda), method='L-BFGS-B', options={'disp': False, 'maxiter': 1000}, jac = gradient_of_Energy_s,
                                                                bounds=bounds_array, tol=ema)

                    elif symmetry == 'd':
                        R, I = calc_RI(EPDeta_start)
                        Jz, J, redJz, redJplus = calc_Jz_J_redJz_redJplus(n, Delta_start)

                        lamda3 = calc_lamda3_d(EPDeta_start, W_start, Delta_d_start, mu_start, lamda3_start, n, lamda,
                                               Delta_start)
                        K, F = calc_KF(R, I, redJz, redJplus)
                        epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_d(K, W_start, mu_start, F, lamda3, Delta_d_start)
                        W, n, Delta_d, Delta = calc_W_n_Delta_d_Delta(ak, bk, tk, ek)
                        mu = mu_start + 0.5 * (n_el - n)
                        delta_mu = abs(mu - mu_start)
                        mu_start = array(mu)
                        delta_x = linalg.norm(array([W - W_start, Delta_d - Delta_d_start, Delta - Delta_start]))
                        #if cnt % 10 == 0:
                        #    print('cnt = ', cnt, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                        #    print('W = ', W, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)

                        W_start = alpha_GA * W + (1.0 - alpha_GA) * W_start
                        Delta_d_start = alpha_GA * Delta_d + (1.0 - alpha_GA) * Delta_d_start
                        Delta_start = alpha_GA * Delta + (1.0 - alpha_GA) * Delta_start

                        Jz, J, redJz, redJplus = calc_Jz_J_redJz_redJplus(n, Delta_start)
                        lamda3 = calc_lamda3_d(EPDeta_start, W_start, Delta_d_start, mu_start, lamda3_start, n, lamda, Delta_start)
                        energy_minimization = optimize.minimize(calc_Energy_d, EPDeta_start, args=(
                            Jz, J, redJz, redJplus, W, Delta_d, mu, lamda3, n, Delta, lamda), method='L-BFGS-B', jac = gradient_of_Energy_d,
                                                                options={'disp': False, 'maxiter': 1000},
                                                                bounds=bounds_array, tol=ema)


                    elif symmetry == 'sid':
                        R, I = calc_RI(EPDeta_start)
                        Jz, J, redJz, redJplus = calc_Jz_J_redJz_redJplus(n, Delta_start)

                        lamda3 = calc_lamda3_sid(EPDeta_start, W_start, Delta_s_start, Delta_d_start, mu_start, lamda3_start, n, lamda,
                                               Delta_start)
                        K, F = calc_KF(R, I, redJz, redJplus)
                        epsk, ak, bk, ek, tk = calc_epsk_Ak_Bk_Ek_Tk_s_id(K, W_start, mu_start, F, lamda3, Delta_s_start, Delta_d_start)
                        W, n, Delta_s, Delta_d, Delta = calc_W_n_Delta_s_Delta_d_Delta(ak, bk, tk, ek)
                        mu = mu_start + 0.5 * (n_el - n)
                        delta_mu = abs(mu - mu_start)
                        mu_start = array(mu)
                        delta_x = linalg.norm(array([W - W_start, Delta_s - Delta_s_start, Delta_d - Delta_d_start, Delta - Delta_start]))
                        #if cnt % 10 == 0:
                        #    print('cnt = ', cnt, ', delta_mu = ', delta_mu, ', delta_x = ', delta_x)
                        #    print('W = ', W,', Delta_s = ', Delta_s, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)

                        W_start = alpha_GA * W + (1.0 - alpha_GA) * W_start
                        Delta_s_start = alpha_GA * Delta_s + (1.0 - alpha_GA) * Delta_s_start
                        Delta_d_start = alpha_GA * Delta_d + (1.0 - alpha_GA) * Delta_d_start
                        Delta_start = alpha_GA * Delta + (1.0 - alpha_GA) * Delta_start

                        Jz, J, redJz, redJplus = calc_Jz_J_redJz_redJplus(n, Delta_start)
                        lamda3 = calc_lamda3_sid(EPDeta_start, W_start, Delta_s_start, Delta_d_start, mu_start, lamda3_start, n, lamda, Delta_start)
                        energy_minimization = optimize.minimize(calc_Energy_sid, EPDeta_start, args=(
                            Jz, J, redJz, redJplus, W, Delta_s, Delta_d, mu, lamda3, n, Delta, lamda), method='L-BFGS-B', jac = gradient_of_Energy_sid,
                                                                options={'disp': False, 'maxiter': 1000},
                                                                bounds=bounds_array, tol=ema)

                    else:
                        raise ValueError


                    EPDeta_start = (energy_minimization.x).reshape(4)


                    if cnt > 5:
                        lamda_1_2 = array([5.0e4, 5.0e4])
                    if cnt > 10:
                        lamda_1_2 = array([1.0e5, 1.0e5])
                    if cnt > 15:
                        lamda_1_2 = array([1.0e6, 1.0e6])
                    if cnt > 20:
                        lamda_1_2 = array([1.0e7, 1.0e7])
                    if cnt > 25:
                        lamda_1_2 = array([1.0e8, 1.0e8])

                    if cnt > 35:
                        lamda_1_2 = array([1.0e9, 1.0e9])
                    if cnt > 50:
                        lamda_1_2 = array([1.0e10, 1.0e10])

                    if cnt == 100:
                        alpha_GA = 0.2
                        #print('reduce alpha to', alpha_GA)
                    if cnt == 200:
                        alpha_GA = 0.1
                        #print('reduce alpha to', alpha_GA)
                    if cnt == 300:
                        alpha_GA = 0.05
                        #print('reduce alpha to', alpha_GA)
                    if U > 6.0:
                        if cnt == 400:
                            alpha_GA = 0.01
                            #print('reduce alpha to', alpha_GA)
                        if cnt == 500:
                            alpha_GA = 0.005
                            #print('reduce alpha to', alpha_GA)
                        if cnt == 600:
                            alpha_GA = 0.001
                            #print('reduce alpha to', alpha_GA)
                    if U > 6.5:
                        accuracy_delta_x = 1.0e-1

                    cnt += 1
                    if cnt > max_noi:
                        print('After', cnt, 'iterations: ', ' delta_mu =', delta_mu, 'delta_x =', delta_x,
                              ', n_el_rel_deviation =',
                              (n - n_el) / n_el, ', energy =', energy_minimization.fun, ', conditions:',
                              conditions_1, conditions_2)
                        raise ValueError('No convergence in time dependent GA')

                    conditions_1 = linalg.norm(cond1(EPDeta_start))
                    conditions_2 = linalg.norm(cond1(EPDeta_start))

                print('After', cnt, 'iterations: ', ' delta_mu =', delta_mu, 'delta_x =', delta_x, ', n_el_rel_deviation =',
                      (n - n_el) / n_el, ', energy =', energy_minimization.fun, ', conditions:',
                      conditions_1, conditions_2, '\n')

                if symmetry == 's':
                    GA_energy_s = energy_minimization.fun
                    #print('s-symmetry results:')
                    #print('W = ', W, ', Delta_s = ', Delta_s, ', Delta = ', Delta, ', mu = ', mu_start)
                    #print('EPDeta: ', EPDeta_start, '\n')
                elif symmetry == 'd':
                    GA_energy_d = energy_minimization.fun
                    #print('d-symmetry results:')
                    #print('W = ', W, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)
                    #print('EPDeta: ', EPDeta_start, '\n')
                elif symmetry == 'sid':
                    GA_energy_sid = energy_minimization.fun
                    #print('s+id-symmetry results:')
                    #print('W = ', W, ', Delta_s = ', Delta_s, ', Delta_d = ', Delta_d, ', Delta = ', Delta, ', mu = ', mu_start)
                    #print('EPDeta: ', EPDeta_start, '\n')

            phase_diagramm_time_dependent_GA[n_el_array_index, U_index] = argmin(array([GA_energy_s, GA_energy_d, GA_energy_sid]))









# create phase diagram


# 0: s-symmetry, 1: d-symmetry, 2: s+id-symmetry

d_nel = n_el_array[1] - n_el_array[0]
dU = U_array[1] - U_array[0]


fs = 24
fs2 = 27
#
#print('Phase diagramm HF:')
#print(phase_diagramm_HF)
#print('Phase diagramm GA:')
# print(phase_diagramm_GA)
# print('Phase diagramm time-dependent GA:')
# print(phase_diagramm_time_dependent_GA)
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

show()









