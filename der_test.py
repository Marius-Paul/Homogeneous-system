from numpy import *
from matplotlib.pyplot import *
from scipy.optimize import fsolve


def GA_CRI_ZZ(D, n, cidciu):
    Jx = cidciu
    Jz = 0.5 * (n - 1.0)
    J = sqrt(Jx ** 2 + Jz ** 2)
    return 2.0 * (0.5 + Jz - D) / (0.25 - J ** 2) * (D - Jz + sqrt(D - Jz - J) * sqrt(D - Jz + J))


def d_GA_CRI_ZZ_dJx(D, n, cidciu):
    Jx = cidciu
    Jz = 0.5 * (n - 1.0)
    J = sqrt(Jx ** 2 + Jz ** 2)
    u = 0.25 - J ** 2
    v = n / 2.0 - D
    z2 = GA_CRI_ZZ(D, n, cidciu)
    print('sqrt = ',  (D - Jz - J))
    return 2.0 * Jx / (0.25 - J ** 2) * (z2 - (n / 2.0 - D) / ( z2*u/(2.0*v) - 0.5+v))


def d_GA_CRI_ZZ_dD(D, n, cidciu):
    Jx = cidciu
    Jz = 0.5 * (n - 1.0)
    J = sqrt(Jx ** 2 + Jz ** 2)
    return GA_CRI_ZZ(D, n, cidciu)  / (sqrt(D - Jz - J) * sqrt(D - Jz + J) )- GA_CRI_ZZ(D, n, cidciu)/(n/2.0 - D)

def d_GA_CRI_ZZ_dD_numerisch(D, n, cidciu):
    delta = 1e-6
    return (GA_CRI_ZZ(D + delta, n, cidciu) - GA_CRI_ZZ(D - delta, n, cidciu)) / (2.0 * delta)

def d_GA_CRI_ZZ_dJx_numerisch(D, n, cidciu):
    delta = 1e-6
    return (GA_CRI_ZZ(D, n, cidciu + delta) - GA_CRI_ZZ(D, n, cidciu - delta)) / (2.0 * delta)

c_vec = linspace(0.0, 0.5, 10)
n_vec = linspace(0.0, 1.0, 10)
n_test = 0.0532
D_test = 0.32532


def Q(D, n, cidciu):
    Jx = cidciu
    Jz = 0.5 * (n - 1.0)
    J = sqrt(Jx ** 2 + Jz ** 2)
    u = 0.25 - J ** 2
    v = n/2.0 - D
    A = GA_CRI_ZZ(D, n, cidciu)*(u/(2.0*v))
    sXY = sqrt(D - Jz - J) * sqrt(D - Jz + J)
    return A/Jx * (sXY - v)/(v-GA_CRI_ZZ(D, n, cidciu)*sXY)





for c in c_vec:
    for n in n_vec:
        print('n', n, 'c', c)
        #print('dD', d_GA_CRI_ZZ_dD(D_test, n, c) - d_GA_CRI_ZZ_dD_numerisch(D_test, n, c))
        print('dQ', (Q(D_test, n, c) - d_GA_CRI_ZZ_dD_numerisch(D_test, n, c)/d_GA_CRI_ZZ_dJx_numerisch(D_test, n, c))/Q(D_test, n, c), '\n')


D_vec = linspace(-0.8, 0.2, 1000)
c_test = 0.023686/20.0

#plot(D_vec, GA_CRI_ZZ(D_vec, n_test, c_test))
#show()


n_t = 0.1111111111111111
c_t = 0.2222222222222222
print('d_GA_CRI_ZZ_dJx = ', d_GA_CRI_ZZ_dJx(D_test, n_t, c_t) )



#for D in D_vec:
#    print(d_GA_CRI_ZZ_dJx(D, n_test, c_test) - d_GA_CRI_ZZ_dJx_numerisch(D, n_test, c_test))

print(GA_CRI_ZZ(c_test**2 + n_test**2/4, n_test, c_test))

print('Verhältnis:', d_GA_CRI_ZZ_dD(D_test, n_test, c_test) / d_GA_CRI_ZZ_dJx(D_test, n_test, c_test))

print(d_GA_CRI_ZZ_dD(D_test, n_test, c_test))
print(d_GA_CRI_ZZ_dJx(D_test, n_test, c_test))


U = 4.0
lamda_test = 0.05
def dE_dD(D, n, cidciu, lamda):
    Jx = cidciu
    Jz = 0.5 * (n - 1.0)
    J = sqrt(Jx ** 2 + Jz ** 2)
    u = 0.25 - J ** 2
    v = n/2.0 - D
    z2 = GA_CRI_ZZ(D, n, cidciu)
    A = z2*(u/(2.0*v))
    sXY = sqrt(D - Jz - J) * sqrt(D - Jz + J)
    return -2.0*lamda*A*(v - sXY) + U*((v - z2*sXY)*Jx)

print(fsolve(dE_dD, 0.05, args=(n_test, c_test, lamda_test)))

#plot(D_vec, dE_dD(D_vec, n_test, c_test, lamda_test))
#grid()
#show()