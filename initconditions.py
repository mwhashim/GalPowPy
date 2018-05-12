'''
Here we define the initial conditions for the system of differential equations

independent variables: k
constants: H0, delta_H, A_s, n_s, beta
initial values of independent variables: z_p, z_i


'''
import sys
import numpy as np
from quintinitconds import quintini
from constants import *

import matplotlib.pylab as plt
sys.dont_write_bytecode = True

def transfer_func(k, **cosmo):
    Omega0_c, Omega0_b = cosmo['Omega0_c'], cosmo['Omega0_b']; Omega0_m = Omega0_c + Omega0_b
    #---: transfer k_eq
    Gamma =   Omega0_m * 0.7 * np.exp(-Omega0_b * (1. - np.sqrt(2. * 0.7)/Omega0_m)); k_eq = k/Gamma
    return np.log(1. + 2.34 * k_eq)/(2.34 * k_eq) * (1. + 3.89 * k_eq + (16.19 * k_eq)**2 + (5.46 * k_eq)**3 + (16.71 * k_eq)**4)**(-1./4.)

def initconds(ics, k, z_p, z_i, alpha, beta, Background, clustering, **cosmo):
    Omega0_c, Omega0_b, H0, n_s, A_s, delta_H = cosmo['Omega0_c'], cosmo['Omega0_b'], cosmo['H0'], cosmo['n_s'], cosmo['A_s'], cosmo['delta_H']
    Omega0_m = Omega0_c + Omega0_b    
    # initial conditions
    x_i, y_i, r_i, v_i, h_i, w_q_i, Omega_q_i, Omega_c_i, Omega_m_i = ics
    Omega_m_i = 1. - Omega_q_i
    # primordial gravitational field phi
    A = 50./9. * np.pi**2 * delta_H**2 * A_s**2
    Phi_p =  np.sqrt(A) * (H0 * k/c)**(-3./2.) * (k)**((n_s - 1.)/2.) * Omega0_m

#    Pow_p = A_s * (k/ks)**(n_s - 4.0); Phi_p = np.sqrt(Pow_p)
    
    #
#    ks = 0.05
#    Phi_p =  - np.sqrt(A) * (k/ks)**((n_s - 1.)/2.)

    # coupling
    phi_dash = np.sqrt(6) * x_i
    Q_m_i = np.sqrt(2. * (8. * np.pi * G)/3.) * beta * phi_dash * Omega_m_i; Q_q_i = - Q_m_i

    # some definitions
    w_m_eff_i = - Q_m_i/(3. * h_i * Omega_m_i)
    w_q_eff_i = w_q_i - Q_q_i/(3. * h_i * Omega_q_i)

    rhodash_mq_i = (Omega_q_i/Omega_m_i) * (1. + w_q_eff_i)/(1. + w_m_eff_i)
    mu_delta = rhodash_mq_i/(1. - rhodash_mq_i)
    
    #- Phi_i
    Phi_i = 9./10. * Phi_p * transfer_func(H0 * k/c, **cosmo)
    #- Delta_m_i
    Delta_m_i = 2./3. * (k/((1./(1. + z_i)) * h_i))**2 * Phi_i/Omega_m_i * (1. + mu_delta) # changing sign check()
    #- u_m_i
    u_m_i = 2./3. * 1./((1. + w_q_i * Omega_q_i) * h_i) * Phi_i
    #- Delta_q_i
    Delta_q_i = 2./3. * (k/((1./(1. + z_i)) * h_i))**2 * Phi_i/Omega_q_i * mu_delta
    #- u_q_i
    u_q_i = u_m_i
    
    if alpha == 0.0 and beta == 0.0:
        Delta_q_i = 0.0; u_q_i = 0.0
    
    if Background:
        initial_conds = np.array([x_i, y_i, r_i, v_i, h_i])
    else:
        if clustering:
            initial_conds = np.array([x_i, y_i, r_i, v_i, h_i, Delta_m_i, Delta_q_i, u_m_i, u_q_i, Phi_i])
        else:
            initial_conds = np.array([x_i, y_i, r_i, v_i, h_i, Delta_m_i, 0.0, u_m_i, 0.0, Phi_i])
    return initial_conds

if __name__ == "__main__":
    cosmo = {'Omega0_q':  0.728815, 'Omega0_c': 0.2260002018, 'Omega0_b': 0.0451, 'Omega0_r': 8.47982e-05, 'H0': 70.3, 'n_s': 0.966 , 'A_s': 1.06, 'delta_H': 1.9 * 10**(-5)}; H0 = cosmo['H0']
    beta = 0.05; alpha = 0.08
    z_p = 10**7; z_i = 100.0
    lgk = np.linspace(np.log(1e-5), np.log(1e0), 100); k = np.exp(lgk)
    
    ics = quintini(z_p, z_i, 0.08, 0.0, **cosmo); qics = quintini(z_p, z_i, 0.08, 0.05, **cosmo)
    initconds(ics, 1/H0, z_p, z_i, 0.08, 0.0, False, True, **cosmo)
    initconds(qics, 1/H0, z_p, z_i, 0.08, 0.05, False, True, **cosmo)

#    #plt.loglog(k, transfer_func(k, **cosmo),'--')
#    plt.loglog(k, Delta_m_i0**2, '-.')
#    plt.axvline(1/H0)
#    plt.show()


