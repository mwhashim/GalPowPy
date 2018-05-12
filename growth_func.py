import sys
import numpy as np
import matplotlib.pylab as plt

from quintinitconds import quintini

from powerspectrum import PowerSpectrum_m
sys.dont_write_bytecode = True


if __name__ == "__main__":
    cosmo = {'Omega0_q':  0.728815, 'Omega0_c': 0.2260002018, 'Omega0_b': 0.0451, 'Omega0_r': 8.47982e-05, 'H0': 70.3, 'n_s': 0.966 , 'A_s': 1.06, 'delta_H': 1.9 * 10**(-5)}
    H0 = cosmo['H0']
    
    z_p = 10.0**7; z_i = 1100; z_f = 0.0
    lga = np.linspace(np.log(1./(1 + z_i)), np.log(1./(1. + z_f)), 1000); a = np.exp(lga)
    lgk = np.linspace(np.log(1e-5), np.log(1e-1), 100); k = np.exp(lgk)

    D_m, PowSpec_m,  D_m_z, d_m_z = PowerSpectrum_m(a, k, z_p, z_i, 0.0, 0.0, False, True, **cosmo)
    DD_m = (D_m_z/D_m_z[0]); dD_m = (d_m_z/d_m_z[0]) * a[0];
#print D_m_z
    plt.figure()
    plt.semilogx(1./a -1., (DD_m/DD_m[-1]), '--')
    #plt.semilogx(1./a -1., dD_m, '-.')
    plt.show()
