import sys
import numpy as np
import scipy.optimize as opt
import lmfit

import matplotlib.pylab as plt
from powerspectrum import GalaxyPowerSpectrum

sys.dont_write_bytecode = True
def degenline(beta, fNL):
    p = lmfit.Parameters()
    p.add_many(('fNL', fNL, True, -200.0, 100.0),
               ('beta', beta, False, 0.0, 0.1)
               )
    PowSpec_g_ref = GalaxyPowerSpectrum(a, k, z_p, z_i, 0.0, 0.0, 0.0, False, True, True, **cosmo)
    def residual(p):
       v = p.valuesdict()
       model = GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, v['beta'], v['fNL'], False, True, True, **cosmo)
       return  1.0 - model/PowSpec_g_ref

    mini = lmfit.Minimizer(residual, p)
    out1 = mini.minimize(method='Nelder'); out2 = mini.minimize(method='leastsq', params=out1.params)
    beta_bestfit = out2.params['beta'].value; fNL_bestfit = out2.params['fNL'].value
    lmfit.report_fit(out2.params, min_correl=0.5)
    return beta_bestfit, fNL_bestfit


if __name__ == "__main__":
    cosmo = {'Omega0_q':  0.728815, 'Omega0_c': 0.2260002018, 'Omega0_b': 0.0451, 'Omega0_r': 8.47982e-05, 'H0': 70.3, 'n_s': 0.966 , 'A_s': 1.06, 'delta_H': 1.9 * 10**(-5)}
    H0 = cosmo['H0']
    
    z_cmb = 1100; z_p = 10.0**7; z_i = z_cmb; z_f = 0.0
    lga = np.linspace(np.log(1./(1 + z_i)), np.log(1./(1. + z_f)), 100); a = np.exp(lga)
    lgk = np.linspace(np.log(1e-4), np.log(1e0), 100); k = np.exp(lgk)
    
    beta = 0.05; fNL = -10.0
    dbeta = beta/25.
    gbeta_array=np.array([0.0]); gbeta_array = np.append(gbeta_array, np.arange(dbeta, 2. * beta, dbeta)); gbeta_array = np.append(gbeta_array, 0.1)
    #gbeta_array=np.arange(0.0, 0.1, 0.01)
    print gbeta_array

    beta_array = np.empty(len(gbeta_array)); fNL_array =np.empty(len(gbeta_array))
    for i in range(len(gbeta_array)):
        beta_bestfit, fNL_bestfit = degenline(gbeta_array[i], fNL)
        beta_array[i] = beta_bestfit; fNL_array[i] = fNL_bestfit

    degen_data = beta_array, fNL_array
    np.save('Degen_data/beta_fNL_degen_clstr.npy', degen_data)

    beta_bestfit = 0.0; fNL_bestfit = 0.0
    Pg_LCDM = GalaxyPowerSpectrum(a, k, z_p, z_i, 0.0, 0.0, 0.0, False, False, False, **cosmo)

    plt.figure()
    plt.semilogx(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, 0.0, 0.0, False, False, False, **cosmo)/Pg_LCDM, '-.', label = 'beta: %s' %beta_bestfit)
    plt.semilogx(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, 0.15, 0.0, False, False, False, **cosmo)/Pg_LCDM, '-.', label = 'fNL:%s' %fNL_bestfit)
    #    plt.semilogx(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, beta_bestfit, fNL_bestfit, False, False, False, **cosmo)/Pg_LCDM, '--', label = 'beta: %s - fNL: %s' %(beta_bestfit, fNL_bestfit))
    plt.axvline(1./H0)
    plt.axhline(1.0, linestyle = '--')
    plt.ylim((0.0, 3))
    plt.legend()
    plt.show()



