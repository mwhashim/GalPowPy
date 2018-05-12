'''
This module for solving the LPT differential equations
'''
import sys
import os.path
import numpy as np
from scipy.integrate import odeint, trapz, quad
from scipy.interpolate import interp1d, UnivariateSpline
import scipy.optimize as opt
import lmfit

from quintinitconds import quintini
from initconditions import initconds, transfer_func
from equations import equations
from constants import *

import matplotlib.pylab as plt
sys.dont_write_bytecode = True


def FTW(R, k):
    return 3*(np.sin(k*R)-k*R*np.cos(k*R)) / (k*R)**3

def growth(a, **cosmo):
    Omega0_q, Omega0_c, Omega0_b = cosmo['Omega0_q'], cosmo['Omega0_c'], cosmo['Omega0_b']; Omega0_m = Omega0_c# + Omega0_b

    Hubble_a = np.sqrt(Omega0_m/(a * a * a) + (1.0 - Omega0_m - Omega0_q) / (a * a) + Omega0_q)
    growth_int = lambda a: (a/(Omega0_m + a * (1.0 - Omega0_m - Omega0_q) + a * a * a * Omega0_q))**1.5

    return Hubble_a * quad(growth_int, 0, a)[0]

def PowerSpectrum_m(a, k, z_p, z_i, alpha, beta, Background, clustering, smoothing, **cosmo):
    H0 = cosmo['H0']
    fname_ic = 'ICs/ic_%s_%s_zi_%s.npy'%(alpha, beta, z_i)
    
    if os.path.isfile(fname_ic):
        ics = np.load(fname_ic)
    else:
        ics = quintini(z_p, z_i, alpha, beta, **cosmo);
        np.save(fname_ic, ics)

    Delta_m_i = np.array([initconds(ics, k[i], z_p, z_i, alpha, beta, Background, clustering, **cosmo)[5] for i in range(len(k))])
    vars = [odeint(equations, initconds(ics, k[i], z_p, z_i, alpha, beta, Background, clustering, **cosmo), a, args = (k[i], c2_q, alpha, beta, Background, clustering, smoothing), mxstep = 10**3) for i in range(len(k))]
    
    vars_z = odeint(equations, initconds(ics, 1./H0, z_p, z_i, alpha, beta, Background, clustering, **cosmo), a, args = (1./H0, c2_q, alpha, beta, Background, clustering,smoothing), mxstep = 10**3)
    
    ##------ Variable definitions ------------------
    Delta_m = np.array([vars[j][:,5][-1] for j  in range(len(k))]);  D_m_k = (Delta_m/Delta_m_i)/a[-1]
    Delta_m_z= vars_z[:,5]; D_m = (Delta_m_z/Delta_m_z[0])/a

    ##--Power Spectrum calculations ------
    amp_integrand = k**(2+sp_ind)*Delta_m**2 * FTW(8, k)**2; amp_integral = trapz(amp_integrand, k)
    amp_0 = 2*np.pi**2/amp_integral; amp = amp_0*s_8**2
    
    #PowSpec_m = k**sp_ind * Delta_m**2 * amp;
    PowSpec_m = Delta_m**2
    
    sigma_integrand = k**2 * PowSpec_m * FTW(8., k)**2
    s_8_check = pow(trapz(sigma_integrand, k)/(2. * np.pi**2), 0.5)

    print "s_8_calculated: ", s_8_check
    
    PowSpec_Data = k, PowSpec_m; D_m_Data = k, D_m_k
    #D_m_Data = a, D_m

    np.save('Data/Dm_%s_%s_z_%s_clsrt_%s_smth_%s.npy' %(alpha, beta, (1./a[-1] - 1.),clustering, smoothing), D_m_Data)
    np.save('Data/Pm_%s_%s_z_%s_clsrt_%s_smth_%s.npy' %(alpha, beta, (1./a[-1] - 1.),clustering, smoothing), PowSpec_Data)
    return D_m_k, PowSpec_m

def GalaxyPowerSpectrum(a, k, z_p, z_i, alpha, beta, fNL, Background, clustering, smoothing, **cosmo):
    print 'running model:: beta: ',  beta, '  fNL: ',  fNL
    Omega0_c, Omega0_b = cosmo['Omega0_c'], cosmo['Omega0_b']; Omega0_m = Omega0_c + Omega0_b; H0 = cosmo['H0']
    
    fname  = 'Data/Pm_%s_%s_z_%s_clsrt_%s_smth_%s.npy' %(alpha, beta, (1./a[-1] - 1.), clustering, smoothing)
    fname_class  = 'class_data/Pm_%s_%s_z_%s_clsrt_%s.dat' %(alpha, beta, (1./a[-1] - 1.), clustering)
    fname_D = 'Data/Dm_%s_%s_z_%s_clsrt_%s_smth_%s.npy' %(alpha, beta, (1./a[-1] - 1.), clustering, smoothing)
    if os.path.isfile(fname):
        PowSpec_m = np.loadtxt(fname_class, unpack = True)[1]; D_m = np.load(fname_D)[1]
    else:
        D_m, PowSpec_m = PowerSpectrum_m(a, k, z_p, z_i, alpha, beta, Background, clustering, smoothing, **cosmo)

    #k_class, PowSpec_m_class  = plt.loadtxt(fname_class, unpack=True); PowSpec_m = interp1d(k_class, PowSpec_m_class, kind='cubic')(k)
    bias = b_g + 3 * (b_g - 1) * fNL * (delta_c * Omega0_m * H0**2)/((c*k)**2 * transfer_func(k, **cosmo) * D_m)
    #bias = b_g + 3 * (b_g - 1) * fNL * (delta_c * Omega0_m * H0**2)/((c*k)**2 * transfer_func(k, **cosmo) * D_m[-1])

    PowSpec_g = bias**2 * PowSpec_m
    GalaxyPowSpec_Data = k, PowSpec_g
    #np.save('Data/Pg_%s_%s_fNL_%s_z_%s_clsrt_%s_smth_%s.npy' %(alpha, beta, fNL, (1./a[-1] - 1.),clustering, smoothing), GalaxyPowSpec_Data)
    return k, PowSpec_g

#
if __name__ == "__main__":
    cosmo = {'Omega0_q':  0.728815, 'Omega0_c': 0.2260002018, 'Omega0_b': 0.0451, 'Omega0_r': 8.47982e-05, 'H0': 70.3, 'n_s': 0.966 , 'A_s': 1.06, 'delta_H': 1.9 * 10**(-5)}
    H0 = cosmo['H0']

    z_cmb = 1100
    z_p = 10.0**7; z_i = z_cmb; z_f = 0.0
    lga = np.linspace(np.log(1./(1 + z_i)), np.log(1./(1. + z_f)), 100); a = np.exp(lga)
    lgk = np.linspace(np.log(1e-5), np.log(1e1), 100); k = np.exp(lgk)


#    # growth rate
##    z, g_D_m = plt.loadtxt('class_data/LCDM_background.dat', unpack=True, usecols = [0,13])
##    D_m_class = g_D_m/(1./(1. + z)); print D_m_class[-1]
##
##    D_m1 = PowerSpectrum_m(a, k, z_p, z_i, 0.08, 0.05, False, False, False, **cosmo)[0]; print D_m1
##
##    g_m = [growth(a[i], **cosmo) for i in range(len(a))]; D_m = g_m; print D_m[-1]
##
##    plt.semilogx(1./a - 1., D_m/a, '--', label = 'analytic')
##    plt.semilogx(1./a - 1., D_m1, '-.', label = 'my code')
##    plt.semilogx(z, D_m_class, '-', label = 'class')
##    plt.legend()
##    plt.xlim((0.0, 1e3))
##    plt.ylim((1.0, 3.0))
##    plt.show()
#
##    Omega_q, w_x = PowerSpectrum_m(a, k, z_p, z_i, 0.08, 0.0, False, False, **cosmo)[:2]
##    plt.semilogx(a, w_x)
##    plt.show()
##
#    PowSpec_g_ref = GalaxyPowerSpectrum(a, k, z_p, z_i, 0.0, 0.0, 0.0, False, False, False,**cosmo)
#
#    p = lmfit.Parameters()
#    p.add_many(('fNL', -10.0, True, -500.0, 0.0),
#               ('beta', 0.05, False, 0.01, 0.2)
#               )
#
#    def residual(p):
#        v = p.valuesdict()
#        model = GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, v['beta'], v['fNL'], False, False, False, **cosmo)
#        return  model - PowSpec_g_ref
#
#    mini = lmfit.Minimizer(residual, p)
#
#    out1 = mini.minimize(method='Nelder'); out2 = mini.minimize(method='leastsq', params=out1.params)
#    fNL_bestfit = out2.params['fNL'].value
#
#    lmfit.report_fit(out2.params, min_correl=0.5)
#    kk, PowSpecg = GalaxyPowerSpectrum(a, k, z_p, z_i, 0.0, 0.0, -10.0, False, True, False, **cosmo)
#    kk1, PowSpecg1 = GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, 0.05, -1.0, False, True, False, **cosmo)
    plt.figure()
#    plt.loglog(kk, PowSpecg, '--', label = 'LCDM')
#    plt.loglog(kk1, PowSpecg1, ':', label = 'qLCDM')
    plt.loglog(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.0, 0.0, 0.0, False, True, False, **cosmo), '--', label = 'qCDM-beta: 0.05')
###    plt.loglog(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, 0.05, 0.0, False, False, True, **cosmo), '--', label = 'qCDM-beta: 0.05-False')
###    plt.loglog(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, 0.05, 0.0, False, True, False, **cosmo), '--', label = 'qCDM-beta: 0.05-smoothing')
###    plt.loglog(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, 0.05, -10.0, False, True, **cosmo), '--', label = 'qCDM-beta: 0.05-fNL:-10')
###    plt.loglog(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, 0.15, -10.0, False, True, **cosmo), '--', label = 'qCDM-beta: 0.15-fNL:-10')
##
###    plt.loglog(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, 0.0, 0.0, False, False, **cosmo), '-.', label = 'qCDM')
###    plt.loglog(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, 0.05, 0.0, False, False, **cosmo), '--', label = 'qCDM-beta: 0.05')
###    plt.loglog(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, 0.05, 0.0, False, True, **cosmo), '--', label = 'qCDM-beta: 0.05')
##    plt.loglog(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, 0.05, 0.0, False, False, False, **cosmo), '--', label = 'qCDM-beta: 0.05')
##    plt.loglog(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.0, 0.0, -10.0, False, False, False, **cosmo), '--', label = 'qCDM-beta: 0.05')
##
#    plt.loglog(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.0, 0.0, fNL_bestfit, False, False, False, **cosmo), '-.', color = 'red', label = 'LCDM-fNL')
#    plt.loglog(k,  GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, 0.05, fNL_bestfit, False, False, False, **cosmo), '--', color = 'blue', label =  'qCDM-fNL')
##    plt.axvline(1./H0)
    plt.legend()
    plt.show()

#-----------------------------------------------
#   Bayesian method
#    sigma  = 1.0
#    def loglike(p):
#        #return 0.5 * np.sum( ( (PowSpec_g_ref - GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, p[0], p[1], False, True, **cosmo))/sigma )**2  )
#        return 0.5 * np.sum( ( ( 1.0 - GalaxyPowerSpectrum(a, k, z_p, z_i, 0.08, p[0], p[1], False, True, **cosmo)/PowSpec_g_ref)/sigma )**2  )
#
#    dbeta = 0.05/30.
#    betaarray = np.arange(dbeta, 2. * 0.05, dbeta)
#
#    dfNL = fNL_bestfit/60.
#    fNLarray = np.arange(dfNL, 2. * fNL_bestfit, dfNL)
#
#    post = np.empty([len(betaarray),len(fNLarray)])
#    for i, beta in enumerate(betaarray):
#        for j, fNL in enumerate(fNLarray):
#            post[i,j] = np.exp(-loglike(np.array([beta, fNL])))
#
#    X, Y = np.meshgrid(fNLarray, betaarray)
#
#    post = post/np.sum(post)/dfNL/dbeta
#    plt.contour(X,Y,post)
#    plt.xlabel(r'$\rm fNL$')
#    plt.ylabel(r'$\beta$')
#
#    np.save('fNL_beta_post_noclstr_z%s.npy' %z_f, post)
#    plt.savefig('2dcountor.png')
#    plt.show()

