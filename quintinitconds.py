'''
This module for generating the initial conditions for coupled quintessence background variables

constants: alpha, A_pot
'''

# define the differential system!!
from __future__ import division
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d, UnivariateSpline
import sys
sys.dont_write_bytecode = True

import matplotlib.pylab as plt

def model(vars, N, alpha, beta):
    x, y, r, v, H = vars[0], vars[1], vars[2], vars[3], vars[4]
    
    dxdN = x/2. * (3. * x**2 - 3. * y**2 + r**2 - 3.) + alpha * y**2 + beta * (1. - x**2 - y**2 - r**2 - v**2)
    dydN = y/2. * (3. * x**2 - 3. * y**2 + r**2 + 3.) - alpha * x * y
    drdN = r/2. * (3. * x**2 - 3. * y**2 + r**2 - 1.)
    dvdN = v/2. * (3. * x**2 - 3. * y**2 + r**2)
    dHdN = -H/2. * (3. * x**2 - 3. * y**2 + r**2 + 3.)
    
    return np.array([dxdN, dydN, drdN, dvdN, dHdN])

def quintini(z_i, z_init, alpha, beta, **cosmo):
    Omega0_q, Omega0_b, Omega0_r, H0 = cosmo['Omega0_q'], cosmo['Omega0_b'], cosmo['Omega0_r'], cosmo['H0']
    r_0 = np.sqrt(Omega0_r); v_0 = np.sqrt(Omega0_b)
    
    # initial guess
    w0_qi = -0.9
    # redshift range
    
    N_i = np.log(1./(1.+ z_i)); N_revs = np.linspace(N_i, 0.0, 10**3)[::-1]
    N_init = np.log(1./(1.+ z_init))
    
    n_step = 10**3; n = 10*3; RNorm = 0.0; pow = -3.0; m = 1.0; mm = 1.0; acc = -3.0

    Stat = False
    if alpha == 0.0 and beta == 0.0:
        x_0 = 0.0; y_0 = np.sqrt(Omega0_q)
        ICs_revs = np.array([x_0, y_0, r_0, v_0, H0])
        Norm_ICs = odeint(model, ICs_revs, N_revs, args = (alpha, beta), mxstep = n_step)
        X,Y,R,V,H = Norm_ICs[:,0],Norm_ICs[:,1],Norm_ICs[:,2],Norm_ICs[:,3],Norm_ICs[:,4]
        OmegaN_L = X**2 + Y**2; OmegaN_r = R**2; OmegaN_b = V**2; OmegaN_c = 1.0 - OmegaN_L - OmegaN_b - OmegaN_r
        OmegaN_m = 1.0 - OmegaN_L - OmegaN_r
        w_x = (X**2 - Y**2)/(X**2 + Y**2)
    else:
        while (Stat == False):
            for i in range(n):
                x_0 = np.sqrt(Omega0_q/2. * (1 + w0_qi))
                y_0 = np.sqrt(Omega0_q/2. * (1 - w0_qi))
                ICs_revs = np.array([x_0, y_0, r_0, v_0, H0])
                
                #setup initial conditions !!
                Norm_ICs = odeint(model, ICs_revs, N_revs, args = (alpha, beta), mxstep = n_step)#,rtol = 1.49012e-8 , atol = 1.49012e-8)
                X,Y,R,V,H = Norm_ICs[:,0],Norm_ICs[:,1],Norm_ICs[:,2],Norm_ICs[:,3],Norm_ICs[:,4]
                Stat = np.isclose(R[-1], 1.0, rtol=10**acc, atol=10**acc)
                
                if RNorm > R[-1]**2:
                    print 'r : ', RNorm, '   w_x : ', w0_qi
                    w0_qi = w0_qi + m * 1.0 * 10**pow; m = 10**(-pow - mm); pow = pow - 1; mm = mm + 1
                    RNorm = 0.0
                    break
                
                elif Stat == True:
                    print 'Normalized EoS: ', w0_qi, ', Max value of r^2: ', R[-1]**2
                    OmegaN_L = X**2 + Y**2; OmegaN_r = R**2; OmegaN_b = V**2; OmegaN_c = 1.0 - OmegaN_L - OmegaN_b - OmegaN_r
                    OmegaN_m = 1.0 - OmegaN_L - OmegaN_r
                    w_x = (X**2 - Y**2)/(X**2 + Y**2)
                    break
        
                elif  RNorm == R[-1]**2 :
                    print 'Can not procced anymore !! Decrease the accuracy level or decrease the number of e-folds !!'
                    print 'accuracy : ',  acc
                    sys.exit()
                
                else:
                    #print 'r : ', R[-1] ,'Status : ', Stat, '   w_x : ', w0_xi
                    pass
                
                RNorm = R[-1]**2
                w0_qi = w0_qi - 1.0 * 10**pow

    X_i = interp1d(N_revs, X, kind='cubic')(N_init)
    Y_i = interp1d(N_revs, Y, kind='cubic')(N_init)
    R_i = interp1d(N_revs, R, kind='cubic')(N_init)
    V_i = interp1d(N_revs, V, kind='cubic')(N_init)
    h_i = interp1d(N_revs, H, kind='cubic')(N_init)/H0
    w_q_i = interp1d(N_revs, w_x, kind='cubic')(N_init)
    Omega_q_i = interp1d(N_revs, OmegaN_L, kind='cubic')(N_init)
    Omega_c_i = interp1d(N_revs, OmegaN_c, kind='cubic')(N_init)
    Omega_m_i = interp1d(N_revs, OmegaN_m, kind='cubic')(N_init)
    
#    ics_data = X_i, Y_i, R_i, V_i, h_i, w_q_i, Omega_q_i, Omega_m_i
#    np.save('./ICs/ic_%s_%s_zi_%s.npy' %(alpha, beta, z_init), ics_data)
    return X_i, Y_i, R_i, V_i, h_i, w_q_i, Omega_q_i, Omega_c_i, Omega_m_i

if __name__ == "__main__":
    cosmo = {'Omega0_q':  0.728815, 'Omega0_c': 0.2260002018, 'Omega0_b': 0.0451, 'Omega0_r': 8.47982e-05, 'H0': 1.0}
    alpha = 0.08; beta = 0.05;
    z_i = 10**7; z_f = 0.0
    lga = np.linspace(np.log(1./(1 + z_i)), np.log(1./(1. + z_f)), 100); a = np.exp(lga); z = 1./a - 1.
    
    #quintini(10**7, 1100, alpha, beta, **cosmo)

    w_q_i, Omega_q_i, Omega_m_i = quintini(z_i, z, alpha, beta, **cosmo)[5:]#, quintini(z_i, z, alpha, beta, **cosmo)[7]

    plt.figure()
    plt.semilogx(1./(1. + z), Omega_m_i)
    plt.semilogx(1./(1. + z), Omega_q_i)
    plt.show()

    plt.figure()
    plt.semilogx(1./(1. + z), w_q_i)
    plt.show()

#print quintini(10**7, 0.0, alpha, beta, **cosmo)


