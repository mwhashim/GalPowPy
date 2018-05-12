'''
This equations module is only for quientessence interacting DE

independent variables: a, k
background variables: x1, x2, h=H/H0; Omega_m,q, w_q, h
perturbations variables:

constants: H0, alpha, c2_q
parameters: beta
'''
import sys
import numpy as np
from constants import *
sys.dont_write_bytecode = True

def equations(vars, a, k, c2_q, alpha, beta, Background, clustering, smoothing):
    if Background:
        x, y, r, v, h = vars[0], vars[1], vars[2], vars[3], vars[4]
    
        dxda = (x/2. * (3. * x**2 - 3. * y**2 + r**2 - 3.) + alpha * y**2 + beta * (1. - x**2 - y**2 - r**2 - v**2))/a
        dyda = (y/2. * (3. * x**2 - 3. * y**2 + r**2 + 3.) - alpha * x * y)/a
        drda = (r/2. * (3. * x**2 - 3. * y**2 + r**2 - 1.))/a
        dvda = (v/2. * (3. * x**2 - 3. * y**2 + r**2))/a
        dhda = (-h/2. * (3. * x**2 - 3. * y**2 + r**2 + 3.))/a
    
        diff_sys = np.array([dxda, dyda, drda, dvda, dhda])
    else:
        # background
        x, y, r, v, h = vars[0], vars[1], vars[2], vars[3], vars[4]
        
        dxda = (x/2. * (3. * x**2 - 3. * y**2 + r**2 - 3.) + alpha * y**2 + beta * (1. - x**2 - y**2 - r**2 - v**2))/a
        dyda = (y/2. * (3. * x**2 - 3. * y**2 + r**2 + 3.) - alpha * x * y)/a
        drda = (r/2. * (3. * x**2 - 3. * y**2 + r**2 - 1.))/a
        dvda = (v/2. * (3. * x**2 - 3. * y**2 + r**2))/a
        dhda = (-h/2. * (3. * x**2 - 3. * y**2 + r**2 + 3.))/a
        
        # some definitions
        Omega_m = 1. - x**2 - y**2; Omega_q = x**2 + y**2
        Omega_c = 1.0 - x**2 - y**2 - r**2 - v**2
        w_q = (x**2 - y**2)/(x**2 + y**2); w_eff = w_q * Omega_q

        w_q_dash = (2. * a)/Omega_q * (x * dxda * (1. - w_q) - y * dyda * (1. + w_q))


        phi_dash = np.sqrt(6) * x

        #  coupling terms
        Q_m = np.sqrt(2. * (8. * np.pi * G)/3.) * beta * phi_dash * Omega_m; Q_q = - Q_m

        # some definitions
        w_m_eff = - Q_m/(3. * h * Omega_m); w_q_eff = w_q - Q_q/(3. * h * Omega_q)
        c2_a = w_q - w_q_dash/(3. * (1. + w_q_eff))

        # LPT
        Delta_m, Delta_q, u_m, u_q, Phi = vars[5], vars[6], vars[7], vars[8], vars[9]
        # some definitions
        u_tot = 1./(1. + w_eff) * (Omega_m * u_m + (1. + w_q) * Omega_q * u_q)

        delta_m = Delta_m + 3 * a * h * (1 + w_m_eff) * u_m
        
        if clustering:
            delta_q = Delta_q + 3 * a * h * (1 + w_q_eff) * u_q
            TT = -3 * a * h * (1 + w_q) * (c2_q - c2_a) * u_q/(delta_q); Cs2 = c2_q + TT
        else:
            delta_q = 0.0; TT = 0.0; Cs2 = c2_q + TT


        
        if alpha == 0.0 and beta == 0.0:
            delta_Q_m = Q_m * (Delta_q + 3. * a * h * (1 + w_q_eff) * u_q); delta_Q_q = - delta_Q_m
        else:
            delta_Q_m = Q_m * (delta_m + 3./2. * Phi + 3. * Omega_q * delta_q/(2 * phi_dash**2) * (1. + Cs2)); delta_Q_q = - delta_Q_m


        f_m = Q_m * (u_tot - u_m); f_q = - f_m
        #f_m = Q_m * (u_tot - u_q); f_q = - f_m

        if alpha == 0.0 and beta == 0.0:
            Omega_dash_m = 0.0; Omega_dash_q = 0.0
        else:
            Omega_dash_m = (x/2. * (3. * x**2 - 3. * y**2 - 3.) + alpha * y**2 + beta * (1. - x**2 - y**2))/x  - 3./2. * (x**2 - y**2 + 1.) # check()
            Omega_dash_q = Omega_dash_m - 3. * (w_m_eff - w_q_eff)
        
        #- dPhi/da
        dPhida =  Phi/a + (3./2.) * h * (1. + w_eff) * u_tot # sign check(v)
        #- du_m/da
        RHS_u_m = 1./(Omega_m * a * h) * (Q_m * (u_tot - u_m) + f_m)
        du_mda = - u_m/a + Phi/(a**2 * h) +  RHS_u_m # check Phi sign ()

        #- dDelta_m/da
        RHS_Delta_m = (  a * Q_m/Omega_m * Omega_dash_m * u_m
                       - a * Q_m/Omega_m * (3 + Q_m/(h * Omega_m)) * (u_tot - u_m)
                       - a/Omega_m * (3. + Q_m/(h * Omega_m)) * f_m
                       - Q_m/(Omega_m * h) * Delta_m
                       + 2. * Q_m/(Omega_m * h) * Phi
                       + a * Q_m/Omega_m * (3. + Q_m/(h * Omega_m)) * u_m
                       + delta_Q_m/(Omega_m * h)
                       )
        dDelta_mda = k**2/(a**2 * h) * u_m + 9./2. * h * (1. + w_eff) * (u_m - u_tot) +  RHS_Delta_m

        if alpha == 0.0 and beta == 0.0:
            dDelta_qda = 0.0; du_qda = 0.0
        else:
            if clustering:
                #- dDelta_q/da
                RHS_Delta_q = (  a * Q_q/Omega_q * Omega_dash_q * u_q
                               - a * Q_q/Omega_q * (3. + Q_q/((1. + w_q) * h * Omega_q)) * (u_tot - u_q)
                               - a/Omega_q * (3. + Q_q/((1 + w_q) * h * Omega_q)) * f_q
                               - Q_q/(Omega_q * h) * (c2_q/(1 + w_q) + 1.) * Delta_q
                               + 2. * Q_q/(Omega_q * h) * Phi
                               + a * Q_q/Omega_q * (3. * (1. + w_q) + Q_q/(h * Omega_q)) * u_q
                               + delta_Q_q/(Omega_q * h)
                               )
                dDelta_qda = 3. * w_q/a * Delta_q + k**2/(a**2 * h) * (1. + w_q) * u_q + 9./2. * h * (1. + w_q) * (1. + w_eff) * (u_q - u_tot) +  RHS_Delta_q
                #- du_q/da
                RHS_u_q = 1./((1. + w_q) * Omega_q * a * h) * (Q_q * (u_tot - u_q) + f_q)
                if smoothing:
                    #-: as an assumption we set phi' = 0.0 since
                    dDelta_qda = 0.0; du_qda =  -u_q/a  + Phi/(a**2 * h) + RHS_u_q
                else:
                    du_qda =  -u_q/a - c2_q/((1. + w_q) * a**2 * h) * Delta_q + Phi/(a**2 * h) + RHS_u_q # check Phi sign ()

            else:
                dDelta_qda = 0.0; du_qda = 0.0
        
        diff_sys = np.array([dxda, dyda, drda, dvda, dhda, dDelta_mda, dDelta_qda, du_mda, du_qda, dPhida])

    return diff_sys
