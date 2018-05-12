import numpy as np
import scipy.optimize as opt
import lmfit

import matplotlib.pylab as plt


beta, fNL = np.load('Degen_data/beta_fNL_degen_clstr.npy')
beta1, fNL1 = np.load('Degen_data/beta_fNL_degen_nclstr.npy')

p = lmfit.Parameters()
p.add_many(('alpha', 1.0, True, -1000, 100.0),
           ('gamma', 1.0, True, -100.0, 100.0),
           ('const', 0.0, True, -10.0, 10.0)
           )
def curvefit(alpha, gamma, cc):
    return 1000.0 * alpha* beta**gamma + cc
def residual(p):
   v = p.valuesdict()
   return  fNL - curvefit(v['alpha'], v['gamma'],v['const'])

mini = lmfit.Minimizer(residual, p)
out1 = mini.minimize(method='Nelder'); out2 = mini.minimize(method='leastsq', params=out1.params)
alpha_bestfit = out2.params['alpha'].value; gamma_bestfit = out2.params['gamma'].value; cc = out2.params['const']

lmfit.report_fit(out2.params, min_correl=0.5)

plt.plot(beta, fNL, '+')
plt.plot(beta1, fNL1, '--+')
plt.plot(beta, curvefit(alpha_bestfit, gamma_bestfit, cc), '--')
plt.hlines(fNL[25], 0, beta[25], color='red', linestyle = '--')
plt.vlines(beta[25], fNL1.min(), fNL[25], color='red', linestyle = '--')
plt.hlines(fNL1[25], 0, beta1[25], color='blue', linestyle = '--')
plt.vlines(beta1[25], fNL1.min(), fNL1[25], color='blue', linestyle = '--')
plt.xlim((0.0, 0.1))
plt.ylim((fNL1.min(), 0.0))
plt.show()
