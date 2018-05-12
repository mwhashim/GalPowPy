import numpy as np
import matplotlib.pylab as plt

H0 = 70.3

plt.figure()

beta = [0.0, 0.05]; fNL = [0.0,-100, 100]
k, PowSpec_g_lcdm = np.load('Pg_%s_%s_fNL_%s_z_%s.npy' %(0.0, 0.0, 0.0, 0.0))

#k, Dm = np.load('Dm_%s_%s_z_%s.npy' %(0.0, 0.0, 0.0)); print Dm
for j in range(len(fNL)):
    k, PowSpec_g = np.load('Pg_%s_%s_fNL_%s_z_%s.npy' %(0.0, beta[0], fNL[j], 0.0))
    plt.semilogx(k, PowSpec_g/PowSpec_g_lcdm, '--', label = r'$\rm fNL: $%s' %fNL[j])

#for i in range(len(beta)):
#    k, PowSpec_g = np.load('Pg_%s_%s_fNL_%s_z_%s.npy' %(0.08, beta[i], fNL[0], 0.0))
#    #plt.loglog(k, PowSpec_g, '--', label = r'$\rm fNL: $%s' %fNL[j])
#    plt.semilogx(k, PowSpec_g/PowSpec_g_lcdm, '--', label = r'$\beta: $%s' %beta[i])
#
#for i in range(len(fNL)):
#    k, PowSpec_g = np.load('Pg_%s_%s_fNL_%s_z_%s.npy' %(0.08, beta[1], fNL[i], 0.0))
#    #plt.loglog(k, PowSpec_g, '--', label = r'$\rm fNL: $%s' %fNL[j])
#    plt.semilogx(k, PowSpec_g/PowSpec_g_lcdm, '--', label = r'$\beta: 0.05, fNL : $%s' %fNL[i])

plt.axvline(1./H0, linewidth=1, color='black', linestyle='--')
plt.legend()
plt.show()

