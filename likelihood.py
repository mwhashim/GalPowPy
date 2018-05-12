import numpy as np
import matplotlib.pylab as plt
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

from matplotlib import colors, ticker, cm

#__________ Plotting setup _______
plt.rc("font", size="15"); plt.rc("axes",labelsize="15")
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

# This will be usefull later, in order to calculate contours levels
def find_level(posterior,fraction) :
    tot = np.sum(posterior)
    max = np.max(posterior)
    min = np.min(posterior)
    ## initialize level to halfway between max and min
    level = 0.5*(max + min)
    ## initialize fraction for this level
    frac = np.sum( posterior[ posterior >= level ] )/tot
    ## initialize resolution = +/- smallest pixel as fraction of total
    res = np.min( posterior[ posterior >= level ] )/tot
    ## iterate until frac is within res of the input fraction
    while( abs(frac - fraction) > res ) :
        ## update max or min
        if( frac > fraction) :
            min = level
        else :
            max = level
        ## update level by bisecting
        level = 0.5*(max + min)
        ## update frac and res
        frac = np.sum( posterior[ posterior >= level ] )/tot
        res = np.min( posterior[ posterior >= level ] )/tot
    ## output the level and its actual fraction
    return level, frac


beta_bestfit = 0.05; fNL_bestfit = -56.3018242; fNL_bestfit1 = -97.1230472637

dbeta = beta_bestfit/30.
betaarray = np.arange(dbeta, 2. * beta_bestfit, dbeta)

dfNL = fNL_bestfit/60.
fNLarray = np.arange(dfNL, 2. * fNL_bestfit, dfNL)


dfNL1 = fNL_bestfit1/60.
fNLarray1 = np.arange(dfNL1, 2. * fNL_bestfit1, dfNL1)

X1, Y1 = np.meshgrid(fNLarray, betaarray)
X2, Y2 = np.meshgrid(fNLarray1, betaarray)

post_clstr = np.load('fNL_beta_post_clstr.npy')
post_nclstr = np.load('fNL_beta_post_noclstr.npy')

contour_level_0,real_fraction_1 = find_level(post_clstr,0.68/2.)
contour_level_1,real_fraction_1 = find_level(post_clstr,0.68)
contour_level_2,real_fraction_2 = find_level(post_clstr,0.95)
contour_level_3,real_fraction_3 = find_level(post_clstr,0.99)

contour_level_10,real_fraction_1 = find_level(post_nclstr,0.68/2.)
contour_level_11,real_fraction_1 = find_level(post_nclstr,0.68)
contour_level_12,real_fraction_2 = find_level(post_nclstr,0.95)
contour_level_13,real_fraction_3 = find_level(post_nclstr,0.99)

plt.figure()
CS2=plt.contourf(X2, Y2, post_nclstr, levels = [contour_level_13, contour_level_12, contour_level_11, contour_level_10], cmap=cm.cool)
CS1=plt.contourf(X2, Y2, post_clstr, levels = [contour_level_3, contour_level_2, contour_level_1, contour_level_0], cmap=cm.hot)

pc1 = CS1.collections[-1]; pc2 = CS2.collections[-1]; pc_list = [pc1, pc2]
proxy = [plt.Rectangle((0,0), 1, 1, fc = pc.get_facecolor()[0]) for pc in pc_list]
plt.legend(proxy, ["clustering", "homogenous"])

plt.xlabel(r'$\rm fNL$')
plt.ylabel(r'$\beta$')
plt.tight_layout()
plt.savefig('fNLbeta2dcountor.png')
plt.show()
