
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.model_selection import train_test_split
import os, pickle
import numpy as np
#from numpy import log, sum, exp, prod
#from numpy.linalg import det
#from numpy.random import beta, binomial, dirichlet, uniform, gamma, seed, multinomial, gumbel, rand, multivariate_normal
#from scipy.stats import wishart #, norm, randint, bernoulli, beta, multinomial, gamma, dirichlet, uniform
#from scipy.special import digamma
from imp import reload
from copy import deepcopy
#import seaborn as sns
import pandas as pd

import bayespy.plot as bpplt
from bayespy.nodes import Dirichlet, Categorical, Gaussian, Wishart, Mixture
from bayespy.inference import VB

os.chdir("C:\\Users\\Alexander\\Documents\\GitHub\\gauss_mix")
#os.chdir("C:\\Users\\Alexander\\Documents\\Python_stuff\\gauss_mix")   # sony

from gaussmix.utils import gmm_utils as gmm

os.getcwd()

#os.listdir(path='.')    # list files in current directory

reload(gmm)

#seed(12)

N = 10**2       # sample size
K = 3           # number of mixture components
D = 2          # dimensions / number of features     

mvt = gmm.mvt_tmix(seed=12)

# X and true cluster assignements:
#-----------------------------------
y, latent_true = mvt.draw(K = D, N = N, m = K, gaussian = True)    
y.shape
#mvt.plot(plot_type='2D')

bpplt.pyplot.plot(y[:,0], y[:,1], 'rx')
plt.show()

# Define nodes of the factor graph:
#------------------------------------
alpha = Dirichlet(1e-5*np.ones(K), name='alpha')
Z = Categorical(alpha, plates=(N,), name='z')

mu = Gaussian(np.zeros(D), 1e-5*np.identity(D), plates=(K,), name='mu')

Lambda = Wishart(D, 1e-5*np.identity(D), plates=(K,), name='Lambda')

Y = Mixture(Z, Gaussian, mu, Lambda, name='Y')

Z.initialize_from_random()

Q = VB(Y, mu, Lambda, Z, alpha)

Y.observe(y)

Q.update(repeat=1000)

bpplt.gaussian_mixture_2d(Y, alpha=alpha, scale=2)

Q.compute_lowerbound()
Y.random()

from sklearn.mixture import BayesianGaussianMixture

# DD
fin_gmm = BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_distribution",
        covariance_type='full', 
        weight_concentration_prior=1.2,
        n_components=10, reg_covar=0, init_params='random',
        max_iter=1500, mean_precision_prior=.8)

fitted = fin_gmm.fit(X)

z_max = fitted.predict(X)
z_max

post_z_X = fitted.predict_proba(X)
post_z_X

def prob_outlier(p):
    return 1-prod(1-p)

# Calculate prob. of belonging to any of the K clusters
# # if below p < 0.05 for example label as outlier:     
np.apply_along_axis(prob_outlier, 1, post_z_X)

# DP
inf_gmm = BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=2,
        n_components=10, reg_covar=0, init_params='random',
        max_iter=1500, mean_precision_prior=.8)

fitted = inf_gmm.fit(X)

z_max = fitted.predict(X)
z_max

post_z_X = fitted.predict_proba(X)
post_z_X.shape
