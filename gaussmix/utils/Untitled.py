import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import wishart, multivariate_normal, norm, randint, bernoulli, beta, multinomial, gamma, dirichlet, uniform
import os

K = 5           # dimensions
N = 1*10**5
m = 3                 # components

seed = None
gaussian = False

alphas = gamma.rvs(5, size=m)               # shape parameter
print(sum(alphas))                              # equivalent sample size
p = dirichlet.rvs(alpha = alphas, size = 1)[0]

thetas, var, covs = dict(), tuple(), tuple()
for i in range(m):

      thetas["mean"+str(i+1)] = norm.rvs(size = K, loc = 1, scale = .5)
      thetas["Sigma"+str(i+1)] = np.eye(K)*(1/gamma.rvs(5, size=K))
      thetas["nu"+str(i+1)] = randint.rvs(K+2, K+10, size=1)[0]

      if gaussian:
        covs += (thetas['Sigma'+str(i+1)], )
      else:
        covs += (wishart.rvs(df = thetas['nu'+str(i+1)], scale = thetas['Sigma'+str(i+1)], size=1),)
        var += (thetas["nu"+str(i+1)]/(thetas["nu"+str(i+1)]-2)*covs[i],)       # variance covariance matrix of first Student-t component

thetas

phi_is = multinomial.rvs(1, p, size=N)       # draw from categorical p.m.f

rows, cols = np.where(phi_is)

