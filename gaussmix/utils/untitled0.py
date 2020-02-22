
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import wishart, multivariate_normal, norm, randint, bernoulli, beta, multinomial, gamma, dirichlet, uniform
import os

K = 10           # dimensions
N = 1*10**5
m = 3                 # components

seed = None
gaussian = False


class mvt_tmix:
    
    def __init__(self, seed = None):
         self.seed = seed
    
    def draw(self, K = 10, N = 1*10**5, m = 3, gaussian = False):
        
        if self.seed is not None:
            np.random.seed(self.seed)
     
        alphas = gamma.rvs(5, size=m)               # shape parameter
        #print(sum(alphas))                              # equivalent sample size
        self.p = dirichlet.rvs(alpha = alphas, size = 1)[0]
        self.phi_is = multinomial.rvs(1, self.p, size=N)       # draw from categorical p.m.f
        
        self.x_draws = np.zeros((N,K))
        self.hyper_loc, self.hyper_scale, self.thetas, self.var, self.covs, self.rdraws = dict(), dict(), dict(), tuple(), tuple(), tuple()
        
        for i in range(m):
        
              self.hyper_loc["mean"+str(i+1)] = norm.rvs(size = 1, loc = 0, scale = 5)
              self.hyper_scale["scale"+str(i+1)] = 1/gamma.rvs(5, size=1)
              
              self.thetas["mean"+str(i+1)] = norm.rvs(size = K, loc = self.hyper_loc["mean"+str(i+1)], 
                          scale = self.hyper_scale["scale"+str(i+1)])
              self.thetas["Sigma"+str(i+1)] = np.eye(K)*(1/gamma.rvs(5, size=K))
              self.thetas["nu"+str(i+1)] = randint.rvs(K+2, K+10, size=1)[0]
        
              if gaussian:
                 self.covs += (self.thetas['Sigma'+str(i+1)], )
              else:
                 self.covs += (wishart.rvs(df = self.thetas['nu'+str(i+1)], scale = self.thetas['Sigma'+str(i+1)], size=1),)
                 self.var += (self.thetas["nu"+str(i+1)]/(self.thetas["nu"+str(i+1)]-2)*self.covs[i],)       # variance covariance matrix of first Student-t component
              self.rdraws += (np.random.multivariate_normal(self.thetas["mean"+str(i+1)], self.covs[i], N),)
        
              self.Phi = np.tile(self.phi_is[:,i], K).reshape(K,N).T              # repeat phi vector to match with random matrix
              self.x_draws += np.multiply(self.Phi, self.rdraws[i])                
        return self.x_draws


    def plot(self, draws = None, save_plot=False, legend_on = True, plot_type = ['2D', '3D'], **kwargs):
        """
        Make scatter plot for first two dimensions of the random draws
        """
        if draws is not None:
            self.draws = draws
            
        #if plot_type[0] == '2D':
        x_comp1,y_comp1 = self.x_draws[:,0], self.x_draws[:,1]
        #x_comp2,y_comp2 = self.sum2[:,0], self.sum2[:,1]
        fig = plt.figure() ; 
        la = plt.scatter(x_comp1, y_comp1, c="blue", **kwargs)
        #lb = plt.scatter(x_comp2, y_comp2, c="orange", **kwargs)
        #lc = plt.scatter([self.thetas['mean1'][0], self.thetas['mean2'][0]], 
        #                 [self.thetas['mean1'][1],self.thetas['mean2'][1]], c="black", s=6**2, alpha=.5)
        #plt.title("Draws from 2-component \nmultivariate Student-t mixture \n(first two dimensions shown)")
        plt.xlabel(r'$x_{1}$') ; plt.ylabel(r'$x_{2}$')
            
        
        if plot_type[0] == '3D':
            fig = plt.figure() ; ax = Axes3D(fig)
            x_comp1, y_comp1, z_comp1 = self.x_draws[:,0], self.x_draws[:,1], self.x_draws[:,2]
            #x_comp2,y_comp2, z_comp2 = self.sum2[:,0], self.sum2[:,1], self.sum2[:,2]
            la = ax.scatter(x_comp1, y_comp1, z_comp1, c="blue", **kwargs) 
            #lb = ax.scatter(x_comp2, y_comp2, z_comp2, c="orange", **kwargs)  
            #lc = ax.scatter([self.thetas['mean1'][0], self.thetas['mean2'][0]], 
            #             [self.thetas['mean1'][1],self.thetas['mean2'][1]], 
            #             [self.thetas['mean1'][2],self.thetas['mean2'][2]], c="black", s=6**2, alpha=.2)
    
            #plt.title("Draws from 2-component \nmultivariate mixture \n(first three dimensions shown)")
            ax.set_xlabel(r'$x_{1}$') ; ax.set_ylabel(r'$x_{2}$') ;ax.set_zlabel(r'$x_{3}$')
            #if legend_on:
            #    ax.legend((la, lb), ('Outlier', 'Inlier'),
            #                scatterpoints=1, loc='lower left', ncol=3, fontsize=8)                            
        #if legend_on:
        #    plt.legend((la, lb), ('Outlier', 'Inlier'),
        #                    scatterpoints=1, loc='lower right', ncol=3, fontsize=8)
        plt.show() ;
        #if save_plot:
        #    fig.savefig('mixturePlot2D.jpg')
        #    print("Saved to:", os.getcwd())


mvt = mvt_tmix(seed = 12)

X = mvt.draw(K = 2, N = 1*10**4, m = 4, gaussian = True)

mvt.plot(plot_type='2D')


x_comp1,y_comp1 = self.x_draws[:,0], self.x_draws[:,1]
#x_comp2,y_comp2 = self.sum2[:,0], self.sum2[:,1]
fig = plt.figure() ; 
la = plt.scatter(x_comp1, y_comp1, c="blue", **kwargs)
#lb = plt.scatter(x_comp2, y_comp2, c="orange", **kwargs)
#lc = plt.scatter([self.thetas['mean1'][0], self.thetas['mean2'][0]], 
#                 [self.thetas['mean1'][1],self.thetas['mean2'][1]], c="black", s=6**2, alpha=.5)
#plt.title("Draws from 2-component \nmultivariate Student-t mixture \n(first two dimensions shown)")
plt.xlabel(r'$x_{1}$') ; plt.ylabel(r'$x_{2}$')




i=1
Phi = np.tile(phi_is[:,i], K).reshape(K,N).T
Phi
mix_sum += np.multiply(Phi, rdraws[i])

#phi_is = multinomial.rvs(1, p, size=N)       # draw from categorical p.m.f
#phi_is


phi_is[:,0]

Phi = np.tile(phi_is[:,0], K).reshape(K,N).T              # repeat phi vector to match with random matrix
Phi

sum1 = np.multiply(Phi, rdraws[0])
sum1


#_, max_phi_is = np.where(phi_is)

len(rdraws)

rdraws[0].shape

#uniform.rvs(2)


#assert (len(self.thetas['mean1']) == k) & (self.thetas['Sigma1'].shape[0] == k), 'Number of dimensions does not match k!'

self.phi_is = bernoulli.rvs(p = self.p, size = self.n_samples)          # m=2

Phi = np.tile(self.phi_is, self.k).reshape(self.k,self.n_samples).T              # repeat phi vector to match with random matrix

rn1 = np.random.multivariate_normal(self.thetas['mean1'], cov1, self.n_samples)
rn2 = np.random.multivariate_normal(self.thetas['mean2'], cov2, self.n_samples)

self.sum1 = np.multiply(Phi, rn1)
self.sum2 = np.multiply(1-Phi, rn2)
self.x_draws = np.add(self.sum1,self.sum2)
return self.phi_is, self.x_draws





