import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import wishart, multivariate_normal, bernoulli, multinomial
import os


class mvt2mixture:
    
    def __init__(self, thetas = {'mean1' : None, 'mean2' : None, \
                               'Sigma1' : None, 'Sigma2' : None, \
                               'nu1': None, 'nu2': None}, seed = None, gaussian = False):
        """
        Multivariate 2-component Student-t mixture random generator. 
        Direct random sampling via using the Student-t representation as a continous scale mixture distr.   
        -------
        Input:
        -------
        thetas: Component-wise parameters; note that Sigma1,2 are the scale matrices of the 
                Wishart priors of the precision matrices of the Student-t's.
        gaussian: boolean, generate from Gaussian mixture if True, otherwise from Student-t    
        seed: set seed for rng.
        """
        self.thetas = thetas ; self.seed = seed ; self.gaussian = gaussian
        if self.seed is not None:
            np.random.seed(seed)
    
    def draw(self, n_samples = 100, k = 2, p = .5): 
        """
        Random number generator:
        Input:
        -------
        n_samples: Number of realizations to generate
        k:         Number of features (Dimension of the t-distr.)
        p:         Success probability Bernoulli(p) p.m.f. 
        """
        self.n_samples = n_samples ; self.k = k; self.p = p ; 
        m = 2                # number of mixture components
        assert (len(self.thetas['mean1']) == k) & (self.thetas['Sigma1'].shape[0] == k), 'Number of dimensions does not match k!'

        if self.gaussian:
            cov1, cov2 = self.thetas['Sigma1'], self.thetas['Sigma2']  
        else:    
            cov1 = wishart.rvs(df = self.thetas['nu1'], scale = self.thetas['Sigma1'], size=1)
            cov2 = wishart.rvs(df = self.thetas['nu2'], scale = self.thetas['Sigma2'], size=1)

        self.var1 = self.thetas['nu1']/(self.thetas['nu1']-2)*cov1       # variance covariance matrix of first Student-t component
        self.var2 = self.thetas['nu2']/(self.thetas['nu2']-2)*cov2
        self.phi_is = bernoulli.rvs(p = self.p, size = self.n_samples)          # m=2
        Phi = np.tile(self.phi_is, self.k).reshape(self.k,self.n_samples).T              # repeat phi vector to match with random matrix
        rn1 = np.random.multivariate_normal(self.thetas['mean1'], cov1, self.n_samples)
        rn2 = np.random.multivariate_normal(self.thetas['mean2'], cov2, self.n_samples)
        self.sum1 = np.multiply(Phi, rn1)
        self.sum2 = np.multiply(1-Phi, rn2)
        self.x_draws = np.add(self.sum1,self.sum2)
        return self.phi_is, self.x_draws

    def show2D(self, save_plot=False, legend_on = True, **kwargs):
        """
        Make scatter plot for first two dimensions of the random draws
        """
        x_comp1,y_comp1 = self.sum1[:,0], self.sum1[:,1]
        x_comp2,y_comp2 = self.sum2[:,0], self.sum2[:,1]
        fig = plt.figure() ; 
        la = plt.scatter(x_comp1, y_comp1, c="blue", **kwargs)
        lb = plt.scatter(x_comp2, y_comp2, c="orange", **kwargs)
        lc = plt.scatter([self.thetas['mean1'][0], self.thetas['mean2'][0]], 
                         [self.thetas['mean1'][1],self.thetas['mean2'][1]], c="black", s=6**2, alpha=.5)
        #plt.title("Draws from 2-component \nmultivariate Student-t mixture \n(first two dimensions shown)")
        plt.xlabel(r'$x_{1}$') ; plt.ylabel(r'$x_{2}$')
        if legend_on:
            plt.legend((la, lb), ('Outlier', 'Inlier'),
                            scatterpoints=1, loc='lower right', ncol=3, fontsize=8)
        plt.show() ;
        if save_plot:
            fig.savefig('mixturePlot2D.jpg')
            print("Saved to:", os.getcwd())

    def show3D(self, save_plot=False, legend_on = True, **kwargs):
        """
        Make scatter plot for first three dimensions of the random draws
        """
        fig = plt.figure() ; ax = Axes3D(fig)
        x_comp1,y_comp1, z_comp1 = self.sum1[:,0], self.sum1[:,1], self.sum1[:,2]
        x_comp2,y_comp2, z_comp2 = self.sum2[:,0], self.sum2[:,1], self.sum2[:,2]
        la = ax.scatter(x_comp1, y_comp1, z_comp1, c="blue", **kwargs) 
        lb = ax.scatter(x_comp2, y_comp2, z_comp2, c="orange", **kwargs)  
        lc = ax.scatter([self.thetas['mean1'][0], self.thetas['mean2'][0]], 
                     [self.thetas['mean1'][1],self.thetas['mean2'][1]], 
                     [self.thetas['mean1'][2],self.thetas['mean2'][2]], c="black", s=6**2, alpha=.2)

        #plt.title("Draws from 2-component \nmultivariate mixture \n(first three dimensions shown)")
        ax.set_xlabel(r'$x_{1}$') ; ax.set_ylabel(r'$x_{2}$') ;ax.set_zlabel(r'$x_{3}$')
        if legend_on:
            ax.legend((la, lb), ('Outlier', 'Inlier'),
                        scatterpoints=1, loc='lower left', ncol=3, fontsize=8)    
        plt.show();
        if save_plot:
            fig.savefig('mixturePlot3D.jpg')
            print("Saved to:", os.getcwd())
