
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.model_selection import train_test_split
import os, pickle, time
import numpy as np
from numpy import log, sum, exp, prod
from numpy.linalg import det
from numpy.random import beta, binomial, dirichlet, uniform, gamma, seed, multinomial, gumbel, rand, normal, multivariate_normal
from scipy.stats import wishart #, norm, randint, bernoulli, beta, multinomial, gamma, dirichlet, uniform
from scipy.special import digamma
from imp import reload
from copy import deepcopy
#import seaborn as sns
import pandas as pd

os.chdir("C:\\Users\\Alexander\\Documents\\GitHub\\gauss_mix")
#os.chdir("C:\\Users\\Alexander\\Documents\\Python_stuff\\gauss_mix")   # sony

from gaussmix.utils import gmm_utils as gmm

os.getcwd()

#os.listdir(path='.')    # list files in current directory

reload(gmm)

#seed(12)
N = 10**3       # sample size
K = 4           # number of mixture components
D = 2          # dimensions / number of features     

mvt = gmm.mvt_tmix()

# X and true cluster assignements:
#-----------------------------------
X, latent_true = mvt.draw(K = D, N = N, m = K, gaussian = True)    

#mvt.plot(plot_type='2D')

# Set starting values for parameters:
#----------------------------------------
#seed(12)
MCsim = 10         # MC iterations

####################
# Declare:
####################

beta0 = 2.
alpha0 = 1.2
nu_0 = D-1 + 2                    # constraint D-1
W = np.empty((D,D, K, MCsim))       
m0 = np.empty((D,MCsim))
nu = np.empty((K,MCsim))            # posterior dof of W and beta_k's
betas = np.empty((K,MCsim))            
Ns = np.empty((K,MCsim))            # posterior dof of W and beta_k's
log_Lambda = np.empty((K,MCsim))            
alpha = np.empty((K,MCsim))            # posterior dof of W and beta_k's
log_pi = np.empty((K,MCsim))            # posterior dof of W and beta_k's
m_mean = np.empty((D,K,MCsim))                            # posterior means of mu
rho = np.empty((N,K, MCsim))
log_rho = np.empty((N,K, MCsim))
rho_norm = np.empty((N,K, MCsim))
S = np.empty((D,D, K, MCsim))
W0_inv = np.eye(D)*1
x_mean = np.zeros((K,D))

#------------------
# Initializations
#------------------
seed(42)

it = 0
var_m0 = np.zeros((D,D)) ; np.fill_diagonal(var_m0, 1)
m0[:,it] = multivariate_normal(np.zeros((D)), var_m0, size=1)    # prior mean of mu

# Initilaze W cov. matrix
w_scales = np.zeros((D,D)) ; np.fill_diagonal(w_scales, 0.5)
W_init = wishart.rvs(df = D-1+10, scale = w_scales, size=K)        # random initialization
for k in range(K): 
    W[:,:,k,it] = W_init[k,:,:]
#rho_norm[:,:,it] = np.full((N,K),1/K)         # initialize matrix
#rho[:,:,it] = np.full((N,K),1/K)         
alp = gamma(shape=4,size=K)
rho_norm[:,:,it] = dirichlet(alpha=alp, size=N)     # normalized responsibilities
rho[:,:,it] = rho_norm[:,:,it]

#betas[:,it] = gamma(shape=4,size=K)
#nu[:,it] = [nu_0]*K
Ns[:,it] = rho_norm[:,:,it].sum(axis=0)                 # (10.51)
betas[:,it] = beta0 + Ns[:,it] 
nu[:,it] = nu_0 + Ns[:,it] + 1                    # degrees of freedom
log_Lambda[:,it] = gamma(shape=4,size=K)
alpha[:,it] = uniform(size=K)
log_pi[:,it] = log(uniform(size=K))
m_mean[:,:,it] = normal(size=D*K).reshape(D,K)

#Nks = np.tile(1/Ns[k,it],(N,D))
#rn = np.tile(rho_norm[:,k,it],(D,1)).T
#np.multiply(rn*X, Nks).sum(axis=0)

ns = iter(range(N))
ks = iter(range(K))
its = iter(range(MCsim))
#k = next(ks) ; print(k)
#n = next(ns); print(n)
#it = next(its) ; print(it)


for it in range(MCsim): 

    print('Iter. {}'.format(it))
    ###########
    # E-step:
    ###########
    for n in range(N): 
        for k in range(K):
            log_rho[n,k,it] = log_pi[k,it] + .5*log_Lambda[k,it] -D/(2*betas[k,it]) -.5*nu[k,it]*(X[n,:] - m_mean[:,k,it]).reshape(1,D).dot(W[:,:,k,it]).dot((X[n,:] - m_mean[:,k,it]).reshape(D,1))
            #print('n: {} k: {} value: {}'.format(n,k, log_rho[n,k,it]))

        #rho[n,:,it] = exp(log_rho[n,:,it])
        #rho_norm[n,:,it] = rho[n,:,it]/sum(rho[n,:,it])
        rho_norm[n,:,it] = gmm.exp_normalize(log_rho[n,:,it])

    Ns[:,it] = rho_norm[:,:,it].sum(axis=0)                 # (10.51)
    betas[:,it] = beta0 + Ns[:,it] 
    nu[:,it] = nu_0 + Ns[:,it] + 1

    ###########
    # M-step:
    ###########
    for k in range(K):
        alpha[k,it] = alpha0 + Ns[k,it]
        if Ns[k,it] == 0.: Ns[k,it] = 10**(-4)
        Nks = np.nan_to_num(np.tile(1/Ns[k,it],(N,D)))
        rn = np.tile(rho_norm[:,k,it],(D,1)).T
        x_mean[k,:] = np.multiply(rn*X, Nks).sum(axis=0)
        m_mean[:,k,it] = (beta0 * m0[:,it] + Ns[k,it] * x_mean[k,:])/betas[k,it]

        Sk = 0 ;
        for n in range(N): Sk += rho_norm[n,k,it] * (X[n,:] - x_mean[k,:]).reshape(D,1).dot((X[n,:] - x_mean[k,:]).reshape(1,D))/Ns[k,it]
        S[:,:,k,it] = Sk

        Wk_inv = W0_inv + Ns[k,it]*S[:,:,k,it] + beta0*Ns[k,it]*((x_mean[k,:] - m0[:,it]).reshape(D,1).dot((x_mean[k,:] - m0[:,it]).reshape(1,D)))/(beta0 + Ns[k,it])
        W[:,:,k,it] = np.linalg.inv(Wk_inv) 

        log_Lambda_k = []
        for i in range(1,D+1): log_Lambda_k.append(digamma((nu[k,it]+1-i)/2))
        try:
            deter = det(W[:,:,k,it])
        except Exception as ex:
            print(ex)
            deter = 10**(-5)  
        log_Lambda[k,it] = sum(log_Lambda_k) + D*log(2) + log(deter)


it = next(its) ; print(it)



#----------------------------------------------------------------------------------------------------
for k in range(K):

    print(k)
    
    alpha[k,it] = alpha0 + Ns[k,it]
    Nks = np.tile(1/Ns[k,it],(N,D))

    rn = np.tile(rho_norm[:,k,it],(D,1)).T
    x_mean[k,:] = np.multiply(rn*X, Nks).sum(axis=0)

    m_mean[:,k,it] = (beta0 * m0 + Ns[k,it] * x_mean[k,:])/betas[k,it]

    Sk = 0
    for n in range(N): 

        #log_rho[n,k,it] = log_pi[k,it] + 0.5*log_Lambda[k,it] -D/(2*betas[k,it]) -0.5*nu[k,it]*(X[5,1] - m_mean[:,k,it]).reshape(1,D).dot(W[:,:,k,it]).dot((X[5,1] - m_mean[:,k,it]).reshape(D,1))
       
        Sk += rho_norm[n,k,it] * (X[n,:] - x_mean[k,:]).reshape(D,1).dot((X[n,:] - x_mean[k,:]).reshape(1,D))/Ns[k,it]
    S[:,:,k,it] = Sk

    Wk_inv = W0_inv + Ns[k,it]*S[:,:,k,it] + beta0*Ns[k,it]*(x_mean[k,:] - m0).reshape(D,1).dot(x_mean[k,:] - m0)/(beta0 + Ns[k,it])
    W[:,:,k,it] = np.linalg.inv(Wk_inv) 

    log_Lambda_k = []
    for i in range(1,D+1): log_Lambda_k.append(digamma((nu[k,it]+1-i)/2))
    log_Lambda[k,it] = sum(log_Lambda_k) + D*log(2) + log(det(W[:,:,k,it]))
     

norm_const_rho = log_rho.sum(axis=1).shape


alpha_hat = alpha[:,it].sum()

for k in range(K): log_pi[k,it] = digamma(alpha[k,it]) - digamma(alpha_hat) 

nu[:,it]

log_pi[k,it] + 0.5*log_Lambda[k,it] -D/(2*betas[k,it]) -0.5*nu[k,it]*(X[5,1] - m_mean[:,k,it]).reshape(1,D).dot(W[:,:,k,it]).dot((X[5,1] - m_mean[:,k,it]).reshape(D,1))


def inner_prod(x, m=m_mean[:,k,it], w=W[:,:,k,it]):
     return (x - m).reshape(1,D).dot(w).dot((x - m).reshape(D,1))

inner_prod(X[1,:])
np.apply_along_axis(func1d= inner_prod, arr=X, axis=0)

m_mean[:,k,it]

m_til = np.tile(m_mean[:,k,it],(N,1))
X - m_til

#alphas = gamma(shape=2, size=K)               # Dirichlet hyperparameters -> concentration param.
#r = dirichlet(alpha = alphas, size = N)
#p_0 = np.array([1/K]*K)  
#theta_0 = beta(a = 1, b = 1, size = K*D).reshape(D,K)

N_ks = rho.sum(axis=0)                 # (10.51)
N_ks.shape

#pd.DataFrame(r).head()
#pd.DataFrame(r_k).head()

#r_k.shape
#X.sum(axis=0).shape


#----------
# Run EM:    
#----------
#logli, p_em, theta_em = bmm.mixture_EM(X = X, p_0 = p_0, theta_0 = theta_0, n_iter = 500, stopcrit = 10**(-3))


#----------------
# Plot loglike.:
#----------------
burn_in = 5

plt.plot(logli[burn_in:], 'b--')
plt.title("Convergence check")
plt.xlabel('iterations')
plt.ylabel('loglikelihood')
plt.show()

# Compare with ground truth:
print(p_em)
print(p_true)

theta_em
theta_true

##################
# Gibbs sampler
##################

seed(12)

MC = 2000        # Monte Carlo runs
burn_in = 500    # discard those draws for burn-in

#K = 3
N, D = X.shape[0], X.shape[1]

p_draws = np.empty((MC,K))                                  # mixture weights draws
theta_draws = np.empty((MC,X.shape[1],K))                   # theta success rates 
latent_draws = np.empty((MC,N))                             # latent variable draws, Z

alphas = gamma(shape=2, size=K)               # shape parameters
p_0 = dirichlet(alpha = alphas, size = 1)[0]
#p_0 = np.array([1/K]*K)
theta_0 = beta(a = 1.3, b = 1.7, size = K*D).reshape(D,K)

p_draws[0,:], theta_draws[0,:,:] = p_0, theta_0 

gammas, deltas = gamma(shape=1.5, size=K), rand(K)     # uniform random draws   

#----------------------------
# Sample from full cond.:
#----------------------------
for i in range(1,MC):   
    if i%500 == 0:   
        print("Iter.",i)
        
    latent_draws[i,:], p_draws[i,:], theta_draws[i,:,:] = bmm.gibbs_pass(p_draws[i-1,:], 
                                                      theta_draws[i-1,:,:], X, 
                                                      alphas = alphas, 
                                                      hyper_para = {'gammas': gammas, 'deltas': deltas})
print("Finished!")
#-----------------------------------------------------------------------------------------------------------

latent_draws.shape
p_draws.shape
theta_draws.shape

# Bayes estimates:
#---------------------
theta_bayes = np.mean(theta_draws[burn_in:,:, :],axis=0)
theta_bayes#.shape
theta_true

p_bayes = np.mean(p_draws[burn_in:,],axis=0)

print(p_bayes)
print(p_true)

latent_bayes = np.around(np.mean(latent_draws[burn_in:,:],axis=0))
latent_bayes.shape

# Compute performance metrics - 
# compare MAP estimates of cluster assignements with ground truth labels:
#-----------------------------------------------------------------------------
# Note: label switching issue when checking for simple accuracy!!

print(accuracy_score(latent_true, latent_bayes))    

confusion_matrix(latent_true, latent_bayes)


# Plot MCMC results:
#---------------------------------
plt.figure(figsize=(10, 20))
for j in range(p_draws.shape[1]):
    plt.subplot(5,2,j+1)
    plt.plot(p_draws[burn_in:, j]);
    plt.title('Trace for $p_{%d}$' % j)

plt.figure(figsize=(10, 20))
for j in range(theta_draws.shape[1]):
    plt.subplot(25,2,j+1)
    plt.plot(theta_draws[100:,j, 0]);
    plt.title('Trace for $theta_{%d}$' % j)


# Calculate ACF and PACF upto 50 lags
#acf_50 = acf(df.value, nlags=50)
#pacf_50 = pacf(df.value, nlags=50)

plt.figure(figsize=(10, 20))
for j in range(p_draws.shape[1]):
    fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
    plot_acf(p_draws[burn_in:, j], lags=100, ax=axes[0])
    plt.xlabel("Lags"); 
    plt.title('ACF for $p_{%d}$' % j)
    plot_pacf(p_draws[burn_in:, j], lags=100, ax=axes[1])
    #plt.xlabel("Lags"); #plt.ylabel("PACF")
    plt.title('PACF for $p_{%d}$' % j)


#################### REAL DATA ############################
###########################################################

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline


np.random.seed(42)

X_digits, y_digits = load_digits(return_X_y=True)
data = scale(X_digits)

X_digits.shape

n_samples, n_features = data.shape
n_digits = len(np.unique(y_digits))
labels = y_digits

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

pca = PCA(n_components=n_digits)
kmeans = KMeans(n_clusters=n_digits,n_init=1)
predictor = Pipeline([('pca', pca), ('kmeans', kmeans)])

predict = predictor.fit(data).predict(data)
predict

stats = bmm.clusters_stats(predict, labels)
purity = bmm.clusters_purity(stats)

print("Plotting an extract of the 10 clusters, overall purity: %f" % purity)

bmm.plot_clusters(predict, labels, stats, data)

############################################################
# MNIST
############################################################

image_size = 28                  # width and length
no_of_different_labels = 10 
image_pixels = image_size**2
image_pixels

# Read data in:
#os.getcwd()
data_path = "C:\\Users\\Alexander\\Documents\\Github\\bmm_mix\\bernmix\\data\\"

train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")

train_data.shape

test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 

test_data.shape

test_data[:10]

test_data[test_data==255]
test_data.shape

#train_imgs = ((train_data[:, 1:]/255) > .5)*1.
train_imgs = np.asfarray(train_data[:, 1:])/255  # we avoid 0 values as inputs
test_imgs = np.asfarray(test_data[:, 1:])/255

X = train_imgs.copy()
X = test_imgs.copy()

fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01  # we avoid 0 values as inputs which are capable of preventing weight updates
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])


lr = np.arange(10)

for label in range(10):
    one_hot = (lr==label).astype(np.int)
    print("label: ", label, " in one-hot representation: ", one_hot)


lr = np.arange(no_of_different_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99


#------ Plot images -------------------------------
for i in range(10):
    img = train_imgs[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()
#------------------------------------------------------


# Save images for later:
with open(data_path+"mnist_all.pkl", "bw") as fh:
    data = (train_imgs, 
            test_imgs, 
            train_labels,
            test_labels,
            #train_labels_one_hot,
            #test_labels_one_hot
            )
    pickle.dump(data, fh)
    
# Load images:
with open(data_path+"mnist_all.pkl", "br") as fh:
    data = pickle.load(fh)


train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]

train_imgs[2].shape


# Log-sum trick:
x = np.arange(1,1000)

log(sum(exp(x)))

a = max(x) + log(sum(exp(x - max(x))));
a


################################################################





