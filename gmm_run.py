
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.model_selection import train_test_split
import os, pickle
import numpy as np
from numpy import log, sum, exp, prod
from numpy.linalg import det
from numpy.random import beta, binomial, dirichlet, uniform, gamma, seed, multinomial, gumbel, rand, multivariate_normal
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

N = 10**2       # sample size
K = 3           # number of mixture components
D = 2          # dimensions / number of features     

mvt = gmm.mvt_tmix(seed=12)

# X and true cluster assignements:
#-----------------------------------
X, latent_true = mvt.draw(K = D, N = N, m = K, gaussian = True)    

mvt.plot(plot_type='2D')

# Set starting values for parameters:
#----------------------------------------
#seed(12)
MCsim = 1000         # MC iterations
beta0 = 0
alpha0 = 1
nu_0 = D-1 + 2                    # constraint D-1
var_m0 = np.zeros((D,D)) ; np.fill_diagonal(var_m0, 1)
m0 = multivariate_normal(np.zeros((D)), var_m0, size=1)    # prior mean of mu

# Initializations:
####################
W = np.empty((D,D, K, MCsim))       
nu = betas = Ns = log_Lambda = np.empty((K,MCsim))            # posterior dof of W and beta_k's
m_mean = np.empty((D,K,MCsim))                            # posterior means of mu
rho = rho_norm = np.empty((N,K, MCsim))
w_scales = np.zeros((D,D)) ; np.fill_diagonal(w_scales, 0.5)
W_init = wishart.rvs(df = D-1+10, scale = w_scales, size=K)        # random initialization
x_mean = np.zeros((K,D))


# Set iteration:
#---------------
it = 0

for k in range(K): W[:,:,k,it] = W_init[k,:,:]

rho_norm[:,:,it] = rho[:,:,it] = np.full((N,K),1/K)         # initialize matrix
#rho_norm[:,:,it]

Ns[:,it] = rho_norm[:,:,it].sum(axis=0)                 # (10.51)
betas[:,it] = beta0 + Ns[:,it] 
nu[:,it] = nu_0 + Ns[:,it] + 1

n = 1
k = 2

X.shape

#Nks = np.tile(1/Ns[k,it],(N,D))
#rn = np.tile(rho_norm[:,k,it],(D,1)).T
#np.multiply(rn - X, Nks).sum(axis=0)

for k in range(K):
    Nks = np.tile(1/Ns[k,it],(N,D))
    rn = np.tile(rho_norm[:,k,it],(D,1)).T
    x_mean[k,:] = np.multiply(rn - X, Nks).sum(axis=0)
    m_mean[:,k,it] = (beta0 * m0 + Ns[k,it] * x_mean[k,:])/betas[k,it]

x_mean

xm = np.tile(x_mean[k,:],(N,1))
XX = (X - xm).T.dot(X - xm)                   # X*X' , X needs to be transposed 


#for k in range(K):
#   for n in range(N):
#     x_mean[k,:] += rho_norm[n,k,it] * X[n,]
#   x_mean[k,:] = x_mean[k,:]/Ns[k,it]  

Ns[k,it]


for k in range(K):
    log_Lambda_k=0
    for i in range(1,D): log_Lambda_k += digamma((nu[k,it]+1-i)/2) + D*log(2) + log(det(W[:,:,k,it]))
    log_Lambda[k,it] = log_Lambda_k

log_Lambda[:,it]

rho_norm
X.shape
X[0,:]

# Check matrix multipl here next!!!
k=1
n = 20

a = np.array([0, 1, 2])
np.tile(a, (2, 1)).T 


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
x_mean_k = np.multiply(r_k, X.sum(axis=0)).sum(axis=0)/N_ks[k]     # (10.52)
x_mean_k.shape

x_centered = (X[n,:] - x_mean_k).reshape(-1,1)

x_centered.shape
x_centered.T.shape

np.matmul(x_centered, x_centered.T).shape

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





