import numpy as np
import matplotlib.pyplot as plt
from copy import *
%matplotlib inline

#downloading the old faithful data
!wget http://www-958.ibm.com/software/analytics/manyeyes/datasets/old-faithful-duration-min-by-interva/versions/1.txt
!head 1.txt

import pandas
d = pandas.read_csv('1.txt',delimiter='\t')

d = d.dropna()
plt.plot(X);
plt.xlabel('Date');
plt.ylabel('Time Interval Duration Preplay Height Prediction');
plt.savefig("plot3d.png")

def normald(X, mu, sigma):
    """ normald:
       X contains samples, one per row, N x D. 
       mu is mean vector, D x 1.
       sigma is covariance matrix, D x D.  """
    D = X.shape[1]
    detSigma = sigma if D == 1 else np.linalg.det(sigma)
    if detSigma == 0:
        print('mu is\n',mu)
        print('Sigma is\n',sigma)
        raise np.linalg.LinAlgError('normald(): Singular covariance matrix')
    sigmaI = 1.0/sigma if D == 1 else np.linalg.inv(sigma)
    normConstant = 1.0 / np.sqrt((2*np.pi)**D * detSigma)
    diffv = X - mu.T # change column vector mu to be row vector
    return normConstant * np.exp(-0.5 * np.sum(np.dot(diffv, sigmaI) * diffv, axis=1))[:,np.newaxis]
    
#import pdb #python debugging package
#%pdb
def mixtureOfGaussians(X,K,nIterations=10,verbose=True):
    # X is N x D matrix of data samples, each being D-dimensional
    # Mu is K x D matrix of K mean vectors, each one D-dimensional
    # Sigma is K x D x D array of K covariance matrices, each one D x D
    # Pi is K x 1 vector of weightings for each Gaussian
    # Gamma is K x N array of responsibilities of Gaussian k for sample n
    
    N,D = X.shape
    randomSampleIndices = np.random.choice(range(N),K,replace=False)
    Mu = X[randomSampleIndices,:]
    Sigma = np.tile(np.eye(D),K).T.reshape((K,D,D))
    Pi = np.tile(1.0/K, K).reshape((-1,1))
    
    LL = []
    for i in range(nIterations):
        # Expectation Step
        gaussians = np.array([normald(X,mu.reshape((-1,1)),sigma) for mu,sigma in zip(Mu,Sigma)])
        gaussians = gaussians.squeeze()
        Gamma = (Pi * gaussians) / np.sum(Pi * gaussians, axis=0)
        Gamma[np.isnan(Gamma)] = 0 # to take care of entries where above denominator is zero
        #pdb.set_trace()
        # Maximization Step
        Nk = np.sum(Gamma,axis=1).reshape((-1,1))
        Mu = np.dot(Gamma,X) / Nk
        diffsk = X[:,np.newaxis,:] - Mu
        for k in range(K):
            Sigma[k,:,:] = np.dot(Gamma[k,:] * diffsk[:,k,:].T, diffsk[:,k,:]) / Nk[k]
        Pi = Nk / np.sum(Nk)
        Pi = Pi.reshape((-1,1))
        LLi = np.sum(np.log(np.sum(Pi * gaussians, axis=0)))
        LL.append(LLi)
        if verbose:
            for k in range(K):
                print('Gaussian',k)
                print('Mu\n',Mu[k,:])
                print('Sigma\n',Sigma[k,:,:])
                print('Pi\n',Pi[k])
            
    return Mu, Sigma, Pi, Gamma, np.array(LL)
    
m,s,p,g,ll = mixtureOfGaussians(X,3,nIterations=30,verbose=False) #K=3

plt.plot(kp,llp)
plt.xlabel('Number of K');
plt.ylabel('Likelihood');
plt.savefig("likelihood vs k.png")
##############PCA##################
means = np.mean(X,axis=0)
Xn = X - means
Xt = X.T
Xc = np.cov(Xt)
U,s,V = np.linalg.svd(Xc)
V = V.T[:,[0,1]]
proj = np.dot(Xn,V)
Xn

def drawline(v,means,len,color,label):
  p1 = means - v*len/2
  p2 = means + v*len/2
  plt.plot([p1[0],p2[0]],[p1[1],p2[1]],label=label,color=color,linewidth=2)


def plotOriginalAndTransformed(data,V):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(data[:,0],data[:,1],'.')
    means = np.mean(data,axis=0)
    drawline(100*V[:,0],means,8,"red","First")
    drawline(100*V[:,1],means,8,"green","Second")
    leg = plt.legend()
    plt.axis('equal')
    plt.gca().set_aspect('equal')


    plt.subplot(1,2,2)    
    proj = np.dot(data - means, V)
    plt.plot(proj[:,0],proj[:,1],'.')
    plt.axis('equal')
    plt.gca().set_aspect('equal')
    plt.xlabel("First")
    plt.ylabel("Second")
    plt.title("Projected to First and Second Singular Vectors");
plotOriginalAndTransformed(X,V)

####################Sammon Mapping###########################
from IPython.html.widgets import interact
def objective(X,proj,theta,s):
    N = X.shape[0]
    P = proj(X,theta)
    dX = np.array([X[i,:] - X[j,:] for i in range(N-1) for j in range(i+1,N)])
    dP = np.array([P[i,:] - P[j,:] for i in range(N-1) for j in range(i+1,N)])
    return 1/N * np.sum( (diffToDist(dX)/s - diffToDist(dP))**2)

def diffToDist(dX):
    return np.sqrt(np.sum(dX*dX, axis=1))

def proj(X,theta):
    return np.dot(X,theta)

def grad(X,proj,theta,s):
    P = proj(X,theta)
    N = X.shape[0]
    dX = np.array([X[i,:] - X[j,:] for i in range(N-1) for j in range(i+1,N)])
    dP = np.array([P[i,:] - P[j,:] for i in range(N-1) for j in range(i+1,N)])
    return 1/N * np.dot(dX.T, ((diffToDist(dX)/s - diffToDist(dP)).reshape((-1,1)) * -2 * dP)) #if linear

def gradient(X,proj,theta,s):
    N = X.shape[0]
    P = proj(X,theta)
    dX = np.array([X[i,:] - X[j,:] for i in range(N-1) for j in range(i+1,N)])
    dP = np.array([P[i,:] - P[j,:] for i in range(N-1) for j in range(i+1,N)])
    distX = diffToDist(dX)
    distP = diffToDist(dP)
    return -1/N * np.dot((((distX/s - distP) / distP).reshape((-1,1)) * dX).T, dP)
s = 0.5 * np.sqrt(np.max(np.var(X,axis=0)))
u,svalues,v = np.linalg.svd(X)
V = v.T
theta = V[:,:2]

thetas = [theta]
nIterations = 3
vals = []
for i in range(nIterations):
    theta = theta - 0.01 * gradient(X,proj,theta,s)
    v = objective(X,proj,theta,s)
    thetas.append(theta.copy())
    vals.append(v)


mn = 1.5*np.min(X)
mx = 1.5*np.max(X)

strings = [i for i in range(X.shape[0])]

@interact(i=(0,nIterations-1,1))
def plotIteration(i):
    #plt.cla()
    plt.figure(figsize=(8,10))
    theta = thetas[i]
    val = vals[i]
    P = proj(X,theta)
    plt.axis([mn,mx,mn,mx])
    for i in range(X.shape[0]):
        plt.text(X[i,0],X[i,1],strings[i],color='black',size=15) 
    for i in range(P.shape[0]):
        plt.text(P[i,0],P[i,1],strings[i],color='red',size=15) 
    plt.title('2D data, Originals in black. Objective = ' + str(val))
