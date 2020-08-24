import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse as sp

def ER_GNP(n,p): # Erdos-Renyi graph G(n,p)    
    A = np.random.rand(n,n) < p
    A = np.triu(A, k=1)
    return csr_matrix(A + A.T) #adjacency matrix

# A = ER_GNP(1000,.5) 
# A = uniform_weight_edges(A)
# plt.imshow(A,cmap='cool')
# plt.colorbar();

def make_SBM(Ns,PI):
    K = np.shape(PI)[0]     
    n = np.zeros(K+1,dtype=int)
    n[1:] = np.cumsum(Ns)
    ids = []
    for k in range(K):
        start = 0;
        ids.append(np.arange(n[k],n[k+1]))

    one_hot = np.zeros((n[-1],K),dtype = bool)
    for k in range(K):
        one_hot[ids[k],k] = True;

    A = np.random.rand(n[-1],n[-1]) < np.dot(np.dot(one_hot,PI),one_hot.T)
  
    A = np.triu(A,k=1)
    
    n = np.zeros(K+1,dtype=int)
    n[1:] = np.cumsum(Ns)
    id_sets = []
    for k in range(K):
        start = 0;
        id_sets.append(list(np.arange(n[k],n[k+1])))

    return  (A + A.T),id_sets

def uniform_weight_edges(A,mmax): # weight the edges
    N = np.shape(A)[0]
    w = np.random.rand(N,N)*mmax
    w = np.triu(w, k=1) + np.triu(w, k=1).T    
    
    
    #A = A.type(float)

    row,col,val = sp.find(A)
    A[row,col] = val * np.random.rand(len(val))*mmax

    
    return A * w
