import numpy as np
import numpy.matlib as mm

def get_dom_eval_and_evec(A):
    evals,evecs = np.linalg.eigh(A)
    k = np.argmax(evals)
    lam = evals[k] # dominant eigenvalue
    u = abs(evecs.T[k]) # dominant eigenvector
    return lam,u

def rank_by_evec(A):
    _,u = get_dom_eval_and_evec(A)
    rank = np.argsort(-u)
    return rank

def distributed_rank_by_evec(A,id_sets):
    # key function for indep management
    # assumes (mabye) patches same size
    # TODO: cahnge this so teh loop over id sets is in manage nodes
    KK = len(id_sets)
    N = len(A)
    ranks = np.zeros((int(N/KK),KK),dtype=int)
    for t,ids in enumerate(id_sets):        
        ranks[:,t] = rank_by_evec((A[ids].T)[ids]) + t*int(N/KK)
        #  t * int(N/KK) back-converts to the right id in the global matrix
            
    ranks = ranks.flatten()
    return ranks

def create_and_mask(N,ids):
    mask = np.zeros(N,dtype=bool)
    mask[ids] = True
    mask = mm.repmat(mask,N,1)          
    return np.logical_and(mask,mask.T)

def create_or_mask(N,ids):
    mask = np.zeros(N,dtype=bool)
    mask[ids] = True
    mask = mm.repmat(mask,N,1)          
    return np.logical_or(mask,mask.T)

def get_node_ranks(A,rank_opts):
    
    if rank_opts['type']=='random': # select node uniformly at random
        ranks = np.random.permutation(len(A))
        
    if rank_opts['type']=='evec': # select node by eigenvector centrality
        _,u = get_dom_eval_and_evec(A)
        ranks = np.argsort(-u)
            
    if rank_opts['type']=='distributed_evec': # select node by distributed eigenvector centralities
        ranks = distributed_rank_by_evec(A,rank_opts['id_sets'])
        
    if rank_opts['type']=='custom': # allow input of any function with any arguments
        ranks = rank_opts['function'](A,rank_opts['arguments'])
        
    return ranks 

def decrease_nodes_weights(A,node_ids,p=.5):
    N = len(A) ## DT check
    mask = create_or_mask(N,node_ids)# the weights of these edges will decrease
    return A - (mask*A)*p 

def decrease_top_nodes_weights(A,k,rank_opts,p=0.5):
    ranks = get_node_ranks(A,rank_opts)
    AA  = decrease_nodes_weights(A,ranks[:k],p)
    return AA

def manage_nodes(A,Ks,rank_opts,manage_opts):
    
    #compute ranks just once
    if manage_opts['online']==False:
        ranks = get_node_ranks(A,rank_opts)
        # OR swap so this retuns a manager-column matrix with ranks in descending order
        # so add for loop here that encodes info logic of rankings (pooled or not)

        # later down... separate for loop with budget etc and the function that
        # decreases weight would go through each manager iteratively and they choose
        # which to manage -- intersect teh manager patches w/ full ranks (subset managed)
        #print(ranks[:4])
        lams = np.zeros(len(Ks))
        for t,k in enumerate(Ks):
            AA  = decrease_nodes_weights(A,node_ids=ranks[:k],p=manage_opts['p'])
            lams[t],_ = get_dom_eval_and_evec(AA)
            
    if manage_opts['online']==True:
        AA = A.copy()
        lams = np.ones(len(Ks))
        for t,k in enumerate(Ks):
            lams[t],_ = get_dom_eval_and_evec(AA)
            AA = decrease_top_nodes_weights(AA,Ks[1]-Ks[0],rank_opts,p=manage_opts['p'])
    
    return AA,lams
