from functools import reduce
from itertools import chain, zip_longest
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

def distributed_rank_by_evec(A,info_id_sets):
    # A - adjacency matrix
    # info_id_sets - iterable of length k, where k managers, each element defines
    #                the node ids that manager k knows about
    # return:
    # 

    ranks = list()
    for k,ids in enumerate(info_id_sets):        
        ranks.append(rank_by_evec((A[ids].T)[ids]))
        # N = nrow(A)
        # KK = num managers 
        #  k * int(N/KK) back-converts to the right id in the global matrix for even size community
    
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

def ranks_to_ids(ranks, info_id_sets):
    # ranks - iterable of indices of ranks
    # info_id_sets - iterable of length k, where k managers, each element defines
    #                the node ids that manager k knows abuot
    #
    # return:
    # 
    #
    return [ids[rs] for ids, rs in zip(info_id_sets, ranks)]


def get_ranked_nodes(A,rank_opts):
    
    if rank_opts['type']=='random': # select node uniformly at random
        ranks = np.random.permutation(len(A))
        
    if rank_opts['type']=='evec': # select node by eigenvector centrality
        info_ids = [np.arange(len(A))]
        ranks = distributed_rank_by_evec(A, info_ids)
        ranks = ranks_to_ids(ranks, info_ids)
        ranks = np.array(list(chain.from_iterable(zip_longest(*ranks))))
            
    if rank_opts['type']=='distributed_evec': # select node by distributed eigenvector centralities
        info_ids = rank_opts['info_id_sets']
        ranks = distributed_rank_by_evec(A, info_ids)
        ranks = ranks_to_ids(ranks, info_ids)
        ranks = np.array(list(chain.from_iterable(zip_longest(*ranks))))
        
    if rank_opts['type']=='custom': # allow input of any function with any arguments
        ranks = rank_opts['function'](A,rank_opts['arguments'])
        
    return ranks

def decrease_nodes_weights(A,node_ids,p=.5):
    ## TODO add management mask
    N = len(A) ## DT check
    mask = create_or_mask(N,node_ids) # the weights of these edges will decrease
    return A - (mask*A)*p 

def decrease_top_nodes_weights(A,k,rank_opts,p=0.5):
    ranks = get_ranked_nodes(A,rank_opts)
    AA  = decrease_nodes_weights(A,ranks[:k],p)
    return AA

def manage_nodes(A, rank_opts,manage_opts):
    # management of a network by k managers, potentially cooperative
    #
    # A is the adjacency matrix
    # rank_opts['random'] - select node uniformly at random
    # rank_opts['evec'] - select node by eigenvector centrality
    # rank_opts['distributed_evec'] - select node by distributed eigenvector centralities
    # rank_opts['custom'] - allow input of any function with any arguments
    # rank_opts['info_id_sets'] -  iterable of length k, each element defines
    #                the information that manager k knows 
    # rank_opts['budget'] -  iterable of length k, with each element an integer defining maximum budget
    # rank_opts['manage_id_sets'] -  iterable of length k, with each element an integer defining maximum budget
    #
    # manage_opts['online'] - if true, adaptively compute management during selection, otherwise
    #                         compute management from inital state of network
    #
    # return AA final adjacency matrix after management, 
    #        lams vector of dominant eigenvalues during management
    # 
    budgets = rank_opts['budget']
    B = int(np.sum(budgets))
    Ks = np.arange(0, np.sum(budgets), len(budgets), dtype='int')
    #compute ranks just once
    if manage_opts['online']==False:
        ranks = get_ranked_nodes(A,rank_opts)
        # OR swap so this retuns a manager-element list with ranks in descending order
        # within each list
        # so add for loop here that encodes info logic of rankings (pooled or not)

        # later down... separate for loop with budget etc and the function that
        # decreases weight would go through each manager iteratively and they choose
        # which to manage -- intersect teh manager patches w/ full ranks (subset managed)
        #print(ranks[:4])


        lams = np.zeros(len(Ks)) # Ks is jumps by K= num managers
        for t,k in enumerate(Ks):
            AA  = decrease_nodes_weights(A,node_ids=ranks[:k],p=manage_opts['p'])
            lams[t],_ = get_dom_eval_and_evec(AA)
            
    if manage_opts['online']==True:
        # assumes
        AA = A.copy()
        lams = np.ones(len(Ks))
        for t,k in enumerate(Ks):
            lams[t],_ = get_dom_eval_and_evec(AA)
            AA = decrease_top_nodes_weights(AA,Ks[1]-Ks[0],rank_opts,p=manage_opts['p'])
    
    return AA,lams
