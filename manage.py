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

def rank_subset_by_evec(A, id_set):
    # A - adjacency matrix
    # id_set - set of indices defineing the node ids in the submatrix
    # return:
    # 
    # vector of ranked ids for the nodes in id_set
    return np.array(id_set[rank_by_evec((A[id_set].T)[id_set])], dtype='int')

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

def get_info_to_rank(rank_opts):
    # returns
    #   method to rank nodes this is a function that
    #   takes adj mat A as an argument and returns a iterable of integer-vectors
    #   that rank all known nodes
    
    if rank_opts['type']=='random': # select node uniformly at random
        fn = lambda A: [np.random.permutation(len(A))]
        
    if rank_opts['type']=='evec': # select node by eigenvector centrality
        # id_set here passed as all nodes in A
        # returned in a list as supposed to return iterable of iterables
        fn = lambda A: [rank_subset_by_evec(A, id_set=np.arange(len(A)))]
            
    if rank_opts['type']=='distributed_evec': # select node by distributed eigenvector centralities
        info_ids = rank_opts['info_id_sets']
        fn = lambda A: map(lambda ids: rank_subset_by_evec(A, ids), info_ids)
        
    if rank_opts['type']=='custom': # allow input of any function with any arguments
        fn = lambda A: rank_opts['function'](A,rank_opts['arguments'])

    return fn

def get_info_to_control(info_to_rank, rank_opts):
    # returns
    #   method to return ranked control nodes this is a function that
    #   takes adj mat A as an argument and returns a iterable of integer-vectors
    #   that rank all known nodes
    
    fn = lambda A: (ir[np.isin(ir, ctl)] for ir, ctl in
            zip(info_to_rank(A), rank_opts['control_id_sets']))
    return fn


def flatten_by_turns(manager_vec):
    # return
    #
    # flattens over the managers in turn as would be done by
    # numpy.matrix().flatten() if the managers vecs in columns
    return np.array(list(chain.from_iterable(zip_longest(*manager_vec))))

def decrease_nodes_weights(A,node_ids,p=.5):
    ## TODO add management mask
    N = len(A) ## DT check
    mask = create_or_mask(N,node_ids) # the weights of these edges will decrease
    return A - (mask*A)*p 

def decrease_top_nodes_weights(A, k, ranker, p=0.5):
    ranks = ranker(A)
    global_rank = flatten_by_turns(ranks)
    AA  = decrease_nodes_weights(A, global_rank[:k], p)
    return AA

def manage_for_budget(ranks, budgets):
    # ranks - iterable of integer-vectors
    #   that rank controllable nodes
    # budgets - iterable of integers defining max budget
    #
    # return
    # 
    # iterable of integer-vectors that go up to the (max) budget
    return (ctl[:b] for ctl, b in zip(ranks, budgets))

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
    # rank_opts['control_id_sets'] -  iterable of length k, each element defines
    #                the nodes that manager k controls 
    # rank_opts['budget'] -  iterable of length k, with each element an integer defining maximum budget
    # rank_opts['manage_id_sets'] -  iterable of length k, with each element an integer defining maximum budget
    #
    # manage_opts['num_managers'] - num managers (optional) if not present set to len control_id_sets
    # manage_opts['online'] - if true, adaptively compute management during selection, otherwise
    #                         compute management from inital state of network
    #
    # return AA final adjacency matrix after management, 
    #        lams vector of dominant eigenvalues during management
    # 

    budgets = rank_opts['budget']
    B = int(np.sum(budgets))
    try:
        K = manage_opts['num_managers']
    except KeyError:
        K = len(budgets)
    Ks = np.arange(0, B, K, dtype='int')
    info_to_rank = get_info_to_rank(rank_opts)
    info_to_control = get_info_to_control(info_to_rank, rank_opts)

    if manage_opts['online']==False:
        # compute ranks just once
        # logic here is to get a manager-length iterable that ranks in descending order
        # the nodes that will be managed up to the max budget
        # we then 'flatten' those in a turn based way (all managers 1st choices occur
        # before any managers 2nd chioces)
        ranks = info_to_control(A)
        management_plans = manage_for_budget(ranks, budgets)

        global_rank = flatten_by_turns(management_plans)

        lams = np.zeros(len(Ks)) # Ks is jumps by K= num managers
        for t,k in enumerate(Ks):
            AA  = decrease_nodes_weights(A,node_ids=global_rank[:k],p=manage_opts['p'])
            lams[t],_ = get_dom_eval_and_evec(AA)
            
    if manage_opts['online']==True:
        # compute ranks after every round of manager choices
        # logic here is the same, ranks are computed for every value in Ks
        # this conflates the budget determination and the Ks steps --
        # could cause errors so should find a better way to do this
        AA = A.copy()
        lams = np.ones(len(Ks))
        for t,k in enumerate(Ks):
            lams[t],_ = get_dom_eval_and_evec(AA)
            AA = decrease_top_nodes_weights(AA, k=Ks[1]-Ks[0],
                    ranker=info_to_control, p=manage_opts['p'])
    
    return AA,lams
