import numpy as np

def simulate_SIS(A,SIS_opts):    
    N = len(A)
    X = np.zeros((SIS_opts['T'],N))
    x = SIS_opts['x0'].copy()
    X[0] = x

    E = np.array(np.random.rand(SIS_opts['T'],N)<SIS_opts['eta'],dtype=int)# external excitation
    H = np.array(np.random.rand(SIS_opts['T'],N)<SIS_opts['beta'],dtype=int)# health recoveries
    
    for t in range(SIS_opts['T']-1):
        I =  np.array(np.random.rand(N,N)< A* SIS_opts['gamma'],dtype=int)# stochastic interactions
        xx = np.max(np.dot(I,np.diag(x)),1)        
        x = (1-x)*np.max(np.array([xx,E[t]]), 0) + x*(1-H[t])
        X[t+1] = x
    return X    

def get_p_curve(A,SIS_opts,gammas,delay):
    ps = zeros(len(gammas))
    for i,gamma in enumerate(gammas):
        print(i/len(gammas))
        SIS_opts['gamma'] = gamma
        X = simulate_SIS(A,SIS_opts)
        ps[i] = np.mean(X[delay:])
    return ps

def get_p_curves(A,SIS_opts,gammas,delay):
    ps = np.zeros((len(SIS_opts['etas']),len(gammas)))
    for i,eta in enumerate(SIS_opts['etas']):    
        print(i/len(SIS_opts['etas']))
        SIS_opts['eta'] = eta
        ps[i] = get_p_curve(A,SIS_opts,gammas,delay)
    return ps

def make_raster_plots(ax,Xs):
    cmaps = ['Oranges','Blues','Greys']
    for t,X in enumerate(Xs): 
        ax[t].imshow(X.T,interpolation='none',cmap=cmaps[t])
        ax[t].xaxis.tick_bottom()
        ax[t].set_xlabel('time, $t$')
    ax[0].set_ylabel('patch id, $i$')
    return 
