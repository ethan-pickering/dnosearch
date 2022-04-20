import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
import scipy.io as sio
from scipy.spatial import distance


def funmin(fun, jac, inputs, opt_method="l-bfgs-b", args=(), 
           kwargs_op=None, num_restarts=None, parallel_restarts=False, 
           n_jobs=10, init_method=None, n_keep=1, opt_all_pts=False):
    """Scipy-based minimizer allowing multiple parallel restarts.

    Parameters
    ----------
    fun : callable
        Objective function to be minimized.
    jac : callable
        Jacobian of the objective function.
    inputs : instance of Inputs
        To be used for domain definition.
    opt_method : str, optional 
        Type of solver. Should be one of "L-BFGS-B", "SLSQP" or "TNC".
    args : tuple, optional
        Extra arguments passed to fun and jac.
    kwargs_op : dict, optional
        A dictionary of solver options, as in scipy.optimize.minimize.
    num_restarts : int, optional
        Number of restarts for the optimizer. The number of initial
        guesses is 1+num_restarts. If None, min(100,10*d) is supplied, 
        with d the dimension of the input space.
    parallel_restarts : boolean, optional
        Whether or not to solve the optimization problems in parallel.
    n_jobs : int, optional
        Number of workers used by joblib for parallel computation.
    init_method : str, optional
        Sampling method for initial guesses. If None, the points are 
        selected by Latin-Hypercube Sampling.  If "sample_fun", we draw
        1000 uniformly sampled points and retain those with the smallest
        objective value. This approach can help avoid local minima, but 
        it is computationaly more expensive.

    Returns
    -------
    x_opt : array
        The solution array.  For multiple initial guesses, the solution
        array associated with the smallest objective value is returned.

    """
    opt_method = opt_method.lower()
    ndim = np.shape(inputs.domain)[0]
    assert(opt_method in ["l-bfgs-b", "slsqp", "tnc"])

    if kwargs_op is None:
        kwargs_op = dict(options={"disp":False})

    if num_restarts is None:
        num_restarts = min(100, 10*inputs.input_dim)

    n_guess = num_restarts # + 1

    if init_method is None:
        x0 = inputs.draw_samples(n_guess, "lhs")
        scores = fun(np.atleast_2d(x0), *args)  
        sorted_idxs = np.argsort(scores,axis = 0)
        org_scores = np.mean(scores[sorted_idxs[:min(len(scores), n_guess)], :])

    elif init_method == "lhs":
        #np.random.seed(1)
        #x0 = np.random.uniform(-1,1,(n_guess,2))*6
        # print('Loading')
        # d = sio.loadmat('/home/epickeri/git/optimization_testing_x0.mat')
        # x0 = d['x0']
        # x0 = x0.reshape(n_guess,2)
        x0 = inputs.draw_samples(n_guess, "lhs")
        #scores = fun(np.atleast_2d(x0), *args)
        x0 = np.atleast_2d(x0)
        scores = fun(x0, False)
        sorted_idxs = np.argsort(scores,axis = 0)
        scores = scores[sorted_idxs[0:n_guess]]
        scores = scores.reshape(n_guess,)
        init_guess_score = np.sum(scores[sorted_idxs[0:n_guess]])
        init_keep_score = np.sum(scores[sorted_idxs[0:n_keep]])
        x0 = x0[sorted_idxs[0:n_guess], :]

    elif init_method == "sample_fun":
        # x0 = inputs.draw_samples(n_guess, "uni")
        if ndim > 4:
            n_monte = 10**5
            n_end   = 10**4
        else:
            n_monte = 10**ndim
            n_end = 10**(ndim-1)
            
        x0 = inputs.draw_samples(n_monte, "lhs")

        # x0 = inputs.draw_samples(3,"grd")
        # x0 = np.ones((1,np.shape(X0)[1]))*0.5
        # scores = fun(np.atleast_2d(x0), *args)
        # scores
        
        x0 = np.atleast_2d(x0)
        scores = fun(x0, False)
        sorted_idxs = np.argsort(scores,axis = 0)
        scores = scores.reshape(n_monte,)
        # init_guess_score = np.sum(scores[sorted_idxs[0:n_guess]])
        # init_keep_score = np.sum(scores[sorted_idxs[0:n_keep]])
        # #init_guess_score = np.sum(scores[sorted_idxs[:min(len(scores), n_guess)], :])
        # #init_keep_score = np.sum(scores[sorted_idxs[:min(len(scores), n_keep)], :])
        # scores = scores[sorted_idxs[0:n_guess]]
        # x0 = x0[sorted_idxs[0in_guess], :]
        
        # New version where we impose the radius earlier
        sorted_scores = scores[sorted_idxs[0:n_end]]
        sorted_x0 = x0[sorted_idxs[0:n_end], :]
        
        x0_guess = np.zeros((n_guess,ndim))
        score_guess = np.zeros((n_guess,))
    
        x0_guess[0,:] = sorted_x0[0,:]
        score_guess[0] = sorted_scores[0]
    
        # Now we need to remove the optimal from consideration, and remove values within a radius of influence
        max_domain_distance = np.sqrt((inputs.domain[1][1]-inputs.domain[0][0])**2*ndim)
        r_val = 0.025*max_domain_distance
    
        for i in range(1,n_guess):
            # Now remove the optimal value
            sorted_x0 = np.delete(sorted_x0, 0, axis=0)
            sorted_scores = np.delete(sorted_scores, 0)
            distances = np.zeros((np.size(sorted_scores),))
            for j in range(0,np.size(sorted_scores)):
                distances[j] = distance.euclidean(x0_guess[i-1,:], sorted_x0[j,:])
            sorted_x0 = sorted_x0[distances > r_val,:]       
            sorted_scores = sorted_scores[distances > r_val]       
            x0_guess[i,:] = sorted_x0[0,:]
            score_guess[i] = sorted_scores[0]
            # I think I need to make a break here that lets you know we have run out of points
        scores = score_guess
        x0 = x0_guess
        init_guess_score = np.sum(score_guess)
        init_keep_score = np.sum(score_guess[0:n_keep])

        # scores = fun(np.atleast_2d(X0), *args)     
        # sorted_idxs = np.argsort(scores,axis = 0)
        # x0 = X0[sorted_idxs[0:n_guess], :]
        # org_scores = np.sum(scores[sorted_idxs[:min(len(scores), n_guess)], :])
        # print('Number of Guesses:')
        # print(n_guess)
        # print('Original Scores:')
        # print(org_scores)
    
   
        
    if parallel_restarts:
        res = Parallel(n_jobs=n_jobs, backend="loky")(
                       delayed(minimize)(fun, 
                                         np.atleast_2d(x0[i]), 
                                         args=args,
                                         method=opt_method, 
                                         jac=jac,
                                         bounds=inputs.domain,
                                         **kwargs_op)
                       for i in range(x0.shape[0]) )

    else:
    #    res = [ minimize(fun, 
    #                     np.atleast_2d(x0[i]), 
    #                     args=args, 
    #                     method=opt_method,
    #                     jac=jac, 
    #                     bounds=inputs.domain,
    #                     **kwargs_op)
    #            for i in range(x0.shape[0]) ]

    #idx = np.argmin([r.fun for r in res])
    #xopt = res[idx].x
        # res = [ minimize(fun, 
        #                  np.atleast_2d(x0[i]), 
        #                  args=args, 
        #                  method=opt_method,
        #                  jac=jac, 
        #                  bounds=np.array(inputs.domain),
        #                  options={'disp': None,'ftol': 0.01, 'iprint': 1, 'maxls': 20})
        #         for i in range(x0.shape[0]) ]
        if opt_all_pts:
             x0 = x0.reshape(1,ndim*n_guess)
             res = minimize(fun, 
                            np.atleast_2d(x0), 
                            args=args, 
                            method=opt_method,
                            jac=jac, 
                            bounds=np.array(inputs.domain*n_guess),
                            options={'disp': None,'ftol': 0.00000000001, 'iprint': 1, 'maxls': 50})
             #print('Score Improvement!')
             #print(res.fun/np.sum(init__scores))
             xopt = res.x
             xopt = xopt.reshape(n_guess,ndim) 
             vals = fun(xopt, False)
             #print(np.sum(vals)/np.sum(org_scores))
             x0 = x0.reshape(n_guess,ndim)

             
        else:
            x0 = x0.reshape(n_guess,1,ndim)
            res = [ minimize(fun, 
                          np.atleast_2d(x0[i]), 
                          args=args, 
                          method=opt_method,
                          jac=jac, 
                          bounds=np.array(inputs.domain),
                          options={'disp': None,'ftol': 0.001, 'iprint': 1, 'maxls': 50})
                for i in range(x0.shape[0]) ]
            vals = np.zeros((n_guess,))
            xopt = np.zeros((n_guess, ndim))
            for i in range(0,n_guess):
                vals[i] = float(res[i].fun)
                xopt[i,:] = res[i].x
            x0 = x0.reshape(n_guess,ndim)

            #new_scores = np.sum(vals)       
            #print('Score Improvement!')
            #print(new_scores/org_scores)

    # vals = np.zeros((n_guess,))
    # xopt = np.zeros((n_guess, ndim))
    # # for i in range(0,n_guess):
    # #     vals[i] = float(res[i].fun)
    # #     xopt[i,:] = res[i].x
    # # new_scores = np.mean(vals)       
    # # print('Score Improvement!')
    # # print(new_scores/org_scores)
    
    #sio.savemat('/home/epickeri/git/optimization_testing.mat', {'scores':scores, 'new_scores':vals, 'x0':x0, 'xopt':xopt})

    opt_guess_score = np.sum(vals) # First determine how much improvement through the optimization straight up
    print('Guess Score Improvement (Lower is better!)')
    print(opt_guess_score/init_guess_score)
    
    # # Append the optimized vals with the initial guesses
    # total_scores = np.append(vals,scores)
    # total_x = np.append(xopt, x0, axis=0)
    
    # # Round the inputs to the nearest tenth and find uniques solutions
    # # We will loop through the dimensions such that this is imposed in each dimensions
    # x_vals, x_inds = np.unique(np.round(total_x[:,0],1), return_index=True, axis = 0)
    # total_unique_x = total_x[x_inds,:]
    # total_unique_scores = total_scores[x_inds]
    # for i in range(1,ndim):
    #         x_vals, x_inds = np.unique(np.round(total_unique_x[:,i],1), return_index=True, axis = 0)
    #         total_unique_x = total_unique_x[x_inds,:]
    #         total_unique_scores = total_unique_scores[x_inds]
    
    # # Sort and keep the best scores/inputs
    # sorted_idxs = np.argsort(total_unique_scores,axis = 0) # Find sorted indices for the scores
    # sorted_scores = total_unique_scores[sorted_idxs] # Sort the scores
    # sorted_x = total_unique_x[sorted_idxs, :] # Sort the inputs
    # opt_keep_score = np.sum(sorted_scores[0:n_keep]) # Keep the scores related to the top n_keep unique values 
    # xopt = sorted_x[0:n_keep,:] # Keep the top n_keep values
    # print('Keep Score Improvement (Lower is better!)')
    # print(opt_keep_score/init_keep_score)
    # xopt = xopt.reshape(n_keep,ndim)
    




    # New method for imposing a radius of influence for each point.
    x_opt_keep = np.zeros((n_keep,ndim))
    score_opt_keep = np.zeros((n_keep,))
    
    # Append all scores
    total_scores = np.append(vals,scores)
    total_x = np.append(xopt, x0, axis=0)
    # Sort the scores
    sorted_idxs = np.argsort(total_scores,axis = 0) # Find sorted indices for the scores
    sorted_scores = total_scores[sorted_idxs] # Sort the scores
    sorted_x = total_x[sorted_idxs, :] # Sort the inputs
    
    # Keep the optimal
    x_opt_keep[0,:] = sorted_x[0,:]
    score_opt_keep[0] = sorted_scores[0]
    
    # Now we need to remove the optimal from consideration, and remove values within a radius of influence
    max_domain_distance = np.sqrt((inputs.domain[1][1]-inputs.domain[0][0])**2*ndim)
    r_val = 0.05*max_domain_distance
    
    for i in range(1,n_keep):
        # Now remove the optimal value
        sorted_x = np.delete(sorted_x, 0, axis=0)
        sorted_scores = np.delete(sorted_scores, 0)
        distances = np.zeros((np.size(sorted_scores),))
        for j in range(0,np.size(sorted_scores)):
            distances[j] = distance.euclidean(x_opt_keep[i-1,:], sorted_x[j,:])
        sorted_x = sorted_x[distances > r_val,:]       
        sorted_scores = sorted_scores[distances > r_val]       
        x_opt_keep[i,:] = sorted_x[0,:]
        score_opt_keep[i] = sorted_scores[0]
   
    opt_keep_score = np.sum(score_opt_keep)
    print('Keep Score Improvement (Lower is better!)')
    print(opt_keep_score/init_keep_score)


    #sio.savemat('/home/epickeri/git/optimization_testing.mat', {'scores':scores, 'new_scores':sorted_scores, 'x0':x0, 'xopt':x_opt_keep, 'total_x':total_x})
    
    
    
    
    #xopt_sort,inds = np.unique(xopt,axis = 0, return_index=True); xopt_sort[inds,:] = xopt_sort
    #no_left = n_guess - np.shape(xopt_sort)[0]
    #extra_not_opt = X0[sorted_idxs[n_guess-1:n_guess+no_left-1], :].reshape(no_left,np.shape(X0)[1])
    #xopt = np.append(xopt,extra_not_opt,axis=0)
    return x_opt_keep
