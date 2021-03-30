'''
    this file contains functions related to the construction of GSN
'''


import numpy as np
import auxiliary_functions as af



'''
    construct the dictionary (D) from the set of angles (s), training points (x),
    and activation (sigma)

    returned dictionary is weighted by the values of collapsed ridgelet transform (CRf)
'''
def construct_dictionary(x, pts, sigma, D_threshold=1e-6):

    # construct the dictionary
    D = sigma(np.matmul(x, pts[:,:-1].T) + pts[:,-1])
    # threshold and normalize
    norms = np.linalg.norm(D, axis=0)
    if D_threshold:
        dict_ind = np.where(norms > D_threshold)[0]
        D = D[:,dict_ind] / norms[dict_ind]
    else:
        dict_ind = np.where(norms > 1e-15)[0]
        D[:,dict_ind] /= norms[dict_ind]
    return D, dict_ind



'''
    construct greedy approximation of vector (f_) by elements of (D_)

    dictionary (D_) first thresholded to remove the atoms
    with 2-norm smaller than threshold * max_norm(D)

    the algorithm is run for either (max_iter) iterations
    or until approximation accuracy (tol) is reached

    (plot) displays the approximation process
'''
def orthogonal_greedy_algorithm(pts, sigma, x_tr, f_tr, val_data=None, \
                                max_iter=100, tol=1e-16, verbose=False):

    # generate dictionary
    D, dict_ind = construct_dictionary(x_tr, pts, sigma)
    D_tr = D.copy()
    N, M = D.shape

    # check if validation data is provided
    if val_data is not None:
        x_vl, f_vl = val_data
        f_vl /= np.linalg.norm(f_vl)
        D_vl, _ = construct_dictionary(x_vl, pts[dict_ind], sigma, None)
        validation = True
    else:
        validation = False

    # normalize target element
    f = (f_tr / np.linalg.norm(f_tr)).reshape((N,1))
    f_tr = f.ravel()
    print('\nOGA: space dimensionality: {:d}, dictionary size: {:d}, validation data {:s} provided'\
        .format(N, M, 'is' if validation else 'is not'))

    # preallocate variables
    ind = np.zeros(max_iter, dtype=int) - 1
    E = np.zeros((N,max_iter))
    err = np.zeros((2,max_iter))

    # construct m-term approximation for f
    for m in range(min(max_iter, M)):

        # compute the projection remainder
        rem = f - np.matmul(f.T, D) * D
        # select the next atom
        ind[m] = np.argmin(np.linalg.norm(rem, axis=0))
        E[:,m] = D[:,ind[m]]
        # update the element
        f = rem[:,ind[m]].reshape((N,1))
        err[0,m] = np.linalg.norm(f)

        # compute validation error
        if validation:
            coef = np.linalg.lstsq(D_tr[:,ind[:m+1]], f_tr, rcond=None)[0]
            err[1,m] = np.linalg.norm(f_vl - np.matmul(D_vl[:,ind[:m+1]], coef))

        # display the approximation error
        if verbose and ((m+1) % 10 == 0):
            print('  iteration {:3d}: approximation error {:.2e} / {:.2e}, index {:d}'\
                .format(m+1, err[0,m], err[1,m], ind[m]))

        # check if required accuracy is reached
        if np.linalg.norm(f) <= tol:
            break

        # orthogonalize the dictionary
        D -= E[:,m].reshape((N,1)) * np.matmul(E[:,m].reshape((1,N)), D)
        D_norms = np.linalg.norm(D, axis=0)
        D[:,np.nonzero(D_norms)[0]] /= D_norms[D_norms > 0]

    # number of performed iterations
    iters = (ind >= 0).sum()

    return dict_ind[ind[:iters]], err[:,:iters]



'''
    select the value for (num_nodes) and construct nodes with the weights (A,B,C)

    (s) is the array of points on d-sphere in form of angles
    (ind) is the set of indices of points (s) selected by greedy algorithm
    (x,f) is the training data
    (sigma) is the activation function

    (nodes) is either an integer, in which case (num_nodes = nodes) or an error sequence
    from the greedy selection, in which case (num_nodes) is selected using (threshold) value
'''
def construct_nodes(s, ind, f, x, sigma, nodes=None, threshold=5e-3, plot=False):

    # select the number of nodes
    if isinstance(nodes, np.ndarray):
        # if validation error is provided then select
        # the number of nodes which minimizes the validation error
        if nodes[1,0] > 0:
            num_nodes = np.argmin(nodes[1,:]) + 1
        # otherwise select the number of nodes which provides the decay of the training error
        else:
            num_nodes = int(np.argwhere(nodes[0,:-1] - nodes[0,1:] > threshold)[-1]) + 2
    elif isinstance(nodes, int):
        num_nodes = min(nodes, len(ind))
    else:
        print('error: \'nodes\' parameter should be either an integer or an error sequence\n')
        return

    # extract weights and biases
    A, B = np.hsplit(s[ind[:num_nodes]], [-1])
    # compute output weights
    C = np.linalg.lstsq(sigma(np.matmul(A,x.T) + B).T, f, rcond=None)[0].reshape((num_nodes,-1))

    # report the outcome
    print('constructed {:d} nodes\n'.format(num_nodes))

    # plot the greedy selection process and constructed nodes
    if isinstance(nodes, np.ndarray):
        af.plot_nodes(nodes, num_nodes, \
            labels=['training','validation','nodes'], \
            title='approximation by the orthogonal greedy algorithm', \
            ticks_num=[21,11], display=plot)

    return A, B, C, num_nodes



