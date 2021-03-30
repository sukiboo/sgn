'''
    this file contains functions for various discretization routines
    such as points sampling, constructing ridgelet and dual ridgelet transforms
'''

import numpy as np
import sympy as sym
import auxiliary_functions as af



'''
    generate training/test sets by uniformly sampling (num_tr/num_ts) points
    from the domain (dom_x) and evaluating (f) at the sampled points
'''
def sample_data(d, func, dom_x, num_x):

    # if num_x is a list
    if isinstance(num_x, int) and (num_x > 0):
        # sample the points randomly
        x = np.array(\
            [np.random.uniform(dom_x[k][0], dom_x[k][1], num_x) for k in range(d)]).T
        # sort by the first coordinate
        x = x[x[:,0].argsort()]

    # if num_x is an integer
    elif isinstance(num_x, list) and (len(num_x) == d):
        # construct the grid
        x_mesh = np.meshgrid(\
                *[np.linspace(dom_x[k][0], dom_x[k][1], num_x[k]) for k in range(d)],\
                indexing='ij')
        # form the array of points
        x = np.array(\
            [x_mesh[k].flatten() for k in range(d)]).T

    # other cases
    else:
        print('\nerror: num_x should be either a list of length d, or a positive integer')
        return None, None

    # evaluate the function
    f = func(*x.T)

    return x, f



'''
    uniformly sample (num_points) from the (d)-dimensional sphere

    d = 1:
        uniformly sample the interval [-pi,pi)

    d = 2:
        sample the shpere with a golden spiral method
        (plot) displays the sampling process
'''
def sample_points_sphere(d, num_points, plot=False):

    # 1d case, sample points deterministically
    if d == 1:
        phi = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        a = np.array(phi).reshape((-1,1))
        s = np.array([np.sin(phi), np.cos(phi)]).T

    # 2d case, sample points via 'golden spiral'
    elif d == 2:
        # generate indices based on the number of points
        ind = np.arange(0, num_points) + 2/(1 + np.sqrt(5))# + .5
        # compute angles phi and psi
        phi = np.pi * (1 + np.sqrt(5)) * ind % (2*np.pi)
        psi = np.arccos(1 - 2 * ind / num_points)
        a = np.array([phi, psi]).T
        s = np.array([np.sin(phi) * np.sin(psi), np.cos(phi) * np.sin(psi), np.cos(psi)]).T

    # other cases, sample points via 'normalized gaussians'
    else:
        a = []
        s = np.random.randn(num_points, d+1)
        s /= np.linalg.norm(s, axis=-1).reshape((-1,1))
        # sort by the first coordinate
        s = s[s[:,0].argsort()]

    # plot the points
    if d == 2:
        af.plot_scatter_3d(s[:,0], s[:,1], s[:,2], \
        cmap='winter', title='points sampled from the sphere', display=plot)

    return s, a



'''
    generate the array of step sizes for a sequence of points (x)
    step size is calculated as the half of lengths of the left and right intervals
    if (period) is given, step sizes for the endpoints are adjusted accordingly
'''
def step_size(x, period=None):

    # calculate distances between points
    dist = (x[1:] - x[:-1]) / 2
    # calculate step sizes
    step_size = np.concatenate(([dist[0]], dist[1:] + dist[:-1], [dist[-1]]))
    # adjust for the periodic case
    if period:
        step_size[[0,-1]] += (period + x[0] - x[-1]) / 2

    return step_size



'''
    compute the ridgelet transform of (f) with respect to (tau)
    (x, s) are arrays of points and angles
'''
def ridgelet_transform(f, x, s, r=np.linspace(0,10,50)[1:]):

    # define admissibility constant and function
    z = sym.symbols('z')
    d = x.shape[-1]
    K = -np.sqrt(2/np.pi) * (2*np.pi)**d
    tau = sym.lambdify(z, sym.diff(sym.exp(-z**2/2) / K, z, 4))

    # 1d case
    if d == 1:
        print('\ncomputing ridgelet transform on the training data...')
        # calculate step size for x
        step_size_x = step_size(x)
        # compute the argument of tau
        tau_arg = x * np.sin(s).ravel() + np.cos(s).ravel()
        tau_arg = tau_arg.reshape(tau_arg.shape + (1,)) * r

    # 2d case
    elif d == 2:
        print('\ncomputing ridgelet transform on the training data...')
        # calculate step sizes for x,y
        step_size_x = np.outer(step_size(np.unique(x[:,0])), step_size(np.unique(x[:,1])))
        # compute the argument of tau
        tau_arg = np.matmul(x, np.array([np.sin(s[:,0]), np.cos(s[:,0])]))
        tau_arg = tau_arg * np.sin(s[:,1]) + np.cos(s[:,1])
        tau_arg = tau_arg.reshape(tau_arg.shape + (1,)) * r

    # other cases
    else:
        print('\nskipping dictionary reduction due to large dimensionality...')
        return None

    # evaluate the integrand
    integrand = f.reshape(f.shape + (1,1)) * tau(tau_arg)
    # compute the ridgelet transform
    Rf = (step_size_x.reshape((-1,1,1)) * integrand).sum(axis=0)

    return Rf



'''
    collapse the ridgelet transform (Rf) along the radial direction

    in 1d case (plot) displays the 3d-contour plot of (Rf) and the collapsed ridgelet transform
    in 2d case (plot) displays the 3d-scatter plot of the collapsed ridgelet transform
'''
def collapse_ridgelet(Rf, s, plot=False):

    # calculate step size for r
    r = np.linspace(0,10,50)[1:]
    step_size_r = np.amax(r) / np.size(r)

    # 1d case
    if np.array(s).shape[-1] == 1:
        # compute the collapsed ridgelet transform
        CRf = (Rf * r**2 * step_size_r).sum(axis=-1)
        # plot the collapsed ridgelet transform
        if plot:
            af.display_plots([\
            ('contour_3d', Rf, s, r, [], ['a','b','Rf'], 'ridgelet transform'),\
            ('approx', s, CRf, [], [], 'collapsed ridgelet transform')], display=True)

    # 2d case
    elif np.array(s).shape[-1] == 2:
        # compute the collapsed ridgelet transform
        CRf = (Rf * r**3 * step_size_r).sum(axis=-1)
        # plot the collapsed ridgelet transform
        if plot:
            af.plot_scatter_3d(s[:,0], s[:,1], CRf,\
                cmap=af.generate_colormap(0, 0, 1, 1, blending=.8, blend_color=(1,1,0)),\
                title='collapsed ridgelet transform', colorbar=True, display=True)

    # other cases
    else:
        return None

    return CRf


