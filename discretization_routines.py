'''
	this file contains functions for various discretization routines
	such as points sampling, constructing ridgelet and dual ridgelet transforms
'''


import numpy as np
import sympy as sym
import auxilary_functions as af




'''
	generate training/test sets by uniformly sampling (num_tr/num_ts) points
	from the domain (dom_x) and evaluating (f) at the sampled points
'''
def sample_data(f, dom_x, num_tr, num_ts=None):

	# define symbolic varibles
	x, y = sym.symbols('x, y')

	# 1d case
	if len(np.array(dom_x).shape) == 1:
		# construct the training set
		x_tr = np.linspace(dom_x[0], dom_x[1], num_tr)
		f_tr = sym.lambdify(x, f)(x_tr)
		# construct the test set
		if num_ts:
			x_ts = np.linspace(dom_x[0], dom_x[1], num_ts)
			f_ts = sym.lambdify(x, f)(x_ts)
		else:
			x_ts = f_ts = None

	# 2d case
	elif len(np.array(dom_x).shape) == 2:
		# construct the training set
		xx = np.linspace(dom_x[0][0], dom_x[0][1], num_tr[0])
		yy = np.linspace(dom_x[1][0], dom_x[1][1], num_tr[1])
		X_tr, Y_tr = np.meshgrid(xx, yy, indexing='ij')
		x_tr = np.array([X_tr.flatten(), Y_tr.flatten()])
		f_tr = sym.lambdify([x,y], f)(*x_tr)
		# construct the test set
		if num_ts:
			X_ts, Y_ts = np.meshgrid(	np.linspace(dom_x[0][0], dom_x[0][1], num_ts[0]), \
										np.linspace(dom_x[1][0], dom_x[1][1], num_ts[1]), \
										indexing='ij')
			x_ts = np.array([X_ts.flatten(), Y_ts.flatten()])
			f_ts = sym.lambdify([x,y], f)(*x_ts)
		else:
			x_ts = f_ts = None

	# other cases
	else:
		print('\nerror: dimensionality of the spatial domain is not 1 or 2')
		x_tr = f_tr = x_ts = f_ts = None

	return x_tr, f_tr, x_ts, f_ts




'''
	uniformly sample (num_points) from the (d)-dimensional sphere

	d = 1:
		uniformly sample the interval [-pi,pi)

	d = 2:
		sample the shpere with a golden spiral method
		(plot) displays the sampling process
'''
def sample_points_sphere(num_points, d, plot=False):

	# 1d case
	if d == 1:
		s = np.linspace(-np.pi, np.pi, num_points, endpoint=False)

	# 2d case
	elif d == 2:
		# generate indices bsed on the number of point
		ind = np.arange(0, num_points) + .5
		# compute angles phi and psi
		phi = np.pi * (1 + np.sqrt(5)) * ind % (2*np.pi)
		psi = np.arccos(1 - 2 * ind / num_points)
		s = [phi, psi]
		# plot the points
		if plot:
			af.plot_scatter_3d(np.sin(phi)*np.sin(psi), np.cos(phi)*np.sin(psi), np.cos(psi), \
				cmap='winter', title='points sampled from the sphere')

	return s




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
	(x, s, r) are arrays of points, weights, and biases
	functions (f) and (tau) are symbolic
'''
def ridgelet_transform(f, x, s, r, tau):

	# 1d case
	if len(np.array(x).shape) == 1:
		# calculate step size for x
		step_size_x = step_size(x)
		# compute the argument of tau
		tau_arg = x.reshape((-1,1)) * np.sin(s) + np.cos(s)
		tau_arg = tau_arg.reshape(tau_arg.shape + (1,)) * r

	# 2d case
	elif len(np.array(x).shape) == 2:
		# calculate step sizes for x,y
		step_size_x = np.outer(step_size(np.unique(x[0,:])), step_size(np.unique(x[1,:])))
		# compute the argument of tau
		tau_arg = np.matmul(x.T, np.array([np.sin(s[0]), np.cos(s[0])]))
		tau_arg = tau_arg * np.sin(s[1]) + np.cos(s[1])
		tau_arg = tau_arg.reshape(tau_arg.shape + (1,)) * r

	# evaluate the integrand
	integrand = f.reshape(f.shape + (1,1)) * tau(tau_arg)
	# compute the ridgelet transform
	Rf = (step_size_x.reshape((-1,1,1)) * integrand).sum(axis=0)

	return Rf




'''
	compute the dual ridgelet transform of (g) with respect to (sigma)
	(x, s, r) are arrays of points, weights, and biases
	function (g) is discrete, function (tau) is symbolic
'''
def dual_ridgelet_transform(Rf, x, s, r, sigma):

	# find dimensionality of the problem
	d = len(np.array(s).shape)

	# 1d case
	if d == 1:
		# calculate step sizes for r,phi,psi
		step_size_sr = 2 * np.pi / s.size * np.amax(r) / np.size(r)
		# compute the argument of sigma
		sigma_arg = x.reshape((-1,1)) * np.sin(s) + np.cos(s)

	# 2d case
	elif d == 2:
		# calculate step sizes for r,phi,psi
		step_size_sr = 4 * np.pi / s[0].size * np.amax(r) / np.size(r)
		# compute the argument of sigma
		sigma_arg = np.matmul(x.T, np.array([np.sin(s[0]), np.cos(s[0])]))
		sigma_arg = sigma_arg * np.sin(s[1]) + np.cos(s[1])

	# evaluate the integrand
	integrand = Rf.reshape((1,) + Rf.shape) * \
				sigma(sigma_arg).reshape(sigma_arg.shape + (1,)) * r**(d+1)
	# compute the dual ridgelet transform
	RRf = (step_size_sr * integrand).sum(axis=(-2,-1))

	return RRf




'''
	collapse the ridgelet transform (Rf) along the radial direction

	in 1d case (plot) displays the 3d-contour plot of (Rf) and the collapsed ridgelet transform
	in 2d case (plot) displays the 3d-scatter plot the collapsed ridgelet transform
'''
def collapse_ridgelet(Rf, s, r, plot=False):

	# calculate step size for r
	step_size_r = np.amax(r) / np.size(r)

	# 1d case
	if len(np.array(s).shape) == 1:
		# compute the collapsed ridgelet transform
		CRf = (Rf * r**2 * step_size_r).sum(axis=-1)
		# plot the collapsed ridgelet transform
		if plot:
			af.display_plots([\
			('contour_3d', Rf, s, r, [], ['a','b','Rf'], 'ridgelet transform'),\
			('approx', s, CRf, [], [], 'collapsed ridgelet transform')])

	# 2d case
	elif len(np.array(s).shape) == 2:
		# compute the collapsed ridgelet transform
		CRf = (Rf * r**3 * step_size_r).sum(axis=-1)
		# plot the collapsed ridgelet transform
		if plot:
			af.plot_scatter_3d(s[0], s[1], CRf, \
				cmap=af.generate_colormap(0, 0, 1, 1, blending=.8, blend_color=(1,1,0)), \
				colorbar=True, title='collapsed ridgelet transform')

	return CRf



