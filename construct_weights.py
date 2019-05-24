'''
	this file contains functions related to the construction of GSN
'''


import numpy as np
from sklearn.cluster import DBSCAN
import auxilary_functions as af




'''
	construct the dictionary (D) from the set of angles (s), training points (x),
	and activation (sigma)

	returned dictionary is weighted by the values of collapsed ridgelet transform (CRf)
'''
def construct_dictionary(CRf, x, s, sigma):

	# 1d case
	if len(np.array(s).shape) == 1:
		# compute the argument of sigma
		sigma_arg = x.reshape((-1,1)) * np.sin(s) + np.cos(s)

	# 2d case
	elif len(np.array(s).shape) == 2:
		# compute the argument of sigma
		sigma_arg = np.matmul(x.T, np.array([np.sin(s[0]), np.cos(s[0])]))
		sigma_arg = sigma_arg * np.sin(s[1]) + np.cos(s[1])

	# construct the dictionary
	D = CRf * sigma(sigma_arg)

	return D




'''
	construct greedy approximation of vector (f_) by elements of (D_)

	dictionary (D_) first thresholded to remove the atoms
	with 2-norm smaller than threshold * max_norm(D)

	the algorithm is run for either (max_iter) iterations
	or until approximation accuracy (tol) is reached

	(plot) displays the approximation process
'''
def orthogonal_greedy_algorithm(f_, D_, threshold=.01, max_iter=40, tol=1e-6, plot=False):

	# normalize element and the dictionary
	norms = np.linalg.norm(D_, axis=0)
	dict_ind = np.where(norms >= threshold * np.max(norms))[0]
	D = D_[:,dict_ind] / norms[dict_ind]
	N, M = D.shape
	f = (f_ / np.linalg.norm(f_)).reshape((N,1))
	print('\nspace dimensionality: {0}, dictionary size: {1}'.format(N, M))

	# initialize variables
	ind = np.zeros(max_iter, dtype=int) - 1
	E = np.zeros((N,max_iter))
	err = np.zeros(max_iter)

	# construct m-term approximation of f
	for m in range(min(max_iter, M)):
		# compute the projection remainder
		rem = f - np.matmul(f.T, D) * D
		# select the next atom
		ind[m] = np.argmin(np.linalg.norm(rem, axis=0))
		E[:,m] = D[:,ind[m]]
		# update the element
		f = rem[:,ind[m]].reshape((N,1))
		err[m] = np.linalg.norm(f)
		print('  iteration {:2d}: accuracy {:.2e}, index {:d}'.format(m+1, err[m], ind[m]))

		# check if required accuracy is reached
		if np.linalg.norm(f) <= tol:
			break

		# orthogonalize the dictionary
		D -= E[:,m].reshape((N,1)) * np.matmul(E[:,m].reshape((1,N)), D)
		D_norms = np.linalg.norm(D, axis=0)
		D[:,np.nonzero(D_norms)[0]] /= D_norms[D_norms > 0]

	# number of performed iterations
	iters = (ind >= 0).sum()

	# plot the greedy selection process
	if plot:
		af.plot_approximation(np.arange(iters)+1, err[:iters], \
			title='approximation by the orthogonal greedy algorithm', \
			ticks_num=[min(iters, 14), 10])

	return dict_ind[ind[:iters]], err[:iters]




'''
	select the value for (num_nodes) and construct nodes with the weights (A,B,C)

	(s) is the array of points on d-sphere in form of angles
	(ind) is the set of indices of points (s) selected by greedy algorithm
	(x,f) is the training data
	(sigma) is the activation function

	(nodes) is either an integer, in which case (num_nodes = nodes) or an error sequence
	from the greedy selection, in which case (num_nodes) is selected using (threshold) value
'''
def construct_nodes(s, ind, f, x, sigma, nodes=None, threshold=5e-3):

	# select the number of nodes
	if isinstance(nodes, np.ndarray):
		num_nodes = int(np.argwhere(nodes[:-1] - nodes[1:] > threshold)[-1]) + 2
	elif isinstance(nodes, int):
		num_nodes = min(nodes, len(ind))
	else:
		print('error: \'nodes\' parameter should be either an integer or an error sequence\n')
		return

	# construct nodes
	print('\nconstructing {:d} nodes'.format(num_nodes))

	# 1d case
	if len(np.array(s).shape) == 1:
		W = s[ind[:num_nodes]]
		# compute weights and biases
		A = np.sin(W).reshape((num_nodes,1))
		B = np.cos(W).reshape((num_nodes,1))
		# compute output weights
		C = np.linalg.lstsq(sigma(A*x + B).T, f, rcond=None)[0].reshape((num_nodes,1))

	# 2d case
	elif len(np.array(s).shape) == 2:
		phi, psi = s[0][ind[:num_nodes]], s[1][ind[:num_nodes]]
		# compute weights and biases
		A = np.array([np.sin(phi) * np.sin(psi), np.cos(phi) * np.sin(psi)]).T
		B = np.cos(psi).reshape((num_nodes,1))
		# compute output weights
		X, Y = np.meshgrid(x[0], x[1], indexing='ij')
		C = np.linalg.lstsq(sigma(np.matmul(A, x) + B).T, f, rcond=None)[0].reshape((num_nodes,1))

	return A, B, C, num_nodes



