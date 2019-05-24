
import numpy as np
import sympy as sym
import discretization_routines as dr
import auxilary_functions as af
import construct_weights as cw
import network_functions as nf

# set print options
np.set_printoptions(precision=3, linewidth=115, \
	suppress=True, formatter={'float':'{: 0.3f}'.format})
# define symbolic variables
x, y = sym.symbols('x, y')




'''
	this program performs numerical experiments described in the paper
	"Greedy Shallow Networks: A New Approach for Constructing and Training Neural Networks"

	to run the code, first select an example to define problem setting
	exmaple 1,2,3,4 are presented in the paper

	example 1: f = sym.sin(2*sym.pi*x) * sym.exp(-x**2)
	example 2: f = sym.cos(sym.exp(3*x)) / (1 + 25*x**2)
	example 3: f = sym.sin(sym.pi*x) * sym.cos(sym.pi*y) * sym.exp(-(x**2 + y**2))
	example 4: f = sym.sin(sym.pi*(x-y)) * sym.exp(x+y)

	values other than 1,2,3,4 result in a custom setting which can be configured below
'''
# select an example
example = 1




''' set the input data '''

### custom setting
if example not in [1,2,3,4]:

	# target function
	f = sym.sin(sym.exp(sym.pi*x)) * sym.exp(-sym.Abs(sym.pi*x))

	# spatial domain and training points
	d = 1
	dom_x = [-1,1]
	num_tr = 50
	num_ts = 200

	# discretization parameters
	max_r = 25
	num_r = 250
	num_s = 500

	# SGN parameters
	D_threshold = .001
	nodes_threshold = .005


### example 1
elif example == 1:

	# target function
	f = sym.sin(2*sym.pi*x) * sym.exp(-x**2)

	# spatial domain and training points
	d = 1
	dom_x = [-1,1]
	num_tr = 50
	num_ts = 200

	# discretization parameters
	max_r = 25
	num_r = 250
	num_s = 500

	# SGN parameters
	D_threshold = .001
	nodes_threshold = .005


### example 2
elif example == 2:

	# target function
	f = sym.cos(sym.exp(3*x)) / (1 + 25*x**2)

	# spatial domain and training points
	d = 1
	dom_x = [-1,1]
	num_tr = 50
	num_ts = 200

	# discretization parameters
	max_r = 25
	num_r = 250
	num_s = 500

	# SGN parameters
	D_threshold = .001
	nodes_threshold = .005


### example 3
elif example == 3:

	# target function
	f = sym.sin(sym.pi*x) * sym.cos(sym.pi*y) * sym.exp(-(x**2 + y**2))

	# spatial domain and training points
	d = 2
	dom_x = [[-1,1], [-1,1]]
	num_tr = [20,20]
	num_ts = [40,40]

	# discretization parameters
	max_r = 15
	num_r = 150
	num_s = 1000

	# SGN parameters
	D_threshold = .001
	nodes_threshold = .005


### example 4
elif example == 4:

	# target function
	f = sym.sin(sym.pi*(x-y)) * sym.exp(x+y)

	# spatial domain and training points
	d = 2
	dom_x = [[-1,1], [-1,1]]
	num_tr = [20,20]
	num_ts = [40,40]

	# discretization parameters
	max_r = 15
	num_r = 150
	num_s = 1000

	# SGN parameters
	D_threshold = .001
	nodes_threshold = .005




''' activation and admissibility functions '''

# ReLU activation function
sigma = lambda z : z*(z>0)

# admissibility constant and function
K = -np.sqrt(2/np.pi) * (2*np.pi)**d
tau = sym.lambdify(x, sym.diff(sym.exp(-x**2/2) / K, x, 4))




''' generate the approximation variables '''

# check the dimensionality
if d != 1 and d != 2:
	print('\nerror: sorry, this code only works in the setting d=1 or d=2\n')
	raise SystemExit(0)

# report selected options
af.print_problem_setting(f, dom_x, num_tr, num_ts, max_r, num_r, num_s)

# construct the training and test sets
x_tr, f_tr, x_ts, f_ts = dr.sample_data(f, dom_x, num_tr, num_ts)

# discretize the inner weights
r = np.linspace(0, max_r, num_r+1)[1:]
s = dr.sample_points_sphere(num_s, d, plot=True)




''' approximation via the reconstruction formula '''

# compute the ridgelet transform
Rf = dr.ridgelet_transform(f_tr, x_tr, s, r, tau)

# evaluate discretization of the reconstruction formula
RRf_tr = dr.dual_ridgelet_transform(Rf, x_tr, s, r, sigma)
RRf_ts = dr.dual_ridgelet_transform(Rf, x_ts, s, r, sigma)

# report the approximation result
print('\napproximation by reconstruction formula:')
af.print_approximation_errors(f_tr, RRf_tr, f_ts, RRf_ts)




''' approximation via GSN initialization '''

# compute the collapsed ridgelet transform
CRf = dr.collapse_ridgelet(Rf, s, r, plot=True)

# construct dictionary
D = cw.construct_dictionary(CRf, x_tr, s, sigma)

# construct nodes by greedy algorithm
oga_ind, err = cw.orthogonal_greedy_algorithm(f_tr, D, D_threshold, plot=True)
A, B, C, num_nodes = cw.construct_nodes(s, oga_ind, f_tr, x_tr, sigma, err, nodes_threshold)

# evaluate the initialized network
ga_init_tr = nf.eval_nn(A, B, C, 0, sigma, x_tr)
ga_init_ts = nf.eval_nn(A, B, C, 0, sigma, x_ts)

# report the approximation result
print('approximation by GSN initialization:')
af.print_approximation_errors(f_tr, ga_init_tr, f_ts, ga_init_ts)




''' approximation via neural networks with GSN initialization '''

# train the network with constructed weights
A, B, C, c = nf.train_nn(x_tr, f_tr, num_nodes, weights_init=[A,B,C], summary=True, plot=True)

# evaluate the trained network
ga_train_tr = nf.eval_nn(A, B, C, c, sigma, x_tr)
ga_train_ts = nf.eval_nn(A, B, C, c, sigma, x_ts)

# report the approximation result
print('approximation by trained network with GSN initialization:')
af.print_approximation_errors(f_tr, ga_train_tr, f_ts, ga_train_ts)




''' approximation via neural networks with random initialization '''

# train the network
A, B, C, c = nf.train_nn(x_tr, f_tr, num_nodes, weights_init=None, summary=True, plot=True)

# evaluate the trained network
rand_tr = nf.eval_nn(A, B, C, c, sigma, x_tr)
rand_ts = nf.eval_nn(A, B, C, c, sigma, x_ts)

# report the approximation result
print('approximation by trained network with greedy initialization:')
af.print_approximation_errors(f_tr, rand_tr, f_ts, rand_ts)




''' report the approximation results '''

# print approximation errors
print('\n\n================ resulting approximation errors ================\n')
print('approximation by reconstruction formula:')
af.print_approximation_errors(f_tr, RRf_tr, f_ts, RRf_ts)
print('approximation by trained network with random initialization:')
af.print_approximation_errors(f_tr, rand_tr, f_ts, rand_ts)
print('approximation by GSN initialization:')
af.print_approximation_errors(f_tr, ga_init_tr, f_ts, ga_init_ts)
print('approximation by trained network by GSN initialization:')
af.print_approximation_errors(f_tr, ga_train_tr, f_ts, ga_train_ts)
print()




''' plot the constructed approximations '''

### 1d case
if d == 1:

	# display all 4 approximations on one figure
	af.display_plots([\
		('approx', x_ts, f_ts, RRf_ts, ['target function','reconstruction formula']),\
		('approx', x_ts, f_ts, rand_ts, ['target function','random trained']), \
		('approx', x_ts, f_ts, ga_init_ts, ['target function','GSN initial']), \
		('approx', x_ts, f_ts, ga_train_ts, ['target function','GSN trained']) \
		])


### 2d case
elif d == 2:

	# construct variables
	X, Y = x_ts[0,:].reshape(*num_ts), x_ts[1,:].reshape(*num_ts)
	Z = f_ts.reshape(*num_ts)

	# display all 4 approximations on one figure
	af.display_plots([\
		('contours_3d', X, Y, Z, RRf_ts.reshape(*num_ts), \
			(.1,.2,1), 100, 'approximation by reconstruction formula'), \
		('contours_3d', X, Y, Z, rand_ts.reshape(*num_ts), \
			(1,.1,.2), 100, 'approximation by network with random initialization'), \
		('contours_3d', X, Y, Z, ga_init_ts.reshape(*num_ts), \
			(0,.5,.5), 100, 'approximation by GSN initialization'), \
		('contours_3d', X, Y, Z, ga_train_ts.reshape(*num_ts), \
			(0,.5,.5), 100, 'approximation by network with GSN initialization'), \
		])



