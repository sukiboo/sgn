
import numpy as np
import sympy as sym
import discretization_routines as dr
import auxiliary_functions as af
import construct_weights as cw
import network_functions as nf
import seaborn as sns
import time

# set print options
np.set_printoptions(precision=3, linewidth=115, suppress=True, formatter={'float':'{: 0.3f}'.format})
# fix the seed for reproducibility
np.random.seed(0)
sns.set_theme(style='whitegrid', rc={"lines.linewidth": 4})


''' set the input data '''
example = '1'


if example == '1':
    # set dimensionality
    d = 1
    # define symbolic variables
    x = sym.symbols('x')
    # target function
    f = sym.cos(2*sym.pi*x) * sym.exp(x)
    # spatial domain and training/validation/test points
    dom_x = [[-1,1]]
    num_tr = [50]
    num_vl = 15
    num_ts = [1000]
    # gsn parameters
    num_pts = 10000
    max_iter = 40

if example == '2':
    # set dimensionality
    d = 1
    # define symbolic variables
    x = sym.symbols('x')
    # target function
    f = sym.sin(2*sym.pi*x) * sym.exp(-x**2) + sym.cos(17*x) * sym.exp(x)
    # spatial domain and training/validation/test points
    dom_x = [[-1,1]]
    num_tr = [100]
    num_vl = 20
    num_ts = [1000]
    # gsn parameters
    num_pts = 10000
    max_iter = 60

elif example == '3':
    # set dimensionality
    d = 2
    # define symbolic variables
    x = sym.symbols('x,y')
    # target function
    f = sym.sin(sym.pi*x[0]) * sym.cos(sym.pi*x[1]) * sym.exp(-(x[0]**2 + x[1]**2))
    # spatial domain and training/validation/test points
    dom_x = [[-1,1], [-1,1]]
    num_tr = [16,16]
    num_vl = 50
    num_ts = [100,100]
    # gsn parameters
    num_pts = 20000
    max_iter = 60

elif example == '4':
    # set dimensionality
    d = 2
    # define symbolic variables
    x = sym.symbols('x,y')
    # target function
    f = sym.cos(5*(x[0] + x[1])) * sym.sin(3*(x[0] - x[1])) * sym.exp(-(x[0]**2 + x[1]**2))
    # spatial domain and training/validation/test points
    dom_x = [[-1,1], [-1,1]]
    num_tr = [32,32]
    num_vl = 200
    num_ts = [100,100]
    # gsn parameters
    num_pts = 20000
    max_iter = 100

elif example == '5':
    # set dimensionality
    d = 4
    # define symbolic variables
    x = sym.symbols('x0:{:d}'.format(d))
    # target function
    f = sym.sin(2*sym.pi * np.sum([x[k] for k in range(d)]))
    # spatial domain and training/validation/test points
    dom_x = [[-1,1] for _ in range(d)]
    num_tr = 4000
    num_vl = 400
    num_ts = 10000
    # gsn parameters
    num_pts = 40000
    max_iter = 200


''' generate the approximation variables '''
# activation function
def ReLU(z):
    relu = z * (z > 0)
    return relu
sigma = ReLU
# report selected options
af.print_problem_setting(d, f, sigma, num_tr, num_vl, num_ts, num_pts)
# construct the training and test sets
ff = sym.lambdify(x, f)
x_tr, f_tr = dr.sample_data(d, ff, dom_x, num_tr)
x_vl, f_vl = dr.sample_data(d, ff, dom_x, num_vl)
x_ts, f_ts = dr.sample_data(d, ff, dom_x, num_ts)
# discretize the inner weights
pts, ang = dr.sample_points_sphere(d, num_pts, plot=False)
# compute and plot ridgelet transform
Rf = dr.ridgelet_transform(f_tr, x_tr, ang)
CRf = dr.collapse_ridgelet(Rf, ang, plot=True)


''' approximation via GSN initialization '''
# greedy approximation on the training data
oga_ind, oga_err = cw.orthogonal_greedy_algorithm(pts, sigma, x_tr, f_tr,\
                    val_data=[x_vl, f_vl], max_iter=max_iter, verbose=True)
# construct nodes
A, B, C, num_nodes = cw.construct_nodes(pts, oga_ind, f_tr, x_tr, sigma, oga_err, plot=True)

# evaluate the initialized network
gsn_tr = nf.eval_nn(A, B, C, 0, sigma, x_tr)
gsn_ts = nf.eval_nn(A, B, C, 0, sigma, x_ts)
# report the approximation result
print('approximation by GSN initialization:')
af.print_approximation_errors(f_tr, gsn_tr, f_ts, gsn_ts)


''' approximation via neural networks with GSN initialization '''
# train the network with constructed weights
A, B, C, c = nf.train_nn(d, x_tr, f_tr, num_nodes, weights_init=[A,B,C], summary=True, plot=True)
# evaluate the trained network
nn_gsn_tr = nf.eval_nn(A, B, C, c, sigma, x_tr)
nn_gsn_ts = nf.eval_nn(A, B, C, c, sigma, x_ts)
# report the approximation result
print('approximation by trained network with GSN initialization:')
af.print_approximation_errors(f_tr, nn_gsn_tr, f_ts, nn_gsn_ts)


''' approximation via neural networks with random initialization '''
# train the network
A, B, C, c = nf.train_nn(d, x_tr, f_tr, num_nodes, weights_init=None, summary=True, plot=True)
# evaluate the trained network
nn_rnd_tr = nf.eval_nn(A, B, C, c, sigma, x_tr)
nn_rnd_ts = nf.eval_nn(A, B, C, c, sigma, x_ts)
# report the approximation result
print('approximation by trained network with random initialization:')
af.print_approximation_errors(f_tr, nn_rnd_tr, f_ts, nn_rnd_ts)


''' report the approximation results '''
# print approximation errors
print('\n\n================ resulting approximation errors ================\n')
print('approximation by network with random initialization:')
af.print_approximation_errors(f_tr, nn_rnd_tr, f_ts, nn_rnd_ts)
print('approximation by GSN initialization:')
af.print_approximation_errors(f_tr, gsn_tr, f_ts, gsn_ts)
print('approximation by network with GSN initialization:')
af.print_approximation_errors(f_tr, nn_gsn_tr, f_ts, nn_gsn_ts)
print()

# plot approximations
# 1d case
if d == 1:
    # display all 3 approximations on one figure
    af.display_plots([\
        ('approx', x_ts, f_ts, nn_rnd_ts, ['target function','random trained'],\
            'approximation error: {:.2e}'.format(\
                np.linalg.norm(f_ts - nn_rnd_ts.ravel()) / np.linalg.norm(f_ts))),\
        ('approx', x_ts, f_ts, gsn_ts, ['target function','GSN initialization'],\
            'approximation error: {:.2e}'.format(\
                np.linalg.norm(f_ts - gsn_ts.ravel()) / np.linalg.norm(f_ts))),\
        ('approx', x_ts, f_ts, nn_gsn_ts, ['target function','GSN trained'],\
            'approximation error: {:.2e}'.format(\
                np.linalg.norm(f_ts - nn_gsn_ts.ravel()) / np.linalg.norm(f_ts)))\
        ])

# 2d case
elif d == 2:
    # construct variables
    X, Y = x_ts[:,0].reshape(*num_ts), x_ts[:,1].reshape(*num_ts)
    Z = f_ts.reshape(*num_ts)
    # display all 3 approximations on one figure
    af.display_plots([\
        ('contours_3d', X, Y, Z, nn_rnd_ts.reshape(*num_ts), (1.,.1,.2), 100,\
            'random initialization: {:.2e}'.format(\
                np.linalg.norm(f_ts.ravel() - nn_rnd_ts.ravel()) / np.linalg.norm(f_ts.ravel()))),\
        ('contours_3d', X, Y, Z, gsn_ts.reshape(*num_ts), (.0,.5,.5,), 100,\
            'gsn initialization: {:.2e}'.format(\
                np.linalg.norm(f_ts.ravel() - gsn_ts.ravel()) / np.linalg.norm(f_ts.ravel()))),\
        ('contours_3d', X, Y, Z, nn_gsn_ts.reshape(*num_ts), (.0,.5,.5,), 100,\
            'gsn network: {:.2e}'.format(\
                np.linalg.norm(f_ts.ravel() - nn_gsn_ts.ravel()) / np.linalg.norm(f_ts.ravel())))\
        ])


