'''
	this file contains auxilary functions related to plotting and printing various data
'''


import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import sympy as sym




''' display multiple plots on the same figure '''
def display_plots(plots, shape=None):

	# find the number of plots
	num_plots = len(plots)
	# determine the plots positioning
	if shape:
		col_num, row_num = shape
	else:
		col_num = int(np.floor(np.sqrt(num_plots)))
		row_num = int(np.ceil(num_plots / col_num))
	# create the figure
	fig = plt.figure(figsize=(9.4*row_num, 6.4*col_num))

	# construct plots
	for n in range(min(num_plots, col_num*row_num)):
		# add subplot
		if '3d' in plots[n][0]:
			axs = fig.add_subplot(col_num, row_num, n+1, projection='3d')
		elif 'polar' in plots[n][0]:
			axs = fig.add_subplot(col_num, row_num, n+1, projection='polar')
		else:
			axs = fig.add_subplot(col_num, row_num, n+1)

		# construct the plot on the created axes
		if plots[n][0] == 'approx':
			plot_approximation(*plots[n][1:], axs=axs)
		elif plots[n][0] == 'contour_2d':
			plot_contour_2d(*plots[n][1:], axs=axs)
		elif plots[n][0] == 'contour_3d':
			plot_contour_3d(*plots[n][1:], axs=axs)
		elif plots[n][0] == 'surface_3d':
			plot_surface_3d(*plots[n][1:], axs=axs)
		elif plots[n][0] == 'contours_3d':
			plot_contours_3d(*plots[n][1:], axs=axs)
		elif plots[n][0] == 'scatter_3d':
			plot_scatter_3d(*plots[n][1:], axs=axs)

	# display the plots
	plt.tight_layout()
	plt.show()
	return




''' plot the function (f) and its approximation (g) on the interval (x) '''
def plot_approximation(x, f, g=[], labels=[], title=None, ticks_num=[5,5], axs=None):

	colors = ['#4488ff', '#51cc8e', '#ff4422', '#d8ed3d', '#8844ff', '#f98220']
	# set up a figure
	if axs:
		ax = axs
	else:
		fig = plt.figure(figsize=(10,6))
		ax = plt.axes()

	# plot the function
	ax.plot(x, f, ('--' if g!=[] else '-'), color=colors[0])

	# plot single approximation
	if isinstance(g, np.ndarray):
		# adjust labels
		if labels:
			if len(labels) == 2:
				labels[1] += ' {:.3f}'.format(norm(f-g,2) / norm(f,2))
			else:
				labels = ['target function', ' {:.3f}'.format(norm(f-g,2) / norm(f,2))]
		# plot the approximation
		ax.plot(x, g, color='r')

	# plot multiple approximations
	else:
		# number of approximations
		num_approx = len(g)
		# adjust labels
		if labels and len(labels) != 1 + num_approx:
			labels = ['target function'] + ['']*num_approx
		# plot the approximations
		for n in range(num_approx):
			ax.plot(x, g[n], color=colors[n+1])
			labels[n+1] += ' {:.3f}'.format(norm(f-g[n],2) / norm(f,2))

	# add legend
	if labels:
		ax.legend(labels, prop={'family':'monospace', 'size':12})
	# add title
	if title:
		ax.set_title(title, fontfamily='monospace', fontsize=14)
	# configure ticks
	ax.set_xticks(np.linspace(np.min(x), np.max(x), ticks_num[0]))
	ax.set_yticks(np.linspace(np.min(f), np.max(f), ticks_num[1]))
	# adjust ticks for polar coordinates
	if np.abs(2*np.pi - x[-1] + x[0]) < .1:
		adjust_ticks_polar(x, ax)

	# show the plot
	if not axs:
		plt.tight_layout()
		plt.show()
	return




'''
	generate 2d meshes (X,Y,Z) based on the values (x,y,z)
	if a tuple of indices (ind) is given, take z[ind] instead
	if (z) is rank 1, take the values of (z) at the points (x,y[ind])
'''
def process_variables_3d(z, x, y, ind, x_lim, y_lim):

	# if x,y are polar
	if np.all(np.abs(x) < 2*np.pi) and np.all(y > 0):
		# restrist the values of r
		x_domain = (x < np.inf)
		y_domain = (y <= y_lim[1])
		X = np.outer(np.sin(x), y[y_domain])
		Y = np.outer(np.cos(x), y[y_domain])
		z_domain = np.outer(x_domain, y_domain)

	# if x,y are cartesian
	else:
		# restrist the values of x,y
		x_domain = (x >= x_lim[0]) * (x <= x_lim[1])
		y_domain = (y >= y_lim[0]) * (y <= y_lim[1])
		X, Y = np.meshgrid(x[x_domain], y[y_domain], indexing='ij')
		z_domain = np.outer(x_domain, y_domain)

	# adjust the values of z
	if ind:
		Z = np.zeros(z_domain.shape)
		if len(z.shape) == 1:
			Z[ind] = z
		else:
			Z[ind] = z[ind]
		Z = Z[z_domain].reshape(X.shape)
	else:
		Z = z[z_domain].reshape(X.shape)

	return X, Y, Z




''' construct a 3d-plot of contour/surface of the function (Z) on the grid (X,Y) '''
def plot_3d(type, X, Y, Z, labels=None, title=None, ticks_num=[5,5], colormap='bwr', axs=None):
	# set up a figure
	if axs:
		ax = axs
	else:
		fig = plt.figure(figsize=(10,8))
		ax = plt.axes(projection='3d')

	# plot the contour/surface
	if type == 'contour':
		ax.contourf(X, Y, Z, levels=100, cmap=colormap)
	elif type == 'surface':
		ax.plot_surface(X, Y, Z, rcount=100, ccount=100, cmap=colormap)
	ax.view_init(elev=15, azim=-50)

	# configure the axes
	if labels:
		ax.set_xlabel(labels[0], fontfamily='monospace', fontsize=12)
		ax.set_ylabel(labels[1], fontfamily='monospace', fontsize=12)
		ax.set_zlabel(labels[2], fontfamily='monospace', fontsize=12)
	if title:
		ax.set_title(title, fontfamily='monospace', fontsize=14)
	# configure the ticks
	ax.set_xticks(np.linspace(np.amin(X), np.amax(X), ticks_num[0]))
	ax.set_yticks(np.linspace(np.amin(Y), np.amax(Y), ticks_num[1]))
	# adjust ticks for polar coordinates
	if np.abs(2*np.pi - Y[0][-1] + Y[0][0]) < .1:
		adjust_ticks_polar(Y[0], ax, label='y')

	# show the plot
	if not axs:
		plt.tight_layout()
		plt.show()
	return




''' plot 3d-contour of the function (z) on the grid (x) x (y) '''
def plot_contour_3d(z, x, y, ind=[], labels=None, title=None, ticks_num=[5,5],\
					x_lim=[-10,10], y_lim=[-10,10], colormap='bwr', axs=None):
	# construct the values for X, Y, Z
	X, Y, Z = process_variables_3d(z, x, y, ind=ind, x_lim=x_lim, y_lim=y_lim)
	# pass arguments to the plot_3d function
	plot_3d('contour', X, Y, Z, labels=labels, title=title, \
			ticks_num=ticks_num, colormap=colormap, axs=axs)
	return




''' plot 3d-surface of the function (z) on the grid (x) x (y) '''
def plot_surface_3d(z, x, y, ind=[], labels=None, title=None, ticks_num=[5,5], \
					x_lim=[-10,10], y_lim=[-10,10], colormap='bwr', axs=None):
	# construct the values for X, Y, Z
	X, Y, Z = process_variables_3d(z, x, y, ind=ind, x_lim=x_lim, y_lim=y_lim)
	# pass arguments to the plot_3d function
	plot_3d('surface', X, Y, Z, labels=labels, title=None, \
			ticks_num=ticks_num, colormap=colormap, axs=axs)
	return




''' plot two 3d-contours (Z,W) on the grid (X,Y) '''
def plot_contours_3d(X, Y, Z, W, color=(0,.5,.5), num_lvl=100, title=None, axs=None):
	# set up a figure
	if axs:
		ax = axs
	else:
		fig = plt.figure(figsize=(12,9))
		ax = plt.axes(projection='3d')
	ax.view_init(elev=25, azim=75)

	# construct levels
	levels = np.linspace(np.amax([Z, W]), np.amin([Z, W]), num_lvl)
	if np.amax(Z) > np.amax(W):
		lvl_Z, lvl_W = levels[::2][::-1], levels[1::2][::-1]
	else:
		lvl_Z, lvl_W = levels[1::2][::-1], levels[::2][::-1]

	# plot the contours
	ax.contourf(X, Y, Z, levels=lvl_Z, cmap=generate_colormap(.8,.8,0))
	ax.contourf(X, Y, W, levels=lvl_W, cmap=generate_colormap(*color))
	# add title
	if title:
		ax.set_title(title, fontfamily='monospace', fontsize=14)

	# show the plot
	if not axs:
		plt.tight_layout()
		plt.show()
	return




''' plot the points (x,y,z) in 3d space '''
def plot_scatter_3d(x, y, z, cmap=None, title=None, colorbar=False, axs=None):
	# set up a figure
	if axs:
		ax = axs
	else:
		fig = plt.figure(figsize=(10,9))
		ax = plt.axes(projection='3d')

	# plot the points
	scatter_3d = ax.scatter(x, y, z, c=z, cmap=cmap)
	# add colorbar
	if colorbar:
		fig.set_figwidth(11)
		plt.colorbar(scatter_3d, ax=ax, shrink=.75, aspect=15)
	# add title
	if title:
		ax.set_title(title, fontfamily='monospace', fontsize=14)

	# show the plot
	if not axs:
		plt.tight_layout()
		plt.show()
	return




''' generate custom colormap to use in plots '''
def generate_colormap(rd=1, gr=0, bl=0, alpha=1, \
						blending=.75, blend_color=(1,1,1)):

	# extract parameters
	a, b = 1-blending, blending
	rd_, gr_, bl_ = blend_color

	# generate color dictionary
	color_dict = {
		'red': (
			(0.0, rd, rd),
			(0.5, a*rd+b*rd_, a*rd+b*rd_),
			(1.0, rd, rd)),
		'green': (
			(0.0, gr, gr),
			(0.5, a*gr+b*gr_, a*gr+b*gr_),
			(1.0, gr, gr)),
		'blue': (
			(0.0, bl, bl),
			(0.5, a*bl+b*bl_, a*bl+b*bl_),
			(1.0, bl, bl)),
		'alpha': (
			(0.0, 1.0, 1.0),
			(0.5, alpha, alpha),
			(1.0, 1.0, 1.0)),
		}

	# construct colormap
	colormap = LinearSegmentedColormap('cmap'+str((rd,gr,bl,alpha,blending,blend_color)), color_dict)

	return colormap




''' adjust the ticks to pi/8 increments for the polar coordinates '''
def adjust_ticks_polar(a, axis, label='x'):
	# adjust x-axis
	if label == 'x':
		axis.set_xticks(np.linspace(np.min(a), np.max(a), 17))
		axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda val,pos: \
			r'$\frac{%.1g\pi}{8}$'%(8*val / np.pi) if np.abs(val/np.pi - np.around(val/np.pi)) > .1 \
			else '$-\pi$' if np.around(val / np.pi) == -1 \
			else '$0$' if np.around(val / np.pi) == 0 \
			else '$\pi$' if np.around(val / np.pi) == 1 \
			else '$%d\pi$'%(np.around(val / np.pi))))
		axis.grid(True, axis=label, linestyle='--')
	# adjust y-axis
	else:
		axis.set_yticks(np.linspace(np.min(a), np.max(a), 17))
		axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda val,pos: \
			r'$\frac{%.1g\pi}{8}$'%(8*val / np.pi) if np.abs(val/np.pi - np.around(val/np.pi)) > .1 \
			else '$-\pi$' if np.around(val / np.pi) == -1 \
			else '$0$' if np.around(val / np.pi) == 0 \
			else '$\pi$' if np.around(val / np.pi) == 1 \
			else '$%d\pi$'%(np.around(val / np.pi))))
	return




''' print the launch parameters '''
def print_problem_setting(f, dom_x, num_tr, num_ts, max_r, num_r, num_s):
	# problem setting
	print('\nproblem setting:')
	print('target function: f =', f)
	# hyperparameters
	print('\ndiscretization parameters:')
	d = len(np.array(dom_x).shape)
	# 1d case
	if d == 1:
		print(' {0} < x < {1}, training/testing points: {2}/{3}'.\
			format(dom_x[0], dom_x[1], num_tr, num_ts))
	if d == 2:
		# 2d case
		print(' {0} < x < {1}, training/testing points: {2}/{3}'.\
			format(dom_x[0][0], dom_x[0][1], num_tr[0], num_ts[0]))
		print(' {0} < y < {1}, training/testing points: {2}/{3}'.\
			format(dom_x[1][0], dom_x[1][1], num_tr[1], num_ts[1]))
	print('{0} < r < {1}, sampling points: {2}'.format('  0', max_r, num_r))
	print('  s is in S{0}, sampling points: {1}'.format(d, num_s))
	print()
	return




''' print the relative errors of approximating (f) by (g) in the provided (norms) '''
def print_approximation_errors(f_0, g_0, f_1=None, g_1=None, norms=[2], sep=False):
	# display errors in the corresponding norms
	if f_1 is not None and g_1 is not None:
		for p in np.array(norms).reshape(-1):
			print(' {:>.3s}-error:  {:.2e} / {:.2e}'.format(str(p), \
				norm(np.ravel(f_0 - g_0), p) / norm(np.ravel(f_0), p), \
				norm(np.ravel(f_1 - g_1), p) / norm(np.ravel(f_1), p)))
	else:
		for p in np.array(norms).reshape(-1):
			print(' {:>.3s}-error:  {:.2e}'.format(str(p), \
				norm(np.ravel(f_0 - g_0), p) / norm(np.ravel(f_0), p)))
	if sep:
		print('\n')
	return



