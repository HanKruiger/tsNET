# Copyright (c) 2016 Paulo Eduardo Rauber

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#                 ^
#                / \
#                 |
#                 |
#
# License included because this module is a heavily modified version based on
# Paulo Rauber's implementation of dynamic t-SNE.
# (https://github.com/paulorauber/thesne)

import math
import numpy as np
import theano
import theano.tensor as T
from sklearn.utils import check_random_state
from scipy.spatial.distance import pdist

epsilon = 1e-16
floath = np.float32


class SigmaTooLowException(Exception):
    pass


class NaNException(Exception):
    pass


# Squared Euclidean distance between all pairs of row-vectors
def sqeuclidean_var(X):
    N = X.shape[0]
    ss = (X ** 2).sum(axis=1)
    return ss.reshape((N, 1)) + ss.reshape((1, N)) - 2 * X.dot(X.T)


# Euclidean distance between all pairs of row-vectors
def euclidean_var(X):
    return T.maximum(sqeuclidean_var(X), epsilon) ** 0.5


# Conditional probabilities of picking (ordered) pairs in high-dim space.
def p_ij_conditional_var(X, sigma):
    N = X.shape[0]

    sqdistance = X**2

    esqdistance = T.exp(-sqdistance / ((2 * (sigma**2)).reshape((N, 1))))
    esqdistance_zd = T.fill_diagonal(esqdistance, 0)

    row_sum = T.sum(esqdistance_zd, axis=1).reshape((N, 1))

    return esqdistance_zd / row_sum  # Possibly dangerous


# Symmetrized probabilities of picking pairs in high-dim space.
def p_ij_sym_var(p_ij_conditional):
    return (p_ij_conditional + p_ij_conditional.T) / (2 * p_ij_conditional.shape[0])


# Probabilities of picking pairs in low-dim space (using Student
# t-distribution).
def q_ij_student_t_var(Y):
    sqdistance = sqeuclidean_var(Y)
    one_over = T.fill_diagonal(1 / (sqdistance + 1), 0)
    return one_over / one_over.sum()


# Probabilities of picking pairs in low-dim space (using Gaussian).
def q_ij_gaussian_var(Y):
    sqdistance = sqeuclidean_var(Y)
    gauss = T.fill_diagonal(T.exp(-sqdistance), 0)
    return gauss / gauss.sum()


# Per point cost function
def cost_var(X, Y, sigma, l_kl, l_c, l_r, r_eps):
    N = X.shape[0]

    # Used to normalize s.t. the l_*'s sum up to one.
    l_sum = l_kl + l_c + l_r

    p_ij_conditional = p_ij_conditional_var(X, sigma)
    p_ij = p_ij_sym_var(p_ij_conditional)
    q_ij = q_ij_student_t_var(Y)

    p_ij_safe = T.maximum(p_ij, epsilon)
    q_ij_safe = T.maximum(q_ij, epsilon)

    # Kullback-Leibler term
    kl = T.sum(p_ij * T.log(p_ij_safe / q_ij_safe), axis=1)

    # Compression term
    compression = (1 / (2 * N)) * T.sum(Y**2, axis=1)

    # Repulsion term
    repulsion = -(1 / (2 * N**2)) * T.sum(T.fill_diagonal(T.log(euclidean_var(Y) + r_eps), 0), axis=1)

    # Sum of all terms.
    cost = (l_kl / l_sum) * kl + (l_c / l_sum) * compression + (l_r / l_sum) * repulsion

    return cost


# Binary search on sigma for a given perplexity
def find_sigma(X_shared, sigma_shared, N, perplexity, sigma_iters, verbose=0):
    X = T.fmatrix('X')
    sigma = T.fvector('sigma')

    target = np.log(perplexity)

    P = T.maximum(p_ij_conditional_var(X, sigma), epsilon)

    entropy = -T.sum(P * T.log(P), axis=1)

    # Setting update for binary search interval
    sigmin_shared = theano.shared(np.full(N, np.sqrt(epsilon), dtype=floath))
    sigmax_shared = theano.shared(np.full(N, np.inf, dtype=floath))

    sigmin = T.fvector('sigmin')
    sigmax = T.fvector('sigmax')

    upmin = T.switch(T.lt(entropy, target), sigma, sigmin)
    upmax = T.switch(T.gt(entropy, target), sigma, sigmax)

    givens = {X: X_shared, sigma: sigma_shared, sigmin: sigmin_shared,
              sigmax: sigmax_shared}
    updates = [(sigmin_shared, upmin), (sigmax_shared, upmax)]

    update_intervals = theano.function([], entropy, givens=givens, updates=updates)

    # Setting update for sigma according to search interval
    upsigma = T.switch(T.isinf(sigmax), sigma * 2, (sigmin + sigmax) / 2.)

    givens = {sigma: sigma_shared, sigmin: sigmin_shared,
              sigmax: sigmax_shared}
    updates = [(sigma_shared, upsigma)]

    update_sigma = theano.function([], sigma, givens=givens, updates=updates)

    for i in range(sigma_iters):
        e = update_intervals()
        update_sigma()
        if verbose:
            print('Finding sigmas... Iteration {0}/{1}: Perplexities in [{2:.4f}, {3:.4f}].'.format(i + 1, sigma_iters, np.exp(e.min()), np.exp(e.max())), end='\r')
        if np.any(np.isnan(np.exp(e))):
            raise SigmaTooLowException('Invalid sigmas. The perplexity is probably too low.')
    if verbose:
        print('\nDone. Perplexities in [{0:.4f}, {1:.4f}].'.format(np.exp(e.min()), np.exp(e.max())))


# Perform momentum-based gradient descent on the cost function with the given
# parameters. Return the vertex coordinates and per-vertex cost.
def find_Y(X_shared, Y_shared, sigma_shared, N, output_dims, n_epochs,
           initial_lr, final_lr, lr_switch, init_stdev, initial_momentum,
           final_momentum, momentum_switch,
           initial_l_kl, final_l_kl, l_kl_switch,
           initial_l_c, final_l_c, l_c_switch,
           initial_l_r, final_l_r, l_r_switch,
           r_eps, autostop=False, window_size=10, verbose=0):
    # Optimization hyperparameters
    initial_lr = np.array(initial_lr, dtype=floath)
    final_lr = np.array(final_lr, dtype=floath)
    initial_momentum = np.array(initial_momentum, dtype=floath)
    final_momentum = np.array(final_momentum, dtype=floath)

    # Hyperparameters used within Theano
    lr = T.fscalar('lr')
    lr_shared = theano.shared(initial_lr)
    momentum = T.fscalar('momentum')
    momentum_shared = theano.shared(initial_momentum)

    # Cost parameters
    initial_l_kl = np.array(initial_l_kl, dtype=floath)
    final_l_kl = np.array(final_l_kl, dtype=floath)
    initial_l_c = np.array(initial_l_c, dtype=floath)
    final_l_c = np.array(final_l_c, dtype=floath)
    initial_l_r = np.array(initial_l_r, dtype=floath)
    final_l_r = np.array(final_l_r, dtype=floath)

    # Cost parameters used within Theano
    l_kl = T.fscalar('l_kl')
    l_kl_shared = theano.shared(initial_l_kl)
    l_c = T.fscalar('l_c')
    l_c_shared = theano.shared(initial_l_c)
    l_r = T.fscalar('l_r')
    l_r_shared = theano.shared(initial_l_r)

    # High-dimensional observations (connectivities of vertices)
    X = T.fmatrix('X')
    # 2D projection (coordinates of vertices)
    Y = T.fmatrix('Y')

    # Standard deviations used for Gaussians to attain perplexity
    sigma = T.fvector('sigma')

    # Y velocities (for momentum-based descent)
    Yv = T.fmatrix('Yv')
    Yv_shared = theano.shared(np.zeros((N, output_dims), dtype=floath))

    # Function for retrieving cost for all individual data points
    costs = cost_var(X, Y, sigma, l_kl, l_c, l_r, r_eps)

    # Sum of all costs (scalar)
    cost = T.sum(costs)

    # Gradient of the cost w.r.t. Y
    grad_Y = T.grad(cost, Y)

    # Returns relative magnitude of stepsize, normalized by N, lr, and the range of the layout.
    stepsize = T.sum(T.sum(Yv ** 2, axis=1) ** 0.5) / (N * lr * T.max(T.max(Y, axis=0) - T.min(Y, axis=0)))

    # Update step for velocity
    update_Yv = theano.function(
        [], stepsize, # Returns the normalized stepsize
        givens={
            X: X_shared,
            sigma: sigma_shared,
            Y: Y_shared,
            Yv: Yv_shared,
            lr: lr_shared,
            momentum: momentum_shared,
            l_kl: l_kl_shared,
            l_c: l_c_shared,
            l_r: l_r_shared
        },
        updates=[
            (Yv_shared, momentum * Yv - lr * grad_Y)
        ]
    )

    # Gradient descent step
    update_Y = theano.function(
        [], None,
        givens={
            Y: Y_shared, Yv: Yv_shared
        },
        updates=[
            (Y_shared, Y + Yv)
        ]
    )

    # Build function to retrieve cost
    get_cost = theano.function(
        [], cost,
        givens={
            X: X_shared,
            sigma: sigma_shared,
            Y: Y_shared,
            l_kl: l_kl_shared,
            l_c: l_c_shared,
            l_r: l_r_shared
        }
    )

    # Build function to retrieve per-vertex cost
    get_costs = theano.function(
        [], costs,
        givens={
            X: X_shared,
            sigma: sigma_shared,
            Y: Y_shared,
            l_kl: l_kl_shared,
            l_c: l_c_shared,
            l_r: l_r_shared
        }
    )

    # Build a list of the stepsizes over time. For convergence detection.
    stepsize_over_time = np.zeros(n_epochs)

    # Function that checks if we're converged.
    def is_converged(epoch, stepsize_over_time, tol=1e-8):
        if epoch > window_size:
            max_stepsize = stepsize_over_time[epoch - window_size:epoch].max()
            return max_stepsize < tol
        return False

    # Optimization loop
    converged = False
    for epoch in range(n_epochs):

        # Switch parameters if a switching point is reached.
        if epoch == lr_switch:
            lr_shared.set_value(final_lr)
        if epoch == momentum_switch:
            momentum_shared.set_value(final_momentum)
        if epoch == l_kl_switch:
            l_kl_shared.set_value(final_l_kl)
        if epoch == l_c_switch:
            l_c_shared.set_value(final_l_c)
        if epoch == l_r_switch:
            s1 = epoch
            l_r_shared.set_value(final_l_r)

        # Do update step for velocity
        dY_norm = update_Yv()
        stepsize_over_time[epoch] = dY_norm # Save normalized stepsize
        
        # Do a gradient descent step
        update_Y()


        c = get_cost()
        if np.isnan(float(c)):
            raise NaNException('Encountered NaN for cost.')

        if verbose:
            if autostop and epoch >= window_size:
                dlast_period = stepsize_over_time[epoch - window_size:epoch]
                max_stepsize = dlast_period.max()
                print('Epoch: {0}. Cost: {1:.6f}. Max step size of last {2}: {3:.2e}'.format(epoch + 1, float(c), window_size, max_stepsize), end='\r')
            else:
                print('Epoch: {0}. Cost: {1:.6f}.'.format(epoch + 1, float(c)), end='\r')

        # Switch phases if we're converged. Or exit if we're already in the last phase.
        if autostop and is_converged(epoch, stepsize_over_time, tol=autostop):
            if epoch < lr_switch:
                lr_switch = epoch + 1
                momentum_switch = epoch + 1
                l_kl_switch = epoch + 1
                l_c_switch = epoch + 1
                l_r_switch = epoch + 1
                print('\nAuto-switching at epoch {0}'.format(epoch))
            elif epoch > lr_switch + window_size:
                print('\nAuto-stopping at epoch {0}'.format(epoch))
                converged = True
                break

    if not converged:
        print('\nWarning: Did not converge!')

    return np.array(Y_shared.get_value())


def tsnet(X, perplexity=30, Y=None, output_dims=2, n_epochs=1000,
         initial_lr=10, final_lr=4, lr_switch=None, init_stdev=1e-4,
         sigma_iters=50, initial_momentum=0.5, final_momentum=0.8,
         momentum_switch=250,
         initial_l_kl=None, final_l_kl=None, l_kl_switch=None,
         initial_l_c=None, final_l_c=None, l_c_switch=None,
         initial_l_r=None, final_l_r=None, l_r_switch=None,
         r_eps=1, random_state=None,
         autostop=False, window_size=10, verbose=1):
    random_state = check_random_state(random_state)

    # Number of vertices/observations
    N = X.shape[0]

    X_shared = theano.shared(np.asarray(X, dtype=floath))
    sigma_shared = theano.shared(np.ones(N, dtype=floath))

    # Randomly initialize Y if it's not defined.
    if Y is None:
        Y = random_state.normal(0, init_stdev, size=(N, output_dims))
    Y_shared = theano.shared(np.asarray(Y, dtype=floath))

    # Find sigmas to attain the given perplexity.
    find_sigma(X_shared, sigma_shared, N, perplexity, sigma_iters, verbose)

    # Do the optimization to find Y (the node coordinates).
    Y = find_Y(
        X_shared, Y_shared, sigma_shared, N, output_dims, n_epochs,
        initial_lr, final_lr, lr_switch, init_stdev, initial_momentum,
        final_momentum, momentum_switch,
        initial_l_kl, final_l_kl, l_kl_switch,
        initial_l_c, final_l_c, l_c_switch,
        initial_l_r, final_l_r, l_r_switch,
        r_eps, autostop, window_size, verbose
    )

    # Return the vertex coordinates.
    return Y
