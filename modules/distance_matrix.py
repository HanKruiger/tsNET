import graph_tool.all as gt
import numpy as np
import itertools


def get_modified_adjacency_matrix(g, k):
    # Get regular adjacency matrix
    adj = gt.adjacency(g)

    # Initialize the modified adjacency matrix
    X = np.zeros(adj.shape)

    # Loop over nonzero elements
    for i, j in zip(*adj.nonzero()):
        X[i, j] = 1 / adj[i, j]

    adj_max = adj.max()

    # Loop over zero elements
    for i, j in set(itertools.product(range(adj.shape[0]), range(adj.shape[1]))).difference(zip(*adj.nonzero())):
        X[i, j] = k * adj_max

    return X


def get_shortest_path_distance_matrix(g, k=10, weights=None):
    # Used to find which vertices are not connected. This has to be this weird,
    # since graph_tool uses maxint for the shortest path distance between
    # unconnected vertices.
    def get_unconnected_distance():
        g_mock = gt.Graph()
        g_mock.add_vertex(2)
        shortest_distances_mock = gt.shortest_distance(g_mock)
        unconnected_dist = shortest_distances_mock[0][1]
        return unconnected_dist

    # Get the value (usually maxint) that graph_tool uses for distances between
    # unconnected vertices.
    unconnected_dist = get_unconnected_distance()
    
    # Get shortest distances for all pairs of vertices in a NumPy array.
    X = gt.shortest_distance(g, weights=weights).get_2d_array(range(g.num_vertices()))

    if len(X[X == unconnected_dist]) > 0:
        print('[distance_matrix] There were disconnected components!')

    # Get maximum shortest-path distance (ignoring maxint)
    X_max = X[X != unconnected_dist].max()

    # Set the unconnected distances to k times the maximum of the other
    # distances.
    X[X == unconnected_dist] = k * X_max
    
    return X


# Return the distance matrix of g, with the specified metric.
def get_distance_matrix(g, distance_metric, normalize=True, k=10.0, verbose=True, weights=None):
    if verbose:
        print('[distance_matrix] Computing distance matrix (metric: {0})'.format(distance_metric))
    
    if distance_metric == 'shortest_path' or distance_metric == 'spdm':
        X = get_shortest_path_distance_matrix(g, weights=weights)
    elif distance_metric == 'modified_adjacency' or distance_metric == 'mam':
        X = get_modified_adjacency_matrix(g, k)
    else:
        raise Exception('Unknown distance metric.')

    # Just to make sure, symmetrize the matrix.
    X = (X + X.T) / 2

    # Force diagonal to zero
    X[range(X.shape[0]), range(X.shape[1])] = 0

    # Normalize matrix s.t. max is 1.
    if normalize:
        X /= np.max(X)
    if verbose:
        print('[distance_matrix] Done!')

    return X
