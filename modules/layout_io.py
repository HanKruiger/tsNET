import os
import numpy as np
import graph_tool.all as gt
import modules.graph_io as graph_io


def load_layout(in_file):
    extension = os.path.splitext(in_file)[1]
    if extension == '.vna':
        return load_vna(in_file)

    
def save_layout(out_file, g, Y):
    extension = os.path.splitext(out_file)[1]
    if extension == '.vna':
        save_vna(out_file, g, Y)

def load_vna(in_file):
    with open(in_file) as f:
        all_lines = f.read().splitlines()

        it = iter(all_lines)

        # Ignore preamble
        line = next(it)
        while not line.lower().startswith('*node data'):
            line = next(it)
        
        node_data = [word.lower() for word in next(it).split(' ')]
        assert('id' in node_data)

        vertices = dict()
        line = next(it)
        gt_idx = 0 # Index for gt
        while not line.lower().startswith('*'):
            entries = line.split(' ')
            vna_id = entries[0]
            vertex = dict()
            vertex['id'] = gt_idx # Replace VNA ID by numerical gt index
            vertices[vna_id] = vertex # Retain VNA ID as key of the vertices dict
            
            gt_idx += 1
            line = next(it)

        assert(line.lower().startswith('*node properties'))
        line = next(it)
        node_properties = [word.lower() for word in line.split(' ')]
        assert('x' in node_properties and 'y' in node_properties)

        line = next(it)
        # Read node properties (x and y)
        while not line.lower().startswith('*'):
            entries = line.split(' ')
            vna_id = entries[0]
            vertex = vertices[vna_id]
            for i, prop in enumerate(node_properties[1:]):
                vertex[prop] = entries[i + 1]
            line = next(it)

        assert(line.lower().startswith('*tie data'))
        edge_properties = next(it).split(' ')
        assert(edge_properties[0] == 'from' and edge_properties[1] == 'to')

        edges = []
        try:
            while True:
                line = next(it)
                entries = line.split(' ')
                v_i = vertices[entries[0]]['id']
                v_j = vertices[entries[1]]['id']
                edges.append((v_i, v_j))
        except StopIteration:
            pass

        g = gt.Graph(directed=False)
        g.add_vertex(len(vertices))
        for v_i, v_j in edges:
            g.add_edge(v_i, v_j)

        gt.remove_parallel_edges(g)

        Y = np.zeros((g.num_vertices(), 2))
        for v in vertices.keys():
            Y[vertices[v]['id'], 0] = float(vertices[v]['x'])
            Y[vertices[v]['id'], 1] = float(vertices[v]['y'])
        pos = g.new_vertex_property('vector<double>')
        pos.set_2d_array(Y.T)

        return g, Y
    return None


def save_vna(out_file, g, Y):
    with open(out_file, 'w') as f:
        f.write('*Node data\n')
        f.write('ID\n')
        for v in g.vertices():
            f.write('{0}\n'.format(int(v)))
        f.write('*Node properties\n')
        f.write('ID x y\n')
        for v in g.vertices():
            x = Y[int(v), 0]
            y = Y[int(v), 1]
            f.write('{0} {1} {2}\n'.format(int(v), x, y))
        f.write('*Tie data\n')
        f.write('from to strength\n')
        for v1, v2 in g.edges():
            f.write('{0} {1} 1\n'.format(int(v1), int(v2)))
        f.close()


# Normalize in [0, 1] x [0, 1] (without changing aspect ratio)
def normalize_layout(Y, verbose=False):
    Y_cpy = Y.copy()
    # Translate s.t. smallest values for both x and y are 0.
    for dim in range(Y.shape[1]):
        Y_cpy[:, dim] += -Y_cpy[:, dim].min()
        
    # Scale s.t. max(max(x, y)) = 1 (while keeping the same aspect ratio!)
    scaling = 1 / (np.absolute(Y_cpy).max())
    Y_cpy *= scaling

    if verbose:
        print("[layout_io] Normalized layout by factor {0}".format(scaling))

    return Y_cpy