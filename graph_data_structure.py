"""Directed Graph 1
    Vertices: {1, 2, 3}
    Edges: {{1, 2},
            {1, 3},
            {2, 3}}
"""

# Edge list representation - memory space and search time O(Edges)
dg_1 = [[1, 2],
        [1, 3],
        [2, 3]]

# Adjacency matrix representation
dg_2 = [[0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]]

"""Undirected Graph 1
    Vertices: {1, 2, 3}
    Edges: {(2, 1),
            (1, 3),
            (3, 2)}
"""

# Edge list representation
ug_1 = [(2, 1),
        (1, 3),
        (3, 2)]

# Adjacency matrix representation - directed since matrix non-symmetric
ug_2 = [[0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]]
