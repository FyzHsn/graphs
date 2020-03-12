class Graph:
    def __init__(self, graph):
        self.graph = graph
        self.node_num = None
        self.node_vec_map = None
        self.vec_node_map = None
        self.matrix = None

    def _vector_repr_init(self):
        self.node_num = len(self.graph.keys())

        self.node_vec_map = {node: index for index, node in
                             enumerate(self.graph.keys())}
        self.vec_node_map = {v: k for (k, v) in self.node_vec_map.items()}

        self.matrix = [[0 for i in range(0, self.node_num)] for j in
                       range(0, self.node_num)]

    def vectorize(self):
        self._vector_repr_init()
        for node, adj_nodes in self.graph.items():
            node_index = self.node_vec_map[node]

            for adj_node in adj_nodes:
                self.matrix[node_index][self.node_vec_map[adj_node]] = 1

    def node_geodesic(self, node_1, node_2):
        """Geodesic between two nodes

        This method finds the shortest path between two nodes in a graph. This
        function can be applied to any connected or disconnected undirected
        graphs.

        :param node_1: first node of interest
        :type node_1: int
        :param node_2: second node of interest different than first node
        :type node_2: int
        :return: None (if the points are disconnected) or the shortest distance
        between the two.
        :rtype: None or int
        """

        paths = [[node_1]]
        finished_traverses = []

        while len(paths):
            new_gen_paths = []
            for path in paths:
                for adj_node in self.graph[path[-1]]:
                    if adj_node == node_2:
                        finished_traverses.append(path + [adj_node])
                    elif adj_node not in path:
                        new_gen_paths.append(path + [adj_node])
            paths = new_gen_paths

        if not len(finished_traverses):
            return None

        return min([len(path) for path in finished_traverses])






