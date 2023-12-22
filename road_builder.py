
import numpy as np
import numpy.typing as npt

class Graph:
    adjacency_matrix: npt.NDArray[np.int64]

    vert_num: int

    def __init__(self, num: int):
        self.adjacency_matrix = np.zeros(shape=(num, num))
        self.vert_num = num

    # used when using edmonds-karp algorithm on graphs where for each edge there is no reversed edge
    # isn't used in example below because reversed edges are set manually
    def update_reversed_edges(self):
        for i in range(self.vert_num):
            for j in range(self.vert_num):
                if self.adjacency_matrix[i, j] > 0:
                    self.adjacency_matrix[j, i] = -self.adjacency_matrix[i, j]

    def get_source(self) -> int:
        for i in range(self.vert_num):
            if not np.any(self.adjacency_matrix[:, i] > 0):
                return i
        
        raise Exception('Graph must contain source')

    def get_sink(self) -> int:
        for i in range(self.vert_num):
            if not np.any(self.adjacency_matrix[i, :] > 0):
                return i
            
        raise Exception('Graph must contain sink')
    
    def get_adjacent_verticies(self, i: int) -> list:
        adjacent_verticies = list()
        for j in range(self.vert_num):
            if self.adjacency_matrix[i, j] != 0:
                adjacent_verticies.append(j)
        
        return adjacent_verticies

# bfs algorithm to find some path from source to sink
def source_to_sink_bfs(g: Graph, source: int, sink: int) -> (bool, list):
    visited = [False] * g.vert_num
    queue = [source]
    visited[source] = True
    predecessors = [-1] * g.vert_num
    while len(queue) != 0:
        current = queue.pop(0)
        for adj_vert in g.get_adjacent_verticies(current):
            if visited[adj_vert] == False and g.adjacency_matrix[current, adj_vert] > 0:
                queue.append(adj_vert)
                visited[adj_vert] = True
                predecessors[adj_vert] = current
    
    return (visited[sink], predecessors)

from copy import deepcopy

# edmonds-karp algorithm
def edmonds_karp(g: Graph) -> int:
    g = deepcopy(g)
    flow = 0
    source = g.get_source()
    sink = g.get_sink()
    while (result := source_to_sink_bfs(g, source, sink))[0]:
        predecessors = result[1]
        vertex = sink
        path_flow = g.adjacency_matrix[predecessors[vertex], vertex]
        vertex = predecessors[vertex]
        while vertex != source:
            path_flow = min(path_flow, g.adjacency_matrix[predecessors[vertex], vertex])
            vertex = predecessors[vertex]
        
        vertex = sink
        while vertex != source:
            g.adjacency_matrix[predecessors[vertex], vertex] -= path_flow
            g.adjacency_matrix[vertex, predecessors[vertex]] += path_flow
            vertex = predecessors[vertex]
        
        flow += path_flow
    
    return flow

# example
g = Graph(12)
g.adjacency_matrix[0, 1] = 34
g.adjacency_matrix[0, 3] = 4
g.adjacency_matrix[0, 2] = 11
g.adjacency_matrix[1, 3] = 18
g.adjacency_matrix[1, 2] = 15
g.adjacency_matrix[2, 1] = 12
g.adjacency_matrix[2, 4] = 6
g.adjacency_matrix[2, 5] = 10
g.adjacency_matrix[3, 4] = 24
g.adjacency_matrix[4, 5] = 10
g.adjacency_matrix[4, 6] = 22
g.adjacency_matrix[5, 3] = 8
g.adjacency_matrix[5, 7] = 24
g.adjacency_matrix[6, 8] = 16
g.adjacency_matrix[7, 8] = 9
g.adjacency_matrix[7, 10] = 38
g.adjacency_matrix[8, 7] = 13
g.adjacency_matrix[8, 9] = 31
g.adjacency_matrix[9, 10] = 3
g.adjacency_matrix[9, 11] = 17
g.adjacency_matrix[10, 9] = 7
g.adjacency_matrix[10, 11] = 28

from itertools import combinations

# computes maximum flow with adding new roads listed in combination
def compute_flow_with_roads(g: Graph, roads: dict, combination: tuple):
    for r in combination:
        g.adjacency_matrix[*r] = roads[r]

    flow = edmonds_karp(g)
    for r in combination:
        g.adjacency_matrix[*r] = 0    
    
    return flow

# roads proposed
roads = {(5, 8): 8, (2, 7): 5, (4, 8): 9}

# we can build only 2 roads
roads_to_build = 2
roads_combs = list(combinations(roads.keys(), roads_to_build))

print(f'Maximum flow in a road network without building new roads: {edmonds_karp(g)}')

# dictionary for new roads combinations and corresponding flows 
flows = dict()
for comb in roads_combs:
    print('-'*30)
    flows.update({comb:compute_flow_with_roads(g, roads, comb)})
    print(f'When building roads: {comb}')
    print(f'Maximum flow in a road network with new roads above: {flows[comb]}')

# finding maximum flow and corresponding roads combinations
max_flow = max(list(flows.values()))
max_combinations = list()
for comb in flows.keys():
    if flows[comb] == max_flow:
        max_combinations.append(comb)

print('-'*30)

print(f'Maximum flow is: {max_flow}')
print(f'It is achieved with such roads combination{'s' if len(max_combinations) > 1 else ''}:')
for comb in max_combinations:
    print(comb)