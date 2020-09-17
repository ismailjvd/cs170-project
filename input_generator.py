import networkx as nx
import numpy as np
# import matplotlib.pyplot as plt
from gurobipy import *
import math
import random
from student_utils import *
from solver import *
from input_validator import *
from output_validator import *

# Variables for Input Graph Creation
BETA = 0.3 # 0.3, 0.15
ALPHA = 0.8 # 0.8, 0.4
SEED = 20

random.seed(SEED)

# Returns a metric graph using the Waxman Graph library
def generate_graph(n, max_weight, a, b):
    return nx.generators.geometric.waxman_graph(n, L=max_weight, beta=b, alpha=a, domain=(0,0,max_weight,max_weight), seed=SEED)

# Creates visual of graph. Same type as Graph returned by Waxman graph.
# Input:
# Graph g(g.nodes = (node_num, {pos: x, y}), g.edges = (node_1, node_2, {}))
# String or List[String]: String if all edges are the same color, otherwise list of strings with len(edges) == len(edge_colors)
# String or List[String]: String if all nodes are the same color, otherwise list of strings with len(nodes) == len(node_colors)
def visualize_graph(g, edge_colors, node_colors):
    node_positions = {node[0]: (node[1]['pos'][0], node[1]['pos'][1]) for node in g.nodes(data=True)}
   # plt.figure(figsize=(12, 9))
    nx.draw(g, pos=node_positions, node_size=20, node_color=node_colors, edge_color=edge_colors)
   # plt.title('Graph Representation', size=15)
   # plt.show()

# Returns adjacency matrix from with Graph type returned by Waxman graph
# Input: 
# Graph g(g.nodes = (node_num, {pos: x, y}), g.edges = (node_1, node_2, {}))
# Output: 
# List[List[String/Float]] 
def create_adjacency_matrix(G):
    n = G.number_of_nodes()
    adjacency_matrix = [['x']*n for i in range(n)]
    node_dict = {node[0]: node[1]['pos'] for node in G.nodes(data=True)}
    for e in G.edges(data=True):
        node_1, node_2 = e[0], e[1]
        pos_1, pos_2 = node_dict[node_1], node_dict[node_2]
        d = round(distance(pos_1, pos_2), 3)
        adjacency_matrix[node_1][node_2] = d
        adjacency_matrix[node_2][node_1] = d
    return adjacency_matrix

# Returns the Euclidean distance (also weight for metric graphs) between two nodes
# node = (node_num, {pos: x, y})
def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# Checks if graph created using student_util adjacency_matrix conversion is valid (metric, connected)
def check_graph_valid(g):
    if not is_metric(g):
        raise AssertionError("ERROR: G is not metric!")
    # number of connected components should be equal to number of nodes
    # assert(len(list(nx.connected_components(g))) == 1, "G has multiple components: " + 
    #           str([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]))

""" GRAPH GENERATORS """

# Hardcoded the edges in this graph, have better solution for hundrend and two hundred graphs
# Returns Waxman Graph and list of Home Nodes (G and H)
def create_fifty_graph():
    # Seed 20, BETA = 0.3, ALPHA = 0.8, 30 vertices
    # 4: (572, 169)
    # 28: (344.777444184001, 116.34846707287849)
    # 2: (259.82744748897693, 635.7258696059891)
    # 26: (863.17055762585, 689.8028785376858)
    # 25: (164.6715879870595, 928.834220072079)
    #  1: (766.5092563626442, 904.6162378132736)
    # 24: (555.6893112719441, 432.9010185355702) 
    # 6: (103.24779991117994, 319.1391488492897)
    # 9: (908.6358448961128, 335.5688104684739)
    # 20: (644.8584468489341, 300.7460778906799)
    # 7: (950.0391079535002, 449.4007558523254)

    # 22 (599.7335517502819, 447.3885514326741)
    # 30 (884.573, 471.932)
    # 31 (624.592, 399.580)
    # 32 (666.655, 445.231)
    # 33 (969.346, 441.921)
    # 34 (662, 862)
    # 35 (808, 922)
    # 1
    # 36 (129, 840)
    # 37 (817, 727)
    # 38 (687, 658)
    # 39 (731, 607)
    # 40 (211, 564)
    # 41 (259, 548)
    # 42 (307, 578)
    # 43 (310, 250)
    # 44 (102, 230)
    # 45 (88, 386)
    # 46 (34, 298)
    # 47 (180, 870)
    # 48 (469, 203)
    # 49 (609, 230)
    G = generate_graph(30, 1000, 0.3, 0.8)
    start_vertex = 4
    G.add_node(30, pos = (884.573, 471.932))
    G.add_node(31, pos = (624.592, 399.580))
    G.add_node(32, pos = (666.655, 445.231))
    G.add_node(33, pos = (969.346, 441.921))
    G.add_node(34, pos = (662.642, 862.125))
    G.add_node(35, pos = (808.527, 922.176))
    G.add_node(36, pos = (129.843, 840.542))
    G.add_node(37, pos = (817.632, 727.665))
    G.add_node(38, pos = (687.333, 658.533))
    G.add_node(39, pos = (731.743, 607.122))
    G.add_node(40, pos = (211.754, 564.532))
    G.add_node(41, pos = (259.153, 548.580))
    G.add_node(42, pos = (307.655, 578.231))
    G.add_node(43, pos = (310.346, 250.921))
    G.add_node(44, pos = (102.642, 230.125))
    G.add_node(45, pos = (88.527, 386.176))
    G.add_node(46, pos = (34.843, 298.542))
    G.add_node(47, pos = (180.632, 870.665))
    G.add_node(48, pos = (469.333, 203.533))
    G.add_node(49, pos = (609.743, 230.122))
    G.add_edge(7, 30)
    G.add_edge(7, 33)
    G.add_edge(21, 31)
    G.add_edge(21, 32)
    G.add_edge(1, 34)
    G.add_edge(1, 35)
    G.add_edge(34, 35)
    G.add_edge(25, 36)
    G.add_edge(25, 47)
    G.add_edge(26, 37)
    G.add_edge(37, 38)
    G.add_edge(38, 39)
    G.add_edge(39, 26)
    G.add_edge(2, 40)
    G.add_edge(2, 41)
    G.add_edge(2, 42)
    G.add_edge(28, 43)
    G.add_edge(6, 44)
    G.add_edge(44, 46)
    G.add_edge(45, 46)
    G.add_edge(4, 48)
    G.add_edge(4, 49)
    nodes_in_path = {
        4, 28, 2, 26, 25, 1, 25, 24, 6, 9, 7
        }
    node_list = list(G.nodes(data=True))
    home_nodes = {1, 9, 22}
    for node in node_list[30:]:
        home_nodes.add(node[0])
    for node in node_list[30:]:
        count = 0
        edges_to_add = random.randint(2,4)
        while count < edges_to_add:
            v = node_list[random.randint(0,29)]
            if (v[0] not in nodes_in_path and (node[1]['pos'][0] < 500 and v[1]['pos'][0] >= 500)
                or (node[1]['pos'][0] >= 500 and v[1]['pos'][0] < 500)):
                G.add_edge(v[0], node[0])
                count += 1
    edges_in_path = {
        '4,28',
        '2,28',
        '2,26',
        '25,26',
        '1,25',
        '1,24',
        '6,24',
        '6,9',
        '7,9',
        '4,7'
        }
    node_colors = ['black' if v[0] not in home_nodes else 'red' for v in G.nodes(data=True)]
    edge_colors = ['red' if str(e[0]) + ',' + str(e[1]) in edges_in_path else 'blue' for e in G.edges(data=True)]
    # for e in G.edges(data=True):
        # print(e)
    # visualize_graph(G, edge_colors, node_colors)
    return G, home_nodes, start_vertex

def create_neighbors(pos, n, size):
    max_range = 0.1*size
    result = []
    for i in range(n):
        x_pos = random.uniform(pos[0] - max_range, pos[0] + max_range)
        while x_pos < 0 or x_pos > size:
            x_pos = random.uniform(pos[0] - max_range, pos[0] + max_range)
        y_pos = random.uniform(pos[1] - max_range, pos[1] + max_range)
        while y_pos < 0 or y_pos > size:
            y_pos = random.uniform(pos[1] - max_range, pos[1] + max_range)
        result.append((x_pos, y_pos))
    return result


def create_hundred_graph():
    G = generate_graph(50, 10000, 0.2, 0.5)
    position_dict = {node[0]: node[1]['pos'] for node in G.nodes(data=True)}
    cycle_list = nx.algorithms.cycles.cycle_basis(G)
    largest_cycle = max(cycle_list, key=len)
    print(list(largest_cycle))
    # visualizing purposes
    cycle_edges = get_edges_from_path(largest_cycle + [largest_cycle[0]])
    edge_set = set()
    for e in cycle_edges:
        u, v = min(e), max(e)
        edge_set.add(str(u) + ',' + str(v))
    edge_colors = ['red' if str(e[0]) + ',' + str(e[1]) in edge_set else 'blue' for e in G.edges(data=True)]
    # visualize_graph(G, edge_colors, 'black')

    home_nodes = set()
    neighbor_dict = {v: [] for v in largest_cycle}
    # create two neighbors for each nodes in the path and attach them to the node
    for v in largest_cycle:
        node_num_1, node_num_2 = len(G.nodes()), len(G.nodes()) + 1
        node_pos_1, node_pos_2 = create_neighbors(position_dict[v], 2, 10000)
        home_nodes.add(node_num_1)
        home_nodes.add(node_num_2)
        neighbor_dict[v].extend([(node_num_1, node_pos_1), (node_num_2, node_pos_2)])
        G.add_node(node_num_1, pos = node_pos_1)
        G.add_node(node_num_2, pos = node_pos_2)
        G.add_edge(v, node_num_1)
        G.add_edge(v, node_num_2)
        # 1/2 probability that these nodes will have an edge between each other
        add_edge_between = random.randint(0, 1)
        if add_edge_between == 1:
            G.add_edge(node_num_1, node_num_2)
    # create 1-2 neighbors for each nodes in the path and attach them to the existing homes
    for v in largest_cycle:
        if len(G.nodes()) >= 99:
            break
        num_neighbors = random.randint(1,2)
        if num_neighbors == 1:
            node_num = len(G.nodes())
            node_pos = create_neighbors(position_dict[v], 1, 10000)[0]
            home_nodes.add(node_num)
            neighbor_dict[v].extend([(node_num, node_pos)])
            G.add_node(node_num, pos = node_pos)
            add_edge_to_v = random.randint(0, 1)
            if add_edge_to_v == 1:
                G.add_edge(v, node_num)
            else:
                G.add_edge(neighbor_dict[v][0][0], node_num)
                G.add_edge(neighbor_dict[v][1][0], node_num)
        else:
            node_num_1, node_num_2 = len(G.nodes()), len(G.nodes()) + 1
            node_pos_1, node_pos_2 = create_neighbors(position_dict[v], 2, 10000)
            home_nodes.add(node_num_1)
            home_nodes.add(node_num_2)
            neighbor_dict[v].extend([(node_num_1, node_pos_1), (node_num_2, node_pos_2)])
            G.add_node(node_num_1, pos = node_pos_1)
            G.add_node(node_num_2, pos = node_pos_2)
            G.add_edge(neighbor_dict[v][0][0], node_num_1)
            G.add_edge(neighbor_dict[v][1][0], node_num_2)
            # 1/2 probability that these nodes will have an edge between each other
            add_edge_between = random.randint(0, 1)
            if add_edge_between == 1:
                G.add_edge(node_num_1, node_num_2)
    node_colors = ['black' if v[0] not in home_nodes else 'red' for v in G.nodes(data=True)]
    # visualize_graph(G, edge_colors, node_colors)
    # attach the homes to other nodes that are not in the cycle
    nodes_not_in_path = set(range(50)) - set(largest_cycle)
    for v, homes in neighbor_dict.items():
        for h in homes:
            num_edges = random.randint(2,6)
            new_neighbors = random.choices(list(nodes_not_in_path), k=num_edges)
            for neighbor in new_neighbors:
                G.add_edge(neighbor, h[0])
    print(len(home_nodes))
    # visualize_graph(G, edge_colors, node_colors)
    start_vertex = 10
    if len(G.nodes()) == 99:
        G.add_node(99, pos=(0,0))
        G.add_edge(0, 99)

    # visualize_graph(G, edge_colors, node_colors)
    return G, home_nodes, start_vertex

def create_two_hundred_graph():
    G = generate_graph(100, 100000, 0.15, 0.4)
    position_dict = {node[0]: node[1]['pos'] for node in G.nodes(data=True)}
    cycle_list = nx.algorithms.cycles.cycle_basis(G)
    largest_cycle = max(cycle_list, key=len)
    print(list(largest_cycle))
    # visualizing purposes
    cycle_edges = get_edges_from_path(largest_cycle + [largest_cycle[0]])
    edge_set = set()
    for e in cycle_edges:
        u, v = min(e), max(e)
        edge_set.add(str(u) + ',' + str(v))
    edge_colors = ['red' if str(e[0]) + ',' + str(e[1]) in edge_set else 'blue' for e in G.edges(data=True)]
    # visualize_graph(G, edge_colors, 'black')

    home_nodes = set()
    neighbor_dict = {v: [] for v in largest_cycle}
    # create three neighbors for each nodes in the path and attach them to the node
    for v in largest_cycle:
        node_num_1, node_num_2, node_num_3 = len(G.nodes()), len(G.nodes()) + 1, len(G.nodes())+ 2
        node_pos_1, node_pos_2, node_pos_3 = create_neighbors(position_dict[v], 3, 100000)
        home_nodes.add(node_num_1)
        home_nodes.add(node_num_2)
        home_nodes.add(node_num_3)
        neighbor_dict[v].extend([(node_num_1, node_pos_1), (node_num_2, node_pos_2), (node_num_3, node_pos_3)])
        G.add_node(node_num_1, pos = node_pos_1)
        G.add_node(node_num_2, pos = node_pos_2)
        G.add_node(node_num_3, pos = node_pos_3)
        G.add_edge(v, node_num_1)
        G.add_edge(v, node_num_2)
        G.add_edge(v, node_num_3)
        # 1/2 probability that these nodes will have an edge between each other
        add_edge_between = random.randint(0, 1)
        if add_edge_between == 1:
            G.add_edge(node_num_1, node_num_2)
        elif add_edge_between == 2:
            G.add_edge(node_num_1, node_num_3)
    # create 1-2 neighbors for each nodes in the path and attach them to the existing homes
    for v in largest_cycle:
        if len(G.nodes()) >= 199:
            break
        node_num_1, node_num_2 = len(G.nodes()), len(G.nodes()) + 1
        node_pos_1, node_pos_2 = create_neighbors(position_dict[v], 2, 100000)
        home_nodes.add(node_num_1)
        home_nodes.add(node_num_2)
        neighbor_dict[v].extend([(node_num_1, node_pos_1), (node_num_2, node_pos_2)])
        G.add_node(node_num_1, pos = node_pos_1)
        G.add_node(node_num_2, pos = node_pos_2)
        G.add_edge(neighbor_dict[v][0][0], node_num_1)
        G.add_edge(neighbor_dict[v][1][0], node_num_2)
        # 1/2 probability that these nodes will have an edge between each other
        add_edge_between = random.randint(0, 1)
        if add_edge_between == 1:
            G.add_edge(node_num_1, node_num_2)
    node_colors = ['black' if v[0] not in home_nodes else 'red' for v in G.nodes(data=True)]
    # visualize_graph(G, edge_colors, node_colors)
    # attach the homes to other nodes that are not in the cycle
    nodes_not_in_path = set(range(50)) - set(largest_cycle)
    for v, homes in neighbor_dict.items():
        for h in homes:
            num_edges = random.randint(2,6)
            new_neighbors = random.choices(list(nodes_not_in_path), k=num_edges)
            for neighbor in new_neighbors:
                G.add_edge(neighbor, h[0])
    print(len(home_nodes))
    # visualize_graph(G, edge_colors, node_colors)
    start_vertex = random.choice(largest_cycle)
    if len(G.nodes()) == 199:
        G.add_node(199, pos=(0,0))
        G.add_edge(0, 199)
    return G, home_nodes, start_vertex

"""
location_list (List[String]): list of locations
home_list (List[String]): list of homes
start (int): start index in locations
matrix: (List[List[String/Float]]): adjacency matrix
"""
def convertToInputFile(location_list, home_list, start, matrix, path_to_file):
    string = ''
    string += str(len(location_list)) + '\n' + str(len(home_list)) + '\n'
    
    for node in location_list:
        string += node + ' '
    string = string.strip()
    string += '\n'

    for node in home_list:
        string += node + ' '
    string = string.strip()
    string += '\n'

    string += location_list[start] + '\n'

    for row in matrix:
        # assert len(row) == 50, "Adjacency matrix has incorrect row len"
        for col in range(len(row)):
            string += str(row[col]) + ' '
        string = string.strip()
        string += '\n'
    # print(string)
    utils.write_to_file(path_to_file, string)

"""
FILE CREATION
"""

def create_input_file(G, H, s, adj_matrix):
    location_list = [str(node) for node in G.nodes()]
    home_list = [str(h) for h in H]
    path_to_file = './' + str(len(location_list)) + '.in'
    convertToInputFile(location_list, home_list, s, adj_matrix, path_to_file)

"""
FUNCTIONS FOR USER TO CALL
"""
    
def create_fifty_files():
    G, H, s = create_fifty_graph()
    matrix = create_adjacency_matrix(G)
    G, message = adjacency_matrix_to_graph(matrix)
    # Plot the new graph (doesn't follow metric anymore)
    # nx.draw(G, node_size=20)
    # plt.show()
    # Ensure that the graph created is valid
    # check_graph_valid(G)
    # shortest_paths = nx.floyd_warshall(G)
    create_input_file(G, H, s, matrix)
    #create_path_mapping_everyone_walks(G, H, s)
    efficient_drive_all_homes(G, H, s)
    #validate_input('./50.in')
    validate_output('./50.in', './50.out')

def create_hundred_files():
    G, H, s = create_hundred_graph()
    matrix = create_adjacency_matrix(G)
    G, message = adjacency_matrix_to_graph(matrix)
    create_input_file(G, H, s, matrix)
    # create_path_mapping_everyone_walks(G, H, s)
    efficient_drive_all_homes(G, H, s)
    # validate_input('./100.in')
    validate_output('./100.in', './100.out')

def create_two_hundred_files():
    G, H, s = create_two_hundred_graph()
    matrix = create_adjacency_matrix(G)
    G, message = adjacency_matrix_to_graph(matrix)
    create_input_file(G, H, s, matrix)
    create_path_mapping_everyone_walks(G, H, s)
    # efficient_drive_all_homes(G, H, s)
    # validate_input('./200.in')
    validate_output('./200.in', './200.out')

# create_fifty_files()
# create_hundred_files()
# create_two_hundred_files()  




