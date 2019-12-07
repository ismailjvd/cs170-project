import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
from gurobipy import *
import math

from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

# doesn't actually create an output file
# returns the path and dropoff mapping if everyone walked, the car didn't drive anywhere
def create_output_file_everyone_walks(G, H, s):
    n = len(G.nodes())
    path = [s]
    dropoff_mapping = {s: list(H)}
    path_to_file = './' + str(n) + '.out'
    list_locations = [str(i) for i in range(n)]
    return path, dropoff_mapping

# given a graph G, a set of dropoff locations D (usually homes), and a starting index s
# find the shortest path in G that goes through all of the locations in D
def create_and_solve_lp(G, D, s):
    def subtourelim(model, where):
      if where == GRB.callback.MIPSOL:
        selected = []
        # make a list of edges selected in the solution
        for i in range(n):
          sol = model.cbGetSolution([model._vars[i,j] for j in range(n)])
          selected += [(i,j) for j in range(n) if sol[j] > 0.5]
        # find the shortest cycle in the selected edge list
        tour = subtour(selected)
        if len(tour) < n:
          # add a subtour elimination constraint
          expr = 0
          for i in range(len(tour)):
            for j in range(i+1, len(tour)):
              expr += model._vars[tour[i], tour[j]]
          model.cbLazy(expr <= len(tour)-1)

    def subtour(edges):
      visited = [False]*n
      cycles = []
      lengths = []
      selected = [[] for i in range(n)]
      for x,y in edges:
        selected[x].append(y)
      while True:
        current = visited.index(False)
        thiscycle = [current]
        while True:
          visited[current] = True
          neighbors = [x for x in selected[current] if not visited[x]]
          if len(neighbors) == 0:
            break
          current = neighbors[0]
          thiscycle.append(current)
        cycles.append(thiscycle)
        lengths.append(len(thiscycle))
        if sum(lengths) == n:
          break
      return cycles[lengths.index(min(lengths))]

    m = Model()
    # Create a new "graph" with only home nodes and shortest distance between home nodes
    new_nodes = list(D)
    new_nodes.append(s)
    new_nodes.sort()
    new_edges = {}
    n = len(new_nodes)
    for i in range(n):
        for j in range(i + 1):
            node_1, node_2 = new_nodes[i], new_nodes[j]
            path = nx.shortest_path(G, source=node_1, target=node_2, weight='weight')
            path_length = nx.shortest_path_length(G, source=node_1, target=node_2, weight='weight')
            key = str(node_1) + "," + str(node_2)
            new_edges[key] = (path, path_length)
    #for edge, path_info in new_edges.items():
    #    print(edge + ": " + str(path_info))

    # Create binary variables for each edge
    vars = {}
    for i in range(n):
        for j in range(i + 1):
            node_1, node_2 = str(new_nodes[i]), str(new_nodes[j])
            key = node_1 + "," + node_2
            vars[i,j] = m.addVar(obj=new_edges[key][1], vtype = GRB.BINARY, name="e"+node_1+"_"+node_2)
            vars[j,i] = vars[i,j]
    m.update()
    # add constraint for no repeated vertices
    for i in range(n):
        m.addConstr(quicksum(vars[i,j] for j in range(n)) == 2)
        vars[i,i].ub = 0
    m.update()
    # perform LP operations
    m._vars = vars
    m.params.LazyConstraints = 1
    print(n)
    m.optimize(subtourelim)
    solution = m.getAttr('x', vars)
    selected = [(i,j) for i in range(n) for j in range(n) if solution[i,j] > 0.5]
    # Retrieve path in G from edges in selected
    adj_edge_list = {v:[] for v in new_nodes}
    for edge in selected:
        index_1, index_2 = edge[0], edge[1]
        u, v = new_nodes[index_1], new_nodes[index_2]
        adj_edge_list[u].append(v)
#    for key, value in adj_edge_list.items():
#        print(str(key) + ": " + str(value))

    def dfs(u):
        visited_set.add(u)
        visited_list.append(u)
        for v in adj_edge_list[u]:
            if v not in visited_set:
                dfs(v)

    visited_set = set()
    visited_list = []
    dfs(s)
    visited_list.append(s)
    tsp_path = get_edges_from_path(visited_list)

    total_path = [s]
    for e in tsp_path:
        node_1, node_2 = e
        key = str(node_1) + "," + str(node_2)
        if key in new_edges:
            path_1_to_2 = new_edges[key][0][1:]
            total_path.extend(path_1_to_2)
        else:
            key = str(node_2) + "," + str(node_1)
            path_1_to_2 = new_edges[key][0][:-1]
            path_2_to_1 = path_1_to_2[::-1]
            total_path.extend(path_2_to_1)

    return total_path

# Takes a car cycle and a list of homes and and iteratively removes two-cycles if it improves cost
def remove_two_cycles_from_path(path, homes, G):
    new_path = path
    dropoff_locations = {h: [h] for h in homes}
    best_solution = (list(new_path), dict(dropoff_locations))
    change_needed = True
    # outer loop decides if it should keep iterating to remove two cycles
    while change_needed and len(new_path) > 2:
        change_needed = False
        path = new_path
        new_path = []
        i = 0
        # inner loop removes all two-cycles for this iteration
        while i < len(path) - 2:
            if path[i] == path[i + 2] and len(dropoff_locations[path[i+1]]) == 1:
                change_needed = True
                home = path[i + 1]
                if path[i] in dropoff_locations:
                    dropoff_locations[path[i]] = dropoff_locations[path[i]] + list(dropoff_locations[home])
                    del dropoff_locations[home]
                else:
                    dropoff_locations[path[i]] = list(dropoff_locations[home])
                    del dropoff_locations[home]
                if not new_path or new_path[-1] != path[i]:
                    new_path.append(path[i])
                i += 2
            else:
                if not new_path or new_path[-1] != path[i]:
                    new_path.append(path[i])
                i += 1
        if len(path) >= 3 and path[-1] != path[-3]:
            if new_path[-1] != path[-2]:
                new_path.append(path[-2])
            new_path.append(path[-1])
        print(path)
        print(new_path)
        # we will no longer continue if the keeping the two-cycles on the previous iteration is more efficient
        cost_a, message = cost_of_solution(G, new_path, dropoff_locations)
        # print(cost_a)
        cost_b, message = cost_of_solution(G, best_solution[0], best_solution[1])
        # print(cost_b)
        # optional print statements
        # print(new_path)
        # for location, dropoffs in dropoff_locations.items():
        #   print(str(location) + ": " + str(dropoffs))
        # isInstance check added in case there's an edge case that's incorrect, causes cost_a to be infinity
        if isinstance(cost_a, str) or cost_a >= cost_b:
            return best_solution
        best_solution = list(new_path), dict(dropoff_locations)

    return new_path, dropoff_locations


# uses ILP to find the shortest cost of visiting all homes, and then removes all two cycles
def efficient_drive_all_homes(G, H, s):

    total_path = create_and_solve_lp(G, H, s)
    new_path, dropoff_locations = remove_two_cycles_from_path(total_path, H, G)

    dropoff_set = set(dropoff_locations.keys())
    # new_path = create_and_solve_lp(G, dropoff_set, s)

    # create output file
    n = len(G.nodes())
    # new_path = total_path
    path = new_path
    # dropoff_locations = {h: [h] for h in H}
    dropoff_mapping = dropoff_locations
    path_to_file = './' + str(n) + '.out'
    list_locations = [str(i) for i in range(n)]
    return path, dropoff_mapping
    # convertToFile(path, dropoff_mapping, path_to_file, list_locations)

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """
    G, message = adjacency_matrix_to_graph(adjacency_matrix)
    location_to_index = {list_of_locations[i]: i for i in range(len(list_of_locations))}
    home_indices = set()
    for home in list_of_homes:
        home_indices.add(location_to_index[home])
    start_index = location_to_index[starting_car_location]
    try:
        path, dropoffs = efficient_drive_all_homes(G, home_indices, start_index)
        cost_a, message_a = cost_of_solution(G, path, dropoffs)
        naive_path, naive_dropoffs = create_output_file_everyone_walks(G, home_indices, start_index)
        cost_b, message_b = cost_of_solution(G, naive_path, naive_dropoffs)
        if cost_b < cost_a:
            path, dropoffs, message_a = naive_path, naive_dropoffs, message_b
        print(message_a)
    except:
        print("An exception occurred with params {}".format(str(params)))
        path, dropoffs = create_output_file_everyone_walks(G, home_indices, start_index)
    return path, dropoffs
    pass

"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        """
        basename, filename = os.path.split(input_file)
        if '200' in filename:
            continue
        group, type = filename.split("_")
        if int(group) >= 10:
            continue
        """
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
