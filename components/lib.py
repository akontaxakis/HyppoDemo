import inspect
import os
import pickle
import random
import hashlib
import re
import time
import webbrowser

from IPython.display import Image, display

from sklearn.metrics import silhouette_score
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from components.augmenter import map_node

def load_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="graphs", mode="eq_"):
    os.makedirs(graph_dir, exist_ok=True)
    file_name = uid + "_AG_" + str(sum) + "_" + mode + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.pkl")
    if os.path.exists(ag_path):
        with open(ag_path, 'rb') as f:
            print("load " + ag_path)
            artifact_graph = pickle.load(f)
    return artifact_graph


def store_EDGES_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="graphs"):
    file_name = uid + "_EDGES_AG_" + str(sum) + "_" + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.txt")
    with open(ag_path, "w") as outfile:
        # Iterate over edges and write to file
        for u, v, data in artifact_graph.edges(data=True):
            cost = data['weight']
            cost_2 = data['weight']
            # cost_3 = data['memory_usage']
            outfile.write(f'"{u}","{v}",{cost},{cost_2}\n')
            # outfile.write(f'"{u}","{v}",{cost},{cost_2},{cost_3}\n')
    return artifact_graph


def store_or_load_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="graphs"):
    os.makedirs(graph_dir, exist_ok=True)
    file_name = uid + "_AG_" + str(sum) + "_" + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.pkl")
    if os.path.exists(ag_path):
        with open(ag_path, 'rb') as f:
            print("load " + ag_path)
            artifact_graph = pickle.load(f)
    else:
        with open(ag_path, 'wb') as f:
            pickle.dump(artifact_graph, f)

    file_name = uid + "_EDGES_AG_" + str(sum) + "_" + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.txt")

    with open(ag_path, "w") as outfile:
        # Iterate over edges and write to file
        for u, v, data in artifact_graph.edges(data=True):
            cost = data['cost']
            outfile.write(f'"{u}","{v}",{cost}\n')
    return artifact_graph


def create_artifact_graph(artifacts):
    G = nx.DiGraph()
    for i, (step_name, artifact) in enumerate(artifacts.items()):
        G.add_node(step_name, artifact=artifact)
        if i > 0:
            prev_step_name = list(artifacts.keys())[i - 1]
            G.add_edge(prev_step_name, step_name)
    return G


def plot_artifact_graph(G, uid, type):
    plt.figure(figsize=(20, 18))
    pos = nx.drawing.layout.spring_layout(G, seed=620, scale=4)
    nx.draw(G, pos, with_labels=True, node_size=120, node_color="skyblue", font_size=5)
    folder_path = "plots/"
    file_path = os.path.join(folder_path, uid + "_" + type + "_plot.pdf")
    plt.savefig(file_path)
    plt.show()


def load_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="graphs", mode="eq_"):
    os.makedirs(graph_dir, exist_ok=True)
    file_name = uid + "_AG_" + str(sum) + "_" + mode + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.pkl")
    if os.path.exists(ag_path):
        with open(ag_path, 'rb') as f:
            print("load " + ag_path)
            artifact_graph = pickle.load(f)
    return artifact_graph


def extract_artifact_graph(artifact_graph, graph_dir, uid):
    shared_graph_file = uid + "_shared_graph"
    shared_graph_path = os.path.join(graph_dir, f"{shared_graph_file}.plk")
    if os.path.exists(shared_graph_path):
        with open(shared_graph_path, 'rb') as f:
            print("load" + shared_graph_path)
            artifact_graph = pickle.load(f)
    return artifact_graph, shared_graph_path


def required_artifact(new_tasks):
    source_nodes = {edge[0] for edge in new_tasks}


def pretty_graph_drawing(G):
    graph_size = 5
    pos = nx.spring_layout(G)
    pos_1 = nx.spring_layout(G)
    depth = G.number_of_nodes()
    for node in G.nodes:
        G.nodes[node]['depth'] = None

    # Compute and set depth for each node
    compute_depth(G, 'source')

    for node_id in G.nodes:
        if node_id == 'source':
            # pos[node_id] = np.array([0, depth - G.nodes[node_id]['depth']])
            pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), 0])
            G.nodes[node_id]['color'] = 'red'
            G.nodes[node_id]['size'] = 100

        elif G.nodes[node_id]['type'] == 'split':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'blue'
            G.nodes[node_id]['size'] = 10

        elif G.nodes[node_id]['type'] == 'super':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'blue'
            G.nodes[node_id]['size'] = 10


        elif G.nodes[node_id]['type'] == 'fitted_operator':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'green'
            G.nodes[node_id]['size'] = 100

        else:
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'purple'
            G.nodes[node_id]['size'] = 100

    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    print(pos)
    nx.draw(G, pos=pos, with_labels=True, font_size=2, node_size=node_sizes, node_color=node_colors)
    plt.figure(figsize=(100, 100))

    plt.savefig("output.pdf", format="pdf")
    plt.show()


## graphviz styles can be found here: https://graphviz.org/docs/attr-types/style/
def graphviz_draw_with_requests_and_new_tasks(K, mode, requested_nodes, new_tasks):
    G = K.copy()

    for node in G.nodes:
        G.nodes[node]['depth'] = None

    # Compute and set depth for each node
    compute_depth(G, 'source')
    blue_nodes = []
    if mode == "eq":
        eq_requested_nodes = []
        for node in requested_nodes:
            eq_requested_nodes.append(map_node(node, "no_fit"))
        requested_nodes = eq_requested_nodes

    disconnected_nodes, disconnected_edges = find_disconnected_nodes_edges(G, requested_nodes)
    for node_id in G.nodes:
        if node_id == 'source':
            # pos[node_id] = np.array([0, depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), 0])
            G.nodes[node_id]['color'] = 'red'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        elif G.nodes[node_id]['type'] == 'super' or G.nodes[node_id]['type'] == 'split':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'blue'
            G.nodes[node_id]['edgecolors'] = 'blue'
            G.nodes[node_id]['shape'] = 'point'
            G.nodes[node_id]['width'] = 0.1
            blue_nodes.append(node_id)

        elif G.nodes[node_id]['type'] == 'fitted_operator':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'green'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        else:
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'purple'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'
        if node_id in requested_nodes:
            G.nodes[node_id]['color'] = 'black'
            G.nodes[node_id]['shape'] = 'ellipse'
        if node_id in disconnected_nodes:
            G.nodes[node_id]['style'] = "dotted"
        else:
            G.nodes[node_id]['style'] = "bold"

    labels = {node: "" if node in blue_nodes else str(node) for node in G.nodes()}
    for node, label in labels.items():
        G.nodes[node]['label'] = label

    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr['rankdir'] = 'LR'
    for edge in A.edges():
        edge.attr['label'] = int(G[edge[0]][edge[1]]['weight'] * 1000)
        if edge in disconnected_edges:
            edge.attr['style'] = "dotted"
        else:
            edge.attr['style'] = "bold"

    for u, v in new_tasks:
        if mode == "eq":
            A.add_edge(map_node(u, "no_fit"), map_node(v, "no_fit"))
        else:
            A.add_edge(u, v)

    # Legend
    # legend_label = "{ LEGEND | {Dotted Lines and Nodes |Pruned Elements} | {Black Ellipse|New Artifacts} | {Bold Black Ellipse|Requested Artifacts}}"
    #  A.add_node("Legend", shape="record", label=legend_label, rank='sink')

    # Ensure the legend is placed at the bottom
    # A.add_subgraph(["Legend"], rank="sink", name="cluster_legend")

    # Save the graph to a file
    file_path = "graph_output.png"
    A.layout(prog='dot')
    A.draw(file_path)

    # Open the saved image file with the default viewer
    if os.name == 'posix':
        os.system(f'open {file_path}')
    elif os.name == 'nt':  # For Windows
        os.startfile(file_path)

    # nx.draw(G, pos=pos, with_labels=True, font_size=2,node_shape="r", node_size=node_sizes, node_color=node_colors)
    # plt.figure(figsize=(100, 100))

    # plt.savefig("output.pdf", format="pdf")
    # plt.show()


def graphviz_draw_with_requests(G, mode, requested_nodes):
    for node in G.nodes:
        G.nodes[node]['depth'] = None

    # Compute and set depth for each node
    compute_depth(G, 'source')
    blue_nodes = []
    if mode == "eq":
        eq_requested_nodes = []
        for node in requested_nodes:
            eq_requested_nodes.append(map_node(node, "no_fit"))
        requested_nodes = eq_requested_nodes

    disconnected_nodes, disconnected_edges = find_disconnected_nodes_edges(G, requested_nodes)
    for node_id in G.nodes:
        if node_id == 'source':
            # pos[node_id] = np.array([0, depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), 0])
            G.nodes[node_id]['color'] = 'red'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        elif G.nodes[node_id]['type'] == 'super' or G.nodes[node_id]['type'] == 'split':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'blue'
            G.nodes[node_id]['edgecolors'] = 'blue'
            G.nodes[node_id]['shape'] = 'point'
            G.nodes[node_id]['width'] = 0.1
            blue_nodes.append(node_id)

        elif G.nodes[node_id]['type'] == 'fitted_operator':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'green'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        else:
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'purple'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'
        if node_id in requested_nodes:
            G.nodes[node_id]['color'] = 'black'
            G.nodes[node_id]['shape'] = 'ellipse'
        if node_id in disconnected_nodes:
            G.nodes[node_id]['style'] = "dotted"
        else:
            G.nodes[node_id]['style'] = "bold"

    labels = {node: "" if node in blue_nodes else str(node) for node in G.nodes()}
    for node, label in labels.items():
        G.nodes[node]['label'] = label
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr['rankdir'] = 'LR'
    for edge in A.edges():
        edge.attr['label'] = int(G[edge[0]][edge[1]]['weight'] * 10000)
        if edge in disconnected_edges:
            edge.attr['style'] = "dotted"
        else:
            edge.attr['style'] = "bold"
    # Save the graph to a file
    file_path = "graph_output.png"
    A.layout(prog='dot')
    A.draw(file_path)

    # Open the saved image file with the default viewer
    if os.name == 'posix':
        os.system(f'open {file_path}')
    elif os.name == 'nt':  # For Windows
        os.startfile(file_path)

    # nx.draw(G, pos=pos, with_labels=True, font_size=2,node_shape="r", node_size=node_sizes, node_color=node_colors)
    # plt.figure(figsize=(100, 100))

    # plt.savefig("output.pdf", format="pdf")
    # plt.show()
def graphviz_simple_draw(G):
    # Compute and set depth for each node
    blue_nodes = []
    for node_id in G.nodes:
        if node_id == 'source':
            # pos[node_id] = np.array([0, depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), 0])
            G.nodes[node_id]['color'] = 'red'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        elif G.nodes[node_id]['type'] == 'super' or G.nodes[node_id]['type'] == 'split':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'blue'
            G.nodes[node_id]['edgecolors'] = 'blue'
            G.nodes[node_id]['shape'] = 'point'
            G.nodes[node_id]['width'] = 0.1
            blue_nodes.append(node_id)

        elif G.nodes[node_id]['type'] == 'fitted_operator':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'green'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        else:
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'purple'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

    labels = {node: "" if node in blue_nodes else str(G.nodes[node]['alias']) for node in G.nodes()}
    for node, label in labels.items():
        G.nodes[node]['label'] = label
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr['rankdir'] = 'LR'
    for edge in A.edges():
        edge.attr['label'] = int(G[edge[0]][edge[1]]['weight'] * 1000)
        edge.attr['style'] = "bold"
    # Save the graph to a file
    file_path = "graph_output.png"
    A.layout(prog='dot')
    png = A.draw(format='png')
    display(Image(png))
    # Open the saved image file with the default viewer
    #if os.name == 'posix':
    #    os.system(f'open {file_path}')
    #elif os.name == 'nt':  # For Windows
    #    os.startfile(file_path)

def graphviz_draw(G, mode='notebook'):

    # Compute and set depth for each node
    blue_nodes = []
    for node_id in G.nodes:
        if node_id == 'source':
            # pos[node_id] = np.array([0, depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), 0])
            G.nodes[node_id]['color'] = 'red'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        elif G.nodes[node_id]['type'] == 'super' or G.nodes[node_id]['type'] == 'split':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'blue'
            G.nodes[node_id]['edgecolors'] = 'blue'
            G.nodes[node_id]['shape'] = 'point'
            G.nodes[node_id]['width'] = 0.1
            blue_nodes.append(node_id)

        elif G.nodes[node_id]['type'] == 'fitted_operator':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'green'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        else:
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'purple'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

    labels = {node: "" if node in blue_nodes else str(node) for node in G.nodes()}
    for node, label in labels.items():
        G.nodes[node]['label'] = label
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr['rankdir'] = 'LR'
    for edge in A.edges():
        edge.attr['label'] = int(G[edge[0]][edge[1]]['weight'] * 1000)
        edge.attr['style'] = "bold"
    # Save the graph to a file
    A.layout(prog='dot')

    if(mode!='notebook'):
        file_path = 'graph.png'
        A.draw(file_path)
        webbrowser.open('file://' + os.path.realpath(file_path))

    else:
        png = A.draw(format='png')
        display(Image(png))
    # Open the saved image file with the default viewer
    #if os.name == 'posix':
    #    os.system(f'open {file_path}')
    #elif os.name == 'nt':  # For Windows
    #    os.startfile(file_path)


def compute_depth(graph, node, parent=None):
    if parent is None:
        depth = 0
    else:
        depth = graph.nodes[parent]['depth'] + 1
    graph.nodes[node]['depth'] = depth
    for neighbor in graph.neighbors(node):
        if neighbor != parent:
            compute_depth(graph, neighbor, node)


def find_disconnected_nodes_edges(G, targets):
    connected_nodes = set()
    for node in G.nodes():
        for target in targets:
            if nx.has_path(G, node, target):
                connected_nodes.add(node)
                break

    # Step 2: Identify edges connected to these nodes
    connected_edges = {(u, v) for u, v in G.edges() if v in connected_nodes}

    # Step 3: Return nodes and edges that aren't in the above sets
    disconnected_nodes = set(G.nodes()) - connected_nodes
    disconnected_edges = set(G.edges()) - connected_edges
    return disconnected_nodes, disconnected_edges

def keep_two_digits(number):
    str_number = str(number)
    index_of_decimal = str_number.index('.')
    str_number_no_round = str_number[:index_of_decimal + 2]
    return str_number_no_round

def compute_correlation(data1, data2):
    corr_matrix = np.corrcoef(data1, data2, rowvar=False)
    return np.average(np.abs(np.diag(corr_matrix, k=1)))

def compare_pickles_exact(artifact_dir='artifacts'):
    files = [f for f in os.listdir(artifact_dir) if f.endswith('.pkl')]
    num_files = len(files)
    equal_pairs = []

    for i in range(num_files):
        file1 = os.path.join(artifact_dir, files[i])

        with open(file1, 'rb') as f:
            data1 = pickle.load(f)

        for j in range(i + 1, num_files):
            file2 = os.path.join(artifact_dir, files[j])

            with open(file2, 'rb') as f:
                data2 = pickle.load(f)

            if np.array_equal(data1, data2):
                print("found a pair")
                equal_pairs.append((files[i], files[j]))

    return equal_pairs

def compare_pickles(artifact_dir='artifacts', correlation_threshold=0.9):
    files = [f for f in os.listdir(artifact_dir) if f.endswith('.pkl')]
    num_files = len(files)
    highly_correlated_pairs = []
    print(num_files)
    for i in range(num_files):
        file1 = os.path.join(artifact_dir, files[i])

        with open(file1, 'rb') as f:
            data1 = pickle.load(f)

        for j in range(i + 1, num_files):
            file2 = os.path.join(artifact_dir, files[j])

            with open(file2, 'rb') as f:
                data2 = pickle.load(f)

            correlation = compute_correlation(data1, data2)
            print(correlation)
            if correlation >= correlation_threshold:
                highly_correlated_pairs.append((files[i], files[j], correlation))

    return highly_correlated_pairs

def print_metrics(metrics_dir='metrics'):
    n_artifacts = 0;
    os.makedirs(metrics_dir, exist_ok=True)
    file_name = "steps_metrics"
    metrics_path = os.path.join(metrics_dir, f"{file_name}.pkl")
    with open(metrics_path, 'rb') as f:
        print("load " + metrics_path)
        step_times = pickle.load(f)
    for step_name, step_time in step_times:
        if step_name.endswith(("__store", "__score_time")):
            n_artifacts = n_artifacts + 0
        else:
            n_artifacts = n_artifacts + 1
        print("Step '{}' execution time: {}".format(step_name, step_time))
    print("number of artifacts " + str(n_artifacts))

def plot_artifact_graph(G):
    pos = nx.drawing.layout.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10)
    plt.show()

def get_steps(steps):
    mandatory_steps = []
    optional_steps = []
    for step_name, options in steps:
        if (str(step_name)[0].isdigit()):
            optional_steps.append((step_name, options))
        else:
            mandatory_steps.append((step_name, options))
    return optional_steps, mandatory_steps


def get_all_steps(steps):
    mandatory_steps = []
    optional_steps = []
    for step_name, options in steps:
        if (str(step_name)[0].isdigit()):
            optional_steps.append((step_name, options))
        else:
            mandatory_steps.append((step_name, options))
    return optional_steps, mandatory_steps
def get_first_lines(filename, n=10):
    """
    Extract the first n lines of a file.

    Parameters:
    - filename: path to the file
    - n: number of lines to extract

    Returns:
    - list of the first n lines
    """

    with open(filename, 'r', encoding="utf-8") as f:
        lines = [next(f) for _ in range(n)]

    return lines


def fit_pipeline_with_artifacts(pipeline, X_train, y_train):
    artifacts = {}
    X_temp = X_train.copy()

    for step_name, step_transformer in pipeline.steps[:-1]:  # Exclude the classifier step
        X_temp = step_transformer.fit_transform(X_temp, y_train)
        artifacts[step_name] = X_temp.copy()

    # Fit the classifier step
    step_name, step_transformer = pipeline.steps[-1]
    step_transformer.fit(X_temp, y_train)
    artifacts[step_name] = step_transformer

    return artifacts


def create_artifact_graph(artifacts):
    G = nx.DiGraph()

    for i, (step_name, artifact) in enumerate(artifacts.items()):
        G.add_node(step_name, artifact=artifact)
        if i > 0:
            prev_step_name = list(artifacts.keys())[i - 1]
            G.add_edge(prev_step_name, step_name)

    return G




def compute_loading_times(metrics_dir='metrics', artifacts_dir='artifacts'):
    os.makedirs(metrics_dir, exist_ok=True)
    loading_times = {}
    file_name = "loading_metrics"
    metrics_path = os.path.join(metrics_dir, f"{file_name}.pkl")

    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            # print("load " + metrics_path)
            loading_times = pickle.load(f)
    else:
        loading_times = {}

    files = [f for f in os.listdir(artifacts_dir) if f.endswith('.pkl')]

    for file in files:
        file_path = os.path.join(artifacts_dir, file)

        start_time = time.time()
        with open(file_path, 'rb') as f:
            _ = pickle.load(f)
        f.close()
        loading_time = time.time() - start_time
        if (file in loading_times):
            if (loading_time > loading_times[file]):
                loading_times[file] = loading_time
        else:
            loading_times[file] = loading_time
    # print(len(loading_times))
    with open(metrics_path, 'wb') as f:
        pickle.dump(loading_times, f)

    return loading_times



def update_graph(artifact_graph, mem_usage, step_time, param, hs_previous, hs_current, platforms, objective):
    artifact_graph.add_edge(hs_previous, hs_current + "_" + param, type=param, weight=step_time,
                            execution_time=step_time, memory_usage=max(mem_usage), platform=platforms, function=objective)
    return hs_current + "_" + param


def extract_platform(operator):
    split_strings = operator.split('__')
    if (len(split_strings) < 2):
        return "SK"
    else:
        return split_strings[0]


def text_inside_parentheses(s):
    # Find all substrings within parentheses
    matches = re.findall(r'\((.*?)\)', s)
    # Concatenate all matches into a single string, separated by a space (or any other separator you prefer)
    return ' '.join(matches)


def extract_first_two_chars(s, selected_models=[]):
    unified_string = ''.join(selected_models)
    sig = create_4_digit_signature(text_inside_parentheses(s) + unified_string)
    split_strings = s.split('__')
    result = ''.join([substring[:2] for substring in split_strings])
    return result + sig


def create_4_digit_signature(input_string):
    # Create a hash of the input string
    hash_object = hashlib.sha256(input_string.encode())
    hex_dig = hash_object.hexdigest()

    # Convert the hexadecimal hash to an integer
    numeric_hash = int(hex_dig, 16)

    # Reduce the hash to 4 digits. We use modulo 10000 to ensure the result is at most 4 digits
    short_hash = numeric_hash % 10000

    return f"{short_hash:04}"  # Return the number as a zero-padded string


def execute_graph(dataset_id, artifact_graph):
    ### EXECUTING A GRAPH
    pipeline_description = None
    memory_artifacts = {}
    train_data = None
    trainy = None
    testy = None
    test_data = None
    trainy = retrieve_artifact(dataset_id + "_trainy__")
    testy = retrieve_artifact(dataset_id + "_testy__")
    memory_artifacts[dataset_id + "_trainy__"] = trainy
    memory_artifacts[dataset_id + "_testy__"] = testy

    if nx.is_directed_acyclic_graph(artifact_graph):
        # Print edges in topological order
        #print("\nEdges in topological order:")
        for node in nx.topological_sort(artifact_graph):
            if artifact_graph.in_degree(node) == 0:
                memory_artifacts[node] = retrieve_artifact(node)
                if 'test' in node:
                    if test_data == None:
                        test_data = memory_artifacts.get(node)
                elif 'train' in node:
                    if train_data == None:
                        train_data = memory_artifacts.get(node)
            for _, neighbor, data in artifact_graph.edges(node, data=True):
                #print(f"({node}, {neighbor})")
                operator = data.get('function', 'No function attribute')
                function = data.get('type', 'No function attribute')
                ## FIT
                if function == 'fit':
                    args = inspect.signature(operator.fit).parameters
                    requires_y = 'y' in args
                    if requires_y:
                        fit_result = operator.fit(train_data, trainy)
                    else:
                        fit_result = operator.fit(train_data)
                    memory_artifacts[neighbor] = fit_result

                ## TRANSFORM
                elif 'tranform' in function:
                    last_underscore_index = node.rfind('_')
                    # Slice the string up to the last underscore
                    if last_underscore_index != -1:  # Check if '_' is found
                        operator = memory_artifacts.get(node[:last_underscore_index])
                        if function == 'ftranform':
                            fit_result = operator.transform(train_data)
                            memory_artifacts[neighbor] = fit_result
                            train_data = fit_result
                        elif function == 'tetranform':
                            fit_result = operator.transform(test_data)
                            memory_artifacts[neighbor] = fit_result
                            test_data = fit_result
                ## PREDICT
                elif 'predict' in function:
                    last_underscore_index = node.rfind('_')
                    # Slice the string up to the last underscore
                    if last_underscore_index != -1:  # Check if '_' is found
                        operator = memory_artifacts.get(node[:last_underscore_index])
                    predictions = operator.predict(test_data)
                    memory_artifacts[neighbor] = predictions
                ## SCORE
                elif 'score' in function:
                    fitted_operator = operator.fit(testy)
                    predictions = memory_artifacts[node]
                    X_temp = fitted_operator.score(predictions)
                #print(operator)
                #print(function)
    else:
        print("Graph is not a DAG. Cannot perform topological sort.")

def retrieve_artifact(hs_current, directory=None):
    if directory is None:
        directory = 'artifact_storage'  # Default to a 'artifact_storage' subdirectory
        if not os.path.exists(directory):
            os.makedirs(directory)
    file_path = os.path.join(directory, f"{hs_current}.pkl")
    with open(file_path, 'rb') as file:
        return pickle.load(file)
