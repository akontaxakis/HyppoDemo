import os
import pickle
import random
import hashlib
import re
import time

from memory_profiler import memory_usage
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
    depth = G.number_of_nodes();
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
    nx.draw(G, pos=pos, with_labels=True, font_size=2, node_shape="r", node_size=node_sizes, node_color=node_colors)
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


def graphviz_draw(G, mode):
    from IPython.display import Image

    # Convert the networkx graph to a pygraphviz graph

    # Customize appearance if needed
    # For example, you can modify node shapes, colors, edge types, etc.
    graph_size = 5
    pos = nx.spring_layout(G)
    pos_1 = nx.spring_layout(G)
    depth = G.number_of_nodes();
    for node in G.nodes:
        G.nodes[node]['depth'] = None

    # Compute and set depth for each node
    compute_depth(G, 'source')
    blue_nodes = []
    for node_id in G.nodes:
        if node_id == 'source':
            # pos[node_id] = np.array([0, depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), 0])
            G.nodes[node_id]['color'] = 'red'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        elif G.nodes[node_id]['type'] == 'split':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'blue'
            G.nodes[node_id]['size'] = 10
            G.nodes[node_id]['shape'] = 'rectangle'
            G.nodes[node_id]['shape'] = 'circle'
            G.nodes[node_id]['width'] = 0.3
            blue_nodes.append(node_id)

        elif G.nodes[node_id]['type'] == 'super':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'blue'
            G.nodes[node_id]['shape'] = 'circle'
            G.nodes[node_id]['width'] = 0.3
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

    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    labels = {node: "" if node in blue_nodes else str(node) for node in G.nodes()}
    for node, label in labels.items():
        G.nodes[node]['label'] = label
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr['rankdir'] = 'LR'
    for edge in A.edges():
        edge.attr['label'] = int(G[edge[0]][edge[1]]['weight'] * 10000)
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


def fit_pipeline_with_store_or_load_artifacts(pipeline, X_train, y_train, materialization, artifact_dir='artifacts'):
    os.makedirs(artifact_dir, exist_ok=True)
    artifacts = {}
    X_temp = X_train.copy()
    artifact_name = ""
    for step_name, step_transformer in pipeline.steps[:-1]:  # Exclude the classifier step
        artifact_name = artifact_name + str(step_transformer) + "_";
        artifact_path = os.path.join(artifact_dir, f"{artifact_name}.pkl")

        if os.path.exists(artifact_path):
            with open(artifact_path, 'rb') as f:
                print("load" + artifact_name)
                X_temp = pickle.load(f)
        else:
            X_temp = step_transformer.fit_transform(X_temp, y_train)
            if random.randint(1, 100) < materialization:
                with open(artifact_path, 'wb') as f:
                    pickle.dump(X_temp, f)

        artifacts[step_name] = X_temp.copy()

    # Fit the classifier step
    step_name, step_transformer = pipeline.steps[-1]
    step_transformer.fit(X_temp, y_train)
    artifacts[step_name] = step_transformer

    return artifacts


def compute_pipeline_metrics(artifact_graph, pipeline, uid, X_train, X_test, y_train, y_test, artifacts, mode,
                             scores_dir='metrics', artifact_dir='artifacts', models_dir='models',
                             materialization=0):
    os.makedirs(scores_dir, exist_ok=True)
    scores_file = uid + "_scores"
    scores_path = os.path.join(scores_dir, f"{scores_file}.txt")

    if mode == "sampling":
        hs_previous = "2sample_X_train__"
    else:
        hs_previous = "X_train__"
    X_temp = X_train.copy()
    step_full_name = hs_previous
    for step_name, step_obj in pipeline.steps:

        step_start_time = time.time()
        step_full_name = step_full_name + str(step_obj) + "__"
        hs_current = extract_first_two_chars(step_full_name)
        artifact_path = os.path.join(artifact_dir, f"{hs_current}.pkl")
        models_path = os.path.join(models_dir, f"{hs_current}.pkl")
        if hasattr(step_obj, 'fit_transform'):
            X_temp = step_obj.fit_transform(X_temp, y_train)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            mem_usage = memory_usage(lambda: step_obj.fit_transform(X_temp, y_train))
        elif hasattr(step_obj, 'fit'):
            mem_usage = memory_usage(lambda: step_obj.fit(X_temp, y_train))
            X_temp = step_obj.fit(X_temp, y_train)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

        if hasattr(step_obj, 'predict'):
            # step_obj.predict(X_test)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

        if random.randint(1, 100) < materialization:
            if hasattr(step_obj, 'predict'):
                artifacts.append(hs_current)
                with open(models_path, 'wb') as f:
                    pickle.dump(X_temp, f)
            else:
                artifacts.append(hs_current)
                with open(artifact_path, 'wb') as f:
                    pickle.dump(X_temp, f)

        if hs_previous == "":
            artifact_graph.add_node(hs_current)
            artifact_graph.add_edge("source", hs_current, weight=step_time, execution_time=step_time,
                                    memory_usage=max(mem_usage))
            hs_previous = hs_current
        else:
            artifact_graph.add_node(hs_current)
            artifact_graph.add_edge(hs_previous, hs_current, weight=step_time, execution_time=step_time,
                                    memory_usage=max(mem_usage))
            hs_previous = hs_current

    end_time = time.time()
    step_start_time = time.time()

    # Check if the pipeline has a classifier
    has_classifier = any(step_name == 'classifier' for step_name, _ in pipeline.steps)

    if has_classifier:
        step_start_time = time.time()
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        end_time = time.time()
        rounded_score = keep_two_digits(score)
        node_name = str(extract_first_two_chars(step_full_name)[-2:]) + "_" + str(rounded_score)
        with open(scores_path, "a") as outfile:
            outfile.write("\n")
            outfile.write(f'"{step_full_name}","{score}","{node_name}","{end_time - step_start_time}"')

    return artifact_graph, artifacts


def compute_pipeline_metrics_old(artifact_graph, pipeline, uid, X_train, X_test, y_train, y_test,
                                 metrics_dir='metrics'):
    os.makedirs(metrics_dir, exist_ok=True)
    file_name = uid + "_steps_metrics"
    file_name_2 = uid + "_pipelines_score"
    metrics_path = os.path.join(metrics_dir, f"{file_name}.pkl")
    scores_path = os.path.join(metrics_dir, f"{file_name_2}.txt")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            print("load" + metrics_path)
            step_times = pickle.load(f)
    else:
        step_times = []
    start_time = time.time()

    step_full_name = ""
    previous = ""

    for step_name, step_obj in pipeline.steps:
        step_start_time = time.time()
        step_full_name = step_full_name + str(step_obj) + "__"
        hs_previous = extract_first_two_chars(previous)
        hs_current = extract_first_two_chars(step_full_name)

        if hasattr(step_obj, 'fit_transform'):
            step_obj.fit_transform(X_train, y_train)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            step_times.append((step_full_name, step_time))

        elif hasattr(step_obj, 'fit'):
            step_obj.fit(X_train, y_train)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            step_times.append((step_full_name, step_time))

        elif hasattr(step_obj, 'predict'):
            step_obj.predict(X_test)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            step_times.append((step_full_name, step_time))

        if previous == "":
            artifact_graph.add_node(hs_current)
            artifact_graph.add_edge("source", hs_current, cost=step_time)
            previous = step_full_name
        else:
            artifact_graph.add_node(hs_current)
            artifact_graph.add_edge(hs_previous, hs_current, cost=step_time)
            previous = step_full_name

    end_time = time.time()
    step_start_time = time.time()

    # Check if the pipeline has a classifier
    has_classifier = any(step_name == '3.classifier' for step_name, _ in pipeline.steps)

    has_clustering = any(step_name == 'clustering' for step_name, _ in pipeline.steps)

    if has_classifier:
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        step_time = step_end_time - step_start_time
        step_times.append((step_full_name + "score_time", step_time))
        step_times.append((step_full_name + "score", score))
        with open(scores_path, "a") as outfile:
            outfile.write("\n")
            outfile.write(f'"{step_full_name}","{score}"')

    if has_clustering:
        pipeline.fit(X_train, y_train)
        labels = pipeline.predict(X_test)
        score = silhouette_score(X_test, labels)
        step_time = step_end_time - step_start_time
        step_times.append((step_full_name + "score_time", step_time))
        step_times.append((step_full_name + "score", score))
        with open(scores_path, "a") as outfile:
            outfile.write("\n")
            outfile.write(f'"{step_full_name}","{score}"')
    with open(metrics_path, 'wb') as f:
        pickle.dump(step_times, f)
    # print("Pipeline execution time: {}".format(total_time))
    # for step_name, step_time in step_times:
    #     print("Step '{}' execution time: {}".format(step_name, step_time))
    return step_times, artifact_graph


def update_graph(artifact_graph, mem_usage, step_time, param, hs_previous, hs_current, platforms):
    artifact_graph.add_edge(hs_previous, hs_current + "_" + param, type=param, weight=step_time,
                            execution_time=step_time, memory_usage=max(mem_usage), platform=platforms)
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
