import copy
import glob
import json
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd

from components.augmenter import new_edges, create_equivalent_graph_without_fit
from components.history_manager import update_and_merge_graphs, add_load_tasks_to_the_graph
from components.lib import pretty_graph_drawing, graphviz_draw, graphviz_simple_draw, view_dictionary
from components.parser.parser import add_dataset, split_data, execute_pipeline, extract_artifact_graph
from components.parser.sub_parser import pipeline_training, pipeline_evaluation

def CartesianProduct(sets):
    if len(sets) == 0:
        return [[]]
    else:
        CP = []
        current = sets.popitem()
        for c in current[1]:
            for set in CartesianProduct(sets):
                CP.append(set + [c])
        sets[current[0]] = current[1]
        return CP


def bstar(A, v):
    return A.in_edges(v)


def Expand(A, pi):
    PI = []
    E = {}
    #GET THE EDGES
    for v in [v_prime for v_prime in pi['frontier'] if v_prime not in ['source']]:
        E[v] = bstar(A, v)
    #Find all possible moves
    M = CartesianProduct(E)
    for move in M:
        pi_prime = {
            'cost': pi['cost'],
            'visited': pi['visited'].copy(),
            'frontier': [],
            'plan': pi['plan'].copy()
        }
        for e in move:
            edge_data = A.get_edge_data(*e)
            extra_edges = []
            if 'super' in e[0] or 'split' in e[0]:
                head = list(A.successors(e[0]))
                tail = list(A.predecessors(e[0]))
                extra_edges += list(A.in_edges(e[0]))
                extra_edges += list(A.out_edges(e[0]))
            else:
                head = [e[1]]
                tail = [e[0]]
            #if e[1] not in pi_prime['visited']:
            #    new_nodes = e[1]
            new_nodes = [n for n in head if n not in pi_prime['visited']]
            if new_nodes:
                pi_prime['cost'] += int(10000 * edge_data.get('weight', 0))
                if not extra_edges:
                    pi_prime['plan'].append(e)
                else:
                    pi_prime['plan'] += extra_edges
                pi_prime['visited'].append(new_nodes)
                #if e[0] not in (pi_prime['visited'] + pi_prime['frontier']):
                #    pi_prime['frontier'].append(e[0])
                pi_prime['frontier'] += [n for n in tail if n not in (pi_prime['visited'] + pi_prime['frontier'])]

        PI.append(pi_prime)
    return PI


def exhaustive_optimizer(required_artifacts, history):
    Q = [{'cost': 0, 'visited': [], 'frontier': required_artifacts, 'plan': []}]
    plans = []
    while Q:
        pi = Q.pop(0)
        if pi['frontier'] == ['source']:
            plans.append({'plan': pi['plan'], 'cost': pi['cost']})
        else:
            for pi_prime in Expand(history, pi):
                Q.append(pi_prime)
    return plans


def stack_optimizer(required_artifacts, history):
    Q = [{'cost': 0, 'visited': [], 'frontier': required_artifacts, 'plan': []}]
    cost_star = 999999
    pi_star = []
    while Q:
        pi = Q.pop(0)
        if pi['frontier'] == ['source']:
             if pi['cost'] < cost_star:
                 pi_star = pi
                 cost_star = pi['cost']
        else:
            for pi_prime in Expand(history, pi):
                if pi_prime ['cost'] < cost_star:
                    Q.append(pi_prime)
    return pi_star


class HistoryGraph:
    def __init__(self, history_id, directory=None):
        self.history_id = history_id
        if directory is None:
            #directory = 'saved_graphs'
            #directory = r'C:\Users\adoko\PycharmProjects\autoPipe\autoML\saved_graphs'
            directory = r'C:\Users\adoko\PycharmProjects\HyppoDemo\saved_graphs'

        file_path = os.path.join(directory, f"{self.history_id}.pkl")
        if os.path.exists(file_path):
            # Load the graph if it exists
            with open(file_path, 'rb') as file:
                saved_graph = pickle.load(file)
            self.history = saved_graph.history
            self.eq_history = saved_graph.eq_history
            self.dataset_ids = saved_graph.dataset_ids
        else:
            self.history = nx.DiGraph()
            self.eq_history = nx.DiGraph()
            self.history.add_node("source", type="source", size=0, cc=0, alias="storage")
            self.dataset_ids = {}
            self.save_to_file()

    def save_graph_graphml(self):
        for node, data in self.history.nodes(data=True):
            for key, value in data.items():
                if isinstance(value, list):
                    # Convert list to a JSON-formatted string
                    self.history.nodes[node][key] = json.dumps(value)

        # Iterate over all edges in the graph to check and modify the attributes
        for u, v, data in self.history.edges(data=True):
            # Iterate over each attribute in the edge's data dictionary
            for attr_key, attr_value in list(data.items()):
                # Check if the attribute value is of a type that needs to be serialized (e.g., list or dict)
                if attr_key =="function":
                    # Convert the type object to its fully qualified name as a string
                    data[attr_key] = f"k"
                if isinstance(attr_value, (list, dict)):
                    # Convert the value to a JSON string and update the attribute
                    print(data[attr_key])
                    data[attr_key] = json.dumps(attr_value)
                    print(data[attr_key])


        nx.write_graphml(self.history, 'history.graphml')

    def view_datasets(self):
        data = [(key, value) for key, value in self.dataset_ids.items()]

        # Create a DataFrame from the list of tuples
        df = pd.DataFrame(data, columns=['dataset_id', 'split_ratio'])
        # Create a DataFrame
        return df

    def add_dataset(self, dataset):
        """
               :dataset: A unique identifier for the dataset.
               :param split_ratio: the split ratio to train and test
               """
        self.dataset_ids[dataset] = 0

        X, y, self.history, cc = add_dataset(self.history, dataset)
        self.save_to_file()

    # TODO add path to the dataset
    def add_dataset_split(self, dataset, split_ratio):
        """
               :dataset: A unique identifier for the dataset.
               :param split_ratio: the split ratio to train and test
               """
        self.dataset_ids[dataset] = split_ratio
        X, y, self.history, cc = add_dataset(self.history, dataset)
        split_data(self.history, dataset, split_ratio, X, y, cc)
        self.save_to_file()

    def save_to_file(self, directory=None):
        """
        Saves the HistoryGraph to a file named after its history_id.
        :param directory: The directory path where the file will be saved. TODO:select directory
        """
        if directory is None:
            directory = 'saved_graphs'  # Default to a 'saved_graphs' subdirectory
            if not os.path.exists(directory):
                os.makedirs(directory)  # Create the directory if it doesn't exist

        file_path = os.path.join(directory, f"{self.history_id}.pkl")
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_file(history_id, directory=None):
        """
        Loads a HistoryGraph from a file using its history_id.
        :param history_id: The history_id of the HistoryGraph to be loaded.
        :param directory: The directory path where the file is saved.
        :return: The loaded HistoryGraph object.
        """
        if directory is None:
            directory = 'saved_graphs'

        file_path = os.path.join(directory, f"{history_id}.pkl")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No saved file found for history_id '{history_id}' in '{directory}'")

        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def visualize(self, type='none', mode='none', load_edges='none', filter_artifact_id = None, filter='None'):
        if 'eq' in filter:
            G = self.eq_history.copy()
        else:
            G = self.history.copy()

        if filter_artifact_id != None:
            if filter_artifact_id in G:
                target_node = filter_artifact_id
                # Get all predecessors and successors of the target node
                predecessors = set(nx.ancestors(G, target_node))
                successors = set(nx.descendants(G, target_node))
                if 'retrieve' in filter:
                    successors = []
                # Create a set that includes the target node, its predecessors, and its successors
                relevant_nodes = predecessors.union(successors).union({target_node})

                # Create a subgraph with these nodes
                G = G.subgraph(relevant_nodes).copy()
                if mode == "simple":
                    graphviz_simple_draw(G)
                else:
                     graphviz_draw(G, type, mode, load_edges)
            else:
                print("dataset id does not exist")
        else:
            if mode == "simple":
                graphviz_simple_draw(G)
            else:
                graphviz_draw(G, type, mode, load_edges, self.history_id)

    def get_dataset_ids(self):
        print(self.dataset_ids)

    def execute_and_add(self, dataset, pipeline, split_ratio = None):
        if split_ratio == None:
            self.dataset_ids[dataset] = split_ratio

        execution_graph, artifacts, request = execute_pipeline(dataset, pipeline, split_ratio)
        self.history = update_and_merge_graphs(copy.deepcopy(self.history), execution_graph)
        self.history = add_load_tasks_to_the_graph(self.history, artifacts)
        self.save_to_file()
        return request, pipeline

    def generate_plans(self, dataset, pipeline, mode='None'):
        artifact_graph = nx.DiGraph()
        artifacts = []
        artifact_graph = pipeline_training(artifact_graph, dataset, pipeline)
        artifact_graph, request = pipeline_evaluation(artifact_graph, dataset, pipeline)
        print(request)
        required_artifacts, extra_cost_1, new_tasks = new_edges(self.history, artifact_graph)
        print(required_artifacts)
        if mode == 'with_eq':
            A = self.eq_history.copy()
        else:
            A = self.history.copy()
        A.remove_node(dataset)
        A.remove_node(dataset+ "_split")

        #graphviz_draw(A, 'pycharm', 'full')
        plans = exhaustive_optimizer(required_artifacts, A)
        subgraph = []
        i = 0
        for plan in plans:
            subgraph.append([plan['cost'], self.history.edge_subgraph(plan['plan']), required_artifacts])
            #if i>0:
            #    graphviz_draw(self.history.edge_subgraph(plan['plan']), type='pycharm', mode='full')
            #i=i-1
        return subgraph

    def delete(self, artifact, mode = None):
        if mode == "all":
            for node, attr in self.history.nodes(data=True):
                if attr.get('type') != 'source' and attr.get('type') != 'training' and attr.get('type') != 'testing' and attr.get('type') != 'raw' and node !="HIGPPC3810_fit":
                    if self.history.has_edge('source', node):
                        self.history.remove_edge('source', node)

        else:
            if self.history.has_edge('source', artifact):
                self.history.remove_edge('source', artifact)

    def get_augmented_graph(self, dataset_id, filter_artifact_id="None", filter="None"):
        if "eq" in filter:
            G = self.eq_history.copy()
        else:
            G = self.history.copy()
        G.remove_node(dataset_id)
        G.remove_node(dataset_id + "_split")
        if filter_artifact_id != None:
            if filter_artifact_id in G:
                target_node = filter_artifact_id
                # Get all predecessors and successors of the target node
                predecessors = set(nx.ancestors(G, target_node))
                successors = set(nx.descendants(G, target_node))
                if 'retrieve' in filter:
                    successors = []
                # Create a set that includes the target node, its predecessors, and its successors
                relevant_nodes = predecessors.union(successors).union({target_node})
                G = G.subgraph(relevant_nodes).copy()
        return G

    def visualize_augmented(self,dataset_id, type='none', mode='none', load_edges='none', filter_artifact_id=None, filter='None'):
        if "eq" in filter:
            G = self.eq_history.copy()
        else:
            G = self.history.copy()
        G.remove_node(dataset_id)
        G.remove_node(dataset_id + "_split")
        if filter_artifact_id != None:
            if filter_artifact_id in G:
                target_node = filter_artifact_id
                # Get all predecessors and successors of the target node
                predecessors = set(nx.ancestors(G, target_node))
                successors = set(nx.descendants(G, target_node))
                if 'retrieve' in filter:
                    successors = []
                # Create a set that includes the target node, its predecessors, and its successors
                relevant_nodes = predecessors.union(successors).union({target_node})

                # Create a subgraph with these nodes
                G = G.subgraph(relevant_nodes).copy()
                if mode == "simple":
                    graphviz_simple_draw(G)
                else:
                    graphviz_draw(G, type, mode, load_edges)
            else:
                print("dataset id does not exist")
        else:
            if mode == "simple":
                graphviz_simple_draw(G)
            else:
                graphviz_draw(G, type, mode, load_edges)

    def retrieve_artifact(self, artifact, mode=None):
        dataset = 'HIGGS'
        if mode == 'with_eq':
            A = self.eq_history.copy()
        else:
            A = self.history.copy()
        A.remove_node(dataset)
        A.remove_node(dataset + "_split")
        if A.has_node(artifact):
            plans = exhaustive_optimizer([artifact], A)
            subgraph = []
            i = 0
            for plan in plans:
                if mode == 'with_eq':
                    subgraph.append([plan['cost'], self.eq_history.edge_subgraph(plan['plan']), [artifact]])
                else:
                    subgraph.append([plan['cost'], self.history.edge_subgraph(plan['plan']), [artifact]])
            return subgraph
        else:
            print('artifact does not exist in history')

    def retrieve_artifacts(self, artifacts, mode=None):
        dataset = 'HIGGS'
        if mode == 'with_eq':
            A = self.eq_history.copy()
        else:
            A = self.history.copy()
        A.remove_node(dataset)
        A.remove_node(dataset + "_split")
        if A.has_node(artifacts[0]):
            plans = exhaustive_optimizer(artifacts, A)
            subgraph = []
            i = 0
            for plan in plans:
                if mode == 'with_eq':
                    subgraph.append([plan['cost'], self.eq_history.edge_subgraph(plan['plan']), artifacts])
                else:
                    subgraph.append([plan['cost'], self.history.edge_subgraph(plan['plan']), artifacts])
            return subgraph
        else:
            print('artifact does not exist in history')

    def optimal_retrieval_plan(self,dataset_id, artifacts, mode=None):
        dataset = dataset_id
        if mode == 'with_eq':
            new_artifacts = []
            for node in artifacts:
                # if artifact_graph.nodes[node]['type'] != "super":
                if ("_fit_" not in node) and ("_fit" not in node) and (
                        "GL" in node or "GP" in node or "TF" in node or "TR" in node or "SK" in node):
                    modified_node = node.replace("GP", "")
                    modified_node = modified_node.replace("TF", "")
                    modified_node = modified_node.replace("TR", "")
                    modified_node = modified_node.replace("SK", "")
                    modified_node = modified_node.replace("GL", "")
                    new_artifacts.append(modified_node)
                else:
                    new_artifacts.append(node)
            artifacts = new_artifacts
            A = self.eq_history.copy()
        else:
            A = self.history.copy()
        A.remove_node(dataset)
        A.remove_node(dataset + "_split")
        if A.has_node(artifacts[0]):
            plan = stack_optimizer(artifacts, A)
            subgraph = []
            if mode == 'with_eq':
                subgraph.append([plan['cost'], self.eq_history.edge_subgraph(plan['plan']), artifacts])
            else:
                subgraph.append([plan['cost'], self.history.edge_subgraph(plan['plan']), artifacts])
            return subgraph
        else:
            print('artifact does not exist in history')


    def find_equivalent_operators(self):
        self.eq_history = create_equivalent_graph_without_fit(self.history)

    def sort_by_metrics(self, dataset_id, metric):
        G = self.history.copy()
        value_set = set()
        if dataset_id in G:
            target_node = dataset_id
            # Get all predecessors and successors of the target node
            predecessors = set(nx.ancestors(G, target_node))
            successors = set(nx.descendants(G, target_node))
            # Create a set that includes the target node, its predecessors, and its successors
            relevant_nodes = predecessors.union(successors).union({target_node})

            # Create a subgraph with these nodes
            G = G.subgraph(relevant_nodes).copy()
            # Iterate through the nodes to process those with type 'score'
            for node, attr in G.nodes(data=True):
                if attr.get('type') == 'score':
                    for node, attr in G.nodes(data=True):
                        if attr.get('type') == 'score' and attr.get('operator') == metric :
                            operator = attr.get('operator')
                            value = attr.get('value', 0)  # Default value is 0 if not present
                            # Update the highest and lowest value for each operator
                            value_set.add(value)
        return sorted(list(value_set), reverse=True)
    def best_metrics_achieved(self, dataset_id):
        G = self.history.copy()
        if dataset_id in G:
            target_node = dataset_id
            # Get all predecessors and successors of the target node
            predecessors = set(nx.ancestors(G, target_node))
            successors = set(nx.descendants(G, target_node))
            # Create a set that includes the target node, its predecessors, and its successors
            relevant_nodes = predecessors.union(successors).union({target_node})

            # Create a subgraph with these nodes
            G = G.subgraph(relevant_nodes).copy()
            from collections import defaultdict
            highest_values = defaultdict(lambda: float('-inf'))
            lowest_values = defaultdict(lambda: float('inf'))

            # Iterate through the nodes to process those with type 'score'
            for node, attr in G.nodes(data=True):
                if attr.get('type') == 'score':
                    for node, attr in G.nodes(data=True):
                        if attr.get('type') == 'score':
                            operator = attr.get('operator')
                            value = attr.get('value', 0)  # Default value is 0 if not present
                            # Update the highest and lowest value for each operator
                            highest_values[operator] = max(highest_values[operator], value)
                            lowest_values[operator] = min(lowest_values[operator], value)


        import pandas as pd
        df = pd.DataFrame([(operator, highest, lowest) for operator, highest, lowest in
                           zip(highest_values, highest_values.values(), lowest_values.values())],
                          columns=['operator', 'highest_value', 'lowest_value'])
        return df
    def popular_unused_operators(self, dataset_id, objective):

        dictionary = view_dictionary(objective=objective)
        popular_operators = self.popular_operators(dataset_id)

        num_rows = dictionary.shape[0]

        # Generate random integer values for each row
        # For example, random integers between 1 and 100
        dictionary['Frequency'] = np.random.randint(10, 101, size=num_rows)


        # For df2, remove details within parentheses to match 'Implementation' format
        popular_operators['NormalizedOperator'] = popular_operators['Operator'].str.extract(r'([^\(]+)')

        # Now, filter df1 by implementations that do not exist in the normalized 'Operator' of df2
        # We're using ~df1['Implementation'].isin(df2['NormalizedOperator']) to find non-matching rows
        df3 = dictionary[~dictionary['Implementation'].isin(popular_operators['NormalizedOperator'])][['Implementation', 'Frequency']]
        return df3
    def retrieve_best_pipelines(self, dataset_id, metric, N):
        A = self.history.copy()
        df = pd.DataFrame(columns=[metric, "Pipeline"])  # Main DataFrame

        highest_values = self.sort_by_metrics(dataset_id, metric)


        edges_to_remove = [(u, v) for u, v in A.out_edges("source") if A.nodes[v].get('type') not in ['training', 'testing']]
        A.remove_edges_from(edges_to_remove)
        for i in range(min(N, len(highest_values))):  # Ensure loop does not exceed available highest values
            G = A.copy()
            specific_value = highest_values[i]  # No need to check for None, loop controls it

            request = None
            for node, attr in G.nodes(data=True):
                if attr.get('operator') == metric and attr.get('value') == specific_value:
                    request = node
                    break

            if request is None:
                print(f"No node found with operator '{metric}' and value '{specific_value}'")
                continue

            G.remove_node(dataset_id)
            G.remove_node(dataset_id + "_split")

            # Assuming exhaustive_optimizer and its output handling are correct
            plans = exhaustive_optimizer([request], G)
            if not plans:
                continue
            plan = plans.pop(0)
            graph = self.history.edge_subgraph(plan['plan'])

            # Extract the pipeline process based on your specific logic
            topological_sorted_nodes = list(nx.topological_sort(graph))
            filtered_sorted_aliases = [graph.nodes[node]['alias'] for node in topological_sorted_nodes
                                       if 'type' in graph.nodes[node] and graph.nodes[node]['type'] != "super"
                                       and 'alias' in graph.nodes[node]
                                       and graph.nodes[node]['alias'] not in {"predictions", "trainX", "storage",
                                                                              "testX"}]

            unique_array = []
            for item in filtered_sorted_aliases:
                if item not in unique_array:
                    unique_array.append(item)

            # Create a new DataFrame for the current data point
            new_data = {metric: [specific_value], "Pipeline": [unique_array]}
            new_df = pd.DataFrame(new_data)

            # Concatenate the new_df to the main df
            df = pd.concat([df, new_df], ignore_index=True)

        return df




    def retrieve_best_pipeline(self, dataset_id, metric, mode='review'):
        G = self.history.copy()
        metrics = self.best_metrics_achieved(dataset_id)
        highest_value_for_specific_operator = metrics[metrics['operator'] == metric]['highest_value'].max()

        specific_operator = metric
        specific_value = highest_value_for_specific_operator
        request = None
        for node, attr in G.nodes(data=True):
            if attr.get('operator') == specific_operator and attr.get('value') == specific_value:
                request = node
                break
        if request == None:
            print(f"No node found with operator '{specific_operator}' and value '{specific_value}'")

        G.remove_node(dataset_id)
        G.remove_node(dataset_id + "_split")
        #graphviz_draw(G, 'pycharm', 'full')
        if(mode == 'review'):
            edges_to_remove = [(u, v) for u, v in G.out_edges("source") if G.nodes[v].get('type') not in ['training', 'testing']]

        # Remove the identified edges
        G.remove_edges_from(edges_to_remove)

        plans = exhaustive_optimizer([node], G)
        plan = plans.pop(0)
        graph =  self.history.edge_subgraph(plan['plan'])

        # Perform a topological sort on the graph
        topological_sorted_nodes = list(nx.topological_sort(graph))

        # Extracting the 'alias' attribute of each node, following the topological order
        # Exclude nodes with specific types or aliases
        filtered_sorted_aliases = [graph.nodes[node]['alias'] for node in topological_sorted_nodes
                                   if 'type' in graph.nodes[node] and graph.nodes[node]['type'] != "super"
                                   and 'alias' in graph.nodes[node]
                                   and graph.nodes[node]['alias'] not in {"predictions", "trainX", "storage", "testX"}]

        unique_array = []
        for item in filtered_sorted_aliases:
            if item not in unique_array:
                unique_array.append(item)
        return unique_array

    def  retrieve_pipelines(self, dataset_id, score_operator = "AccuracyCalculator", threshold=0.6, mode="review"):
            G = self.history.copy()
            if dataset_id in G:
                target_node = dataset_id
                # Get all predecessors and successors of the target node
                predecessors = set(nx.ancestors(G, target_node))
                successors = set(nx.descendants(G, target_node))
                # Create a set that includes the target node, its predecessors, and its successors
                relevant_nodes = predecessors.union(successors).union({target_node})

                # Create a subgraph with these nodes
                G = G.subgraph(relevant_nodes).copy()
                from collections import defaultdict
                highest_values = defaultdict(lambda: float('-inf'))
                nodes = []
                # Iterate through the nodes to process those with type 'score'
                for node, attr in G.nodes(data=True):
                    if attr.get('type') == 'score':
                        operator = attr.get('operator')
                        value = attr.get('value', 0)
                        if operator == score_operator and value>=threshold:
                            nodes.append(node)

            print(nodes)
            G.add_node("request",type="super", size=0, cc=0)
            for node in nodes:
                G.add_edge(node, 'request', type='super', weight=0,
                                        execution_time=0, memory_usage=0, platform=["None"],
                                        function=None)

            G.remove_node(dataset_id)
            G.remove_node(dataset_id + "_split")

            if (mode == 'review'):
                edges_to_remove = [(u, v) for u, v in G.out_edges("source") if
                                   G.nodes[v].get('type') not in ['training', 'testing']]

                # Remove the identified edges
            G.remove_edges_from(edges_to_remove)

            plans = exhaustive_optimizer(["request"], G)
            subgraph = []
            i = 0
            for plan in plans:
                subgraph.append([plan['cost'], self.history.edge_subgraph(plan['plan']), ["request"]])
            return subgraph

    def  retrieve_optimal_pipelines(self, dataset_id, score_operator = "AccuracyCalculator", threshold=0.6, mode="review"):
            G = self.history.copy()
            if dataset_id in G:
                target_node = dataset_id
                # Get all predecessors and successors of the target node
                predecessors = set(nx.ancestors(G, target_node))
                successors = set(nx.descendants(G, target_node))
                # Create a set that includes the target node, its predecessors, and its successors
                relevant_nodes = predecessors.union(successors).union({target_node})

                # Create a subgraph with these nodes
                G = G.subgraph(relevant_nodes).copy()
                from collections import defaultdict
                highest_values = defaultdict(lambda: float('-inf'))
                nodes = []
                # Iterate through the nodes to process those with type 'score'
                for node, attr in G.nodes(data=True):
                    if attr.get('type') == 'score':
                        operator = attr.get('operator')
                        value = attr.get('value', 0)
                        if operator == score_operator and value>=threshold:
                            nodes.append(node)

            G.add_node("request",type="super", size=0, cc=0)
            for node in nodes:
                G.add_edge(node, 'request', type='super', weight=0,
                                        execution_time=0, memory_usage=0, platform=["None"],
                                        function=None)

            G.remove_node(dataset_id)
            G.remove_node(dataset_id + "_split")

            if (mode == 'review'):
                edges_to_remove = [(u, v) for u, v in G.out_edges("source") if
                                   G.nodes[v].get('type') not in ['training', 'testing']]

                # Remove the identified edges
            G.remove_edges_from(edges_to_remove)

            plan = stack_optimizer(["request"], G)
            subgraph = []
            subgraph.append([plan['cost'], self.history.edge_subgraph(plan['plan']), ["request"]])
            return subgraph


    def equivalent_operators(self,dataset_id,pair):
        self.find_equivalent_operators()

    def printArtifacts(self, mode = None):
        if mode ==None:
            for node in self.history.nodes:
                print(node)
        elif mode == "with_eq":
            for node in self.eq_history.nodes:
                print(node)

    def popular_operators(self, dataset_id):
        G = self.get_augmented_graph(dataset_id, dataset_id)

        # Step 1: Extract Node Frequencies
        node_frequencies = {node: G.nodes[node]['frequency'] for node in G.nodes if node != "source"}
        node_alias = {node: G.nodes[node]['alias'] for node in G.nodes if node != "source"}
        # Step 2: Calculate Operator Frequencies
        operator_frequency = {}
        for edge in G.edges:
            source_node, target_node = edge
            # Skip edges that originate from "source"
            if source_node == "source":
                continue
            if G.edges[edge]['type'] =="super":
                continue
            target_node = edge[1]
            operator = node_alias[target_node]
            # Assuming the edge is directed and you want the frequency of the node the edge points to
            if operator == "predictions":
                continue
            frequency = node_frequencies[target_node]
            if operator in operator_frequency:
                operator_frequency[operator] += frequency
            else:
                operator_frequency[operator] = frequency

        # Step 3: Rank Operators
        # Sort the operators based on the summed frequency
        operators_ranked = sorted(operator_frequency.items(), key=lambda x: x[1], reverse=True)
        df_operators_ranked = pd.DataFrame(operators_ranked, columns=['Operator', 'Frequency'])
        df_filtered = df_operators_ranked[
            ~(df_operators_ranked['Operator'].isin(['trainX', 'testX', 'split', 'trainY', 'testY']))
        ]
        return df_filtered

    def optimize_pipeline(self, dataset_id, pipeline):
         graph, request = extract_artifact_graph(dataset_id,pipeline)
         plans=self.optimal_retrieval_plan(dataset_id, [request])
         return plans.pop(0)


    def list_py_files(self, directory):
        paths = []
        for root, dirs, files in os.walk(directory):
            for file in glob.glob(os.path.join(root, '*.py')):
                paths.append(file)
        return paths

    def view_dictionary(self, type=None, objective=None):

        directory_path = "C:/Users/adoko/PycharmProjects/HyppoDemo/dictionary"
        paths = self.list_py_files(directory_path)

        # Specify the path to your directory

        df = pd.DataFrame(columns=["Type", "Objective", "Implementation"])

        # Prefix to remove from each path
        prefix = "C:/Users/adoko/PycharmProjects/HyppoDemo/dictionary\\"

        # Processing the paths
        data = []
        for path in paths:
            # Remove the prefix
            trimmed_path = path.replace(prefix, "")
            # Split the path to extract the required components
            components = trimmed_path.split("\\")
            if len(components) >= 3:
                Type, Objective, Implementation = components[0], components[1], components[-1]
                # Remove the file extension from Implementation
                Implementation = Implementation.split('.')[0]
                data.append([Type, Objective, Implementation])

        # Create a DataFrame
        df = pd.DataFrame(data, columns=['Type', 'Objective', 'Implementation'])

        if type is not None:
            # Apply the age filter
            df = df[df['Type'] == type]
        if objective is not None:
            # Apply the age filter
            df = df[df['Objective'] == objective]
        return df