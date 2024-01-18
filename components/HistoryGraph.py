import copy
import os
import pickle

import networkx as nx

from components.augmenter import new_edges
from components.history_manager import update_and_merge_graphs, add_load_tasks_to_the_graph
from components.lib import pretty_graph_drawing, graphviz_draw, graphviz_simple_draw
from components.parser.parser import add_dataset, split_data, execute_pipeline
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
                pi_prime['plan'].append(e)
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


class HistoryGraph:
    def __init__(self, history_id, directory=None):
        self.history_id = history_id
        if directory is None:
            directory = 'saved_graphs'
        file_path = os.path.join(directory, f"{self.history_id}.pkl")
        if os.path.exists(file_path):
            # Load the graph if it exists
            with open(file_path, 'rb') as file:
                saved_graph = pickle.load(file)
            self.history = saved_graph.history
            self.dataset_ids = saved_graph.dataset_ids
        else:
            self.history = nx.DiGraph()
            self.history.add_node("source", type="source", size=0, cc=0)
            self.dataset_ids = {}
            self.save_to_file()

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

    def visualize(self, type='none', mode='none'):
        if mode == "simple":
            graphviz_simple_draw(self.history)
        else:
            graphviz_draw(self.history, type, mode)

    def get_dataset_ids(self):
        print(self.dataset_ids)

    def execute_and_add(self, dataset, pipeline, split_ratio):

        self.dataset_ids[dataset] = split_ratio

        execution_graph, artifacts, request = execute_pipeline(dataset, pipeline, split_ratio)
        self.history = update_and_merge_graphs(copy.deepcopy(self.history), execution_graph)
        self.history = add_load_tasks_to_the_graph(self.history, artifacts)
        return request

    def generate_plans(self, dataset, pipeline):
        artifact_graph = nx.DiGraph()
        artifacts = []
        artifact_graph = pipeline_training(artifact_graph, dataset, pipeline)
        artifact_graph, request = pipeline_evaluation(artifact_graph, dataset, pipeline)
        print(request)
        required_artifacts, extra_cost_1, new_tasks = new_edges(self.history, artifact_graph)
        print(required_artifacts)
        plans = exhaustive_optimizer(required_artifacts, self.history)
        subgraph = {}
        i = 10
        for plan in plans:
            subgraph[plan['cost']] = self.history.edge_subgraph(plan['plan'])
            if i>0:
                graphviz_draw(self.history.edge_subgraph(plan['plan']), type='pycharm', mode='full')
                i=i-1
        return subgraph
