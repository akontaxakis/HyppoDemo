import copy
import os
import pickle

import networkx as nx

from components.augmenter import new_edges
from components.history_manager import update_and_merge_graphs, add_load_tasks_to_the_graph
from components.lib import pretty_graph_drawing
from components.parser.parser import add_dataset, split_data, execute_pipeline
from components.parser.sub_parser import pipeline_training, pipeline_evaluation


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

    def visualize(self):
        pretty_graph_drawing(self.history)

    def get_dataset_ids(self):
        print(self.dataset_ids)

    def execute_and_add(self, dataset, pipeline, split_ratio):

        self.dataset_ids[dataset] = split_ratio

        execution_graph, artifacts = execute_pipeline(dataset, pipeline, split_ratio)
        self.history = update_and_merge_graphs(copy.deepcopy(self.history), execution_graph)
        self.history = add_load_tasks_to_the_graph(self.history, artifacts)

    def generate_plans(self, dataset, pipeline):
        artifact_graph = nx.DiGraph()
        artifacts = []
        artifact_graph = pipeline_training(artifact_graph, dataset, pipeline)
        artifact_graph, request = pipeline_evaluation(artifact_graph, dataset, pipeline)
        print(request)
        required_artifacts, extra_cost_1, new_tasks = new_edges(self.history, artifact_graph)
        print(required_artifacts)

        return artifact_graph


