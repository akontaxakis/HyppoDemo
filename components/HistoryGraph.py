import pickle

import networkx as nx

from components.parser.parser import add_dataset, split_data


class HistoryGraph:
    def __init__(self, history_id):
        self.history_id = history_id
        self.history = nx.DiGraph()
        self.history.add_node("source", type="source", size=0, cc=0)
        self.dataset_ids = []

    def add_dataset(self, dataset):
        """
               :dataset: A unique identifier for the dataset.
               :param split_ratio: the split ratio to train and test
               """
        self.dataset_ids.append(dataset)
        X, y, self.history, cc = add_dataset(self.history, dataset)

    def add_dataset_split(self, dataset, split_ratio):
        """
               :dataset: A unique identifier for the dataset.
               :param split_ratio: the split ratio to train and test
               """
        self.dataset_ids.append(dataset)
        X, y, self.history, cc = add_dataset(self.history, dataset)
        split_data(self.history, dataset, split_ratio, X, y, cc)

    def save_to_file(self, directory=""):
        """
        Saves the HistoryGraph to a file named after its history_id.
        :param directory: The directory path where the file will be saved. TODO:select directory
        """
        file_path = f"{directory}/{self.history_id}.pkl"
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_file(file_path):
        """
        Loads a HistoryGraph from a file.
        :param file_path: The path to the file from which to load the graph.
        :return: The loaded HistoryGraph object.
        """
        with open(file_path, 'rb') as file:
            return pickle.load(file)