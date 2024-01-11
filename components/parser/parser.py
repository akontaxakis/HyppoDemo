import time
import networkx as nx
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from components.parser.sub_parser import compute_pipeline_metrics_training_ad, compute_pipeline_metrics_evaluation_ad, \
    compute_pipeline_metrics_training, \
    compute_pipeline_metrics_evaluation, compute_pipeline_metrics_evaluation_helix, \
    compute_pipeline_metrics_training_helix


def sample(X_train, y_train, rate):
    sample_size = int(X_train.shape[0] * rate)
    # Generate a list of indices based on the length of your training set
    indices = np.arange(X_train.shape[0])
    # Randomly select indices for your sample
    sample_indices = np.random.choice(indices, size=sample_size, replace=False)
    # Use these indices to sample from X_train and y_train
    sample_X_train = X_train[sample_indices]
    sample_y_train = y_train[sample_indices]
    return sample_X_train, sample_y_train


def init_graph(dataset, split_ratio, dataset_multiplier=1):
    # Load the Breast Cancer Wisconsin dataset
    cc = 0
    G = nx.DiGraph()
    G.add_node("source", type="source", size=0, cc=cc)
    start_time = time.time()
    if dataset == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        X = np.random.rand(100000, 100)
        y = np.random.rand(100000)
        y = (y > 0.5).astype(int)
    elif dataset == "HIGGS":
        data = np.loadtxt('C:/Users/adoko/Downloads/HIGGS.csv', delimiter=',')
        # Extract and modify the first column based on your condition
        # (e.g., setting it to 0 or 1 if it's greater than 0.5)
        y = np.where(data[:, 0] > 0.5, 1, 0).astype(float)

        # Store the original first column in a separate array
        y = data[:, 0].copy()

        # Drop the first column from the data
        X = data[:, 1:]
        print(data.shape)
        print(data.shape)
    elif dataset == "TAXI":
        data = pd.read_csv('C:/Users/adoko/PycharmProjects/pythonProject1/datasets/taxi_train.csv')
        data['trip_duration'] = data['trip_duration'].replace(-1, 0)
        y = data['trip_duration'].values
        X = data.drop('trip_duration', axis=1).values
    else:
        data = pd.read_csv('C:/Users/adoko/Υπολογιστής/BBC.csv')
        data['target'] = data['target'].replace(-1, 0)
        y = data['target'].values
        X = data.drop('target', axis=1).values

    end_time = time.time()
    cc = end_time - start_time
    G.add_node(dataset, type="raw", size=X.size * X.itemsize, cc=cc, frequency=1)
    platforms = ["python"]
    G.add_edge("source", dataset, type="load", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time,
               memory_usage=0, platform=platforms)
    return X, y, G, cc


def add_dataset(G, dataset, dataset_multiplier=1):
    # Load the Breast Cancer Wisconsin dataset
    cc = 0
    start_time = time.time()
    if dataset == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        X = np.random.rand(100000, 100)
        y = np.random.rand(100000)
        y = (y > 0.5).astype(int)
    elif dataset == "HIGGS":
        data = np.loadtxt('C:/Users/adoko/Downloads/HIGGS.csv', delimiter=',')
        # Extract and modify the first column based on your condition
        # (e.g., setting it to 0 or 1 if it's greater than 0.5)
        y = np.where(data[:, 0] > 0.5, 1, 0).astype(float)

        # Store the original first column in a separate array
        y = data[:, 0].copy()

        # Drop the first column from the data
        X = data[:, 1:]
        print(data.shape)
        print(data.shape)
    elif dataset == "TAXI":
        data = pd.read_csv('C:/Users/adoko/PycharmProjects/pythonProject1/datasets/taxi_train.csv')
        data['trip_duration'] = data['trip_duration'].replace(-1, 0)
        y = data['trip_duration'].values
        X = data.drop('trip_duration', axis=1).values
    else:
        data = pd.read_csv('C:/Users/adoko/Υπολογιστής/BBC.csv')
        data['target'] = data['target'].replace(-1, 0)
        y = data['target'].values
        X = data.drop('target', axis=1).values

    end_time = time.time()
    cc = end_time - start_time
    G.add_node(dataset, type="raw", size=X.size * X.itemsize, cc=cc, frequency=1)
    platforms = ["python"]
    G.add_edge("source", dataset, type="load", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time,
               memory_usage=0, platform=platforms)
    return X, y, G, cc


def execute_pipeline(artifact_graph, uid, steps, mode, cc, X_train, y_train, X_test, y_test):
    artifacts = []
    pipeline = steps
    new_pipeline = clone(pipeline)
    cc1 = cc
    artifact_graph, artifacts, new_pipeline = compute_pipeline_metrics_training(artifact_graph, new_pipeline, uid,
                                                                                X_train, y_train, artifacts, mode, cc1)
    artifact_graph, artifacts, request = compute_pipeline_metrics_evaluation(artifact_graph, new_pipeline, uid, X_test,
                                                                             y_test, artifacts)
    return artifact_graph, artifacts, request


def execute_pipeline_helix(dataset, artifact_graph, uid, steps, mode, cc, X_train, y_train, X_test, y_test, budget):
    artifacts = []
    pipeline = steps
    new_pipeline = clone(pipeline)
    cc1 = cc
    artifact_graph, artifacts, new_pipeline, materialized_artifacts, budget = compute_pipeline_metrics_training_helix(
        artifact_graph, new_pipeline, uid, X_train, y_train, artifacts, mode, cc1, budget)
    artifact_graph, artifacts, request, materialized_artifacts = compute_pipeline_metrics_evaluation_helix(
        artifact_graph, new_pipeline, uid, X_test, y_test, artifacts, materialized_artifacts, budget)
    return artifact_graph, artifacts, request, materialized_artifacts


def execute_pipeline_ad(dataset, artifact_graph, uid, steps, mode, cc, X_train, y_train, X_test, y_test):
    artifacts = []
    pipeline = steps
    new_pipeline = clone(pipeline)
    cc1 = cc
    artifact_graph, artifacts, new_pipeline, selected_models = compute_pipeline_metrics_training_ad(artifact_graph,
                                                                                                    new_pipeline, uid,
                                                                                                    X_train, y_train,
                                                                                                    artifacts, mode,
                                                                                                    cc1)
    artifact_graph, artifacts, request = compute_pipeline_metrics_evaluation_ad(artifact_graph, new_pipeline, uid,
                                                                                X_test, y_test, artifacts)

    return artifact_graph, artifacts, request


def split_data(artifact_graph, dataset, split_ratio, X, y, cc):
    platforms = []
    platforms.append("python")
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
    end_time = time.time()
    mem_usage = [0, 0]  # memory_usage(lambda: train_test_split(X, y, test_size=0.2, random_state=42))
    # G.add_node("y_test")
    step_time = end_time - start_time
    cc = cc + step_time

    artifact_graph.add_node(dataset+"_train__", type="training", size=X_train.__sizeof__(), cc=cc, frequency=1)
    artifact_graph.add_node(dataset+"_test__", type="test", size=X_test.__sizeof__(), cc=cc, frequency=1)
    artifact_graph.add_node("split", type="split", size=0, cc=0, frequency=1)

    artifact_graph.add_edge(dataset, "split", type="split", weight=step_time, execution_time=step_time,
                            memory_usage=max(mem_usage), platform=platforms)
    artifact_graph.add_edge("split", dataset+"_train__", type="split", weight=0.000001, execution_time=0,
                            memory_usage=0, platform=platforms)
    # G.add_edge("split", "X_test",weight=0, execution_time=0,
    #           memory_usage=0)
    artifact_graph.add_edge("split", dataset+"_test__", type="split", weight=0.000001, execution_time=0,
                            memory_usage=0, platform=platforms)

    return X_test, X_train, y_test, y_train, cc
