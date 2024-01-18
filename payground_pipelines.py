import inspect

import networkx as nx
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from components.lib import graphviz_draw, graphviz_simple_draw, execute_graph
from components.parser.parser import extract_artifact_graph, graph_to_pipeline
from dictionary.Evaluation.F1_score import F1ScoreCalculator
from dictionary.PCA.GPU__PCA import GPU__PCA
import os


if __name__ == '__main__':
    from components.HistoryGraph import HistoryGraph
    dataset_id = "HIGGS"
    History = HistoryGraph("test_history")
    #History.visualize("sd")
    History.add_dataset_split(dataset_id, 0.3)
    #History.visualize("sd")

    user1_pipe = Pipeline(
        [('scaler', StandardScaler()), ('pca', GPU__PCA(n_components=3)), ('svc', SVC()), ('F1', F1ScoreCalculator())])
    artifact_graph = extract_artifact_graph(dataset_id, user1_pipe)
    split_ratio = 0.3
    request = History.execute_and_add(dataset_id, user1_pipe, split_ratio)

    History.visualize("pycharm","full")

    plans = History.generate_plans(dataset_id, user1_pipe)
    #plan = plans.pop(0)
   #graphviz_draw(plan,type='pycharm', mode='full')
    #graphviz_draw(artifact_graph,"normal")

    #execute_graph(dataset_id, artifact_graph)