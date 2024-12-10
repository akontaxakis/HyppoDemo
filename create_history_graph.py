import inspect

import networkx as nx
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from components.lib import graphviz_draw, graphviz_simple_draw, execute_graph
from components.parser.parser import extract_artifact_graph, graph_to_pipeline
from dictionary.Evaluator.Classification.ComputeAccuracy import AccuracyCalculator
from dictionary.Evaluator.Classification.F1_score import F1ScoreCalculator
from dictionary.Evaluator.Regression.MAECalculator import MAECalculator
from dictionary.Preprocessor.PCA.GPU__PCA import GPU__PCA

if __name__ == '__main__':
    from components.HistoryGraph import HistoryGraph

    History = HistoryGraph("History")
    #datasets=["wine","HIGGS"]
    #datasets = ["breast_cancer", "iris", "diabetes", "digits", "linnerud", "wine", "HIGGS", "TAXI", "BBC"]
    datasets = ["breast_cancer", "iris", "diabetes", "digits", "wine", "HIGGS"]
    for dataset in datasets:
        History.add_dataset_split(dataset, 0.3)

        dataset_id = dataset
        df = History.view_datasets()
        print(df)

        print(History.best_metrics_achieved(dataset_id))
        print(History.sort_by_metrics(dataset_id, "F1ScoreCalculator"))

        print(History.retrieve_best_pipelines(dataset_id, "F1ScoreCalculator",5))

        split_ratio = 0.3

        user2_pipe = Pipeline([('scaler', StandardScaler()), ('PCA',PCA(n_components=3)), ('RFC', RandomForestClassifier(random_state=42)), ('F1', F1ScoreCalculator())])
        request = History.execute_and_add(dataset_id, user2_pipe, split_ratio)
        artifact_graph, request = extract_artifact_graph(dataset_id, user2_pipe)

        request = History.execute_and_add(dataset_id, user2_pipe, split_ratio)

        user2_pipe = Pipeline(
            [('scaler', StandardScaler()),('pca', PCA(n_components=3)), ('SVC', SVC()), ('F1', AccuracyCalculator())])
        request = History.execute_and_add(dataset_id, user2_pipe, split_ratio)

        user2_pipe = Pipeline(
            [('scaler', StandardScaler()), ('SVC', SVC()), ('F1', AccuracyCalculator())])
        request = History.execute_and_add(dataset_id, user2_pipe, split_ratio)


        user2_pipe = Pipeline(
            [('scaler', StandardScaler()), ('pca', PCA(n_components=3)), ('dtc', DecisionTreeClassifier()), ('F1', F1ScoreCalculator())])
        request = History.execute_and_add(dataset_id, user2_pipe, split_ratio)
        user2_pipe = Pipeline(
            [('imputation', SimpleImputer(strategy='mean')), ('dtc', DecisionTreeClassifier()),
             ('F1', F1ScoreCalculator())])
        request = History.execute_and_add(dataset_id, user2_pipe, split_ratio)

        user2_pipe = Pipeline(
            [('imputation', SimpleImputer(strategy='mean')),('scaler', StandardScaler()), ('dtc', DecisionTreeClassifier()),
             ('F1', AccuracyCalculator())])
        request = History.execute_and_add(dataset_id, user2_pipe, split_ratio)

        user2_pipe = Pipeline(
            [('imputation', SimpleImputer(strategy='mean')), ('pca', PCA(n_components=3)),
             ('dtc', DecisionTreeClassifier()),
             ('F1', AccuracyCalculator())])
        request = History.execute_and_add(dataset_id, user2_pipe, split_ratio)
        user1_pipe = Pipeline(
            [('pca', PCA(n_components=3)), ('svc', SVC()), ('MAE', MAECalculator())])
        request = History.execute_and_add(dataset_id, user1_pipe, split_ratio)

        user1_pipe = Pipeline(
            [('pca', PCA(n_components=3)), ('svc', SVC()), ('MAE', MAECalculator())])
        request = History.execute_and_add(dataset_id, user1_pipe, split_ratio)

    datasets = ["linnerud","TAXI","BBC"]