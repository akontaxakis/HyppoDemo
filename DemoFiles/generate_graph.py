import inspect

import networkx as nx
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
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
    dataset_id = "HIGGS"
    History = HistoryGraph("History")
    History.add_dataset_split(dataset_id, 0.3)
    df = History.view_datasets()
    print(df)
    print(History.best_metrics_achieved(dataset_id))
    print(History.sort_by_metrics(dataset_id, "F1ScoreCalculator"))

    print(History.retrieve_best_pipelines(dataset_id, "F1ScoreCalculator",5))

    split_ratio = 0.3

    user2_pipe = Pipeline([('scaler', StandardScaler()), ('PCA',PCA(n_components=3)), ('RFC', RandomForestClassifier(random_state=42)), ('F1', F1ScoreCalculator())])
    request = History.execute_and_add(dataset_id, user2_pipe, split_ratio)
    artifact_graph, request = extract_artifact_graph(dataset_id, user2_pipe)
    graphviz_draw(artifact_graph, type='pycharm', mode='use_alias')

    user2_pipe = Pipeline([('scaler', StandardScaler()), ('PCA',GPU__PCA(n_components=3)), ('RFC', RandomForestClassifier(random_state=42)), ('F1', F1ScoreCalculator())])
    request = History.execute_and_add(dataset_id, user2_pipe, split_ratio)
    artifact_graph, request = extract_artifact_graph(dataset_id, user2_pipe)
    #graphviz_draw(artifact_graph, type='pycharm', mode='use_alias')


    #History.delete("HISKStSKPCSKRa8256_predict")
    #History.delete("HISKStGPPCSKRa8256_predict")
    #History.delete("HISKStSKPCSKRaSKF12886_score")
    #History.delete("HISKStGPPCSKRaSKF12886_score")
    History.visualize(type='pycharm', mode='use_alias')

    History.equivalent_operators(dataset_id, ["PCA", "GPU_PCA"])
    History.visualize_augmented(dataset_id, type='pycharm', mode='use_alias',
                                filter_artifact_id="HIStPCRaF12886_score", filter='eq_retrieve')

    pipe = History.optimal_retrieval_plan(dataset_id, ["HIStPCRaF12886_score"], mode="with_eq")
   # graphviz_draw(pipe[0][1], 'pycharm', 'use_alias')

    user2_pipe = Pipeline(
        [('scaler', StandardScaler()),('pca', PCA(n_components=3)), ('SVC', SVC()), ('F1', AccuracyCalculator())])
    request = History.execute_and_add(dataset_id, user2_pipe, split_ratio)

    user2_pipe = Pipeline(
        [('scaler', StandardScaler()), ('pca', PCA(n_components=3)), ('dtc', DecisionTreeClassifier()), ('F1', F1ScoreCalculator())])
    request = History.execute_and_add(dataset_id, user2_pipe, split_ratio)


    artifact_graph = extract_artifact_graph(dataset_id, user2_pipe)
    #graphviz_draw(artifact_graph, type='pycharm', mode='use_alias')
    #graphviz_draw(pipe[0][1], 'notebook', 'use_alias')
    user1_pipe = Pipeline([('scaler', StandardScaler()), ('pca', GPU__PCA(n_components=3)), ('SVC', SVC()), ('ac', AccuracyCalculator())])
    History.execute_and_add(dataset_id, user1_pipe, split_ratio)
    artifact_graph = extract_artifact_graph(dataset_id, user1_pipe)
    user1_pipe = Pipeline(
        [('pca', PCA(n_components=3)), ('svc', SVC()), ('MAE', MAECalculator())])
    artifact_graph = extract_artifact_graph(dataset_id, user1_pipe)

    History.execute_and_add(dataset_id, user1_pipe, split_ratio)

    artifact_graph = extract_artifact_graph(dataset_id, user1_pipe)

    History.execute_and_add(dataset_id, user1_pipe, split_ratio)
    History.printArtifacts()
    dict = History.best_metrics_achieved(dataset_id)


    #dict = History.best_metrics_achieved(dataset_id)
    #pipeline = History.retrieve_best_pipeline(dataset_id, "AccuracyCalculator")
    #History.visualize(type='pycharm',load_edges='without_load', mode='use_alias')
    #History.delete("", "all")
    History.equivalent_operators(dataset_id, ["PCA", "GPU_PCA"])
    History.visualize_augmented(dataset_id, type='pycharm', mode='use_alias',
                                filter_artifact_id="HIPCSV9009_predict", filter='eq_retrieve')
    History.printArtifacts(mode="with_eq")
    History.save_to_file()
    #History.save_graph_graphml()
    #pipe = History.optimal_retrieval_plan(["HIPCSV9009_predict"], mode="with_eq")

    History.compareHistoryAgainstDictionary(dataset_id=dataset_id
                                            , objective="PCA")
    #graphviz_draw(pipe[0][1], 'pycharm', 'use_alias')
