from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from components.lib import graphviz_draw, graphviz_simple_draw, retrieve_artifact
from components.parser.parser import extract_artifact_graph
from dictionary.Evaluation.F1_score import F1ScoreCalculator
from dictionary.PCA.GPU__PCA import GPU__PCA
import os

if __name__ == '__main__':
    from components.HistoryGraph import HistoryGraph

    History = HistoryGraph("test_history")
    dataset_id = "HIGGS"
    # History.visualize()

    History.add_dataset(dataset_id)

    History.add_dataset_split(dataset_id, 0.3)
    # History.visualize()
    History.get_dataset_ids()

    user1_pipe = Pipeline(
        [('scaler', StandardScaler()), ('pca', GPU__PCA(n_components=3)), ('svc', SVC()), ('F1', F1ScoreCalculator())])

    split_ratio = 0.3
    History.execute_and_add(dataset_id, user1_pipe, split_ratio)

    # History.visualize()

    artifact = retrieve_artifact('HISKStGPPCSKSV8256_predict')
    print(artifact)
    artifact_graph = extract_artifact_graph(dataset_id, user1_pipe)
    # graphviz_draw(artifact_graph)
    graphviz_simple_draw(artifact_graph)
    request = History.execute_and_add(dataset_id, user1_pipe, split_ratio)
    print(request)
    # History.generate_plans(dataset_id, user1_pipe)
    artifact = retrieve_artifact(request)
    print(artifact)
    user2_pipe = Pipeline(
        [('scaler', StandardScaler()), ('pca', PCA(n_components=3)), ('svc', SVC()), ('F1', F1ScoreCalculator())])
    all_plans = History.generate_plans(dataset_id, user2_pipe)  ##Exhaustive
    # optimized_user2_pipe = History.generate_optimal_plan(user1_pipe)
