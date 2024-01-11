from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from components.parser.parser import execute_pipeline, vizualize_pipeline
from dictionary.Evaluation.F1_score import F1ScoreCalculator
from dictionary.PCA.GPU__PCA import GPU__PCA
import os
os.chdir("C:/Users/adoko/PycharmProjects/HyppoDemo/")


if __name__ == '__main__':

    from components.HistoryGraph import HistoryGraph


    History = HistoryGraph("test_history")
    History.visualize()

    History.add_dataset("HIGGS")

    History.add_dataset_split("HIGGS", 0.3)
    History.visualize()
    History.get_dataset_ids()

    user1_pipe = Pipeline([('scaler', StandardScaler()), ('pca', GPU__PCA(n_components=3)), ('svc', SVC()), ('F1', F1ScoreCalculator())])

    split_ratio = 0.3
    History.execute_and_add("HIGGS", user1_pipe, split_ratio)
    History.visualize()

