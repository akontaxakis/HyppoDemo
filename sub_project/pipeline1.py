import inspect
import time
import networkx as nx
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from components.lib import graphviz_draw, graphviz_simple_draw, execute_graph
from components.parser.parser import extract_artifact_graph, graph_to_pipeline, get_dataset, get_split
from dictionary.Evaluating.ComputeAUC import ComputeAUC
from dictionary.Evaluating.ComputeAccuracy import AccuracyCalculator
from dictionary.Evaluating.F1_score import F1ScoreCalculator
import os

from dictionary.Evaluating.MAECalculator import MAECalculator
from dictionary.Preproceser.PCA.GPU__PCA import GPU__PCA

if __name__ == '__main__':
    from components.HistoryGraph import HistoryGraph
    dataset_id = "HIGGS"
    History = HistoryGraph("HIGGS_example2")
    History.add_dataset_split(dataset_id, 0.3)

    split_ratio = 0.3

    #user2_pipe = Pipeline([('scaler', StandardScaler()), ('PCA',PCA(n_components=3)), ('RFC', RandomForestClassifier()), ('F1', F1ScoreCalculator())])
    #user2_pipe = Pipeline([('scaler', StandardScaler()), ('RFC', LGBMClassifier(random_state=42))])
    #user2_pipe = Pipeline([('scaler', StandardScaler()), ('RFC', DecisionTreeClassifier(max_depth=10, random_state=42))])
    #user2_pipe = Pipeline([('scaler', StandardScaler()), ('RFC',  SVC(kernel="linear", C=0.025, random_state=42))])
    user2_pipe = Pipeline([('scaler', StandardScaler()), ('RFC', DecisionTreeClassifier(random_state=42))])


    X, y = get_dataset(dataset_id)
    X_test, X_train, y_test, y_train = get_split(X, y, split_ratio)
    start_time = time.time()
    user2_pipe.fit(X_train,y_train)
    int_ac =user2_pipe.score(X_test,y_test)
    end_time = time.time()
    training_time = end_time - start_time
    print(str(int_ac)+","+ str(training_time) +","+str(int_ac/training_time))
    classifier = user2_pipe.named_steps['RFC']

    #SVC
    #coef = classifier.coef_[0]
    #feature_importances = np.abs(coef)

    #classifers with features importance
    feature_importances = classifier.feature_importances_

    # Normalize the feature importances to sum to 1
    normalized_importances = feature_importances / np.sum(feature_importances)

    #feature_importances = classifier.feature_importances_

    #normalized_importances = feature_importances / feature_importances.sum()

    # Filter your data to include only these features
    for i in [1,2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]:
        top_n_indices = np.argsort(feature_importances)[-i:]
        top_n_importance = normalized_importances[top_n_indices]
        cumulative_importance = top_n_importance.sum()
        expected_accuracy = cumulative_importance * int_ac
        #expected_accuracy = int_ac - expected_accuracy_loss
        user2_pipe = Pipeline([('scaler', StandardScaler()), ('RFC',  DecisionTreeClassifier(random_state=42))])
        start_time = time.time()
        X_train_filtered = X_train[:, top_n_indices]
        X_test_filtered = X_test[:, top_n_indices]
        user2_pipe.fit(X_train_filtered, y_train)
        ac = user2_pipe.score(X_test_filtered, y_test)
        end_time = time.time()
        training_time = end_time - start_time
        #time
        print(str(i)+ "," + str(training_time))
        #accuracy
        #print(str(i) + "," + str(ac) + "," + str(expected_accuracy))

