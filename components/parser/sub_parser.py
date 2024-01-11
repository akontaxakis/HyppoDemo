import os
import pickle
import random
import time

from sklearn.pipeline import Pipeline
from pympler import asizeof

from components.lib import extract_platform, extract_first_two_chars, update_graph, get_steps

os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")


def compute_pipeline_metrics_training(artifact_graph, pipeline, uid, X_train, y_train, artifacts, mode, cc,
                                      scores_dir='metrics', artifact_dir='artifacts', models_dir='models',
                                      materialization=0):
    os.makedirs(scores_dir, exist_ok=True)
    from joblib import dump

    folder_name = "taxi_models_2"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    scores_file = uid + "_scores"
    scores_path = os.path.join(scores_dir, f"{scores_file}.txt")

    if mode == "sampling":
        hs_previous = "2sample_X_train__"
    else:
        hs_previous = "X_train__"
    X_temp = X_train.copy()
    y_temp = y_train.copy()
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:

        platforms = []
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)

        platforms.append(extract_platform(name))
        step_full_name = step_full_name + name + "__"
        hs_current = extract_first_two_chars(step_full_name)
        # artifact_path = os.path.join(artifact_dir, f"{hs_current}.pkl")
        # models_path = os.path.join(models_dir, f"{hs_current}.pkl")
        if hasattr(step_obj, 'fit'):
            if str(step_obj).startswith("F1ScoreCalculator"):
                continue
            if str(step_obj).startswith("AccuracyCalculator"):
                continue
            if str(step_obj).startswith("ComputeAUC"):
                continue
            if str(step_obj).startswith("KS"):
                continue
            if str(step_obj).startswith("MSECalculator"):
                continue
            if str(step_obj).startswith("MAECalculator"):
                continue
            if str(step_obj).startswith("MPECalculator"):
                continue
            step_start_time = time.time()
            y_temp = y_temp[:len(X_temp)]

            fitted_operator = step_obj.fit(X_temp, y_temp)
            step_end_time = time.time()
            import copy
            fitted_operator_copy = copy.deepcopy(fitted_operator)
            if (
                    "Ra" in name or "La" in name or "KN" in name or "LG" in name or "Gr" in name or "Ri" in name or "Li" in name):
                file_path = os.path.join(folder_name, hs_current)
                with open(file_path, 'wb') as f:
                    pickle.dump(fitted_operator_copy, f)

            step_time = step_end_time - step_start_time
            cc = cc + step_time
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.fit(X_temp, y_train))
            artifact_graph.add_node(hs_current + "_fit", type="fitted_operator", size=asizeof.asizeof(fitted_operator),
                                    cc=cc, frequency=1)
            fitted_operator_name = update_graph(artifact_graph, mem_usage, step_time, "fit", hs_previous, hs_current,
                                                platforms)

        if hasattr(step_obj, 'transform'):
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            step_start_time = time.time()
            X_temp = step_obj.transform(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            cc = cc + step_time
            # tmp = X_temp.__sizeof__()
            artifact_graph.add_node(fitted_operator_name + "_super", type="super", size=0, cc=0, frequency=1)
            artifact_graph.add_node(hs_current + "_ftranform", type="train", size=X_temp.size * X_temp.itemsize, cc=cc,
                                    frequency=1)
            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "ftranform",
                                       fitted_operator_name + "_super", hs_current, platforms)

    return artifact_graph, artifacts, pipeline


def compute_pipeline_metrics_evaluation(artifact_graph, pipeline, uid, X_test, y_test, artifacts):
    X_temp = X_test.copy()
    y_temp = y_test.copy()
    hs_previous = "X_test__"
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:
        platforms = []
        platforms.append(extract_platform(str(step_obj)))
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj) in str(
                step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)
        step_full_name = step_full_name + str(name) + "__"
        hs_current = extract_first_two_chars(step_full_name)
        # artifact_path = os.path.join(artifact_dir, f"{hs_current}.pkl")
        # models_path = os.path.join(models_dir, f"{hs_current}.pkl")
        fitted_operator_name = hs_current + "_" + "fit"
        # print(fitted_operator_name)
        if hasattr(step_obj, 'transform'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            step_start_time = time.time()
            X_temp = step_obj.transform(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            artifact_graph.add_node(hs_current + "_tetranform", type="test", size=X_temp.size * X_temp.itemsize,
                                    cc=cc + step_time, frequency=1)
            artifact_graph.add_node(fitted_operator_name + "_Tsuper", type="super", size=0, cc=0, frequency=1)

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "tetranform",
                                       fitted_operator_name + "_Tsuper", hs_current, platforms)

        if hasattr(step_obj, 'predict'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            step_start_time = time.time()
            predictions = step_obj.predict(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            artifact_graph.add_node(hs_current + "_predict", type="test", size=predictions.size * predictions.itemsize,
                                    cc=cc + step_time, frequency=1)

            artifact_graph.add_node(fitted_operator_name + "_Psuper", type="super", size=0, cc=0, frequency=1)

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)

            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.predict(X_temp ))
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "predict",
                                       fitted_operator_name + "_Psuper", hs_current, platforms)
        if hasattr(step_obj, 'score'):
            if str(step_obj).startswith("F1ScoreCalculator") or str(step_obj).startswith("AccuracyCalculator") or str(
                    step_obj).startswith("MPECalculator") or str(step_obj).startswith("MSE") or str(
                    step_obj).startswith("MAE") or str(step_obj).startswith("KS") or str(step_obj).startswith("VIZ"):
                step_start_time = time.time()
                y_temp = y_temp[:len(predictions)]
                fitted_operator = step_obj.fit(y_temp)

                X_temp = fitted_operator.score(predictions)
                print(X_temp)
                step_end_time = time.time()
                step_time = step_end_time - step_start_time

                artifact_graph.add_node(hs_current + "_score", type="score",
                                        size=X_temp.size * X_temp.itemsize, cc=cc, frequency=1)

                hs_previous = update_graph(artifact_graph, mem_usage, step_time, "score", hs_previous, hs_current,
                                           platforms)

    return artifact_graph, artifacts, hs_previous


def compute_pipeline_metrics_training_ad(artifact_graph, pipeline, uid, X_train, y_train, artifacts, mode, cc,
                                         scores_dir='metrics', artifact_dir='artifacts', models_dir='models',
                                         materialization=0):
    os.makedirs(scores_dir, exist_ok=True)
    from joblib import dump
    folder_name = "taxi_models"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    scores_file = uid + "_scores"
    scores_path = os.path.join(scores_dir, f"{scores_file}.txt")

    if mode == "sampling":
        hs_previous = "2sample_X_train__"
    else:
        hs_previous = "X_train__"
    X_temp = X_train.copy()
    y_temp = y_train.copy()
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:

        platforms = []
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)

        platforms.append(extract_platform(name))
        step_full_name = step_full_name + name + "__"
        hs_current = extract_first_two_chars(step_full_name)
        # artifact_path = os.path.join(artifact_dir, f"{hs_current}.pkl")
        # models_path = os.path.join(models_dir, f"{hs_current}.pkl")
        if hasattr(step_obj, 'fit'):
            if str(step_obj).startswith("F1ScoreCalculator"):
                continue
            if str(step_obj).startswith("AccuracyCalculator"):
                continue
            if str(step_obj).startswith("ComputeAUC"):
                continue
            if str(step_obj).startswith("KS"):
                continue
            if str(step_obj).startswith("MSECalculator"):
                continue
            if str(step_obj).startswith("MAECalculator"):
                continue
            if str(step_obj).startswith("MPECalculator"):
                continue
            step_start_time = time.time()
            y_temp = y_temp[:len(X_temp)]

            fitted_operator = step_obj.fit(X_temp, y_temp)
            step_end_time = time.time()
            import copy
            fitted_operator_copy = copy.deepcopy(fitted_operator)
            # if ("Ra" in name or "De" in name or "KN" in name or "LG" in name or "Gr" in name or "Ri" in name or "Li" in name):
            #    file_path = os.path.join(folder_name, hs_current)
            #    with open(file_path, 'wb') as f:
            #        pickle.dump(fitted_operator_copy, f)

            step_time = step_end_time - step_start_time
            cc = cc + step_time
            if hasattr(step_obj, 'get_selected_models'):
                selected_models = step_obj.get_selected_models()
                print(selected_models)
                hs_current = extract_first_two_chars(step_full_name, selected_models)
                mem_usage = [0, 0]  # memory_usage(lambda: step_obj.fit(X_temp, y_train))
                artifact_graph.add_node(hs_current + "_fit", type="fitted_operator",
                                        size=asizeof.asizeof(fitted_operator), cc=cc, frequency=1)
                artifact_graph.add_node(hs_current + "_Fsuper", type="super", size=0, cc=0, frequency=1)
                artifact_graph.add_edge(hs_current + "_Fsuper", hs_current + "_fit", type="super", weight=step_time,
                                        execution_time=step_time, memory_usage=max(mem_usage), platform=platforms)

                artifact_graph.add_edge(hs_previous, hs_current + "_Fsuper", type="super",
                                        weight=0,
                                        execution_time=0, memory_usage=0, platform=platforms)
                for model in selected_models:
                    artifact_graph.add_node(model + "_fit", type="fitted_operator",
                                            size=asizeof.asizeof(fitted_operator), cc=cc, frequency=1)

                    artifact_graph.add_edge(model + "_fit", hs_current + "_Fsuper", type="super",
                                            weight=0,
                                            execution_time=0, memory_usage=0, platform=platforms)
            else:
                mem_usage = [0, 0]  # memory_usage(lambda: step_obj.fit(X_temp, y_train))
                artifact_graph.add_node(hs_current + "_fit", type="fitted_operator",
                                        size=asizeof.asizeof(fitted_operator), cc=cc, frequency=1)
                fitted_operator_name = update_graph(artifact_graph, mem_usage, step_time, "fit", hs_previous,
                                                    hs_current, platforms)

        if hasattr(step_obj, 'transform'):
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            step_start_time = time.time()
            X_temp = step_obj.transform(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            cc = cc + step_time
            # tmp = X_temp.__sizeof__()
            #### ADDING SUPER EDGE FOR TRANFORM
            artifact_graph.add_node(fitted_operator_name + "_super", type="super", size=0, cc=0, frequency=1)
            artifact_graph.add_node(hs_current + "_ftranform", type="train", size=X_temp.size * X_temp.itemsize, cc=cc,
                                    frequency=1)
            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "ftranform",
                                       fitted_operator_name + "_super", hs_current, platforms)

    return artifact_graph, artifacts, pipeline, selected_models


def compute_pipeline_metrics_evaluation_ad(artifact_graph, pipeline, uid, X_test, y_test, artifacts):
    X_temp = X_test.copy()
    y_temp = y_test.copy()
    hs_previous = "X_test__"
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:
        platforms = []
        platforms.append(extract_platform(str(step_obj)))
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj) in str(
                step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)
        step_full_name = step_full_name + str(name) + "__"

        hs_current = extract_first_two_chars(step_full_name)
        if hasattr(step_obj, 'get_selected_models'):
            selected_models = step_obj.get_selected_models()
            hs_current = extract_first_two_chars(step_full_name, selected_models)
        # artifact_path = os.path.join(artifact_dir, f"{hs_current}.pkl")
        # models_path = os.path.join(models_dir, f"{hs_current}.pkl")
        fitted_operator_name = hs_current + "_" + "fit"
        # print(fitted_operator_name)
        if hasattr(step_obj, 'transform'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            step_start_time = time.time()
            X_temp = step_obj.transform(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            artifact_graph.add_node(hs_current + "_tetranform", type="test", size=X_temp.size * X_temp.itemsize,
                                    cc=cc + step_time, frequency=1)
            artifact_graph.add_node(fitted_operator_name + "_Tsuper", type="super", size=0, cc=0, frequency=1)

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "tetranform",
                                       fitted_operator_name + "_Tsuper", hs_current, platforms)

        if hasattr(step_obj, 'predict'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            step_start_time = time.time()
            predictions = step_obj.predict(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            artifact_graph.add_node(hs_current + "_predict", type="test", size=predictions.size * predictions.itemsize,
                                    cc=cc + step_time, frequency=1)

            artifact_graph.add_node(fitted_operator_name + "_Psuper", type="super", size=0, cc=0, frequency=1)

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)

            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.predict(X_temp ))
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "predict",
                                       fitted_operator_name + "_Psuper", hs_current, platforms)
        if hasattr(step_obj, 'score'):
            if str(step_obj).startswith("F1ScoreCalculator") or str(step_obj).startswith("AccuracyCalculator") or str(
                    step_obj).startswith("MPECalculator") or str(step_obj).startswith("MSE") or str(
                    step_obj).startswith("MAE") or str(step_obj).startswith("KS") or str(step_obj).startswith("VIZ"):
                step_start_time = time.time()
                fitted_operator = step_obj.fit(y_temp)
                X_temp = fitted_operator.score(predictions)
                print(X_temp)
                step_end_time = time.time()
                step_time = step_end_time - step_start_time

                artifact_graph.add_node(hs_current + "_score", type="score",
                                        size=X_temp.size * X_temp.itemsize, cc=cc, frequency=1)

                hs_previous = update_graph(artifact_graph, mem_usage, step_time, "score", hs_previous, hs_current,
                                           platforms)

    return artifact_graph, artifacts, hs_previous


def compute_pipeline_metrics_training_helix(artifact_graph, pipeline, uid, X_train, y_train, artifacts, mode, cc,
                                            budget,
                                            scores_dir='metrics'):
    loading_speed = 566255240
    materialized_artifacts = []
    if mode == "sampling":
        hs_previous = "2sample_X_train__"
    else:
        hs_previous = "X_train__"
    X_temp = X_train.copy()
    y_temp = y_train.copy()
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:

        platforms = []
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)

        platforms.append(extract_platform(name))
        step_full_name = step_full_name + name + "__"
        hs_current = extract_first_two_chars(step_full_name)
        if hasattr(step_obj, 'fit'):
            if str(step_obj).startswith("F1ScoreCalculator"):
                continue
            if str(step_obj).startswith("AccuracyCalculator"):
                continue
            if str(step_obj).startswith("ComputeAUC"):
                continue
            if str(step_obj).startswith("KS"):
                continue
            if str(step_obj).startswith("MSECalculator"):
                continue
            if str(step_obj).startswith("MAECalculator"):
                continue
            if str(step_obj).startswith("MPECalculator"):
                continue
            step_start_time = time.time()
            y_temp = y_temp[:len(X_temp)]

            fitted_operator = step_obj.fit(X_temp, y_temp)
            step_end_time = time.time()

            step_time = step_end_time - step_start_time
            cc = cc + step_time
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.fit(X_temp, y_train))
            a_size = asizeof.asizeof(fitted_operator)
            loading_time = a_size / loading_speed
            if cc > loading_time * 2 and a_size < budget:
                budget = budget - a_size
                materialized_artifacts.append(hs_current + "_fit")

            artifact_graph.add_node(hs_current + "_fit", type="fitted_operator", size=a_size, cc=cc, frequency=1)
            fitted_operator_name = update_graph(artifact_graph, mem_usage, step_time, "fit", hs_previous, hs_current,
                                                platforms)

        if hasattr(step_obj, 'transform'):
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            step_start_time = time.time()
            X_temp = step_obj.transform(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            cc = cc + step_time
            # tmp = X_temp.__sizeof__()
            loading_time = (X_temp.size * X_temp.itemsize) / loading_speed
            if cc > loading_time * 2 and (X_temp.size * X_temp.itemsize) < budget:
                budget = budget - asizeof.asizeof(X_temp.size * X_temp.itemsize)
                materialized_artifacts.append(hs_current + "_ftranform")

            artifact_graph.add_node(fitted_operator_name + "_super", type="super", size=0, cc=0, frequency=1)
            artifact_graph.add_node(hs_current + "_ftranform", type="train", size=X_temp.size * X_temp.itemsize, cc=cc,
                                    frequency=1)
            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "ftranform",
                                       fitted_operator_name + "_super", hs_current, platforms)

    return artifact_graph, artifacts, pipeline, materialized_artifacts, budget

def compute_pipeline_metrics_evaluation_helix(artifact_graph, pipeline, uid, X_test, y_test, artifacts,
                                              materialized_artifacts, budget):
    loading_speed = 566255240
    X_temp = X_test.copy()
    y_temp = y_test.copy()
    hs_previous = "X_test__"
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:
        platforms = []
        platforms.append(extract_platform(str(step_obj)))
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj) in str(
                step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)
        step_full_name = step_full_name + str(name) + "__"
        hs_current = extract_first_two_chars(step_full_name)
        fitted_operator_name = hs_current + "_" + "fit"
        if hasattr(step_obj, 'transform'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            step_start_time = time.time()
            X_temp = step_obj.transform(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            loading_time = (X_temp.size * X_temp.itemsize) / loading_speed
            if cc > loading_time * 2 and (X_temp.size * X_temp.itemsize) < budget:
                budget = budget - asizeof.asizeof(X_temp.size * X_temp.itemsize)
                materialized_artifacts.append(hs_current + "_tetranform")

            artifact_graph.add_node(hs_current + "_tetranform", type="test", size=X_temp.size * X_temp.itemsize,
                                    cc=cc + step_time, frequency=1)
            artifact_graph.add_node(fitted_operator_name + "_Tsuper", type="super", size=0, cc=0, frequency=1)

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "tetranform",
                                       fitted_operator_name + "_Tsuper", hs_current, platforms)

        if hasattr(step_obj, 'predict'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            step_start_time = time.time()
            predictions = step_obj.predict(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            loading_time = (predictions.size * predictions.itemsize) / loading_speed
            if cc > loading_time * 2 and (predictions.size * predictions.itemsize) < budget:
                budget = budget - asizeof.asizeof(predictions.size * predictions.itemsize)
                materialized_artifacts.append(hs_current + "_predict")

            artifact_graph.add_node(hs_current + "_predict", type="test", size=predictions.size * predictions.itemsize,
                                    cc=cc + step_time, frequency=1)

            artifact_graph.add_node(fitted_operator_name + "_Psuper", type="super", size=0, cc=0, frequency=1)

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)

            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.predict(X_temp ))
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "predict",
                                       fitted_operator_name + "_Psuper", hs_current, platforms)
        if hasattr(step_obj, 'score'):
            if str(step_obj).startswith("F1ScoreCalculator") or str(step_obj).startswith("AccuracyCalculator") or str(
                    step_obj).startswith("MPECalculator") or str(step_obj).startswith("MSE") or str(
                    step_obj).startswith("MAE") or str(step_obj).startswith("KS") or str(step_obj).startswith("VIZ"):

                step_start_time = time.time()
                y_temp = y_temp[:len(predictions)]
                fitted_operator = step_obj.fit(y_temp)

                X_temp = fitted_operator.score(predictions)
                print(X_temp)
                step_end_time = time.time()
                step_time = step_end_time - step_start_time

                cc = artifact_graph.nodes[hs_previous]['cc']

                loading_time = (X_temp.size * X_temp.itemsize) / loading_speed
                if cc > loading_time * 2 and (X_temp.size * X_temp.itemsize) < budget:
                    budget = budget - asizeof.asizeof(X_temp.size * X_temp.itemsize)
                    materialized_artifacts.append(hs_current + "_score")

                artifact_graph.add_node(hs_current + "_score", type="score",
                                        size=X_temp.size * X_temp.itemsize, cc=cc, frequency=1)

                hs_previous = update_graph(artifact_graph, mem_usage, step_time, "score", hs_previous, hs_current,
                                           platforms)

    return artifact_graph, artifacts, hs_previous, materialized_artifacts

def generate_pipeline(steps, number_of_steps, task='random_no'):
    pipeline_steps = []
    optional_steps, mandatory_steps = get_steps(steps[:-1])
    if task == "random":
        steps_count = random.randint(1, len(optional_steps))
        selected_steps = random.sample(optional_steps, steps_count)
        selected_steps = mandatory_steps + selected_steps
    else:
        selected_steps = mandatory_steps

    selected_steps.append(steps[number_of_steps - 1])
    for step_name, options in selected_steps:
        selected_option = random.choice(options)
        pipeline_steps.append((step_name, selected_option))
    # print(pipeline_steps)
    return Pipeline(pipeline_steps)