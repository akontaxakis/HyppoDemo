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
from dictionary.Evaluating.ComputeAUC import ComputeAUC
from dictionary.Evaluating.ComputeAccuracy import AccuracyCalculator
from dictionary.Evaluating.F1_score import F1ScoreCalculator
import os
from neo4j import GraphDatabase
from dictionary.Evaluating.MAECalculator import MAECalculator
from dictionary.Preproceser.PCA.GPU__PCA import GPU__PCA

if __name__ == '__main__':
    from components.HistoryGraph import HistoryGraph
    dataset_id = "HIGGS"
    History = HistoryGraph("HIGGS_example2")
    History.save_graph_graphml()


    import networkx as nx

    driver = GraphDatabase.driver('bolt:localhost:7687',auth=("neo4j", "12345678"))

    query = """
    MATCH (n)-[r]->(c) RETURN *
    """

    results = driver.session().run(query)

    G = nx.MultiDiGraph()

    nodes = list(results.graph()._nodes.values())
    for node in nodes:
        G.add_node(node.element_id, labels=node._labels, properties=node._properties)

    rels = list(results.graph()._relationships.values())
    for rel in rels:
        G.add_edge(rel.start_node.element_id, rel.end_node.element_id, key=rel.element_id, type=rel.type, properties=rel._properties)