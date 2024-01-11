
if __name__ == '__main__':
    import os
    from components.HistoryGraph import HistoryGraph
    os.chdir("C:/Users/adoko/PycharmProjects/HyppoDemo/")
    History = HistoryGraph("test_history")
    History.visualize()

    History.add_dataset_split("HIGGS", 0.3)
    History.visualize()