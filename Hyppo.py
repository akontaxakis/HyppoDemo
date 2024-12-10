

from components.HistoryGraph import HistoryGraph

import warnings

warnings.filterwarnings('ignore')
Hyppo = HistoryGraph("History_TPOT")
if __name__ == '__main__':
    Hyppo.visualize()