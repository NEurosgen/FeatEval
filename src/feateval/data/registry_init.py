# src/data/registry_init.py (или прямо в loader.py)
from .registry import Registry
from .adapters.pyg import load_graph_list, xy_from_graphs

reg = Registry()

# Группа A: графы с признаками x_A
reg.register("A", lambda: xy_from_graphs(load_graph_list("path/to/A_graphs.pt"),
                                         x_attr="x_A", y_attr="y"))

# Группа B: графы с признаками x_B
reg.register("B", lambda: xy_from_graphs(load_graph_list("path/to/B_graphs.pt"),
                                         x_attr="x_B", y_attr="y"))
