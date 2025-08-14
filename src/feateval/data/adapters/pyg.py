from typing import List, Tuple
import numpy as np
from torch_geometric.data import Data
import torch

def load_graph_list(path: str) -> List[Data]:
    return torch.load(path)

def xy_from_graphs(graphs: List[Data], x_attr: str = "x", y_attr: str = "y") -> Tuple[np.ndarray, np.ndarray]:

    X_list, y_list = [], []
    for g in graphs:
        x = getattr(g, x_attr)
        y = getattr(g, y_attr)
        if x.dim() == 1: x = x.unsqueeze(0)
        X_list.append(x.cpu().numpy())
        y_np = y.cpu().numpy()
        if y_np.ndim == 0: y_np = np.array([y_np]) 
        y_list.append(y_np)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y
