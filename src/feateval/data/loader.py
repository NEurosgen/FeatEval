from .schema import FeatureSpec

from .validate import validate_shapes


from .registry import Registry
from .adapters.pyg import load_graph_list, xy_from_graphs

reg = Registry()

# Группа A: графы с признаками x_A
reg.register("A", lambda: xy_from_graphs(load_graph_list("path/to/A_graphs.pt"),
                                         x_attr="x_A", y_attr="y"))

# Группа B: графы с признаками x_B
reg.register("B", lambda: xy_from_graphs(load_graph_list("path/to/B_graphs.pt"),
                                         x_attr="x_B", y_attr="y"))


def load_group(spec: FeatureSpec, name: str):
    X, y = reg.get(name)            # A → (X_A, y), B → (X_B, y)
    validate_shapes(X, y)           # базовая проверка
    return X, y

def load_union(spec: FeatureSpec, names: list[str]):
    # объединить представления (по колонкам / конкатенировать признаки)
    pairs = [reg.get(n) for n in names]
    # проверяем, что y совпадает
    ys = [p[1] for p in pairs]
    # (проверка идентичности y по всем группам)
    # X = concat по признакам
    import numpy as np
    Xs = [p[0] for p in pairs]
    X = np.concatenate(Xs, axis=1) if isinstance(Xs[0], np.ndarray) else ...
    return X, ys[0]
