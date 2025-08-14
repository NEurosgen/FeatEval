from typing import Callable,Dict,Tuple,Any
import numpy as np
import pandas as pd


DataGetter = Callable[[],Tuple[Any,np.ndarray]]

class Registry:
    def __init__ (self):
        self._getters : Dict[str,DataGetter] = {}
    def register (self,name:str,getter:DataGetter):
        self._getters[name] = getter
    def get(self, name: str)->Tuple[Any,np.ndarray]:
        return self._getters[name]()