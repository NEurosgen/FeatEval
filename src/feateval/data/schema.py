
from dataclasses import dataclass
from typing import Callable, Literal, Dict

Modality = Literal["table", "graph"]

@dataclass(frozen=True)
class FeatureSpec:
    target:str
    modality :Modality
    groups: Dict[str,str]