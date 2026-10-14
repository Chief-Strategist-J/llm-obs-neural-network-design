"""Domain types and DTOs for network_spec feature."""

from dataclasses import dataclass
from typing import Mapping, Any


@dataclass(frozen=True)
class NetworkSpecCreateDTO:
    name: str
    architecture_type: str
    layer_count: int
    parameter_budget: int


@dataclass(frozen=True)
class NetworkSpecModel:
    id: str
    name: str
    architecture_type: str
    layer_count: int
    parameter_budget: int
    status: str
