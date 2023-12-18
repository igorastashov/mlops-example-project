from dataclasses import dataclass


@dataclass
class Paths:
    data: str


@dataclass
class Params:
    epoch_count: int
    lr: float
    batch_size: int
    momentum: float


@dataclass
class ConvNetConfig:
    paths: Paths
    params: Params
