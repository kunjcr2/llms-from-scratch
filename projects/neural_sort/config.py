from dataclasses import dataclass

@dataclass
class Config:
    n: int = 16
    value_range: tuple = (-100, 100)
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    batch_size: int = 256
    lr: float = 1e-3
    curriculum_steps: int = 20000

config = Config()
