# configs/base_config.py
from dataclasses import dataclass, field
import torch
from typing import List, Dict, Any

@dataclass
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    grad_clip: float = 1.0
    use_amp: bool = False
    multi_gpu: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_interval: int = 10


@dataclass
class LossConfig:
    name: str = "ssim3d"
    weight: float = 0.85
    params: Dict[str, Any] = field(default_factory=lambda: {
        "data_range": 1.0,
        "window_size": 7
    })


@dataclass
class InferenceConfig:
    tile_size: tuple = (128, 256, 256)
    tile_step_size: float = 0.5
    batch_size: int = 8
    max_tif_worker: int = 2
    use_amp: bool = False


@dataclass
class ModelConfig:
    name: str = "unet3d_basic"
    params: dict = field(default_factory=dict)

@dataclass
class DataConfig:
    crop_size: tuple = (64, 128, 128)
    batch_size: int = 8
    batch_size_val: int = 8
    num_workers: int = 4
    transform: bool = False
    target_resolution = (0.2, 0.2, 1)


@dataclass
class PathsConfig:
    root_dir: str
    train_dir: str = field(init=False)
    valid_dir: str = field(init=False)
    test_dir: str = field(init=False)
    gt_dir: str = field(init=False)
    results_dir_base: str = field(init=False)
    trained_model_path: str = field(init=False)

    def update_paths(self, task_name: str, create_dir=True):
        import os
        from utils.general import mkdir_x

        self.train_dir = os.path.join(self.root_dir, "Train")
        self.valid_dir = os.path.join(self.root_dir, "Val")
        self.test_dir = os.path.join(self.root_dir, "Test")
        self.gt_dir = os.path.join(self.root_dir, "Gt")
        self.results_dir_base = os.path.join(self.root_dir, "Results")
        self.log_dir = os.path.join(self.root_dir, 'log', task_name)
        if create_dir:
            mkdir_x(self.results_dir_base, emptyFlag=False)
            mkdir_x(self.log_dir, emptyFlag=True)

        self.results_dir = os.path.join(self.results_dir_base, task_name)
        if create_dir:
            mkdir_x(self.results_dir, emptyFlag=False)


        self.pred_dir = os.path.join(self.results_dir, "pred")
        self.save_model_dir = os.path.join(self.results_dir, "models")
        if create_dir:
            mkdir_x(self.pred_dir, emptyFlag=False)
            mkdir_x(self.save_model_dir, emptyFlag=False)



@dataclass
class BaseConfig():
    task_name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self):
        pass

    def update_all_paths(self, create_dir=True):
        if self.paths is None:
            raise ValueError("paths not setting")
        self.paths.update_paths(self.task_name, create_dir=create_dir)
