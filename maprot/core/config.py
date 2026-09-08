"""
MapRot Configuration Management.
Provides YAML configuration parsing, path resolution, and parameter validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from yaml import SafeLoader


class MapRotConfig:
    """
    MapRot centralized configuration loader.
    Resolves project root and paths automatically without requiring sys.path hacks.
    """

    def __init__(self, config_dict: Dict[str, Any], root_dir: Optional[Path] = None):
        self._raw = config_dict
        self.root_dir = root_dir or Path.cwd().resolve()
        self._parse()

    @classmethod
    def from_yaml(cls, config_path: str | Path, root_dir: Optional[Path] = None) -> "MapRotConfig":
        path = Path(config_path)
        if not path.is_absolute():
            resolved_root = root_dir or Path.cwd().resolve()
            path = resolved_root / path

        if not path.exists():
            raise FileNotFoundError(f"MapRot configuration file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=SafeLoader) or {}

        return cls(data, root_dir=root_dir or path.parent.parent.resolve())

    def _parse(self):
        # Top-level sections with defaults
        self.project_name: str = self._raw.get("project_name", "MapRot")
        self.device: str = self._raw.get("device", "cuda")
        self.log_level: str = self._raw.get("log_level", "INFO")

        # Directories
        dirs = self._raw.get("dirs", {})
        self.data_dir = self._resolve_path(dirs.get("data_dir", "data"))
        self.train_dir = self._resolve_path(dirs.get("train_dir", "data/train"))
        self.test_dir = self._resolve_path(dirs.get("test_dir", "data/test"))
        self.weights_dir = self._resolve_path(dirs.get("weights_dir", "weights"))
        self.outputs_dir = self._resolve_path(dirs.get("outputs_dir", "outputs"))
        self.logs_dir = self._resolve_path(dirs.get("logs_dir", "logs"))

        # Data & Patching
        data_cfg = self._raw.get("data", {})
        self.file_type: str = data_cfg.get("file_type", ".tif")
        self.patch_size: int = data_cfg.get("patch_size", 512)
        self.discard_rate: float = data_cfg.get("discard_rate", 0.95)
        self.batch_size: int = data_cfg.get("batch_size", 16)
        self.num_workers: int = data_cfg.get("num_workers", 0)

        # Classes
        cls_cfg = self._raw.get("classes", {})
        self.all_classes: List[str] = cls_cfg.get("all_classes", ["background", "building", "woodland", "water", "road"])
        self.train_classes: List[str] = cls_cfg.get("train_classes", ["background", "building", "woodland", "water"])
        self.inference_classes: List[str] = cls_cfg.get("inference_classes", ["background", "building", "water"])

        # Model
        model_cfg = self._raw.get("model", {})
        self.arch: str = model_cfg.get("arch", "Unet")
        self.encoder: str = model_cfg.get("encoder", "efficientnet-b0")
        self.encoder_weights: str = model_cfg.get("encoder_weights", "imagenet")
        self.activation: str = model_cfg.get("activation", "softmax2d")
        self.checkpoint_name: str = model_cfg.get("checkpoint_name", "maprot_unet_efficientnet_b0.pth")

        # Training
        train_cfg = self._raw.get("training", {})
        self.epochs: int = train_cfg.get("epochs", 20)
        self.optimizer: str = train_cfg.get("optimizer", "Adam")
        self.init_lr: float = train_cfg.get("init_lr", 0.0003)
        self.lr_reduce_factor: float = train_cfg.get("lr_reduce_factor", 0.5)
        self.lr_patience: int = train_cfg.get("lr_patience", 5)
        self.lr_min: float = train_cfg.get("lr_min", 1e-6)

    def _resolve_path(self, p: str | Path) -> Path:
        path = Path(p)
        if not path.is_absolute():
            return (self.root_dir / path).resolve()
        return path.resolve()

    @property
    def checkpoint_path(self) -> Path:
        return self.weights_dir / self.checkpoint_name

    def to_dict(self) -> Dict[str, Any]:
        return self._raw
