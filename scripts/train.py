"""
MapRot Training Script.
Usage: python scripts/train.py [configs/maprot_default.yaml]
"""

import sys
from pathlib import Path

# Resolve project root from scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maprot.core.config import MapRotConfig
from maprot.pipelines.trainer import MapRotTrainer


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/maprot_default.yaml"
    config = MapRotConfig.from_yaml(config_path, root_dir=PROJECT_ROOT)
    trainer = MapRotTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
