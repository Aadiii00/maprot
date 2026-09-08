"""
MapRot System Constants.
Defines supported land-cover classes, visual color mappings, and filesystem conventions.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Supported land-cover categories in the MapRot benchmark
MAPROT_CLASSES: Tuple[str, ...] = (
    "background",
    "building",
    "woodland",
    "water",
    "road",
)

CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(MAPROT_CLASSES)}
INDEX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(MAPROT_CLASSES)}

# High-contrast, publication-quality RGB color palette for segmentation visualization
# background: Charcoal dark / transparent
# building: Crimson red
# woodland: Emerald forest green
# water: Azure blue
# road: Amber gold
MAPROT_COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "background": (30, 30, 30),
    "building": (220, 50, 47),
    "woodland": (46, 204, 113),
    "water": (52, 152, 219),
    "road": (241, 196, 15),
}

# Color palette indexed by class ID (RGB)
PALETTE_RGB: List[Tuple[int, int, int]] = [
    MAPROT_COLOR_PALETTE[name] for name in MAPROT_CLASSES
]

DEFAULT_CONFIG_PATH = Path("configs/maprot_default.yaml")
