"""
MapRot Model Training Pipeline.
Executes dataset tiling, DataLoader setup, optimization with lr-scheduling,
and model checkpointing.
"""

from pathlib import Path
from typing import Optional, Union
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils

from maprot.core.config import MapRotConfig
from maprot.core.logger import get_maprot_logger
from maprot.data.dataset import MapRotSegmentationDataset
from maprot.data.transforms import get_training_augmentation, get_preprocessing_pipeline
from maprot.models.factory import build_segmentation_model


class MapRotTrainer:
    """
    Orchestrates deep learning training workflows for MapRot segmentation models.
    """

    def __init__(self, config: MapRotConfig):
        self.cfg = config
        self.logger = get_maprot_logger(
            name="MapRot-Trainer",
            log_file=self.cfg.logs_dir / "maprot_train.log",
            level=self.cfg.log_level,
        )

        device_name = self.cfg.device if torch.cuda.is_available() and self.cfg.device != "cpu" else "cpu"
        self.device = torch.device(device_name)
        self.logger.info(f"Initialized MapRotTrainer on device: {self.device}")

    def run(self) -> Path:
        """
        Executes full training routine and saves the best model checkpoint.
        """
        self.logger.info("Initializing model architecture...")
        model = build_segmentation_model(
            arch=self.cfg.arch,
            encoder_name=self.cfg.encoder,
            encoder_weights=self.cfg.encoder_weights,
            classes=len(self.cfg.train_classes),
            activation=self.cfg.activation,
        ).to(self.device)

        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            self.cfg.encoder, self.cfg.encoder_weights
        )

        # Datasets
        train_img_dir = self.cfg.train_dir / "images"
        train_msk_dir = self.cfg.train_dir / "masks"

        if not train_img_dir.exists() or len(list(train_img_dir.glob("*.*"))) == 0:
            msg = f"Training images directory empty or not found: {train_img_dir}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        train_dataset = MapRotSegmentationDataset(
            images_dir=train_img_dir,
            masks_dir=train_msk_dir,
            all_classes=self.cfg.all_classes,
            classes=self.cfg.train_classes,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing_pipeline(preprocessing_fn),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )

        self.logger.info(f"Loaded {len(train_dataset)} training samples into {len(train_loader)} batches.")

        # Loss & Metrics
        loss = smp_utils.losses.DiceLoss()
        metrics = [
            smp_utils.metrics.IoU(threshold=0.5),
            smp_utils.metrics.Fscore(threshold=0.5),
        ]

        # Optimizer & Scheduler
        optimizer = getattr(torch.optim, self.cfg.optimizer)(
            params=model.parameters(), lr=self.cfg.init_lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.cfg.lr_reduce_factor,
            patience=self.cfg.lr_patience,
            min_lr=self.cfg.lr_min,
        )

        train_epoch = smp_utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=self.device,
            verbose=True,
        )

        best_score = 0.0
        self.cfg.weights_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_out = self.cfg.weights_dir / self.cfg.checkpoint_name

        self.logger.info(f"Beginning training for {self.cfg.epochs} epochs...")
        for epoch in range(1, self.cfg.epochs + 1):
            self.logger.info(f"Epoch: {epoch}/{self.cfg.epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
            train_logs = train_epoch.run(train_loader)

            current_score = train_logs["iou_score"]
            scheduler.step(current_score)

            if current_score > best_score:
                best_score = current_score
                torch.save(model, checkpoint_out)
                self.logger.info(f"✓ New best score ({best_score:.4f}). Saved checkpoint: {checkpoint_out.name}")

        self.logger.info(f"Training complete. Best IoU: {best_score:.4f}. Model stored at: {checkpoint_out}")
        return checkpoint_out
