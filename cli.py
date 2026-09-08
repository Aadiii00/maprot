"""
MapRot Unified Command Line Interface (CLI).
Provides commands for training, evaluation benchmarking, raster prediction, and web demo.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    # Common argument parser for inherited options like --config
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        type=str,
        default="configs/maprot_default.yaml",
        help="Path to YAML configuration file (default: configs/maprot_default.yaml)",
    )

    parser = argparse.ArgumentParser(
        prog="maprot",
        parents=[base_parser],
        description="MapRot: Geospatial Surface Analytics & Semantic Segmentation Framework",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available MapRot workflows")

    # Command: train
    train_parser = subparsers.add_parser(
        "train",
        parents=[base_parser],
        help="Train a segmentation model on geospatial data",
    )
    train_parser.add_argument("--epochs", type=int, help="Override number of training epochs")
    train_parser.add_argument("--batch-size", type=int, help="Override training batch size")

    # Command: evaluate
    eval_parser = subparsers.add_parser(
        "evaluate",
        parents=[base_parser],
        help="Benchmark model performance on test dataset",
    )
    eval_parser.add_argument("--weights", type=str, help="Custom checkpoint path to evaluate")

    # Command: predict
    predict_parser = subparsers.add_parser(
        "predict",
        parents=[base_parser],
        help="Execute inference on satellite/aerial rasters",
    )
    predict_parser.add_argument("--input", type=str, help="Directory containing rasters to process")
    predict_parser.add_argument("--weights", type=str, help="Custom checkpoint path for inference")
    predict_parser.add_argument(
        "--classes",
        nargs="+",
        help="Selective class prompt/filter (e.g. --classes building water)",
    )
    predict_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generating side-by-side visualization plots",
    )
    predict_parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable 4-fold Test-Time Augmentation (TTA) multi-flip ensemble",
    )
    predict_parser.add_argument(
        "--no-geojson",
        action="store_true",
        help="Disable exporting GIS vector polygons to GeoJSON",
    )

    # Command: demo
    demo_parser = subparsers.add_parser(
        "demo",
        parents=[base_parser],
        help="Launch the interactive MapRot web dashboard and visual demo",
    )
    demo_parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    demo_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    from maprot.core.config import MapRotConfig
    config = MapRotConfig.from_yaml(args.config, root_dir=PROJECT_ROOT)

    if args.command == "train":
        from maprot.pipelines.trainer import MapRotTrainer
        if args.epochs:
            config.epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        trainer = MapRotTrainer(config)
        trainer.run()

    elif args.command == "evaluate":
        from maprot.pipelines.evaluator import MapRotEvaluator
        evaluator = MapRotEvaluator(config)
        evaluator.evaluate(checkpoint_path=args.weights)

    elif args.command == "predict":
        from maprot.pipelines.predictor import MapRotPredictor
        predictor = MapRotPredictor(config)
        predictor.predict_directory(
            input_dir=args.input,
            selected_classes=args.classes,
            checkpoint_path=args.weights,
            use_tta=args.tta,
            export_vector_geojson=not args.no_geojson,
            save_plots=not args.no_plots,
        )

    elif args.command == "demo":
        from app import start_server
        start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
