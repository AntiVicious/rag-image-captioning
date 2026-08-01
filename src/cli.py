"""
Command-line interface for the RAG Image Captioning pipeline.

Usage:
    python -m src.cli setup
    python -m src.cli build-db
    python -m src.cli caption --image path/to/image.jpg [--skip-detr]
    python -m src.cli --backend blip caption --image path/to/image.jpg
"""
import argparse
import sys

from .config import Config
from .pipeline import Pipeline


def _build_config(args: argparse.Namespace) -> Config:
    config = Config()
    if args.backend:
        config.caption_backend = args.backend
    if getattr(args, "skip_detr", False):
        config.enable_segmentation = False
        config.enable_object_detection = False
    return config


def cmd_setup(args: argparse.Namespace) -> int:
    pipeline = Pipeline(_build_config(args))
    pipeline.setup()
    print("Setup complete.")
    return 0


def cmd_build_db(args: argparse.Namespace) -> int:
    pipeline = Pipeline(_build_config(args))
    pipeline.setup()
    if not pipeline.build_database_from_coco():
        print("Build failed: COCO dataset not found (see Config.coco_img_dir / coco_ann_file).")
        return 1
    stats = pipeline.db_manager.get_stats()
    print(f"Database built: {stats['total_embeddings']} embeddings in '{stats['collection_name']}'.")
    return 0


def cmd_caption(args: argparse.Namespace) -> int:
    pipeline = Pipeline(_build_config(args))
    pipeline.setup()
    result = pipeline.caption_image(args.image, use_advanced=not args.skip_detr)
    print(f"Backend: {result['backend']}")
    print(f"Caption: {result['generated_caption']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG Image Captioning")
    parser.add_argument(
        "--backend",
        choices=["retrieval", "blip"],
        default=None,
        help="Override Config.caption_backend (default: retrieval)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("setup", help="Load models and initialize the database")
    subparsers.add_parser("build-db", help="Build the ChromaDB index from the COCO dataset")

    caption_parser = subparsers.add_parser("caption", help="Caption a single image")
    caption_parser.add_argument("--image", required=True, help="Path to an image file")
    caption_parser.add_argument(
        "--skip-detr", action="store_true", help="Skip segmentation/object-detection crops"
    )

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    commands = {"setup": cmd_setup, "build-db": cmd_build_db, "caption": cmd_caption}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
