#!/usr/bin/env python3
"""Train the final TRACE model from a published experiment configuration."""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training import Trainer
from utils import project_root, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRACE")
    parser.add_argument(
        "--config", default="src/configs/example.yaml"
    )
    parser.add_argument("--resume", default=None, help="Checkpoint used to resume training")
    args = parser.parse_args()

    config_path = resolve_path(args.config, base_dir=str(project_root()))
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    trainer = Trainer(config)
    if args.resume:
        trainer.load_model(resolve_path(args.resume, base_dir=str(project_root())))
    try:
        trainer.train()
    except KeyboardInterrupt:
        output = os.path.join(trainer.checkpoint_dir, "interrupted_model.pth")
        trainer.save_model(output)
        trainer.logger.info("Training interrupted; checkpoint saved to %s", output)


if __name__ == "__main__":
    main()
