#!/usr/bin/env python3
import argparse
import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def import_module_from_path(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Script not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_ddqn():
    script_path = PROJECT_ROOT / "RL" / "train_ddqn.py"
    module = import_module_from_path("train_ddqn", script_path)

    if hasattr(module, "train"):
        module.train()
    else:
        raise AttributeError(f"'train' function not found in {script_path}")


def run_ppo():
    script_path = PROJECT_ROOT / "RL" / "train_ppo.py"
    module = import_module_from_path("train_ppo", script_path)

    if hasattr(module, "train"):
        module.train()
    else:
        raise AttributeError(f"'train' function not found in {script_path}")


def run_yolo():
    script_path = PROJECT_ROOT / "pest detection" / "pobed_yolo.py"
    module = import_module_from_path("pobed_yolo", script_path)

    if hasattr(module, "main"):
        module.main()
    else:
        raise AttributeError(
            f"'main' function not found in {script_path}. "
            "Wrap the YOLO script in a main() function."
        )


def run_cnn():
    script_path = PROJECT_ROOT / "pest detection" / "Potato_disease_cnn.py"
    module = import_module_from_path("Potato_disease_cnn", script_path)

    if hasattr(module, "main"):
        module.main()
    else:
        raise AttributeError(
            f"'main' function not found in {script_path}. "
            "Wrap the CNN script in a main() function."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Unified entry point for UAV-IQA-RL-Control"
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["ddqn", "ppo", "yolo", "cnn"],
        help="Task to run"
    )

    args = parser.parse_args()

    try:
        if args.task == "ddqn":
            print("Running DDQN training...")
            run_ddqn()

        elif args.task == "ppo":
            print("Running PPO training...")
            run_ppo()

        elif args.task == "yolo":
            print("Running YOLO training...")
            run_yolo()

        elif args.task == "cnn":
            print("Running CNN training...")
            run_cnn()

    except Exception as e:
        print(f"\nExecution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
