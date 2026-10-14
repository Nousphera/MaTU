"""Command-line entry point for deterministic sequential simulation."""

import argparse
import json
from pathlib import Path
from .training import DEFAULTS, run


def main():
    parser = argparse.ArgumentParser(description="MaTU many-task federated training")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--flower",
        action="store_true",
        help="Run the Flower App through local Ray workers",
    )
    args = parser.parse_args()
    config = {**DEFAULTS, **json.loads(args.config.read_text())}
    if args.output:
        config["output"] = str(args.output)
    if args.flower:
        from .flower import simulate

        simulate(config)
    else:
        print(json.dumps(run(config), indent=2))


if __name__ == "__main__":
    main()
