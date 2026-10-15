#!/usr/bin/env python3
"""Prepare the pinned pretrained model before starting clients."""

import argparse
import json

from matu.model_preparation import DEFAULT_WEIGHTS_DIR, prepare_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=DEFAULT_WEIGHTS_DIR, help="Checkpoint directory"
    )
    args = parser.parse_args()
    print(json.dumps(prepare_model(args.output), indent=2))


if __name__ == "__main__":
    main()
