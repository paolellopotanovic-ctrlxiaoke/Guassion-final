#!/usr/bin/env python3
"""Unified command-line entrypoint for Gaussian-direct128."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gaussian_direct.app import GaussianDirectApp
from gaussian_direct.config import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train with the typed config")
    _add_data_arguments(train)
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--devices", type=int)
    train.add_argument("--accelerator", choices=("auto", "cpu", "gpu"))
    train.add_argument("--precision", choices=("16-mixed", "32"))

    infer = subparsers.add_parser("infer", help="Predict residue probabilities")
    _add_data_arguments(infer)
    infer.add_argument("--checkpoint", type=Path)
    infer.add_argument("--npz", action="append", type=Path)
    infer.add_argument("--residue-ids")
    infer.add_argument("--output", type=Path)
    infer.add_argument("--device")
    infer.add_argument("--batch-size", type=int)
    infer.add_argument("--threshold", type=float)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a compact-NPZ panel")
    _add_data_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--device")
    evaluate.add_argument("--batch-size", type=int)

    subparsers.add_parser("verify", help="Verify configured release assets")
    subparsers.add_parser("show-config", help="Print the resolved config as JSON")
    preprocess = subparsers.add_parser("preprocess", help="Build compact training NPZ files")
    preprocess.add_argument("--manifest", type=Path)
    preprocess.add_argument("--output-dir", type=Path)
    preprocess.add_argument("--workers", type=int)
    preprocess.add_argument("--torch-threads", type=int)
    preprocess.add_argument("--max-num-nn", type=int)
    preprocess.add_argument("--payload", choices=("base", "training"))
    preprocess.add_argument("--limit", type=int)
    preprocess.add_argument("--progress-every", type=int)
    preprocess.add_argument("--chunksize", type=int)
    preprocess.add_argument("--trust-final-filter", action="store_true")
    preprocess.add_argument("--overwrite", action="store_true")
    return parser


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--compact-dir", type=Path)


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig.from_file(args.config)
    if args.command == "train":
        training_data = replace(
            config.training.data,
            **{
                key: value
                for key, value in {
                    "train_manifest": args.manifest,
                    "train_compact_dir": args.compact_dir,
                }.items()
                if value is not None
            },
        )
        training = replace(config.training, data=training_data)
        if args.output_dir is not None:
            training = replace(training, output_dir=args.output_dir)
        runtime = config.runtime
        if args.devices is not None:
            runtime = replace(runtime, devices=args.devices)
        if args.accelerator is not None:
            runtime = replace(runtime, accelerator=args.accelerator)
        if args.precision is not None:
            runtime = replace(runtime, precision=args.precision)
        config = replace(config, training=training, runtime=runtime)
    elif args.command in {"infer", "evaluate"}:
        current = config.inference if args.command == "infer" else config.evaluation
        fields = {
            "manifest": args.manifest,
            "compact_dir": args.compact_dir,
            "checkpoint": args.checkpoint,
            "output": args.output,
            "device": args.device,
            "batch_size": args.batch_size,
        }
        if args.command == "infer":
            fields.update(
                {
                    "npz": tuple(args.npz) if args.npz else current.npz,
                    "residue_ids": args.residue_ids,
                    "threshold": args.threshold,
                }
            )
        current = replace(
            current,
            **{key: value for key, value in fields.items() if value is not None},
        )
        if args.command == "infer":
            config = replace(config, inference=current, evaluation=replace(config.evaluation, checkpoint=current.checkpoint))
        else:
            config = replace(config, evaluation=current, inference=replace(config.inference, checkpoint=current.checkpoint))

    app = GaussianDirectApp(config)
    if args.command == "train":
        app.train()
    elif args.command == "infer":
        app.infer()
    elif args.command == "evaluate":
        app.evaluate()
    elif args.command == "verify":
        print(json.dumps(app.verify(), indent=2))
    elif args.command == "preprocess":
        preprocessing = replace(
            config.preprocessing,
            **{
                key: value
                for key, value in {
                    "manifest": args.manifest,
                    "output_dir": args.output_dir,
                    "workers": args.workers,
                    "torch_threads": args.torch_threads,
                    "max_num_nn": args.max_num_nn,
                    "payload": args.payload,
                    "limit": args.limit,
                    "progress_every": args.progress_every,
                    "chunksize": args.chunksize,
                    "trust_final_filter": args.trust_final_filter,
                    "overwrite": args.overwrite,
                }.items()
                if value is not None
            },
        )
        GaussianDirectApp(replace(config, preprocessing=preprocessing)).preprocess()
    else:
        print(json.dumps(config.to_dict(), indent=2))


if __name__ == "__main__":
    main()
