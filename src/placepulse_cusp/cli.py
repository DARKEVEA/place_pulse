from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from placepulse_cusp.config import load_config
from placepulse_cusp.data.fetch import fetch_data
from placepulse_cusp.data.schema import standardise_votes
from placepulse_cusp.data.splits import prepare_data
from placepulse_cusp.data.validate import validate_votes
from placepulse_cusp.hardware import device_report, gpu_smoke
from placepulse_cusp.pipeline import run_all, run_cusp_stage, run_dimension
from placepulse_cusp.reporting.report import build_report
from placepulse_cusp.simulation.generate import generate_vote_table
from placepulse_cusp.simulation.recovery import (
    validate_density_recovery,
    validate_model_recovery,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ppc", description="Place Pulse CUSP experiment")
    sub = parser.add_subparsers(dest="group", required=True)

    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="action", required=True)
    fetch = data_sub.add_parser("fetch")
    fetch.add_argument("--source")
    for name in ("validate", "prepare"):
        data_sub.add_parser(name)

    simulate = sub.add_parser("simulate")
    simulate_sub = simulate.add_subparsers(dest="action", required=True)
    generate = simulate_sub.add_parser("generate")
    generate.add_argument(
        "--mechanism",
        choices=("null", "scalar", "continuous", "mixture", "cusp"),
        default="mixture",
    )
    simulate_sub.add_parser("validate-models")
    simulate_sub.add_parser("validate-density")

    run = sub.add_parser("run")
    run_sub = run.add_subparsers(dest="action", required=True)
    for name in ("scalar", "heterogeneity", "bimodality", "cusp", "all"):
        run_sub.add_parser(name)

    report = sub.add_parser("report")
    report_sub = report.add_subparsers(dest="action", required=True)
    report_sub.add_parser("build")

    clean = sub.add_parser("clean")
    clean_sub = clean.add_subparsers(dest="action", required=True)
    clean_sub.add_parser("artifacts")

    gpu = sub.add_parser("gpu")
    gpu_sub = gpu.add_subparsers(dest="action", required=True)
    gpu_check = gpu_sub.add_parser("check")
    gpu_check.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    gpu_benchmark = gpu_sub.add_parser("benchmark")
    gpu_benchmark.add_argument("--device", choices=("mps", "cuda"), required=True)
    gpu_benchmark.add_argument("--size", type=int, default=2048)
    gpu_benchmark.add_argument("--iterations", type=int, default=5)

    for command in (fetch, generate):
        command.add_argument("--config", default="configs/confirmatory.yaml")
        command.add_argument("--resume", action="store_true")
    for group_parser in (data, simulate, run, report, clean, gpu):
        group_parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    # Accept --config after every leaf command.
    for leaf in [
        *[x for x in data_sub.choices.values()],
        *[x for x in simulate_sub.choices.values()],
        *[x for x in run_sub.choices.values()],
        *[x for x in report_sub.choices.values()],
        *[x for x in clean_sub.choices.values()],
        *[x for x in gpu_sub.choices.values()],
    ]:
        if not any(action.dest == "config" for action in leaf._actions):
            leaf.add_argument("--config", default="configs/confirmatory.yaml")
        if not any(action.dest == "resume" for action in leaf._actions):
            leaf.add_argument("--resume", action="store_true")
    return parser


def _ensure_prepared(config):
    interim = Path(config["data"]["interim_dir"]) / "votes.parquet"
    processed = Path(config["data"]["processed_dir"]) / "votes.parquet"
    split_manifest = Path(config["data"]["processed_dir"]) / "splits.json"
    if not interim.exists():
        standardise_votes(config)
    validation = validate_votes(config, interim)
    if validation["status"] != "ok":
        raise RuntimeError("DATA_INSUFFICIENT: " + ", ".join(validation["reasons"]))
    split_current = False
    if split_manifest.exists():
        try:
            split_current = (
                json.loads(split_manifest.read_text("utf-8")).get(
                    "split_schema_version"
                )
                == 2
            )
        except (OSError, json.JSONDecodeError):
            split_current = False
    if not processed.exists() or not split_current:
        prepare_data(config)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config or "configs/confirmatory.yaml")
    if args.group == "data":
        if args.action == "fetch":
            result = fetch_data(config, args.source)
        elif args.action == "validate":
            if not args.resume or not (
                Path(config["data"]["interim_dir"]) / "votes.parquet"
            ).exists():
                standardise_votes(config)
            result = validate_votes(config)
        else:
            if not args.resume or not (
                Path(config["data"]["interim_dir"]) / "votes.parquet"
            ).exists():
                standardise_votes(config)
            result = {"outputs": [str(path) for path in prepare_data(config)]}
    elif args.group == "simulate":
        if args.action == "generate":
            result = {"path": str(generate_vote_table(config, mechanism=args.mechanism))}
        elif args.action == "validate-models":
            result = validate_model_recovery(config, resume=args.resume)
        else:
            result = validate_density_recovery(config, resume=args.resume)
    elif args.group == "run":
        if args.action == "all":
            result = run_all(config, resume=args.resume)
        elif args.action in {"scalar", "heterogeneity"}:
            _ensure_prepared(config)
            calibration_root = config["simulation"].get("calibration_artifacts")
            metrics_root = (
                Path(calibration_root)
                if calibration_root
                else Path(config["reporting"]["artifacts_dir"]) / "metrics"
            )
            model_path = metrics_root / "model_recovery.json"
            density_path = metrics_root / "simulation_recovery.json"
            if not model_path.exists() or not density_path.exists():
                raise RuntimeError(
                    "Model and density calibration must be completed before a "
                    "real-data heterogeneity run."
                )
            model_status = json.loads(model_path.read_text("utf-8"))["status"]
            density_status = json.loads(density_path.read_text("utf-8"))["status"]
            config["_simulation_ok"] = model_status == "ok" and density_status == "ok"
            result = run_dimension(
                config,
                config["data"]["primary_dimension"],
                detailed=True,
                resume=args.resume,
            )[0]
        else:
            _ensure_prepared(config)
            metrics_path = (
                Path(config["reporting"]["artifacts_dir"])
                / "metrics"
                / f"{config['data']['primary_dimension']}_model_comparison.json"
            )
            if not metrics_path.exists():
                primary = run_dimension(
                    config, config["data"]["primary_dimension"], detailed=True
                )[0]
            else:
                primary = json.loads(metrics_path.read_text("utf-8"))
            verdict, payload = run_cusp_stage(config, primary)
            result = {"verdict": verdict, "details": payload}
    elif args.group == "report":
        result = {"path": str(build_report(config))}
    elif args.group == "clean":
        target = Path(config["reporting"]["artifacts_dir"])
        if target.resolve() in {Path("/").resolve(), Path.home().resolve()}:
            raise RuntimeError(f"Refusing unsafe clean target: {target}")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        (target / ".gitkeep").touch()
        result = {"cleaned": str(target)}
    elif args.group == "gpu":
        if args.action == "check":
            result = device_report(args.device)
        else:
            result = {
                "report": device_report(args.device),
                "benchmark": gpu_smoke(args.device, args.size, args.iterations),
            }
    else:
        raise AssertionError("unreachable")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
