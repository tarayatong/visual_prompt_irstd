from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


CONFIG_SUFFIXES = {".yaml", ".yml"}
CONTENT_DIRS = {"artifacts", "figures", "metadata", "metrics", "reports"}
FIGURE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".pdf", ".png", ".svg", ".webp"}
ARTIFACT_SUFFIXES = {".npy", ".npz", ".pkl", ".pt", ".pth"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def content_dir(path: Path) -> str:
    if path.name in {"command.txt", "config.resolved.yaml", "run_metadata.json"}:
        return "metadata"
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "metrics"
    if suffix in FIGURE_SUFFIXES:
        return "figures"
    if suffix in ARTIFACT_SUFFIXES:
        return "artifacts"
    if suffix in {".log", ".md", ".txt"}:
        return "reports"
    return "artifacts"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def collect_configs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.suffix.lower() in CONFIG_SUFFIXES
        )
    if path.suffix.lower() not in CONFIG_SUFFIXES:
        raise ValueError(f"Expected a YAML config or directory: {path}")
    return [path]


def as_cli_value(value):
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    if value is None:
        return "none"
    return str(value)


def append_args(command: list[str], args_map: dict | None):
    if not args_map:
        return
    for key, value in args_map.items():
        if isinstance(value, bool):
            command.append(f"--{key}" if value else f"--no-{key}")
        elif value is None:
            continue
        else:
            command.extend([f"--{key}", as_cli_value(value)])


def build_output(config: dict, config_path: Path, root: Path) -> tuple[Path, dict]:
    output = dict(config.get("output") or {})
    output_root = root / output.get("root", "outputs/runs/misc")
    subdir = output.get("subdir") or config.get("name") or config_path.stem
    output_dir = output_root / subdir
    output_arg = output.get("arg")
    output_file = output.get("file")

    output_args = {}
    if output_arg:
        if output_file:
            output_args[output_arg] = str(output_dir / content_dir(Path(output_file)) / output_file)
        else:
            output_args[output_arg] = str(output_dir)
    return output_dir, output_args


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def write_run_metadata(
    output_dir: Path,
    config_path: Path,
    config: dict,
    command: list[str],
    dry_run: bool,
):
    metadata_dir = output_dir / "metadata"
    if dry_run:
        metadata_dir = metadata_dir / "dry-run"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = metadata_dir / "config.resolved.yaml"
    with resolved_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    (metadata_dir / "command.txt").write_text(command_text(command) + "\n", encoding="utf-8")

    metadata = {
        "config_path": str(config_path),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "python": sys.executable,
        "cwd": str(repo_root()),
        "command": command,
    }
    (metadata_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def organize_output_files(output_dir: Path):
    files = [path for path in output_dir.rglob("*") if path.is_file()]
    for source in files:
        relative = source.relative_to(output_dir)
        if relative.parts[0] in CONTENT_DIRS:
            continue
        destination = output_dir / content_dir(source) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)

    directories = sorted(
        (path for path in output_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        if directory.parent == output_dir and directory.name in CONTENT_DIRS:
            continue
        try:
            directory.rmdir()
        except OSError:
            pass


def run_one(config_path: Path, dry_run: bool, print_command: bool) -> int:
    root = repo_root()
    config = load_config(config_path)
    module = config.get("module")
    script = config.get("script")
    if not module and not script:
        raise ValueError(f"Missing required key 'module' or 'script' in {config_path}")

    if module:
        command = [sys.executable, "-m", module]
    else:
        script_path = root / script
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found for {config_path}: {script_path}")
        command = [sys.executable, str(script_path)]

    output_dir, output_args = build_output(config, config_path, root)
    args_map = dict(config.get("args") or {})
    args_map.update(output_args)

    append_args(command, args_map)
    append_args(command, config.get("flags") or {})

    if print_command or dry_run:
        print(command_text(command))
    if dry_run:
        return 0

    write_run_metadata(output_dir, config_path, config, command, dry_run=False)

    completed = subprocess.run(command, cwd=str(root))
    organize_output_files(output_dir)
    return int(completed.returncode)


def parse_args():
    parser = argparse.ArgumentParser("Run visual_prompt experiments from YAML configs.")
    parser.add_argument("--config", required=True, type=str, help="YAML config file or directory.")
    parser.add_argument("--dry-run", action="store_true", help="Write metadata and print command only.")
    parser.add_argument("--print-command", action="store_true", help="Print each resolved command before running.")
    parser.add_argument("--keep-going", action="store_true", help="Continue when one config fails.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_paths = collect_configs((repo_root() / args.config).resolve())
    if not config_paths:
        raise RuntimeError(f"No YAML configs found under {args.config}")

    failures = []
    for config_path in config_paths:
        print(f"[experiment] {config_path.relative_to(repo_root())}")
        code = run_one(config_path, dry_run=args.dry_run, print_command=args.print_command)
        if code != 0:
            failures.append((config_path, code))
            if not args.keep_going:
                break

    if failures:
        for path, code in failures:
            print(f"[failed] {path} exit={code}", file=sys.stderr)
        raise SystemExit(failures[0][1])


if __name__ == "__main__":
    main()
