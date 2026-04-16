from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIRS = {
    "trajectory": PROJECT_ROOT / "DeepRUOT",
    "embedding": PROJECT_ROOT / "autoencoder",
    "classification": PROJECT_ROOT / "classification",
    "regression": PROJECT_ROOT / "regression",
}

LEGACY_DIRS = {
    category: path / "legacy"
    for category, path in SOURCE_DIRS.items()
}


def available_scripts(category: str) -> list[str]:
    source_dir = SOURCE_DIRS[category]
    return sorted(
        path.stem
        for path in source_dir.glob("*.py")
        if path.name != "__init__.py"
    )


def resolve_script(category: str, script_name: str) -> Path:
    normalized = script_name if script_name.endswith(".py") else f"{script_name}.py"
    script_path = SOURCE_DIRS[category] / normalized
    if not script_path.exists():
        known = ", ".join(available_scripts(category))
        raise FileNotFoundError(
            f"Unknown {category} script: {script_name}. Available scripts: {known}"
        )
    return script_path


def available_legacy_scripts(category: str) -> list[str]:
    legacy_dir = LEGACY_DIRS[category]
    if not legacy_dir.exists():
        return []
    return sorted(
        path.stem
        for path in legacy_dir.glob("*.py")
        if path.name != "__init__.py"
    )


def resolve_legacy_script(category: str, script_name: str) -> Path:
    normalized = script_name if script_name.endswith(".py") else f"{script_name}.py"
    script_path = LEGACY_DIRS[category] / normalized
    if not script_path.exists():
        known = ", ".join(available_legacy_scripts(category))
        raise FileNotFoundError(
            f"Unknown legacy {category} script: {script_name}. Available scripts: {known}"
        )
    return script_path


def _normalize_passthrough_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def run_legacy_script(category: str, script_name: str, script_args: list[str]) -> None:
    script_path = resolve_script(category, script_name)
    cmd = [sys.executable, str(script_path), *_normalize_passthrough_args(script_args)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), str(SOURCE_DIRS[category]), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def run_archived_script(category: str, script_name: str, script_args: list[str]) -> None:
    script_path = resolve_legacy_script(category, script_name)
    cmd = [sys.executable, str(script_path), *_normalize_passthrough_args(script_args)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(PROJECT_ROOT),
            str(SOURCE_DIRS[category]),
            str(LEGACY_DIRS[category]),
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _ensure_sys_path(path: Path) -> None:
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _clear_modules(names: list[str]) -> None:
    for name in names:
        sys.modules.pop(name, None)


def run_trajectory_training(config_path: str, evaluate: bool = False) -> dict[str, Any]:
    _ensure_sys_path(SOURCE_DIRS["trajectory"])
    _clear_modules(["train_RUOT"])

    train_module = importlib.import_module("train_RUOT")
    utils_module = importlib.import_module("DeepRUOT.utils")

    resolved_config = str(Path(config_path).expanduser().resolve())
    config = utils_module.load_and_merge_config(resolved_config)

    pipeline = train_module.TrainingPipeline(config)
    pipeline.train()

    evaluation_rows = 0
    if evaluate:
        evaluation = pipeline.evaluate()
        evaluation_rows = len(evaluation)

    return {
        "config": resolved_config,
        "exp_dir": str(pipeline.exp_dir),
        "evaluated": evaluate,
        "evaluation_rows": evaluation_rows,
    }


def run_embedding_training(config: dict[str, Any]) -> None:
    _ensure_sys_path(SOURCE_DIRS["embedding"])
    _clear_modules(
        [
            "train_model",
            "dataio",
            "utils",
            "netmodel",
            "models",
            "models.layers",
            "models.dgi",
            "models.encoder_decoder",
        ]
    )

    train_module = importlib.import_module("train_model")
    cfg = train_module.TrainConfig(**config)
    train_module.main(cfg)
