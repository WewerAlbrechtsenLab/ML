from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import sklearn

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .config import PipelineConfig


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _to_serializable(v) for k, v in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def _safe_write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _hash_bytes(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        if chunk:
            digest.update(chunk)
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    if not path or not path.exists():
        return "missing"
    with path.open("rb") as handle:
        return _hash_bytes(iter(lambda: handle.read(1024 * 1024), b""))


def _hash_config(config: PipelineConfig) -> str:
    payload = _to_serializable(config.to_dict())
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _get_git_sha(cwd: Path) -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL)
    except Exception:
        return "unknown"
    return output.decode("utf-8").strip()


def _compute_run_id(config_hash: str, data_hash: str, git_sha: str) -> str:
    combined = "|".join([config_hash, data_hash, git_sha])
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:12]


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _resolve_path(
    candidate: str | Path,
    bases: Sequence[Path] | Path,
    *,
    allow_outside: bool = True,
    require_exists: bool = False,
) -> Path:
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        return candidate_path

    if isinstance(bases, Path):
        base_list = [bases]
    else:
        base_list = [Path(base) for base in bases]

    if not base_list:
        base_list = [Path.cwd()]

    primary_base = base_list[0]

    selected: Path | None = None
    for base in base_list:
        resolved = (base / candidate_path).resolve()
        if not allow_outside and not _is_within(resolved, primary_base):
            continue
        if require_exists and not resolved.exists():
            continue
        selected = resolved
        if resolved.exists() or not require_exists:
            break

    if selected is None:
        selected = (primary_base / candidate_path).resolve()
        if not allow_outside and not _is_within(selected, primary_base):
            selected = (primary_base / candidate_path.name).resolve()

    return selected


def _prepare_run_dir(root: Path, run_id: str) -> Path:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _select_best_model(leaderboard: pd.DataFrame, models: Dict[str, BaseEstimator]) -> tuple[str | None, BaseEstimator | None, Dict[str, Any] | None]:
    if leaderboard.empty:
        return None, None, None
    best_row = leaderboard.iloc[0].to_dict()
    best_model_name = best_row.get("model")
    best_model = models.get(best_model_name)
    return best_model_name, best_model, best_row


def log_training_run(
    *,
    config: PipelineConfig,
    leaderboard: pd.DataFrame,
    trained_models: Dict[str, BaseEstimator],
    project_root: Path | None = None,
) -> Path:
    project_root = project_root or Path(__file__).resolve().parents[2]
    base_candidates = [project_root]
    config_dir = project_root / "config"
    if config_dir.exists():
        base_candidates.append(config_dir)
    notebooks_dir = project_root / "notebooks"
    if notebooks_dir.exists():
        base_candidates.append(notebooks_dir)
    output_dir = _resolve_path(config.output_dir, base_candidates, allow_outside=False)
    config_hash = _hash_config(config)
    data_path = _resolve_path(
        config.data_path,
        base_candidates + [Path.cwd()],
        allow_outside=True,
        require_exists=True,
    )
    data_hash = _hash_file(data_path)
    git_sha = _get_git_sha(project_root)
    run_id = _compute_run_id(config_hash, data_hash, git_sha)
    run_dir = _prepare_run_dir(output_dir, run_id)

    if yaml is not None:
        config_yaml = yaml.safe_dump(config.to_dict(), sort_keys=False)
    else:  # pragma: no cover
        config_yaml = json.dumps(_to_serializable(config.to_dict()), indent=2)
    _safe_write_text(run_dir / "config.yaml", config_yaml)
    _safe_write_text(run_dir / "git_sha.txt", git_sha + "\n")
    _safe_write_text(run_dir / "data_hash.txt", data_hash + "\n")

    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)

    best_model_name, best_model, best_row = _select_best_model(leaderboard, trained_models)
    metrics_payload: Dict[str, Any] = {
        "best_model": _to_serializable(best_row) if best_row else None,
        "all_models": [_to_serializable(record) for record in leaderboard.to_dict(orient="records")],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, default=_json_default), encoding="utf-8")

    if best_model is not None:
        joblib.dump(best_model, run_dir / "best_model.joblib")

    manifest_payload: Dict[str, Any] = {
        "run_id": run_id,
        "run_directory": str(run_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": _safe_get_user(),
        "random_state": config.random_state,
        "git_sha": git_sha,
        "data_hash": data_hash,
        "config_hash": config_hash,
        "environment": {
            "python": _get_python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest_payload, indent=2, default=_json_default), encoding="utf-8")
    return run_dir


def _get_python_version() -> str:
    import platform

    return platform.python_version()


def _safe_get_user() -> str:
    try:
        import getpass

        return getpass.getuser()
    except Exception:
        return "unknown"
