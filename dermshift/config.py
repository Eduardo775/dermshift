"""Configuration loading: ``base.yaml`` <- model file <- ``configs/local.yaml`` <- dotted overrides.

The resolved configuration is saved in every run folder.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"Unknown config section '{k}' in override '{dotted}'")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"Unknown config key '{dotted}'")
    node[keys[-1]] = value


def _load_tree(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_rel = cfg.pop("base", None)
    if base_rel is not None:
        cfg = deep_merge(_load_tree(path.parent / base_rel), cfg)
    return cfg


def find_local_yaml(path: Path) -> Path | None:
    """configs/local.yaml next to base.yaml (machine-specific paths and workers), if present."""
    for d in (path.parent, path.parent.parent):
        f = d / "local.yaml"
        if f.exists() and (d / "base.yaml").exists():
            return f
    return None


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None, use_local: bool = True) -> dict:
    """base.yaml <- experiment YAML <- configs/local.yaml <- dotted overrides."""
    path = Path(path)
    cfg = _load_tree(path)
    loc = find_local_yaml(path) if use_local else None
    if loc is not None:
        with open(loc, "r", encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for k, v in (overrides or {}).items():
        set_dotted(cfg, k, v)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def config_hash(cfg: dict) -> str:
    """Short hash of the settings that affect training; a run refuses to resume if it changes."""
    import hashlib

    c = copy.deepcopy(cfg)
    c.pop("paths", None)
    c.get("data", {}).pop("num_workers", None)
    c.get("run", {}).pop("checkpoint_minutes", None)
    c.get("run", {}).pop("channels_last", None)      # speed-only settings
    c.get("run", {}).pop("deterministic", None)
    for k in ("sets", "nsd_tolerances", "nsd_tolerances_pct", "metrics_version"):   # evaluation only
        c.get("eval", {}).pop(k, None)
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]
