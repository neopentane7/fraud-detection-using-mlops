"""Tests for the champion/challenger promotion logic in scripts/promote_model.py.

The retrain workflow can't exercise the `>= min_delta` comparison branch (a CI
run starts from a fresh registry with no Production champion, so it promotes
unconditionally). These unit tests cover the comparison logic directly — both
the "challenger wins" and "challenger does not win" outcomes — so the gate that
protects Production is genuinely validated somewhere.
"""

from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest

# scripts/ is not an installed package; load the module from its file path.
_PM_PATH = Path(__file__).resolve().parents[1] / "scripts" / "promote_model.py"
_spec = importlib.util.spec_from_file_location("promote_model", _PM_PATH)
assert _spec and _spec.loader
pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pm)


def _write_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, f1: float) -> None:
    path = tmp_path / "eval_metrics.json"
    path.write_text(json.dumps({"f1_fraud": f1}), encoding="utf-8")
    monkeypatch.setattr(pm, "EVAL_METRICS_PATH", path)


def _fake_client_with_prod_f1(prod_f1: float) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        get_model_version=lambda name, v: types.SimpleNamespace(run_id="r1"),
        get_run=lambda rid: types.SimpleNamespace(
            data=types.SimpleNamespace(metrics={"f1_fraud": prod_f1})
        ),
    )


def test_no_eval_metrics_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No holdout metrics file -> no-op, never promotes."""
    monkeypatch.setattr(pm, "EVAL_METRICS_PATH", tmp_path / "missing.json")
    promoted = {"called": False}
    monkeypatch.setattr(
        pm, "promote_staging_to_production", lambda: promoted.update(called=True) or 0
    )
    assert pm.compare_and_promote(0.02) == 0
    assert promoted["called"] is False


def test_no_champion_promotes_unconditionally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no Production champion, the challenger is promoted unconditionally."""
    _write_eval(tmp_path, monkeypatch, f1=0.50)
    monkeypatch.setattr(pm, "_latest_version", lambda stage: None)
    promoted = {"called": False}
    monkeypatch.setattr(
        pm, "promote_staging_to_production", lambda: promoted.update(called=True) or 0
    )
    pm.compare_and_promote(0.02)
    assert promoted["called"] is True


def test_promotes_when_challenger_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Challenger beating the champion by >= min_delta is promoted."""
    _write_eval(tmp_path, monkeypatch, f1=0.80)  # vs champion 0.70 -> +0.10
    monkeypatch.setattr(pm, "_latest_version", lambda stage: "1")
    monkeypatch.setattr(pm, "_client", lambda: _fake_client_with_prod_f1(0.70))
    promoted = {"called": False}
    monkeypatch.setattr(
        pm, "promote_staging_to_production", lambda: promoted.update(called=True) or 0
    )
    pm.compare_and_promote(0.02)
    assert promoted["called"] is True


def test_keeps_champion_when_not_better(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Challenger that doesn't clear min_delta must NOT be promoted."""
    _write_eval(tmp_path, monkeypatch, f1=0.71)  # vs champion 0.70 -> +0.01 < 0.02
    monkeypatch.setattr(pm, "_latest_version", lambda stage: "1")
    monkeypatch.setattr(pm, "_client", lambda: _fake_client_with_prod_f1(0.70))
    promoted = {"called": False}
    monkeypatch.setattr(
        pm, "promote_staging_to_production", lambda: promoted.update(called=True) or 0
    )
    assert pm.compare_and_promote(0.02) == 0
    assert promoted["called"] is False


def test_promote_staging_is_noop_without_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Staging version -> no-op (returns 0), never calls a stage transition."""
    monkeypatch.setattr(pm, "_latest_version", lambda stage: None)
    transitions = {"called": False}
    client = types.SimpleNamespace(
        transition_model_version_stage=lambda **kw: transitions.update(called=True)
    )
    monkeypatch.setattr(pm, "_client", lambda: client)
    assert pm.promote_staging_to_production() == 0
    assert transitions["called"] is False
