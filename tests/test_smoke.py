"""Smoke tests: package imports and tier config validation (no network)."""

import pytest

import repo_audit
from repo_audit import config


def test_version():
    assert repo_audit.__version__


def test_missing_env_raises(monkeypatch):
    for k in ("CHEAP_MODEL", "CHEAP_API_KEY", "CHEAP_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="CHEAP_"):
        config.cheap_tier()


def test_tier_reads_env(monkeypatch):
    monkeypatch.setenv("CHEAP_MODEL", "test-model")
    monkeypatch.setenv("CHEAP_API_KEY", "sk-test")
    monkeypatch.setenv("CHEAP_BASE_URL", "https://example.com/v1")
    tier = config.cheap_tier()
    assert tier.model == "test-model"
