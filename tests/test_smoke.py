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


def test_real_mode_requires_both_tiers_before_constructing_either(monkeypatch):
    monkeypatch.delenv("REPO_AUDIT_MODE", raising=False)
    for prefix in ("CHEAP", "FLAGSHIP"):
        for suffix in ("MODEL", "API_KEY", "BASE_URL"):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    monkeypatch.setenv("CHEAP_MODEL", "cheap")
    monkeypatch.setenv("CHEAP_API_KEY", "secret")
    monkeypatch.setenv("CHEAP_BASE_URL", "https://example.test")
    with pytest.raises(RuntimeError, match="FLAGSHIP_"):
        config.try_cheap_tier()


def test_demo_ignores_keys_and_langfuse(monkeypatch):
    monkeypatch.setenv("REPO_AUDIT_MODE", "demo")
    for prefix in ("CHEAP", "FLAGSHIP"):
        for suffix in ("MODEL", "API_KEY", "BASE_URL"):
            monkeypatch.setenv(f"{prefix}_{suffix}", "dummy")
    for suffix in ("PUBLIC_KEY", "SECRET_KEY", "HOST"):
        monkeypatch.setenv(f"LANGFUSE_{suffix}", "dummy")
    assert config.try_cheap_tier() is None
    assert config.try_flagship_tier() is None
    assert config.langfuse_handler() is None


def test_invalid_mode_rejected(monkeypatch):
    monkeypatch.setenv("REPO_AUDIT_MODE", "automatic")
    with pytest.raises(ValueError, match="REPO_AUDIT_MODE"):
        config.try_flagship_tier()


def test_client_bounds_network_wait_and_retries(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(config, "ChatOpenAI", fake_client)
    config.ModelTier("cheap", "m", "k", "https://example.test").client()
    assert captured["timeout"] == 60
    assert captured["max_retries"] == 1


def test_verifier_enabled_defaults_on(monkeypatch):
    monkeypatch.delenv("VERIFIER_ENABLED", raising=False)
    assert config.verifier_enabled() is True


def test_verifier_enabled_reads_truthy_values(monkeypatch):
    for val in ("1", "true", "True", "YES", "on"):
        monkeypatch.setenv("VERIFIER_ENABLED", val)
        assert config.verifier_enabled() is True, f"{val!r} 应判为开启"


def test_verifier_enabled_reads_falsy_values(monkeypatch):
    for val in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("VERIFIER_ENABLED", val)
        assert config.verifier_enabled() is False, f"{val!r} 应判为关闭"
