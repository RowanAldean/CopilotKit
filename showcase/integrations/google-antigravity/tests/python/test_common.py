"""Shared agent wiring in src/agents/_common.py."""

from __future__ import annotations

import pytest

from agents import _common


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        # Compose sets the OpenAI SDK convention (`.../v1`); the harness
        # appends `/v1/chat/completions` itself, so the shim needs the root.
        ("http://aimock:4010/v1", "http://aimock:4010"),
        ("http://aimock:4010/v1/", "http://aimock:4010"),
        ("https://api.openai.com/v1", "https://api.openai.com"),
        ("http://proxy.local", "http://proxy.local"),
        ("http://proxy.local/", "http://proxy.local"),
    ],
)
def test_upstream_is_the_root_the_harness_appends_to(monkeypatch, base, expected):
    monkeypatch.setenv("OPENAI_BASE_URL", base)
    assert _common._upstream() == expected


def test_upstream_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert _common._upstream() == "https://api.openai.com"


def test_only_finish_is_enabled():
    from google.antigravity.types import BuiltinTools

    capabilities = _common.chat_only_capabilities()
    # search_web loops without Google credentials and ask_question would park
    # on an interrupt no demo answers; the workspace's file and shell tools
    # stay off on a public deployment.
    assert list(capabilities.enabled_tools) == [BuiltinTools.FINISH]
    assert capabilities.enable_subagents is False
