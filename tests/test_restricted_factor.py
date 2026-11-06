"""Tests for :func:`plp.restricted_factor.compile_phi`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from plp.restricted_factor import compile_phi


def _state(vars_: dict[str, str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(vars=vars_ or {}, log_weight=0.0)


def test_compile_phi_reads_args() -> None:
    code = "def phi(state, *args, **kwargs):\n    return float(len(args[0]))"
    fn = compile_phi(code, "f")
    assert fn(_state(), "hello") == 5.0


def test_compile_phi_reads_state_vars() -> None:
    code = (
        "def phi(state, *args, **kwargs):\n"
        "    return 1.0 if state.vars['x'] == 'ok' else -1.0"
    )
    fn = compile_phi(code, "f")
    assert fn(_state({"x": "ok"})) == 1.0
    assert fn(_state({"x": "bad"})) == -1.0


def test_compile_phi_syntax_error() -> None:
    with pytest.raises(ValueError, match="syntax"):
        compile_phi("def phi(", "bad")


def test_compile_phi_missing_phi() -> None:
    with pytest.raises(ValueError, match="phi"):
        compile_phi("x = 1", "bad")
