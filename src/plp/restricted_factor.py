"""
RestrictedPython compilation for synthesized ``phi(state, *args, **kwargs)``
factor code produced by the Circuit Controller.

The code must define

    def phi(state, *args, **kwargs) -> float:
        ...

where ``state.vars`` is a read-only mapping from site name to output string
and ``args`` holds the parents' outputs in declaration order. Imports are
stripped; a small allowlist of modules is injected.
"""

from __future__ import annotations

import collections
import functools
import itertools
import json
import math
import re
import string
from typing import Any, Callable

from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import guarded_unpack_sequence

ALLOWED_MODULES = {
    "re": re,
    "math": math,
    "json": json,
    "collections": collections,
    "itertools": itertools,
    "functools": functools,
    "string": string,
}

_EXTRA_BUILTINS = {
    "int": int, "float": float, "str": str, "bool": bool,
    "len": len, "abs": abs, "round": round,
    "max": max, "min": min, "sum": sum,
    "range": range, "enumerate": enumerate,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "sorted": sorted, "reversed": reversed,
    "any": any, "all": all, "zip": zip, "map": map, "filter": filter,
    "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
    "True": True, "False": False, "None": None,
    "ValueError": ValueError, "TypeError": TypeError,
    "Exception": Exception,
}


def _make_inplacevar(op: str, x, y):
    ops = {
        "+=": lambda a, b: a + b, "-=": lambda a, b: a - b,
        "*=": lambda a, b: a * b, "/=": lambda a, b: a / b,
        "//=": lambda a, b: a // b, "%=": lambda a, b: a % b,
        "**=": lambda a, b: a ** b,
    }
    fn = ops.get(op)
    if fn is None:
        raise ValueError(f"Unsupported in-place operator: {op!r}")
    return fn(x, y)


def compile_phi(code: str, name: str) -> Callable[..., float]:
    """Compile a ``phi(state, *args, **kwargs) -> float`` factor in a sandbox.

    The compiled function is returned as a plain callable. Callers invoke it
    with a ``state``-like object (anything exposing ``state.vars`` and
    ``state.log_weight``), plus the parent outputs as positional args.

    Raises ``ValueError`` if compilation fails, if the code does not define
    a callable ``phi``, or if the function signature does not accept
    ``(state, *args, **kwargs)``.
    """
    clean = "\n".join(
        line for line in code.splitlines()
        if not line.strip().startswith(("import ", "from "))
    )

    try:
        byte_code = compile_restricted(clean, filename=f"<factor:{name}>", mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Factor '{name}' syntax error: {e}") from e

    ns: dict[str, Any] = dict(safe_globals)
    ns["__builtins__"] = {**safe_globals.get("__builtins__", {}), **_EXTRA_BUILTINS}
    ns["_getiter_"] = iter
    ns["_getitem_"] = lambda obj, key: obj[key]
    ns["_getattr_"] = getattr
    ns["_write_"] = lambda obj: obj
    ns["_unpack_sequence_"] = guarded_unpack_sequence
    ns["_iter_unpack_sequence_"] = guarded_unpack_sequence
    ns["_inplacevar_"] = _make_inplacevar
    ns.update(ALLOWED_MODULES)

    try:
        exec(byte_code, ns)  # noqa: S102
    except Exception as e:
        raise ValueError(f"Factor '{name}' exec failed: {e}") from e

    fn = ns.get("phi")
    if fn is None or not callable(fn):
        raise ValueError(f"Factor '{name}' did not define callable 'phi'")
    return fn
