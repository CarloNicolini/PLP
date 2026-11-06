"""
Restricted Python execution host for generative program synthesis.

Provides a sandboxed ``exec`` environment with a curated import hook,
bounded stdout capture, and optional async ``llm_query`` bridging.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import textwrap
from typing import Any, Callable

logger = logging.getLogger(__name__)

RESERVED_NAMES = frozenset(
    {
        "__builtins__",
        "context",
        "llm_query",
        "REPL_RESULT",
        "json",
        "_context_json",
        "_namespace_json",
    }
)


class SandboxExecutor:
    """Execute Python snippets in a restricted local environment."""

    async def run_python(
        self,
        code: str,
        context: dict[str, str],
        llm_query: Callable[[str, dict[str, str] | None], Any] | None = None,
        namespace: dict[str, Any] | None = None,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """
        Execute code in a restricted environment.

        Returns:
            ``{"ok": True, "result": ..., "namespace": {...}}`` on success,
            ``{"ok": False, "error": "...", "namespace": {...}}`` on failure.
        """
        logger.info(
            "restricted_exec code_chars=%d context_keys=%d ns_keys=%d timeout=%.0fs",
            len(code),
            len(context),
            len(namespace or {}),
            timeout_seconds,
        )
        logger.debug("restricted_exec code:\n%s", code[:500])
        code_to_run = textwrap.dedent(code).strip("\n")

        import json as json_module
        import re as re_module
        import math as math_module
        import cmath as cmath_module
        import decimal as decimal_module
        import fractions as fractions_module
        import random as random_module
        import statistics as statistics_module
        import pprint as pprint_module
        import datetime as datetime_module
        import tomllib as tomllib_module
        import collections as collections_module
        import hashlib as hashlib_module
        import secrets as secrets_module
        import difflib as difflib_module
        import itertools as itertools_module
        import functools as functools_module
        import operator as operator_module
        import string as string_module
        import unicodedata as unicodedata_module

        _allowed_modules = {
            "asyncio": asyncio,
            "queue": queue,
            "json": json_module,
            "re": re_module,
            "math": math_module,
            "cmath": cmath_module,
            "decimal": decimal_module,
            "fractions": fractions_module,
            "random": random_module,
            "statistics": statistics_module,
            "pprint": pprint_module,
            "datetime": datetime_module,
            "tomllib": tomllib_module,
            "collections": collections_module,
            "hashlib": hashlib_module,
            "secrets": secrets_module,
            "difflib": difflib_module,
            "itertools": itertools_module,
            "functools": functools_module,
            "operator": operator_module,
            "string": string_module,
            "unicodedata": unicodedata_module,
        }

        def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in _allowed_modules:
                fromlist = kwargs.get("fromlist")
                if fromlist is None and len(args) >= 3:
                    fromlist = args[2]
                if "." in name and not fromlist:
                    top_level = name.split(".", 1)[0]
                    if top_level in _allowed_modules:
                        return _allowed_modules[top_level]
                return _allowed_modules[name]
            raise ImportError(
                f"Module '{name}' is not available. "
                f"Allowed: {', '.join(sorted(_allowed_modules))}"
            )

        stdout_limit_chars = 8_000
        stdout_chars = 0
        stdout_chunks: list[str] = []
        stdout_truncated = False

        def _sandbox_print(*args: Any, **kwargs: Any) -> None:
            nonlocal stdout_chars, stdout_truncated
            if stdout_truncated:
                return
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            text = sep.join(str(a) for a in args) + end
            remaining = stdout_limit_chars - stdout_chars
            if remaining <= 0:
                stdout_truncated = True
                return
            if len(text) <= remaining:
                stdout_chunks.append(text)
                stdout_chars += len(text)
                return
            stdout_chunks.append(text[:remaining])
            stdout_chars += remaining
            stdout_truncated = True

        def _stdout_payload() -> dict[str, Any]:
            if not stdout_chunks:
                return {}
            stdout = "".join(stdout_chunks)
            if stdout_truncated:
                stdout += "\n... [sandbox stdout truncated]\n"
            return {"stdout": stdout}

        safe_builtins = {
            "__import__": _safe_import,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "any": any,
            "all": all,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "print": _sandbox_print,
            "isinstance": isinstance,
            "type": type,
            "hasattr": hasattr,
            "getattr": getattr,
            "next": next,
            "iter": iter,
            "chr": chr,
            "ord": ord,
            "repr": repr,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "ImportError": ImportError,
            "StopIteration": StopIteration,
            "Exception": Exception,
        }

        globs: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "context": context,
            "REPL_RESULT": None,
            "json": json_module,
            "re": re_module,
            "math": math_module,
        }

        if namespace:
            for k, v in namespace.items():
                if k not in RESERVED_NAMES:
                    globs[k] = v

        if llm_query is not None:
            import nest_asyncio

            nest_asyncio.apply()

            def _sync_llm_query(
                query: str, subcontext: dict[str, str] | None = None
            ) -> Any:
                loop = asyncio.get_event_loop()
                response = loop.run_until_complete(llm_query(query, subcontext))
                if (
                    isinstance(response, dict)
                    and response.get("ok") is False
                    and response.get("error")
                    in {
                        "budget-exhausted",
                        "depth-exhausted",
                        "budget-and-depth-exhausted",
                        "subquery-too-large",
                    }
                ):
                    raise RuntimeError(f"llm_query terminal: {response['error']}")
                return response

            globs["llm_query"] = _sync_llm_query

        def _exec_in_thread() -> dict[str, Any]:
            exec(code_to_run, globs)
            result = globs.get("REPL_RESULT")
            new_namespace = {}
            for k, v in globs.items():
                if k not in RESERVED_NAMES and not k.startswith("_"):
                    if isinstance(v, set):
                        new_namespace[k] = sorted(v)
                    elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
                        new_namespace[k] = v
            return {"result": result, "namespace": new_namespace}

        t0 = time.monotonic()
        try:

            def _run_with_timeout_in_daemon_thread(
                timeout: float,
            ) -> dict[str, Any] | None:
                result_queue: queue.Queue[dict[str, Any] | Exception] = queue.Queue(
                    maxsize=1
                )

                def _target() -> None:
                    try:
                        result_queue.put(_exec_in_thread())
                    except Exception as e:  # pragma: no cover
                        result_queue.put(e)

                worker = threading.Thread(target=_target, daemon=True)
                worker.start()
                worker.join(timeout)
                if worker.is_alive():
                    return None
                data = result_queue.get_nowait()
                if isinstance(data, Exception):
                    raise data
                return data

            if timeout_seconds and timeout_seconds > 0:
                exec_data = await asyncio.to_thread(
                    _run_with_timeout_in_daemon_thread,
                    timeout_seconds,
                )
                if exec_data is None:
                    elapsed = time.monotonic() - t0
                    logger.warning(
                        "restricted_exec TIMEOUT (%.1fs > %.1fs limit)",
                        elapsed,
                        timeout_seconds,
                    )
                    return {
                        "ok": False,
                        "error": f"Execution timed out after {timeout_seconds:.0f}s",
                        "namespace": namespace or {},
                        **_stdout_payload(),
                    }
            else:
                exec_data = await asyncio.to_thread(_exec_in_thread)

            elapsed = time.monotonic() - t0
            logger.info(
                "restricted_exec ok (%.3fs) has_result=%s ns_keys=%d stdout_chars=%d",
                elapsed,
                exec_data["result"] is not None,
                len(exec_data["namespace"]),
                stdout_chars,
            )
            return {"ok": True, **exec_data, **_stdout_payload()}
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.warning(
                "restricted_exec error (%.3fs): %s",
                elapsed,
                str(e)[:200],
                exc_info=True,
            )
            return {
                "ok": False,
                "error": str(e),
                "namespace": namespace or {},
                **_stdout_payload(),
            }


Executor = SandboxExecutor
