"""Keep the first-run documentation executable without a hosted model."""
import re
from pathlib import Path

import pytest


@pytest.mark.parametrize("page", ["README.md", "docs/getting-started/quickstart.md"])
def test_offline_examples(page, capsys):
    source = (Path(__file__).resolve().parents[1] / page).read_text()
    blocks = re.findall(r"```python\n(.*?)```", source, re.DOTALL)
    # The README's second Python block intentionally configures a hosted client.
    if page == "README.md":
        blocks = blocks[:1]
    namespace = {}
    for block in blocks:
        exec(compile(block, page, "exec"), namespace)
    assert namespace["answer"] == "Paris"
    assert namespace["state"].log_weight == 0.0
    assert "Paris" in capsys.readouterr().out


def test_offline_inference_example(capsys):
    page = Path(__file__).resolve().parents[1] / "docs/examples/offline-inference.md"
    block, = re.findall(r"```python\n(.*?)```", page.read_text(), re.DOTALL)
    exec(compile(block, str(page), "exec"), {})
    assert capsys.readouterr().out.splitlines() == [
        "ImportanceSampling",
        "4 0.0",
        "5 -2.0",
        "4 0.0",
    ]
