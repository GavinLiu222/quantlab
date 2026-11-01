"""Execute the self-contained public API examples and quickstart code blocks."""

import doctest
import os
import re
import sys
from pathlib import Path

# These libraries share a process in the examples on macOS.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def main() -> None:
    from quantlab import api
    from quantlab.analysis import factor_report
    from quantlab.api import _factor_report, _report
    from quantlab.dataset import factor_store

    runner = doctest.DocTestRunner(
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    selected = {
        "quantlab.analysis.factor_report.FactorAnalyzer",
        "quantlab.analysis.factor_report.FactorAnalyzer.analyze_pair",
        "quantlab.analysis.factor_report.FactorAnalysis.summary",
    }
    finder = doctest.DocTestFinder()
    for module in (api, _factor_report, _report, factor_report, factor_store):
        for example in finder.find(module):
            if module is factor_report and example.name not in selected:
                continue
            runner.run(example)
    result = runner.summarize()
    if result.failed:
        raise SystemExit(1)
    print(f"Passed {result.attempted} doctest examples.")

    document = Path(__file__).resolve().parents[1] / "docs/getting-started/quickstart.md"
    blocks = re.findall(r"^```python\n(.*?)^```", document.read_text(), re.MULTILINE | re.DOTALL)
    if not blocks:
        raise SystemExit("Quickstart has no Python blocks to check.")
    namespace = {"__name__": "__quickstart_docs__"}
    try:
        for number, block in enumerate(blocks, 1):
            exec(compile(block, f"{document}:block-{number}", "exec"), namespace)  # noqa: S102 -- Run repository-owned documentation examples.
        if not namespace.get("same"):
            raise SystemExit("Quickstart did not reproduce its equity curve.")
    finally:
        if workspace := namespace.get("workspace"):
            workspace.cleanup()
    print(f"Passed all {len(blocks)} quickstart Python blocks in document order.")


if __name__ == "__main__":
    main()
