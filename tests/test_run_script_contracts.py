from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
STAGE_1_PATTERN = re.compile(
    r'if \[\[ "\$START_STAGE" -le 1 && "\$STOP_STAGE" -ge 1 \]\]; then(?P<body>.*?)\nfi',
    re.DOTALL,
)
FREEZE_PATTERN = re.compile(r"--freeze-backbone-epochs\s+(?P<value>\S+)")


class RunScriptContractsTest(unittest.TestCase):
    def test_stage1_market_pretraining_does_not_freeze_foundation_backbone(self) -> None:
        stage1 = _stage1_body()
        match = FREEZE_PATTERN.search(stage1)

        self.assertIsNotNone(match, "Stage 1 must declare its backbone-freeze policy explicitly")
        self.assertEqual(match.group("value"), "0")


def _stage1_body() -> str:
    script = (ROOT / "run.sh").read_text(encoding="utf-8")
    match = STAGE_1_PATTERN.search(script)
    if match is None:
        raise AssertionError("Cannot locate Stage 1 block in run.sh")
    return match.group("body")


if __name__ == "__main__":
    unittest.main()
