import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from recognizer_contract import (  # noqa: E402
    generation_kwargs,
    normalize_ocr_target,
    prompts_for_type,
    type_balance_multipliers,
)


class RecognizerContractTest(unittest.TestCase):
    def test_prompts_are_type_aware(self):
        handwritten = prompts_for_type("handwritten")
        formula = prompts_for_type("formula")
        table = prompts_for_type("table")
        self.assertNotEqual(handwritten, formula)
        self.assertIn("formula", formula[1].lower())
        self.assertIn("table", table[1].lower())

    def test_natural_text_collapses_newlines(self):
        value = normalize_ocr_target("  Один\n  два\tтри ", "handwritten")
        self.assertEqual(value, "Один два три")

    def test_formula_preserves_lines(self):
        value = normalize_ocr_target("x = 1\n y = 2 ", "formula")
        self.assertEqual(value, "x = 1\ny = 2")

    def test_table_preserves_rows_and_normalizes_pipes(self):
        value = normalize_ocr_target(
            " Name | Value \n A  |  10 \n\n B | 20 ",
            "table",
        )
        self.assertEqual(value, "Name|Value\nA|10\nB|20")

    def test_type_weights_are_capped(self):
        multipliers, counts = type_balance_multipliers(
            ["handwritten"] * 100 + ["formula"] * 25 + ["table"],
            cap=4.0,
        )
        self.assertEqual(counts["handwritten"], 100)
        self.assertEqual(multipliers["handwritten"], 1.0)
        self.assertEqual(multipliers["formula"], 2.0)
        self.assertEqual(multipliers["table"], 4.0)

    def test_generation_is_longer_for_structured_types(self):
        natural = generation_kwargs("handwritten")
        formula = generation_kwargs("formula")
        table = generation_kwargs("table")
        self.assertLess(natural["max_new_tokens"], formula["max_new_tokens"])
        self.assertLess(formula["max_new_tokens"], table["max_new_tokens"])
        self.assertIn("repetition_penalty", natural)
        self.assertNotIn("repetition_penalty", formula)

    def test_generation_budget_can_be_overridden(self):
        with patch.dict(os.environ, {"MAX_TOKENS_FORMULA": "192"}):
            self.assertEqual(
                generation_kwargs("formula")["max_new_tokens"],
                192,
            )


if __name__ == "__main__":
    unittest.main()
