#!/usr/bin/env python3
"""Focused tests for the semantic manuscript checker."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_manuscript as checker


ROOT = Path(__file__).resolve().parents[1]


class InputGraphTests(unittest.TestCase):
    def test_resolves_only_live_uncommented_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sections").mkdir()
            (root / "main.tex").write_text(
                "\\input{sections/one}\n% \\input{sections/stale}\n",
                encoding="utf-8",
            )
            (root / "sections" / "one.tex").write_text(
                "\\input{two}\n", encoding="utf-8"
            )
            (root / "sections" / "two.tex").write_text(
                "live\n", encoding="utf-8"
            )

            resolved_root, documents = checker.resolve_input_graph(
                root / "main.tex"
            )

            self.assertEqual(resolved_root, root.resolve())
            self.assertEqual(
                set(documents),
                {"main.tex", "sections/one.tex", "sections/two.tex"},
            )

    def test_rejects_an_input_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.tex").write_text("\\input{b}\n", encoding="utf-8")
            (root / "b.tex").write_text("\\input{a}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cyclic TeX input"):
                checker.resolve_input_graph(root / "a.tex")


class SemanticClaimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metrics = checker.build_metrics(ROOT / "artifact")
        cls.claims = {
            claim.claim_id: claim for claim in checker.build_claims(cls.metrics)
        }

    def test_rounding_is_explicit_and_conventional(self) -> None:
        self.assertEqual(checker._render(0.8758931204598783, 2), "0.88")
        self.assertEqual(checker._render(0.754245851599709, 2), "0.75")

    def test_baseline_registry_includes_generative_result(self) -> None:
        self.assertIn(
            "$-0.8959$", self.claims["table.baseline.generative"].fragment
        )

    def test_right_number_in_wrong_context_does_not_pass(self) -> None:
        claim = self.claims["table.main.B16"]
        wrong_row = claim.fragment.replace("$0.033309$", "$0.033500$")
        document = checker.Document(
            path=Path(claim.file),
            relative=claim.file,
            text="",
            normalized=wrong_row + " elsewhere $0.033309$",
        )

        errors, _covered = checker.audit_claims(
            {claim.file: document}, [claim]
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("table.main.B16", errors[0])

    def test_registered_current_abstract_ranges_match(self) -> None:
        _root, documents = checker.resolve_input_graph(
            ROOT / "iclr2027_conference.tex"
        )
        selected = [
            self.claims["abstract.complete_vitb.rounded_range"],
            self.claims["abstract.lora.rounded"],
        ]

        errors, _covered = checker.audit_claims(documents, selected)

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
