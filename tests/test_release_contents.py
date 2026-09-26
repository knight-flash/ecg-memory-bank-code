from __future__ import annotations

import ast
import io
import re
import tokenize
import unittest
import runpy
from pathlib import Path


class ReleaseContentsTests(unittest.TestCase):
    def test_archive_text_has_no_chinese(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text_suffixes = {".py", ".md", ".csv", ".json", ".yaml", ".yml", ".toml", ".txt"}
        for path in root.rglob("*"):
            if not path.is_file() or ".git" in path.relative_to(root).parts:
                continue
            if path.suffix.lower() in text_suffixes:
                self.assertNotRegex(path.read_text(encoding="utf-8"), r"[\u3400-\u9fff]", str(path))

    def test_substantive_modules_explain_their_data_contracts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sections = (
            "Purpose\n-------",
            "Method\n------",
            "Inputs\n------",
            "Outputs\n-------",
            "Paper parameters\n----------------",
            "Shapes and meaning\n------------------",
        )
        for directory in (root / "src", root / "scripts"):
            for path in directory.rglob("*.py"):
                if path.name == "__init__.py":
                    continue
                docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
                self.assertIsNotNone(docstring, str(path))
                for section in sections:
                    self.assertIn(section, docstring, str(path))
                self.assertGreaterEqual(len(docstring.splitlines()), 24, str(path))

    def test_source_comments_are_consistent(self) -> None:
        root = Path(__file__).resolve().parents[1]
        code_like = re.compile(
            r"#\s*(?:from |import |def |class |if |for |while |return |parser\.|os\.|torch\.|model\.|print\(|[A-Za-z_][\w.]*\s*=)"
        )
        for directory in (root / "src", root / "scripts"):
            for path in directory.rglob("*.py"):
                source = path.read_text(encoding="utf-8")
                for token in tokenize.generate_tokens(io.StringIO(source).readline):
                    if token.type == tokenize.COMMENT:
                        self.assertNotRegex(token.string, r"[\u3400-\u9fff]", str(path))
                        self.assertIsNone(code_like.match(token.string), str(path))
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        docstring = ast.get_docstring(node, clean=False)
                        if docstring:
                            self.assertNotRegex(docstring, r"[\u3400-\u9fff]", str(path))

    def test_only_final_manuscript_extraction_datasets(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = runpy.run_path(str(root / "scripts" / "run_extraction.py"))
        self.assertEqual(set(runner["SCRIPTS"]), {"heedb", "mimic", "code15"})
        for path in (root / "src").rglob("*.py"):
            self.assertNotIn("cpsc", path.read_text(encoding="utf-8").lower(), str(path))

    def test_formal_runner_locks_paper_parameters(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = runpy.run_path(str(root / "scripts" / "run_paper.py"))
        fixed = runner["locked_flags"]
        self.assertEqual(fixed("broad-heedb")["sampling-mode"], "anchored_nested")
        self.assertEqual(len(fixed("broad-mimic")["gallery-fractions"]), 8)
        self.assertEqual(fixed("broad-mimic")["split-by"], "subject")
        self.assertEqual(fixed("chagas-addition")["positive-test-size"], 1000)
        self.assertEqual(fixed("prototype-chagas")["max-prototypes-per-class"], 30)

    def test_direct_broad_scripts_reject_legacy_sampling(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for dataset in ("heedb", "mimic"):
            path = root / "src" / "ecg_memory_bank" / "experiments" / f"broad_case_addition_{dataset}.py"
            source = path.read_text(encoding="utf-8")
            self.assertIn("validate_broad_addition_protocol(", source)
            self.assertNotIn('"fixed_seed_fraction_sample"', source)
            self.assertNotIn('"--sampling-order-path"', source)

    def test_no_experiment_outputs_or_credentials(self) -> None:
        root = Path(__file__).resolve().parents[1]
        forbidden = {".pt", ".pth", ".npy", ".npz", ".jsonl", ".png", ".pdf", ".log", ".netrc", ".pyc"}
        for path in root.rglob("*"):
            if path.is_file():
                if ".git" in path.relative_to(root).parts:
                    continue
                self.assertNotIn(path.suffix.lower(), forbidden, str(path))
                self.assertNotIn("outputs", path.relative_to(root).parts, str(path))
                if path.suffix == ".py" and path.relative_to(root).parts[0] in {"src", "scripts"}:
                    code = path.read_text(encoding="utf-8")
                    self.assertNotIn("/data2/2shared/yanmingke", code, str(path))
                    self.assertNotIn("/home/yanmingke", code, str(path))


if __name__ == "__main__":
    unittest.main()
