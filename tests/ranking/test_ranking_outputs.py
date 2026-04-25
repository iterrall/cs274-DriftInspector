"""
This checks that the expected output files are actually created and non-empty.
"""

import csv
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


class TestRankingOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[2]
        cls.python_exe = cls.project_root / ".venv" / "Scripts" / "python.exe"

        if not cls.python_exe.exists():
            cls.python_exe = Path(sys.executable)

        cls.output_dir = cls.project_root / "test-report-metrics"

    def setUp(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)

        cmd = [
            str(self.python_exe),
            str(self.project_root / "src" / "ranking_summary.py"),
            "--models-ckpt", str(self.project_root / "models-ckpt"),
            "--baseline-checkpoint", "adult_model",
            "--adaptive-checkpoint", "adult_model",
            "--win-size", "5",
            "--output-dir", str(self.output_dir),
        ]

        self.result = subprocess.run(
            cmd,
            cwd=self.project_root,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_script_exit_code(self):
        if self.result.returncode != 0:
            print("STDOUT:\n", self.result.stdout)
            print("STDERR:\n", self.result.stderr)
        self.assertEqual(self.result.returncode, 0)

    def test_overall_csv_exists(self):
        path = self.output_dir / "ranking_summary_overall.csv"
        self.assertTrue(path.exists(), f"Missing file: {path}")

    def test_by_support_csv_exists(self):
        path = self.output_dir / "ranking_summary_by_support.csv"
        self.assertTrue(path.exists(), f"Missing file: {path}")

    def test_overall_csv_has_rows(self):
        path = self.output_dir / "ranking_summary_overall.csv"
        self.assertTrue(path.exists(), f"Missing file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        self.assertGreater(len(rows), 1, "ranking_summary_overall.csv has no data rows")

    def test_overall_csv_has_expected_columns(self):
        path = self.output_dir / "ranking_summary_overall.csv"
        self.assertTrue(path.exists(), f"Missing file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

        expected = {"implementation", "method", "nDCG", "nDCG_10", "nDCG_100", "Pearson", "Spearman"}
        self.assertTrue(expected.issubset(set(fieldnames)),
                        f"Missing expected columns. Found: {fieldnames}")