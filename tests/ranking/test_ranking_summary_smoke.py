"""
This test runs the ranking summary script and checks that it exits successfully.
"""

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


class TestRankingSummarySmoke(unittest.TestCase):
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

    def test_ranking_summary_runs(self):
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

        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            env=env,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)

        self.assertEqual(result.returncode, 0, "ranking_summary.py did not run successfully")