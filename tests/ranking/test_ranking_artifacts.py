import os
import unittest
from pathlib import Path


class TestRankingArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[2]
        cls.models_ckpt = cls.project_root / "models-ckpt"
        cls.checkpoint = "adult_model"

    def test_models_ckpt_exists(self):
        self.assertTrue(self.models_ckpt.exists(), f"Missing directory: {self.models_ckpt}")

    def test_matches_file_exists(self):
        matches_file = self.models_ckpt / f"matches-{self.checkpoint}.pkl"
        self.assertTrue(matches_file.exists(), f"Missing matches file: {matches_file}")

    def test_supwise_dir_exists(self):
        supwise_dir = self.models_ckpt / "sup-wise"
        self.assertTrue(supwise_dir.exists(), f"Missing sup-wise directory: {supwise_dir}")

    def test_supwise_positive_files_exist(self):
        supwise_dir = self.models_ckpt / "sup-wise"
        files = list(supwise_dir.glob(f"{self.checkpoint}-noise-0.50-*.pkl"))
        self.assertGreater(len(files), 0, "No positive-noise sup-wise files found")

    def test_supwise_negative_files_exist(self):
        supwise_dir = self.models_ckpt / "sup-wise"
        files = list(supwise_dir.glob(f"{self.checkpoint}-noise-0.00-*.pkl"))
        self.assertGreater(len(files), 0, "No zero-noise sup-wise files found")