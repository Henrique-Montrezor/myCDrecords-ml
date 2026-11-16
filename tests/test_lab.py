import json
import tempfile
import unittest
from pathlib import Path

from mycdrecords_ml.lab import build_fixture, evaluate, train, publish, load_model


class LabTests(unittest.TestCase):
    def test_metrics_match_a_small_hand_calculated_ranking(self):
        self.assertEqual(evaluate([["a", "b"], ["b", "a"]], ["a", "a"], 1), {"recall": 0.5, "ndcg": 0.5})
        with self.assertRaises(ValueError):
            evaluate([["a"]], [], 1)

    def test_learning_beats_popularity_on_synthetic_holdouts_and_excludes_training_items(self):
        snapshot, model = train()
        self.assertLess(snapshot["training"]["finalLoss"], snapshot["training"]["initialLoss"] * 0.5)
        self.assertGreaterEqual(snapshot["evaluation"]["model"]["recall"], 0.75)
        self.assertGreater(snapshot["evaluation"]["model"]["ndcg"], snapshot["evaluation"]["baseline"]["ndcg"])
        self.assertFalse(snapshot["promotionAllowed"])
        self.assertEqual(snapshot["realWorldQuality"], "not_measured")
        for profile in snapshot["profiles"]:
            ranked = [item["albumId"] for item in profile["recommendations"]]
            self.assertTrue(set(ranked).isdisjoint(profile["trainingAlbums"]))
            self.assertEqual(len(ranked), len(set(ranked)))
        repeated, _ = train()
        self.assertEqual(snapshot["snapshotId"], repeated["snapshotId"])
        self.assertEqual(snapshot["profiles"], repeated["profiles"])
        self.assertEqual(model.user_embeddings.weight.shape[0], len(build_fixture()))

    def test_published_results_and_restricted_checkpoint_survive_reload(self):
        snapshot, model = train()
        with tempfile.TemporaryDirectory() as directory:
            path = publish(snapshot, model, Path(directory))
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored, snapshot)
            restored = load_model(Path(directory) / f'{snapshot["snapshotId"]}.pt')
            self.assertTrue(model.user_embeddings.weight.equal(restored.user_embeddings.weight))
            self.assertTrue(model.album_embeddings.weight.equal(restored.album_embeddings.weight))


if __name__ == "__main__":
    unittest.main()
