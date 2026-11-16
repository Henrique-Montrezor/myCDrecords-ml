import json
import tempfile
import unittest
from pathlib import Path

from mycdrecords_ml.serving import build_model, publish, rank_artists


class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = build_model()

    def test_unseen_profiles_beat_popularity_without_real_user_training(self):
        for split in ("validation", "test"):
            metrics = self.artifact["evaluation"][split]
            self.assertGreater(metrics["recallAt3"], metrics["popularityRecallAt3"])
            self.assertGreaterEqual(metrics["recallAt3"], 0.8)
        training = self.artifact["training"]
        self.assertLess(training["finalLoss"], training["initialLoss"])

    def test_new_profiles_change_affinities_and_negative_signals_push_away(self):
        model = self.artifact["artists"]
        goth = dict(rank_artists(model, [("The Cure", 1.0)]))
        jazz = dict(rank_artists(model, [("Miles Davis", 1.0)]))
        self.assertGreater(goth["Bauhaus"], goth["John Coltrane"])
        self.assertGreater(jazz["John Coltrane"], jazz["Bauhaus"])
        mixed = dict(rank_artists(model, [("The Cure", 1.0), ("Bauhaus", -1.0)]))
        self.assertLess(mixed["Bauhaus"], goth["Bauhaus"])
        with self.assertRaisesRegex(ValueError, "supported positive"):
            rank_artists(model, [("Unknown artist", 1.0)])

    def test_published_vectors_reload_without_user_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "artist-bpr-v1.json"
            publish(self.artifact, target)
            loaded = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(
                rank_artists(loaded["artists"], [("The Cure", 1.0)]),
                rank_artists(self.artifact["artists"], [("The Cure", 1.0)]),
            )
            self.assertNotIn("user_embeddings", target.read_text(encoding="utf-8"))
            self.assertEqual(loaded["trainedOn"], "fictional-profiles")


if __name__ == "__main__":
    unittest.main()
