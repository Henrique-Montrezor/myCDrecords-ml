import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mycdrecords_ml.temporal import (
    Album, Event, build_temporal_fixture, prepare, run_temporal, save_report,
)


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.albums, self.events, self.cuts = build_temporal_fixture()

    def test_as_of_replay_handles_edits_deletes_and_late_arrivals(self):
        events = [
            Event("a", "synthetic-user-01", "synthetic-album-01", 5, "2026-01-01T10:00:00Z", "2026-01-01T10:00:00Z"),
            Event("b", "synthetic-user-01", "synthetic-album-01", 1, "2026-01-02T10:00:00Z", "2026-01-02T10:00:00Z"),
            Event("c", "synthetic-user-01", "synthetic-album-02", 5, "2026-01-01T10:00:00Z", "2026-01-04T10:00:00Z"),
            Event("d", "synthetic-user-02", "synthetic-album-02", 5, "2026-01-01T10:00:00Z", "2026-01-01T10:00:00Z"),
            Event("e", "synthetic-user-02", "synthetic-album-02", None, "2026-01-02T10:00:00Z", "2026-01-02T10:00:00Z"),
        ]
        dataset = prepare(self.albums, events, self.cuts)
        self.assertEqual(dataset.training, {})
        self.assertEqual(dataset.seen["synthetic-user-01"], {"synthetic-album-01"})
        self.assertEqual(dataset.seen["synthetic-user-02"], set())

    def test_future_changes_do_not_change_training_maps_masks_or_fingerprint(self):
        original = prepare(self.albums, self.events, self.cuts)
        changed = [replace(event, rating=1) if event.occurred_at > self.cuts[0] else event for event in self.events]
        altered = prepare(self.albums, changed, self.cuts)
        self.assertEqual(original.training, altered.training)
        self.assertEqual(original.users, altered.users)
        self.assertEqual(original.items, altered.items)
        self.assertEqual(original.seen, altered.seen)
        self.assertEqual(original.training_hash, altered.training_hash)
        self.assertNotEqual(original.dataset_hash, altered.dataset_hash)
        self.assertEqual(altered.validation.targets, {})
        self.assertNotIn("synthetic-album-13", original.items)
        self.assertNotIn("synthetic-user-13", original.users)

    def test_temporal_targets_and_cold_start_are_separate(self):
        dataset = prepare(self.albums, self.events, self.cuts)
        self.assertEqual(len(dataset.training), 12)
        self.assertTrue(all(len(items) == 2 for items in dataset.training.values()))
        self.assertEqual(len(dataset.validation.targets), 12)
        self.assertEqual(len(dataset.test.targets), 12)
        self.assertEqual(dataset.test.cold_users, 1)
        self.assertEqual(dataset.test.cold_items, 1)
        self.assertEqual(dataset.test.unavailable_targets, 1)
        for user, target in dataset.test.targets.items():
            self.assertNotIn(target, dataset.test.exclusions[user])
            self.assertIn(dataset.validation.targets[user], dataset.test.exclusions[user])

    def test_catalog_only_items_are_cold_not_random_embedding_candidates(self):
        albums = [replace(album, available_at="2026-01-01T00:00:00Z") for album in self.albums]
        dataset = prepare(albums, self.events, self.cuts)
        self.assertNotIn("synthetic-album-13", dataset.items)
        self.assertEqual(dataset.test.cold_items, 1)
        self.assertEqual(dataset.test.unavailable_targets, 0)

    def test_cut_boundaries_are_inclusive_and_shuffled_inputs_are_reproducible(self):
        events = [
            Event("a", "synthetic-user-01", "synthetic-album-01", 5, self.cuts[0], self.cuts[0]),
            Event("b", "synthetic-user-02", "synthetic-album-02", 5, self.cuts[0], self.cuts[0]),
            Event("c", "synthetic-user-03", "synthetic-album-03", 5, self.cuts[0], self.cuts[0]),
            Event("d", "synthetic-user-01", "synthetic-album-02", 5, self.cuts[1], self.cuts[1]),
            Event("e", "synthetic-user-01", "synthetic-album-03", 5, self.cuts[2], self.cuts[2]),
        ]
        dataset = prepare(self.albums, events, self.cuts)
        self.assertEqual(dataset.training["synthetic-user-01"], ["synthetic-album-01"])
        self.assertEqual(dataset.validation.targets, {"synthetic-user-01": "synthetic-album-02"})
        self.assertEqual(dataset.test.targets, {"synthetic-user-01": "synthetic-album-03"})
        self.assertEqual(dataset, prepare(list(reversed(self.albums)), list(reversed(events)), self.cuts))

    def test_invalid_input_is_rejected_instead_of_producing_success(self):
        invalid_cases = [
            (self.albums, [], self.cuts),
            (self.albums, self.events + [self.events[0]], self.cuts),
            (self.albums, self.events + [replace(self.events[0], id="ambiguous", rating=1)], self.cuts),
            (self.albums, [replace(self.events[0], source="spotify")], self.cuts),
            (self.albums, [replace(self.events[0], rating=6)], self.cuts),
            (self.albums, [replace(self.events[0], rating=True)], self.cuts),
            (self.albums, [replace(self.events[0], album_id="synthetic-album-99")], self.cuts),
            (self.albums, [replace(self.events[0], occurred_at="2026-01-01")], self.cuts),
            (self.albums, [replace(self.events[0], observed_at="2025-01-01T00:00:00Z")], self.cuts),
            ([replace(self.albums[0], source="musicbrainz")], self.events, self.cuts),
            ([self.albums[0], self.albums[0]], self.events, self.cuts),
            (self.albums, self.events, (self.cuts[0], self.cuts[0], self.cuts[2])),
            ([Album("synthetic-album-01", "2027-01-01T00:00:00Z")], [self.events[0]], self.cuts),
        ]
        for albums, events, cuts in invalid_cases:
            with self.subTest(events=events[:1], cuts=cuts):
                with self.assertRaises(ValueError):
                    prepare(albums, events, cuts)

    def test_validation_selection_is_independent_of_test_labels_and_report_persists(self):
        report = run_temporal(self.albums, self.events, self.cuts)
        changed = [replace(event, rating=1) if event.occurred_at > self.cuts[1] else event for event in self.events]
        repeated = run_temporal(self.albums, changed, self.cuts)
        self.assertEqual(report["selection"], repeated["selection"])
        self.assertEqual(report["trainingHash"], repeated["trainingHash"])
        self.assertFalse(report["promotionAllowed"])
        self.assertEqual(report["realWorldQuality"], "not_measured")
        self.assertEqual(report["test"]["evaluatedUsers"], 12)
        self.assertEqual(repeated["test"]["evaluatedUsers"], 0)
        self.assertIsNone(repeated["test"]["model"])
        self.assertEqual(len(report["selection"]["trials"]), 3)
        self.assertEqual(report, run_temporal(list(reversed(self.albums)), list(reversed(self.events)), self.cuts))
        with tempfile.TemporaryDirectory() as directory:
            path = save_report(report, Path(directory))
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored, report)
            self.assertFalse((Path(directory) / "snapshot.json").exists())

    def test_cli_writes_temporal_report_only_and_requires_synthetic_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-m", "mycdrecords_ml", "temporal", "--synthetic", "--output", directory],
                capture_output=True, text=True, check=True,
            )
            output = json.loads(result.stdout)
            self.assertEqual(output["protocol"], "synthetic-temporal-bpr-v1")
            self.assertTrue((Path(directory) / "temporal-report.json").is_file())
            self.assertFalse((Path(directory) / "snapshot.json").exists())
            denied = subprocess.run(
                [sys.executable, "-m", "mycdrecords_ml", "temporal", "--output", directory],
                capture_output=True, text=True,
            )
            self.assertEqual(denied.returncode, 2)
            self.assertIn("--synthetic", denied.stderr)


if __name__ == "__main__":
    unittest.main()
