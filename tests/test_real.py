import copy
import hashlib
import io
import json
import subprocess
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mycdrecords_ml.real import read_real_exchange, train_real
from mycdrecords_ml.temporal import build_temporal_fixture


def fixture():
    albums, events, cuts = build_temporal_fixture()

    def token(value):
        return hashlib.sha256(value.encode()).hexdigest()

    def micro(value):
        return value.replace("Z", ".000000Z")

    return {
        "schemaVersion": 2, "mode": "real",
        "origin": "mycdrecords-consented-ratings-v2", "noticeVersion": "ml-pilot-v2",
        "generation": "9007199254740993", "captured_at": "2026-01-08T00:00:00.000000Z",
        "cuts": [micro(cut) for cut in cuts],
        "albums": [{"id": token(album.id), "available_at": micro(album.available_at)} for album in albums],
        "events": [{
            "sequence": str(index + 1), "user_id": token(event.user_id), "album_id": token(event.album_id),
            "rating": event.rating, "occurred_at": micro(event.occurred_at),
            "observed_at": micro(event.observed_at), "expires_at": "2026-01-30T00:00:00.000000Z",
        } for index, event in enumerate(events)],
    }


NOW = datetime(2026, 1, 8, tzinfo=timezone.utc)


class RealTrainingTests(unittest.TestCase):
    def test_trains_evaluates_without_exporting_maps_or_weights(self):
        payload = fixture()
        report = train_real(read_real_exchange(io.StringIO(json.dumps(payload))), now=NOW)
        self.assertEqual(report["status"], "trained")
        self.assertEqual(report["mode"], "real")
        self.assertEqual(report["test"]["evaluatedUsers"], 12)
        self.assertLess(report["training"]["finalLoss"], report["training"]["initialLoss"])
        self.assertFalse(report["promotionAllowed"])
        self.assertFalse(report["persistedWeights"])
        encoded = json.dumps(report)
        self.assertNotIn(payload["events"][0]["user_id"], encoded)
        self.assertNotIn(payload["albums"][0]["id"], encoded)
        self.assertNotIn("userMap", encoded)

    def test_rejects_unapproved_fields_expired_events_and_collection_only_notice(self):
        original = fixture()
        cases = []
        for field, value in [("noticeVersion", "ml-pilot-v1"), ("mode", "synthetic"), ("email", "fixture")]:
            payload = copy.deepcopy(original)
            payload[field] = value
            cases.append(payload)
        for field, value in [("rating", True), ("expires_at", "2026-01-07T00:00:00.000000Z"),
                             ("user_id", "1"), ("sequence", "01"), ("review_text", "fixture")]:
            payload = copy.deepcopy(original)
            payload["events"][0][field] = value
            cases.append(payload)
        for payload in cases:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    train_real(payload, now=NOW)

    def test_sequence_breaks_microsecond_ties_and_future_edits_do_not_tune_selection(self):
        original = fixture()
        edited = copy.deepcopy(original)
        event = copy.deepcopy(edited["events"][0])
        event["sequence"] = "9007199254740993"
        event["rating"] = 1
        edited["events"].append(event)
        # A later mutation at the exact same microsecond supersedes the first positive.
        from mycdrecords_ml.real import prepare_real
        dataset = prepare_real(edited, now=NOW)
        self.assertNotIn(event["album_id"], dataset.training.get(event["user_id"], []))
        changed = copy.deepcopy(original)
        for row in changed["events"]:
            if row["occurred_at"] > changed["cuts"][1]:
                row["rating"] = 1
        first = train_real(original, now=NOW)
        with self.assertRaisesRegex(ValueError, "WARM_TEST_TARGETS_REQUIRED"):
            train_real(changed, now=NOW)
        changed = copy.deepcopy(original)
        changed["events"][-2]["rating"] = 1
        second = train_real(changed, now=NOW)
        self.assertEqual(first["training"], second["training"])

    def test_no_success_shaped_model_for_insufficient_data_or_oversized_input(self):
        payload = fixture()
        payload["events"] = []
        with self.assertRaisesRegex(ValueError, "NO_ELIGIBLE_EVENTS"):
            train_real(payload, now=NOW)
        with self.assertRaises(ValueError):
            read_real_exchange(io.StringIO(" " * (2 * 1024 * 1024 + 1)))
        with self.assertRaises(ValueError):
            read_real_exchange(io.StringIO('{"mode":"real","mode":"synthetic"}'))

    def test_node_to_pytorch_pipe_with_disposable_fixture_and_aggregate_output(self):
        clock = datetime.now(timezone.utc)

        def stamp(days):
            return (clock - timedelta(days=days)).isoformat(timespec="microseconds").replace("+00:00", "Z")

        rows = []
        for user in range(12):
            for position in range(4):
                album = (user // 4) * 4 + (user % 4 + position) % 4 + 1
                occurred = stamp((6, 5, 2, 1)[position])
                rows.append({
                    "sequence": str(len(rows) + 1), "user_id": user + 1,
                    "album_id": f"00000000-0000-4000-8000-{album:012d}",
                    "rating": 5, "occurred_at": occurred, "observed_at": occurred,
                    "expires_at": stamp(-20), "available_at": stamp(10),
                })
        script = """
const {buildRealExchange} = require('./src/modules/recommendations/recommendations.real');
const {runRealPython} = require('./src/modules/recommendations/recommendations.real.cli');
const {readFileSync} = require('node:fs');
runRealPython(buildRealExchange(JSON.parse(readFileSync(0, 'utf8'))))
  .then(report => console.log(JSON.stringify(report)))
  .catch(error => { console.error(error.message); process.exitCode = 1; });
"""
        result = subprocess.run(
            ["node", "-r", "ts-node/register", "-e", script],
            input=json.dumps({"generation": "1", "capturedAt": stamp(0), "rows": rows}),
            cwd=Path(__file__).resolve().parents[2] / "myCDrecords-API",
            capture_output=True, text=True, timeout=120, check=True,
        )
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "trained")
        self.assertEqual(report["test"]["evaluatedUsers"], 12)
        self.assertLess(report["training"]["finalLoss"], report["training"]["initialLoss"])
        self.assertFalse(report["promotionAllowed"])
        self.assertFalse(report["persistedWeights"])
        self.assertNotIn("userMap", report)


if __name__ == "__main__":
    unittest.main()
