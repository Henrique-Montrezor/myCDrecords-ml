import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mycdrecords_ml.exchange import read_synthetic_exchange
from mycdrecords_ml.temporal import prepare, run_temporal


class ExchangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api = Path(__file__).resolve().parents[2] / "myCDrecords-API"
        result = subprocess.run(
            ["node", "-r", str(api / "node_modules" / "ts-node" / "register"),
             str(api / "src" / "modules" / "recommendations" / "recommendations.exchange.cli.ts"), "--synthetic"],
            cwd=api, capture_output=True, text=True, check=True, timeout=30,
        )
        cls.raw = result.stdout
        cls.payload = json.loads(cls.raw)

    def test_api_exchange_trains_and_evaluates_in_python_without_real_data(self):
        albums, events, cuts = read_synthetic_exchange(io.StringIO(self.raw))
        report = run_temporal(albums, events, cuts)
        self.assertEqual(report["test"]["evaluatedUsers"], 12)
        self.assertEqual(report["test"]["coldUsers"], 1)
        self.assertFalse(report["promotionAllowed"])
        self.assertEqual(report["realWorldQuality"], "not_measured")

    def test_rejects_non_synthetic_sources_personal_fields_and_invalid_shapes(self):
        for update in [
            {"mode": "real"}, {"origin": "database"}, {"email": "fixture@example.invalid"},
            {"events": [{**self.payload["events"][0], "text": "not permitted"}]},
            {"albums": [{**self.payload["albums"][0], "source": "spotify"}]},
            {"events": [{**self.payload["events"][0], "rating": True}]},
            {"events": [{**self.payload["events"][0], "user_id": 1}]},
            {"events": self.payload["events"] * 2}, {"cuts": [True, False, None]},
        ]:
            with self.subTest(update=list(update)):
                with self.assertRaises(ValueError):
                    read_synthetic_exchange(io.StringIO(json.dumps({**self.payload, **update})))
        for raw in ["", "not-json", '{"mode":"synthetic","mode":"real"}', " " * 131073]:
            with self.subTest(raw_length=len(raw)):
                with self.assertRaises(ValueError):
                    read_synthetic_exchange(io.StringIO(raw))

    def test_mutations_and_reordered_export_preserve_as_of_contract(self):
        payload = json.loads(self.raw)
        original = payload["events"][0]
        payload["events"].append({
            **original, "id": "edit", "rating": 1,
            "occurred_at": "2026-01-02T18:00:00Z", "observed_at": "2026-01-02T18:00:00Z",
        })
        dataset = prepare(*read_synthetic_exchange(io.StringIO(json.dumps(payload))))
        self.assertNotIn(original["album_id"], dataset.training[original["user_id"]])
        payload["events"].reverse()
        self.assertEqual(dataset, prepare(*read_synthetic_exchange(io.StringIO(json.dumps(payload)))))

    def test_cli_stdin_validates_before_publication_and_rejects_real_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-m", "mycdrecords_ml", "temporal", "--synthetic", "--stdin", "--output", directory]
            result = subprocess.run(command, input=self.raw, text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout)["test"]["evaluatedUsers"], 12)
            report_path = Path(directory) / "temporal-report.json"
            previous = report_path.read_bytes()
            invalid = json.dumps({**self.payload, "mode": "real"})
            denied = subprocess.run(command, input=invalid, text=True, capture_output=True)
            self.assertNotEqual(denied.returncode, 0)
            self.assertEqual(denied.stdout, "")
            self.assertEqual(report_path.read_bytes(), previous)
            self.assertFalse((Path(directory) / "snapshot.json").exists())
