import json
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import TextIO, TypedDict

from .exchange import _record, _string, _unique_pairs
from .lab import MatrixFactorization, Training, fit_bpr
from .temporal import Album, Dataset, Event, WindowMetrics, _measure, _moment, _prepare_temporal

MAX_BYTES = 2 * 1024 * 1024


class RealReport(TypedDict):
    status: str
    mode: str
    protocol: str
    users: int
    albums: int
    events: int
    training: Training
    validation: WindowMetrics
    test: WindowMetrics
    promotionAllowed: bool
    persistedWeights: bool


def read_real_exchange(stream: TextIO) -> object:
    raw = stream.read(MAX_BYTES + 1)
    if len(raw.encode("utf-8")) > MAX_BYTES:
        raise ValueError("EXCHANGE_TOO_LARGE")
    return json.loads(raw, object_pairs_hook=_unique_pairs)


def _token(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("INVALID_PSEUDONYM")
    return value


def prepare_real(value: object, *, now: datetime | None = None) -> Dataset:
    payload = _record(value, {"schemaVersion", "mode", "origin", "noticeVersion", "generation",
                              "captured_at", "cuts", "albums", "events"})
    if (type(payload["schemaVersion"]) is not int or payload["schemaVersion"] != 2
            or payload["mode"] != "real" or payload["origin"] != "mycdrecords-consented-ratings-v2"
            or payload["noticeVersion"] != "ml-pilot-v2"):
        raise ValueError("INVALID_REAL_TRAINING_CONTRACT")
    generation = _string(payload["generation"])
    if not re.fullmatch(r"[1-9]\d{0,19}", generation) or int(generation) > 2**64 - 1:
        raise ValueError("INVALID_GENERATION")
    captured = _moment(_string(payload["captured_at"]))
    clock = now or datetime.now(timezone.utc)
    if not captured - timedelta(minutes=5) <= clock <= captured + timedelta(minutes=5):
        raise ValueError("SNAPSHOT_TOO_OLD_OR_CLOCK_SKEW")
    cut_values = payload["cuts"]
    if not isinstance(cut_values, list) or len(cut_values) != 3:
        raise ValueError("THREE_TEMPORAL_CUTS_REQUIRED")
    cuts = (_string(cut_values[0]), _string(cut_values[1]), _string(cut_values[2]))
    if not _moment(cuts[0]) < _moment(cuts[1]) < _moment(cuts[2]) <= captured:
        raise ValueError("INVALID_TEMPORAL_CUTS")
    album_rows, event_rows = payload["albums"], payload["events"]
    if not isinstance(album_rows, list) or not isinstance(event_rows, list):
        raise ValueError("INVALID_REAL_ARRAYS")
    if not event_rows:
        raise ValueError("NO_ELIGIBLE_EVENTS")
    if len(event_rows) > 5000 or len(album_rows) > 200:
        raise ValueError("PILOT_CAPACITY_EXCEEDED")
    albums = []
    events = []
    for item in album_rows:
        row = _record(item, {"id", "available_at"})
        albums.append(Album(_token(row["id"]), _string(row["available_at"]), "independent-core-cc0"))
    for item in event_rows:
        row = _record(item, {"sequence", "user_id", "album_id", "rating", "occurred_at", "observed_at", "expires_at"})
        sequence = _string(row["sequence"])
        if not re.fullmatch(r"[1-9]\d{0,19}", sequence) or int(sequence) > 2**64 - 1:
            raise ValueError("INVALID_SEQUENCE")
        occurred, observed, expires = (_moment(_string(row[key]))
                                      for key in ("occurred_at", "observed_at", "expires_at"))
        if not occurred <= observed <= captured or not clock < expires <= occurred + timedelta(days=30):
            raise ValueError("EXPIRED_OR_INVALID_EVENT")
        rating = row["rating"]
        if rating is not None and (type(rating) is not int or not 1 <= rating <= 5):
            raise ValueError("INVALID_RATING")
        events.append(Event(sequence.zfill(20), _token(row["user_id"]), _token(row["album_id"]), rating,
                            _string(row["occurred_at"]), _string(row["observed_at"]), "independent-local-rating"))
    if len({event.user_id for event in events}) > 100:
        raise ValueError("PILOT_CAPACITY_EXCEEDED")
    return _prepare_temporal(albums, events, cuts)


def train_real(value: object, *, now: datetime | None = None) -> RealReport:
    dataset = prepare_real(value, now=now)
    if len(dataset.users) < 2 or len(dataset.items) < 2:
        raise ValueError("TRAINING_OVERLAP_REQUIRED")
    if not dataset.validation.targets:
        raise ValueError("WARM_VALIDATION_TARGETS_REQUIRED")
    if not dataset.test.targets:
        raise ValueError("WARM_TEST_TARGETS_REQUIRED")
    positives = [[dataset.items.index(album) for album in dataset.training[user]] for user in dataset.users]
    triple_count = sum(len(items) * (len(dataset.items) - len(items)) for items in positives)
    if any(len(items) == len(dataset.items) for items in positives):
        raise ValueError("TRAINING_CONTRASTS_REQUIRED")
    if triple_count > 100000:
        raise ValueError("PILOT_CAPACITY_EXCEEDED")
    best_model: MatrixFactorization | None = None
    best_training: Training | None = None
    best_validation: WindowMetrics | None = None
    best_score = -1.0
    for epochs in (50, 100, 200):
        model, training = fit_bpr(positives, len(dataset.items), epochs)
        validation = _measure(model, dataset, dataset.validation)
        metrics = validation["model"]
        if metrics is None:
            raise RuntimeError("VALIDATION_FAILED")
        if metrics["ndcg"] > best_score:
            best_model, best_training, best_validation, best_score = model, training, validation, metrics["ndcg"]
    if best_model is None or best_training is None or best_validation is None:
        raise RuntimeError("MODEL_SELECTION_FAILED")
    test = _measure(best_model, dataset, dataset.test)
    # Revalidate expiry after fitting; the Node owner also rechecks DB authorization before accepting results.
    prepare_real(value, now=now)
    if not isinstance(value, dict) or not isinstance(value["events"], list):
        raise RuntimeError("INVALID_EXCHANGE")
    return {
        "status": "trained", "mode": "real", "protocol": "real-global-time-60-20-20-bpr-v1",
        "users": len(dataset.users), "albums": len(dataset.items), "events": len(value["events"]),
        "training": best_training, "validation": best_validation, "test": test,
        "promotionAllowed": False, "persistedWeights": False,
    }


def main() -> None:
    try:
        report = train_real(read_real_exchange(sys.stdin))
    except (ValueError, RuntimeError) as error:
        code = str(error)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,80}", code):
            code = "INVALID_REAL_TRAINING_INPUT"
        print(json.dumps({"status": "blocked", "code": code, "mode": "real", "promotionAllowed": False}))
        sys.exit(1)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
