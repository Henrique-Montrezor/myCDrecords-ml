import hashlib
import json
import math
import platform
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import torch

from .lab import MatrixFactorization, Metrics, Training, evaluate, fit_bpr


@dataclass(frozen=True)
class Album:
    id: str
    available_at: str
    source: str = "synthetic"


@dataclass(frozen=True)
class Event:
    id: str
    user_id: str
    album_id: str
    rating: int | None
    occurred_at: str
    observed_at: str
    source: str = "synthetic"


@dataclass(frozen=True)
class Window:
    targets: dict[str, str]
    exclusions: dict[str, set[str]]
    cold_users: int
    cold_items: int
    unavailable_targets: int


@dataclass(frozen=True)
class Dataset:
    users: list[str]
    items: list[str]
    training: dict[str, list[str]]
    seen: dict[str, set[str]]
    validation: Window
    test: Window
    dataset_hash: str
    training_hash: str


class WindowMetrics(TypedDict):
    evaluatedUsers: int
    coldUsers: int
    coldItems: int
    unavailableTargets: int
    model: Metrics | None
    baseline: Metrics | None


class Trial(TypedDict):
    epochs: int
    metrics: WindowMetrics


class Selection(TypedDict):
    criterion: str
    epochs: int
    trials: list[Trial]
    training: Training


class Report(TypedDict):
    schemaVersion: int
    mode: str
    protocol: str
    datasetHash: str
    trainingHash: str
    cuts: list[str]
    fields: list[str]
    userMap: list[str]
    albumMap: list[str]
    pythonVersion: str
    k: int
    selection: Selection
    test: WindowMetrics
    promotionAllowed: bool
    realWorldQuality: str


def _time(value: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise ValueError("Synthetic timestamps must be UTC ISO seconds ending in Z.")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _moment(value: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{6})?Z", value):
        raise ValueError("Expected UTC ISO seconds or microseconds ending in Z.")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def build_temporal_fixture() -> tuple[list[Album], list[Event], tuple[str, str, str]]:
    cuts = ("2026-01-03T00:00:00Z", "2026-01-05T00:00:00Z", "2026-01-07T00:00:00Z")
    albums = [Album(f"synthetic-album-{index:02d}", "2026-01-01T00:00:00Z") for index in range(1, 13)]
    albums.append(Album("synthetic-album-13", "2026-01-06T00:00:00Z"))
    events = []
    for user in range(12):
        family = list(range((user // 4) * 4 + 1, (user // 4) * 4 + 5))
        ordered = family[user % 4:] + family[:user % 4]
        for position, album in enumerate(ordered):
            day = (1, 2, 4, 6)[position]
            timestamp = f"2026-01-{day:02d}T12:00:00Z"
            events.append(Event(f"event-{user:02d}-{position}", f"synthetic-user-{user + 1:02d}",
                                f"synthetic-album-{album:02d}", 5, timestamp, timestamp))
    events.append(Event("cold", "synthetic-user-13", "synthetic-album-13", 5,
                        "2026-01-06T12:00:00Z", "2026-01-06T12:00:00Z"))
    return albums, events, cuts


def prepare(albums: list[Album], events: list[Event], cuts: tuple[str, str, str]) -> Dataset:
    for cut in cuts:
        _time(cut)
    for album in albums:
        if album.source != "synthetic" or not re.fullmatch(r"synthetic-album-\d{2}", album.id):
            raise ValueError("Only independently generated synthetic albums are accepted.")
        _time(album.available_at)
    for event in events:
        if event.source != "synthetic" or not re.fullmatch(r"synthetic-user-\d{2}", event.user_id):
            raise ValueError("Only independently generated synthetic events are accepted.")
        _time(event.occurred_at)
        _time(event.observed_at)
    if len({(event.user_id, event.album_id, event.occurred_at) for event in events}) != len(events):
        raise ValueError("Ambiguous simultaneous mutations for the same user and album.")
    return _prepare_temporal(albums, events, cuts)


def _prepare_temporal(albums: list[Album], events: list[Event], cuts: tuple[str, str, str]) -> Dataset:
    if not albums or not events or len(cuts) != 3:
        raise ValueError("Expected a non-empty catalog/events and three cuts.")
    times = [_moment(cut) for cut in cuts]
    if not times[0] < times[1] < times[2]:
        raise ValueError("Cuts must be strictly increasing.")
    catalog = {album.id: album for album in albums}
    if len(catalog) != len(albums) or len({event.id for event in events}) != len(events):
        raise ValueError("Duplicate album or event ID.")
    for album in albums:
        _moment(album.available_at)
    for event in events:
        if not event.id or not event.user_id:
            raise ValueError("Expected event and user identities.")
        if event.album_id not in catalog:
            raise ValueError("Event references an unknown album.")
        if event.rating is not None and (type(event.rating) is not int or not 1 <= event.rating <= 5):
            raise ValueError("Expected rating 1..5 or None for deletion.")
        if _moment(event.observed_at) < _moment(event.occurred_at):
            raise ValueError("Observation precedes occurrence.")
        if _moment(event.occurred_at) < _moment(catalog[event.album_id].available_at):
            raise ValueError("Interaction precedes catalog availability.")
    ordered = sorted(events, key=lambda event: (_moment(event.occurred_at), event.id))

    def state(cut: str) -> dict[tuple[str, str], Event]:
        current: dict[tuple[str, str], Event] = {}
        for event in ordered:
            if _moment(event.occurred_at) <= _moment(cut) and _moment(event.observed_at) <= _moment(cut):
                current[(event.user_id, event.album_id)] = event
        return current

    train_state = state(cuts[0])
    seen: dict[str, set[str]] = {}
    training: dict[str, list[str]] = {}
    for (user, album), event in sorted(train_state.items()):
        seen.setdefault(user, set())
        if event.rating is not None:
            seen[user].add(album)
        if event.rating is not None and event.rating >= 4:
            training.setdefault(user, []).append(album)
    users = sorted(training)
    items = sorted({album for positives in training.values() for album in positives})

    def window(start: str, end: str) -> Window:
        exclusions: dict[str, set[str]] = {}
        for (user, album), event in state(start).items():
            if event.rating is not None:
                exclusions.setdefault(user, set()).add(album)
        # One target per user: first new positive still positive at window end.
        eligible = sorted(state(end).values(), key=lambda event: (_moment(event.occurred_at), event.id))
        targets: dict[str, str] = {}
        cold_users: set[str] = set()
        cold_items: set[str] = set()
        unavailable = 0
        for event in eligible:
            if _moment(event.occurred_at) <= _moment(start) or event.rating is None or event.rating < 4:
                continue
            if event.album_id in exclusions.get(event.user_id, set()):
                continue
            if event.user_id not in users:
                cold_users.add(event.user_id)
            if event.album_id not in items:
                cold_items.add(event.album_id)
            if _moment(catalog[event.album_id].available_at) > _moment(start):
                unavailable += 1
            if event.user_id in users and event.album_id in items and _moment(catalog[event.album_id].available_at) <= _moment(start):
                targets.setdefault(event.user_id, event.album_id)
        return Window(targets, exclusions, len(cold_users), len(cold_items), unavailable)

    catalog_rows = [asdict(catalog[key]) for key in sorted(catalog)]
    event_rows = [asdict(event) for event in ordered]
    return Dataset(
        users, items, training, seen, window(cuts[0], cuts[1]), window(cuts[1], cuts[2]),
        _hash({"catalog": catalog_rows, "events": event_rows, "cuts": cuts}),
        _hash({"catalog": [row for row in catalog_rows if _moment(row["available_at"]) <= _moment(cuts[0])],
               "events": [asdict(event) for event in ordered if _moment(event.occurred_at) <= _moment(cuts[0]) and _moment(event.observed_at) <= _moment(cuts[0])],
               "cut": cuts[0]}),
    )


def _measure(model: MatrixFactorization, dataset: Dataset, window: Window) -> WindowMetrics:
    rankings: list[list[str]] = []
    baselines: list[list[str]] = []
    targets: list[str] = []
    popularity = {album: sum(album in positives for positives in dataset.training.values()) for album in dataset.items}
    with torch.inference_mode():
        for user, target in sorted(window.targets.items()):
            index = dataset.users.index(user)
            scores = model(torch.full((len(dataset.items),), index, dtype=torch.long),
                           torch.arange(len(dataset.items))).tolist()
            if not all(math.isfinite(score) for score in scores):
                raise RuntimeError("Temporal inference produced non-finite scores.")
            candidates = [album for album in dataset.items if album not in window.exclusions.get(user, set())]
            score_map = dict(zip(dataset.items, scores, strict=True))
            rankings.append(sorted(candidates, key=lambda album: (-score_map[album], album)))
            baselines.append(sorted(candidates, key=lambda album: (-popularity[album], album)))
            targets.append(target)
    return {
        "evaluatedUsers": len(targets), "coldUsers": window.cold_users,
        "coldItems": window.cold_items, "unavailableTargets": window.unavailable_targets,
        "model": evaluate(rankings, targets, 3) if targets else None,
        "baseline": evaluate(baselines, targets, 3) if targets else None,
    }


def run_temporal(albums: list[Album], events: list[Event], cuts: tuple[str, str, str]) -> Report:
    dataset = prepare(albums, events, cuts)
    if not dataset.training or not dataset.validation.targets:
        raise ValueError("Temporal experiment needs training positives and warm validation targets.")
    positives = [[dataset.items.index(album) for album in dataset.training[user]] for user in dataset.users]
    trials: list[Trial] = []
    best_model: MatrixFactorization | None = None
    best_training: Training | None = None
    best_ndcg = -1.0
    for epochs in (50, 100, 200):
        model, training = fit_bpr(positives, len(dataset.items), epochs)
        metrics = _measure(model, dataset, dataset.validation)
        trials.append({"epochs": epochs, "metrics": metrics})
        validation = metrics["model"]
        if validation is None:
            raise RuntimeError("Missing validation metrics.")
        if validation["ndcg"] > best_ndcg:
            best_model, best_training, best_ndcg = model, training, validation["ndcg"]
    if best_model is None or best_training is None:
        raise RuntimeError("No model selected.")
    # Freeze selection before measuring test; no refit or test-driven tuning.
    test = _measure(best_model, dataset, dataset.test)
    return {
        "schemaVersion": 1, "mode": "synthetic", "protocol": "synthetic-temporal-bpr-v1",
        "datasetHash": dataset.dataset_hash, "trainingHash": dataset.training_hash,
        "cuts": list(cuts),
        "fields": ["album.id", "album.available_at", "album.source", "event.id", "event.user_id",
                   "event.album_id", "event.rating", "event.occurred_at", "event.observed_at", "event.source"],
        "userMap": dataset.users, "albumMap": dataset.items, "pythonVersion": platform.python_version(), "k": 3,
        "selection": {"criterion": "validation NDCG@3; ties prefer fewer epochs",
                      "epochs": best_training["epochs"], "trials": trials, "training": best_training},
        "test": test, "promotionAllowed": False, "realWorldQuality": "not_measured",
    }


def save_report(report: Report, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / f".temporal-{uuid.uuid4().hex}.tmp"
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = directory / "temporal-report.json"
    pending.replace(target)
    return target
