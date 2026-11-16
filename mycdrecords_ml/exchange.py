import json
from typing import TextIO

from .temporal import Album, Event, prepare


def _record(value: object, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Exchange has missing or unapproved fields.")
    return value


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Exchange expects string fields.")
    return value


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Exchange contains duplicate JSON keys.")
        result[key] = value
    return result


def read_synthetic_exchange(stream: TextIO) -> tuple[list[Album], list[Event], tuple[str, str, str]]:
    raw = stream.read(131073)
    if len(raw.encode("utf-8")) > 131072:
        raise ValueError("Synthetic exchange exceeds 128 KiB.")
    payload = _record(json.loads(raw, object_pairs_hook=_unique_pairs),
                      {"schemaVersion", "mode", "origin", "cuts", "albums", "events"})
    if (type(payload["schemaVersion"]) is not int or payload["schemaVersion"] != 1
            or payload["mode"] != "synthetic" or payload["origin"] != "mycdrecords-internal-fixture-v1"):
        raise ValueError("Only version 1 internal synthetic exchanges are supported; real data is blocked.")
    cuts_value = payload["cuts"]
    if not isinstance(cuts_value, list) or len(cuts_value) != 3 or not all(isinstance(cut, str) for cut in cuts_value):
        raise ValueError("Exchange requires three string cuts.")
    cuts = (_string(cuts_value[0]), _string(cuts_value[1]), _string(cuts_value[2]))
    album_rows, event_rows = payload["albums"], payload["events"]
    if not isinstance(album_rows, list) or not isinstance(event_rows, list):
        raise ValueError("Exchange albums and events must be arrays.")
    albums = []
    events = []
    for value in album_rows:
        row = _record(value, {"id", "available_at", "source"})
        albums.append(Album(_string(row["id"]), _string(row["available_at"]), _string(row["source"])))
    for value in event_rows:
        row = _record(value, {"id", "user_id", "album_id", "rating", "occurred_at", "observed_at", "source"})
        rating = row["rating"]
        if rating is not None and type(rating) is not int:
            raise ValueError("Exchange rating must be an integer or null.")
        events.append(Event(_string(row["id"]), _string(row["user_id"]), _string(row["album_id"]),
                            rating, _string(row["occurred_at"]), _string(row["observed_at"]), _string(row["source"])))
    prepare(albums, events, cuts)
    return albums, events, cuts
