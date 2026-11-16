import argparse
import hashlib
import itertools
import json
import math
import uuid
from pathlib import Path
from typing import TypedDict

import torch

from .lab import fit_bpr


# Independently authored toy affinities, not listening histories, tags or fan statistics.
ARTIST_GROUPS = (
    ("Radiohead", "Pink Floyd", "David Bowie", "The Beatles"),
    ("Metallica", "Iron Maiden", "Black Sabbath", "Megadeth"),
    ("The Cure", "Bauhaus", "Joy Division", "Siouxsie and the Banshees"),
    ("Miles Davis", "John Coltrane", "Dave Brubeck", "Bill Evans"),
    ("Milton Nascimento", "Caetano Veloso", "Gilberto Gil", "Chico Buarque"),
    ("Daft Punk", "Kraftwerk", "The Chemical Brothers", "Massive Attack"),
)
MINIMUM_AFFINITY = 0.35


class Artist(TypedDict):
    name: str
    vector: list[str]


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("Degenerate embedding.")
    return [value / norm for value in vector]


def rank_artists(artists: list[Artist], signals: list[tuple[str, float]]) -> list[tuple[str, float]]:
    vectors = {artist["name"]: _unit([float(value) for value in artist["vector"]]) for artist in artists}
    supported = [(name, weight) for name, weight in signals if name in vectors]
    if not any(weight > 0 for _, weight in supported):
        raise ValueError("Expected a supported positive signal.")
    profile = _unit([sum(vectors[name][axis] * weight for name, weight in supported) for axis in range(4)])
    scores = [(name, sum(a * b for a, b in zip(vector, profile, strict=True))) for name, vector in vectors.items()]
    # Keep index order for ties, matching the runtime's artifact order.
    return sorted(scores, key=lambda item: -item[1])


def _evaluate(artists: list[Artist], positives: list[list[int]], width: int) -> dict[str, float | int]:
    names = [artist["name"] for artist in artists]
    popularity = [sum(index in profile for profile in positives) for index in range(len(names))]
    recalls: list[float] = []
    baselines: list[float] = []
    for start in range(0, len(names), 4):
        family = list(range(start, start + 4))
        for observed in itertools.combinations(family, width):
            targets = {names[index] for index in family if index not in observed}
            seen = {names[index] for index in observed}
            ranked = [name for name, score in rank_artists(artists, [(name, 1.0) for name in names if name in seen])
                      if name not in seen and score >= MINIMUM_AFFINITY]
            baseline = [names[index] for index in sorted(range(len(names)), key=lambda index: (-popularity[index], index)) if names[index] not in seen]
            recalls.append(len(targets.intersection(ranked[:3])) / len(targets))
            baselines.append(len(targets.intersection(baseline[:3])) / len(targets))
    return {"profiles": len(recalls), "recallAt3": sum(recalls) / len(recalls),
            "popularityRecallAt3": sum(baselines) / len(baselines)}


def build_model() -> dict:
    names = [name for group in ARTIST_GROUPS for name in group]
    positives = [[index for index in range(start, start + 4) if index != held_out]
                 for start in range(0, len(names), 4) for held_out in range(start, start + 4)]
    model, training = fit_bpr(positives, len(names), 200)
    with torch.inference_mode():
        artists: list[Artist] = [{"name": name, "vector": [f"{value:.8f}" for value in _unit(vector)]}
                                for name, vector in zip(names, model.album_embeddings.weight.tolist(), strict=True)]
    validation = _evaluate(artists, positives, 2)
    test = _evaluate(artists, positives, 1)
    for metrics in (validation, test):
        if metrics["recallAt3"] < 0.8 or metrics["recallAt3"] <= metrics["popularityRecallAt3"]:
            raise RuntimeError("Fictional fold-in quality gate failed; do not publish.")
    encoded = json.dumps([[artist["name"], *artist["vector"]] for artist in artists],
                         ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    probe = [("The Cure", 1.0), ("Metallica", -0.25)]
    return {
        "schemaVersion": 1,
        "modelVersion": "artist-bpr-fictional-v1",
        "trainedOn": "fictional-profiles",
        "curation": "independent-manual-affinities-v1",
        "realWorldQuality": "not_measured",
        "dimensions": 4,
        "minimumAffinity": MINIMUM_AFFINITY,
        "training": {**training, "profiles": len(positives), "interactions": sum(map(len, positives))},
        "evaluation": {"protocol": "fictional-new-profile-fold-in-v1", "validation": validation, "test": test},
        "embeddingSha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "artists": artists,
        "probe": {"signals": [{"artist": name, "weight": weight} for name, weight in probe],
                  "scores": [{"artist": name, "score": round(score, 8)} for name, score in rank_artists(artists, probe)]},
    }


def publish(artifact: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f".{target.name}-{uuid.uuid4().hex}.tmp")
    pending.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish only independently simulated artist embeddings; no DB or user input.")
    parser.add_argument("--output", type=Path, action="append", required=True)
    args = parser.parse_args()
    artifact = build_model()
    for target in args.output:
        publish(artifact, target)
    print(json.dumps({key: artifact[key] for key in ("modelVersion", "trainedOn", "embeddingSha256", "training", "evaluation", "realWorldQuality")}))


if __name__ == "__main__":
    main()
