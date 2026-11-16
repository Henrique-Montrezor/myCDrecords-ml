import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F


class Metrics(TypedDict):
    recall: float
    ndcg: float


class RankedItem(TypedDict):
    albumId: str
    score: float


class Profile(TypedDict):
    id: str
    trainingAlbums: list[str]
    heldOutAlbum: str
    recommendations: list[RankedItem]
    baseline: list[RankedItem]


class Training(TypedDict):
    initialLoss: float
    finalLoss: float
    epochs: int
    seed: int
    torchVersion: str


class Evaluation(TypedDict):
    protocol: str
    k: int
    model: Metrics
    baseline: Metrics


class Snapshot(TypedDict):
    schemaVersion: int
    mode: str
    modelVersion: str
    datasetVersion: str
    snapshotId: str
    generatedAt: str
    promotionAllowed: bool
    realWorldQuality: str
    training: Training
    evaluation: Evaluation
    profiles: list[Profile]


def build_fixture() -> list[tuple[int, list[int], int]]:
    profiles = []
    for user in range(12):
        family = list(range((user // 4) * 4, (user // 4) * 4 + 4))
        held_out = family[user % 4]
        profiles.append((user, [album for album in family if album != held_out], held_out))
    return profiles


class MatrixFactorization(nn.Module):
    def __init__(self, users: int = 12, albums: int = 12) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(users, 4)
        self.album_embeddings = nn.Embedding(albums, 4)
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.album_embeddings.weight, std=0.1)

    def forward(self, users: torch.Tensor, albums: torch.Tensor) -> torch.Tensor:
        return (self.user_embeddings(users) * self.album_embeddings(albums)).sum(dim=-1)


def evaluate(rankings: list[list[str]], targets: list[str], k: int) -> Metrics:
    if not rankings or len(rankings) != len(targets) or k < 1:
        raise ValueError("Expected non-empty matched rankings/targets and k >= 1.")
    hits = 0
    gain = 0.0
    for ranking, target in zip(rankings, targets, strict=True):
        if len(ranking) != len(set(ranking)):
            raise ValueError("Ranking contains duplicate albums.")
        if target in ranking[:k]:
            position = ranking.index(target)
            hits += 1
            gain += 1 / math.log2(position + 2)
    return {"recall": hits / len(targets), "ndcg": gain / len(targets)}


def fit_bpr(positives_by_user: list[list[int]], album_count: int, epochs: int) -> tuple[MatrixFactorization, Training]:
    if not positives_by_user or album_count < 2 or epochs < 1:
        raise ValueError("Expected training users, at least two albums and positive epochs.")
    if any(not positives or len(positives) != len(set(positives))
           or any(album < 0 or album >= album_count for album in positives)
           or len(positives) == album_count for positives in positives_by_user):
        raise ValueError("Each user needs unique eligible positives and at least one contrast.")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = MatrixFactorization(len(positives_by_user), album_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    # Unseen items include the holdout: sampling never consults test labels.
    triples = [(user, positive, contrast) for user, positives in enumerate(positives_by_user)
               for positive in positives for contrast in range(album_count) if contrast not in positives]
    batch = torch.tensor(triples, dtype=torch.long)

    def loss() -> torch.Tensor:
        difference = model(batch[:, 0], batch[:, 1]) - model(batch[:, 0], batch[:, 2])
        regularization = sum(parameter.square().mean() for parameter in model.parameters())
        return F.softplus(-difference).mean() + 0.005 * regularization

    initial_loss = loss().item()
    for _ in range(epochs):
        optimizer.zero_grad()
        objective = loss()
        if not torch.isfinite(objective):
            raise RuntimeError("Training produced a non-finite loss.")
        objective.backward()
        optimizer.step()
    model.eval()
    final_loss = loss().item()
    if not math.isfinite(final_loss):
        raise RuntimeError("Training produced a non-finite final loss.")
    return model, {"initialLoss": initial_loss, "finalLoss": final_loss, "epochs": epochs, "seed": 42, "torchVersion": str(torch.__version__)}


def train() -> tuple[Snapshot, MatrixFactorization]:
    fixture = build_fixture()
    model, training = fit_bpr([positives for _, positives, _ in fixture], 12, 200)
    popularity = [sum(album in positives for _, positives, _ in fixture) for album in range(12)]
    profiles: list[Profile] = []
    all_rankings: list[list[str]] = []
    all_baselines: list[list[str]] = []
    targets: list[str] = []

    def album_id(index: int) -> str:
        return f"synthetic-album-{index + 1:02d}"

    with torch.inference_mode():
        for user, positives, held_out in fixture:
            scores = model(torch.full((12,), user, dtype=torch.long), torch.arange(12)).tolist()
            if not all(math.isfinite(score) for score in scores):
                raise RuntimeError("Inference produced non-finite scores.")
            candidates = [album for album in range(12) if album not in positives]
            ranked = sorted(candidates, key=lambda album: (-scores[album], album))
            baseline = sorted(candidates, key=lambda album: (-popularity[album], album))
            all_rankings.append([album_id(album) for album in ranked])
            all_baselines.append([album_id(album) for album in baseline])
            targets.append(album_id(held_out))
            profiles.append({
                "id": f"synthetic-user-{user + 1:02d}",
                "trainingAlbums": [album_id(album) for album in positives],
                "heldOutAlbum": album_id(held_out),
                "recommendations": [{"albumId": album_id(album), "score": round(scores[album], 6)} for album in ranked[:6]],
                "baseline": [{"albumId": album_id(album), "score": float(popularity[album])} for album in baseline[:6]],
            })

    evaluation: Evaluation = {"protocol": "synthetic-leave-one-out", "k": 3, "model": evaluate(all_rankings, targets, 3), "baseline": evaluate(all_baselines, targets, 3)}
    identity = json.dumps({"profiles": profiles, "training": training, "evaluation": evaluation}, sort_keys=True, allow_nan=False)
    snapshot: Snapshot = {
        "schemaVersion": 1, "mode": "synthetic", "modelVersion": "pytorch-bpr-v1", "datasetVersion": "synthetic-families-v1",
        "snapshotId": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "promotionAllowed": False, "realWorldQuality": "not_measured",
        "training": training, "evaluation": evaluation, "profiles": profiles,
    }
    return snapshot, model


def publish(snapshot: Snapshot, model: MatrixFactorization, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / f'{snapshot["snapshotId"]}.pt'
    torch.save(model.state_dict(), checkpoint)
    pending = directory / f".snapshot-{uuid.uuid4().hex}.tmp"
    pending.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = directory / "snapshot.json"
    pending.replace(target)
    return target


def load_model(checkpoint: Path) -> MatrixFactorization:
    model = MatrixFactorization()
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    model.eval()
    return model
