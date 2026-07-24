from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch

from placepulse_cusp.constants import CHOICE_TO_INDEX


def select_device(requested: str = "auto") -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false. "
                "Check the NVIDIA driver and install a CUDA-enabled PyTorch wheel."
            )
        return torch.device("cuda")
    if requested == "mps":
        if not (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "MPS was requested but is unavailable. Run outside restricted sandboxes "
                "on an Apple-silicon Mac with an MPS-enabled PyTorch build."
            )
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "auto":
        raise ValueError(f"Unsupported device: {requested}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_deterministic(seed: int, deterministic: bool = True) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False


@dataclass
class EncodedVotes:
    left: torch.Tensor
    right: torch.Tensor
    voter: torch.Tensor
    choice: torch.Tensor
    image_ids: list[str]
    voter_ids: list[str]
    vote_ids: list[str]

    @property
    def n_images(self) -> int:
        return len(self.image_ids)

    @property
    def n_voters(self) -> int:
        return len(self.voter_ids)

    @property
    def n_votes(self) -> int:
        return len(self.vote_ids)

    def to(self, device: torch.device) -> "EncodedVotes":
        return EncodedVotes(
            self.left.to(device),
            self.right.to(device),
            self.voter.to(device),
            self.choice.to(device),
            self.image_ids,
            self.voter_ids,
            self.vote_ids,
        )


class VoteEncoder:
    def __init__(self, image_ids: list[str] | None = None, voter_ids: list[str] | None = None):
        self.image_ids = image_ids or []
        self.voter_ids = voter_ids or []
        self.image_index = {value: i for i, value in enumerate(self.image_ids)}
        self.voter_index = {value: i for i, value in enumerate(self.voter_ids)}

    def fit(self, votes: pl.DataFrame) -> "VoteEncoder":
        self.image_ids = sorted(
            set(votes["left_image_id"].to_list()) | set(votes["right_image_id"].to_list())
        )
        self.voter_ids = sorted(
            value for value in votes["voter_id"].drop_nulls().unique().to_list()
        )
        self.image_index = {value: i for i, value in enumerate(self.image_ids)}
        self.voter_index = {value: i for i, value in enumerate(self.voter_ids)}
        return self

    def transform(self, votes: pl.DataFrame, device: torch.device | str = "cpu") -> EncodedVotes:
        left = np.asarray([self.image_index.get(x, -1) for x in votes["left_image_id"]], dtype=np.int64)
        right = np.asarray(
            [self.image_index.get(x, -1) for x in votes["right_image_id"]], dtype=np.int64
        )
        voter = np.asarray(
            [self.voter_index.get(x, -1) if x is not None else -1 for x in votes["voter_id"]],
            dtype=np.int64,
        )
        choice = np.asarray([CHOICE_TO_INDEX[x] for x in votes["choice"]], dtype=np.int64)
        valid = (left >= 0) & (right >= 0)
        if not valid.all():
            raise ValueError("Evaluation contains images absent from the training encoder.")
        target = torch.device(device)
        return EncodedVotes(
            torch.as_tensor(left, dtype=torch.long, device=target),
            torch.as_tensor(right, dtype=torch.long, device=target),
            torch.as_tensor(voter, dtype=torch.long, device=target),
            torch.as_tensor(choice, dtype=torch.long, device=target),
            self.image_ids,
            self.voter_ids,
            votes["vote_id"].to_list(),
        )


def davidson_logits(
    delta: torch.Tensor, log_tie: torch.Tensor, voter_tie: torch.Tensor | float = 0.0
) -> torch.Tensor:
    equal = torch.log(torch.tensor(2.0, device=delta.device)) + log_tie + voter_tie
    if equal.ndim == 0:
        equal = equal.expand_as(delta)
    return torch.stack((delta / 2.0, -delta / 2.0, equal), dim=-1)


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def result_payload(model_name: str, history: list[float], extra: dict[str, Any]) -> dict[str, Any]:
    return {"model": model_name, "loss_history": history, **extra}
