"""Schema and conversion helpers for chunk-level online AWBC collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class OnlineAWBCFrame:
    """One low-level LeRobot action row emitted during an online episode."""

    dataset_index: int
    episode_index: int
    frame_index: int
    source: str
    phi: float

    def validate(self) -> None:
        if min(self.dataset_index, self.episode_index, self.frame_index) < 0:
            raise ValueError("online AWBC frame indices must be non-negative")
        if self.source not in {"policy", "expert"}:
            raise ValueError("online AWBC frame source must be policy or expert")
        if not 0.0 <= self.phi <= 1.0:
            raise ValueError("online AWBC phi must be in [0, 1]")


@dataclass(frozen=True)
class OnlineAWBCChunk:
    """The decision-level transition represented by the first row of a chunk."""

    dataset_index: int
    episode_index: int
    frame_index: int
    next_frame_index: int
    source: str
    phi: float
    phi_next: float
    vfd_score: float
    threshold: float
    success: bool

    def validate(self) -> None:
        if min(self.dataset_index, self.episode_index, self.frame_index) < 0:
            raise ValueError("online AWBC chunk indices must be non-negative")
        if self.next_frame_index <= self.frame_index:
            raise ValueError("online AWBC chunk must advance at least one frame")
        if self.source not in {"policy", "expert"}:
            raise ValueError("online AWBC chunk source must be policy or expert")
        if not 0.0 <= self.phi <= 1.0 or not 0.0 <= self.phi_next <= 1.0:
            raise ValueError("online AWBC chunk phi values must be in [0, 1]")

    def diagnostic_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_online_awbc_manifest(
    frames: Sequence[OnlineAWBCFrame], chunks: Iterable[OnlineAWBCChunk]
) -> list[dict[str, Any]]:
    """Create a full AWBC manifest with valid rows only at chunk boundaries."""

    ordered_frames = sorted(frames, key=lambda frame: frame.dataset_index)
    if [frame.dataset_index for frame in ordered_frames] != list(range(len(frames))):
        raise ValueError("online AWBC frames must use contiguous dataset indices")
    frame_by_index = {frame.dataset_index: frame for frame in ordered_frames}
    for frame in ordered_frames:
        frame.validate()

    chunk_by_index: dict[int, OnlineAWBCChunk] = {}
    chunks_per_episode: dict[int, int] = {}
    for chunk in chunks:
        chunk.validate()
        if chunk.dataset_index in chunk_by_index:
            raise ValueError(f"duplicate online AWBC chunk index {chunk.dataset_index}")
        frame = frame_by_index.get(chunk.dataset_index)
        if frame is None:
            raise ValueError(f"online AWBC chunk index {chunk.dataset_index} is out of range")
        if (frame.episode_index, frame.frame_index, frame.source) != (
            chunk.episode_index,
            chunk.frame_index,
            chunk.source,
        ):
            raise ValueError("online AWBC chunk does not match its frame metadata")
        chunk_by_index[chunk.dataset_index] = chunk
        chunks_per_episode[chunk.episode_index] = (
            chunks_per_episode.get(chunk.episode_index, 0) + 1
        )

    rows: list[dict[str, Any]] = []
    for frame in ordered_frames:
        chunk = chunk_by_index.get(frame.dataset_index)
        if chunk is None:
            rows.append(
                {
                    "dataset_index": frame.dataset_index,
                    "episode_index": frame.episode_index,
                    "frame_index": frame.frame_index,
                    "next_frame_index": frame.frame_index,
                    "phi": None,
                    "phi_next": None,
                    "delta_phi": None,
                    "valid": False,
                    "confidence": 0.0,
                    "episode_length_chunks": max(
                        1, chunks_per_episode.get(frame.episode_index, 0)
                    ),
                    "source": frame.source,
                    "success": False,
                }
            )
            continue

        rows.append(
            {
                "dataset_index": chunk.dataset_index,
                "episode_index": chunk.episode_index,
                "frame_index": chunk.frame_index,
                "next_frame_index": chunk.next_frame_index,
                "phi": chunk.phi,
                "phi_next": chunk.phi_next,
                "delta_phi": chunk.phi_next - chunk.phi,
                "valid": True,
                "confidence": 1.0,
                "episode_length_chunks": max(
                    1, chunks_per_episode.get(chunk.episode_index, 0)
                ),
                "source": chunk.source,
                "success": chunk.success,
            }
        )
    return rows
