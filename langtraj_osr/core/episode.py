"""Factorized Semantic Episode representation for LangTraj-OSR.

Each GPS/check-in trajectory is converted into a sequence of SemanticEpisodes,
where every continuous field is discretized into a compact vocabulary token.
A SemanticTrajectory bundles the episode sequence with metadata and labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, List, Optional


@dataclass(slots=True)
class SemanticEpisode:
    """Single stay-or-move segment, fully factorized into discrete tokens.

    Attributes:
        zone_id: Spatial zone identifier (grid cell or cluster id).
        poi_role: Urban functional role from a 64-class vocabulary (0-63).
        time_bin: Temporal bin = hour_of_day * 7 + day_of_week, giving 168 bins.
        dwell_bin: Discretized stay duration bucket (0-15).
        transition_type: How the user arrived: 0=walk, 1=drive, 2=transit, 3=jump.
        trip_length_change: Ratio of this trip segment length vs the user's
            historical average (>1 means longer than usual).
        event_flag: Whether a special event is happening nearby (0 or 1).
        companion_flag: Whether the user is co-located with a known companion
            (0 or 1).
    """

    zone_id: int = 0
    poi_role: int = 0
    time_bin: int = 0
    dwell_bin: int = 0
    transition_type: int = 0
    trip_length_change: float = 1.0
    event_flag: int = 0
    companion_flag: int = 0

    # ----- vocabulary sizes (class-level constants) -----
    POI_ROLE_VOCAB: ClassVar[int] = 64
    TIME_BIN_VOCAB: ClassVar[int] = 168
    DWELL_BIN_VOCAB: ClassVar[int] = 16
    TRANSITION_VOCAB: ClassVar[int] = 4

    def to_list(self) -> List[float]:
        """Serialise to a flat list of 8 numeric values."""
        return [
            float(self.zone_id),
            float(self.poi_role),
            float(self.time_bin),
            float(self.dwell_bin),
            float(self.transition_type),
            self.trip_length_change,
            float(self.event_flag),
            float(self.companion_flag),
        ]

    @staticmethod
    def from_list(values: List[float]) -> "SemanticEpisode":
        """Reconstruct from the flat 8-element list produced by `to_list`."""
        if len(values) != 8:
            raise ValueError(f"Expected 8 values, got {len(values)}")
        return SemanticEpisode(
            zone_id=int(values[0]),
            poi_role=int(values[1]),
            time_bin=int(values[2]),
            dwell_bin=int(values[3]),
            transition_type=int(values[4]),
            trip_length_change=float(values[5]),
            event_flag=int(values[6]),
            companion_flag=int(values[7]),
        )


@dataclass
class SemanticTrajectory:
    """A complete trajectory represented as a sequence of semantic episodes.

    Attributes:
        episodes: Ordered list of SemanticEpisode objects.
        user_id: Unique user identifier string.
        trip_id: Unique trip / trajectory identifier string.
        label: Anomaly concept id.  0 = normal, >0 = known anomaly concept,
            -1 = unknown anomaly (no concept definition available).
        primitive_labels: Per-episode 10-dim binary vector indicating which
            behavioural primitives are active.  ``None`` when labels are
            unavailable (e.g. unlabelled test data).
    """

    episodes: List[SemanticEpisode] = field(default_factory=list)
    user_id: str = ""
    trip_id: str = ""
    label: int = 0
    primitive_labels: Optional[List[List[int]]] = None

    # ---- convenience helpers ------------------------------------------------

    def __len__(self) -> int:
        return len(self.episodes)

    def to_tensor_list(self) -> List[List[float]]:
        """Return episodes as a list-of-lists (L x 8), ready for tensor conversion."""
        return [ep.to_list() for ep in self.episodes]

    def truncate(self, max_len: int) -> "SemanticTrajectory":
        """Return a copy truncated to *max_len* episodes."""
        return SemanticTrajectory(
            episodes=self.episodes[:max_len],
            user_id=self.user_id,
            trip_id=self.trip_id,
            label=self.label,
            primitive_labels=(
                self.primitive_labels[:max_len]
                if self.primitive_labels is not None
                else None
            ),
        )
