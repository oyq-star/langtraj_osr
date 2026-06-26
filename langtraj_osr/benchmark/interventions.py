"""Anomaly intervention operators for MobDef-Bench.

Eight executable interventions that transform normal SemanticTrajectory instances
into anomalous ones while setting the appropriate primitive_labels on each
affected episode.

Primitive label dimensions (10-dim binary vector per episode) — aligned with
core/concepts.py:
    0: unusual_time          — visit at an atypical hour for the user
    1: unusual_zone          — visit to a spatially atypical zone
    2: unusual_poi           — visit to a novel POI role category
    3: long_dwell            — abnormally long stay duration
    4: short_dwell           — abnormally short stay duration
    5: unusual_transition    — rare or impossible transport mode / order
    6: high_trip_deviation   — trip segment much longer than user average
    7: event_co_occurrence   — special event at the location
    8: companion_absence     — user is alone when usually accompanied
    9: companion_anomaly     — accompanied by unusual companion pattern
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..core.episode import SemanticEpisode, SemanticTrajectory

# Number of primitive dimensions
NUM_PRIMITIVES: int = 10

# Primitive index constants — MUST match core/concepts.py exactly
P_UNUSUAL_TIME = 0
P_UNUSUAL_ZONE = 1
P_UNUSUAL_POI = 2
P_LONG_DWELL = 3
P_SHORT_DWELL = 4
P_UNUSUAL_TRANSITION = 5
P_HIGH_TRIP_DEVIATION = 6
P_EVENT_CO_OCCURRENCE = 7
P_COMPANION_ABSENCE = 8
P_COMPANION_ANOMALY = 9


def _zero_labels(n_episodes: int) -> List[List[int]]:
    """Return n_episodes copies of a 10-dim zero vector."""
    return [[0] * NUM_PRIMITIVES for _ in range(n_episodes)]


def _ensure_labels(traj: SemanticTrajectory) -> List[List[int]]:
    """Return existing primitive_labels or create fresh zeros."""
    if traj.primitive_labels is not None:
        return [list(row) for row in traj.primitive_labels]
    return _zero_labels(len(traj.episodes))


def _deep_copy_traj(traj: SemanticTrajectory) -> SemanticTrajectory:
    """Deep-copy a trajectory so that the original is never mutated."""
    return SemanticTrajectory(
        episodes=[copy.copy(ep) for ep in traj.episodes],
        user_id=traj.user_id,
        trip_id=traj.trip_id,
        label=traj.label,
        primitive_labels=(
            [list(row) for row in traj.primitive_labels]
            if traj.primitive_labels is not None
            else None
        ),
    )


class InterventionEngine:
    """Applies eight anomaly intervention operators to semantic trajectories."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # 1. Time-shift
    # ------------------------------------------------------------------
    def time_shift(
        self,
        traj: SemanticTrajectory,
        target_hour_range: Tuple[int, int] = (0, 5),
    ) -> Tuple[SemanticTrajectory, List[List[int]]]:
        """Move all episodes into an unusual hour range.

        The *target_hour_range* is expressed as (start_hour, end_hour) in 24-h
        format.  Every episode's ``time_bin`` (= hour*7 + dow) is rewritten so
        that the hour component falls uniformly within the target range while
        preserving the day-of-week.
        """
        out = _deep_copy_traj(traj)
        labels = _ensure_labels(out)

        lo, hi = target_hour_range
        for i, ep in enumerate(out.episodes):
            dow = ep.time_bin % 7
            new_hour = int(self.rng.randint(lo, max(hi, lo + 1)))
            ep.time_bin = new_hour * 7 + dow
            labels[i][P_UNUSUAL_TIME] = 1  # prim 0

        out.primitive_labels = labels
        return out, labels

    # ------------------------------------------------------------------
    # 2. Role-swap
    # ------------------------------------------------------------------
    def role_swap(
        self,
        traj: SemanticTrajectory,
        target_role: Optional[int] = None,
    ) -> Tuple[SemanticTrajectory, List[List[int]]]:
        """Replace destination POI roles with an unusual target role.

        If *target_role* is ``None``, one is sampled from roles outside the
        trajectory's current role set.  Only non-home/non-work episodes are
        affected (heuristic: first and last episodes are treated as home, the
        episode with the longest dwell is treated as work).
        """
        out = _deep_copy_traj(traj)
        labels = _ensure_labels(out)

        existing_roles = {ep.poi_role for ep in out.episodes}
        if target_role is None:
            candidates = [r for r in range(SemanticEpisode.POI_ROLE_VOCAB)
                          if r not in existing_roles]
            if not candidates:
                candidates = list(range(SemanticEpisode.POI_ROLE_VOCAB))
            target_role = int(self.rng.choice(candidates))

        # Identify "swappable" indices — skip first, last (home), longest dwell (work)
        if len(out.episodes) <= 2:
            swap_indices = list(range(len(out.episodes)))
        else:
            dwells = [ep.dwell_bin for ep in out.episodes]
            work_idx = int(np.argmax(dwells[1:-1])) + 1
            swap_indices = [
                i for i in range(len(out.episodes))
                if i not in (0, len(out.episodes) - 1, work_idx)
            ]
            if not swap_indices:
                swap_indices = list(range(1, len(out.episodes) - 1)) or [0]

        # Swap at least one, up to all swappable
        n_swap = self.rng.randint(1, len(swap_indices) + 1)
        chosen = self.rng.choice(swap_indices, size=n_swap, replace=False)
        for idx in chosen:
            out.episodes[idx].poi_role = target_role
            labels[idx][P_UNUSUAL_POI] = 1  # prim 2

        out.primitive_labels = labels
        return out, labels

    # ------------------------------------------------------------------
    # 3. Destination-substitution
    # ------------------------------------------------------------------
    def destination_substitution(
        self,
        traj: SemanticTrajectory,
        target_zone: Optional[int] = None,
    ) -> Tuple[SemanticTrajectory, List[List[int]]]:
        """Route one or more episodes to an unusual zone.

        If *target_zone* is ``None``, a zone far from the user's habitual set
        is sampled (zone_id offset by a large random amount).
        """
        out = _deep_copy_traj(traj)
        labels = _ensure_labels(out)

        existing_zones = {ep.zone_id for ep in out.episodes}
        if target_zone is None:
            max_zone = max(existing_zones) if existing_zones else 100
            target_zone = int(max_zone + self.rng.randint(50, 200))

        # Pick one or two episodes to redirect
        n_sub = min(self.rng.randint(1, 3), len(out.episodes))
        indices = self.rng.choice(len(out.episodes), size=n_sub, replace=False)
        for idx in indices:
            out.episodes[idx].zone_id = target_zone
            out.episodes[idx].trip_length_change = float(
                self.rng.uniform(2.0, 5.0)
            )
            labels[idx][P_UNUSUAL_ZONE] = 1       # prim 1
            labels[idx][P_HIGH_TRIP_DEVIATION] = 1  # prim 6

        out.primitive_labels = labels
        return out, labels

    # ------------------------------------------------------------------
    # 4. Detour-insertion
    # ------------------------------------------------------------------
    def detour_insertion(
        self,
        traj: SemanticTrajectory,
        detour_length: int = 3,
        unusual_roles: Optional[List[int]] = None,
    ) -> Tuple[SemanticTrajectory, List[List[int]]]:
        """Insert an off-route segment of *detour_length* episodes.

        The detour is placed at a random interior position.  Each inserted
        episode uses a zone/role unlikely for the user.
        """
        out = _deep_copy_traj(traj)
        labels = _ensure_labels(out)

        if unusual_roles is None or len(unusual_roles) == 0:
            existing_roles = {ep.poi_role for ep in out.episodes}
            unusual_roles = [
                r for r in range(SemanticEpisode.POI_ROLE_VOCAB)
                if r not in existing_roles
            ]
            if not unusual_roles:
                unusual_roles = list(range(SemanticEpisode.POI_ROLE_VOCAB))

        # Insertion point: somewhere in the middle
        insert_pos = self.rng.randint(1, max(2, len(out.episodes)))

        existing_zones = [ep.zone_id for ep in out.episodes]
        max_zone = max(existing_zones) if existing_zones else 50

        detour_episodes: List[SemanticEpisode] = []
        detour_labels: List[List[int]] = []
        for _ in range(detour_length):
            ep = SemanticEpisode(
                zone_id=int(max_zone + self.rng.randint(10, 100)),
                poi_role=int(self.rng.choice(unusual_roles)),
                time_bin=out.episodes[min(insert_pos, len(out.episodes) - 1)].time_bin,
                dwell_bin=int(self.rng.randint(1, 8)),
                transition_type=int(self.rng.choice([1, 2, 3])),
                trip_length_change=float(self.rng.uniform(1.5, 4.0)),
                event_flag=0,
                companion_flag=0,
            )
            detour_episodes.append(ep)
            row = [0] * NUM_PRIMITIVES
            row[P_UNUSUAL_ZONE] = 1       # prim 1
            row[P_HIGH_TRIP_DEVIATION] = 1  # prim 6
            detour_labels.append(row)

        out.episodes = (
            out.episodes[:insert_pos]
            + detour_episodes
            + out.episodes[insert_pos:]
        )
        labels = labels[:insert_pos] + detour_labels + labels[insert_pos:]

        out.primitive_labels = labels
        return out, labels

    # ------------------------------------------------------------------
    # 5. Missing-stop removal
    # ------------------------------------------------------------------
    def missing_stop_removal(
        self,
        traj: SemanticTrajectory,
        stop_index: Optional[int] = None,
    ) -> Tuple[SemanticTrajectory, List[List[int]]]:
        """Remove a habitual stop from the trajectory.

        The *stop_index* identifies which episode to remove.  If ``None``, the
        episode with the highest dwell (excluding first/last) is removed —
        simulating a skipped regular stop.  Neighbouring episodes are flagged.
        """
        if len(traj.episodes) <= 2:
            # Too short to remove; return a copy with no modification
            out = _deep_copy_traj(traj)
            labels = _ensure_labels(out)
            out.primitive_labels = labels
            return out, labels

        out = _deep_copy_traj(traj)
        labels = _ensure_labels(out)

        if stop_index is None:
            dwells = [ep.dwell_bin for ep in out.episodes[1:-1]]
            stop_index = int(np.argmax(dwells)) + 1

        stop_index = max(0, min(stop_index, len(out.episodes) - 1))

        # Remove the episode
        out.episodes.pop(stop_index)
        labels.pop(stop_index)

        # Flag neighbours
        for ni in [stop_index - 1, stop_index]:
            if 0 <= ni < len(labels):
                labels[ni][P_UNUSUAL_TRANSITION] = 1  # prim 5

        out.primitive_labels = labels
        return out, labels

    # ------------------------------------------------------------------
    # 6. Dwell-inflation
    # ------------------------------------------------------------------
    def dwell_inflation(
        self,
        traj: SemanticTrajectory,
        factor: float = 5.0,
        episode_idx: Optional[int] = None,
    ) -> Tuple[SemanticTrajectory, List[List[int]]]:
        """Extend the stay duration of one or more episodes.

        The ``dwell_bin`` is multiplied by *factor* (clamped to the vocab max).
        """
        out = _deep_copy_traj(traj)
        labels = _ensure_labels(out)

        if episode_idx is not None:
            indices = [max(0, min(episode_idx, len(out.episodes) - 1))]
        else:
            # Pick 1-2 random episodes
            n = min(self.rng.randint(1, 3), len(out.episodes))
            indices = list(self.rng.choice(len(out.episodes), size=n, replace=False))

        max_dwell = SemanticEpisode.DWELL_BIN_VOCAB - 1
        for idx in indices:
            original = out.episodes[idx].dwell_bin
            if factor >= 1.0:
                inflated = min(int(original * factor), max_dwell)
                out.episodes[idx].dwell_bin = max(inflated, min(max_dwell, original + 5))
                labels[idx][P_LONG_DWELL] = 1   # prim 3
            else:
                deflated = max(0, int(original * factor))
                out.episodes[idx].dwell_bin = min(deflated, max(0, original - 3))
                labels[idx][P_SHORT_DWELL] = 1  # prim 4

        out.primitive_labels = labels
        return out, labels

    # ------------------------------------------------------------------
    # 7. Order-permutation
    # ------------------------------------------------------------------
    def order_permutation(
        self,
        traj: SemanticTrajectory,
        segment_start: Optional[int] = None,
        segment_end: Optional[int] = None,
    ) -> Tuple[SemanticTrajectory, List[List[int]]]:
        """Scramble the visit order of a contiguous sub-sequence.

        The first and last episodes (home anchors) are preserved by default.
        """
        out = _deep_copy_traj(traj)
        labels = _ensure_labels(out)

        n = len(out.episodes)
        if n <= 3:
            # Too short to meaningfully permute; flag everything
            for i in range(n):
                labels[i][P_UNUSUAL_TRANSITION] = 1  # prim 5
            if n > 1:
                mid = list(range(1, n - 1)) if n > 2 else list(range(n))
                self.rng.shuffle(mid)
                if n > 2:
                    reordered = [out.episodes[0]]
                    reordered += [out.episodes[i] for i in mid]
                    reordered.append(out.episodes[-1])
                    out.episodes = reordered
            out.primitive_labels = labels
            return out, labels

        if segment_start is None:
            segment_start = 1
        if segment_end is None:
            segment_end = n - 1

        segment_start = max(0, min(segment_start, n - 1))
        segment_end = max(segment_start + 1, min(segment_end, n))

        interior = list(range(segment_start, segment_end))
        self.rng.shuffle(interior)

        reordered_episodes = list(out.episodes[:segment_start])
        reordered_labels = list(labels[:segment_start])

        for idx in interior:
            reordered_episodes.append(out.episodes[idx])
            row = list(labels[idx])
            row[P_UNUSUAL_TRANSITION] = 1  # prim 5
            reordered_labels.append(row)

        reordered_episodes.extend(out.episodes[segment_end:])
        reordered_labels.extend(labels[segment_end:])

        out.episodes = reordered_episodes
        out.primitive_labels = reordered_labels
        return out, reordered_labels

    # ------------------------------------------------------------------
    # 8. Event-conflict
    # ------------------------------------------------------------------
    def event_conflict(
        self,
        traj: SemanticTrajectory,
    ) -> Tuple[SemanticTrajectory, List[List[int]]]:
        """Create a trip that conflicts with the event context.

        Episodes with ``event_flag=1`` get their roles and zones changed to
        something incompatible with the event.  If no event is flagged, one or
        more episodes are given ``event_flag=1`` and then made conflicting.
        """
        out = _deep_copy_traj(traj)
        labels = _ensure_labels(out)

        event_indices = [i for i, ep in enumerate(out.episodes) if ep.event_flag == 1]

        if not event_indices:
            # Inject event flags on 1-2 episodes
            n_flag = min(self.rng.randint(1, 3), len(out.episodes))
            event_indices = list(
                self.rng.choice(len(out.episodes), size=n_flag, replace=False)
            )
            for idx in event_indices:
                out.episodes[idx].event_flag = 1

        existing_roles = {ep.poi_role for ep in out.episodes}
        conflict_roles = [
            r for r in range(SemanticEpisode.POI_ROLE_VOCAB)
            if r not in existing_roles
        ]
        if not conflict_roles:
            conflict_roles = list(range(SemanticEpisode.POI_ROLE_VOCAB))

        for idx in event_indices:
            ep = out.episodes[idx]
            ep.poi_role = int(self.rng.choice(conflict_roles))
            ep.zone_id = ep.zone_id + int(self.rng.randint(30, 100))
            ep.trip_length_change = float(self.rng.uniform(2.0, 5.0))
            labels[idx][P_EVENT_CO_OCCURRENCE] = 1  # prim 7
            labels[idx][P_UNUSUAL_POI] = 1          # prim 2
            labels[idx][P_UNUSUAL_ZONE] = 1         # prim 1

        out.primitive_labels = labels
        return out, labels
