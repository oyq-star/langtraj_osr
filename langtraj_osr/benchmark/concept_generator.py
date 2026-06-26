"""Anomaly concept generator for MobDef-Bench.

Composes 1-3 interventions per concept definition to produce anomalous
trajectories.  25 concepts are organised into four splits:

    A_seen      (12)  — single / double primitive compositions (training)
    A_zs_comp   ( 6)  — novel compositions of seen primitives (zero-shot)
    A_zs_family ( 4)  — held-out operator family: companion-based
    A_unknown   ( 3)  — no definition, no training signal
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.episode import SemanticEpisode, SemanticTrajectory
from .interventions import (
    NUM_PRIMITIVES,
    P_COMPANION_ABSENCE,
    P_COMPANION_ANOMALY,
    P_EVENT_CO_OCCURRENCE,
    P_HIGH_TRIP_DEVIATION,
    P_LONG_DWELL,
    P_SHORT_DWELL,
    P_UNUSUAL_POI,
    P_UNUSUAL_TIME,
    P_UNUSUAL_TRANSITION,
    P_UNUSUAL_ZONE,
    InterventionEngine,
    _deep_copy_traj,
    _ensure_labels,
)

# ======================================================================
# Concept definitions — maps concept_id -> (name, split, pipeline)
# ======================================================================

# Splits
SPLIT_SEEN = "A_seen"
SPLIT_ZS_COMP = "A_zs_comp"
SPLIT_ZS_FAMILY = "A_zs_family"
SPLIT_UNKNOWN = "A_unknown"


@dataclass
class ConceptDef:
    """Declarative definition of one anomaly concept."""
    concept_id: int
    name: str
    split: str
    description: str
    # Each step is (method_name, kwargs_dict)
    pipeline: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)


# ----- A_seen (12) — aligned with core/concepts.py --------------------
# IMPORTANT: concept_id, name, and primitives MUST match core/concepts.py exactly.

CONCEPT_DEFS: List[ConceptDef] = [
    # 1: late_night_industrial (primitives [0,1])
    ConceptDef(
        concept_id=1,
        name="late_night_industrial",
        split=SPLIT_SEEN,
        description="Residential user visiting industrial zones between midnight and 5 AM.",
        pipeline=[
            ("time_shift", {"target_hour_range": (0, 5)}),
            ("role_swap", {"target_role": 10}),  # 10 = industrial
        ],
    ),
    # 2: excessive_dwell_commercial (primitives [2,3])
    ConceptDef(
        concept_id=2,
        name="excessive_dwell_commercial",
        split=SPLIT_SEEN,
        description="User spending more than 6 hours at a commercial venue they have never visited before.",
        pipeline=[
            ("destination_substitution", {}),       # novel POI (unusual_poi prim 2)
            ("dwell_inflation", {"factor": 8.0}),   # >6 h dwell (long_dwell prim 3)
        ],
    ),
    # 3: rapid_cross_city_jump (primitives [5,6])
    ConceptDef(
        concept_id=3,
        name="rapid_cross_city_jump",
        split=SPLIT_SEEN,
        description="Instantaneous teleportation-like jump across the city without plausible transit time.",
        pipeline=[
            ("destination_substitution", {}),       # far-away unusual zone (prim 6)
            ("order_permutation", {}),              # breaks expected transit sequence (prim 5)
        ],
    ),
    # 4: weekend_office_district (primitives [0,1])
    ConceptDef(
        concept_id=4,
        name="weekend_office_district",
        split=SPLIT_SEEN,
        description="User visiting a central business district on a weekend when they normally only go on weekdays.",
        pipeline=[
            ("time_shift", {"target_hour_range": (10, 16), "target_weekday": False}),  # weekend (prim 0)
            ("role_swap", {"target_role": 5}),    # 5 = office zone (prim 1)
        ],
    ),
    # 5: fleeting_hospital_visit (primitives [2,4])
    ConceptDef(
        concept_id=5,
        name="fleeting_hospital_visit",
        split=SPLIT_SEEN,
        description="Extremely brief visit (under 3 minutes) to a hospital by a user with no medical POI history.",
        pipeline=[
            ("role_swap", {"target_role": 11}),      # 11 = medical/hospital (prim 2)
            ("dwell_inflation", {"factor": 0.04}),   # <3 min (short_dwell prim 4)
        ],
    ),
    # 6: midnight_park_loiter (primitives [0,3])
    ConceptDef(
        concept_id=6,
        name="midnight_park_loiter",
        split=SPLIT_SEEN,
        description="User staying in a public park for over 2 hours between 11 PM and 5 AM.",
        pipeline=[
            ("time_shift", {"target_hour_range": (23, 5)}),  # midnight window (prim 0)
            ("role_swap", {"target_role": 2}),               # 2 = park/green (prim 3: unusual for this time)
            ("dwell_inflation", {"factor": 6.0}),            # 2+ hours (long_dwell prim 3)
        ],
    ),
    # 7: solo_nightclub_deviation (primitives [0,2,8])
    ConceptDef(
        concept_id=7,
        name="solo_nightclub_deviation",
        split=SPLIT_SEEN,
        description="User who normally visits nightlife venues with companions arrives alone at a nightclub at an unusual hour.",
        pipeline=[
            ("time_shift", {"target_hour_range": (1, 4)}),   # unusual hour (prim 0)
            ("role_swap", {"target_role": 20}),              # 20 = nightclub (prim 2)
            ("_companion_remove", {}),                       # alone (companion_absence prim 8)
        ],
    ),
    # 8: event_zone_avoidance (primitives [1,7])
    ConceptDef(
        concept_id=8,
        name="event_zone_avoidance",
        split=SPLIT_SEEN,
        description="User deliberately detouring around an area hosting a major event they would normally attend.",
        pipeline=[
            ("destination_substitution", {}),  # unusual zone detour (prim 1)
            ("event_conflict", {}),            # event co-occurrence flag set (prim 7)
        ],
    ),
    # 9: transit_mode_switch_loop (primitives [5,6])
    ConceptDef(
        concept_id=9,
        name="transit_mode_switch_loop",
        split=SPLIT_SEEN,
        description="Rapid alternation between walking and driving within a single trip, creating a loop pattern.",
        pipeline=[
            ("order_permutation", {}),          # disrupts normal transit sequence (prim 5)
            ("detour_insertion", {"detour_length": 3}),  # loop/detour adds high deviation (prim 6)
        ],
    ),
    # 10: repeated_short_visits_different_zones (primitives [1,4,6])
    ConceptDef(
        concept_id=10,
        name="repeated_short_visits_different_zones",
        split=SPLIT_SEEN,
        description="Sequence of very short visits (under 5 min each) to 4+ distinct zones within one hour.",
        pipeline=[
            ("destination_substitution", {}),     # unusual zone (prim 1)
            ("dwell_inflation", {"factor": 0.05}),  # very short stays (short_dwell prim 4)
        ],
    ),
    # 11: companion_switch_at_sensitive_poi (primitives [2,9])
    ConceptDef(
        concept_id=11,
        name="companion_switch_at_sensitive_poi",
        split=SPLIT_SEEN,
        description="User arrives at a government or financial POI with an unusual companion pattern.",
        pipeline=[
            ("role_swap", {"target_role": 12}),   # 12 = government/financial (unusual_poi prim 2)
            ("_companion_add", {}),               # new co-travelling device (companion_anomaly prim 9)
        ],
    ),
    # 12: long_trip_to_airport_no_history (primitives [1,2,6])
    ConceptDef(
        concept_id=12,
        name="long_trip_to_airport_no_history",
        split=SPLIT_SEEN,
        description="User with no prior airport visits makes a long-distance trip to an airport zone.",
        pipeline=[
            ("destination_substitution", {}),    # unusual far zone (prim 1 + prim 6)
            ("role_swap", {"target_role": 14}),  # 14 = airport (unusual_poi prim 2)
        ],
    ),

    # ----- A_zs_comp (6) — aligned with core/concepts.py ----------------
    # 13: midnight_commercial_with_event (primitives [0,2,7])
    ConceptDef(
        concept_id=13,
        name="midnight_commercial_with_event",
        split=SPLIT_ZS_COMP,
        description="User visiting a commercial venue after midnight while a major event is underway.",
        pipeline=[
            ("time_shift", {"target_hour_range": (0, 4)}),   # midnight (prim 0)
            ("role_swap", {"target_role": 7}),               # 7 = commercial (prim 2)
            ("event_conflict", {}),                          # event co-occurrence (prim 7)
        ],
    ),
    # 14: solo_long_dwell_unfamiliar_residential (primitives [1,3,8])
    ConceptDef(
        concept_id=14,
        name="solo_long_dwell_unfamiliar_residential",
        split=SPLIT_ZS_COMP,
        description="User staying alone for an extended period in an unfamiliar residential zone.",
        pipeline=[
            ("destination_substitution", {}),    # unfamiliar zone (prim 1)
            ("dwell_inflation", {"factor": 7.0}),  # extended stay (prim 3)
            ("_companion_remove", {}),           # alone (prim 8)
        ],
    ),
    # 15: rapid_mode_switch_with_companion_change (primitives [5,9])
    ConceptDef(
        concept_id=15,
        name="rapid_mode_switch_with_companion_change",
        split=SPLIT_ZS_COMP,
        description="User rapidly alternates transport modes while companion signature changes mid-trip.",
        pipeline=[
            ("order_permutation", {}),   # mode disruption sequence (prim 5)
            ("_companion_add", {}),      # companion change (prim 9)
        ],
    ),
    # 16: short_dwell_chain_with_long_trip (primitives [4,6])
    ConceptDef(
        concept_id=16,
        name="short_dwell_chain_with_long_trip",
        split=SPLIT_ZS_COMP,
        description="Chain of short dwell times combined with a far-reaching trip segment.",
        pipeline=[
            ("dwell_inflation", {"factor": 0.08}),   # short dwells (prim 4)
            ("destination_substitution", {}),         # long spatial jump (prim 6)
        ],
    ),
    # 17: unusual_time_event_zone_alone (primitives [0,7,8])
    ConceptDef(
        concept_id=17,
        name="unusual_time_event_zone_alone",
        split=SPLIT_ZS_COMP,
        description="User alone at an event zone during an unusual hour for them, while the event is active.",
        pipeline=[
            ("time_shift", {"target_hour_range": (2, 5)}),  # unusual hour (prim 0)
            ("event_conflict", {}),                          # event zone (prim 7)
            ("_companion_remove", {}),                       # alone (prim 8)
        ],
    ),
    # 18: novel_poi_with_dwell_and_transition_anomaly (primitives [2,3,5])
    ConceptDef(
        concept_id=18,
        name="novel_poi_with_dwell_and_transition_anomaly",
        split=SPLIT_ZS_COMP,
        description="First-time visit to a new POI type with both an abnormally long stay and an unusual transit mode.",
        pipeline=[
            ("role_swap", {}),                       # new POI role (prim 2)
            ("dwell_inflation", {"factor": 5.0}),    # excessive dwell (prim 3)
            ("order_permutation", {}),               # unusual transit mode/order (prim 5)
        ],
    ),

    # ----- A_zs_family (4) — spatial deviation (aligns with core/concepts.py) ---
    # 19: systematic_boundary_probing (primitives [1,6])
    ConceptDef(
        concept_id=19,
        name="systematic_boundary_probing",
        split=SPLIT_ZS_FAMILY,
        description="User repeatedly visits the edges of their usual activity space, probing zones just beyond their normal boundary.",
        pipeline=[
            ("destination_substitution", {}),         # peripheral unusual zone (prim 1)
            ("detour_insertion", {"detour_length": 4}),  # high spatial deviation (prim 6)
        ],
    ),
    # 20: spatial_anchor_shift (primitives [1,3])
    ConceptDef(
        concept_id=20,
        name="spatial_anchor_shift",
        split=SPLIT_ZS_FAMILY,
        description="User's primary activity anchor abruptly shifts to a new zone with long dwell times.",
        pipeline=[
            ("destination_substitution", {}),       # shift to new zone (prim 1)
            ("dwell_inflation", {"factor": 8.0}),   # long dwell at new anchor (prim 3)
        ],
    ),
    # 21: oscillating_zone_revisit (primitives [1,5,6])
    ConceptDef(
        concept_id=21,
        name="oscillating_zone_revisit",
        split=SPLIT_ZS_FAMILY,
        description="User oscillates between two distant zones multiple times in a single day, using different transport modes.",
        pipeline=[
            ("destination_substitution", {}),          # unusual distant zone (prim 1)
            ("order_permutation", {}),                 # different transport order (prim 5)
            ("detour_insertion", {"detour_length": 3}),  # back-and-forth deviation (prim 6)
        ],
    ),
    # 22: coverage_maximisation_pattern (primitives [1,4,6])
    ConceptDef(
        concept_id=22,
        name="coverage_maximisation_pattern",
        split=SPLIT_ZS_FAMILY,
        description="User visits an unusually large number of distinct zones in a single day with minimal dwell at each.",
        pipeline=[
            ("destination_substitution", {}),           # many new zones (prim 1)
            ("dwell_inflation", {"factor": 0.05}),      # minimal dwell (prim 4)
        ],
    ),

    # ----- A_unknown (3) — no definition, no training --------------------
    # 23
    ConceptDef(
        concept_id=23,
        name="crypto_meeting",
        split=SPLIT_UNKNOWN,
        description="Complex multi-primitive anomaly without predefined structure.",
        pipeline=[
            ("time_shift", {"target_hour_range": (2, 4)}),
            ("destination_substitution", {}),
            ("dwell_inflation", {"factor": 3.0}),
            ("role_swap", {}),
        ],
    ),
    # 24
    ConceptDef(
        concept_id=24,
        name="surveillance_pattern",
        split=SPLIT_UNKNOWN,
        description="Loop + dwell + destination pattern suggesting surveillance.",
        pipeline=[
            ("detour_insertion", {"detour_length": 4}),
            ("dwell_inflation", {"factor": 6.0}),
            ("destination_substitution", {}),
        ],
    ),
    # 25
    ConceptDef(
        concept_id=25,
        name="dead_drop",
        split=SPLIT_UNKNOWN,
        description="Detour + dwell + missing stop pattern.",
        pipeline=[
            ("detour_insertion", {"detour_length": 2}),
            ("dwell_inflation", {"factor": 5.0}),
            ("missing_stop_removal", {}),
        ],
    ),
]

# Quick lookup by name / id
CONCEPT_BY_NAME: Dict[str, ConceptDef] = {c.name: c for c in CONCEPT_DEFS}
CONCEPT_BY_ID: Dict[int, ConceptDef] = {c.concept_id: c for c in CONCEPT_DEFS}


class ConceptGenerator:
    """Generate anomalous trajectories by composing interventions per concept."""

    def __init__(self, intervention_engine: InterventionEngine) -> None:
        self.engine = intervention_engine

    # ------------------------------------------------------------------
    # Companion helpers (used by A_zs_family concepts)
    # ------------------------------------------------------------------

    def _companion_remove(
        self,
        traj: SemanticTrajectory,
        **kwargs: Any,
    ) -> Tuple[SemanticTrajectory, List[List[int]]]:
        """Remove companion flags from episodes that have them."""
        out = _deep_copy_traj(traj)
        labels = _ensure_labels(out)
        for i, ep in enumerate(out.episodes):
            if ep.companion_flag == 1:
                ep.companion_flag = 0
                labels[i][P_COMPANION_ABSENCE] = 1  # prim 8
        # If no companions were present, flag a random episode
        if not any(row[P_COMPANION_ABSENCE] for row in labels):
            idx = int(self.engine.rng.randint(0, len(out.episodes)))
            labels[idx][P_COMPANION_ABSENCE] = 1
        out.primitive_labels = labels
        return out, labels

    def _companion_add(
        self,
        traj: SemanticTrajectory,
        **kwargs: Any,
    ) -> Tuple[SemanticTrajectory, List[List[int]]]:
        """Add unexpected companion flags."""
        out = _deep_copy_traj(traj)
        labels = _ensure_labels(out)
        n_add = min(self.engine.rng.randint(1, 4), len(out.episodes))
        indices = self.engine.rng.choice(
            len(out.episodes), size=n_add, replace=False
        )
        for idx in indices:
            out.episodes[idx].companion_flag = 1
            labels[idx][P_COMPANION_ANOMALY] = 1
        out.primitive_labels = labels
        return out, labels

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _resolve_method(self, method_name: str) -> Callable[..., Tuple[SemanticTrajectory, List[List[int]]]]:
        """Return the callable for a given method name."""
        # Companion helpers live on this class
        if method_name.startswith("_companion"):
            return getattr(self, method_name)
        return getattr(self.engine, method_name)

    def generate_concept(
        self,
        traj: SemanticTrajectory,
        concept_def: ConceptDef,
    ) -> Tuple[SemanticTrajectory, int, List[List[int]]]:
        """Apply the concept's intervention pipeline to *traj*.

        Returns:
            (anomalous_trajectory, concept_id, primitive_labels)
        """
        current = _deep_copy_traj(traj)
        # Initialise labels
        current.primitive_labels = _ensure_labels(current)

        for method_name, kwargs in concept_def.pipeline:
            method = self._resolve_method(method_name)

            # Handle special stop_index=-1 (last episode)
            call_kwargs = dict(kwargs)
            if "stop_index" in call_kwargs and call_kwargs["stop_index"] == -1:
                call_kwargs["stop_index"] = len(current.episodes) - 1

            current, step_labels = method(current, **call_kwargs)
            # Merge labels (OR): step_labels may have different length if
            # episodes were inserted/removed, so we take whatever the method
            # produced.
            current.primitive_labels = step_labels

        current.label = concept_def.concept_id
        current.trip_id = f"{traj.trip_id}_c{concept_def.concept_id}"
        return current, concept_def.concept_id, current.primitive_labels

    def generate_all_concepts(
        self,
        normal_trajectories: List[SemanticTrajectory],
        concepts_config: Optional[List[ConceptDef]] = None,
    ) -> List[Tuple[SemanticTrajectory, int, List[List[int]]]]:
        """Generate anomalous trajectories for every concept.

        For each concept, a random subset of *normal_trajectories* is selected
        and the concept pipeline is applied.

        Args:
            normal_trajectories: Pool of normal trajectories to perturb.
            concepts_config: List of ConceptDef to use.  Defaults to all 25.

        Returns:
            List of (anomalous_traj, concept_id, primitive_labels) tuples.
        """
        if concepts_config is None:
            concepts_config = CONCEPT_DEFS

        results: List[Tuple[SemanticTrajectory, int, List[List[int]]]] = []
        n_pool = len(normal_trajectories)

        if n_pool == 0:
            return results

        for cdef in concepts_config:
            # Sample ~5 % of pool per concept, at least 1
            n_sample = max(1, n_pool // 20)
            indices = self.engine.rng.choice(n_pool, size=n_sample, replace=False)
            for idx in indices:
                traj = normal_trajectories[idx]
                if len(traj.episodes) < 2:
                    continue
                try:
                    anom_traj, cid, prim = self.generate_concept(traj, cdef)
                    results.append((anom_traj, cid, prim))
                except Exception:
                    # Skip trajectories that are too short / incompatible
                    continue

        return results
