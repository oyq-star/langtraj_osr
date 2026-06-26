"""B4: Canonical-JSON baseline — hand-crafted JSON definitions as fixed vectors.

Controls for information content: same information as language definitions,
but expressed as structured JSON mappings from concepts to primitive weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.episode_encoder import EpisodeEncoder
from ..models.trajectory_encoder import TrajectoryEncoder
from ..models.user_history import UserHistoryModule


# Default canonical schema: all 25 MobDef-Bench concepts mapped to primitive weights.
# Primitive ordering (10 dims) matches core/concepts.py:
#   0:unusual_time, 1:unusual_zone, 2:unusual_poi, 3:long_dwell, 4:short_dwell,
#   5:unusual_transition, 6:high_trip_deviation, 7:event_co_occurrence,
#   8:companion_absence, 9:companion_anomaly
_DEFAULT_CANONICAL_SCHEMA: Dict[str, Dict[str, float]] = {
    # A_seen (12)
    "late_night_industrial":              {"unusual_time":0.9,"unusual_zone":0.8,"unusual_poi":0.3,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    "excessive_dwell_commercial":         {"unusual_time":0.0,"unusual_zone":0.0,"unusual_poi":0.9,"long_dwell":0.9,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    "rapid_cross_city_jump":              {"unusual_time":0.0,"unusual_zone":0.0,"unusual_poi":0.0,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.9,"high_trip_deviation":0.9,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    "weekend_office_district":            {"unusual_time":0.9,"unusual_zone":0.7,"unusual_poi":0.0,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    "fleeting_hospital_visit":            {"unusual_time":0.0,"unusual_zone":0.0,"unusual_poi":0.9,"long_dwell":0.0,"short_dwell":0.9,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    "midnight_park_loiter":               {"unusual_time":0.9,"unusual_zone":0.0,"unusual_poi":0.0,"long_dwell":0.9,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    "solo_nightclub_deviation":           {"unusual_time":0.8,"unusual_zone":0.0,"unusual_poi":0.8,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.0,"companion_absence":0.9,"companion_anomaly":0.0},
    "event_zone_avoidance":               {"unusual_time":0.0,"unusual_zone":0.8,"unusual_poi":0.0,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.9,"companion_absence":0.0,"companion_anomaly":0.0},
    "transit_mode_switch_loop":           {"unusual_time":0.0,"unusual_zone":0.0,"unusual_poi":0.0,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.9,"high_trip_deviation":0.8,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    "repeated_short_visits_different_zones": {"unusual_time":0.0,"unusual_zone":0.8,"unusual_poi":0.0,"long_dwell":0.0,"short_dwell":0.9,"unusual_transition":0.0,"high_trip_deviation":0.7,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    "companion_switch_at_sensitive_poi":  {"unusual_time":0.0,"unusual_zone":0.0,"unusual_poi":0.9,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.9},
    "long_trip_to_airport_no_history":    {"unusual_time":0.0,"unusual_zone":0.8,"unusual_poi":0.8,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.9,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    # A_zs_comp (6)
    "midnight_commercial_with_event":     {"unusual_time":0.9,"unusual_zone":0.0,"unusual_poi":0.8,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.9,"companion_absence":0.0,"companion_anomaly":0.0},
    "solo_long_dwell_unfamiliar_residential": {"unusual_time":0.0,"unusual_zone":0.9,"unusual_poi":0.0,"long_dwell":0.9,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.0,"companion_absence":0.9,"companion_anomaly":0.0},
    "rapid_mode_switch_with_companion_change": {"unusual_time":0.0,"unusual_zone":0.0,"unusual_poi":0.0,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.9,"high_trip_deviation":0.0,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.9},
    "short_dwell_chain_with_long_trip":   {"unusual_time":0.0,"unusual_zone":0.0,"unusual_poi":0.0,"long_dwell":0.0,"short_dwell":0.9,"unusual_transition":0.0,"high_trip_deviation":0.9,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    "unusual_time_event_zone_alone":      {"unusual_time":0.9,"unusual_zone":0.0,"unusual_poi":0.0,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.9,"companion_absence":0.9,"companion_anomaly":0.0},
    "novel_poi_with_dwell_and_transition_anomaly": {"unusual_time":0.0,"unusual_zone":0.0,"unusual_poi":0.9,"long_dwell":0.9,"short_dwell":0.0,"unusual_transition":0.9,"high_trip_deviation":0.0,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    # A_zs_family (4)
    "systematic_boundary_probing":        {"unusual_time":0.0,"unusual_zone":0.9,"unusual_poi":0.0,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.9,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    "spatial_anchor_shift":               {"unusual_time":0.0,"unusual_zone":0.9,"unusual_poi":0.0,"long_dwell":0.9,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    "oscillating_zone_revisit":           {"unusual_time":0.0,"unusual_zone":0.9,"unusual_poi":0.0,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.9,"high_trip_deviation":0.9,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    "coverage_maximisation_pattern":      {"unusual_time":0.0,"unusual_zone":0.9,"unusual_poi":0.0,"long_dwell":0.0,"short_dwell":0.9,"unusual_transition":0.0,"high_trip_deviation":0.9,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    # A_unknown (3) — these have no text definition; all-zero canonical representation
    "unknown_anomaly_A":                  {"unusual_time":0.0,"unusual_zone":0.0,"unusual_poi":0.0,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    "unknown_anomaly_B":                  {"unusual_time":0.0,"unusual_zone":0.0,"unusual_poi":0.0,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
    "unknown_anomaly_C":                  {"unusual_time":0.0,"unusual_zone":0.0,"unusual_poi":0.0,"long_dwell":0.0,"short_dwell":0.0,"unusual_transition":0.0,"high_trip_deviation":0.0,"event_co_occurrence":0.0,"companion_absence":0.0,"companion_anomaly":0.0},
}

# Canonical primitive names — must match core/concepts.py primitive index legend:
#   0:unusual_time, 1:unusual_zone, 2:unusual_poi, 3:long_dwell, 4:short_dwell,
#   5:unusual_transition, 6:high_trip_deviation, 7:event_co_occurrence,
#   8:companion_absence, 9:companion_anomaly
PRIMITIVE_NAMES: List[str] = [
    "unusual_time", "unusual_zone", "unusual_poi", "long_dwell", "short_dwell",
    "unusual_transition", "high_trip_deviation", "event_co_occurrence",
    "companion_absence", "companion_anomaly",
]
N_PRIMITIVES = len(PRIMITIVE_NAMES)


class CanonicalJSONEncoder(nn.Module):
    """Encode hand-crafted JSON concept definitions into d_model embeddings.

    Loads a JSON file (or dict) mapping concept names to primitive weight dicts,
    converts them to fixed vectors, and projects to d_model.

    Parameters
    ----------
    d_model : int
        Output dimensionality.
    json_path : str | Path | None
        Path to JSON file with definitions. If None, uses default schema.
    definitions : dict | None
        Direct dict of definitions (alternative to json_path).
    """

    def __init__(
        self,
        d_model: int = 256,
        json_path: Optional[Union[str, Path]] = None,
        definitions: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        if json_path is not None:
            with open(json_path, "r") as f:
                definitions = json.load(f)
        if definitions is None:
            definitions = _DEFAULT_CANONICAL_SCHEMA

        # Convert definitions to fixed weight matrix
        concept_names = sorted(definitions.keys())
        self.concept_names = concept_names
        n_concepts = len(concept_names)

        weight_matrix = torch.zeros(n_concepts, N_PRIMITIVES)
        for i, name in enumerate(concept_names):
            prim_dict = definitions[name]
            for j, prim_name in enumerate(PRIMITIVE_NAMES):
                weight_matrix[i, j] = prim_dict.get(prim_name, 0.0)

        self.register_buffer("weight_matrix", weight_matrix)  # (K, N_PRIM)

        # Projection from primitive space to d_model
        self.proj = nn.Sequential(
            nn.Linear(N_PRIMITIVES, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(self) -> torch.Tensor:
        """Encode all canonical definitions.

        Returns
        -------
        Tensor (K, d_model) — one embedding per concept.
        """
        return self.proj(self.weight_matrix)  # (K, d_model)

    @property
    def n_concepts(self) -> int:
        return self.weight_matrix.shape[0]


class CanonicalJSONModel(nn.Module):
    """Baseline B4: trajectory scoring with canonical JSON definitions.

    Same backbone (EpisodeEncoder + TrajectoryEncoder) but definitions
    come from hand-crafted JSON rather than language or DSL.

    Parameters
    ----------
    d_model : int
        Hidden dimensionality.
    json_path : str | Path | None
        Path to JSON definitions file.
    definitions : dict | None
        Direct definition dict.
    """

    def __init__(
        self,
        poi_vocab_size: int = 64,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        n_prototypes: int = 8,
        dropout: float = 0.1,
        json_path: Optional[Union[str, Path]] = None,
        definitions: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.episode_encoder = EpisodeEncoder(
            poi_vocab_size=poi_vocab_size,
            hidden_dim=d_model,
        )
        self.trajectory_encoder = TrajectoryEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.user_history = UserHistoryModule(
            d_model=d_model,
            n_prototypes=n_prototypes,
        )
        self.json_encoder = CanonicalJSONEncoder(
            d_model=d_model,
            json_path=json_path,
            definitions=definitions,
        )

        # Bilinear alignment
        self.W = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

        # Primitive head for attribute alignment
        self.primitive_head = nn.Linear(d_model, N_PRIMITIVES)

    def encode_trajectory(
        self,
        episodes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ep_emb = self.episode_encoder(episodes)
        return self.trajectory_encoder(ep_emb, mask=mask)

    def forward(
        self,
        episodes: Dict[str, torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z_x, h_i = self.encode_trajectory(episodes, mask=mask)
        E_norm, dev_feats = self.user_history(z_x, user_prototypes)

        # Canonical definition embeddings
        c_json = self.json_encoder()  # (K, D)

        # Bilinear global score
        z_proj = z_x @ self.W                     # (B, D)
        global_score = z_proj @ c_json.T           # (B, K)

        # Attribute alignment via primitive head vs canonical weights
        v_x = self.primitive_head(z_x)                             # (B, N_PRIM)
        attr_score = v_x @ self.json_encoder.weight_matrix.T      # (B, K)

        definition_scores = global_score + 0.5 * attr_score

        return {
            "E_norm": E_norm,
            "deviation_features": dev_feats,
            "definition_scores": definition_scores,
            "z_x": z_x,
        }

    @torch.no_grad()
    def predict(
        self,
        episodes: Dict[str, torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        out = self.forward(episodes, user_prototypes, mask=mask)
        return out["definition_scores"], out["E_norm"]

    def compute_loss(
        self,
        episodes: Dict[str, torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.forward(episodes, user_prototypes, mask=mask)
        def_scores = out["definition_scores"]
        E_norm = out["E_norm"]

        n_cls = def_scores.shape[1]
        known_mask = (labels > 0) & (labels <= n_cls)
        if known_mask.any():
            cls_loss = F.cross_entropy(def_scores[known_mask], (labels[known_mask] - 1).clamp(0, n_cls - 1))
        else:
            cls_loss = torch.tensor(0.0, device=labels.device)

        is_normal = (labels == 0).float()
        is_anomaly = (labels > 0).float()
        margin = 10.0
        norm_loss = (
            (E_norm * is_normal).sum() / is_normal.sum().clamp(min=1)
            + (F.relu(margin - E_norm) * is_anomaly).sum() / is_anomaly.sum().clamp(min=1)
        )

        return cls_loss + norm_loss
