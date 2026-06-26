"""B3: NL->DSL baseline — parse natural language definitions into DSL slots, then score.

Uses a rule-based parser to extract structured slot values from text definitions,
then delegates scoring to the DSL-XL model.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .dsl_xl import DSL_SLOT_NAMES, DSLXLModel, NUM_DSL_SLOTS


class NL2DSLParser:
    """Rule-based parser that extracts DSL slot values from natural language definitions.

    Each slot is extracted via keyword/regex heuristics. Missing slots default to 0.
    """

    # --- time range keywords ---
    _TIME_KEYWORDS: Dict[str, int] = {
        "morning": 1, "dawn": 1, "early": 1,
        "midday": 2, "noon": 2, "lunch": 2,
        "afternoon": 3,
        "evening": 4, "dusk": 4, "dinner": 4,
        "night": 5, "late night": 6, "midnight": 6,
        "overnight": 7,
        "weekend": 8, "weekday": 9,
        "rush hour": 10, "commute": 10,
    }

    # --- POI role keywords ---
    _POI_KEYWORDS: Dict[str, int] = {
        "home": 1, "residence": 1, "house": 1,
        "work": 2, "office": 2, "workplace": 2,
        "gym": 3, "fitness": 3, "exercise": 3,
        "restaurant": 4, "dining": 4, "food": 4, "eat": 4,
        "bar": 5, "pub": 5, "nightclub": 5, "club": 5,
        "hospital": 6, "medical": 6, "clinic": 6, "doctor": 6,
        "school": 7, "university": 7, "campus": 7, "education": 7,
        "park": 8, "garden": 8, "outdoor": 8,
        "mall": 9, "shopping": 9, "store": 9, "shop": 9,
        "airport": 10, "station": 11, "transit": 11,
        "hotel": 12, "accommodation": 12,
        "church": 13, "temple": 13, "mosque": 13, "religious": 13,
    }

    # --- transition keywords ---
    _TRANSITION_KEYWORDS: Dict[str, int] = {
        "walk": 0, "walking": 0, "foot": 0, "pedestrian": 0,
        "drive": 1, "driving": 1, "car": 1, "taxi": 1, "uber": 1,
        "transit": 2, "bus": 2, "subway": 2, "metro": 2, "train": 2,
        "jump": 3, "teleport": 3, "sudden": 3,
    }

    def parse(self, text: str) -> List[int]:
        """Parse a natural language definition into 12 DSL slot values.

        Parameters
        ----------
        text : str
            Free-form text definition (e.g., "Late night visits to bars
            without companions, arriving by taxi").

        Returns
        -------
        list[int]
            12 integer slot values corresponding to DSL_SLOT_NAMES.
        """
        text_lower = text.lower()
        slots = [0] * NUM_DSL_SLOTS

        # Slot 0: target_time_range
        for keyword, val in self._TIME_KEYWORDS.items():
            if keyword in text_lower:
                slots[0] = val
                break

        # Slot 1: target_poi_roles
        for keyword, val in self._POI_KEYWORDS.items():
            if keyword in text_lower:
                slots[1] = val
                break

        # Slot 2: forbidden_poi_roles — look for "not at", "avoid", "never"
        forbidden_match = re.search(
            r"(?:not at|avoid|never|without visiting|skip)\s+(\w+)", text_lower
        )
        if forbidden_match:
            word = forbidden_match.group(1)
            slots[2] = self._POI_KEYWORDS.get(word, 0)

        # Slot 3: min_dwell — look for minimum duration mentions
        min_dwell_match = re.search(
            r"(?:at least|minimum|min|more than)\s+(\d+)\s*(?:min|hour|h)", text_lower
        )
        if min_dwell_match:
            val = int(min_dwell_match.group(1))
            if "hour" in min_dwell_match.group(0) or "h" in min_dwell_match.group(0):
                val *= 60
            slots[3] = min(val // 10, 15)  # bin into 0-15

        # Slot 4: max_dwell
        max_dwell_match = re.search(
            r"(?:at most|maximum|max|less than|no more than)\s+(\d+)\s*(?:min|hour|h)",
            text_lower,
        )
        if max_dwell_match:
            val = int(max_dwell_match.group(1))
            if "hour" in max_dwell_match.group(0) or "h" in max_dwell_match.group(0):
                val *= 60
            slots[4] = min(val // 10, 15)

        # Slot 5: expected_transitions
        for keyword, val in self._TRANSITION_KEYWORDS.items():
            if keyword in text_lower:
                slots[5] = val
                break

        # Slot 6: forbidden_transitions
        forbidden_trans = re.search(
            r"(?:not by|never by|without|no)\s+(\w+)", text_lower
        )
        if forbidden_trans:
            word = forbidden_trans.group(1)
            slots[6] = self._TRANSITION_KEYWORDS.get(word, 0)

        # Slot 7: spatial_zone_type
        if any(w in text_lower for w in ["urban", "city", "downtown"]):
            slots[7] = 1
        elif any(w in text_lower for w in ["suburban", "residential"]):
            slots[7] = 2
        elif any(w in text_lower for w in ["rural", "countryside"]):
            slots[7] = 3
        elif any(w in text_lower for w in ["industrial", "factory"]):
            slots[7] = 4

        # Slot 8: temporal_regularity
        if any(w in text_lower for w in ["regular", "routine", "habitual", "daily"]):
            slots[8] = 1
        elif any(w in text_lower for w in ["irregular", "unusual", "sporadic", "rare"]):
            slots[8] = 2
        elif any(w in text_lower for w in ["periodic", "weekly", "monthly"]):
            slots[8] = 3

        # Slot 9: event_context
        if any(w in text_lower for w in ["event", "concert", "festival", "game", "match"]):
            slots[9] = 1
        elif any(w in text_lower for w in ["holiday", "vacation"]):
            slots[9] = 2
        elif any(w in text_lower for w in ["emergency", "urgent"]):
            slots[9] = 3

        # Slot 10: companion_required
        if any(w in text_lower for w in ["with companion", "with friend", "together", "group"]):
            slots[10] = 1
        elif any(w in text_lower for w in ["alone", "solo", "without companion", "unaccompanied"]):
            slots[10] = 0

        # Slot 11: negation_flag
        if any(w in text_lower for w in ["not", "never", "without", "no ", "absence", "lack"]):
            slots[11] = 1

        return slots

    def parse_batch(self, texts: List[str]) -> torch.Tensor:
        """Parse multiple definitions into a slot tensor.

        Returns
        -------
        Tensor (N, 12) — LongTensor of slot values.
        """
        return torch.tensor([self.parse(t) for t in texts], dtype=torch.long)


class NL2DSLModel(nn.Module):
    """Baseline B3: parse NL definitions to DSL, then score via DSL-XL.

    Wraps NL2DSLParser + DSLXLModel. At init/load time, definitions are
    parsed once; at forward time, the pre-parsed DSL slots are used.

    Parameters
    ----------
    definitions : list[str] | None
        Natural language definitions. Parsed to DSL at init.
    """

    def __init__(
        self,
        definitions: Optional[List[str]] = None,
        poi_vocab_size: int = 64,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        n_prototypes: int = 8,
        n_primitives: int = 10,
        lambda_attr: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.parser = NL2DSLParser()
        self.dsl_model = DSLXLModel(
            poi_vocab_size=poi_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            n_prototypes=n_prototypes,
            n_primitives=n_primitives,
            lambda_attr=lambda_attr,
            dropout=dropout,
        )

        # Pre-parse definitions if provided
        if definitions is not None:
            parsed = self.parser.parse_batch(definitions)
            self.register_buffer("dsl_slots", parsed)
        else:
            self.register_buffer("dsl_slots", torch.zeros(1, NUM_DSL_SLOTS, dtype=torch.long))

    def set_definitions(self, definitions: List[str]) -> None:
        """Parse and cache new definitions."""
        parsed = self.parser.parse_batch(definitions)
        self.dsl_slots = parsed.to(next(self.parameters()).device)

    def forward(
        self,
        episodes: Dict[str, torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using pre-parsed DSL slots."""
        return self.dsl_model.forward(
            episodes, self.dsl_slots, user_prototypes, mask=mask
        )

    @torch.no_grad()
    def predict(
        self,
        episodes: Dict[str, torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict definition scores and normality energy."""
        self.eval()
        return self.dsl_model.predict(
            episodes, self.dsl_slots, user_prototypes, mask=mask
        )

    def compute_loss(
        self,
        episodes: Dict[str, torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss delegating to DSL-XL model."""
        return self.dsl_model.compute_loss(
            episodes, self.dsl_slots, user_prototypes, labels, mask=mask
        )
