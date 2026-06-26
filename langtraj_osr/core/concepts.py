"""Anomaly concept definitions for the MobDef-Bench benchmark.

25 urban mobility anomaly concepts organised into four evaluation splits:

* **A_seen** (12 concepts) -- used during training; the model observes both
  the natural-language definition and labelled trajectories.
* **A_zs_comp** (6 concepts) -- zero-shot compositional; novel conjunctions
  of primitives that were individually seen during training.
* **A_zs_family** (4 concepts) -- zero-shot family; an entire operator
  family (spatial deviation) is held out from training.
* **A_unknown** (3 concepts) -- no definition, no training examples; the
  system must flag these as *unknown* anomalies at test time.

Primitive index legend (10-dim binary vector)
---------------------------------------------
0 : unusual_time          -- visit at an atypical hour for the user
1 : unusual_zone          -- visit to a spatially atypical zone
2 : unusual_poi           -- visit to a novel POI role category
3 : long_dwell            -- abnormally long stay duration
4 : short_dwell           -- abnormally short stay duration
5 : unusual_transition    -- rare or impossible transport mode
6 : high_trip_deviation   -- trip segment much longer than user average
7 : event_co_occurrence   -- special event at the location
8 : companion_absence     -- user is alone when usually accompanied
9 : companion_anomaly     -- accompanied by unusual companion pattern
"""

from __future__ import annotations

from typing import Any, Dict, List

# ============================================================================
# A_seen -- 12 concepts visible during training
# ============================================================================

_SEEN: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "late_night_industrial",
        "primitives": [0, 2],
        "canonical": (
            "A residential user visiting industrial zones between midnight and 5 AM."
        ),
        "paraphrases": [
            "Residential user going to factory areas in the small hours of the morning.",
            "A normally home-bound person found in an industrial district after midnight.",
            "Night-time presence in manufacturing zones by someone who usually stays in residential areas.",
            "User whose home is in a residential neighbourhood appears in an industrial area between 00:00 and 05:00.",
        ],
        "adversarial_paraphrases": [
            "A factory shift-worker commuting to their regular night shift.",
            "Industrial zone visit during daylight hours by a residential user.",
            "A resident walking past an industrial area at 7 PM.",
        ],
    },
    {
        "id": 2,
        "name": "excessive_dwell_commercial",
        "primitives": [1, 2, 3, 6],
        "canonical": (
            "User spending more than 6 hours at a commercial venue they have never visited before."
        ),
        "paraphrases": [
            "Unusually long stay (6 h+) at a brand-new commercial location.",
            "First-time commercial venue visit lasting an abnormally long time.",
            "Spending an entire workday at an unfamiliar shopping or retail location.",
            "A previously unvisited commercial establishment where the user remains for over six hours.",
        ],
        "adversarial_paraphrases": [
            "Spending 30 minutes shopping at a new store.",
            "Six-hour visit to a commercial venue the user frequents weekly.",
            "Extended dwell at the user's own workplace, which is a commercial building.",
        ],
    },
    {
        "id": 3,
        "name": "rapid_cross_city_jump",
        "primitives": [1, 5, 6],
        "canonical": (
            "Instantaneous teleportation-like jump across the city without plausible transit time."
        ),
        "paraphrases": [
            "Two consecutive check-ins on opposite sides of the city with no feasible travel time.",
            "GPS trace showing the user jumping 30 km in under 2 minutes.",
            "Impossible spatial displacement: user appears far away with zero transit gap.",
            "Back-to-back records at distant locations, implying faster-than-possible travel.",
        ],
        "adversarial_paraphrases": [
            "A regular subway commute across the city taking 40 minutes.",
            "GPS jitter causing a 50-metre jump between consecutive points.",
            "Driving on a highway at 120 km/h for 20 minutes.",
        ],
    },
    {
        "id": 4,
        "name": "weekend_office_district",
        "primitives": [0, 2],
        "canonical": (
            "User visiting a central business district on a weekend when they normally only go on weekdays."
        ),
        "paraphrases": [
            "Weekend appearance in an office area the user only visits Monday-to-Friday.",
            "Saturday/Sunday trip to the CBD by someone who exclusively works there on weekdays.",
            "A strictly weekday-office worker found in their office district over the weekend.",
            "User breaks their weekday-only CBD pattern by visiting on a Saturday or Sunday.",
        ],
        "adversarial_paraphrases": [
            "A weekend retail worker going to their regular Saturday shift in the CBD.",
            "Visiting a restaurant in the CBD on a Friday evening.",
            "User who goes to the CBD every day, including weekends.",
        ],
    },
    {
        "id": 5,
        "name": "fleeting_hospital_visit",
        "primitives": [2, 4],
        "canonical": (
            "Extremely brief visit (under 3 minutes) to a hospital by a user with no medical POI history."
        ),
        "paraphrases": [
            "A user with no prior hospital visits checks into a medical facility for less than 3 minutes.",
            "Flash visit to a hospital: in and out in under three minutes, no prior medical POI pattern.",
            "Under-3-minute hospital stop by someone who never visits healthcare locations.",
            "User without medical-POI history pings at a hospital for a suspiciously short duration.",
        ],
        "adversarial_paraphrases": [
            "A nurse arriving at the hospital for a 12-hour shift.",
            "Dropping someone off at the hospital entrance (expected brief stop).",
            "A 2-hour outpatient appointment at a clinic the user visits monthly.",
        ],
    },
    {
        "id": 6,
        "name": "midnight_park_loiter",
        "primitives": [0, 2, 3],
        "canonical": (
            "User staying in a public park for over 2 hours between 11 PM and 5 AM."
        ),
        "paraphrases": [
            "Late-night loitering in a park: remaining for 2+ hours after 11 PM.",
            "Extended after-midnight presence in a public green space.",
            "A person spending the small hours of the night sitting in a park.",
            "Park visit lasting more than two hours during the 23:00-05:00 window.",
        ],
        "adversarial_paraphrases": [
            "An evening jog through the park ending at 10 PM.",
            "A daytime picnic lasting 3 hours on a Sunday afternoon.",
            "Passing through a park on a midnight walk home (5-minute transit).",
        ],
    },
    {
        "id": 7,
        "name": "solo_nightclub_deviation",
        "primitives": [0, 2, 8],
        "canonical": (
            "User who normally visits nightlife venues with companions arrives alone at a nightclub at an unusual hour."
        ),
        "paraphrases": [
            "An always-accompanied nightclub-goer shows up solo at an off-peak hour.",
            "Lone nightclub visit by a user who historically only goes with friends, at an atypical time.",
            "Unaccompanied and off-schedule visit to a nightlife venue.",
            "User breaks both their companion and timing patterns at a nightclub.",
        ],
        "adversarial_paraphrases": [
            "Going to a nightclub with friends on a typical Friday night.",
            "User who always goes alone visiting a club at their usual time.",
            "Solo dinner at a restaurant near a nightclub district.",
        ],
    },
    {
        "id": 8,
        "name": "event_zone_avoidance",
        "primitives": [1, 2, 6, 7],
        "canonical": (
            "User deliberately detouring around an area hosting a major event they would normally attend."
        ),
        "paraphrases": [
            "Spatial avoidance of a zone with an active event the user usually participates in.",
            "User reroutes to bypass a major event venue they have historically visited.",
            "Detour pattern around an event zone the user typically frequents.",
            "Anomalous route that avoids a major-event area the user would ordinarily visit.",
        ],
        "adversarial_paraphrases": [
            "Avoiding a road closure due to construction.",
            "Taking a different route because of daily traffic patterns.",
            "Skipping an event the user has never attended before.",
        ],
    },
    {
        "id": 9,
        "name": "transit_mode_switch_loop",
        "primitives": [1, 5, 6],
        "canonical": (
            "Rapid alternation between walking and driving within a single trip, creating a loop pattern."
        ),
        "paraphrases": [
            "User switches between walk and drive modes multiple times in quick succession, tracing a loop.",
            "Oscillating transport mode (walk-drive-walk-drive) forming a closed geographic loop.",
            "Suspicious back-and-forth mode switching along a looping trajectory.",
            "Trip segment with repeated walk/drive transitions that returns to its starting point.",
        ],
        "adversarial_paraphrases": [
            "Walking to a parked car and driving away (single mode switch).",
            "A delivery driver making multiple stops along a route.",
            "Switching from bus to walking for the last mile of a commute.",
        ],
    },
    {
        "id": 10,
        "name": "repeated_short_visits_different_zones",
        "primitives": [1, 4, 6],
        "canonical": (
            "Sequence of very short visits (under 5 min each) to 4+ distinct zones within one hour."
        ),
        "paraphrases": [
            "Four or more sub-5-minute stops in different spatial zones in a single hour.",
            "Rapid zone hopping: brief presence in many distinct areas in quick succession.",
            "Under-5-minute visits to multiple scattered zones, all within a 60-minute window.",
            "High-frequency, low-dwell multi-zone traversal in a compressed timeframe.",
        ],
        "adversarial_paraphrases": [
            "A postal worker delivering mail along their regular route.",
            "Driving through several neighbourhoods on a highway without stopping.",
            "Visiting three shops in the same mall over an hour.",
        ],
    },
    {
        "id": 11,
        "name": "companion_switch_at_sensitive_poi",
        "primitives": [2, 9],
        "canonical": (
            "User arrives at a government or financial POI with an unusual companion pattern "
            "(new co-located device)."
        ),
        "paraphrases": [
            "Visiting a bank or government office while co-located with a previously unseen device.",
            "Anomalous companion signature at a sensitive POI (e.g. embassy, courthouse).",
            "User's companion set changes right when entering a financial institution.",
            "New co-travelling device appears exactly at a government/financial venue visit.",
        ],
        "adversarial_paraphrases": [
            "Going to the bank with a long-time friend.",
            "Visiting a government office alone, as usual.",
            "Meeting a new colleague at a coffee shop.",
        ],
    },
    {
        "id": 12,
        "name": "long_trip_to_airport_no_history",
        "primitives": [1, 2, 6],
        "canonical": (
            "User with no prior airport visits makes a long-distance trip to an airport zone."
        ),
        "paraphrases": [
            "First-ever airport visit preceded by a trip segment far longer than the user's average.",
            "Unprecedented airport zone visit after an unusually long drive.",
            "Long-haul trip to an airport by someone who has never visited one before.",
            "User with zero airport-POI history suddenly travels a great distance to the airport.",
        ],
        "adversarial_paraphrases": [
            "A frequent flyer driving to the airport for their weekly business trip.",
            "Living near the airport and passing through the zone daily.",
            "Taking a taxi to a train station 5 km away.",
        ],
    },
]

# ============================================================================
# A_zs_comp -- 6 concepts composed from seen primitives, but novel conjunctions
# ============================================================================

_ZS_COMP: List[Dict[str, Any]] = [
    {
        "id": 13,
        "name": "midnight_commercial_with_event",
        "primitives": [0, 1, 2, 7],
        "canonical": (
            "User visiting a commercial venue after midnight while a major event is underway in the same zone."
        ),
        "paraphrases": [
            "Post-midnight commercial-area visit coinciding with a live event in the vicinity.",
            "Late-night shopping-district presence during an ongoing special event.",
            "User found at a commercial POI past midnight, with an active event flag in the zone.",
            "Concurrent midnight commercial visit and zone-level event occurrence.",
        ],
        "adversarial_paraphrases": [
            "Attending a late-night concert at a dedicated music venue.",
            "Shopping at a 24-hour store on a quiet Tuesday night.",
            "Visiting a commercial area during a daytime festival.",
        ],
    },
    {
        "id": 14,
        "name": "solo_long_dwell_unfamiliar_residential",
        "primitives": [1, 3, 6, 8],
        "canonical": (
            "User staying alone for an extended period in an unfamiliar residential zone."
        ),
        "paraphrases": [
            "Prolonged solo visit to a residential area the user has never been to before.",
            "Unaccompanied multi-hour stay in an unknown residential neighbourhood.",
            "User without companions lingers in a new residential zone for an unusually long time.",
            "Extended dwell in a previously unvisited residential area while alone.",
        ],
        "adversarial_paraphrases": [
            "Staying at a friend's house overnight (regularly visited address).",
            "Visiting a new residential area briefly while accompanied by a friend.",
            "Spending time at home alone on a typical evening.",
        ],
    },
    {
        "id": 15,
        "name": "rapid_mode_switch_with_companion_change",
        "primitives": [5, 9],
        "canonical": (
            "User rapidly alternates transport modes while their companion signature changes mid-trip."
        ),
        "paraphrases": [
            "Transport-mode oscillation paired with a companion-set change during the same trip.",
            "Switching between walk and drive while simultaneously gaining or losing a co-traveller.",
            "Mid-trip companion swap coinciding with a transport-mode change.",
            "Unusual joint change in both transit mode and companion pattern.",
        ],
        "adversarial_paraphrases": [
            "A friend drops the user off and the user walks the rest of the way.",
            "Switching from bus to walking when a colleague gets off at a different stop.",
            "Carpooling with different coworkers on alternating days.",
        ],
    },
    {
        "id": 16,
        "name": "short_dwell_chain_with_long_trip",
        "primitives": [1, 4, 6],
        "canonical": (
            "Series of very short stops connected by trip segments much longer than the user's average."
        ),
        "paraphrases": [
            "Chain of brief visits linked by unusually long travel segments.",
            "Multiple sub-5-minute stops separated by high-distance trips.",
            "Short-dwell, long-trip pattern: quick stops punctuated by above-average travel distances.",
            "User makes several fleeting visits with disproportionately long drives between them.",
        ],
        "adversarial_paraphrases": [
            "A truck driver making scheduled delivery stops along a highway.",
            "A commuter with one transfer making two short waits.",
            "Driving a normal distance between two 30-minute errands.",
        ],
    },
    {
        "id": 17,
        "name": "unusual_time_event_zone_alone",
        "primitives": [0, 1, 2, 7, 8],
        "canonical": (
            "User alone at an event zone during an unusual hour for them, while the event is active."
        ),
        "paraphrases": [
            "Solo presence at an active event venue at an atypical personal hour.",
            "Unaccompanied user at an event-flagged zone outside their normal time pattern.",
            "Being alone at a live-event area at a time the user doesn't usually go out.",
            "Unusual-hour, solo visit coinciding with an active event in the zone.",
        ],
        "adversarial_paraphrases": [
            "Attending a concert alone at the usual showtime.",
            "Being at an event zone during off-hours when no event is running.",
            "Going to a festival with friends at a normal time.",
        ],
    },
    {
        "id": 18,
        "name": "novel_poi_with_dwell_and_transition_anomaly",
        "primitives": [2, 3, 5],
        "canonical": (
            "First-time visit to a new POI type with both an abnormally long stay and an unusual transit mode."
        ),
        "paraphrases": [
            "User visits an entirely new POI category, stays far too long, and arrives by an unusual mode.",
            "Novel POI visit combining excessive dwell and atypical transport.",
            "Triple anomaly: new venue type, long dwell, and rare transition mode.",
            "Unprecedented POI category visit with jointly unusual duration and travel mode.",
        ],
        "adversarial_paraphrases": [
            "Visiting a new restaurant for a normal-length dinner, arriving by car.",
            "A long stay at a familiar POI type reached by an unusual bus route.",
            "Brief first visit to a new gym, arriving on foot.",
        ],
    },
]

# ============================================================================
# A_zs_family -- 4 concepts from a held-out operator family (spatial deviation)
# ============================================================================

_ZS_FAMILY: List[Dict[str, Any]] = [
    {
        "id": 19,
        "name": "systematic_boundary_probing",
        "primitives": [1, 6],
        "canonical": (
            "User repeatedly visits the edges of their usual activity space, "
            "probing zones just beyond their normal boundary."
        ),
        "paraphrases": [
            "Systematic exploration of the periphery of the user's habitual mobility region.",
            "Repeated trips to zones that lie just outside the user's historical convex hull.",
            "Boundary-testing pattern: visits cluster at the fringes of the user's activity space.",
            "User incrementally extends their spatial range by visiting adjacent unexplored zones.",
        ],
        "adversarial_paraphrases": [
            "Gradually exploring a new neighbourhood after moving house.",
            "A runner varying their jogging route slightly each day.",
            "Visiting a friend in a nearby suburb once a month.",
        ],
    },
    {
        "id": 20,
        "name": "spatial_anchor_shift",
        "primitives": [1, 3, 6],
        "canonical": (
            "User's primary activity anchor (e.g. home/work centroid) abruptly shifts "
            "to a new zone with long dwell times."
        ),
        "paraphrases": [
            "Sudden relocation of the user's home or work anchor to a new zone with extended stays.",
            "Abrupt change in the user's spatial centroid accompanied by long dwell at the new location.",
            "The user's primary base suddenly moves to an unfamiliar zone where they spend many hours.",
            "Overnight anchor shift: the user begins spending most time in a previously unvisited zone.",
        ],
        "adversarial_paraphrases": [
            "Moving to a new apartment (legitimate relocation).",
            "A business trip with hotel stays in a different city.",
            "Working from a different office branch for a scheduled rotation.",
        ],
    },
    {
        "id": 21,
        "name": "oscillating_zone_revisit",
        "primitives": [1, 5, 6],
        "canonical": (
            "User oscillates between two distant zones multiple times in a single day, "
            "using different transport modes each time."
        ),
        "paraphrases": [
            "Repeated back-and-forth between two far-apart zones with varying transit modes.",
            "Same-day ping-pong pattern between distant locations via different transport.",
            "Multi-round-trip oscillation between two zones, each leg using a new mode.",
            "User bounces between two distant areas several times, switching how they travel.",
        ],
        "adversarial_paraphrases": [
            "Commuting between home and work once each way.",
            "Driving to a store and back once in the afternoon.",
            "Taking the same bus route to and from school.",
        ],
    },
    {
        "id": 22,
        "name": "coverage_maximisation_pattern",
        "primitives": [1, 4, 6],
        "canonical": (
            "User visits an unusually large number of distinct zones in a single day "
            "with minimal dwell at each, maximising spatial coverage."
        ),
        "paraphrases": [
            "Spatial-coverage maximisation: many unique zones visited briefly in one day.",
            "Sweep pattern covering a high fraction of the city with minimal stops.",
            "User systematically passes through many zones with very short stays each.",
            "High zone-count, low-dwell day suggesting deliberate area scanning.",
        ],
        "adversarial_paraphrases": [
            "A tourist visiting multiple sightseeing spots over a full day.",
            "A courier making deliveries across the city.",
            "Driving through several zones on a long commute without stopping.",
        ],
    },
]

# ============================================================================
# A_unknown -- 3 concepts with no definition and no training examples
# ============================================================================

_UNKNOWN: List[Dict[str, Any]] = [
    {
        "id": 23,
        "name": "unknown_anomaly_A",
        "primitives": [0, 1, 2, 3, 6],
        "canonical": "",
        "paraphrases": [],
        "adversarial_paraphrases": [],
    },
    {
        "id": 24,
        "name": "unknown_anomaly_B",
        "primitives": [1, 3, 6],
        "canonical": "",
        "paraphrases": [],
        "adversarial_paraphrases": [],
    },
    {
        "id": 25,
        "name": "unknown_anomaly_C",
        "primitives": [1, 3, 5, 6],
        "canonical": "",
        "paraphrases": [],
        "adversarial_paraphrases": [],
    },
]

# ============================================================================
# Public aggregate dictionary
# ============================================================================

ANOMALY_CONCEPTS: Dict[str, List[Dict[str, Any]]] = {
    "seen": _SEEN,
    "zs_comp": _ZS_COMP,
    "zs_family": _ZS_FAMILY,
    "unknown": _UNKNOWN,
}

# ---------------------------------------------------------------------------
# Convenience look-ups
# ---------------------------------------------------------------------------

CONCEPT_BY_ID: Dict[int, Dict[str, Any]] = {}
for _split_concepts in ANOMALY_CONCEPTS.values():
    for _c in _split_concepts:
        CONCEPT_BY_ID[_c["id"]] = _c

CONCEPT_NAMES: Dict[int, str] = {cid: c["name"] for cid, c in CONCEPT_BY_ID.items()}

PRIMITIVE_NAMES: List[str] = [
    "unusual_time",
    "unusual_zone",
    "unusual_poi",
    "long_dwell",
    "short_dwell",
    "unusual_transition",
    "high_trip_deviation",
    "event_co_occurrence",
    "companion_absence",
    "companion_anomaly",
]

NUM_PRIMITIVES: int = len(PRIMITIVE_NAMES)


def get_concept_ids_for_split(split: str) -> List[int]:
    """Return sorted concept ids belonging to *split* (seen | zs_comp | zs_family | unknown)."""
    if split not in ANOMALY_CONCEPTS:
        raise ValueError(
            f"Unknown split '{split}'. Choose from {list(ANOMALY_CONCEPTS.keys())}."
        )
    return sorted(c["id"] for c in ANOMALY_CONCEPTS[split])


def get_all_definitions(include_paraphrases: bool = False) -> Dict[int, List[str]]:
    """Return a mapping from concept id to its textual definitions.

    Parameters
    ----------
    include_paraphrases : bool
        When True, the list includes the canonical text followed by all
        paraphrases.  When False, only the canonical string is returned
        (still wrapped in a list for uniform API).
    """
    out: Dict[int, List[str]] = {}
    for cid, concept in CONCEPT_BY_ID.items():
        canonical = concept["canonical"]
        if not canonical:
            out[cid] = []
            continue
        if include_paraphrases:
            out[cid] = [canonical] + list(concept.get("paraphrases", []))
        else:
            out[cid] = [canonical]
    return out
