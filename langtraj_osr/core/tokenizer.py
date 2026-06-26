"""Convert raw trajectory DataFrames into SemanticTrajectory sequences.

Each public ``tokenize_*`` method accepts a pandas DataFrame whose schema
matches the corresponding dataset and returns a list of SemanticTrajectory
objects ready for downstream modelling.

Supported datasets
------------------
* **GeoLife** — GPS traces with latitude, longitude, timestamp columns.
* **Porto** — Taxi trajectories stored as polylines with trip metadata.
* **Foursquare** — Check-in records with venue category and user id.
* **NumoSim** — Synthetic agent-based mobility simulation exports.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .episode import SemanticEpisode, SemanticTrajectory


# ---------------------------------------------------------------------------
# Default dwell-time bin edges (in minutes).  16 bins covering short stops
# through overnight stays.
# ---------------------------------------------------------------------------
_DEFAULT_DWELL_EDGES: List[float] = [
    0, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 480, 720, 1440,
]

# ---------------------------------------------------------------------------
# Transition mode speed thresholds (m/s) used for GeoLife-style GPS data.
# ---------------------------------------------------------------------------
_WALK_SPEED_MAX: float = 2.0     # ~7 km/h
_DRIVE_SPEED_MIN: float = 10.0   # ~36 km/h
_TRANSIT_SPEED_MIN: float = 4.0  # between walk and drive


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in metres."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _infer_transition(speed_ms: float) -> int:
    """Heuristic transition-type label from average segment speed."""
    if speed_ms < _WALK_SPEED_MAX:
        return 0  # walk
    if speed_ms < _TRANSIT_SPEED_MIN:
        return 2  # transit (bus / tram speeds)
    if speed_ms < _DRIVE_SPEED_MIN:
        return 2  # still transit-ish
    return 1  # drive


def _stable_zone_id(lat: float, lon: float, resolution: float = 0.005) -> int:
    """Map (lat, lon) to a deterministic spatial zone id via grid hashing."""
    grid_lat = int(round(lat / resolution))
    grid_lon = int(round(lon / resolution))
    h = hashlib.md5(f"{grid_lat},{grid_lon}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


class TrajectoryTokenizer:
    """Converts raw mobility data into factorised SemanticTrajectory objects.

    Parameters
    ----------
    poi_vocab_size : int
        Number of urban role classes (default 64).
    time_bins : int
        Number of temporal bins (default 168 = 24 h * 7 dow).
    dwell_bins : int
        Number of dwell-duration buckets (default 16).
    role_mapping : dict | None
        Optional mapping from raw POI category strings to ``poi_role`` ints.
        When *None*, categories are hashed into ``[0, poi_vocab_size)``.
    """

    def __init__(
        self,
        poi_vocab_size: int = 64,
        time_bins: int = 168,
        dwell_bins: int = 16,
        role_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        self.poi_vocab_size = poi_vocab_size
        self.time_bins = time_bins
        self.dwell_bins = dwell_bins
        self.role_mapping = role_mapping or {}
        self._dwell_edges = np.asarray(
            _DEFAULT_DWELL_EDGES[: self.dwell_bins], dtype=np.float64
        )

    # ------------------------------------------------------------------
    # Public tokenisation entry points
    # ------------------------------------------------------------------

    def tokenize_geolife(self, raw_df: pd.DataFrame) -> List[SemanticTrajectory]:
        """Tokenize GeoLife GPS traces.

        Expected columns: ``user_id``, ``latitude``, ``longitude``,
        ``timestamp`` (parseable datetime), and optionally
        ``transport_mode``, ``poi_category``, ``label``.
        """
        raw_df = raw_df.copy()
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
        raw_df.sort_values(["user_id", "timestamp"], inplace=True)

        trajectories: List[SemanticTrajectory] = []

        for user_id, user_df in raw_df.groupby("user_id"):
            # Segment into trips using a 20-minute gap threshold.
            user_df = user_df.reset_index(drop=True)
            time_diffs = user_df["timestamp"].diff().dt.total_seconds().fillna(0)
            trip_ids = (time_diffs > 1200).cumsum()

            user_avg_dist = self._user_avg_segment_distance_gps(user_df)

            for trip_idx, trip_df in user_df.groupby(trip_ids):
                trip_df = trip_df.reset_index(drop=True)
                if len(trip_df) < 2:
                    continue

                episodes: List[SemanticEpisode] = []
                stay_points = self._detect_stay_points_gps(trip_df)

                for sp in stay_points:
                    lat, lon = sp["lat"], sp["lon"]
                    ts: pd.Timestamp = sp["arrival"]
                    dwell_min = sp["dwell_minutes"]

                    zone = _stable_zone_id(lat, lon)
                    poi_role = self._resolve_poi_role(
                        sp.get("poi_category", "")
                    )
                    time_bin = self._compute_time_bin(ts.hour, ts.dayofweek)
                    dwell_bin = self._discretize_dwell(dwell_min)

                    seg_dist = sp.get("approach_dist_m", 0.0)
                    trip_len_change = (
                        seg_dist / user_avg_dist if user_avg_dist > 0 else 1.0
                    )

                    transition = sp.get("transition", 0)
                    event_flag = int(sp.get("event_flag", 0))
                    companion_flag = int(sp.get("companion_flag", 0))

                    episodes.append(
                        SemanticEpisode(
                            zone_id=zone,
                            poi_role=poi_role,
                            time_bin=time_bin,
                            dwell_bin=dwell_bin,
                            transition_type=transition,
                            trip_length_change=round(trip_len_change, 4),
                            event_flag=event_flag,
                            companion_flag=companion_flag,
                        )
                    )

                label = int(trip_df["label"].iloc[0]) if "label" in trip_df.columns else 0

                trajectories.append(
                    SemanticTrajectory(
                        episodes=episodes,
                        user_id=str(user_id),
                        trip_id=f"{user_id}_{trip_idx}",
                        label=label,
                    )
                )

        return trajectories

    def tokenize_porto(self, raw_df: pd.DataFrame) -> List[SemanticTrajectory]:
        """Tokenize Porto taxi trajectories.

        Expected columns: ``TRIP_ID``, ``POLYLINE`` (JSON list of [lon, lat]),
        ``TIMESTAMP`` (Unix epoch), ``MISSING_DATA``, ``DAY_TYPE``,
        and optionally ``label``.
        """
        import json

        trajectories: List[SemanticTrajectory] = []

        for _, row in raw_df.iterrows():
            trip_id = str(row["TRIP_ID"])

            polyline = row["POLYLINE"]
            if isinstance(polyline, str):
                try:
                    polyline = json.loads(polyline)
                except json.JSONDecodeError:
                    continue
            if not polyline or len(polyline) < 2:
                continue

            if row.get("MISSING_DATA", False) is True:
                continue

            base_ts = int(row["TIMESTAMP"])
            step_s = 15  # Porto traces are sampled every 15 seconds.

            # Compute user-average segment distance for normalisation.
            dists: List[float] = []
            for i in range(1, len(polyline)):
                d = _haversine_m(
                    polyline[i - 1][1], polyline[i - 1][0],
                    polyline[i][1], polyline[i][0],
                )
                dists.append(d)
            avg_dist = float(np.mean(dists)) if dists else 1.0

            # Convert polyline coordinates into episodes.  We subsample
            # to stay points roughly every 5 pings (~75 s) to avoid
            # extremely long sequences.
            episodes: List[SemanticEpisode] = []
            subsample = max(1, len(polyline) // 20)

            for idx in range(0, len(polyline), subsample):
                lon, lat = polyline[idx]
                elapsed = idx * step_s
                ts = pd.Timestamp(base_ts + elapsed, unit="s")

                zone = _stable_zone_id(lat, lon)
                poi_role = self._resolve_poi_role("")  # no POI info in Porto
                time_bin = self._compute_time_bin(ts.hour, ts.dayofweek)
                dwell_bin = self._discretize_dwell(subsample * step_s / 60.0)

                if idx > 0:
                    prev_lon, prev_lat = polyline[idx - subsample] if idx - subsample >= 0 else polyline[0]
                    seg_dist = _haversine_m(prev_lat, prev_lon, lat, lon)
                    speed = seg_dist / (subsample * step_s) if subsample * step_s > 0 else 0
                    transition = _infer_transition(speed)
                    trip_len_change = seg_dist / avg_dist if avg_dist > 0 else 1.0
                else:
                    transition = 1  # taxi = drive
                    trip_len_change = 1.0

                episodes.append(
                    SemanticEpisode(
                        zone_id=zone,
                        poi_role=poi_role,
                        time_bin=time_bin,
                        dwell_bin=dwell_bin,
                        transition_type=transition,
                        trip_length_change=round(trip_len_change, 4),
                        event_flag=0,
                        companion_flag=0,
                    )
                )

            label = int(row["label"]) if "label" in row.index else 0

            trajectories.append(
                SemanticTrajectory(
                    episodes=episodes,
                    user_id=str(row.get("TAXI_ID", "unknown")),
                    trip_id=trip_id,
                    label=label,
                )
            )

        return trajectories

    def tokenize_foursquare(self, raw_df: pd.DataFrame) -> List[SemanticTrajectory]:
        """Tokenize Foursquare check-in data.

        Expected columns: ``user_id``, ``venue_id``, ``venue_category``,
        ``latitude``, ``longitude``, ``timestamp``, and optionally
        ``label``.
        """
        raw_df = raw_df.copy()
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
        raw_df.sort_values(["user_id", "timestamp"], inplace=True)

        trajectories: List[SemanticTrajectory] = []

        for user_id, user_df in raw_df.groupby("user_id"):
            user_df = user_df.reset_index(drop=True)

            # Compute user-average inter-checkin distance.
            inter_dists: List[float] = []
            for i in range(1, len(user_df)):
                d = _haversine_m(
                    user_df.iloc[i - 1]["latitude"],
                    user_df.iloc[i - 1]["longitude"],
                    user_df.iloc[i]["latitude"],
                    user_df.iloc[i]["longitude"],
                )
                inter_dists.append(d)
            user_avg_dist = float(np.mean(inter_dists)) if inter_dists else 1.0

            # Segment into daily trajectories.
            user_df["date"] = user_df["timestamp"].dt.date
            for date_val, day_df in user_df.groupby("date"):
                day_df = day_df.reset_index(drop=True)
                if len(day_df) < 1:
                    continue

                episodes: List[SemanticEpisode] = []
                for i in range(len(day_df)):
                    row = day_df.iloc[i]
                    ts: pd.Timestamp = row["timestamp"]
                    lat, lon = row["latitude"], row["longitude"]

                    zone = _stable_zone_id(lat, lon)
                    poi_role = self._resolve_poi_role(
                        str(row.get("venue_category", ""))
                    )
                    time_bin = self._compute_time_bin(ts.hour, ts.dayofweek)

                    # Estimate dwell as gap to next check-in (capped at 12 h).
                    if i + 1 < len(day_df):
                        next_ts = day_df.iloc[i + 1]["timestamp"]
                        dwell_min = (next_ts - ts).total_seconds() / 60.0
                        dwell_min = min(dwell_min, 720.0)
                    else:
                        dwell_min = 30.0  # default for last check-in

                    dwell_bin = self._discretize_dwell(dwell_min)

                    # Transition from previous check-in.
                    if i > 0:
                        prev = day_df.iloc[i - 1]
                        seg_dist = _haversine_m(
                            prev["latitude"], prev["longitude"], lat, lon
                        )
                        dt_sec = (ts - prev["timestamp"]).total_seconds()
                        speed = seg_dist / dt_sec if dt_sec > 0 else 0
                        transition = _infer_transition(speed)
                        trip_len_change = seg_dist / user_avg_dist if user_avg_dist > 0 else 1.0
                    else:
                        transition = 3  # jump (first check-in of the day)
                        trip_len_change = 1.0

                    episodes.append(
                        SemanticEpisode(
                            zone_id=zone,
                            poi_role=poi_role,
                            time_bin=time_bin,
                            dwell_bin=dwell_bin,
                            transition_type=transition,
                            trip_length_change=round(trip_len_change, 4),
                            event_flag=0,
                            companion_flag=0,
                        )
                    )

                label = int(day_df["label"].iloc[0]) if "label" in day_df.columns else 0
                trajectories.append(
                    SemanticTrajectory(
                        episodes=episodes,
                        user_id=str(user_id),
                        trip_id=f"{user_id}_{date_val}",
                        label=label,
                    )
                )

        return trajectories

    def tokenize_numosim(self, raw_df: pd.DataFrame) -> List[SemanticTrajectory]:
        """Tokenize NumoSim synthetic mobility traces.

        Expected columns: ``agent_id``, ``trip_id``, ``step``,
        ``latitude``, ``longitude``, ``timestamp``, ``activity_type``,
        ``mode``, ``poi_type``, ``event_flag``, ``companion_flag``,
        and optionally ``label``, ``primitive_labels``.
        """
        raw_df = raw_df.copy()
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
        raw_df.sort_values(["agent_id", "trip_id", "step"], inplace=True)

        trajectories: List[SemanticTrajectory] = []

        # Per-agent average distance for normalisation.
        agent_avg_dist: Dict[str, float] = {}
        for agent_id, adf in raw_df.groupby("agent_id"):
            dists: List[float] = []
            for i in range(1, len(adf)):
                d = _haversine_m(
                    adf.iloc[i - 1]["latitude"],
                    adf.iloc[i - 1]["longitude"],
                    adf.iloc[i]["latitude"],
                    adf.iloc[i]["longitude"],
                )
                dists.append(d)
            agent_avg_dist[str(agent_id)] = float(np.mean(dists)) if dists else 1.0

        mode_map = {"walk": 0, "drive": 1, "transit": 2, "jump": 3}

        for (agent_id, trip_id), trip_df in raw_df.groupby(["agent_id", "trip_id"]):
            trip_df = trip_df.reset_index(drop=True)
            if len(trip_df) < 1:
                continue

            avg_d = agent_avg_dist.get(str(agent_id), 1.0)
            episodes: List[SemanticEpisode] = []
            prim_labels: Optional[List[List[int]]] = None

            has_prims = "primitive_labels" in trip_df.columns
            if has_prims:
                prim_labels = []

            for i in range(len(trip_df)):
                row = trip_df.iloc[i]
                ts: pd.Timestamp = row["timestamp"]
                lat, lon = row["latitude"], row["longitude"]

                zone = _stable_zone_id(lat, lon)
                poi_role = self._resolve_poi_role(str(row.get("poi_type", "")))
                time_bin = self._compute_time_bin(ts.hour, ts.dayofweek)

                # Dwell: time to next step or default 15 min.
                if i + 1 < len(trip_df):
                    next_ts = trip_df.iloc[i + 1]["timestamp"]
                    dwell_min = (next_ts - ts).total_seconds() / 60.0
                    dwell_min = max(0.0, min(dwell_min, 1440.0))
                else:
                    dwell_min = 15.0

                dwell_bin = self._discretize_dwell(dwell_min)

                mode_str = str(row.get("mode", "walk")).lower()
                transition = mode_map.get(mode_str, 0)

                if i > 0:
                    prev = trip_df.iloc[i - 1]
                    seg_dist = _haversine_m(
                        prev["latitude"], prev["longitude"], lat, lon
                    )
                    trip_len_change = seg_dist / avg_d if avg_d > 0 else 1.0
                else:
                    trip_len_change = 1.0

                event_flag = int(row.get("event_flag", 0))
                companion_flag = int(row.get("companion_flag", 0))

                episodes.append(
                    SemanticEpisode(
                        zone_id=zone,
                        poi_role=poi_role,
                        time_bin=time_bin,
                        dwell_bin=dwell_bin,
                        transition_type=transition,
                        trip_length_change=round(trip_len_change, 4),
                        event_flag=event_flag,
                        companion_flag=companion_flag,
                    )
                )

                if has_prims and prim_labels is not None:
                    raw_prim = row["primitive_labels"]
                    if isinstance(raw_prim, str):
                        import json as _json
                        raw_prim = _json.loads(raw_prim)
                    prim_labels.append(list(raw_prim))

            label = int(trip_df["label"].iloc[0]) if "label" in trip_df.columns else 0

            trajectories.append(
                SemanticTrajectory(
                    episodes=episodes,
                    user_id=str(agent_id),
                    trip_id=f"{agent_id}_{trip_id}",
                    label=label,
                    primitive_labels=prim_labels,
                )
            )

        return trajectories

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discretize_dwell(self, minutes: float) -> int:
        """Map a continuous dwell duration (minutes) to a bin index 0..dwell_bins-1."""
        minutes = max(0.0, minutes)
        bin_idx = int(np.searchsorted(self._dwell_edges, minutes, side="right")) - 1
        return max(0, min(bin_idx, self.dwell_bins - 1))

    def _compute_time_bin(self, hour: int, dow: int) -> int:
        """Compute temporal bin index from hour-of-day and day-of-week.

        Parameters
        ----------
        hour : int
            Hour of day in [0, 23].
        dow : int
            Day of week in [0, 6] where 0 = Monday.

        Returns
        -------
        int
            Bin index in [0, 167].
        """
        hour = max(0, min(23, int(hour)))
        dow = max(0, min(6, int(dow)))
        return hour * 7 + dow

    def _resolve_poi_role(self, category: str) -> int:
        """Map a raw POI category string to a role index in [0, poi_vocab_size)."""
        if not category:
            return 0
        if category in self.role_mapping:
            return self.role_mapping[category]
        # Deterministic hash fallback.
        h = hashlib.md5(category.encode()).hexdigest()
        return int(h[:8], 16) % self.poi_vocab_size

    # ------------------------------------------------------------------
    # Stay-point detection for raw GPS data (GeoLife)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_stay_points_gps(
        df: pd.DataFrame,
        dist_thresh_m: float = 200.0,
        time_thresh_s: float = 300.0,
    ) -> List[Dict]:
        """Simple time-distance stay-point detection on a GPS trace.

        Returns a list of dicts with keys: lat, lon, arrival (Timestamp),
        dwell_minutes, approach_dist_m, transition, poi_category,
        event_flag, companion_flag.
        """
        stays: List[Dict] = []
        n = len(df)
        i = 0

        while i < n:
            j = i + 1
            while j < n:
                dist = _haversine_m(
                    df.iloc[i]["latitude"],
                    df.iloc[i]["longitude"],
                    df.iloc[j]["latitude"],
                    df.iloc[j]["longitude"],
                )
                if dist > dist_thresh_m:
                    break
                j += 1

            dt_sec = (df.iloc[j - 1]["timestamp"] - df.iloc[i]["timestamp"]).total_seconds()

            if dt_sec >= time_thresh_s or i == 0:
                mean_lat = df.iloc[i:j]["latitude"].mean()
                mean_lon = df.iloc[i:j]["longitude"].mean()

                # Approach distance and speed from previous stay.
                approach_dist = 0.0
                transition = 3  # jump for the first stay
                if stays:
                    approach_dist = _haversine_m(
                        stays[-1]["lat"], stays[-1]["lon"], mean_lat, mean_lon
                    )
                    gap_sec = (
                        df.iloc[i]["timestamp"] - stays[-1]["departure"]
                    ).total_seconds()
                    speed = approach_dist / gap_sec if gap_sec > 0 else 0
                    transition = _infer_transition(speed)

                poi_cat = ""
                if "poi_category" in df.columns:
                    poi_cat = str(df.iloc[i].get("poi_category", ""))

                stays.append(
                    {
                        "lat": mean_lat,
                        "lon": mean_lon,
                        "arrival": df.iloc[i]["timestamp"],
                        "departure": df.iloc[j - 1]["timestamp"],
                        "dwell_minutes": dt_sec / 60.0,
                        "approach_dist_m": approach_dist,
                        "transition": transition,
                        "poi_category": poi_cat,
                        "event_flag": int(df.iloc[i].get("event_flag", 0))
                        if "event_flag" in df.columns
                        else 0,
                        "companion_flag": int(df.iloc[i].get("companion_flag", 0))
                        if "companion_flag" in df.columns
                        else 0,
                    }
                )

            i = j

        return stays

    @staticmethod
    def _user_avg_segment_distance_gps(df: pd.DataFrame) -> float:
        """Mean inter-point haversine distance for a single user's trace."""
        if len(df) < 2:
            return 1.0
        dists: List[float] = []
        for i in range(1, len(df)):
            d = _haversine_m(
                df.iloc[i - 1]["latitude"],
                df.iloc[i - 1]["longitude"],
                df.iloc[i]["latitude"],
                df.iloc[i]["longitude"],
            )
            dists.append(d)
        return float(np.mean(dists)) if dists else 1.0
