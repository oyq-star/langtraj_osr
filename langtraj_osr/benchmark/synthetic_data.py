"""Synthetic mobility data generator for development and testing.

Creates realistic-looking SemanticTrajectory instances so that the benchmark
pipeline can be exercised without real GPS / check-in datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..core.episode import SemanticEpisode, SemanticTrajectory

# Role constants (subset of the 64-class vocabulary used for generation)
ROLE_HOME = 0
ROLE_RESIDENTIAL = 1
ROLE_WORK_OFFICE = 5
ROLE_RESTAURANT = 8
ROLE_INDUSTRIAL = 10
ROLE_SHOPPING = 12
ROLE_ENTERTAINMENT = 15
ROLE_NIGHTCLUB = 20
ROLE_TRANSIT = 25
ROLE_PARK = 30
ROLE_GYM = 35
ROLE_SCHOOL = 40


@dataclass
class UserProfile:
    """Holds the habitual mobility profile for a synthetic user."""
    user_id: str
    home_zone: int
    work_zone: int
    regular_zones: List[int]
    regular_roles: List[int]
    work_start_hour: int  # typical arrival hour
    work_end_hour: int
    has_companion: bool
    companion_zones: Set[int]


class SyntheticMobilityGenerator:
    """Generate synthetic trajectory data for development / sanity checks."""

    def __init__(
        self,
        n_users: int = 200,
        n_zones: int = 50,
        n_roles: int = 64,
        seed: int = 42,
    ) -> None:
        self.n_users = n_users
        self.n_zones = n_zones
        self.n_roles = n_roles
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._profiles: List[UserProfile] = []

    # ------------------------------------------------------------------
    # Profile generation
    # ------------------------------------------------------------------

    def _generate_profiles(self) -> List[UserProfile]:
        """Create user profiles with home, work, and regular POIs."""
        profiles: List[UserProfile] = []
        for uid in range(self.n_users):
            home_zone = int(self.rng.randint(0, self.n_zones))
            work_zone = int(self.rng.randint(0, self.n_zones))
            while work_zone == home_zone:
                work_zone = int(self.rng.randint(0, self.n_zones))

            # 3-7 regular zones (shopping, restaurant, gym, etc.)
            n_regular = self.rng.randint(3, 8)
            regular_zones = list(
                self.rng.choice(self.n_zones, size=n_regular, replace=False)
            )
            regular_roles = list(
                self.rng.choice(
                    [ROLE_RESTAURANT, ROLE_SHOPPING, ROLE_ENTERTAINMENT,
                     ROLE_PARK, ROLE_GYM, ROLE_SCHOOL, ROLE_TRANSIT],
                    size=n_regular,
                    replace=True,
                )
            )

            has_companion = bool(self.rng.random() < 0.4)
            companion_zones: Set[int] = set()
            if has_companion:
                companion_zones = set(
                    self.rng.choice(regular_zones, size=min(2, len(regular_zones)), replace=False)
                )

            profiles.append(UserProfile(
                user_id=f"user_{uid:04d}",
                home_zone=home_zone,
                work_zone=work_zone,
                regular_zones=regular_zones,
                regular_roles=regular_roles,
                work_start_hour=int(self.rng.choice([7, 8, 9])),
                work_end_hour=int(self.rng.choice([17, 18, 19])),
                has_companion=has_companion,
                companion_zones=companion_zones,
            ))
        return profiles

    # ------------------------------------------------------------------
    # Single-trip generation
    # ------------------------------------------------------------------

    def _make_episode(
        self,
        zone_id: int,
        poi_role: int,
        hour: int,
        dow: int,
        dwell_bin: int,
        transition: int = 1,
        trip_length_change: float = 1.0,
        event_flag: int = 0,
        companion_flag: int = 0,
    ) -> SemanticEpisode:
        return SemanticEpisode(
            zone_id=zone_id,
            poi_role=poi_role,
            time_bin=hour * 7 + dow,
            dwell_bin=min(dwell_bin, SemanticEpisode.DWELL_BIN_VOCAB - 1),
            transition_type=transition,
            trip_length_change=trip_length_change,
            event_flag=event_flag,
            companion_flag=companion_flag,
        )

    def _generate_weekday_trip(
        self, profile: UserProfile, trip_idx: int, dow: int,
    ) -> SemanticTrajectory:
        """home -> commute -> work -> lunch -> work -> commute -> home."""
        eps: List[SemanticEpisode] = []

        # Home departure
        depart_hour = profile.work_start_hour + int(self.rng.choice([-1, 0, 0, 0]))
        eps.append(self._make_episode(
            zone_id=profile.home_zone,
            poi_role=ROLE_HOME,
            hour=max(0, depart_hour - 1),
            dow=dow,
            dwell_bin=int(self.rng.randint(8, 14)),
            transition=0,  # walk out
            companion_flag=int(profile.home_zone in profile.companion_zones),
        ))

        # Commute transit
        transit_zone = int(self.rng.choice(profile.regular_zones))
        eps.append(self._make_episode(
            zone_id=transit_zone,
            poi_role=ROLE_TRANSIT,
            hour=depart_hour,
            dow=dow,
            dwell_bin=int(self.rng.randint(1, 4)),
            transition=int(self.rng.choice([1, 2])),
        ))

        # Work morning
        eps.append(self._make_episode(
            zone_id=profile.work_zone,
            poi_role=ROLE_WORK_OFFICE,
            hour=profile.work_start_hour,
            dow=dow,
            dwell_bin=int(self.rng.randint(10, 15)),
            transition=int(self.rng.choice([1, 2])),
        ))

        # Lunch
        lunch_zone = int(self.rng.choice(profile.regular_zones))
        lunch_role_idx = profile.regular_zones.index(lunch_zone) if lunch_zone in profile.regular_zones else 0
        lunch_role = profile.regular_roles[lunch_role_idx % len(profile.regular_roles)]
        eps.append(self._make_episode(
            zone_id=lunch_zone,
            poi_role=lunch_role if self.rng.random() < 0.5 else ROLE_RESTAURANT,
            hour=12 + int(self.rng.choice([0, 1])),
            dow=dow,
            dwell_bin=int(self.rng.randint(2, 5)),
            transition=0,
        ))

        # Work afternoon
        eps.append(self._make_episode(
            zone_id=profile.work_zone,
            poi_role=ROLE_WORK_OFFICE,
            hour=13 + int(self.rng.choice([0, 1])),
            dow=dow,
            dwell_bin=int(self.rng.randint(8, 13)),
            transition=0,
        ))

        # Occasional after-work activity (30 % chance)
        if self.rng.random() < 0.3:
            aw_idx = int(self.rng.randint(0, len(profile.regular_zones)))
            aw_zone = profile.regular_zones[aw_idx]
            aw_role = profile.regular_roles[aw_idx]
            companion = int(
                profile.has_companion and aw_zone in profile.companion_zones
            )
            eps.append(self._make_episode(
                zone_id=aw_zone,
                poi_role=aw_role,
                hour=profile.work_end_hour,
                dow=dow,
                dwell_bin=int(self.rng.randint(3, 7)),
                transition=int(self.rng.choice([0, 1])),
                companion_flag=companion,
            ))

        # Commute home
        eps.append(self._make_episode(
            zone_id=transit_zone,
            poi_role=ROLE_TRANSIT,
            hour=profile.work_end_hour + int(self.rng.choice([0, 1])),
            dow=dow,
            dwell_bin=int(self.rng.randint(1, 4)),
            transition=int(self.rng.choice([1, 2])),
        ))

        # Home arrival
        eps.append(self._make_episode(
            zone_id=profile.home_zone,
            poi_role=ROLE_HOME,
            hour=min(23, profile.work_end_hour + int(self.rng.choice([1, 2]))),
            dow=dow,
            dwell_bin=int(self.rng.randint(8, 15)),
            transition=0,
            companion_flag=int(profile.home_zone in profile.companion_zones),
        ))

        # Occasional event (5 % chance on any episode)
        for ep in eps:
            if self.rng.random() < 0.05:
                ep.event_flag = 1

        return SemanticTrajectory(
            episodes=eps,
            user_id=profile.user_id,
            trip_id=f"{profile.user_id}_trip{trip_idx:04d}",
            label=0,
            primitive_labels=None,
        )

    def _generate_weekend_trip(
        self, profile: UserProfile, trip_idx: int, dow: int,
    ) -> SemanticTrajectory:
        """home -> shopping/entertainment -> restaurant -> home."""
        eps: List[SemanticEpisode] = []

        start_hour = int(self.rng.randint(9, 12))

        # Home
        eps.append(self._make_episode(
            zone_id=profile.home_zone,
            poi_role=ROLE_HOME,
            hour=start_hour - 1,
            dow=dow,
            dwell_bin=int(self.rng.randint(10, 15)),
            transition=0,
            companion_flag=int(profile.home_zone in profile.companion_zones),
        ))

        # 1-3 leisure stops
        n_stops = self.rng.randint(1, 4)
        hour = start_hour
        for _ in range(n_stops):
            s_idx = int(self.rng.randint(0, len(profile.regular_zones)))
            zone = profile.regular_zones[s_idx]
            role = profile.regular_roles[s_idx]
            companion = int(
                profile.has_companion and zone in profile.companion_zones
            )
            eps.append(self._make_episode(
                zone_id=zone,
                poi_role=role,
                hour=min(23, hour),
                dow=dow,
                dwell_bin=int(self.rng.randint(3, 9)),
                transition=int(self.rng.choice([0, 1])),
                companion_flag=companion,
            ))
            hour += int(self.rng.randint(1, 3))

        # Restaurant
        rest_zone = int(self.rng.choice(profile.regular_zones))
        eps.append(self._make_episode(
            zone_id=rest_zone,
            poi_role=ROLE_RESTAURANT,
            hour=min(23, hour),
            dow=dow,
            dwell_bin=int(self.rng.randint(3, 7)),
            transition=int(self.rng.choice([0, 1])),
            companion_flag=int(
                profile.has_companion and rest_zone in profile.companion_zones
            ),
        ))

        # Home return
        eps.append(self._make_episode(
            zone_id=profile.home_zone,
            poi_role=ROLE_HOME,
            hour=min(23, hour + int(self.rng.randint(1, 3))),
            dow=dow,
            dwell_bin=int(self.rng.randint(8, 15)),
            transition=0,
            companion_flag=int(profile.home_zone in profile.companion_zones),
        ))

        # Occasional event
        for ep in eps:
            if self.rng.random() < 0.08:
                ep.event_flag = 1

        return SemanticTrajectory(
            episodes=eps,
            user_id=profile.user_id,
            trip_id=f"{profile.user_id}_trip{trip_idx:04d}",
            label=0,
            primitive_labels=None,
        )

    # ------------------------------------------------------------------
    # Novel destination injection (sparse users get fewer, rich get more)
    # ------------------------------------------------------------------

    def _maybe_inject_novel(
        self, traj: SemanticTrajectory, profile: UserProfile,
    ) -> SemanticTrajectory:
        """With small probability, replace one stop with a novel destination."""
        if self.rng.random() > 0.08 or len(traj.episodes) < 3:
            return traj

        idx = int(self.rng.randint(1, len(traj.episodes) - 1))
        novel_zone = int(self.rng.randint(0, self.n_zones))
        novel_role = int(self.rng.randint(0, self.n_roles))
        traj.episodes[idx].zone_id = novel_zone
        traj.episodes[idx].poi_role = novel_role
        traj.episodes[idx].trip_length_change = float(self.rng.uniform(1.2, 2.5))
        return traj

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self, trips_per_user: int = 100,
    ) -> List[SemanticTrajectory]:
        """Generate the full synthetic dataset.

        Args:
            trips_per_user: Number of trips to generate per user.

        Returns:
            Flat list of SemanticTrajectory instances (all labelled normal).
        """
        self._profiles = self._generate_profiles()
        all_trajectories: List[SemanticTrajectory] = []

        for profile in self._profiles:
            # Some users are sparse (20-50 % of full trips)
            is_sparse = self.rng.random() < 0.25
            actual_trips = (
                int(trips_per_user * self.rng.uniform(0.2, 0.5))
                if is_sparse
                else trips_per_user
            )

            for t in range(actual_trips):
                dow = t % 7  # cycle through days
                is_weekend = dow >= 5

                if is_weekend:
                    traj = self._generate_weekend_trip(profile, t, dow)
                else:
                    traj = self._generate_weekday_trip(profile, t, dow)

                # Add timing noise to trip_length_change
                for ep in traj.episodes:
                    ep.trip_length_change = float(
                        max(0.3, 1.0 + self.rng.normal(0, 0.15))
                    )

                traj = self._maybe_inject_novel(traj, profile)
                all_trajectories.append(traj)

        return all_trajectories
