"""PyTorch datasets and data-loading utilities for MobDef-Bench.

Provides:
* ``MobDefBenchDataset`` -- a map-style ``torch.utils.data.Dataset`` that
  yields padded episode tensors, user-history tensors, textual concept
  definitions, labels, primitive labels, and attention masks.
* ``collate_mobdef`` -- a custom collate function for ``DataLoader``.
* ``MobDefBenchDataModule`` -- a lightweight data-module that builds
  train / val / test ``DataLoader`` instances with proper concept splits.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .concepts import ANOMALY_CONCEPTS, CONCEPT_BY_ID, get_concept_ids_for_split
from .episode import SemanticEpisode, SemanticTrajectory

# Number of numeric fields per episode (see SemanticEpisode.to_list).
_EP_DIM: int = 8
# Number of behavioural primitives.
_NUM_PRIMS: int = 10


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MobDefBenchDataset(Dataset):
    """Map-style dataset for the MobDef-Bench benchmark.

    Parameters
    ----------
    trajectories : list[SemanticTrajectory]
        Pre-tokenised trajectory objects.
    concept_definitions : dict[int, list[str]]
        Mapping from concept id to a list of textual definitions
        (canonical + paraphrases).  Unknown-anomaly concepts may map to
        an empty list.
    user_histories : dict[str, list[SemanticTrajectory]]
        Per-user bank of historical normal trajectories used as context.
    max_len : int
        Maximum number of episodes per trajectory (longer ones are truncated).
    max_history : int
        Maximum number of episodes drawn from the user history bank.
    split : str
        One of ``'train'``, ``'val'``, ``'test'``.  Controls whether
        definition paraphrases are randomly sampled (train) or fixed
        to canonical (val / test).
    """

    def __init__(
        self,
        trajectories: List[SemanticTrajectory],
        concept_definitions: Dict[int, List[str]],
        user_histories: Dict[str, List[SemanticTrajectory]],
        max_len: int = 64,
        max_history: int = 64,
        split: str = "train",
    ) -> None:
        super().__init__()
        self.trajectories = trajectories
        self.concept_definitions = concept_definitions
        self.user_histories = user_histories
        self.max_len = max_len
        self.max_history = max_history
        self.split = split

    # ---- Dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj = self.trajectories[idx]

        # 1. Episode tensor  (L x 8)
        truncated = traj.truncate(self.max_len)
        ep_list = truncated.to_tensor_list()
        seq_len = len(ep_list)
        # Pad to max_len
        while len(ep_list) < self.max_len:
            ep_list.append([0.0] * _EP_DIM)
        episode_tensor = torch.tensor(ep_list, dtype=torch.float32)  # (max_len, 8)

        # 2. Mask  (max_len,)
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:seq_len] = True

        # 3. User history tensor  (K x 8)
        history_eps = self._sample_user_history(traj.user_id)
        hist_len = len(history_eps)
        while len(history_eps) < self.max_history:
            history_eps.append([0.0] * _EP_DIM)
        user_history_tensor = torch.tensor(
            history_eps[: self.max_history], dtype=torch.float32
        )
        history_mask = torch.zeros(self.max_history, dtype=torch.bool)
        history_mask[:hist_len] = True

        # 4. Definition text
        definition_text = self._pick_definition(traj.label)

        # 5. Label
        label = traj.label

        # 6. Primitive labels  (L x 10), padded
        if traj.primitive_labels is not None:
            prim = traj.primitive_labels[: self.max_len]
            prim_np = np.zeros((self.max_len, _NUM_PRIMS), dtype=np.float32)
            for i, p in enumerate(prim):
                prim_np[i, : len(p)] = p[: _NUM_PRIMS]
            primitive_labels = torch.from_numpy(prim_np)
        else:
            primitive_labels = torch.zeros(
                (self.max_len, _NUM_PRIMS), dtype=torch.float32
            )

        return {
            "episode_tensor": episode_tensor,
            "user_history_tensor": user_history_tensor,
            "definition_text": definition_text,
            "label": label,
            "primitive_labels": primitive_labels,
            "mask": mask,
            "history_mask": history_mask,
            "user_id": traj.user_id,
            "trip_id": traj.trip_id,
        }

    # ---- Internal helpers ---------------------------------------------------

    def _sample_user_history(self, user_id: str) -> List[List[float]]:
        """Flatten and sample episodes from the user's history bank."""
        hist_trajs = self.user_histories.get(user_id, [])
        all_eps: List[List[float]] = []
        for ht in hist_trajs:
            all_eps.extend(ht.to_tensor_list())
        if not all_eps:
            return []
        if self.split == "train" and len(all_eps) > self.max_history:
            all_eps = random.sample(all_eps, self.max_history)
        return all_eps[: self.max_history]

    def _pick_definition(self, label: int) -> str:
        """Select a textual definition for the given concept label."""
        if label <= 0:
            # Normal or unknown anomaly -- return empty string.
            defs = self.concept_definitions.get(label, [])
            if defs:
                return defs[0]
            return ""

        defs = self.concept_definitions.get(label, [])
        if not defs:
            return ""

        if self.split == "train" and len(defs) > 1:
            return random.choice(defs)
        return defs[0]  # canonical for val / test


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_mobdef(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate that stacks tensors and collects strings into lists."""
    episode_tensors = torch.stack([b["episode_tensor"] for b in batch])
    user_history_tensors = torch.stack([b["user_history_tensor"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    history_masks = torch.stack([b["history_mask"] for b in batch])
    primitive_labels = torch.stack([b["primitive_labels"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    definition_texts = [b["definition_text"] for b in batch]
    user_ids = [b["user_id"] for b in batch]
    trip_ids = [b["trip_id"] for b in batch]

    return {
        "episode_tensor": episode_tensors,        # (B, L, 8)
        "user_history_tensor": user_history_tensors,  # (B, K, 8)
        "definition_text": definition_texts,       # list[str]
        "label": labels,                           # (B,)
        "primitive_labels": primitive_labels,       # (B, L, 10)
        "mask": masks,                             # (B, L)
        "history_mask": history_masks,             # (B, K)
        "user_id": user_ids,
        "trip_id": trip_ids,
    }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class MobDefBenchDataModule:
    """Lightweight data-module that constructs DataLoaders for MobDef-Bench.

    Parameters
    ----------
    trajectories : dict[str, list[SemanticTrajectory]]
        Keyed by split name: ``'train'``, ``'val'``, ``'test'``.
    concept_definitions : dict[int, list[str]]
        Textual definitions per concept id.
    user_histories : dict[str, list[SemanticTrajectory]]
        Historical normal trajectories per user.
    split_config : dict
        Configuration for concept splits.  Keys:

        * ``A_seen`` -- list of concept ids for the training set.
        * ``A_zs_comp`` -- ids for zero-shot compositional evaluation.
        * ``A_zs_family`` -- ids for held-out operator family.
        * ``A_unknown`` -- ids for unknown anomaly evaluation.
    batch_size : int
    num_workers : int
    max_len : int
    max_history : int
    """

    def __init__(
        self,
        trajectories: Dict[str, List[SemanticTrajectory]],
        concept_definitions: Dict[int, List[str]],
        user_histories: Dict[str, List[SemanticTrajectory]],
        split_config: Optional[Dict[str, List[int]]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        max_len: int = 64,
        max_history: int = 64,
    ) -> None:
        self.trajectories = trajectories
        self.concept_definitions = concept_definitions
        self.user_histories = user_histories
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len
        self.max_history = max_history

        if split_config is None:
            self.split_config: Dict[str, List[int]] = {
                "A_seen": get_concept_ids_for_split("seen"),
                "A_zs_comp": get_concept_ids_for_split("zs_comp"),
                "A_zs_family": get_concept_ids_for_split("zs_family"),
                "A_unknown": get_concept_ids_for_split("unknown"),
            }
        else:
            self.split_config = split_config

    # ---- Public API ---------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return self._build_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._build_loader("test", shuffle=False)

    def concept_split_dataloaders(
        self,
    ) -> Dict[str, DataLoader]:
        """Return one DataLoader per concept split for fine-grained evaluation.

        Keys: ``A_seen``, ``A_zs_comp``, ``A_zs_family``, ``A_unknown``.
        """
        test_trajs = self.trajectories.get("test", [])
        loaders: Dict[str, DataLoader] = {}

        for split_name, concept_ids in self.split_config.items():
            id_set = set(concept_ids)
            # Include normal trajectories (label 0) in every split for
            # closed-set vs open-set discrimination.
            split_trajs = [
                t for t in test_trajs if t.label in id_set or t.label == 0
            ]
            if not split_trajs:
                continue
            ds = MobDefBenchDataset(
                split_trajs,
                self.concept_definitions,
                self.user_histories,
                max_len=self.max_len,
                max_history=self.max_history,
                split="test",
            )
            loaders[split_name] = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_mobdef,
                pin_memory=True,
            )
        return loaders

    # ---- Convenience class method -------------------------------------------

    @classmethod
    def load_dataset(
        cls,
        dataset_name: str,
        split_config: Optional[Dict[str, List[int]]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        max_len: int = 64,
        max_history: int = 64,
    ) -> "MobDefBenchDataModule":
        """Factory that loads a pre-processed dataset by name.

        Parameters
        ----------
        dataset_name : str
            One of ``'geolife'``, ``'porto'``, ``'foursquare'``, ``'numosim'``.
        split_config : dict | None
            Concept split configuration.  If *None*, the default 4-split
            layout defined by :pymod:`core.concepts` is used.
        batch_size : int
        num_workers : int
        max_len : int
        max_history : int

        Returns
        -------
        MobDefBenchDataModule
            Ready-to-use data module with train / val / test loaders.

        Raises
        ------
        FileNotFoundError
            If the pre-processed cache for *dataset_name* does not exist.
        """
        import json
        import os

        cache_root = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "processed",
            dataset_name,
        )

        if not os.path.isdir(cache_root):
            raise FileNotFoundError(
                f"Processed dataset cache not found at {cache_root}. "
                f"Run the preprocessing pipeline first."
            )

        trajectories: Dict[str, List[SemanticTrajectory]] = {}
        for split in ("train", "val", "test"):
            split_path = os.path.join(cache_root, f"{split}.json")
            if not os.path.isfile(split_path):
                trajectories[split] = []
                continue
            with open(split_path, "r") as f:
                raw = json.load(f)
            trajectories[split] = _deserialise_trajectories(raw)

        # Load concept definitions.
        from .concepts import get_all_definitions

        concept_definitions = get_all_definitions(include_paraphrases=True)

        # Load user histories.
        hist_path = os.path.join(cache_root, "user_histories.json")
        user_histories: Dict[str, List[SemanticTrajectory]] = {}
        if os.path.isfile(hist_path):
            with open(hist_path, "r") as f:
                raw_hist = json.load(f)
            for uid, traj_list in raw_hist.items():
                user_histories[uid] = _deserialise_trajectories(traj_list)

        return cls(
            trajectories=trajectories,
            concept_definitions=concept_definitions,
            user_histories=user_histories,
            split_config=split_config,
            batch_size=batch_size,
            num_workers=num_workers,
            max_len=max_len,
            max_history=max_history,
        )

    # ---- Private helpers ----------------------------------------------------

    def _build_loader(self, split: str, shuffle: bool) -> DataLoader:
        trajs = self.trajectories.get(split, [])
        ds = MobDefBenchDataset(
            trajs,
            self.concept_definitions,
            self.user_histories,
            max_len=self.max_len,
            max_history=self.max_history,
            split=split,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_mobdef,
            pin_memory=True,
            drop_last=(split == "train"),
        )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _deserialise_trajectories(
    raw_list: List[Dict[str, Any]],
) -> List[SemanticTrajectory]:
    """Reconstruct SemanticTrajectory objects from JSON-friendly dicts."""
    trajs: List[SemanticTrajectory] = []
    for item in raw_list:
        episodes = [
            SemanticEpisode.from_list(ep) for ep in item.get("episodes", [])
        ]
        prim = item.get("primitive_labels")
        trajs.append(
            SemanticTrajectory(
                episodes=episodes,
                user_id=item.get("user_id", ""),
                trip_id=item.get("trip_id", ""),
                label=item.get("label", 0),
                primitive_labels=prim,
            )
        )
    return trajs
