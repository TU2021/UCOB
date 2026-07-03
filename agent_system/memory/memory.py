# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import re
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import BaseMemory

class SimpleMemory(BaseMemory):
    """
    Memory manager: responsible for storing & fetching per‑environment history records.
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str, default="text_obs"):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str, default="action"):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths
    

class SearchMemory(BaseMemory):
    """
    Memory manager for search tasks: responsible for storing & fetching
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str,
        action_key: str,
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"Step {step_num}:{act} {obs}\n"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths


class SkillLibrary:
    """Persistent skill memory used by Skill1-style training."""

    def __init__(
        self,
        filepath: str = "memory_cache/skill_library.json",
        model_name: str = "all-MiniLM-L6-v2",
        relevance_weight: float = 0.6,
        alpha: float = 0.1,
        temperature: float = 0.5,
        retrieve_type: str = "ucb",
        ucb_scale: float = 1.0,
        use_description_head: bool = False,
        max_size: int = 5000,
        retriever_type: str = "dense",
    ) -> None:
        self.filepath = filepath
        self.relevance_weight = relevance_weight
        self.alpha = alpha
        self.temperature = temperature
        self.retrieve_type = str(retrieve_type).lower()
        self.ucb_scale = ucb_scale
        self.use_description_head = use_description_head
        self.max_size = max_size
        self.retriever_type = str(retriever_type).lower()
        self._last_evicted_count = 0
        self._current_training_step = 0
        self.prune_milestones = [0.25, 0.5, 0.75]
        self.processed_milestones = set()
        self.data: List[Dict[str, Any]] = []
        self._tokenizer = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._sklearn_cosine = None
        self.model = None
        self.task_embeddings = None
        self._backend = "lexical"

        self._setup_backend(model_name=model_name)
        self.load()

    def _setup_backend(self, model_name: str) -> None:
        if self.retriever_type == "tfidf":
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

                self._tfidf_vectorizer = TfidfVectorizer()
                self._sklearn_cosine = sklearn_cosine
                self._backend = "tfidf"
                return
            except Exception:
                self._backend = "lexical"

        if self.retriever_type in {"dense", "sentence", "sentence_transformer"}:
            model_path = os.environ.get("EMBEDDING_MODEL_PATH", model_name)
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_path)
                self._backend = "dense"
                return
            except Exception:
                self._backend = "lexical"

        self._backend = "lexical"

    def __len__(self):
        return len(self.data)

    def load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError:
                    self.data = []
        self._rebuild_embeddings()

    def save(self):
        directory = os.path.dirname(self.filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def _normalize_text(self, text: str) -> str:
        text = str(text or "").lower()
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _token_set(self, text: str) -> set[str]:
        text = self._normalize_text(text)
        return {tok for tok in text.split(" ") if tok}

    def _rebuild_embeddings(self):
        if not self.data:
            self.task_embeddings = None
            self._tfidf_matrix = None
            return
        texts = []
        for item in self.data:
            if self.use_description_head and item.get("description_head"):
                texts.append(item.get("description_head") or item.get("scenario_desc", ""))
            else:
                texts.append(item.get("scenario_desc", ""))
        if self._backend == "tfidf" and self._tfidf_vectorizer is not None:
            try:
                self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
                self.task_embeddings = None
                return
            except Exception:
                self._backend = "lexical"
        if self._backend == "dense" and self.model is not None:
            try:
                self.task_embeddings = self.model.encode(texts, convert_to_tensor=True)
                self._tfidf_matrix = None
                return
            except Exception:
                self._backend = "lexical"
        self.task_embeddings = None
        self._tfidf_matrix = None

    def _compute_similarity_single(self, text_a: str, text_b: str) -> float:
        if self._backend == "tfidf" and self._tfidf_vectorizer is not None and self._sklearn_cosine is not None:
            try:
                vecs = self._tfidf_vectorizer.fit_transform([text_a, text_b])
                return float(self._sklearn_cosine(vecs[0:1], vecs[1:2])[0][0])
            except Exception:
                pass
        if self._backend == "dense" and self.model is not None:
            try:
                emb1 = self.model.encode(text_a, convert_to_tensor=True)
                emb2 = self.model.encode(text_b, convert_to_tensor=True)
                from sentence_transformers import util

                return float(util.cos_sim(emb1, emb2).item())
            except Exception:
                pass
        a_tokens = self._token_set(text_a)
        b_tokens = self._token_set(text_b)
        if not a_tokens or not b_tokens:
            return 0.0
        return float(len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1))

    def _ema_update(self, item, new_score):
        current_avg = float(item.get("utility_score", 0.5))
        item["utility_score"] = (1 - self.alpha) * current_avg + self.alpha * float(new_score)
        item["count"] = int(item.get("count", 1)) + 1

    def admit(
        self,
        scenario_description: str,
        strategy_text: str,
        trajectory: str,
        initial_score: float = 0.5,
        attempt_type: str = "unknown",
        current_progress_ratio: float = 0.0,
        description_head: str = "",
    ):
        for item in self.data:
            if item.get("strategy") == strategy_text:
                self._ema_update(item, initial_score)
                if attempt_type and item.get("attempt_type", "unknown") == "unknown":
                    item["attempt_type"] = attempt_type
                if description_head and not item.get("description_head"):
                    item["description_head"] = description_head
                self.save()
                return

        new_entry = {
            "skill_id": str(uuid.uuid4()),
            "scenario_desc": scenario_description,
            "description_head": description_head,
            "strategy": strategy_text,
            "trajectory": trajectory,
            "utility_score": float(initial_score),
            "count": 1,
            "attempt_type": attempt_type,
            "created_at_progress": float(current_progress_ratio),
            "created_at_step": int(self._current_training_step),
        }
        self.data.append(new_entry)
        self._rebuild_embeddings()
        self.save()
        self._retire()

    def update_utility(self, scenario_description: str, strategy_text: str, score: float):
        for item in self.data:
            if item.get("strategy") == strategy_text:
                self._ema_update(item, score)
                self.save()
                return

    def _retire(self):
        if len(self.data) <= self.max_size:
            self._last_evicted_count = 0
            return
        self.data.sort(
            key=lambda x: float(x.get("utility_score", 0.5)) * math.log2(float(x.get("count", 1)) + 1.0),
            reverse=True,
        )
        self._last_evicted_count = max(len(self.data) - self.max_size, 0)
        self.data = self.data[: self.max_size]
        self._rebuild_embeddings()
        self.save()

    def check_and_prune(self, progress_ratio: float, top_k: int = 3):
        for milestone in self.prune_milestones:
            if progress_ratio >= milestone and milestone not in self.processed_milestones:
                self._prune_memory(top_k=top_k)
                self.processed_milestones.add(milestone)

    def _prune_memory(self, top_k: int):
        if not self.data:
            return
        grouped_memory = defaultdict(list)
        for item in self.data:
            grouped_memory[item.get("scenario_desc", "")].append(item)
        pruned_data = []
        for _, items in grouped_memory.items():
            items.sort(key=lambda x: float(x.get("utility_score", 0.0)), reverse=True)
            kept_items = items[:top_k]
            for item in kept_items:
                item["utility_score"] = float(item.get("utility_score", 0.0)) * 0.9
                item["count"] = 1
            pruned_data.extend(kept_items)
        self.data = pruned_data
        self._rebuild_embeddings()
        self.save()

    def retrieve_trajectory(self, current_scenario_description: str, target_type: str = "success") -> Optional[str]:
        for item in self.data:
            if item.get("scenario_desc") == current_scenario_description and item.get("attempt_type") == target_type:
                return item.get("trajectory")
        return None

    def retrieve(self, current_scenario_description: str, top_k: int = 3, filter_type: str = "both") -> List[Dict[str, str]]:
        if not self.data:
            return []

        if self._backend == "tfidf" and self._tfidf_vectorizer is not None and self._tfidf_matrix is not None:
            try:
                query_vec = self._tfidf_vectorizer.transform([current_scenario_description])
                cos_scores = self._sklearn_cosine(query_vec, self._tfidf_matrix)[0]
            except Exception:
                cos_scores = np.array([self._compute_similarity_single(current_scenario_description, item.get("scenario_desc", "")) for item in self.data])
        elif self._backend == "dense" and self.model is not None and self.task_embeddings is not None:
            try:
                query_embedding = self.model.encode(current_scenario_description, convert_to_tensor=True)
                cos_scores = self._to_numpy(self._dense_cosine(query_embedding, self.task_embeddings))
            except Exception:
                cos_scores = np.array([self._compute_similarity_single(current_scenario_description, item.get("scenario_desc", "")) for item in self.data])
        else:
            cos_scores = np.array([self._compute_similarity_single(current_scenario_description, item.get("scenario_desc", "")) for item in self.data])

        candidates = []
        total_system_counts = max(sum(int(item.get("count", 1)) for item in self.data), 1)
        for idx, item in enumerate(self.data):
            item_type = item.get("attempt_type", "unknown")
            if filter_type != "both" and item_type != filter_type:
                continue
            relevance = float(cos_scores[idx]) if idx < len(cos_scores) else 0.0
            if relevance < 0.05:
                continue
            utility = float(item.get("utility_score", 0.5))
            count = max(int(item.get("count", 1)), 1)
            if self.retrieve_type == "relevance_only":
                final_score = relevance
            elif self.retrieve_type == "ucb":
                exploration_bonus = self.ucb_scale * math.sqrt(math.log(total_system_counts + 1.0) / count)
                final_score = (self.relevance_weight * relevance) + ((1 - self.relevance_weight) * (utility + exploration_bonus))
            else:
                final_score = (self.relevance_weight * relevance) + ((1 - self.relevance_weight) * utility)
            candidates.append(
                {
                    "text": item.get("strategy", ""),
                    "score": final_score,
                    "type": item_type,
                    "similarity_score": relevance,
                    "skill_id": item.get("skill_id", ""),
                    "utility_score": utility,
                }
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:top_k]
        return [
            {
                "text": c["text"],
                "type": c["type"],
                "similarity_score": c["similarity_score"],
                "skill_id": c["skill_id"],
                "utility_score": c["utility_score"],
            }
            for c in candidates
        ]

    def _dense_cosine(self, query_embedding, task_embeddings):
        from sentence_transformers import util

        return util.cos_sim(query_embedding, task_embeddings)[0]

    def _to_numpy(self, value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)
