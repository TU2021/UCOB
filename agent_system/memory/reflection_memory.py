import json
import math
import os
import re
import uuid
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Tuple


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class ReflectionMemory:
    """RetroAgent-style online reflection memory.

    The memory starts empty by default. New entries are produced from post-episode
    reflection prompts and are later retrieved into the acting prompt.
    """

    def __init__(
        self,
        filepath: str,
        alpha: float = 0.6,
        beta: float = 0.05,
        temperature: float = 0.5,
        retrieve_type: str = "ucb",
        ucb_scale: float = 1.0,
        similarity_backend: str = "lexical",
        model_name: str = "all-MiniLM-L6-v2",
        relevance_threshold: float = 0.4,
        similarity_top_m: int = 0,
        merge_exact_task_attempt: bool = True,
        max_lessons_per_task_attempt: int = 4,
        skill_format: str = "retroagent",
        category_filter: bool = False,
        skill_mapping_dir: Optional[str] = None,
        retrieval_query_mode: str = "task",
    ):
        self.filepath = filepath
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temperature = float(temperature)
        self.retrieve_type = str(retrieve_type).lower()
        self.ucb_scale = float(ucb_scale)
        self.similarity_backend = str(similarity_backend).lower()
        self.model_name = model_name
        self.relevance_threshold = float(relevance_threshold)
        self.similarity_top_m = int(similarity_top_m)
        self.skill_format = str(skill_format or "retroagent").lower()
        self.category_filter = _as_bool(category_filter, default=False)
        self.retrieval_query_mode = str(retrieval_query_mode or "task").strip().lower()
        self.skill_mapping_dir = skill_mapping_dir
        self.skill_mapping = self._load_skill_mapping(skill_mapping_dir)
        self.task_to_skill = self.skill_mapping.get("task_to_skill", {})
        self.task_keywords = self.skill_mapping.get("task_keywords", {})
        self.valid_categories = set(self.task_to_skill.values()) | set(self.task_to_skill.keys())
        self.default_category = "other" if "other" in self.valid_categories else "unknown"
        self.merge_exact_task_attempt = _as_bool(merge_exact_task_attempt, default=True)
        if self.skill_format == "claude_style":
            self.merge_exact_task_attempt = False
        self.max_lessons_per_task_attempt = int(max_lessons_per_task_attempt)

        self.data: List[Dict[str, Any]] = []
        self._embedding_model = None
        self._task_embeddings = None
        self._total_count = 1
        self._retrieval_cache_generation = 0
        self._candidate_indices_cache: Dict[Tuple[int, str, str, str], Tuple[int, ...]] = {}
        self._candidate_index_buckets: Dict[Tuple[str, str], Dict[str, Tuple[int, ...]]] = {}
        self._query_embedding_cache: "OrderedDict[str, Any]" = OrderedDict()
        self._query_embedding_cache_limit = 2048

        self.load()

    def __len__(self) -> int:
        return len(self.data)

    def _load_skill_mapping(self, skill_mapping_dir: Optional[str]) -> Dict[str, Any]:
        if not skill_mapping_dir:
            return {}
        mapping_path = os.path.join(str(skill_mapping_dir), "skill_mapping.json")
        if not os.path.exists(mapping_path):
            print(f"[UCOB][skill_memory] skill_mapping.json not found: {mapping_path}")
            return {}
        with open(mapping_path, "r") as f:
            try:
                mapping = json.load(f)
            except json.JSONDecodeError:
                print(f"[UCOB][skill_memory] failed to parse skill mapping: {mapping_path}")
                return {}
        return mapping if isinstance(mapping, dict) else {}

    @staticmethod
    def _normalize_category_name(category: Any) -> str:
        return "_".join(str(category or "").strip().lower().split())

    def infer_task_category(self, task_description: str) -> str:
        text_lower = str(task_description or "").lower()
        for task_type, keywords in self.task_keywords.items():
            if not keywords:
                continue
            if any(str(keyword).lower() in text_lower for keyword in keywords):
                return self._normalize_category(task_type, task_description)
        return self.default_category

    def _normalize_category(self, category: Any, task_description: str = "") -> str:
        normalized = self._normalize_category_name(category)
        if normalized in self.task_to_skill:
            normalized = self._normalize_category_name(self.task_to_skill[normalized])
        if normalized and (not self.valid_categories or normalized in self.valid_categories):
            return normalized
        return self.infer_task_category(task_description)

    def _is_claude_style_item(self, item: Dict[str, Any]) -> bool:
        return item.get("skill_format") == "claude_style" or bool(item.get("title"))

    @staticmethod
    def _clean_field(value: Any, max_chars: int = 800) -> str:
        text = " ".join(str(value or "").strip().split())
        return text[:max_chars].strip()

    def _structured_reflection_text(
        self,
        title: str,
        principle: str,
        when_to_apply: str,
    ) -> str:
        return f"Skill: {title}\nPrinciple: {principle}\nWhen to apply: {when_to_apply}"

    @staticmethod
    def _memory_pool(item: Dict[str, Any]) -> str:
        return "state" if item.get("memory_type") == "state_skill" else "task"

    def _memory_embedding_text(self, item: Dict[str, Any]) -> str:
        if self._memory_pool(item) == "state":
            parts = [
                str(item.get("task_desc", "")),
                str(item.get("source_anchor_obs", "")),
                str(item.get("state_summary", "")),
            ]
            return "\n".join(part for part in parts if part.strip())
        return str(item.get("task_desc", ""))

    def _retrieval_query_text(
        self,
        task_description: str,
        task_category: Optional[str],
        state_context: Optional[str] = None,
        pool: str = "all",
    ) -> str:
        pool = str(pool or "all").strip().lower()
        if pool in {"task", "task_skill", "task_skills"}:
            return str(task_description)
        if pool in {"state", "step", "state_skill", "step_skill", "state_skills", "step_skills"}:
            parts = [str(task_description)]
            if state_context:
                parts.append(f"Current observation: {state_context}")
            return "\n".join(parts)
        if self.retrieval_query_mode in {"task_state", "state", "task_anchor"}:
            parts = [str(task_description)]
            if state_context:
                parts.append(f"Current observation: {state_context}")
            return "\n".join(parts)
        return str(task_description)

    def _normalize_loaded_entries(self) -> bool:
        changed = False
        for item in self.data:
            if not isinstance(item, dict):
                continue
            if not item.get("memory_id"):
                item["memory_id"] = f"mem_{uuid.uuid4().hex[:12]}"
                changed = True
            if self._is_claude_style_item(item):
                if item.get("skill_format") != "claude_style":
                    item["skill_format"] = "claude_style"
                    changed = True
                normalized_category = self._normalize_category(
                    item.get("task_category", ""),
                    item.get("task_desc", ""),
                )
                if item.get("task_category") != normalized_category:
                    item["task_category"] = normalized_category
                    changed = True
                if not item.get("reflection"):
                    item["reflection"] = self._structured_reflection_text(
                        item.get("title", ""),
                        item.get("principle", ""),
                        item.get("when_to_apply", ""),
                    )
                    changed = True
        return changed

    def load(self) -> None:
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                try:
                    loaded = json.load(f)
                except json.JSONDecodeError:
                    loaded = []
            if isinstance(loaded, list):
                self.data = loaded
            else:
                self.data = []
        else:
            self.data = []
        changed = self._normalize_loaded_entries()
        changed = self._coalesce_all_exact_task_attempt_entries() or changed
        if changed:
            self.save()
        self._recompute_total_count()
        self._rebuild_candidate_index_buckets()
        self._rebuild_embeddings()

    def save(self) -> None:
        dirname = os.path.dirname(self.filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def _load_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "algorithm.ucob.skill_memory.similarity_backend='sentence_transformer' "
                "requires sentence-transformers. Use 'lexical' to avoid this dependency."
            ) from exc
        self._embedding_model = SentenceTransformer(self.model_name)
        return self._embedding_model

    def _rebuild_embeddings(self) -> None:
        if self.similarity_backend != "sentence_transformer":
            self._task_embeddings = None
            return
        if not self.data:
            self._task_embeddings = None
            return
        model = self._load_embedding_model()
        texts = [self._memory_embedding_text(item) for item in self.data]
        self._task_embeddings = model.encode(texts, convert_to_tensor=True)

    def _recompute_total_count(self) -> None:
        self._total_count = max(sum(int(item.get("count", 1)) for item in self.data), 1)

    def _invalidate_retrieval_cache(self) -> None:
        self._retrieval_cache_generation += 1
        self._candidate_indices_cache.clear()
        self._query_embedding_cache.clear()

    def _append_candidate_index(self, item: Dict[str, Any], idx: int) -> None:
        if self.skill_format == "claude_style" and not self._is_claude_style_item(item):
            return

        pool = self._memory_pool(item)
        if pool not in {"task", "state"}:
            pool = "task"
        category = self._normalize_category(item.get("task_category", ""), item.get("task_desc", "")) if self.category_filter else ""
        attempt_type = str(item.get("attempt_type", "unknown"))

        for selector in ("all", pool):
            bucket = self._candidate_index_buckets.setdefault((selector, category), {})
            for filter_key in ("both", attempt_type):
                indices = list(bucket.get(filter_key, ()))
                indices.append(idx)
                bucket[filter_key] = tuple(indices)

    def _rebuild_candidate_index_buckets(self) -> None:
        self._candidate_index_buckets = {}
        for idx, item in enumerate(self.data):
            self._append_candidate_index(item, idx)

    def _ensure_task_embeddings(self) -> bool:
        if self.similarity_backend != "sentence_transformer":
            return False
        if self._task_embeddings is None or len(self._task_embeddings) != len(self.data):
            self._rebuild_embeddings()
        return self._task_embeddings is not None

    def _cache_query_embedding(self, query: str, embedding: Any) -> None:
        if self.similarity_backend != "sentence_transformer":
            return
        if getattr(embedding, "ndim", 1) > 1 and getattr(embedding, "shape", [0])[0] == 1:
            embedding = embedding[0]
        cached = embedding.detach().cpu()
        self._query_embedding_cache[query] = cached
        self._query_embedding_cache.move_to_end(query)
        while len(self._query_embedding_cache) > self._query_embedding_cache_limit:
            self._query_embedding_cache.popitem(last=False)

    def _query_embeddings_with_cache(self, queries: List[str]):
        if self.similarity_backend != "sentence_transformer" or not queries:
            return []

        model = self._load_embedding_model()
        results: List[Any] = [None] * len(queries)
        missing_queries: List[str] = []
        missing_positions: List[int] = []

        for idx, query in enumerate(queries):
            cached = self._query_embedding_cache.get(query)
            if cached is None:
                missing_queries.append(query)
                missing_positions.append(idx)
            else:
                self._query_embedding_cache.move_to_end(query)
                results[idx] = cached

        if missing_queries:
            encoded = model.encode(missing_queries, convert_to_tensor=True)
            if getattr(encoded, "ndim", 1) <= 1:
                encoded_rows = [encoded]
            else:
                encoded_rows = list(encoded)
            for query, pos, embedding in zip(missing_queries, missing_positions, encoded_rows):
                self._cache_query_embedding(query, embedding)
                results[pos] = self._query_embedding_cache[query]

        return results

    def _append_entry_embedding(self, item: Dict[str, Any]) -> None:
        if self.similarity_backend != "sentence_transformer":
            return
        if self._task_embeddings is None:
            self._rebuild_embeddings()
            return
        if len(self._task_embeddings) != len(self.data) - 1:
            self._rebuild_embeddings()
            return

        import torch

        model = self._load_embedding_model()
        new_embedding = model.encode([self._memory_embedding_text(item)], convert_to_tensor=True)
        new_embedding = new_embedding.to(self._task_embeddings.device)
        self._task_embeddings = torch.cat((self._task_embeddings, new_embedding), dim=0)

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())

    @staticmethod
    def _normalize_task(text: str) -> str:
        return " ".join(str(text).strip().lower().split())

    def _merge_reflection_text(self, existing: str, new_text: str) -> str:
        lessons = []
        seen = set()
        for text in (existing, new_text):
            for lesson in str(text).split(" | "):
                lesson = lesson.strip()
                if not lesson:
                    continue
                key = " ".join(lesson.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                lessons.append(lesson)

        if self.max_lessons_per_task_attempt > 0:
            lessons = lessons[-self.max_lessons_per_task_attempt :]
        return " | ".join(lessons)

    def _merge_item_into_primary(self, primary: Dict[str, Any], item: Dict[str, Any]) -> None:
        if not primary.get("memory_id"):
            primary["memory_id"] = f"mem_{uuid.uuid4().hex[:12]}"

        primary_count = max(int(primary.get("count", 1)), 1)
        item_count = max(int(item.get("count", 1)), 1)
        combined_count = primary_count + item_count
        primary_utility = float(primary.get("utility_score", 0.5))
        item_utility = float(item.get("utility_score", 0.5))

        primary["utility_score"] = (
            (primary_utility * primary_count) + (item_utility * item_count)
        ) / max(combined_count, 1)
        primary["count"] = combined_count
        primary["reflection"] = self._merge_reflection_text(
            primary.get("reflection", ""),
            item.get("reflection", ""),
        )
        if item.get("trajectory"):
            primary["trajectory"] = item.get("trajectory", "")
        primary["created_at_progress"] = max(
            float(primary.get("created_at_progress", 0.0)),
            float(item.get("created_at_progress", 0.0)),
        )
        if item.get("updated_at_progress") is not None:
            primary["updated_at_progress"] = max(
                float(primary.get("updated_at_progress", primary.get("created_at_progress", 0.0))),
                float(item.get("updated_at_progress", 0.0)),
            )

    def _coalesce_all_exact_task_attempt_entries(self) -> bool:
        if not self.merge_exact_task_attempt or not self.data:
            return False

        groups = {}
        duplicate_ids = set()
        changed = False
        for item in self.data:
            attempt_type = item.get("attempt_type", "unknown")
            if attempt_type == "unknown":
                continue
            key = (self._normalize_task(item.get("task_desc", "")), attempt_type)
            if key in groups:
                self._merge_item_into_primary(groups[key], item)
                duplicate_ids.add(id(item))
                changed = True
            else:
                if not item.get("memory_id"):
                    item["memory_id"] = f"mem_{uuid.uuid4().hex[:12]}"
                    changed = True
                groups[key] = item

        if duplicate_ids:
            self.data = [item for item in self.data if id(item) not in duplicate_ids]
        return changed

    def _coalesce_exact_task_attempt_entries(
        self,
        task_description: str,
        attempt_type: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.merge_exact_task_attempt or attempt_type == "unknown":
            return None

        normalized_task = self._normalize_task(task_description)
        matches = [
            item
            for item in self.data
            if self._normalize_task(item.get("task_desc", "")) == normalized_task
            and item.get("attempt_type", "unknown") == attempt_type
        ]
        if not matches:
            return None

        primary = matches[0]
        if not primary.get("memory_id"):
            primary["memory_id"] = f"mem_{uuid.uuid4().hex[:12]}"

        if len(matches) > 1:
            for item in matches[1:]:
                self._merge_item_into_primary(primary, item)
            duplicate_ids = {id(item) for item in matches[1:]}
            self.data = [item for item in self.data if id(item) not in duplicate_ids]

        return primary

    def _lexical_similarity(self, text_a: str, text_b: str) -> float:
        tokens_a = Counter(self._tokens(text_a))
        tokens_b = Counter(self._tokens(text_b))
        if not tokens_a or not tokens_b:
            return 0.0
        overlap = sum(min(tokens_a[tok], tokens_b[tok]) for tok in tokens_a.keys() & tokens_b.keys())
        norm = math.sqrt(sum(v * v for v in tokens_a.values())) * math.sqrt(sum(v * v for v in tokens_b.values()))
        if norm <= 0:
            return 0.0
        return float(overlap / norm)

    def _candidate_indices(
        self,
        task_category: Optional[str] = None,
        filter_type: str = "both",
        pool: str = "all",
    ) -> List[int]:
        cache_key = (
            self._retrieval_cache_generation,
            self._normalize_category_name(task_category),
            str(filter_type or "both").strip().lower(),
            str(pool or "all").strip().lower(),
        )
        cached = self._candidate_indices_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        normalized_pool = str(pool or "all").strip().lower()
        if normalized_pool in {"state", "step", "state_skill", "step_skill", "state_skills", "step_skills"}:
            normalized_pool = "state"
        elif normalized_pool in {"task", "task_skill", "task_skills"}:
            normalized_pool = "task"
        else:
            normalized_pool = "all"

        normalized_query_category = self._normalize_category_name(task_category) if self.category_filter else ""
        bucket = self._candidate_index_buckets.get((normalized_pool, normalized_query_category))
        if not bucket:
            self._candidate_indices_cache[cache_key] = tuple()
            return []

        if filter_type == "both":
            indices = list(bucket.get("both", ()))
        else:
            indices = list(bucket.get(str(filter_type), ()))
        self._candidate_indices_cache[cache_key] = tuple(indices)
        return indices

    def _similarities(self, query: str, candidate_indices: Optional[List[int]] = None) -> List[float]:
        if not self.data:
            return []
        if candidate_indices is None:
            candidate_indices = list(range(len(self.data)))
        if not candidate_indices:
            return []
        if self.similarity_backend == "sentence_transformer":
            if not self._ensure_task_embeddings():
                return []
            from sentence_transformers import util

            model = self._load_embedding_model()
            query_embedding = self._query_embedding_cache.get(query)
            if query_embedding is None:
                query_embedding = model.encode(query, convert_to_tensor=True)
                self._cache_query_embedding(query, query_embedding)
                query_embedding = self._query_embedding_cache[query]
            candidate_embeddings = self._task_embeddings[candidate_indices]
            if query_embedding.device != candidate_embeddings.device:
                query_embedding = query_embedding.to(candidate_embeddings.device)
            return util.cos_sim(query_embedding, candidate_embeddings)[0].detach().cpu().tolist()
        return [
            self._lexical_similarity(query, self._memory_embedding_text(self.data[idx]))
            for idx in candidate_indices
        ]

    def _format_candidate(
        self,
        item: Dict[str, Any],
        attempt_type: str,
        score: float,
        relevance: float,
        utility: Optional[float] = None,
        memory_branch: str = "good",
    ) -> Dict[str, Any]:
        if utility is None:
            utility = float(item.get("utility_score", 0.5))
        memory_pool = self._memory_pool(item)
        return {
            "memory_id": item.get("memory_id", ""),
            "text": item.get("reflection", ""),
            "type": attempt_type,
            "score": float(score),
            "relevance": float(relevance),
            "utility_score": float(utility),
            "source": item.get("source", "agent_reflection"),
            "skill_format": item.get("skill_format", "retroagent"),
            "task_category": item.get("task_category", ""),
            "title": item.get("title", ""),
            "principle": item.get("principle", ""),
            "when_to_apply": item.get("when_to_apply", ""),
            "memory_type": item.get("memory_type", ""),
            "memory_pool": memory_pool,
            "memory_branch": memory_branch,
        }

    def _rank_candidates(
        self,
        similarities: List[float],
        candidate_indices: Optional[List[int]] = None,
        top_k: int = 3,
        filter_type: str = "both",
    ) -> List[Dict[str, Any]]:
        total_count = self._total_count
        candidate_rows = []
        if candidate_indices is None:
            candidate_indices = list(range(len(self.data)))

        for sim_idx, idx in enumerate(candidate_indices):
            if sim_idx >= len(similarities) or idx >= len(self.data):
                continue
            item = self.data[idx]
            attempt_type = item.get("attempt_type", "unknown")
            if filter_type != "both" and attempt_type != filter_type:
                continue

            relevance = float(similarities[sim_idx])
            if relevance < self.relevance_threshold:
                continue

            candidate_rows.append((idx, item, attempt_type, relevance))

        if self.similarity_top_m > 0 and len(candidate_rows) > self.similarity_top_m:
            candidate_rows.sort(key=lambda row: row[3], reverse=True)
            candidate_rows = candidate_rows[: self.similarity_top_m]

        candidates: List[Dict[str, Any]] = []
        for idx, item, attempt_type, relevance in candidate_rows:
            utility = float(item.get("utility_score", 0.5))
            count = max(int(item.get("count", 1)), 1)
            if self.retrieve_type == "relevance_only":
                score = relevance
            elif self.retrieve_type == "ucb":
                exploration_bonus = self.ucb_scale * math.sqrt(math.log(total_count) / count)
                score = (self.alpha * relevance) + ((1.0 - self.alpha) * (utility + exploration_bonus))
            else:
                score = (self.alpha * relevance) + ((1.0 - self.alpha) * utility)

            candidates.append(
                self._format_candidate(
                    item=item,
                    attempt_type=attempt_type,
                    score=score,
                    relevance=relevance,
                    utility=utility,
                    memory_branch="good",
                )
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[: max(int(top_k), 0)]

    def _rank_low_similarity_candidates(
        self,
        similarities: List[float],
        candidate_indices: Optional[List[int]] = None,
        top_k: int = 3,
        filter_type: str = "both",
    ) -> List[Dict[str, Any]]:
        if candidate_indices is None:
            candidate_indices = list(range(len(self.data)))

        candidates: List[Dict[str, Any]] = []
        for sim_idx, idx in enumerate(candidate_indices):
            if sim_idx >= len(similarities) or idx >= len(self.data):
                continue
            item = self.data[idx]
            attempt_type = item.get("attempt_type", "unknown")
            if filter_type != "both" and attempt_type != filter_type:
                continue

            relevance = float(similarities[sim_idx])
            candidates.append(
                self._format_candidate(
                    item=item,
                    attempt_type=attempt_type,
                    score=relevance,
                    relevance=relevance,
                    memory_branch="bad",
                )
            )

        candidates.sort(key=lambda x: x["relevance"])
        return candidates[: max(int(top_k), 0)]

    def retrieve(
        self,
        current_task_description: str,
        top_k: int = 3,
        filter_type: str = "both",
        state_context: Optional[str] = None,
        pool: str = "all",
    ) -> List[Dict[str, Any]]:
        if not self.data:
            return []

        task_category = self.infer_task_category(current_task_description) if self.category_filter else None
        candidate_indices = self._candidate_indices(
            task_category=task_category,
            filter_type=filter_type,
            pool=pool,
        )
        query = self._retrieval_query_text(
            current_task_description,
            task_category,
            state_context=state_context,
            pool=pool,
        )
        similarities = self._similarities(query, candidate_indices=candidate_indices)
        return self._rank_candidates(
            similarities=similarities,
            candidate_indices=candidate_indices,
            top_k=top_k,
            filter_type=filter_type,
        )

    def retrieve_many(
        self,
        current_task_descriptions: List[str],
        top_k: int = 3,
        filter_type: str = "both",
        state_contexts: Optional[List[str]] = None,
        pool: str = "all",
    ) -> List[List[Dict[str, Any]]]:
        if not current_task_descriptions:
            return []
        if not self.data:
            return [[] for _ in current_task_descriptions]
        if state_contexts is None:
            state_contexts = [None] * len(current_task_descriptions)
        else:
            state_contexts = list(state_contexts)[: len(current_task_descriptions)]
            if len(state_contexts) < len(current_task_descriptions):
                state_contexts.extend([None] * (len(current_task_descriptions) - len(state_contexts)))

        unique_keys = []
        representative_tasks: Dict[Tuple[str, str], str] = {}
        representative_categories: Dict[Tuple[str, str], Optional[str]] = {}
        representative_contexts: Dict[Tuple[str, str, str], Optional[str]] = {}
        task_keys = []
        for task, state_context in zip(current_task_descriptions, state_contexts):
            task_category = self.infer_task_category(task) if self.category_filter else None
            context_key = self._normalize_task(state_context or "")
            key = (self._normalize_task(task), task_category or "", context_key, str(pool or "all"))
            task_keys.append(key)
            if key not in representative_tasks:
                representative_tasks[key] = task
                representative_categories[key] = task_category
                representative_contexts[key] = state_context
                unique_keys.append(key)

        retrieved_by_key: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
        if self.similarity_backend == "sentence_transformer":
            if not self._ensure_task_embeddings():
                return [[] for _ in current_task_descriptions]
            from sentence_transformers import util

            model = self._load_embedding_model()
            queries = [
                self._retrieval_query_text(
                    representative_tasks[key],
                    representative_categories[key],
                    state_context=representative_contexts[key],
                    pool=pool,
                )
                for key in unique_keys
            ]
            query_embeddings = self._query_embeddings_with_cache(queries)
            for query_idx, key in enumerate(unique_keys):
                task_category = representative_categories[key]
                candidate_indices = self._candidate_indices(
                    task_category=task_category,
                    filter_type=filter_type,
                    pool=pool,
                )
                if not candidate_indices:
                    retrieved_by_key[key] = []
                    continue
                candidate_embeddings = self._task_embeddings[candidate_indices]
                query_embedding = query_embeddings[query_idx]
                if query_embedding.device != candidate_embeddings.device:
                    query_embedding = query_embedding.to(candidate_embeddings.device)
                similarities = util.cos_sim(
                    query_embedding.unsqueeze(0),
                    candidate_embeddings,
                )[0].detach().cpu().tolist()
                retrieved_by_key[key] = self._rank_candidates(
                    similarities=similarities,
                    candidate_indices=candidate_indices,
                    top_k=top_k,
                    filter_type=filter_type,
                )
        else:
            for key in unique_keys:
                task = representative_tasks[key]
                task_category = representative_categories[key]
                candidate_indices = self._candidate_indices(
                    task_category=task_category,
                    filter_type=filter_type,
                    pool=pool,
                )
                query = self._retrieval_query_text(
                    task,
                    task_category,
                    state_context=representative_contexts[key],
                    pool=pool,
                )
                similarities = self._similarities(query, candidate_indices=candidate_indices)
                retrieved_by_key[key] = self._rank_candidates(
                    similarities=similarities,
                    candidate_indices=candidate_indices,
                    top_k=top_k,
                    filter_type=filter_type,
                )

        return [retrieved_by_key.get(key, []) for key in task_keys]

    def retrieve_many_dual_pool(
        self,
        current_task_descriptions: List[str],
        task_top_k: int = 1,
        state_top_k: int = 2,
        filter_type: str = "both",
        state_contexts: Optional[List[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        task_results = self.retrieve_many(
            current_task_descriptions,
            top_k=task_top_k,
            filter_type=filter_type,
            state_contexts=state_contexts,
            pool="task",
        ) if task_top_k > 0 else [[] for _ in current_task_descriptions]
        state_results = self.retrieve_many(
            current_task_descriptions,
            top_k=state_top_k,
            filter_type=filter_type,
            state_contexts=state_contexts,
            pool="state",
        ) if state_top_k > 0 else [[] for _ in current_task_descriptions]
        return [
            list(task_items) + list(state_items)
            for task_items, state_items in zip(task_results, state_results)
        ]

    def retrieve_bad_many(
        self,
        current_task_descriptions: List[str],
        top_k: int = 3,
        filter_type: str = "both",
        state_contexts: Optional[List[str]] = None,
        pool: str = "all",
    ) -> List[List[Dict[str, Any]]]:
        if not current_task_descriptions:
            return []
        if not self.data:
            return [[] for _ in current_task_descriptions]
        if state_contexts is None:
            state_contexts = [None] * len(current_task_descriptions)
        else:
            state_contexts = list(state_contexts)[: len(current_task_descriptions)]
            if len(state_contexts) < len(current_task_descriptions):
                state_contexts.extend([None] * (len(current_task_descriptions) - len(state_contexts)))

        unique_keys = []
        representative_tasks: Dict[Tuple[str, str], str] = {}
        representative_categories: Dict[Tuple[str, str], Optional[str]] = {}
        representative_contexts: Dict[Tuple[str, str, str], Optional[str]] = {}
        task_keys = []
        for task, state_context in zip(current_task_descriptions, state_contexts):
            task_category = self.infer_task_category(task) if self.category_filter else None
            context_key = self._normalize_task(state_context or "")
            key = (self._normalize_task(task), task_category or "", context_key, str(pool or "all"))
            task_keys.append(key)
            if key not in representative_tasks:
                representative_tasks[key] = task
                representative_categories[key] = task_category
                representative_contexts[key] = state_context
                unique_keys.append(key)

        retrieved_by_key: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
        if self.similarity_backend == "sentence_transformer":
            if not self._ensure_task_embeddings():
                return [[] for _ in current_task_descriptions]
            from sentence_transformers import util

            model = self._load_embedding_model()
            queries = [
                self._retrieval_query_text(
                    representative_tasks[key],
                    representative_categories[key],
                    state_context=representative_contexts[key],
                    pool=pool,
                )
                for key in unique_keys
            ]
            query_embeddings = self._query_embeddings_with_cache(queries)
            for query_idx, key in enumerate(unique_keys):
                task_category = representative_categories[key]
                candidate_indices = self._candidate_indices(
                    task_category=task_category,
                    filter_type=filter_type,
                    pool=pool,
                )
                if not candidate_indices:
                    retrieved_by_key[key] = []
                    continue
                candidate_embeddings = self._task_embeddings[candidate_indices]
                query_embedding = query_embeddings[query_idx]
                if query_embedding.device != candidate_embeddings.device:
                    query_embedding = query_embedding.to(candidate_embeddings.device)
                similarities = util.cos_sim(
                    query_embedding.unsqueeze(0),
                    candidate_embeddings,
                )[0].detach().cpu().tolist()
                retrieved_by_key[key] = self._rank_low_similarity_candidates(
                    similarities=similarities,
                    candidate_indices=candidate_indices,
                    top_k=top_k,
                    filter_type=filter_type,
                )
        else:
            for key in unique_keys:
                task = representative_tasks[key]
                task_category = representative_categories[key]
                candidate_indices = self._candidate_indices(
                    task_category=task_category,
                    filter_type=filter_type,
                    pool=pool,
                )
                query = self._retrieval_query_text(
                    task,
                    task_category,
                    state_context=representative_contexts[key],
                    pool=pool,
                )
                similarities = self._similarities(query, candidate_indices=candidate_indices)
                retrieved_by_key[key] = self._rank_low_similarity_candidates(
                    similarities=similarities,
                    candidate_indices=candidate_indices,
                    top_k=top_k,
                    filter_type=filter_type,
                )

        return [retrieved_by_key.get(key, []) for key in task_keys]

    def retrieve_bad_many_dual_pool(
        self,
        current_task_descriptions: List[str],
        task_top_k: int = 1,
        state_top_k: int = 2,
        filter_type: str = "both",
        state_contexts: Optional[List[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        task_results = self.retrieve_bad_many(
            current_task_descriptions,
            top_k=task_top_k,
            filter_type=filter_type,
            state_contexts=state_contexts,
            pool="task",
        ) if task_top_k > 0 else [[] for _ in current_task_descriptions]
        state_results = self.retrieve_bad_many(
            current_task_descriptions,
            top_k=state_top_k,
            filter_type=filter_type,
            state_contexts=state_contexts,
            pool="state",
        ) if state_top_k > 0 else [[] for _ in current_task_descriptions]
        return [
            list(task_items) + list(state_items)
            for task_items, state_items in zip(task_results, state_results)
        ]

    def retrieve_good_bad_many(
        self,
        current_task_descriptions: List[str],
        top_k: int = 3,
        filter_type: str = "both",
        state_contexts: Optional[List[str]] = None,
        pool: str = "all",
    ) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
        if not current_task_descriptions:
            return [], []
        if not self.data:
            empty = [[] for _ in current_task_descriptions]
            return empty, [[] for _ in current_task_descriptions]
        if state_contexts is None:
            state_contexts = [None] * len(current_task_descriptions)
        else:
            state_contexts = list(state_contexts)[: len(current_task_descriptions)]
            if len(state_contexts) < len(current_task_descriptions):
                state_contexts.extend([None] * (len(current_task_descriptions) - len(state_contexts)))

        unique_keys = []
        representative_tasks: Dict[Tuple[str, str], str] = {}
        representative_categories: Dict[Tuple[str, str], Optional[str]] = {}
        representative_contexts: Dict[Tuple[str, str, str], Optional[str]] = {}
        task_keys = []
        for task, state_context in zip(current_task_descriptions, state_contexts):
            task_category = self.infer_task_category(task) if self.category_filter else None
            context_key = self._normalize_task(state_context or "")
            key = (self._normalize_task(task), task_category or "", context_key, str(pool or "all"))
            task_keys.append(key)
            if key not in representative_tasks:
                representative_tasks[key] = task
                representative_categories[key] = task_category
                representative_contexts[key] = state_context
                unique_keys.append(key)

        good_by_key: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
        bad_by_key: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}

        if self.similarity_backend == "sentence_transformer":
            if not self._ensure_task_embeddings():
                empty = [[] for _ in current_task_descriptions]
                return empty, [[] for _ in current_task_descriptions]
            from sentence_transformers import util

            model = self._load_embedding_model()
            queries = [
                self._retrieval_query_text(
                    representative_tasks[key],
                    representative_categories[key],
                    state_context=representative_contexts[key],
                    pool=pool,
                )
                for key in unique_keys
            ]
            query_embeddings = self._query_embeddings_with_cache(queries)
            for query_idx, key in enumerate(unique_keys):
                task_category = representative_categories[key]
                candidate_indices = self._candidate_indices(
                    task_category=task_category,
                    filter_type=filter_type,
                    pool=pool,
                )
                if not candidate_indices:
                    good_by_key[key] = []
                    bad_by_key[key] = []
                    continue
                candidate_embeddings = self._task_embeddings[candidate_indices]
                query_embedding = query_embeddings[query_idx]
                if query_embedding.device != candidate_embeddings.device:
                    query_embedding = query_embedding.to(candidate_embeddings.device)
                similarities = util.cos_sim(
                    query_embedding.unsqueeze(0),
                    candidate_embeddings,
                )[0].detach().cpu().tolist()
                good_by_key[key] = self._rank_candidates(
                    similarities=similarities,
                    candidate_indices=candidate_indices,
                    top_k=top_k,
                    filter_type=filter_type,
                )
                bad_by_key[key] = self._rank_low_similarity_candidates(
                    similarities=similarities,
                    candidate_indices=candidate_indices,
                    top_k=top_k,
                    filter_type=filter_type,
                )
        else:
            for key in unique_keys:
                task = representative_tasks[key]
                task_category = representative_categories[key]
                candidate_indices = self._candidate_indices(
                    task_category=task_category,
                    filter_type=filter_type,
                    pool=pool,
                )
                query = self._retrieval_query_text(
                    task,
                    task_category,
                    state_context=representative_contexts[key],
                    pool=pool,
                )
                similarities = self._similarities(query, candidate_indices=candidate_indices)
                good_by_key[key] = self._rank_candidates(
                    similarities=similarities,
                    candidate_indices=candidate_indices,
                    top_k=top_k,
                    filter_type=filter_type,
                )
                bad_by_key[key] = self._rank_low_similarity_candidates(
                    similarities=similarities,
                    candidate_indices=candidate_indices,
                    top_k=top_k,
                    filter_type=filter_type,
                )

        good = [good_by_key.get(key, []) for key in task_keys]
        bad = [bad_by_key.get(key, []) for key in task_keys]
        return good, bad

    def retrieve_good_bad_many_dual_pool(
        self,
        current_task_descriptions: List[str],
        task_top_k: int = 1,
        state_top_k: int = 2,
        filter_type: str = "both",
        state_contexts: Optional[List[str]] = None,
    ) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
        task_good, task_bad = self.retrieve_good_bad_many(
            current_task_descriptions,
            top_k=task_top_k,
            filter_type=filter_type,
            state_contexts=state_contexts,
            pool="task",
        ) if task_top_k > 0 else (
            [[] for _ in current_task_descriptions],
            [[] for _ in current_task_descriptions],
        )
        state_good, state_bad = self.retrieve_good_bad_many(
            current_task_descriptions,
            top_k=state_top_k,
            filter_type=filter_type,
            state_contexts=state_contexts,
            pool="state",
        ) if state_top_k > 0 else (
            [[] for _ in current_task_descriptions],
            [[] for _ in current_task_descriptions],
        )
        good = [
            list(task_items) + list(state_items)
            for task_items, state_items in zip(task_good, state_good)
        ]
        bad = [
            list(task_items) + list(state_items)
            for task_items, state_items in zip(task_bad, state_bad)
        ]
        return good, bad

    def format_retrieved(self, retrieved: List[Dict[str, Any]]) -> str:
        structured = [
            item for item in retrieved
            if item.get("skill_format") == "claude_style" or item.get("title")
        ]
        if structured:
            pools = {
                "Task-level skills": [
                    item for item in structured if item.get("memory_pool", "task") != "state"
                ],
                "Step-level skills": [
                    item for item in structured if item.get("memory_pool", "task") == "state"
                ],
            }
            should_group_by_pool = bool(pools["Step-level skills"])
            lines = ["Relevant category-specific skills:"]

            def append_skill_lines(items: List[Dict[str, Any]]) -> None:
                for item in items:
                    title = self._clean_field(item.get("title", "Untitled Skill"), max_chars=120)
                    principle = self._clean_field(item.get("principle", ""), max_chars=600)
                    when_to_apply = self._clean_field(item.get("when_to_apply", ""), max_chars=400)
                    lines.append(f"- Skill: {title}")
                    if principle:
                        lines.append(f"  Principle: {principle}")
                    if when_to_apply:
                        lines.append(f"  When to apply: {when_to_apply}")

            if should_group_by_pool:
                for section_name, items in pools.items():
                    if not items:
                        continue
                    lines.append(f"{section_name}:")
                    append_skill_lines(items)
            else:
                append_skill_lines(structured)

            lines.append("Use these skills only when they fit the current task category and observation.")
            return "\n".join(lines)

        lines = [item.get("text", "").strip() for item in retrieved if item.get("text", "").strip()]
        if not lines:
            return ""
        return (
            "Past reflections on similar tasks:\n"
            + "\n".join(lines)
            + "\nWarning: These lessons may be outdated. Use them only if they align with your current observation."
        )

    def add(
        self,
        task_description: str,
        reflection_text: str,
        trajectory: str,
        initial_score: float = 0.5,
        attempt_type: str = "unknown",
        current_progress_ratio: float = 0.0,
        source: str = "agent_reflection",
        metadata: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> str:
        reflection_text = str(reflection_text).strip()
        if not reflection_text:
            return ""

        for item in self.data:
            if item.get("reflection", "") == reflection_text:
                if self._lexical_similarity(item.get("task_desc", ""), task_description) > 0.85:
                    old_attempt_type = item.get("attempt_type", "unknown")
                    self._update_score_internal(item, initial_score)
                    if item.get("attempt_type", "unknown") == "unknown":
                        item["attempt_type"] = attempt_type
                    self._total_count += 1
                    if item.get("attempt_type", "unknown") != old_attempt_type:
                        self._rebuild_candidate_index_buckets()
                    self._invalidate_retrieval_cache()
                    if save:
                        self.save()
                    return item.get("memory_id", "")

        exact_task_entry = self._coalesce_exact_task_attempt_entries(task_description, attempt_type)
        if exact_task_entry is not None:
            exact_task_entry["task_desc"] = task_description
            exact_task_entry["reflection"] = self._merge_reflection_text(
                exact_task_entry.get("reflection", ""),
                reflection_text,
            )
            if trajectory:
                exact_task_entry["trajectory"] = trajectory
            exact_task_entry["source"] = source
            exact_task_entry["updated_at_progress"] = float(current_progress_ratio)
            self._update_score_internal(exact_task_entry, initial_score)
            self._total_count += 1
            self._invalidate_retrieval_cache()
            if save:
                self.save()
            return exact_task_entry.get("memory_id", "")

        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        new_entry = {
            "memory_id": memory_id,
            "task_desc": task_description,
            "reflection": reflection_text,
            "trajectory": trajectory,
            "utility_score": float(initial_score),
            "count": 1,
            "attempt_type": attempt_type,
            "created_at_progress": float(current_progress_ratio),
            "source": source,
        }
        if metadata:
            for key, value in metadata.items():
                if value is not None:
                    new_entry[key] = value
        self.data.append(new_entry)
        self._append_entry_embedding(new_entry)
        self._total_count += 1
        self._append_candidate_index(new_entry, len(self.data) - 1)
        if save:
            self.save()
        self._invalidate_retrieval_cache()
        return memory_id

    def add_skill(
        self,
        task_description: str,
        task_category: str,
        title: str,
        principle: str,
        when_to_apply: str,
        evidence: str,
        trajectory: str,
        initial_score: float = 0.5,
        attempt_type: str = "unknown",
        current_progress_ratio: float = 0.0,
        source: str = "agent_skill_reflection",
        metadata: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> str:
        title = self._clean_field(title, max_chars=160)
        principle = self._clean_field(principle, max_chars=900)
        when_to_apply = self._clean_field(when_to_apply, max_chars=600)
        evidence = self._clean_field(evidence, max_chars=600)
        task_category = self._normalize_category(task_category, task_description)
        if len(title) < 2 or len(principle) < 8 or len(when_to_apply) < 8:
            return ""

        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        new_entry = {
            "memory_id": memory_id,
            "skill_format": "claude_style",
            "task_category": task_category,
            "task_desc": task_description,
            "title": title,
            "principle": principle,
            "when_to_apply": when_to_apply,
            "evidence": evidence,
            "reflection": self._structured_reflection_text(
                title=title,
                principle=principle,
                when_to_apply=when_to_apply,
            ),
            "trajectory": trajectory,
            "utility_score": float(initial_score),
            "count": 1,
            "attempt_type": attempt_type,
            "created_at_progress": float(current_progress_ratio),
            "source": source,
        }
        if metadata:
            for key, value in metadata.items():
                if value is not None:
                    new_entry[key] = value

        self.data.append(new_entry)
        self._append_entry_embedding(new_entry)
        self._total_count += 1
        self._append_candidate_index(new_entry, len(self.data) - 1)
        self._invalidate_retrieval_cache()
        if save:
            self.save()
        return memory_id

    def _update_score_internal(self, item: Dict[str, Any], new_score: float, beta: Optional[float] = None) -> None:
        current_avg = float(item.get("utility_score", 0.5))
        update_beta = self.beta if beta is None else float(beta)
        item["utility_score"] = ((1.0 - update_beta) * current_avg) + (update_beta * float(new_score))
        item["count"] = int(item.get("count", 1)) + 1

    def update_utility_by_ids(self, memory_ids: List[str], score: float, beta: Optional[float] = None) -> int:
        return self.update_utility_by_id_scores([(mid, score) for mid in memory_ids if mid], beta=beta)

    def update_utility_by_id_scores(self, updates: List[Tuple[str, float]], beta: Optional[float] = None) -> int:
        updates = [(mid, score) for mid, score in updates if mid]
        if not updates:
            return 0
        memory_by_id = {item.get("memory_id", ""): item for item in self.data}
        updated = 0
        for memory_id, score in updates:
            item = memory_by_id.get(memory_id)
            if item is not None:
                self._update_score_internal(item, score, beta=beta)
                updated += 1
        if updated:
            self._recompute_total_count()
            self.save()
        return updated

    def memory_pool_by_id(self, memory_id: str) -> str:
        for item in self.data:
            if item.get("memory_id", "") == memory_id:
                return self._memory_pool(item)
        return ""

    def get_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        for item in self.data:
            if item.get("memory_id", "") == memory_id:
                return item
        return None

    def retrieve_trajectory(self, current_task_description: str, target_type: str = "success") -> Optional[str]:
        for item in reversed(self.data):
            if item.get("task_desc") == current_task_description and item.get("attempt_type") == target_type:
                trajectory = item.get("trajectory", "")
                if trajectory:
                    return trajectory
        return None

    @staticmethod
    def _extract_json(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
        content = str(text)
        if "</think>" in content:
            content = content.split("</think>")[-1]

        reflection_match = re.search(r"<reflection>\s*(\{.*?\})\s*</reflection>", content, re.DOTALL)
        if reflection_match:
            json_str = reflection_match.group(1)
        else:
            code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                start_idx = content.find("{")
                end_idx = content.rfind("}")
                if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                    return None, ""
                json_str = content[start_idx : end_idx + 1]

        try:
            return json.loads(json_str), json_str
        except json.JSONDecodeError:
            return None, json_str

    def add_from_reflection_output(
        self,
        task_description: str,
        reflection_output: str,
        trajectory: str,
        actual_success: bool,
        current_progress_ratio: float = 0.0,
        validate_success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        initial_score: float = 0.5,
        save: bool = True,
    ) -> Tuple[bool, str]:
        reflection_data, _ = self._extract_json(reflection_output)
        if not isinstance(reflection_data, dict):
            return False, "parse_failed"

        predicted_success = reflection_data.get("task_success", False)
        if isinstance(predicted_success, str):
            predicted_success = predicted_success.strip().lower() in {"true", "1", "yes", "success", "successful"}
        predicted_success = bool(predicted_success)
        if validate_success and predicted_success != bool(actual_success):
            return False, "success_mismatch"

        title = reflection_data.get("title")
        principle = reflection_data.get("principle")
        when_to_apply = reflection_data.get("when_to_apply")
        if self.skill_format == "claude_style" or any([title, principle, when_to_apply]):
            if not (title and principle and when_to_apply):
                return False, "empty_skill"
            task_category = self._normalize_category(
                reflection_data.get("task_category", ""),
                task_description,
            )
            memory_id = self.add_skill(
                task_description=task_description,
                task_category=task_category,
                title=title,
                principle=principle,
                when_to_apply=when_to_apply,
                evidence=reflection_data.get("evidence", ""),
                trajectory=trajectory,
                initial_score=initial_score,
                attempt_type="success" if actual_success else "failure",
                current_progress_ratio=current_progress_ratio,
                metadata=metadata,
                save=save,
            )
            return bool(memory_id), memory_id or "empty_skill"

        lessons = []
        action_lesson = reflection_data.get("action_lesson")
        navigation_lesson = reflection_data.get("navigation_lesson")
        environment_lesson = reflection_data.get("environment_lesson")
        lesson = reflection_data.get("lesson")

        if action_lesson and len(str(action_lesson).strip()) > 5:
            lessons.append(f"Action Insight: {str(action_lesson).strip()}")
        if navigation_lesson and len(str(navigation_lesson).strip()) > 5:
            lessons.append(f"Navigation Insight: {str(navigation_lesson).strip()}")
        if not lessons and environment_lesson and len(str(environment_lesson).strip()) > 5:
            lessons.append(f"Environment Insight: {str(environment_lesson).strip()}")
        if not lessons and lesson and len(str(lesson).strip()) > 5:
            lessons.append(f"Lesson: {str(lesson).strip()}")

        if not lessons:
            return False, "empty_lesson"

        memory_id = self.add(
            task_description=task_description,
            reflection_text=" | ".join(lessons),
            trajectory=trajectory,
            initial_score=initial_score,
            attempt_type="success" if actual_success else "failure",
            current_progress_ratio=current_progress_ratio,
            metadata=metadata,
            save=save,
        )
        return bool(memory_id), memory_id or "add_failed"


class DynamicMemoryProvider:
    def __init__(self, config: Dict[str, Any]):
        filepath = config.get("filepath", "outputs/dynamic_memory/reflection_memory.json")
        self.top_k = int(config.get("top_k", 1))
        self.eval_top_k = int(config.get("eval_top_k", self.top_k))
        self.dual_pool_retrieval = _as_bool(config.get("dual_pool_retrieval", False), default=False)
        self.memory_pool_mode = str(
            config.get("memory_pool_mode", config.get("pool_mode", "both"))
        ).strip().lower()
        pool_mode_aliases = {
            "all": "both",
            "dual": "both",
            "dual_pool": "both",
            "both": "both",
            "task": "task_only",
            "task_only": "task_only",
            "task-only": "task_only",
            "task_skill": "task_only",
            "task_skills": "task_only",
            "state": "state_only",
            "step": "state_only",
            "state_only": "state_only",
            "state-only": "state_only",
            "step_only": "state_only",
            "step-only": "state_only",
            "state_skill": "state_only",
            "state_skills": "state_only",
        }
        self.memory_pool_mode = pool_mode_aliases.get(self.memory_pool_mode, self.memory_pool_mode)
        if self.memory_pool_mode not in {"both", "task_only", "state_only"}:
            raise ValueError(
                "Unsupported algorithm.ucob.skill_memory.memory_pool_mode="
                f"{self.memory_pool_mode!r}; choose 'both', 'task_only', or 'state_only'."
            )
        self.task_top_k = int(config.get("task_top_k", config.get("top_k_task", self.top_k)))
        self.state_top_k = int(
            config.get(
                "state_top_k",
                config.get("step_top_k", config.get("top_k_step", self.top_k)),
            )
        )
        self.eval_task_top_k = int(config.get("eval_task_top_k", config.get("eval_top_k_task", self.task_top_k)))
        self.eval_state_top_k = int(
            config.get(
                "eval_state_top_k",
                config.get("eval_step_top_k", config.get("eval_top_k_step", self.state_top_k)),
            )
        )
        self.filter_type = str(config.get("filter_type", "both"))
        self.skill_format = str(config.get("skill_format", "retroagent")).lower()
        self.category_filter = _as_bool(config.get("category_filter", False))
        self.validate_success = _as_bool(config.get("validate_success", True), default=True)
        self.write_enable = _as_bool(config.get("write_enable", True), default=True)
        self.utility_update = str(config.get("utility_update", "outcome")).strip().lower()
        default_utility_beta = float(config.get("beta", 0.05))
        self.task_utility_beta = float(
            config.get("task_utility_beta", config.get("utility_beta_task", default_utility_beta))
        )
        self.state_utility_beta = float(
            config.get(
                "state_utility_beta",
                config.get("step_utility_beta", config.get("utility_beta_step", default_utility_beta)),
            )
        )
        if config.get("initial_utility_score") is not None:
            self.initial_utility_score = float(config.get("initial_utility_score"))
        elif self.utility_update in {"baseline_relative", "d2skill", "d2skill_relative"}:
            self.initial_utility_score = 0.0
        else:
            self.initial_utility_score = 0.5
        self.max_trajectory_steps = int(config.get("max_trajectory_steps", 50))
        self.max_reflection_items = int(config.get("max_reflection_items", 6))
        self.current_progress_ratio = 0.0

        self.memory = ReflectionMemory(
            filepath=filepath,
            alpha=config.get("alpha", 0.6),
            beta=config.get("beta", 0.05),
            temperature=config.get("temperature", 0.5),
            retrieve_type=config.get("retrieve_type", "ucb"),
            ucb_scale=config.get("ucb_scale", 1.0),
            similarity_backend=config.get("similarity_backend", "sentence_transformer"),
            model_name=config.get("model_name", "all-MiniLM-L6-v2"),
            relevance_threshold=config.get("relevance_threshold", 0.4),
            similarity_top_m=config.get("similarity_top_m", 0),
            merge_exact_task_attempt=config.get("merge_exact_task_attempt", True),
            max_lessons_per_task_attempt=config.get("max_lessons_per_task_attempt", 4),
            skill_format=self.skill_format,
            category_filter=self.category_filter,
            skill_mapping_dir=config.get("skill_mapping_dir", None),
            retrieval_query_mode=config.get("retrieval_query_mode", config.get("query_mode", "task")),
        )
        self.skill_format = self.memory.skill_format
        self.category_filter = self.memory.category_filter

    def allows_pool(self, pool: str) -> bool:
        normalized_pool = str(pool or "task").strip().lower()
        if normalized_pool in {"step", "state_skill", "step_skill"}:
            normalized_pool = "state"
        if normalized_pool not in {"task", "state"}:
            return self.memory_pool_mode == "both"
        if self.memory_pool_mode == "both":
            return True
        if self.memory_pool_mode == "task_only":
            return normalized_pool == "task"
        if self.memory_pool_mode == "state_only":
            return normalized_pool == "state"
        return True

    def _top_k_for_pool(self, pool: str, is_eval: bool = False) -> int:
        if pool == "state":
            return self.eval_state_top_k if is_eval else self.state_top_k
        return self.eval_task_top_k if is_eval else self.task_top_k

    def set_training_progress(self, current_step: int, total_steps: int) -> None:
        if total_steps and total_steps > 0:
            self.current_progress_ratio = max(0.0, min(1.0, float(current_step) / float(total_steps)))

    def infer_task_category(self, task_description: str) -> str:
        return self.memory.infer_task_category(task_description)

    def retrieve_for_tasks(
        self,
        tasks: List[str],
        is_eval: bool = False,
        state_contexts: Optional[List[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        if self.memory_pool_mode in {"task_only", "state_only"}:
            pool = "task" if self.memory_pool_mode == "task_only" else "state"
            return self.memory.retrieve_many(
                tasks,
                top_k=self._top_k_for_pool(pool, is_eval=is_eval),
                filter_type=self.filter_type,
                state_contexts=state_contexts,
                pool=pool,
            )
        if self.dual_pool_retrieval:
            task_top_k = self.eval_task_top_k if is_eval else self.task_top_k
            state_top_k = self.eval_state_top_k if is_eval else self.state_top_k
            return self.memory.retrieve_many_dual_pool(
                tasks,
                task_top_k=task_top_k,
                state_top_k=state_top_k,
                filter_type=self.filter_type,
                state_contexts=state_contexts,
            )
        top_k = self.eval_top_k if is_eval else self.top_k
        return self.memory.retrieve_many(
            tasks,
            top_k=top_k,
            filter_type=self.filter_type,
            state_contexts=state_contexts,
        )

    def retrieve_bad_for_tasks(
        self,
        tasks: List[str],
        is_eval: bool = False,
        state_contexts: Optional[List[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        if self.memory_pool_mode in {"task_only", "state_only"}:
            pool = "task" if self.memory_pool_mode == "task_only" else "state"
            return self.memory.retrieve_bad_many(
                tasks,
                top_k=self._top_k_for_pool(pool, is_eval=is_eval),
                filter_type=self.filter_type,
                state_contexts=state_contexts,
                pool=pool,
            )
        if self.dual_pool_retrieval:
            task_top_k = self.eval_task_top_k if is_eval else self.task_top_k
            state_top_k = self.eval_state_top_k if is_eval else self.state_top_k
            return self.memory.retrieve_bad_many_dual_pool(
                tasks,
                task_top_k=task_top_k,
                state_top_k=state_top_k,
                filter_type=self.filter_type,
                state_contexts=state_contexts,
            )
        top_k = self.eval_top_k if is_eval else self.top_k
        return self.memory.retrieve_bad_many(
            tasks,
            top_k=top_k,
            filter_type=self.filter_type,
            state_contexts=state_contexts,
        )

    def retrieve_good_bad_for_tasks(
        self,
        tasks: List[str],
        is_eval: bool = False,
        state_contexts: Optional[List[str]] = None,
    ) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
        if self.memory_pool_mode in {"task_only", "state_only"}:
            pool = "task" if self.memory_pool_mode == "task_only" else "state"
            return self.memory.retrieve_good_bad_many(
                tasks,
                top_k=self._top_k_for_pool(pool, is_eval=is_eval),
                filter_type=self.filter_type,
                state_contexts=state_contexts,
                pool=pool,
            )
        if self.dual_pool_retrieval:
            task_top_k = self.eval_task_top_k if is_eval else self.task_top_k
            state_top_k = self.eval_state_top_k if is_eval else self.state_top_k
            return self.memory.retrieve_good_bad_many_dual_pool(
                tasks,
                task_top_k=task_top_k,
                state_top_k=state_top_k,
                filter_type=self.filter_type,
                state_contexts=state_contexts,
            )
        top_k = self.eval_top_k if is_eval else self.top_k
        return self.memory.retrieve_good_bad_many(
            tasks,
            top_k=top_k,
            filter_type=self.filter_type,
            state_contexts=state_contexts,
        )

    def format_retrieved(self, retrieved: List[Dict[str, Any]]) -> str:
        return self.memory.format_retrieved(retrieved)

    def update_retrieved_utility(self, retrieved: List[Dict[str, Any]], success: bool) -> int:
        if self.utility_update == "none":
            return 0
        memory_ids = [item.get("memory_id", "") for item in retrieved]
        return self.memory.update_utility_by_ids(memory_ids, 1.0 if success else 0.0)

    def update_retrieved_utilities(self, retrieved_success_pairs: List[Tuple[List[Dict[str, Any]], bool]]) -> int:
        if self.utility_update == "none":
            return 0
        updates = []
        for retrieved, success in retrieved_success_pairs:
            score = 1.0 if success else 0.0
            for item in retrieved:
                memory_id = item.get("memory_id", "")
                if memory_id:
                    updates.append((memory_id, score))
        return self.memory.update_utility_by_id_scores(updates)

    def update_utility_by_id_scores(self, updates: List[Tuple[str, float]], beta: Optional[float] = None) -> int:
        if self.utility_update == "none":
            return 0
        return self.memory.update_utility_by_id_scores(updates, beta=beta)

    def memory_pool_by_id(self, memory_id: str) -> str:
        return self.memory.memory_pool_by_id(memory_id)

    def get_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        return self.memory.get_by_id(memory_id)

    def update_retrieved_utilities_relative(
        self,
        retrieved_by_env: List[List[Dict[str, Any]]],
        successes: List[bool],
        skill_mask: List[bool],
        group_n: int,
    ) -> int:
        if self.utility_update == "none":
            return 0

        batch_size = min(len(retrieved_by_env), len(successes), len(skill_mask))
        if batch_size <= 0:
            return 0

        group_n = max(int(group_n), 1)
        updates = []
        for start in range(0, batch_size, group_n):
            group_indices = list(range(start, min(start + group_n, batch_size)))
            no_skill_indices = [idx for idx in group_indices if not bool(skill_mask[idx])]
            skill_indices = [idx for idx in group_indices if bool(skill_mask[idx])]
            if not no_skill_indices or not skill_indices:
                continue

            no_skill_mean = sum(1.0 if successes[idx] else 0.0 for idx in no_skill_indices) / len(no_skill_indices)
            skill_mean = sum(1.0 if successes[idx] else 0.0 for idx in skill_indices) / len(skill_indices)
            relative_score = 0.5 + (0.5 * (skill_mean - no_skill_mean))
            relative_score = max(0.0, min(1.0, relative_score))

            seen_memory_ids = set()
            for idx in skill_indices:
                for item in retrieved_by_env[idx]:
                    memory_id = item.get("memory_id", "")
                    if memory_id and memory_id not in seen_memory_ids:
                        updates.append((memory_id, relative_score))
                        seen_memory_ids.add(memory_id)

        return self.memory.update_utility_by_id_scores(updates)

    def retrieve_reference_trajectory(self, task_description: str, target_type: str) -> Optional[str]:
        return self.memory.retrieve_trajectory(task_description, target_type=target_type)

    def save(self) -> None:
        self.memory.save()

    def add_from_reflection_output(
        self,
        task_description: str,
        reflection_output: str,
        trajectory: str,
        actual_success: bool,
        metadata: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> Tuple[bool, str]:
        if not self.write_enable:
            return False, "write_disabled"
        metadata = dict(metadata or {})
        write_pool = "state" if metadata.get("memory_type") == "state_skill" else "task"
        if not self.allows_pool(write_pool):
            return False, f"{write_pool}_pool_disabled"
        return self.memory.add_from_reflection_output(
            task_description=task_description,
            reflection_output=reflection_output,
            trajectory=trajectory,
            actual_success=actual_success,
            current_progress_ratio=self.current_progress_ratio,
            validate_success=self.validate_success,
            metadata=metadata,
            initial_score=self.initial_utility_score,
            save=save,
        )
