from __future__ import annotations

from typing import List

from uuid import uuid4

from rag_service.backends.base import RetrievalBackend
from rag_service.config import RagConfig
from rag_service.models import RetrievalRequest, RetrievalResponse
from rag_service.relevance import identifier_tokens


class RagService:
    """Stable, read-only facade shared by every agent adapter."""

    def __init__(self, config: RagConfig, backend: RetrievalBackend):
        self.config = config
        self.backend = backend

    def search(self, request: RetrievalRequest) -> RetrievalResponse:
        request = self._normalize_request(request)
        results = self.backend.search(request)
        response = RetrievalResponse(
            query=request.query,
            knowledge_base=request.knowledge_base,
            results=results,
            total=len(results),
            backend=self.backend.name,
            embedding_model=self.config.embedding_model,
            request_id=uuid4().hex,
            applied_filters=dict(request.filters),
        )
        operational_state = getattr(self.backend, "search_state", lambda: {})()
        degraded = [
            result.metadata["degraded"]
            for result in results
            if result.metadata.get("degraded")
        ]
        degraded_value = degraded[0] if degraded else operational_state.get("degraded")
        if degraded_value:
            response.degraded = degraded_value
            response.warnings.append(
                "embedding provider unreachable: results are lexical-only and scores are "
                "not cosine similarities; recall is narrower than normal"
            )
        if results:
            response.next_cursor = results[0].metadata.get("next_cursor")
            response.truncated = any(
                bool(result.metadata.get("truncated")) for result in results
            )
        else:
            response.no_match = True

        filters = request.filters or {}
        knowledge_base = request.knowledge_base or self.config.default_knowledge_base

        # A broken exclusion is silent in the unsafe direction (nothing dropped),
        # so check it independently of the result count.
        excluded = filters.get("exclude_source_prefix")
        if excluded:
            prefixes = excluded if isinstance(excluded, list) else [excluded]
            dead = self._unmatched_exclude_prefixes(
                knowledge_base, [str(prefix) for prefix in prefixes]
            )
            if dead:
                response.warnings.append(
                    "exclude_source_prefix matched no indexed source: "
                    f"{', '.join(dead)}; prefixes are literal, so a typo excludes nothing"
                )

        # A non-default lexical_weight silently moves the gate; say so.
        weight_warning = self._lexical_weight_warning(request)
        if weight_warning:
            response.warnings.append(weight_warning)

        if not results:
            if filters:
                category_value = filters.get("category")
                if isinstance(category_value, str):
                    available = self.config.content_categories(knowledge_base)
                    if available and category_value not in available:
                        response.warnings.append(
                            f"filter category '{category_value}' is not an indexed content "
                            f"prefix; available prefixes: {', '.join(available)}"
                        )
                        return response
                if not request.query.strip():
                    browse_scope = (
                        filters.get("source")
                        or filters.get("source_prefix")
                        or filters.get("category")
                        or "the corpus"
                    )
                    response.warnings.append(
                        f"no document matched browse scope {browse_scope!r}; "
                        "verify the exact path via the empty-query corpus index"
                    )
                else:
                    response.warnings.append(
                        "no documents matched the requested filters for this query; "
                        "consider removing filters or increasing top_k"
                    )
            else:
                # An unfiltered query returning nothing used to carry no warning
                # at all, so the caller read "no evidence in the corpus" — a
                # false negative when the cause was a threshold or pool cutoff.
                response.warnings.append(
                    "no document scored above the threshold for this query; this is not "
                    "proof the corpus lacks the topic — the query may be too generic, may "
                    "use different wording than the corpus, or the target document may "
                    "rank below the rerank pool. Retry with concrete technical anchors "
                    "(software + version + mechanism), or lower score_threshold to inspect "
                    "near-misses."
                )
        if results:
            # A browse response is navigation, not a match, so grading it would
            # report "no distinctive term" about a listing that never had a query.
            response.confidence = self._confidence(request, results)
            top_score = results[0].score
            low = (
                top_score is not None
                and self.config.low_score_warn > 0
                and top_score < self.config.low_score_warn
            )
            if low:
                response.warnings.append(
                    f"top result score is low ({top_score:.3f} < {self.config.low_score_warn:.2f}); "
                    "the query may be too vague, out of corpus, or matching only noise"
                )
            elif response.confidence == "semantic":
                # Only reached once the hit has cleared the coarse filter: the
                # interesting failure is a match that *looks* confident by score
                # (the measured 0.49-0.55 band where off-corpus probes land) but
                # shares no term with the query. Below the threshold the low-score
                # warning is the honest message, and saying both would be noise.
                response.warnings.append(
                    "top result matched on meaning alone: it shares no distinctive term "
                    "with the query. Confirm the topic against the cited source before "
                    "relying on it, or retry with concrete anchors (software + version + "
                    "mechanism)."
                )
        return response

    def _confidence(self, request: RetrievalRequest, results: List) -> str | None:
        """Grade the top hit by the evidence behind it, not by its raw score.

        Measured on the real corpus: the top-1/top-2 margin does *not* indicate
        confidence here (a correct hit scored a 0.0011 margin because several
        distinct documents cover the topic, while off-corpus queries sat at
        0.004-0.006). What does discriminate is whether the query and the winning
        document share actual terms -- and, most precisely, whether an identifier
        the caller supplied (CVE, version, port) appears in that document.
        """
        if not results or not request.query.strip():
            return None
        top = results[0]
        content = (top.content or "").lower()
        identifiers = identifier_tokens(request.query)
        if identifiers and any(token.lower() in content for token in identifiers):
            return "anchored"
        lexical = (top.metadata or {}).get("lexical_score")
        if not isinstance(lexical, (int, float)):
            # No lexical component was computed (dense-only mode, or a backend
            # that does not emit one). Absence of the signal is not evidence the
            # match was semantic, so decline to grade rather than guess.
            return None
        return "lexical" if lexical >= 0.5 else "semantic"

    def _lexical_weight_warning(self, request: RetrievalRequest) -> str | None:
        """Explain how a raised ``lexical_weight`` moves the score gate.

        ``score_threshold`` gates the *fused* score ``(1-w)*dense + w*lexical``.
        Raising ``w`` therefore raises the dense similarity a result needs when
        it has no lexical hits: at the default 0.35 that floor is 0.692, at 0.6
        it is 1.125 — no cosine can reach it, so every query returns empty.
        Measured: ``lexical_weight`` 0.6/1.0 turned ``CVE-2017-0144 eternalblue``
        and ``one gadget __free_hook`` into ``no_match``. The knob reads as
        "strengthen keyword matching", so the arithmetic has to be stated rather
        than left to be inferred.
        """
        weight = request.lexical_weight
        if weight is None or weight <= self.config.lexical_weight + 0.05:
            return None
        if weight >= 1.0:
            detail = "no score can pass at all"
        else:
            floor = self.config.default_score_threshold / (1.0 - weight)
            detail = f"a result with no lexical hits needs dense >= {floor:.3f}"
        default_floor = self.config.default_score_threshold / (
            1.0 - self.config.lexical_weight
        )
        return (
            f"lexical_weight={weight:.2f} also raises the score_threshold gate, because the "
            f"threshold applies to the fused score: {detail} (at the default "
            f"{self.config.lexical_weight:.2f} it needs dense >= {default_floor:.3f}). "
            "Tune lexical_weight only alongside `python -m rag_service.evaluate`."
        )

    def _unmatched_exclude_prefixes(
        self, knowledge_base: str, prefixes: List[str]
    ) -> List[str]:
        """Return exclude prefixes matching no indexed source.

        An exclusion that matches nothing is nearly always a typo, and it fails
        in the unsafe direction: nothing is dropped while the caller believes
        noise was removed. Silent, so it is worth a warning — also when results
        are returned, because a partially-working filter hides the typo.
        """
        try:
            sources = getattr(self.backend, "known_sources", None)
            if sources is None:
                return []
            known = sources(knowledge_base)
        except Exception:
            return []
        if not known:
            return []
        return [
            prefix
            for prefix in prefixes
            if prefix and not any(source.startswith(prefix) for source in known)
        ]

    def _normalize_request(self, request: RetrievalRequest) -> RetrievalRequest:
        values = request.model_dump()
        if not values["knowledge_base"]:
            values["knowledge_base"] = self.config.default_knowledge_base
        if values["top_k"] > self.config.max_top_k:
            values["top_k"] = self.config.max_top_k
        if values["limit"] is not None and values["limit"] > self.config.max_top_k:
            values["limit"] = self.config.max_top_k
        if values["score_threshold"] is None:
            values["score_threshold"] = self.config.default_score_threshold
        if len(values["query"]) > self.config.max_query_length:
            raise ValueError("query exceeds configured maximum length")
        self.config.require_allowed(values["knowledge_base"])
        return RetrievalRequest.model_validate(values)
