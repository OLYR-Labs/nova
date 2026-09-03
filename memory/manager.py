import re
from typing import Optional

from memory.procedural import ProceduralMemory
from memory.store import MemoryStore
from memory.working import WorkingMemory
from memory.episodic import EpisodicMemory


class MemoryManager:
    """
    Persistent, searchable long-term memory for NOVA.

    Memory layers:
    - Working memory
    - Conversation memory
    - Episodic memory
    - Semantic/long-term memory
    - Procedural memory
    """

    VALID_CATEGORIES = {
        "identity",
        "preference",
        "project",
        "goal",
        "fact",
        "instruction",
        "technical_context",
        "relationship",
    }

    STOP_WORDS = {
        "what", "when", "where", "which", "who", "why", "how",
        "is", "are", "was", "were", "the", "this", "that",
        "my", "your", "our", "their", "about", "does", "do",
        "did", "and", "or", "for", "with", "from", "into",
        "have", "has", "had",
    }

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
    ):
        self.store = store or MemoryStore()

        # ---------------------------------------------------------
        # WORKING MEMORY
        # ---------------------------------------------------------

        self.working = WorkingMemory(
            max_items=20
        )

        # ---------------------------------------------------------
        # EPISODIC MEMORY
        # ---------------------------------------------------------

        self.episodic = EpisodicMemory(
            self.store
        )

        # ---------------------------------------------------------
        # PROCEDURAL MEMORY
        # ---------------------------------------------------------

        self.procedural = ProceduralMemory(
            self.store
        )

    # =============================================================
    # TEXT PROCESSING
    # =============================================================

    @classmethod
    def _words(
        cls,
        text: str,
    ) -> set[str]:

        words = re.findall(
            r"[a-zA-Z0-9_]+",
            text.lower(),
        )

        return {
            word
            for word in words
            if word not in cls.STOP_WORDS
        }

    # =============================================================
    # LONG-TERM MEMORY
    # =============================================================

    def save(
        self,
        category: str,
        key: str,
        value: str,
    ) -> bool:

        category = category.strip().lower()
        key = key.strip()
        value = value.strip()

        if category not in self.VALID_CATEGORIES:
            return False

        if not key or not value:
            return False

        data = self.store.all()

        if category not in data:
            data[category] = {}

        if not isinstance(
            data[category],
            dict,
        ):
            data[category] = {}

        data[category][key] = value

        self.store._save(data)

        return True

    def get(
        self,
        category: str,
        key: str,
        default=None,
    ):

        data = self.store.all()

        category_data = data.get(
            category,
            {},
        )

        if not isinstance(
            category_data,
            dict,
        ):
            return default

        return category_data.get(
            key,
            default,
        )

    def all(self) -> dict:
        return self.store.all()

    # =============================================================
    # SEARCH LONG-TERM MEMORY
    # =============================================================

    def search(
        self,
        query: str,
    ) -> list[dict]:

        query_words = self._words(query)

        if not query_words:
            return []

        results = []

        for category, values in self.store.all().items():

            # -----------------------------------------------------
            # Non-dictionary memory
            # -----------------------------------------------------

            if not isinstance(
                values,
                dict,
            ):

                searchable = (
                    f"{category} {values}"
                )

                memory_words = self._words(
                    searchable
                )

                score = len(
                    query_words & memory_words
                )

                if score:

                    results.append({
                        "category": "fact",
                        "key": category,
                        "value": values,
                        "score": score,
                    })

                continue

            # -----------------------------------------------------
            # Categorized memory
            # -----------------------------------------------------

            for key, value in values.items():

                searchable = (
                    f"{category} "
                    f"{key} "
                    f"{value}"
                )

                memory_words = self._words(
                    searchable
                )

                score = len(
                    query_words & memory_words
                )

                if score:

                    results.append({
                        "category": category,
                        "key": key,
                        "value": value,
                        "score": score,
                    })

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results

    # =============================================================
    # DELETE LONG-TERM MEMORY
    # =============================================================

    def delete(
        self,
        category: str,
        key: str,
    ) -> bool:

        data = self.store.all()

        category_data = data.get(
            category
        )

        if not isinstance(
            category_data,
            dict,
        ):
            return False

        if key not in category_data:
            return False

        del category_data[key]

        if not category_data:
            del data[category]

        self.store._save(data)

        return True

    # =============================================================
    # WORKING MEMORY
    # =============================================================

    def add_working_memory(
        self,
        content: str,
    ):
        """
        Add temporary information to NOVA's
        current reasoning context.
        """

        if not content:
            return

        self.working.add(
            content.strip()
        )

    def get_working_memory(self) -> list:
        """
        Return the current working-memory items.
        """

        return self.working.recent()

    def working_context(self) -> str:
        """
        Build a text representation of working memory.
        """

        context = self.working.context()

        if not context:
            return (
                "WORKING MEMORY:\n"
                "No active working memory."
            )

        return (
            "WORKING MEMORY:\n"
            + context
        )

    def clear_working_memory(self):
        """
        Clear temporary reasoning state.
        """

        self.working.clear()

    # =============================================================
    # EPISODIC MEMORY
    # =============================================================

    def remember_episode(
        self,
        event_type: str,
        content: str,
        metadata: Optional[dict] = None,
    ):
        """
        Record an important event in NOVA's
        persistent episodic memory.
        """

        if not content:
            return

        self.episodic.record(
            event_type=event_type,
            content=content,
            metadata=metadata,
        )

    def recent_episodes(
        self,
        limit: int = 20,
    ) -> list:
        """
        Return recent episodic memories.
        """

        return self.episodic.recent(
            limit=limit
        )

    # =============================================================
    # PROCEDURAL MEMORY
    # =============================================================

    def save_procedure(
        self,
        task: str,
        procedure: str,
    ) -> bool:
        """
        Store a reusable procedure that NOVA
        has learned for a particular task.
        """

        task = str(task).strip()
        procedure = str(procedure).strip()

        if not task or not procedure:
            return False

        self.procedural.save(
            task=task,
            procedure=procedure,
        )

        return True

    def get_procedure(
        self,
        task: str,
    ):
        """
        Retrieve a learned procedure.
        """

        if not task:
            return None

        return self.procedural.get(
            task
        )

    def all_procedures(self) -> dict:
        """
        Return every stored procedural memory.
        """

        return self.procedural.all()

    # =============================================================
    # MEMORY CONTEXT
    # =============================================================

    def build_context(
        self,
        query: Optional[str] = None,
    ) -> str:

        # ---------------------------------------------------------
        # LONG-TERM MEMORY
        # ---------------------------------------------------------

        if query:

            memories = self.search(
                query
            )

        else:

            memories = []

            for category, values in self.store.all().items():

                # -------------------------------------------------
                # Ignore internal procedural storage here.
                # It gets exposed separately when relevant.
                # -------------------------------------------------

                if category == "procedures":
                    continue

                if not isinstance(
                    values,
                    dict,
                ):

                    memories.append({
                        "category": "fact",
                        "key": category,
                        "value": values,
                        "score": 0,
                    })

                    continue

                for key, value in values.items():

                    memories.append({
                        "category": category,
                        "key": key,
                        "value": value,
                        "score": 0,
                    })

        lines = []

        # ---------------------------------------------------------
        # LONG-TERM MEMORY
        # ---------------------------------------------------------

        lines.append(
            "LONG-TERM MEMORY:"
        )

        if memories:

            for memory in memories:

                lines.append(
                    f"- [{memory['category']}] "
                    f"{memory['key']}: "
                    f"{memory['value']}"
                )

        else:

            lines.append(
                "- No relevant persistent memories."
            )

        # ---------------------------------------------------------
        # WORKING MEMORY
        # ---------------------------------------------------------

        lines.append("")

        lines.append(
            self.working_context()
        )

        # ---------------------------------------------------------
        # RECENT EPISODIC MEMORY
        # ---------------------------------------------------------

        episodes = self.recent_episodes(
            limit=10
        )

        lines.append("")

        lines.append(
            "RECENT EPISODIC MEMORY:"
        )

        if episodes:

            for episode in episodes:

                if isinstance(
                    episode,
                    dict,
                ):

                    event_type = episode.get(
                        "event_type",
                        "event",
                    )

                    content = episode.get(
                        "content",
                        "",
                    )

                    timestamp = episode.get(
                        "timestamp",
                        "",
                    )

                    lines.append(
                        f"- [{event_type}] "
                        f"{content}"
                        + (
                            f" ({timestamp})"
                            if timestamp
                            else ""
                        )
                    )

                else:

                    lines.append(
                        f"- {episode}"
                    )

        else:

            lines.append(
                "- No recent episodes."
            )

        # ---------------------------------------------------------
        # PROCEDURAL MEMORY
        # ---------------------------------------------------------

        procedures = self.all_procedures()

        lines.append("")

        lines.append(
            "PROCEDURAL MEMORY:"
        )

        if procedures:

            for task, procedure in procedures.items():

                lines.append(
                    f"- {task}: {procedure}"
                )

        else:

            lines.append(
                "- No learned procedures."
            )

        return "\n".join(lines)

    # =============================================================
    # MEMORY EXTRACTION
    # =============================================================

    @staticmethod
    def extraction_prompt(
        user_message: str,
    ) -> str:

        return f"""
Analyze the user's message for information that should be
remembered across future conversations.

Only save genuinely useful long-term information.

Good examples:
- identity
- persistent preferences
- ongoing projects
- important technical context
- long-term goals
- explicit instructions about NOVA
- reusable procedures or workflows the user explicitly wants NOVA
  to remember

Do NOT save:
- casual conversation
- temporary actions
- one-time questions
- jokes
- greetings
- sensitive information unless explicitly requested
- information that is only useful for the current turn

Return ONLY valid JSON.

If nothing should be remembered:

{{
    "remember": false
}}

Otherwise:

{{
    "remember": true,
    "category": "identity|preference|project|goal|fact|instruction|technical_context|relationship",
    "key": "short stable key",
    "value": "concise persistent fact"
}}

USER MESSAGE:

{user_message}
"""