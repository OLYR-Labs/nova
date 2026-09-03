from datetime import datetime


class EpisodicMemory:
    """
    Persistent episodic memory for NOVA.

    Stores important events and experiences that NOVA
    may need to recall in future sessions.
    """

    MAX_EPISODES = 500

    def __init__(self, store):
        self.store = store

    # =========================================================
    # RECORD EPISODE
    # =========================================================

    def record(
        self,
        event_type: str,
        content: str,
        metadata=None,
    ) -> bool:

        event_type = str(
            event_type
        ).strip()

        content = str(
            content
        ).strip()

        if not event_type or not content:
            return False

        events = self.store.get(
            "episodes",
            [],
        )

        if not isinstance(
            events,
            list,
        ):
            events = []

        events.append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "content": content,
            "metadata": metadata or {},
        })

        events = events[
            -self.MAX_EPISODES:
        ]

        self.store.set(
            "episodes",
            events,
        )

        return True

    # =========================================================
    # RECENT EPISODES
    # =========================================================

    def recent(
        self,
        limit: int = 20,
    ) -> list:

        try:
            limit = int(limit)
        except (
            TypeError,
            ValueError,
        ):
            limit = 20

        if limit <= 0:
            return []

        events = self.store.get(
            "episodes",
            [],
        )

        if not isinstance(
            events,
            list,
        ):
            return []

        return events[
            -limit:
        ]

    # =========================================================
    # SEARCH EPISODES
    # =========================================================

    def search(
        self,
        query: str,
        limit: int = 20,
    ) -> list:

        query = str(
            query or ""
        ).strip().lower()

        if not query:
            return []

        events = self.store.get(
            "episodes",
            [],
        )

        if not isinstance(
            events,
            list,
        ):
            return []

        words = set(
            query.split()
        )

        results = []

        for event in events:

            if not isinstance(
                event,
                dict,
            ):
                continue

            searchable = " ".join([
                str(
                    event.get(
                        "type",
                        "",
                    )
                ),
                str(
                    event.get(
                        "content",
                        "",
                    )
                ),
            ]).lower()

            event_words = set(
                searchable.split()
            )

            score = len(
                words & event_words
            )

            if score:
                results.append({
                    "event": event,
                    "score": score,
                })

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[
            :limit
        ]