class ProceduralMemory:
    """
    Stores learned procedures and reusable workflows.

    Procedural memory answers:

        "How does NOVA know how to do this?"

    Unlike normal facts, procedures represent repeatable
    actions, workflows, or learned development practices.
    """

    def __init__(self, store):
        self.store = store

    # =========================================================
    # INTERNAL LOAD
    # =========================================================

    def _load(self) -> dict:

        procedures = self.store.get(
            "procedures",
            {},
        )

        if not isinstance(
            procedures,
            dict,
        ):
            procedures = {}

        return procedures

    # =========================================================
    # SAVE PROCEDURE
    # =========================================================

    def save(
        self,
        task: str,
        procedure,
    ) -> bool:

        task = str(
            task
        ).strip()

        if not task:
            return False

        if procedure is None:
            return False

        procedures = self._load()

        procedures[task] = procedure

        self.store.set(
            "procedures",
            procedures,
        )

        return True

    # =========================================================
    # GET PROCEDURE
    # =========================================================

    def get(
        self,
        task: str,
        default=None,
    ):

        task = str(
            task
        ).strip()

        if not task:
            return default

        procedures = self._load()

        return procedures.get(
            task,
            default,
        )

    # =========================================================
    # DELETE PROCEDURE
    # =========================================================

    def delete(
        self,
        task: str,
    ) -> bool:

        task = str(
            task
        ).strip()

        if not task:
            return False

        procedures = self._load()

        if task not in procedures:
            return False

        del procedures[task]

        self.store.set(
            "procedures",
            procedures,
        )

        return True

    # =========================================================
    # ALL PROCEDURES
    # =========================================================

    def all(self) -> dict:

        return self._load()

    # =========================================================
    # SEARCH PROCEDURES
    # =========================================================

    def search(
        self,
        query: str,
    ) -> list[dict]:

        query = str(
            query or ""
        ).strip().lower()

        if not query:
            return []

        query_words = set(
            query.split()
        )

        results = []

        for task, procedure in self._load().items():

            searchable = (
                f"{task} {procedure}"
            ).lower()

            procedure_words = set(
                searchable.split()
            )

            score = len(
                query_words
                & procedure_words
            )

            if score:

                results.append({
                    "task": task,
                    "procedure": procedure,
                    "score": score,
                })

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results