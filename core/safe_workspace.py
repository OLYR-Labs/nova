import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


class SafeWorkspace:
    """
    Safe filesystem layer for NOVA.

    Every modification is:

        1. validated
        2. backed up
        3. written
        4. verified
        5. recorded

    Backups live inside:

        .nova/backups/

    The workspace itself remains inside NOVA's sandbox.
    """

    def __init__(
        self,
        root,
    ):

        self.root = Path(
            root
        ).resolve()

        self.meta_root = (
            self.root / ".nova"
        )

        self.backup_root = (
            self.meta_root / "backups"
        )

        self.log_file = (
            self.meta_root / "operations.jsonl"
        )

        self.meta_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.backup_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================
    # SAFE PATH
    # =========================================================

    def safe_path(
        self,
        path,
    ):

        if not path:
            path = "."

        requested = Path(
            path
        )

        if requested.is_absolute():
            target = requested.resolve()
        else:
            target = (
                self.root / requested
            ).resolve()

        try:

            target.relative_to(
                self.root
            )

        except ValueError:

            raise PermissionError(
                f"Path outside NOVA sandbox blocked: {path}"
            )

        return target

    # =========================================================
    # HASH
    # =========================================================

    def hash_file(
        self,
        path,
    ):

        content = Path(
            path
        ).read_bytes()

        return hashlib.sha256(
            content
        ).hexdigest()

    # =========================================================
    # BACKUP
    # =========================================================

    def backup(
        self,
        target,
    ):

        target = Path(
            target
        )

        if not target.exists():
            return None

        timestamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )

        relative = target.relative_to(
            self.root
        )

        backup = (
            self.backup_root
            / timestamp
            / relative
        )

        backup.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if target.is_file():

            shutil.copy2(
                target,
                backup,
            )

        elif target.is_dir():

            shutil.copytree(
                target,
                backup,
                dirs_exist_ok=True,
            )

        return backup

    # =========================================================
    # LOG
    # =========================================================

    def log(
        self,
        operation,
        path,
        success,
        details=None,
    ):

        record = {
            "timestamp": (
                datetime.now().isoformat()
            ),
            "operation": operation,
            "path": str(path),
            "success": bool(success),
            "details": details or {},
        }

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.log_file.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # =========================================================
    # WRITE
    # =========================================================

    def write_file(
        self,
        path,
        content,
    ):

        target = self.safe_path(
            path
        )

        if target.exists():

            backup = self.backup(
                target
            )

        else:

            backup = None

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target.write_text(
            str(content),
            encoding="utf-8",
        )

        verification = self.verify(
            target
        )

        success = verification["success"]

        self.log(
            "write_file",
            target,
            success,
            {
                "backup": (
                    str(backup)
                    if backup
                    else None
                ),
                "verification": verification,
            },
        )

        return {
            "success": success,
            "action": "write_file",
            "path": str(
                target.relative_to(
                    self.root
                )
            ),
            "bytes": target.stat().st_size,
            "backup": (
                str(
                    backup.relative_to(
                        self.root
                    )
                )
                if backup
                else None
            ),
            "verification": verification,
        }

    # =========================================================
    # EDIT
    # =========================================================

    def edit_file(
        self,
        path,
        old_text,
        new_text,
    ):

        target = self.safe_path(
            path
        )

        if not target.exists():
            return {
                "success": False,
                "error": (
                    f"File does not exist: {path}"
                ),
            }

        if not target.is_file():
            return {
                "success": False,
                "error": (
                    f"Path is not a file: {path}"
                ),
            }

        content = target.read_text(
            encoding="utf-8"
        )

        occurrences = content.count(
            old_text
        )

        if occurrences == 0:

            return {
                "success": False,
                "error": (
                    "old_text was not found."
                ),
            }

        if occurrences > 1:

            return {
                "success": False,
                "error": (
                    f"old_text matched "
                    f"{occurrences} times. "
                    "Ambiguous edit refused."
                ),
            }

        backup = self.backup(
            target
        )

        updated = content.replace(
            old_text,
            new_text,
            1,
        )

        target.write_text(
            updated,
            encoding="utf-8",
        )

        verification = self.verify(
            target
        )

        success = verification[
            "success"
        ]

        self.log(
            "edit_file",
            target,
            success,
            {
                "backup": (
                    str(backup)
                    if backup
                    else None
                ),
                "verification": verification,
            },
        )

        return {
            "success": success,
            "action": "edit_file",
            "path": str(
                target.relative_to(
                    self.root
                )
            ),
            "replacements": 1,
            "backup": str(
                backup.relative_to(
                    self.root
                )
            ),
            "verification": verification,
        }

    # =========================================================
    # VERIFY
    # =========================================================

    def verify(
        self,
        path,
    ):

        target = Path(
            path
        )

        if not target.exists():

            return {
                "success": False,
                "error": (
                    "File does not exist."
                ),
            }

        if not target.is_file():

            return {
                "success": False,
                "error": (
                    "Path is not a file."
                ),
            }

        try:

            data = target.read_bytes()

            text = data.decode(
                "utf-8"
            )

            return {
                "success": True,
                "bytes": len(data),
                "sha256": hashlib.sha256(
                    data
                ).hexdigest(),
                "lines": len(
                    text.splitlines()
                ),
            }

        except Exception as error:

            return {
                "success": False,
                "error": str(error),
            }

    # =========================================================
    # RECENT BACKUPS
    # =========================================================

    def list_backups(self):

        if not self.backup_root.exists():
            return []

        return [
            str(
                item.relative_to(
                    self.root
                )
            )
            for item in sorted(
                self.backup_root.rglob("*")
            )
            if item.is_file()
        ]