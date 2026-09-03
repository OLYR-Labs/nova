from pathlib import Path
from datetime import datetime
import shutil


class SandboxFilesystem:

    def __init__(
        self,
        sandbox_root: str,
    ):

        self.root = Path(
            sandbox_root
        ).resolve()

        if not self.root.exists():

            self.root.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.backup_root = (
            self.root / ".nova_backups"
        )

        self.backup_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Files/directories NOVA must never modify.
        self.blocked_names = {
            ".env",
            ".venv",
            "venv",
            "__pycache__",
            ".git",
            ".nova_backups",
        }

        self.blocked_extensions = {
            ".key",
            ".pem",
            ".p12",
            ".pfx",
        }

    # =========================================================
    # PATH SECURITY
    # =========================================================

    def _safe_path(
        self,
        path: str,
    ) -> Path:

        requested = (
            self.root / path
        ).resolve()

        try:

            requested.relative_to(
                self.root
            )

        except ValueError:

            raise PermissionError(
                "Path is outside the NOVA development boundary."
            )

        self._check_blocked(
            requested
        )

        return requested

    def _check_blocked(
        self,
        path: Path,
    ):

        try:

            relative = path.relative_to(
                self.root
            )

        except ValueError:

            raise PermissionError(
                "Path is outside the NOVA development boundary."
            )

        for part in relative.parts:

            if part in self.blocked_names:

                raise PermissionError(
                    f"Access to '{part}' is blocked."
                )

        if path.suffix.lower() in (
            self.blocked_extensions
        ):

            raise PermissionError(
                f"File type '{path.suffix}' is blocked."
            )

    # =========================================================
    # LIST FILES
    # =========================================================

    def list_files(
        self,
        path: str = ".",
    ):

        directory = self._safe_path(
            path
        )

        if not directory.exists():

            raise FileNotFoundError(
                f"Path does not exist: {path}"
            )

        if not directory.is_dir():

            raise NotADirectoryError(
                f"Not a directory: {path}"
            )

        results = []

        for item in sorted(
            directory.iterdir(),
            key=lambda x: x.name.lower(),
        ):

            # Don't expose backup contents.
            if item.name in self.blocked_names:
                continue

            results.append({
                "name": item.name,
                "type": (
                    "directory"
                    if item.is_dir()
                    else "file"
                ),
            })

        return results

    # =========================================================
    # READ FILE
    # =========================================================

    def read_file(
        self,
        path: str,
    ):

        file_path = self._safe_path(
            path
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"File does not exist: {path}"
            )

        if not file_path.is_file():

            raise IsADirectoryError(
                f"Not a file: {path}"
            )

        content = file_path.read_text(
            encoding="utf-8"
        )

        # -----------------------------------------------------
        # LINE-NUMBERED OUTPUT
        # -----------------------------------------------------

        lines = content.splitlines()

        numbered_lines = []

        for line_number, line in enumerate(
            lines,
            start=1,
        ):

            numbered_lines.append(
                f"{line_number:04d} | {line}"
            )

        return "\n".join(
            numbered_lines
        )

    # =========================================================
    # BACKUP
    # =========================================================

    def _backup_file(
        self,
        file_path: Path,
    ):

        if not file_path.exists():
            return None

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        relative = file_path.relative_to(
            self.root
        )

        backup_path = (
            self.backup_root
            / timestamp
            / relative
        )

        backup_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            file_path,
            backup_path,
        )

        return str(
            backup_path.relative_to(
                self.root
            )
        )

    # =========================================================
    # WRITE FILE
    # =========================================================

    def write_file(
        self,
        path: str,
        content: str,
    ):

        file_path = self._safe_path(
            path
        )

        if (
            file_path.exists()
            and not file_path.is_file()
        ):

            raise IsADirectoryError(
                f"Not a file: {path}"
            )

        backup = self._backup_file(
            file_path
        )

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path.write_text(
            content,
            encoding="utf-8",
        )

        return {
            "status": "written",
            "path": str(
                file_path.relative_to(
                    self.root
                )
            ),
            "backup": backup,
        }

    # =========================================================
    # EDIT FILE
    # =========================================================

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
    ):

        file_path = self._safe_path(
            path
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"File does not exist: {path}"
            )

        if not file_path.is_file():

            raise IsADirectoryError(
                f"Not a file: {path}"
            )

        content = file_path.read_text(
            encoding="utf-8"
        )

        occurrences = content.count(
            old_text
        )

        if occurrences == 0:

            raise ValueError(
                "The requested text was not found."
            )

        if occurrences > 1:

            raise ValueError(
                f"The requested text occurs {occurrences} times. "
                "Refusing ambiguous edit."
            )

        backup = self._backup_file(
            file_path
        )

        updated = content.replace(
            old_text,
            new_text,
            1,
        )

        file_path.write_text(
            updated,
            encoding="utf-8",
        )

        return {
            "status": "edited",
            "path": str(
                file_path.relative_to(
                    self.root
                )
            ),
            "backup": backup,
        }