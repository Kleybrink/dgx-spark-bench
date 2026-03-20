"""WIP (work-in-progress) tracker for resumable benchmark runs."""

import json
import logging
import os
import sqlite3

logger = logging.getLogger(__name__)


def _json_default(obj):
    """Handle numpy scalars etc."""
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


class WIPTracker:
    """Manages WIP SQLite state: loading prior rows, tracking completions, appending new rows."""

    def __init__(
        self, wip_dir: str = os.path.join("results", "wip"), quiet: bool = False
    ):
        self.wip_dir = wip_dir
        os.makedirs(wip_dir, exist_ok=True)

        self.db_path = os.path.join(wip_dir, "wip.db")
        self._conn = sqlite3.connect(self.db_path)
        self._init_db()
        self._load_rows(quiet)

    def _init_db(self):
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                think INTEGER NOT NULL,
                prompt_name TEXT NOT NULL,
                run INTEGER NOT NULL,
                data TEXT NOT NULL,
                UNIQUE(model, think, prompt_name, run)
            )
            """
        )
        self._conn.commit()

    def _load_rows(self, quiet):
        cursor = self._conn.execute("SELECT data FROM runs ORDER BY id")
        self.rows: list[dict] = []
        for (data_json,) in cursor.fetchall():
            row = json.loads(data_json)
            if "think" in row:
                row["think"] = bool(row["think"])
            self.rows.append(row)

        self.completed: set[tuple] = {
            (r["model"], bool(r["think"]), str(r["prompt_name"]), int(r["run"]))
            for r in self.rows
        }

        if self.rows and not quiet:
            print(f"Resuming: {len(self.rows)} rows loaded from {self.db_path}")

    def _insert_row(self, row):
        data_json = json.dumps(row, default=_json_default)
        self._conn.execute(
            "INSERT OR IGNORE INTO runs (model, think, prompt_name, run, data) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                row["model"],
                int(bool(row["think"])),
                str(row.get("prompt_name", "default")),
                int(row["run"]),
                data_json,
            ),
        )

    def is_done(self, model: str, think: bool, prompt_name: str, run_idx: int) -> bool:
        return (model, think, prompt_name, run_idx) in self.completed

    def all_done(
        self, model: str, think: bool, prompts: list[dict], runs_per_mode: int
    ) -> bool:
        """Check if all prompt x run combinations are done for a model/think pair."""
        return all(
            self.is_done(model, think, p["name"], ri)
            for p in prompts
            for ri in range(1, runs_per_mode + 1)
        )

    def append(self, row: dict):
        """Record a completed run: add to in-memory list and insert into SQLite."""
        key = (
            row["model"],
            bool(row["think"]),
            str(row["prompt_name"]),
            int(row["run"]),
        )
        self.rows.append(row)
        self.completed.add(key)
        self._insert_row(row)
        self._conn.commit()
        logger.debug(
            "WIP row appended: model=%s think=%s prompt=%s run=%s",
            row.get("model"),
            row.get("think"),
            row.get("prompt_name"),
            row.get("run"),
        )
