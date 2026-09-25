"""Transactional, single-instance release state. No model execution happens here."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import RLock
import uuid

DEFAULT_GUARDRAILS = dict(
    max_error_rate=0.10, max_p95_ms=150, min_samples=20, window_size=100
)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


class Conflict(ValueError):
    pass


class InvalidRelease(ValueError):
    pass


def route_version(policy, key):
    if policy["status"] != "running":
        return policy["champion"]
    digest = hashlib.sha256(f"{policy['rollout_id']}:{key}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return policy["canary"] if fraction < policy["weight"] else policy["champion"]


class PolicyStore:
    def __init__(self, path: Path, models: list[dict]):
        self.models = {m["version"]: m for m in models}
        if not self.models:
            raise InvalidRelease("The verified model registry is empty")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT NOT NULL);
            """)
            with self._transaction():
                if not self.db.execute("SELECT 1 FROM policy WHERE id=1").fetchone():
                    # A new deployment serves the first registered model.
                    self._save(
                        dict(
                            revision=0,
                            champion=next(iter(self.models)),
                            canary=None,
                            weight=0,
                            status="inactive",
                            rollout_id=None,
                            guardrails=DEFAULT_GUARDRAILS.copy(),
                            updated_at=now_iso(),
                            reason=None,
                        )
                    )
                p = self._get()
                if p["champion"] not in self.models:
                    raise InvalidRelease(
                        "Persisted champion is absent from the verified model registry"
                    )
                if p["status"] == "running":
                    self._finish(
                        p,
                        "restart_rollback",
                        "Restart discards unobserved canary; champion restored",
                    )
        except BaseException:
            self.db.close()
            raise

    @contextmanager
    def _transaction(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def _get(self):
        return json.loads(
            self.db.execute("SELECT body FROM policy WHERE id=1").fetchone()[0]
        )

    def get(self):
        with self.lock:
            return self._get()

    def _save(self, p):
        self.db.execute(
            "INSERT OR REPLACE INTO policy(id,body) VALUES(1,?)", (json.dumps(p),)
        )

    def _event(self, kind, message, p, details=None):
        event = dict(
            at=now_iso(),
            type=kind,
            message=message,
            revision=p["revision"],
            details=details or {},
        )
        self.db.execute("INSERT INTO events(body) VALUES(?)", (json.dumps(event),))

    def record_event(self, kind, message, details=None):
        with self._transaction():
            self._event(kind, message, self._get(), details)

    def events(self):
        with self.lock:
            return [
                dict(json.loads(body), id=id_)
                for id_, body in self.db.execute(
                    "SELECT id,body FROM events ORDER BY id DESC LIMIT 100"
                )
            ]

    @staticmethod
    def _check(p, revision, running=False):
        if p["revision"] != revision:
            raise Conflict(
                "Policy changed; refresh and retry with its current revision"
            )
        if running and p["status"] != "running":
            raise Conflict("There is no running canary")

    def start(self, candidate, weight, revision, guardrails=None):
        with self._transaction():
            p = self._get()
            self._check(p, revision)
            if p["status"] == "running":
                raise Conflict("Finish the active rollout before starting another")
            if candidate not in self.models or candidate == p["champion"]:
                raise InvalidRelease(
                    "Select a registered model different from the champion"
                )
            if (
                self.models[candidate]["accuracy"]
                < self.models[p["champion"]]["accuracy"] - 0.03
            ):
                raise InvalidRelease(
                    "Candidate fails holdout quality gate: maximum accuracy loss is 3 percentage points"
                )
            if not 0.05 <= weight <= 0.5:
                raise InvalidRelease("Canary traffic must be between 5% and 50%")
            guard = dict(DEFAULT_GUARDRAILS, **(guardrails or {}))
            if not (
                0 <= guard["max_error_rate"] <= 1
                and guard["max_p95_ms"] > 0
                and 1 <= guard["min_samples"] <= guard["window_size"] <= 10000
            ):
                raise InvalidRelease("Invalid guardrails or sample window")
            p.update(
                revision=p["revision"] + 1,
                canary=candidate,
                weight=weight,
                status="running",
                rollout_id=str(uuid.uuid4()),
                guardrails=guard,
                updated_at=now_iso(),
                reason=None,
            )
            self._save(p)
            self._event(
                "rollout_started",
                f"{candidate} receives {weight:.0%} of traffic",
                p,
                dict(candidate=candidate, guardrails=guard),
            )
            return p

    def _finish(self, p, kind, reason, evidence=None):
        previous = dict(
            champion=p["champion"], canary=p["canary"], rollout_id=p["rollout_id"]
        )
        if kind == "promoted":
            p["champion"] = p["canary"]
        p.update(
            revision=p["revision"] + 1,
            canary=None,
            weight=0,
            status="promoted" if kind == "promoted" else "rolled_back",
            updated_at=now_iso(),
            reason=reason,
        )
        self._save(p)
        self._event(
            kind, reason, p, dict(previous, evidence=evidence) if evidence else previous
        )
        return p

    def rollback(self, revision, reason, automatic=False, evidence=None):
        with self._transaction():
            p = self._get()
            self._check(p, revision, running=True)
            return self._finish(
                p,
                "automatic_rollback" if automatic else "manual_rollback",
                reason,
                evidence,
            )

    def promote(self, revision):
        # Runtime observation gate belongs to the API controller, which calls this without awaiting.
        with self._transaction():
            p = self._get()
            self._check(p, revision, running=True)
            return self._finish(
                p, "promoted", "Healthy canary promoted after minimum observation count"
            )

    def close(self):
        self.db.close()
