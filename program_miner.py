"""Endless independent island-GA program mining for LUTM-1.

Every episode samples one configured task uniformly with replacement, starts
from a new random population, and stores qualifying programs in a compact
SQLite register.  Previously mined programs are never used to seed training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import secrets
import sqlite3
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from island_ga import IslandGAConfig, PopulationFitness, TaichiIslandGA
from taichi_backend import TaichiUTMSimulator, initialize_taichi_cuda
from utils import (
    SimulatorConfig,
    TaskCases,
    effective_program_to_string,
    validate_binary,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "miner_config.json"
DEFAULT_TASKS_PATH = ROOT / "miner_tasks.json"

SIMULATOR_FIELDS = {
    "program_width",
    "left_budget",
    "right_budget",
    "t_max",
}
GA_FIELDS = {
    "islands",
    "population_per_island",
    "elite_fraction",
    "cross_island_fraction",
    "cross_island_interval",
    "random_immigrant_fraction",
    "tournament_delta_thresholds",
    "tournament_best_probabilities",
    "tournament_base_probability",
    "p_crossover",
    "crossover_segments",
    "crossover_segment_min",
    "crossover_segment_max",
    "p_insert",
    "p_delete",
    "p_bit_flip",
    "k_penalty",
}
MINING_FIELDS = {
    "max_generations",
    "post_perfect_generations",
    "min_accuracy",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"configuration file not found: {path}")
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _mapping(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return dict(value)


def _reject_unknown(name: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")


def _merge_section(
    name: str,
    defaults: Mapping[str, Any],
    overrides: object,
    required: set[str],
) -> dict[str, Any]:
    base = dict(defaults)
    _reject_unknown(f"default {name}", base, required)
    missing = required - set(base)
    if missing:
        raise ValueError(f"default {name} is missing fields: {sorted(missing)}")
    update = {} if overrides is None else _mapping(f"{name} overrides", overrides)
    _reject_unknown(f"{name} override", update, required)
    base.update(update)
    return base


def _integer(name: str, value: object, minimum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _real(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class MiningLimits:
    max_generations: int
    post_perfect_generations: int
    min_accuracy: float

    def __post_init__(self) -> None:
        _integer("max_generations", self.max_generations, 0)
        _integer(
            "post_perfect_generations",
            self.post_perfect_generations,
            0,
        )
        threshold = _real("min_accuracy", self.min_accuracy)
        if not 0.0 <= threshold < 1.0:
            raise ValueError("min_accuracy must be in [0, 1)")


@dataclass(frozen=True)
class MiningTask:
    name: str
    cases: TaskCases
    simulator: SimulatorConfig
    ga: IslandGAConfig
    limits: MiningLimits
    definition_json: str
    signature: str


@dataclass(frozen=True)
class MinerSettings:
    database_path: Path
    print_every_generations: int


def _task_snapshot(
    name: str,
    cases: TaskCases,
    simulator: SimulatorConfig,
    ga: IslandGAConfig,
    limits: MiningLimits,
) -> dict[str, Any]:
    return {
        "name": name,
        "inputs": list(cases.inputs),
        "targets": list(cases.targets),
        "simulator": asdict(simulator),
        "ga": asdict(ga),
        "mining": asdict(limits),
    }


def load_miner_setup(
    config_path: Path = DEFAULT_CONFIG_PATH,
    tasks_path: Path = DEFAULT_TASKS_PATH,
) -> tuple[MinerSettings, tuple[MiningTask, ...]]:
    config_path = config_path.resolve()
    tasks_path = tasks_path.resolve()
    config = _read_json_object(config_path)
    _reject_unknown(
        "miner configuration",
        config,
        {"database_path", "print_every_generations", "defaults"},
    )
    if not isinstance(config.get("database_path"), str) or not config[
        "database_path"
    ].strip():
        raise ValueError("database_path must be a nonempty string")
    database_path = Path(config["database_path"])
    if not database_path.is_absolute():
        database_path = config_path.parent / database_path
    settings = MinerSettings(
        database_path=database_path.resolve(),
        print_every_generations=_integer(
            "print_every_generations",
            config.get("print_every_generations"),
            1,
        ),
    )

    defaults = _mapping("defaults", config.get("defaults"))
    _reject_unknown("defaults", defaults, {"simulator", "ga", "mining"})
    default_simulator = _mapping("default simulator", defaults.get("simulator"))
    default_ga = _mapping("default ga", defaults.get("ga"))
    default_mining = _mapping("default mining", defaults.get("mining"))

    task_document = _read_json_object(tasks_path)
    _reject_unknown("task document", task_document, {"tasks"})
    raw_tasks = task_document.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("tasks must be a nonempty JSON array")

    tasks: list[MiningTask] = []
    seen_names: set[str] = set()
    for index, raw_task in enumerate(raw_tasks):
        entry = _mapping(f"task {index}", raw_task)
        _reject_unknown(
            f"task {index}",
            entry,
            {"name", "enabled", "inputs", "targets", "overrides"},
        )
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise TypeError(f"task {index} enabled must be boolean")
        if not enabled:
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"task {index} name must be a nonempty string")
        name = name.strip()
        if name in seen_names:
            raise ValueError(f"duplicate enabled task name: {name!r}")
        seen_names.add(name)

        cases = TaskCases(entry.get("inputs"), entry.get("targets"))
        overrides = _mapping(
            f"task {name} overrides", entry.get("overrides", {})
        )
        _reject_unknown(
            f"task {name} overrides",
            overrides,
            {"simulator", "ga", "mining"},
        )
        simulator_values = _merge_section(
            "simulator",
            default_simulator,
            overrides.get("simulator"),
            SIMULATOR_FIELDS,
        )
        ga_values = _merge_section(
            "ga", default_ga, overrides.get("ga"), GA_FIELDS
        )
        mining_values = _merge_section(
            "mining",
            default_mining,
            overrides.get("mining"),
            MINING_FIELDS,
        )
        simulator = SimulatorConfig(**simulator_values)
        limits = MiningLimits(**mining_values)
        ga_values["tournament_delta_thresholds"] = tuple(
            ga_values["tournament_delta_thresholds"]
        )
        ga_values["tournament_best_probabilities"] = tuple(
            ga_values["tournament_best_probabilities"]
        )
        ga = IslandGAConfig(
            generations=limits.max_generations,
            seed=None,
            print_every=None,
            stop_first_exact=False,
            **ga_values,
        )
        cases.validate_right_budget(simulator.right_budget)
        snapshot = _task_snapshot(name, cases, simulator, ga, limits)
        definition_json = _canonical_json(snapshot)
        signature = hashlib.sha256(definition_json.encode("utf-8")).hexdigest()
        tasks.append(
            MiningTask(
                name=name,
                cases=cases,
                simulator=simulator,
                ga=ga,
                limits=limits,
                definition_json=definition_json,
                signature=signature,
            )
        )
    if not tasks:
        raise ValueError("at least one task must be enabled")
    return settings, tuple(tasks)


def evaluate_miner_fitness(
    simulator: Any,
    programs: np.ndarray,
    task: TaskCases,
    *,
    k_penalty: float,
) -> PopulationFitness:
    """Evaluate bit accuracy while assigning zero matches to invalid cases."""

    raw = np.asarray(programs)
    if raw.ndim != 2 or raw.shape[0] == 0:
        raise ValueError("programs must be a nonempty two-dimensional array")
    if raw.shape[1] != simulator.config.program_width:
        raise ValueError("program width disagrees with the simulator")
    if raw.shape[0] > simulator.batch_capacity:
        raise ValueError("program population exceeds simulator batch_capacity")
    penalty = _real("k_penalty", k_penalty)
    if penalty < 0.0:
        raise ValueError("k_penalty must be non-negative")
    task.validate_right_budget(simulator.config.right_budget)

    population = raw.shape[0]
    matching_bits = np.zeros(population, dtype=np.int64)
    compared_bits = np.zeros(population, dtype=np.int64)
    invalid_cases = np.zeros(population, dtype=np.uint32)
    exact = np.ones(population, dtype=np.bool_)

    for input_bits, target in task.pairs():
        result = simulator.simulate_program_batch_for_fitness(raw, input_bits)
        target_bits = np.fromiter(
            (int(bit) for bit in target),
            dtype=np.uint8,
            count=len(target),
        )
        output_lengths = result.output_lengths.astype(np.int64, copy=False)
        overlap = np.minimum(output_lengths, len(target))
        positions = np.arange(len(target), dtype=np.int64)[None, :]
        matches = (
            (positions < overlap[:, None])
            & (result.output_bits[:, : len(target)] == target_bits[None, :])
            & (~result.invalid[:, None])
        )
        matching_bits += matches.sum(axis=1, dtype=np.int64)
        compared_bits += np.maximum(output_lengths, len(target))
        invalid_cases += result.invalid.astype(np.uint32)
        exact &= (
            ~result.invalid
            & (output_lengths == len(target))
            & np.all(
                result.output_bits[:, : len(target)] == target_bits[None, :],
                axis=1,
            )
        )

    bit_accuracy = matching_bits / compared_bits
    invalid_fraction = invalid_cases / len(task)
    return PopulationFitness(
        fitness=bit_accuracy - penalty * invalid_fraction,
        bit_accuracy=bit_accuracy,
        invalid_fraction=invalid_fraction,
        exact=exact,
    )


class MiningIslandGA(TaichiIslandGA):
    """The established island GA with miner-specific invalid-case scoring."""

    def _evaluate(self) -> PopulationFitness:
        if self.programs is None:
            raise RuntimeError("population is not initialized")
        flat = self.programs.reshape(
            self.config.total_population,
            self.simulator.config.program_width,
        )
        evaluated = evaluate_miner_fitness(
            self.simulator,
            flat,
            self.task,
            k_penalty=self.config.k_penalty,
        )
        shape = (self.config.islands, self.config.population_per_island)
        return PopulationFitness(
            fitness=evaluated.fitness.reshape(shape),
            bit_accuracy=evaluated.bit_accuracy.reshape(shape),
            invalid_fraction=evaluated.invalid_fraction.reshape(shape),
            exact=evaluated.exact.reshape(shape),
        )


def pack_program(program: str) -> tuple[int, bytes]:
    validate_binary("program", program)
    length = len(program)
    size = (length + 7) // 8
    packed = b"" if size == 0 else int(program, 2).to_bytes(size, "big")
    return length, packed


def unpack_program(length: int, packed: bytes) -> str:
    length = _integer("program length", length, 0)
    expected = (length + 7) // 8
    raw = bytes(packed)
    if len(raw) != expected:
        raise ValueError("packed program length is inconsistent")
    if length == 0:
        return ""
    value = int.from_bytes(raw, "big")
    if value >= (1 << length):
        raise ValueError("packed program contains bits outside its length")
    return format(value, f"0{length}b")


@dataclass(frozen=True)
class Candidate:
    program: str
    accuracy: float
    exact: bool


@dataclass(frozen=True)
class ArchiveResult:
    distinct_in_batch: int
    distinct_in_run: int
    new_programs_added: int


@dataclass(frozen=True)
class RunHandle:
    number: int
    uid: str
    task_id: int


class ProgramRegistry:
    """Compact, transactional and deduplicated SQLite program register."""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = path.resolve()
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.path.is_file():
                raise FileNotFoundError(f"program database not found: {self.path}")
            self.connection = sqlite3.connect(
                self.path.as_uri() + "?mode=ro",
                uri=True,
                timeout=30.0,
            )
            self.connection.execute("PRAGMA query_only = ON")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, timeout=30.0)
            self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        if not self.read_only:
            self.connection.execute("PRAGMA synchronous = NORMAL")
            self._create_schema()

    def __enter__(self) -> "ProgramRegistry":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is not None:
            self.connection.rollback()
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id INTEGER PRIMARY KEY,
                signature TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                definition_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_number INTEGER PRIMARY KEY,
                run_uid TEXT NOT NULL UNIQUE,
                task_id INTEGER NOT NULL REFERENCES tasks(task_id),
                started_at TEXT NOT NULL,
                ended_at TEXT,
                generations_completed INTEGER,
                programs_found INTEGER,
                new_programs_added INTEGER,
                best_accuracy REAL,
                first_exact_generation INTEGER,
                stop_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS programs (
                program_id INTEGER PRIMARY KEY,
                bit_length INTEGER NOT NULL,
                packed_bits BLOB NOT NULL,
                UNIQUE(bit_length, packed_bits)
            );

            CREATE TABLE IF NOT EXISTS discoveries (
                task_id INTEGER NOT NULL REFERENCES tasks(task_id),
                program_id INTEGER NOT NULL REFERENCES programs(program_id),
                first_run_number INTEGER NOT NULL REFERENCES runs(run_number),
                generation INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                exact INTEGER NOT NULL CHECK(exact IN (0, 1)),
                PRIMARY KEY(task_id, program_id)
            ) WITHOUT ROWID;

            CREATE INDEX IF NOT EXISTS discoveries_by_program
                ON discoveries(program_id);

            CREATE TEMP TABLE IF NOT EXISTS candidate_batch (
                bit_length INTEGER NOT NULL,
                packed_bits BLOB NOT NULL,
                accuracy REAL NOT NULL,
                exact INTEGER NOT NULL,
                generation INTEGER NOT NULL,
                PRIMARY KEY(bit_length, packed_bits)
            ) WITHOUT ROWID;

            CREATE TEMP TABLE IF NOT EXISTS run_seen (
                bit_length INTEGER NOT NULL,
                packed_bits BLOB NOT NULL,
                PRIMARY KEY(bit_length, packed_bits)
            ) WITHOUT ROWID;
            """
        )
        self.connection.commit()

    def register_task(self, task: MiningTask) -> int:
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO tasks(signature, name, definition_json)
                VALUES (?, ?, ?)
                """,
                (task.signature, task.name, task.definition_json),
            )
            row = self.connection.execute(
                "SELECT task_id FROM tasks WHERE signature = ?",
                (task.signature,),
            ).fetchone()
        if row is None:
            raise RuntimeError("failed to register task")
        return int(row[0])

    def start_run(self, task_id: int) -> RunHandle:
        uid = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-"
            + secrets.token_hex(3)
        )
        with self.connection:
            self.connection.execute("DELETE FROM run_seen")
            cursor = self.connection.execute(
                """
                INSERT INTO runs(run_uid, task_id, started_at)
                VALUES (?, ?, ?)
                """,
                (uid, task_id, _utc_now()),
            )
        return RunHandle(int(cursor.lastrowid), uid, task_id)

    def archive(
        self,
        run: RunHandle,
        generation: int,
        candidates: Sequence[Candidate],
    ) -> ArchiveResult:
        unique: dict[str, Candidate] = {}
        for candidate in candidates:
            previous = unique.get(candidate.program)
            if previous is None or candidate.accuracy > previous.accuracy:
                unique[candidate.program] = candidate
        rows = []
        for candidate in unique.values():
            length, packed = pack_program(candidate.program)
            rows.append(
                (
                    length,
                    sqlite3.Binary(packed),
                    float(candidate.accuracy),
                    int(candidate.exact),
                    int(generation),
                )
            )
        if not rows:
            found = int(
                self.connection.execute("SELECT COUNT(*) FROM run_seen").fetchone()[0]
            )
            return ArchiveResult(0, found, 0)

        with self.connection:
            self.connection.execute("DELETE FROM candidate_batch")
            self.connection.executemany(
                """
                INSERT OR IGNORE INTO candidate_batch(
                    bit_length, packed_bits, accuracy, exact, generation
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            self.connection.execute(
                """
                INSERT OR IGNORE INTO run_seen(bit_length, packed_bits)
                SELECT bit_length, packed_bits FROM candidate_batch
                """
            )
            self.connection.execute(
                """
                INSERT OR IGNORE INTO programs(bit_length, packed_bits)
                SELECT bit_length, packed_bits FROM candidate_batch
                """
            )
            self.connection.execute(
                """
                INSERT OR IGNORE INTO discoveries(
                    task_id, program_id, first_run_number,
                    generation, accuracy, exact
                )
                SELECT ?, p.program_id, ?, c.generation, c.accuracy, c.exact
                FROM candidate_batch AS c
                JOIN programs AS p
                  ON p.bit_length = c.bit_length
                 AND p.packed_bits = c.packed_bits
                """,
                (run.task_id, run.number),
            )
            added = int(self.connection.execute("SELECT changes()").fetchone()[0])
            found = int(
                self.connection.execute("SELECT COUNT(*) FROM run_seen").fetchone()[0]
            )
        return ArchiveResult(len(rows), found, added)

    def finish_run(
        self,
        run: RunHandle,
        *,
        generations_completed: int,
        programs_found: int,
        new_programs_added: int,
        best_accuracy: float,
        first_exact_generation: int | None,
        stop_reason: str,
    ) -> None:
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE runs
                   SET ended_at = ?, generations_completed = ?,
                       programs_found = ?, new_programs_added = ?,
                       best_accuracy = ?, first_exact_generation = ?,
                       stop_reason = ?
                 WHERE run_number = ? AND ended_at IS NULL
                """,
                (
                    _utc_now(),
                    generations_completed,
                    programs_found,
                    new_programs_added,
                    best_accuracy,
                    first_exact_generation,
                    stop_reason,
                    run.number,
                ),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("run was not active when finishing it")

    def lookup_program(self, program: str) -> list[dict[str, Any]]:
        length, packed = pack_program(program)
        rows = self.connection.execute(
            """
            SELECT t.name, t.signature, r.run_number, r.run_uid,
                   r.started_at, d.generation, d.accuracy, d.exact
              FROM programs AS p
              JOIN discoveries AS d ON d.program_id = p.program_id
              JOIN tasks AS t ON t.task_id = d.task_id
              JOIN runs AS r ON r.run_number = d.first_run_number
             WHERE p.bit_length = ? AND p.packed_bits = ?
             ORDER BY t.name, r.run_number
            """,
            (length, sqlite3.Binary(packed)),
        ).fetchall()
        return [
            {
                "program": program,
                "task": str(row[0]),
                "task_signature": str(row[1]),
                "run_number": int(row[2]),
                "run_id": str(row[3]),
                "run_started_at": str(row[4]),
                "generation": int(row[5]),
                "accuracy": float(row[6]),
                "exact": bool(row[7]),
            }
            for row in rows
        ]

    def statistics(self, recent: int = 10) -> dict[str, Any]:
        recent = _integer("recent", recent, 0)
        scalar_queries = {
            "unique_programs": "SELECT COUNT(*) FROM programs",
            "task_program_associations": "SELECT COUNT(*) FROM discoveries",
            "exact_associations": (
                "SELECT COUNT(*) FROM discoveries WHERE exact = 1"
            ),
            "runs": "SELECT COUNT(*) FROM runs",
            "completed_runs": (
                """SELECT COUNT(*) FROM runs
                    WHERE ended_at IS NOT NULL
                      AND stop_reason != 'interrupted'
                      AND stop_reason NOT LIKE 'error:%'"""
            ),
            "active_runs": (
                "SELECT COUNT(*) FROM runs WHERE ended_at IS NULL"
            ),
            "interrupted_runs": (
                "SELECT COUNT(*) FROM runs WHERE stop_reason = 'interrupted'"
            ),
            "error_runs": (
                "SELECT COUNT(*) FROM runs WHERE stop_reason LIKE 'error:%'"
            ),
        }
        totals = {
            name: int(self.connection.execute(query).fetchone()[0])
            for name, query in scalar_queries.items()
        }

        task_rows = self.connection.execute(
            """
            WITH run_stats AS (
                SELECT t.name AS name,
                       COUNT(r.run_number) AS runs,
                       SUM(CASE WHEN r.ended_at IS NOT NULL
                                     AND r.stop_reason != 'interrupted'
                                     AND r.stop_reason NOT LIKE 'error:%'
                                THEN 1 ELSE 0 END)
                           AS completed_runs,
                       MAX(COALESCE(r.ended_at, r.started_at)) AS last_activity
                  FROM tasks AS t
                  LEFT JOIN runs AS r ON r.task_id = t.task_id
                 GROUP BY t.name
            ),
            discovery_stats AS (
                SELECT t.name AS name,
                       COUNT(DISTINCT d.program_id) AS programs,
                       COUNT(DISTINCT CASE WHEN d.exact = 1
                                           THEN d.program_id END)
                           AS exact_programs,
                       MAX(d.accuracy) AS best_accuracy,
                       AVG(d.accuracy) AS average_accuracy
                  FROM tasks AS t
                  LEFT JOIN discoveries AS d ON d.task_id = t.task_id
                 GROUP BY t.name
            )
            SELECT r.name, r.runs, r.completed_runs,
                   d.programs, d.exact_programs,
                   d.best_accuracy, d.average_accuracy, r.last_activity
              FROM run_stats AS r
              JOIN discovery_stats AS d ON d.name = r.name
             ORDER BY r.name
            """
        ).fetchall()
        per_task = [
            {
                "task": str(row[0]),
                "runs": int(row[1]),
                "completed_runs": int(row[2]),
                "programs": int(row[3]),
                "exact_programs": int(row[4]),
                "best_accuracy": (
                    None if row[5] is None else float(row[5])
                ),
                "average_accuracy": (
                    None if row[6] is None else float(row[6])
                ),
                "last_activity": None if row[7] is None else str(row[7]),
            }
            for row in task_rows
        ]

        recent_rows: list[dict[str, Any]] = []
        if recent:
            rows = self.connection.execute(
                """
                SELECT r.run_number, r.run_uid, t.name,
                       r.generations_completed, r.programs_found,
                       r.new_programs_added, r.best_accuracy,
                       r.first_exact_generation, r.stop_reason,
                       r.started_at, r.ended_at
                  FROM runs AS r
                  JOIN tasks AS t ON t.task_id = r.task_id
                 ORDER BY r.run_number DESC
                 LIMIT ?
                """,
                (recent,),
            ).fetchall()
            recent_rows = [
                {
                    "run_number": int(row[0]),
                    "run_id": str(row[1]),
                    "task": str(row[2]),
                    "generations": None if row[3] is None else int(row[3]),
                    "programs_found": None if row[4] is None else int(row[4]),
                    "new_programs_added": (
                        None if row[5] is None else int(row[5])
                    ),
                    "best_accuracy": (
                        None if row[6] is None else float(row[6])
                    ),
                    "first_exact_generation": (
                        None if row[7] is None else int(row[7])
                    ),
                    "stop_reason": None if row[8] is None else str(row[8]),
                    "started_at": str(row[9]),
                    "ended_at": None if row[10] is None else str(row[10]),
                }
                for row in rows
            ]

        database_bytes = sum(
            path.stat().st_size
            for path in (
                self.path,
                Path(str(self.path) + "-wal"),
                Path(str(self.path) + "-shm"),
            )
            if path.exists()
        )
        return {
            **totals,
            "database_bytes": database_bytes,
            "tasks": per_task,
            "tasks_with_exact_programs": [
                item["task"] for item in per_task if item["exact_programs"] > 0
            ],
            "recent_runs": recent_rows,
        }


def _format_bytes(size: int) -> str:
    value = float(size)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def print_statistics(statistics: Mapping[str, Any]) -> None:
    print("LUTM-1 program miner statistics")
    print(
        f"Unique programs: {statistics['unique_programs']:,}\n"
        f"Task-program associations: "
        f"{statistics['task_program_associations']:,}\n"
        f"Exact associations: {statistics['exact_associations']:,}\n"
        f"Runs: {statistics['runs']:,} "
        f"({statistics['completed_runs']:,} completed, "
        f"{statistics['active_runs']:,} active, "
        f"{statistics['interrupted_runs']:,} interrupted, "
        f"{statistics['error_runs']:,} errors)\n"
        f"Database size: {_format_bytes(statistics['database_bytes'])}"
    )

    print("\nPer task")
    print(
        f"{'task':<22} {'runs':>7} {'programs':>10} {'exact':>8} "
        f"{'best':>9} {'average':>9}"
    )
    for item in statistics["tasks"]:
        best = (
            "-"
            if item["best_accuracy"] is None
            else f"{item['best_accuracy']:.6f}"
        )
        average = (
            "-"
            if item["average_accuracy"] is None
            else f"{item['average_accuracy']:.6f}"
        )
        print(
            f"{item['task']:<22} {item['runs']:>7,} "
            f"{item['programs']:>10,} {item['exact_programs']:>8,} "
            f"{best:>9} {average:>9}"
        )

    exact_tasks = statistics["tasks_with_exact_programs"]
    print("\nTasks admitting at least one exact mined program")
    print(", ".join(exact_tasks) if exact_tasks else "None yet")

    if statistics["recent_runs"]:
        print("\nRecent runs")
        for run in statistics["recent_runs"]:
            accuracy = (
                "-"
                if run["best_accuracy"] is None
                else f"{run['best_accuracy']:.6f}"
            )
            print(
                f"#{run['run_number']} {run['task']} "
                f"generations={run['generations']} "
                f"found={run['programs_found']} "
                f"new={run['new_programs_added']} "
                f"best={accuracy} reason={run['stop_reason'] or 'active'}"
            )


def _qualifying_candidates(
    trainer: MiningIslandGA,
    fitness: PopulationFitness,
    threshold: float,
) -> list[Candidate]:
    if trainer.programs is None:
        raise RuntimeError("population is not initialized")
    flat_programs = trainer.programs.reshape(
        trainer.config.total_population,
        trainer.simulator.config.program_width,
    )
    accuracy = fitness.bit_accuracy.reshape(-1)
    invalid = fitness.invalid_fraction.reshape(-1)
    exact = fitness.exact.reshape(-1)
    indices = np.flatnonzero((accuracy > threshold) & (invalid == 0.0))
    table = trainer.simulator.table
    return [
        Candidate(
            program=effective_program_to_string(
                flat_programs[index],
                blank_id=table.blank_id,
                zero_id=table.zero_id,
                one_id=table.one_id,
            ),
            accuracy=float(accuracy[index]),
            exact=bool(exact[index]),
        )
        for index in indices
    ]


class ProgramMiner:
    def __init__(
        self,
        settings: MinerSettings,
        tasks: Sequence[MiningTask],
        registry: ProgramRegistry,
    ) -> None:
        if not tasks:
            raise ValueError("tasks must not be empty")
        self.settings = settings
        self.tasks = tuple(tasks)
        self.registry = registry
        self.task_rng = np.random.default_rng()
        self.simulators: dict[
            tuple[SimulatorConfig, int], TaichiUTMSimulator
        ] = {}
        self.task_ids = {
            task.signature: registry.register_task(task) for task in self.tasks
        }

    def _simulator(self, task: MiningTask) -> TaichiUTMSimulator:
        key = (task.simulator, task.ga.total_population)
        simulator = self.simulators.get(key)
        if simulator is None:
            simulator = TaichiUTMSimulator(
                task.simulator,
                batch_capacity=task.ga.total_population,
            )
            self.simulators[key] = simulator
        return simulator

    @staticmethod
    def _summary_line(
        run: RunHandle,
        task: MiningTask,
        generation: int,
        programs_found: int,
        new_programs_added: int,
        best_accuracy: float,
        first_exact: int | None,
        reason: str,
        elapsed: float,
    ) -> str:
        exact_text = "none" if first_exact is None else str(first_exact)
        return (
            f"run={run.number} id={run.uid} task={task.name} "
            f"generations={generation} programs_found={programs_found:,} "
            f"new_programs={new_programs_added:,} "
            f"best_accuracy={best_accuracy:.6f} "
            f"first_exact={exact_text} reason={reason} "
            f"elapsed={elapsed:.3f}s"
        )

    def _run_task(self, task: MiningTask) -> None:
        task_id = self.task_ids[task.signature]
        run = self.registry.start_run(task_id)
        started = perf_counter()
        generation = 0
        programs_found = 0
        new_programs_added = 0
        best_accuracy = 0.0
        first_exact: int | None = None
        reason = "generation-budget"

        print(
            f"run={run.number} id={run.uid} task={task.name} starting "
            f"population={task.ga.total_population:,} "
            f"max_generations={task.limits.max_generations:,}",
            flush=True,
        )
        try:
            trainer = MiningIslandGA(
                self._simulator(task), task.cases, task.ga
            )
            trainer._initialize(None)
            while True:
                if generation > 0:
                    if trainer.fitness is None:
                        raise RuntimeError("previous generation fitness is missing")
                    trainer._make_next_generation(generation, trainer.fitness)
                fitness = trainer._evaluate()
                trainer.fitness = fitness
                best_accuracy = max(
                    best_accuracy, float(np.max(fitness.bit_accuracy))
                )
                candidates = _qualifying_candidates(
                    trainer, fitness, task.limits.min_accuracy
                )
                archived = self.registry.archive(
                    run, generation, candidates
                )
                programs_found = archived.distinct_in_run
                new_programs_added += archived.new_programs_added

                if first_exact is None and np.any(fitness.exact):
                    first_exact = generation
                perfect_tail_done = (
                    first_exact is not None
                    and generation
                    >= first_exact + task.limits.post_perfect_generations
                )
                budget_done = generation >= task.limits.max_generations
                terminal = perfect_tail_done or budget_done
                if perfect_tail_done:
                    reason = "perfect-tail"

                if (
                    generation % self.settings.print_every_generations == 0
                    or terminal
                ):
                    print(
                        f"run={run.number} task={task.name} "
                        f"generation={generation:,} "
                        f"best_accuracy={best_accuracy:.6f} "
                        f"qualifying={archived.distinct_in_batch:,} "
                        f"programs_found={programs_found:,} "
                        f"new_programs={new_programs_added:,} "
                        f"exact={int(np.count_nonzero(fitness.exact)):,}",
                        flush=True,
                    )
                if terminal:
                    break
                generation += 1
        except KeyboardInterrupt:
            reason = "interrupted"
            self.registry.finish_run(
                run,
                generations_completed=generation,
                programs_found=programs_found,
                new_programs_added=new_programs_added,
                best_accuracy=best_accuracy,
                first_exact_generation=first_exact,
                stop_reason=reason,
            )
            print(
                self._summary_line(
                    run,
                    task,
                    generation,
                    programs_found,
                    new_programs_added,
                    best_accuracy,
                    first_exact,
                    reason,
                    perf_counter() - started,
                ),
                flush=True,
            )
            raise
        except Exception as error:
            reason = f"error:{type(error).__name__}"
            self.registry.finish_run(
                run,
                generations_completed=generation,
                programs_found=programs_found,
                new_programs_added=new_programs_added,
                best_accuracy=best_accuracy,
                first_exact_generation=first_exact,
                stop_reason=reason,
            )
            raise

        self.registry.finish_run(
            run,
            generations_completed=generation,
            programs_found=programs_found,
            new_programs_added=new_programs_added,
            best_accuracy=best_accuracy,
            first_exact_generation=first_exact,
            stop_reason=reason,
        )
        print(
            self._summary_line(
                run,
                task,
                generation,
                programs_found,
                new_programs_added,
                best_accuracy,
                first_exact,
                reason,
                perf_counter() - started,
            ),
            flush=True,
        )

    def run_forever(self, max_runs: int | None = None) -> None:
        completed = 0
        try:
            while max_runs is None or completed < max_runs:
                task = self.tasks[
                    int(self.task_rng.integers(0, len(self.tasks)))
                ]
                self._run_task(task)
                completed += 1
        except KeyboardInterrupt:
            print("Program miner stopped cleanly.", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine LUTM-1 programs forever")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument(
        "--max-runs",
        type=int,
        help="run a finite number of episodes; default is unlimited",
    )
    query = parser.add_mutually_exclusive_group()
    query.add_argument(
        "--find",
        metavar="PROGRAM",
        help="show the tasks and first runs associated with one program",
    )
    query.add_argument(
        "--stats",
        action="store_true",
        help="show database and per-task mining statistics",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=10,
        help="number of recent runs shown by --stats; default: 10",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.max_runs is not None and arguments.max_runs < 1:
        raise ValueError("--max-runs must be positive")
    if arguments.recent < 0:
        raise ValueError("--recent must be non-negative")
    if arguments.max_runs is not None and (
        arguments.find is not None or arguments.stats
    ):
        raise ValueError("--max-runs cannot be combined with query modes")
    settings, tasks = load_miner_setup(arguments.config, arguments.tasks)
    query_mode = arguments.find is not None or arguments.stats
    with ProgramRegistry(
        settings.database_path,
        read_only=query_mode,
    ) as registry:
        if arguments.find is not None:
            print(
                json.dumps(
                    registry.lookup_program(arguments.find),
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            return 0
        if arguments.stats:
            print_statistics(registry.statistics(arguments.recent))
            return 0
        initialize_taichi_cuda()
        ProgramMiner(settings, tasks, registry).run_forever(
            arguments.max_runs
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
