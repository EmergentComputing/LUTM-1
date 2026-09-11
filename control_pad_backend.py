"""Live, resumable island-GA backend for the LUTM-1 control pad.

This module is deliberately separate from the research notebooks and their
backends.  It reuses the verified GA operators, adds a lightweight diagnostic
CUDA result, and exposes one-generation-at-a-time execution with atomic
checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from island_ga import (
    IslandChampion,
    IslandGAConfig,
    PopulationFitness,
    ReproductionStats,
    TaichiIslandGA,
    evaluate_population_fitness_taichi,
)
from lutm import CANONICAL_CSV_SHA256, InvalidReason
from taichi_backend import TaichiFitnessBatch, TaichiUTMSimulator
from utils import (
    SimulatorConfig,
    TaskCases,
    effective_program_lengths,
)


CHECKPOINT_SCHEMA = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


DEFAULT_DEFINITION: dict[str, Any] = {
    "run_id": "",
    "train_inputs": ["00", "11", "01", "10", "1", "0"],
    "train_targets": ["0000", "1111", "0101", "1010", "11", "00"],
    "test_inputs": ["001", "101", "111"],
    "test_targets": ["001001", "101101", "111111"],
    "simulator": {
        "program_width": 400,
        "left_budget": 410,
        "right_budget": 30,
        "t_max": 48_000,
    },
    "structural": {
        "islands": 200,
        "population_per_island": 800,
        "seed": 4,
        "k_penalty": 0.5,
    },
    "mutable": {
        "generation_target": 16_000,
        "elite_fraction": 0.03,
        "cross_island_fraction": 0.02,
        "cross_island_interval": 1_200,
        "random_immigrant_fraction": 0.15,
        "tournament_delta_thresholds": [0.05, 0.20],
        "tournament_best_probabilities": [0.53, 0.58],
        "tournament_base_probability": 0.80,
        "p_crossover": 0.50,
        "crossover_segments": 3,
        "crossover_segment_min": 1,
        "crossover_segment_max": 20,
        "p_insert": 0.04,
        "p_delete": 0.06,
        "p_bit_flip": 0.02,
        "stop_first_exact": True,
        "diagnostic_interval": 10,
        "checkpoint_interval": 100,
    },
}


@dataclass(frozen=True, eq=False)
class ControlPadBatch(TaichiFitnessBatch):
    """Fitness fields plus counters, without copying the final tape."""

    T: np.ndarray
    left_space_used: np.ndarray
    right_space_used: np.ndarray
    halted: np.ndarray


class ControlPadTaichiSimulator(TaichiUTMSimulator):
    """Add a small diagnostics transfer to the unchanged Taichi simulator."""

    def _control_pad_result(self, size: int) -> ControlPadBatch:
        reasons = self.invalid_reasons.to_numpy()[:size].copy()
        return ControlPadBatch(
            output_bits=self.output_bits.to_numpy()[:size].copy(),
            output_lengths=self.output_lengths.to_numpy()[:size].copy(),
            invalid=reasons != int(InvalidReason.NONE),
            invalid_reason=reasons,
            T=self.transition_counts.to_numpy()[:size].copy(),
            left_space_used=self.left_space_used.to_numpy()[:size].copy(),
            right_space_used=self.right_space_used.to_numpy()[:size].copy(),
            halted=self.halted.to_numpy()[:size].astype(bool, copy=True),
        )

    def simulate_program_batch_for_control_pad(
        self,
        programs,
        input_bits: str,
    ) -> ControlPadBatch:
        """Run one padded batch and transfer only fitness/resource fields."""

        population = self._launch_explicit(programs, input_bits)
        return self._control_pad_result(population)


@dataclass(frozen=True, eq=False)
class PopulationDiagnostics:
    fitness: PopulationFitness
    halted_counts: np.ndarray
    mean_halted_T: np.ndarray
    mean_halted_space: np.ndarray


@dataclass(frozen=True)
class RunDefinition:
    run_id: str
    simulator: SimulatorConfig
    train: TaskCases
    test: TaskCases | None
    islands: int
    population_per_island: int
    seed: int | None
    k_penalty: float

    @property
    def total_population(self) -> int:
        return self.islands * self.population_per_island

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "simulator": asdict(self.simulator),
            "train_inputs": list(self.train.inputs),
            "train_targets": list(self.train.targets),
            "test_inputs": [] if self.test is None else list(self.test.inputs),
            "test_targets": [] if self.test is None else list(self.test.targets),
            "structural": {
                "islands": self.islands,
                "population_per_island": self.population_per_island,
                "seed": self.seed,
                "k_penalty": self.k_penalty,
            },
        }


@dataclass(frozen=True)
class LiveSettings:
    generation_target: int
    elite_fraction: float
    cross_island_fraction: float
    cross_island_interval: int
    random_immigrant_fraction: float
    tournament_delta_thresholds: tuple[float, ...]
    tournament_best_probabilities: tuple[float, ...]
    tournament_base_probability: float
    p_crossover: float
    crossover_segments: int
    crossover_segment_min: int
    crossover_segment_max: int
    p_insert: float
    p_delete: float
    p_bit_flip: float
    stop_first_exact: bool
    diagnostic_interval: int
    checkpoint_interval: int

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["tournament_delta_thresholds"] = list(
            self.tournament_delta_thresholds
        )
        result["tournament_best_probabilities"] = list(
            self.tournament_best_probabilities
        )
        return result


def suggested_run_id() -> str:
    return datetime.now().strftime("run-%Y%m%d-%H%M%S")


def validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with a letter or digit and contain only "
            "letters, digits, underscores, or hyphens (maximum 80 characters)"
        )
    return run_id


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _require_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if converted < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return converted


def _require_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


def _require_string_list(name: str, value: object) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise TypeError(f"{name} must be a list of strings")
    return value


def parse_run_definition(payload: Mapping[str, Any]) -> RunDefinition:
    simulator_raw = _require_mapping("simulator", payload.get("simulator"))
    structural = _require_mapping("structural", payload.get("structural"))
    simulator = SimulatorConfig(
        program_width=_require_int(
            "program_width", simulator_raw.get("program_width"), minimum=1
        ),
        left_budget=_require_int(
            "left_budget", simulator_raw.get("left_budget"), minimum=1
        ),
        right_budget=_require_int(
            "right_budget", simulator_raw.get("right_budget"), minimum=1
        ),
        t_max=_require_int("t_max", simulator_raw.get("t_max"), minimum=1),
    )
    train = TaskCases(
        _require_string_list("train_inputs", payload.get("train_inputs")),
        _require_string_list("train_targets", payload.get("train_targets")),
    )
    train.validate_right_budget(simulator.right_budget)

    test_inputs = _require_string_list(
        "test_inputs", payload.get("test_inputs", [])
    )
    test_targets = _require_string_list(
        "test_targets", payload.get("test_targets", [])
    )
    if bool(test_inputs) != bool(test_targets):
        raise ValueError(
            "test_inputs and test_targets must both be empty or both contain cases"
        )
    test = TaskCases(test_inputs, test_targets) if test_inputs else None
    if test is not None:
        test.validate_right_budget(simulator.right_budget)

    seed_raw = structural.get("seed")
    if seed_raw is not None:
        seed = _require_int("seed", seed_raw, minimum=0)
    else:
        seed = None
    k_penalty = _require_float("k_penalty", structural.get("k_penalty"))
    if k_penalty < 0.0:
        raise ValueError("k_penalty must be non-negative")
    return RunDefinition(
        run_id=validate_run_id(payload.get("run_id")),
        simulator=simulator,
        train=train,
        test=test,
        islands=_require_int(
            "islands", structural.get("islands"), minimum=1
        ),
        population_per_island=_require_int(
            "population_per_island",
            structural.get("population_per_island"),
            minimum=2,
        ),
        seed=seed,
        k_penalty=k_penalty,
    )


def parse_live_settings(
    value: Mapping[str, Any],
    definition: RunDefinition,
) -> LiveSettings:
    thresholds_raw = value.get("tournament_delta_thresholds")
    probabilities_raw = value.get("tournament_best_probabilities")
    if not isinstance(thresholds_raw, (list, tuple)):
        raise TypeError("tournament_delta_thresholds must be a list")
    if not isinstance(probabilities_raw, (list, tuple)):
        raise TypeError("tournament_best_probabilities must be a list")
    settings = LiveSettings(
        generation_target=_require_int(
            "generation_target", value.get("generation_target"), minimum=0
        ),
        elite_fraction=_require_float(
            "elite_fraction", value.get("elite_fraction")
        ),
        cross_island_fraction=_require_float(
            "cross_island_fraction", value.get("cross_island_fraction")
        ),
        cross_island_interval=_require_int(
            "cross_island_interval",
            value.get("cross_island_interval"),
            minimum=1,
        ),
        random_immigrant_fraction=_require_float(
            "random_immigrant_fraction",
            value.get("random_immigrant_fraction"),
        ),
        tournament_delta_thresholds=tuple(
            _require_float("tournament threshold", item)
            for item in thresholds_raw
        ),
        tournament_best_probabilities=tuple(
            _require_float("tournament probability", item)
            for item in probabilities_raw
        ),
        tournament_base_probability=_require_float(
            "tournament_base_probability",
            value.get("tournament_base_probability"),
        ),
        p_crossover=_require_float("p_crossover", value.get("p_crossover")),
        crossover_segments=_require_int(
            "crossover_segments", value.get("crossover_segments"), minimum=1
        ),
        crossover_segment_min=_require_int(
            "crossover_segment_min",
            value.get("crossover_segment_min"),
            minimum=1,
        ),
        crossover_segment_max=_require_int(
            "crossover_segment_max",
            value.get("crossover_segment_max"),
            minimum=1,
        ),
        p_insert=_require_float("p_insert", value.get("p_insert")),
        p_delete=_require_float("p_delete", value.get("p_delete")),
        p_bit_flip=_require_float("p_bit_flip", value.get("p_bit_flip")),
        stop_first_exact=_require_bool(
            "stop_first_exact", value.get("stop_first_exact")
        ),
        diagnostic_interval=_require_int(
            "diagnostic_interval",
            value.get("diagnostic_interval"),
            minimum=1,
        ),
        checkpoint_interval=_require_int(
            "checkpoint_interval",
            value.get("checkpoint_interval"),
            minimum=1,
        ),
    )
    build_ga_config(definition, settings)
    return settings


def build_ga_config(
    definition: RunDefinition,
    settings: LiveSettings,
) -> IslandGAConfig:
    return IslandGAConfig(
        islands=definition.islands,
        population_per_island=definition.population_per_island,
        generations=settings.generation_target,
        elite_fraction=settings.elite_fraction,
        cross_island_fraction=settings.cross_island_fraction,
        cross_island_interval=settings.cross_island_interval,
        random_immigrant_fraction=settings.random_immigrant_fraction,
        tournament_delta_thresholds=settings.tournament_delta_thresholds,
        tournament_best_probabilities=(
            settings.tournament_best_probabilities
        ),
        tournament_base_probability=(
            settings.tournament_base_probability
        ),
        p_crossover=settings.p_crossover,
        crossover_segments=settings.crossover_segments,
        crossover_segment_min=settings.crossover_segment_min,
        crossover_segment_max=settings.crossover_segment_max,
        p_insert=settings.p_insert,
        p_delete=settings.p_delete,
        p_bit_flip=settings.p_bit_flip,
        k_penalty=definition.k_penalty,
        seed=definition.seed,
        print_every=None,
        stop_first_exact=settings.stop_first_exact,
    )


def parse_new_run_payload(
    payload: Mapping[str, Any],
) -> tuple[RunDefinition, LiveSettings]:
    definition = parse_run_definition(payload)
    mutable = _require_mapping("mutable", payload.get("mutable"))
    return definition, parse_live_settings(mutable, definition)


def _evaluate_outputs(
    result: TaichiFitnessBatch,
    target: str,
    matching_bits: np.ndarray,
    compared_bits: np.ndarray,
    invalid_cases: np.ndarray,
    exact: np.ndarray,
) -> None:
    target_bits = np.fromiter(
        (int(bit) for bit in target),
        dtype=np.uint8,
        count=len(target),
    )
    output_lengths = result.output_lengths.astype(np.int64, copy=False)
    overlap = np.minimum(output_lengths, len(target))
    positions = np.arange(len(target), dtype=np.int64)[None, :]
    aligned = positions < overlap[:, None]
    matches = aligned & (
        result.output_bits[:, : len(target)] == target_bits[None, :]
    )
    matching_bits += matches.sum(axis=1, dtype=np.int64)
    compared_bits += np.maximum(output_lengths, len(target))
    invalid_cases += result.invalid.astype(np.uint32)
    exact &= (
        ~result.invalid
        & (output_lengths == len(target))
        & np.all(
            result.output_bits[:, : len(target)]
            == target_bits[None, :],
            axis=1,
        )
    )


def evaluate_population_with_diagnostics(
    simulator: ControlPadTaichiSimulator,
    programs: np.ndarray,
    task: TaskCases,
    *,
    k_penalty: float,
) -> PopulationDiagnostics:
    """Evaluate fitness and halted resource use in the same CUDA runs."""

    raw = np.asarray(programs)
    if raw.ndim != 2 or raw.shape[0] == 0:
        raise ValueError("programs must be a nonempty two-dimensional array")
    if raw.shape[1] != simulator.config.program_width:
        raise ValueError("program width disagrees with the simulator")
    if raw.shape[0] > simulator.batch_capacity:
        raise ValueError("program population exceeds simulator batch_capacity")
    task.validate_right_budget(simulator.config.right_budget)
    if not np.isfinite(float(k_penalty)) or float(k_penalty) < 0.0:
        raise ValueError("k_penalty must be finite and non-negative")

    population = raw.shape[0]
    matching_bits = np.zeros(population, dtype=np.int64)
    compared_bits = np.zeros(population, dtype=np.int64)
    invalid_cases = np.zeros(population, dtype=np.uint32)
    exact = np.ones(population, dtype=np.bool_)
    halted_counts = np.zeros(population, dtype=np.uint32)
    halted_T_sum = np.zeros(population, dtype=np.uint64)
    halted_space_sum = np.zeros(population, dtype=np.uint64)

    for input_bits, target in task.pairs():
        result = simulator.simulate_program_batch_for_control_pad(
            raw, input_bits
        )
        _evaluate_outputs(
            result,
            target,
            matching_bits,
            compared_bits,
            invalid_cases,
            exact,
        )
        halted = result.halted
        halted_counts += halted.astype(np.uint32)
        halted_T_sum += np.where(
            halted, result.T.astype(np.uint64), 0
        )
        spaces = (
            result.left_space_used.astype(np.int64)
            + result.right_space_used.astype(np.int64)
        )
        if np.any(spaces < 0):
            raise RuntimeError("simulator returned negative space use")
        halted_space_sum += np.where(
            halted, spaces.astype(np.uint64), 0
        )

    bit_accuracy = matching_bits / compared_bits
    invalid_fraction = invalid_cases / len(task)
    fitness_values = (
        bit_accuracy - float(k_penalty) * invalid_fraction
    )
    mean_T = np.full(population, np.nan, dtype=np.float64)
    mean_space = np.full(population, np.nan, dtype=np.float64)
    valid = halted_counts > 0
    mean_T[valid] = halted_T_sum[valid] / halted_counts[valid]
    mean_space[valid] = halted_space_sum[valid] / halted_counts[valid]
    return PopulationDiagnostics(
        fitness=PopulationFitness(
            fitness=fitness_values,
            bit_accuracy=bit_accuracy,
            invalid_fraction=invalid_fraction,
            exact=exact,
        ),
        halted_counts=halted_counts,
        mean_halted_T=mean_T,
        mean_halted_space=mean_space,
    )


def _zero_reproduction(config: IslandGAConfig) -> ReproductionStats:
    return ReproductionStats(
        elites_per_island=config.elite_count,
        promoted_child_elites=0,
        cross_island_per_island=0,
        random_immigrants_per_island=0,
        children_per_island=0,
        crossed_children=0,
        sampled_segments=0,
        copied_loci=0,
        insertions=0,
        deletions=0,
        flipped_bits=0,
    )


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"mean": None, "q20": None, "q80": None, "count": 0}
    q20, q80 = np.quantile(finite, (0.20, 0.80))
    return {
        "mean": float(np.mean(finite)),
        "q20": float(q20),
        "q80": float(q80),
        "count": int(finite.size),
    }


def _nullable_float_list(values: np.ndarray) -> list[float | None]:
    return [
        float(value) if np.isfinite(value) else None
        for value in np.asarray(values, dtype=np.float64)
    ]


def _champion_to_json(champion: IslandChampion) -> dict[str, Any]:
    return {
        "generation": champion.generation,
        "island": champion.island,
        "island_index": champion.island_index,
        "program": champion.program,
        "padded_program": champion.padded_program,
        "effective_length": champion.effective_length,
        "fitness": champion.fitness,
        "bit_accuracy": champion.bit_accuracy,
        "invalid_fraction": champion.invalid_fraction,
        "exact": champion.exact,
    }


class LiveTrainingSession:
    """Stateful one-generation driver around the unchanged island GA."""

    def __init__(
        self,
        definition: RunDefinition,
        settings: LiveSettings,
        simulator: ControlPadTaichiSimulator,
        *,
        initialize: bool,
    ) -> None:
        self.definition = definition
        self.settings = settings
        self.simulator = simulator
        self.engine = TaichiIslandGA(
            simulator,
            definition.train,
            build_ga_config(definition, settings),
        )
        self.generation = 0
        self.elapsed_before_start = 0.0
        self.started_at = perf_counter()
        self.last_saved_generation: int | None = None
        self.last_saved_at: str | None = None
        self.config_events: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []
        self.diagnostic_records: list[dict[str, Any]] = []
        self.island_fitness: list[np.ndarray] = []
        self.island_accuracy: list[np.ndarray] = []
        self.island_invalid: list[np.ndarray] = []
        self.diag_island_T: list[np.ndarray] = []
        self.diag_island_space: list[np.ndarray] = []
        self.diag_test_island_fitness: list[np.ndarray] = []
        self.diag_test_island_accuracy: list[np.ndarray] = []
        self.diag_test_island_invalid: list[np.ndarray] = []
        self.global_diagnostics: dict[str, Any] = {
            "train": None,
            "test": None,
            "generation": None,
        }
        if initialize:
            self._initialize_new()

    @classmethod
    def create(
        cls,
        definition: RunDefinition,
        settings: LiveSettings,
    ) -> "LiveTrainingSession":
        simulator = ControlPadTaichiSimulator(
            definition.simulator,
            batch_capacity=definition.total_population,
        )
        return cls(definition, settings, simulator, initialize=True)

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_before_start + (perf_counter() - self.started_at)

    def _flat_programs(self) -> np.ndarray:
        if self.engine.programs is None:
            raise RuntimeError("population is not initialized")
        return self.engine.programs.reshape(
            self.definition.total_population,
            self.definition.simulator.program_width,
        )

    def _evaluate(
        self,
        *,
        diagnostics: bool,
    ) -> tuple[PopulationFitness, PopulationDiagnostics | None]:
        flat = self._flat_programs()
        if diagnostics:
            result = evaluate_population_with_diagnostics(
                self.simulator,
                flat,
                self.definition.train,
                k_penalty=self.definition.k_penalty,
            )
            evaluated = result.fitness
            diagnostic = result
        else:
            evaluated = evaluate_population_fitness_taichi(
                self.simulator,
                flat,
                self.definition.train,
                k_penalty=self.definition.k_penalty,
            )
            diagnostic = None
        shape = (
            self.definition.islands,
            self.definition.population_per_island,
        )
        fitness = PopulationFitness(
            fitness=evaluated.fitness.reshape(shape),
            bit_accuracy=evaluated.bit_accuracy.reshape(shape),
            invalid_fraction=evaluated.invalid_fraction.reshape(shape),
            exact=evaluated.exact.reshape(shape),
        )
        if diagnostic is None:
            return fitness, None
        return fitness, PopulationDiagnostics(
            fitness=fitness,
            halted_counts=diagnostic.halted_counts.reshape(shape),
            mean_halted_T=diagnostic.mean_halted_T.reshape(shape),
            mean_halted_space=diagnostic.mean_halted_space.reshape(shape),
        )

    def _initialize_new(self) -> None:
        self.engine._initialize(None)
        fitness, diagnostic = self._evaluate(diagnostics=True)
        self.engine.fitness = fitness
        self._record_generation(
            fitness,
            _zero_reproduction(self.engine.config),
            diagnostic,
        )

    def apply_settings(self, settings: LiveSettings) -> None:
        build_ga_config(self.definition, settings)
        self.settings = settings
        self.engine.config = build_ga_config(self.definition, settings)
        self.config_events.append(
            {
                "effective_generation": self.generation + 1,
                "settings": settings.to_json(),
            }
        )

    def _current_champion_indices(
        self,
        fitness: PopulationFitness,
    ) -> np.ndarray:
        return np.argmax(fitness.fitness, axis=1)

    def _evaluate_global_best_diagnostics(self) -> None:
        champion = self.engine.global_best
        if champion is None:
            return
        padded = champion.symbols.reshape(1, -1)
        train = evaluate_population_with_diagnostics(
            self.simulator,
            padded,
            self.definition.train,
            k_penalty=self.definition.k_penalty,
        )
        train_data = {
            "fitness": float(train.fitness.fitness[0]),
            "accuracy": float(train.fitness.bit_accuracy[0]),
            "invalid": float(train.fitness.invalid_fraction[0]),
            "halted_cases": int(train.halted_counts[0]),
            "mean_T": (
                None
                if np.isnan(train.mean_halted_T[0])
                else float(train.mean_halted_T[0])
            ),
            "mean_space": (
                None
                if np.isnan(train.mean_halted_space[0])
                else float(train.mean_halted_space[0])
            ),
        }
        test_data = None
        if self.definition.test is not None:
            test = evaluate_population_with_diagnostics(
                self.simulator,
                padded,
                self.definition.test,
                k_penalty=self.definition.k_penalty,
            )
            test_data = {
                "fitness": float(test.fitness.fitness[0]),
                "accuracy": float(test.fitness.bit_accuracy[0]),
                "invalid": float(test.fitness.invalid_fraction[0]),
                "halted_cases": int(test.halted_counts[0]),
                "mean_T": (
                    None
                    if np.isnan(test.mean_halted_T[0])
                    else float(test.mean_halted_T[0])
                ),
                "mean_space": (
                    None
                    if np.isnan(test.mean_halted_space[0])
                    else float(test.mean_halted_space[0])
                ),
            }
        self.global_diagnostics = {
            "train": train_data,
            "test": test_data,
            "generation": self.generation,
        }

    def _record_diagnostics(
        self,
        fitness: PopulationFitness,
        diagnostics: PopulationDiagnostics,
        champion_indices: np.ndarray,
    ) -> None:
        rows = np.arange(self.definition.islands)
        champion_T = diagnostics.mean_halted_T[rows, champion_indices]
        champion_space = diagnostics.mean_halted_space[
            rows, champion_indices
        ]
        record: dict[str, Any] = {
            "generation": self.generation,
            "train_T": _summary(champion_T),
            "train_space": _summary(champion_space),
            "test_fitness": None,
            "test_accuracy": None,
            "test_invalid": None,
            "test_T": None,
            "test_space": None,
        }
        self.diag_island_T.append(champion_T.astype(np.float64, copy=True))
        self.diag_island_space.append(
            champion_space.astype(np.float64, copy=True)
        )

        if self.engine.programs is None:
            raise RuntimeError("population is not initialized")
        if self.definition.test is not None:
            champion_programs = self.engine.programs[
                rows, champion_indices
            ]
            test = evaluate_population_with_diagnostics(
                self.simulator,
                champion_programs,
                self.definition.test,
                k_penalty=self.definition.k_penalty,
            )
            record["test_fitness"] = _summary(test.fitness.fitness)
            record["test_accuracy"] = _summary(
                test.fitness.bit_accuracy
            )
            record["test_invalid"] = _summary(
                test.fitness.invalid_fraction
            )
            record["test_T"] = _summary(test.mean_halted_T)
            record["test_space"] = _summary(test.mean_halted_space)
            self.diag_test_island_fitness.append(
                test.fitness.fitness.astype(np.float64, copy=True)
            )
            self.diag_test_island_accuracy.append(
                test.fitness.bit_accuracy.astype(np.float64, copy=True)
            )
            self.diag_test_island_invalid.append(
                test.fitness.invalid_fraction.astype(np.float64, copy=True)
            )
        else:
            empty = np.empty((0,), dtype=np.float64)
            self.diag_test_island_fitness.append(empty)
            self.diag_test_island_accuracy.append(empty)
            self.diag_test_island_invalid.append(empty)

        self._evaluate_global_best_diagnostics()
        record["global_best"] = self.global_diagnostics
        self.diagnostic_records.append(record)

    def _record_generation(
        self,
        fitness: PopulationFitness,
        reproduction: ReproductionStats,
        diagnostics: PopulationDiagnostics | None,
    ) -> None:
        current = self.engine._champion(self.generation, fitness)
        if (
            self.engine.global_best is None
            or current.fitness > self.engine.global_best.fitness
        ):
            self.engine.global_best = current
        if self.engine.global_best is None:
            raise RuntimeError("global best was not initialized")

        indices = self._current_champion_indices(fitness)
        rows = np.arange(self.definition.islands)
        island_fitness = fitness.fitness[rows, indices]
        island_accuracy = fitness.bit_accuracy[rows, indices]
        island_invalid = fitness.invalid_fraction[rows, indices]
        self.island_fitness.append(
            island_fitness.astype(np.float64, copy=True)
        )
        self.island_accuracy.append(
            island_accuracy.astype(np.float64, copy=True)
        )
        self.island_invalid.append(
            island_invalid.astype(np.float64, copy=True)
        )

        record = {
            "generation": self.generation,
            "current_best": _champion_to_json(current),
            "global_best": _champion_to_json(self.engine.global_best),
            "island_fitness": _summary(island_fitness),
            "island_accuracy": _summary(island_accuracy),
            "island_invalid": _summary(island_invalid),
            "population_invalid": _summary(
                fitness.invalid_fraction.reshape(-1)
            ),
            "mean_population_fitness": float(np.mean(fitness.fitness)),
            "exact_programs": int(np.count_nonzero(fitness.exact)),
            "reproduction": asdict(reproduction),
            "elapsed_seconds": self.elapsed_seconds,
        }
        self.records.append(record)
        if diagnostics is not None:
            self._record_diagnostics(fitness, diagnostics, indices)

    def step(self) -> dict[str, Any]:
        if self.engine.fitness is None:
            raise RuntimeError("current fitness is unavailable")
        self.generation += 1
        reproduction = self.engine._make_next_generation(
            self.generation,
            self.engine.fitness,
        )
        diagnostics_due = (
            self.generation % self.settings.diagnostic_interval == 0
        )
        fitness, diagnostics = self._evaluate(
            diagnostics=diagnostics_due
        )
        self.engine.fitness = fitness
        self._record_generation(fitness, reproduction, diagnostics)
        return self.records[-1]

    def should_stop_exact(self) -> bool:
        return bool(
            self.settings.stop_first_exact
            and self.engine.fitness is not None
            and np.any(self.engine.fitness.exact)
        )

    def reached_generation_target(self) -> bool:
        return self.generation >= self.settings.generation_target

    def public_state(self) -> dict[str, Any]:
        if not self.records or self.engine.global_best is None:
            raise RuntimeError("session has no recorded generation")
        return {
            "run_id": self.definition.run_id,
            "generation": self.generation,
            "generation_target": self.settings.generation_target,
            "elapsed_seconds": self.elapsed_seconds,
            "last_record": self.records[-1],
            "global_best": _champion_to_json(self.engine.global_best),
            "global_diagnostics": self.global_diagnostics,
            "active_settings": self.settings.to_json(),
            "structural": self.definition.to_json(),
            "last_saved_generation": self.last_saved_generation,
            "last_saved_at": self.last_saved_at,
        }

    def history_payload(
        self,
        start: int,
        selected_islands: Sequence[int],
    ) -> dict[str, Any]:
        if start < 0:
            raise ValueError("history start must be non-negative")
        for island in selected_islands:
            if not 0 <= island < self.definition.islands:
                raise ValueError(f"island index out of range: {island + 1}")
        selected = list(dict.fromkeys(map(int, selected_islands)))
        # A request may arrive while the main thread is finishing a record.
        # Slice every parallel history at the last fully published record.
        record_count = len(self.records)
        diagnostic_count = len(self.diagnostic_records)
        island_payload: dict[str, Any] = {}
        if selected and self.island_fitness:
            fitness = np.asarray(
                self.island_fitness[:record_count], dtype=np.float64
            )
            accuracy = np.asarray(
                self.island_accuracy[:record_count], dtype=np.float64
            )
            invalid = np.asarray(
                self.island_invalid[:record_count], dtype=np.float64
            )
            diag_generations = [
                int(record["generation"])
                for record in self.diagnostic_records[:diagnostic_count]
            ]
            diag_T = (
                np.asarray(
                    self.diag_island_T[:diagnostic_count],
                    dtype=np.float64,
                )
                if diagnostic_count
                else np.empty((0, self.definition.islands))
            )
            diag_space = (
                np.asarray(
                    self.diag_island_space[:diagnostic_count],
                    dtype=np.float64,
                )
                if diagnostic_count
                else np.empty((0, self.definition.islands))
            )
            for island in selected:
                island_payload[str(island)] = {
                    "fitness": fitness[start:, island].tolist(),
                    "accuracy": accuracy[start:, island].tolist(),
                    "invalid": invalid[start:, island].tolist(),
                    "diagnostic_generations": diag_generations,
                    "mean_T": _nullable_float_list(diag_T[:, island]),
                    "mean_space": _nullable_float_list(
                        diag_space[:, island]
                    ),
                }
        return {
            "start": start,
            "records": self.records[start:record_count],
            "diagnostics": self.diagnostic_records[:diagnostic_count],
            "islands": island_payload,
            "total_records": record_count,
        }

    def _checkpoint_metadata(self, reason: str) -> dict[str, Any]:
        if self.engine.global_best is None:
            raise RuntimeError("global best is unavailable")
        return {
            "schema": CHECKPOINT_SCHEMA,
            "transition_table_sha256": CANONICAL_CSV_SHA256,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "save_reason": reason,
            "generation": self.generation,
            "elapsed_seconds": self.elapsed_seconds,
            "definition": self.definition.to_json(),
            "active_settings": self.settings.to_json(),
            "config_events": self.config_events,
            "rng_state": self.engine.rng.bit_generator.state,
            "global_best": _champion_to_json(self.engine.global_best),
            "global_diagnostics": self.global_diagnostics,
            "records": self.records,
            "diagnostic_records": self.diagnostic_records,
        }

    @staticmethod
    def checkpoint_path(runs_root: Path, run_id: str) -> Path:
        validate_run_id(run_id)
        return runs_root.resolve() / run_id / "latest.npz"

    def save_checkpoint(
        self,
        runs_root: Path,
        *,
        reason: str,
    ) -> Path:
        if (
            self.engine.programs is None
            or self.engine.lengths is None
            or self.engine.is_child is None
            or self.engine.fitness is None
            or self.engine.global_best is None
        ):
            raise RuntimeError("cannot checkpoint an incomplete session")
        target = self.checkpoint_path(
            runs_root, self.definition.run_id
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / (
            f".latest-{os.getpid()}-{self.generation}.tmp"
        )
        arrays: dict[str, np.ndarray] = {
            "metadata_json": np.asarray(
                json.dumps(
                    self._checkpoint_metadata(reason),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            ),
            "programs": self.engine.programs,
            "lengths": self.engine.lengths,
            "is_child": self.engine.is_child,
            "fitness": self.engine.fitness.fitness,
            "bit_accuracy": self.engine.fitness.bit_accuracy,
            "invalid_fraction": self.engine.fitness.invalid_fraction,
            "exact": self.engine.fitness.exact,
            "global_best_symbols": self.engine.global_best.symbols,
            "island_fitness": np.asarray(
                self.island_fitness, dtype=np.float64
            ),
            "island_accuracy": np.asarray(
                self.island_accuracy, dtype=np.float64
            ),
            "island_invalid": np.asarray(
                self.island_invalid, dtype=np.float64
            ),
            "diag_island_T": np.asarray(
                self.diag_island_T, dtype=np.float64
            ),
            "diag_island_space": np.asarray(
                self.diag_island_space, dtype=np.float64
            ),
            "diag_test_island_fitness": np.asarray(
                self.diag_test_island_fitness, dtype=np.float64
            ),
            "diag_test_island_accuracy": np.asarray(
                self.diag_test_island_accuracy, dtype=np.float64
            ),
            "diag_test_island_invalid": np.asarray(
                self.diag_test_island_invalid, dtype=np.float64
            ),
        }
        try:
            with temporary.open("wb") as handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.last_saved_generation = self.generation
        self.last_saved_at = datetime.now(timezone.utc).isoformat()
        return target

    @classmethod
    def load_checkpoint(
        cls,
        runs_root: Path,
        run_id: str,
    ) -> "LiveTrainingSession":
        path = cls.checkpoint_path(runs_root, run_id)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata_json"].item()))
            arrays = {
                name: stored[name].copy()
                for name in stored.files
                if name != "metadata_json"
            }
        if metadata.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported control-pad checkpoint schema")
        if (
            metadata.get("transition_table_sha256")
            != CANONICAL_CSV_SHA256
        ):
            raise ValueError(
                "checkpoint transition table differs from this repository"
            )
        definition_payload = dict(metadata["definition"])
        definition = parse_run_definition(definition_payload)
        if definition.run_id != run_id:
            raise ValueError("checkpoint run_id does not match its directory")
        settings = parse_live_settings(
            metadata["active_settings"], definition
        )
        simulator = ControlPadTaichiSimulator(
            definition.simulator,
            batch_capacity=definition.total_population,
        )
        session = cls(
            definition,
            settings,
            simulator,
            initialize=False,
        )
        shape = (
            definition.islands,
            definition.population_per_island,
            definition.simulator.program_width,
        )
        programs = arrays["programs"]
        lengths = arrays["lengths"]
        is_child = arrays["is_child"]
        if programs.shape != shape:
            raise ValueError("checkpoint population shape is inconsistent")
        if lengths.shape != shape[:2] or is_child.shape != shape[:2]:
            raise ValueError("checkpoint population metadata shape is invalid")
        table = simulator.table
        calculated_lengths = effective_program_lengths(
            programs.reshape(definition.total_population, shape[2]),
            blank_id=table.blank_id,
            zero_id=table.zero_id,
            one_id=table.one_id,
        ).reshape(shape[:2])
        if not np.array_equal(lengths, calculated_lengths):
            raise ValueError("checkpoint program lengths are inconsistent")

        session.engine.programs = programs.astype(np.uint8, copy=True)
        session.engine.lengths = lengths.astype(np.int32, copy=True)
        session.engine.is_child = is_child.astype(np.bool_, copy=True)
        fitness_shape = shape[:2]
        session.engine.fitness = PopulationFitness(
            fitness=np.asarray(arrays["fitness"], dtype=np.float64),
            bit_accuracy=np.asarray(
                arrays["bit_accuracy"], dtype=np.float64
            ),
            invalid_fraction=np.asarray(
                arrays["invalid_fraction"], dtype=np.float64
            ),
            exact=np.asarray(arrays["exact"], dtype=np.bool_),
        )
        for array in (
            session.engine.fitness.fitness,
            session.engine.fitness.bit_accuracy,
            session.engine.fitness.invalid_fraction,
            session.engine.fitness.exact,
        ):
            if array.shape != fitness_shape:
                raise ValueError("checkpoint fitness shape is inconsistent")

        best_data = metadata["global_best"]
        symbols = np.asarray(
            arrays["global_best_symbols"], dtype=np.uint8
        )
        if symbols.shape != (shape[2],):
            raise ValueError("checkpoint global-best program shape is invalid")
        session.engine.global_best = IslandChampion(
            generation=int(best_data["generation"]),
            island=int(best_data["island"]),
            island_index=int(best_data["island_index"]),
            program=str(best_data["program"]),
            padded_program=str(best_data["padded_program"]),
            symbols=symbols,
            effective_length=int(best_data["effective_length"]),
            fitness=float(best_data["fitness"]),
            bit_accuracy=float(best_data["bit_accuracy"]),
            invalid_fraction=float(best_data["invalid_fraction"]),
            exact=bool(best_data["exact"]),
        )
        session.engine.rng.bit_generator.state = metadata["rng_state"]
        session.generation = int(metadata["generation"])
        session.elapsed_before_start = float(metadata["elapsed_seconds"])
        session.started_at = perf_counter()
        session.config_events = list(metadata.get("config_events", []))
        session.records = list(metadata["records"])
        session.diagnostic_records = list(
            metadata.get("diagnostic_records", [])
        )
        session.global_diagnostics = dict(
            metadata.get(
                "global_diagnostics",
                {"train": None, "test": None, "generation": None},
            )
        )

        def rows(name: str) -> list[np.ndarray]:
            array = np.asarray(arrays[name], dtype=np.float64)
            return [row.copy() for row in array]

        session.island_fitness = rows("island_fitness")
        session.island_accuracy = rows("island_accuracy")
        session.island_invalid = rows("island_invalid")
        session.diag_island_T = rows("diag_island_T")
        session.diag_island_space = rows("diag_island_space")
        session.diag_test_island_fitness = rows(
            "diag_test_island_fitness"
        )
        session.diag_test_island_accuracy = rows(
            "diag_test_island_accuracy"
        )
        session.diag_test_island_invalid = rows(
            "diag_test_island_invalid"
        )
        if len(session.records) != session.generation + 1:
            raise ValueError("checkpoint generation history is incomplete")
        if len(session.island_fitness) != len(session.records):
            raise ValueError("checkpoint island history is incomplete")
        if len(session.diagnostic_records) != len(session.diag_island_T):
            raise ValueError("checkpoint diagnostic history is incomplete")
        session.last_saved_generation = session.generation
        session.last_saved_at = str(metadata["saved_at"])
        return session


def list_saved_runs(runs_root: Path) -> list[dict[str, Any]]:
    root = runs_root.resolve()
    if not root.exists():
        return []
    results: list[dict[str, Any]] = []
    for directory in root.iterdir():
        if not directory.is_dir() or not RUN_ID_PATTERN.fullmatch(
            directory.name
        ):
            continue
        checkpoint = directory / "latest.npz"
        if not checkpoint.is_file():
            continue
        try:
            with np.load(checkpoint, allow_pickle=False) as stored:
                metadata = json.loads(
                    str(stored["metadata_json"].item())
                )
            results.append(
                {
                    "run_id": directory.name,
                    "generation": int(metadata["generation"]),
                    "saved_at": str(metadata["saved_at"]),
                    "save_reason": str(metadata["save_reason"]),
                }
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            results.append(
                {
                    "run_id": directory.name,
                    "generation": None,
                    "saved_at": None,
                    "save_reason": "unreadable checkpoint",
                }
            )
    return sorted(
        results,
        key=lambda item: item["saved_at"] or "",
        reverse=True,
    )
