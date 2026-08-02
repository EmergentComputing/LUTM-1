"""NumPy island GA evaluated by the Taichi/CUDA LUTM simulator.

Evolution stays on the CPU in vectorized NumPy.  Only physical LUTM
simulation is delegated to Taichi.  Programs are blank-left-padded binary
suffixes throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter
from typing import Sequence

import numpy as np

from taichi_backend import TaichiUTMSimulator
from utils import (
    TaskCases,
    effective_program_lengths,
    effective_program_to_string,
    padded_program_to_string,
)


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(
        value, (bool, np.bool_)
    )


def _validate_probability(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return converted


def _real_tuple(name: str, values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of real numbers")
    converted: list[float] = []
    for value in values:
        if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must contain only real numbers")
        converted.append(float(value))
    return tuple(converted)


def _validate_delta_schedule(
    thresholds: Sequence[float],
    probabilities: Sequence[float],
    base_probability: float,
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    thresholds_tuple = _real_tuple("thresholds", thresholds)
    probabilities_tuple = _real_tuple("probabilities", probabilities)
    if not thresholds_tuple:
        raise ValueError("thresholds and probabilities must not be empty")
    if len(thresholds_tuple) != len(probabilities_tuple):
        raise ValueError(
            "thresholds and probabilities must have the same length"
        )
    if not np.all(np.isfinite(thresholds_tuple)):
        raise ValueError("thresholds must be finite")
    if any(value < 0.0 for value in thresholds_tuple):
        raise ValueError("thresholds must be non-negative")
    if any(
        right <= left
        for left, right in zip(thresholds_tuple, thresholds_tuple[1:])
    ):
        raise ValueError("thresholds must be strictly increasing")
    if not np.all(np.isfinite(probabilities_tuple)):
        raise ValueError("probabilities must be finite")
    if any(not 0.5 <= value <= 1.0 for value in probabilities_tuple):
        raise ValueError("tournament probabilities must be in [0.5, 1]")
    if any(
        right < left
        for left, right in zip(
            probabilities_tuple, probabilities_tuple[1:]
        )
    ):
        raise ValueError("tournament probabilities must be nondecreasing")
    base = _validate_probability("base_probability", base_probability)
    if base < 0.5:
        raise ValueError("base_probability must be at least 0.5")
    if base < probabilities_tuple[-1]:
        raise ValueError(
            "base_probability must not be below the last interval probability"
        )
    return thresholds_tuple, probabilities_tuple, base


@dataclass(frozen=True)
class IslandGAConfig:
    """Configuration for the delta-tournament island search."""

    islands: int = 20
    population_per_island: int = 2_000
    generations: int = 300
    elite_fraction: float = 0.05
    cross_island_fraction: float = 0.01
    cross_island_interval: int = 30
    random_immigrant_fraction: float = 0.05
    tournament_delta_thresholds: tuple[float, ...] = (0.05, 0.20)
    tournament_best_probabilities: tuple[float, ...] = (0.52, 0.60)
    tournament_base_probability: float = 0.90
    p_crossover: float = 0.50
    crossover_segments: int = 2
    crossover_segment_min: int = 1
    crossover_segment_max: int = 4
    p_insert: float = 0.04
    p_delete: float = 0.06
    p_bit_flip: float = 0.02
    k_penalty: float = 1.0
    seed: int | None = 0
    print_every: int | None = 1
    stop_first_exact: bool = True

    def __post_init__(self) -> None:
        for name in (
            "islands",
            "population_per_island",
            "generations",
            "cross_island_interval",
            "crossover_segments",
            "crossover_segment_min",
            "crossover_segment_max",
        ):
            if not _is_integer(getattr(self, name)):
                raise TypeError(f"{name} must be an integer")
        if self.islands < 1:
            raise ValueError("islands must be positive")
        if self.population_per_island < 2:
            raise ValueError("population_per_island must be at least 2")
        if self.generations < 0:
            raise ValueError("generations must be non-negative")
        if self.cross_island_interval < 1:
            raise ValueError("cross_island_interval must be positive")
        if self.crossover_segments < 1:
            raise ValueError("crossover_segments must be positive")
        if self.crossover_segment_min < 1:
            raise ValueError("crossover_segment_min must be positive")
        if self.crossover_segment_max < self.crossover_segment_min:
            raise ValueError(
                "crossover_segment_max must be at least crossover_segment_min"
            )

        for name in (
            "elite_fraction",
            "cross_island_fraction",
            "random_immigrant_fraction",
            "p_crossover",
            "p_insert",
            "p_delete",
            "p_bit_flip",
        ):
            _validate_probability(name, getattr(self, name))
        if self.p_insert + self.p_delete > 1.0:
            raise ValueError("p_insert + p_delete must not exceed 1")

        thresholds, probabilities, base = _validate_delta_schedule(
            self.tournament_delta_thresholds,
            self.tournament_best_probabilities,
            self.tournament_base_probability,
        )
        object.__setattr__(self, "tournament_delta_thresholds", thresholds)
        object.__setattr__(
            self, "tournament_best_probabilities", probabilities
        )
        object.__setattr__(self, "tournament_base_probability", base)

        if not isinstance(self.k_penalty, Real) or isinstance(
            self.k_penalty, (bool, np.bool_)
        ):
            raise TypeError("k_penalty must be a real number")
        if not np.isfinite(float(self.k_penalty)):
            raise ValueError("k_penalty must be finite")
        if self.k_penalty < 0:
            raise ValueError("k_penalty must be non-negative")
        if self.seed is not None and not _is_integer(self.seed):
            raise TypeError("seed must be an integer or None")
        if self.print_every is not None:
            if not _is_integer(self.print_every):
                raise TypeError("print_every must be an integer or None")
            if self.print_every < 1:
                raise ValueError("print_every must be positive")
        if not isinstance(self.stop_first_exact, (bool, np.bool_)):
            raise TypeError("stop_first_exact must be boolean")

        if self.cross_island_fraction > 0.0 and self.islands < 2:
            raise ValueError(
                "cross-island migration requires at least two islands"
            )
        if (
            self.elite_count
            + self.cross_island_count
            + self.random_immigrant_count
            > self.population_per_island
        ):
            raise ValueError(
                "elite, cross-island, and random-immigrant slots "
                "exceed the island population"
            )

    @property
    def total_population(self) -> int:
        return self.islands * self.population_per_island

    @property
    def elite_count(self) -> int:
        return max(1, int(self.elite_fraction * self.population_per_island))

    @property
    def cross_island_count(self) -> int:
        return int(self.cross_island_fraction * self.population_per_island)

    @property
    def random_immigrant_count(self) -> int:
        return int(
            self.random_immigrant_fraction * self.population_per_island
        )


@dataclass(frozen=True, eq=False)
class PopulationFitness:
    fitness: np.ndarray
    bit_accuracy: np.ndarray
    invalid_fraction: np.ndarray
    exact: np.ndarray


@dataclass(frozen=True, eq=False)
class MutationResult:
    programs: np.ndarray
    effective_lengths: np.ndarray
    insertions: int
    deletions: int
    flipped_bits: int


@dataclass(frozen=True, eq=False)
class SegmentCrossoverResult:
    programs: np.ndarray
    effective_lengths: np.ndarray
    crossed_children: int
    sampled_segments: int
    copied_loci: int


@dataclass(frozen=True)
class IslandChampion:
    generation: int
    island: int
    island_index: int
    program: str
    padded_program: str
    symbols: np.ndarray
    effective_length: int
    fitness: float
    bit_accuracy: float
    invalid_fraction: float
    exact: bool


@dataclass(frozen=True)
class ReproductionStats:
    elites_per_island: int
    promoted_child_elites: int
    cross_island_per_island: int
    random_immigrants_per_island: int
    children_per_island: int
    crossed_children: int
    sampled_segments: int
    copied_loci: int
    insertions: int
    deletions: int
    flipped_bits: int


@dataclass(frozen=True, eq=False)
class IslandGAHistory:
    generations: np.ndarray
    current_best_fitness: np.ndarray
    global_best_fitness: np.ndarray
    mean_fitness: np.ndarray
    std_fitness: np.ndarray
    mean_island_best: np.ndarray
    std_island_best: np.ndarray
    best_bit_accuracy: np.ndarray
    best_invalid_fraction: np.ndarray
    exact_programs: np.ndarray
    island_best_fitness: np.ndarray
    promoted_child_elites: np.ndarray
    cross_island_per_island: np.ndarray
    children_per_island: np.ndarray
    crossed_children: np.ndarray
    sampled_segments: np.ndarray
    copied_loci: np.ndarray
    insertions: np.ndarray
    deletions: np.ndarray
    flipped_bits: np.ndarray


@dataclass(frozen=True, eq=False)
class IslandGAResult:
    best: IslandChampion
    solved: bool
    generations_completed: int
    evaluated_programs: int
    elapsed_seconds: float
    history: IslandGAHistory
    final_programs: np.ndarray
    final_effective_lengths: np.ndarray


def initialize_population(
    simulator: TaichiUTMSimulator,
    population_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample valid padded programs with lengths uniform in ``0..width``."""

    if not _is_integer(population_size):
        raise TypeError("population_size must be an integer")
    if population_size < 1:
        raise ValueError("population_size must be positive")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    size = int(population_size)
    width = simulator.config.program_width
    table = simulator.table
    lengths = rng.integers(0, width + 1, size=size, dtype=np.int32)
    columns = np.arange(width, dtype=np.int32)[None, :]
    active = columns >= (width - lengths)[:, None]
    logical_bits = rng.integers(0, 2, size=(size, width), dtype=np.uint8)
    bit_symbols = np.where(
        logical_bits == 0, table.zero_id, table.one_id
    ).astype(np.uint8)
    programs = np.full((size, width), table.blank_id, dtype=np.uint8)
    programs[active] = bit_symbols[active]
    return programs, lengths


def mutate_population(
    simulator: TaichiUTMSimulator,
    programs: np.ndarray,
    effective_lengths: np.ndarray,
    rng: np.random.Generator,
    *,
    p_insert: float,
    p_delete: float,
    p_bit_flip: float,
) -> MutationResult:
    """Apply one exclusive structural mutation, then per-locus bit flips.

    Insertion prepends a random bit; deletion removes the leftmost bit::

        BB0101 -> B10101  (insert 1)
        BB0101 -> BBB101  (delete)
    """

    raw = np.asarray(programs)
    lengths = np.asarray(effective_lengths)
    if raw.ndim != 2 or raw.shape[0] == 0:
        raise ValueError("programs must be a nonempty two-dimensional array")
    if raw.shape[1] != simulator.config.program_width:
        raise ValueError("program width disagrees with the simulator")
    if lengths.shape != (raw.shape[0],):
        raise ValueError("effective_lengths has the wrong shape")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    for name, value in (
        ("p_insert", p_insert),
        ("p_delete", p_delete),
        ("p_bit_flip", p_bit_flip),
    ):
        _validate_probability(name, value)
    if p_insert + p_delete > 1.0:
        raise ValueError("p_insert + p_delete must not exceed 1")

    table = simulator.table
    validated_lengths = effective_program_lengths(
        raw,
        blank_id=table.blank_id,
        zero_id=table.zero_id,
        one_id=table.one_id,
    ).astype(np.int32)
    if not np.array_equal(lengths, validated_lengths):
        raise ValueError("effective_lengths disagrees with the programs")

    mutated = raw.astype(np.uint8, copy=True)
    new_lengths = validated_lengths.copy()
    population, width = mutated.shape
    structural_draw = rng.random(population, dtype=np.float32)
    delete_mask = (structural_draw < p_delete) & (new_lengths > 0)
    insert_mask = (
        (structural_draw >= p_delete)
        & (structural_draw < p_delete + p_insert)
        & (new_lengths < width)
    )

    delete_rows = np.flatnonzero(delete_mask)
    if delete_rows.size:
        delete_columns = width - new_lengths[delete_rows]
        mutated[delete_rows, delete_columns] = table.blank_id
        new_lengths[delete_rows] -= 1

    insert_rows = np.flatnonzero(insert_mask)
    if insert_rows.size:
        insert_columns = width - new_lengths[insert_rows] - 1
        logical = rng.integers(
            0, 2, size=insert_rows.size, dtype=np.uint8
        )
        inserted = np.where(
            logical == 0, table.zero_id, table.one_id
        ).astype(np.uint8)
        mutated[insert_rows, insert_columns] = inserted
        new_lengths[insert_rows] += 1

    flipped_bits = 0
    if p_bit_flip > 0.0:
        columns = np.arange(width, dtype=np.int32)[None, :]
        active = columns >= (width - new_lengths)[:, None]
        if p_bit_flip == 1.0:
            flip_mask = active
        else:
            flip_mask = (
                rng.random((population, width), dtype=np.float32)
                < p_bit_flip
            ) & active
        selected = mutated[flip_mask]
        mutated[flip_mask] = np.where(
            selected == table.zero_id, table.one_id, table.zero_id
        ).astype(np.uint8)
        flipped_bits = int(np.count_nonzero(flip_mask))

    return MutationResult(
        programs=mutated,
        effective_lengths=new_lengths,
        insertions=int(insert_rows.size),
        deletions=int(delete_rows.size),
        flipped_bits=flipped_bits,
    )


def evaluate_population_fitness_taichi(
    simulator: TaichiUTMSimulator,
    programs: np.ndarray,
    task: TaskCases,
    *,
    k_penalty: float,
) -> PopulationFitness:
    """Evaluate normalized positional Hamming fitness through CUDA.

    Missing and extra output bits are mismatches.  Invalid penalties are
    averaged over task cases::

        fitness = bit_accuracy - k_penalty * invalid_fraction
    """

    raw = np.asarray(programs)
    if raw.ndim != 2 or raw.shape[0] == 0:
        raise ValueError("programs must be a nonempty two-dimensional array")
    if raw.shape[1] != simulator.config.program_width:
        raise ValueError("program width disagrees with the simulator")
    if raw.shape[0] > simulator.batch_capacity:
        raise ValueError(
            "program population exceeds simulator batch_capacity"
        )
    if not isinstance(k_penalty, Real) or isinstance(
        k_penalty, (bool, np.bool_)
    ):
        raise TypeError("k_penalty must be a real number")
    if not np.isfinite(float(k_penalty)):
        raise ValueError("k_penalty must be finite")
    if k_penalty < 0:
        raise ValueError("k_penalty must be non-negative")
    task.validate_right_budget(simulator.config.right_budget)

    population = raw.shape[0]
    matching_bits = np.zeros(population, dtype=np.int64)
    compared_bits = np.zeros(population, dtype=np.int64)
    invalid_cases = np.zeros(population, dtype=np.uint32)
    exact = np.ones(population, dtype=np.bool_)

    for input_bits, target in task.pairs():
        result = simulator.simulate_program_batch_for_fitness(
            raw, input_bits
        )
        target_bits = np.fromiter(
            (int(bit) for bit in target),
            dtype=np.uint8,
            count=len(target),
        )
        output_lengths = result.output_lengths.astype(
            np.int64, copy=False
        )
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

    bit_accuracy = matching_bits / compared_bits
    invalid_fraction = invalid_cases / len(task)
    fitness = bit_accuracy - float(k_penalty) * invalid_fraction
    return PopulationFitness(
        fitness=fitness,
        bit_accuracy=bit_accuracy,
        invalid_fraction=invalid_fraction,
        exact=exact,
    )


def delta_selection_probabilities(
    deltas: np.ndarray,
    thresholds: Sequence[float],
    probabilities: Sequence[float],
    base_probability: float,
) -> np.ndarray:
    """Map nonnegative deltas to the configured discrete probabilities.

    Exact ties map to 0.5.  Each threshold includes its upper boundary;
    values above the final threshold use ``base_probability``.
    """

    threshold_values, probability_values, base = _validate_delta_schedule(
        thresholds, probabilities, base_probability
    )
    values = np.asarray(deltas, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("deltas must be finite and non-negative")
    bins = np.searchsorted(
        np.asarray(threshold_values, dtype=np.float64), values, side="left"
    )
    schedule = np.asarray(
        probability_values + (base,), dtype=np.float64
    )
    selected = schedule[bins]
    return np.where(values == 0.0, 0.5, selected)


def delta_tournament_select_islands(
    fitness: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    thresholds: Sequence[float],
    probabilities: Sequence[float],
    base_probability: float,
) -> np.ndarray:
    """Run independent local k=2 tournaments on every island."""

    values = np.asarray(fitness)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(
            "fitness must have shape (islands, population >= 2)"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("fitness values must be finite")
    if not _is_integer(count):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    _validate_delta_schedule(thresholds, probabilities, base_probability)

    islands, population = values.shape
    if count == 0:
        return np.empty((islands, 0), dtype=np.int64)

    contenders = rng.integers(
        0,
        population,
        size=(islands, int(count), 2),
        dtype=np.int64,
    )
    rows = np.arange(islands)[:, None]
    first = contenders[:, :, 0]
    second = contenders[:, :, 1]
    first_fitness = values[rows, first]
    second_fitness = values[rows, second]
    ties = first_fitness == second_fitness
    first_is_better = first_fitness > second_fitness
    better = np.where(first_is_better, first, second)
    worse = np.where(first_is_better, second, first)
    deltas = np.abs(first_fitness - second_fitness)
    best_probability = delta_selection_probabilities(
        deltas, thresholds, probabilities, base_probability
    )
    choose_better = rng.random((islands, int(count))) < best_probability
    non_tie_choice = np.where(choose_better, better, worse)
    tie_choice = np.where(
        rng.random((islands, int(count))) < 0.5, first, second
    )
    return np.where(ties, tie_choice, non_tie_choice)


def neutral_elite_order(
    fitness: np.ndarray,
    is_child: np.ndarray,
) -> np.ndarray:
    """Rank by fitness, preferring new children only on exact ties."""

    values = np.asarray(fitness)
    child_flags = np.asarray(is_child)
    if values.ndim != 2 or values.shape != child_flags.shape:
        raise ValueError("fitness and is_child must be equal 2D arrays")
    if not np.all(np.isfinite(values)):
        raise ValueError("fitness values must be finite")
    if child_flags.dtype != np.bool_:
        raise ValueError("is_child must have boolean dtype")
    indices = np.broadcast_to(
        np.arange(values.shape[1], dtype=np.int64), values.shape
    )
    return np.lexsort(
        (indices, -child_flags.astype(np.int8), -values), axis=1
    )


def segment_crossover(
    simulator: TaichiUTMSimulator,
    parent1: np.ndarray,
    parent1_lengths: np.ndarray,
    parent2: np.ndarray,
    parent2_lengths: np.ndarray,
    rng: np.random.Generator,
    *,
    p_crossover: float,
    crossover_segments: int,
    segment_min: int,
    segment_max: int,
) -> SegmentCrossoverResult:
    """Copy sampled parent-2 segments within the shared right suffix.

    Child length always comes from parent 1.  Sampled segments may overlap.
    """

    first = np.asarray(parent1)
    second = np.asarray(parent2)
    first_lengths = np.asarray(parent1_lengths)
    second_lengths = np.asarray(parent2_lengths)
    if (
        first.ndim != 2
        or second.shape != first.shape
        or first.shape[0] == 0
    ):
        raise ValueError("parents must be equal nonempty 2D arrays")
    if first.shape[1] != simulator.config.program_width:
        raise ValueError("parent width disagrees with the simulator")
    if first_lengths.shape != (first.shape[0],):
        raise ValueError("parent1_lengths has the wrong shape")
    if second_lengths.shape != (first.shape[0],):
        raise ValueError("parent2_lengths has the wrong shape")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    _validate_probability("p_crossover", p_crossover)
    for name, value in (
        ("crossover_segments", crossover_segments),
        ("segment_min", segment_min),
        ("segment_max", segment_max),
    ):
        if not _is_integer(value):
            raise TypeError(f"{name} must be an integer")
    if crossover_segments < 1:
        raise ValueError("crossover_segments must be positive")
    if segment_min < 1:
        raise ValueError("segment_min must be positive")
    if segment_max < segment_min:
        raise ValueError("segment_max must be at least segment_min")

    table = simulator.table
    validated_first = effective_program_lengths(
        first,
        blank_id=table.blank_id,
        zero_id=table.zero_id,
        one_id=table.one_id,
    ).astype(np.int32)
    validated_second = effective_program_lengths(
        second,
        blank_id=table.blank_id,
        zero_id=table.zero_id,
        one_id=table.one_id,
    ).astype(np.int32)
    if not np.array_equal(first_lengths, validated_first):
        raise ValueError("parent1_lengths disagrees with parent1")
    if not np.array_equal(second_lengths, validated_second):
        raise ValueError("parent2_lengths disagrees with parent2")

    children = first.astype(np.uint8, copy=True)
    child_lengths = validated_first.copy()
    count, width = children.shape
    overlap = np.minimum(validated_first, validated_second)
    if p_crossover == 0.0:
        selected = np.zeros(count, dtype=np.bool_)
    elif p_crossover == 1.0:
        selected = np.ones(count, dtype=np.bool_)
    else:
        selected = rng.random(count) < p_crossover
    active = selected & (overlap > 0)
    copy_mask = np.zeros((count, width), dtype=np.bool_)
    columns = np.arange(width, dtype=np.int32)[None, :]
    effective_min = np.minimum(int(segment_min), overlap)
    effective_max = np.minimum(int(segment_max), overlap)

    for _ in range(int(crossover_segments)):
        length_span = effective_max - effective_min + 1
        segment_lengths = effective_min + np.floor(
            rng.random(count) * length_span
        ).astype(np.int32)
        start_span = overlap - segment_lengths + 1
        offsets = np.floor(rng.random(count) * start_span).astype(np.int32)
        starts = width - overlap + offsets
        ends = starts + segment_lengths
        copy_mask |= (
            active[:, None]
            & (columns >= starts[:, None])
            & (columns < ends[:, None])
        )

    children[copy_mask] = second[copy_mask]
    crossed_children = int(np.count_nonzero(active))
    return SegmentCrossoverResult(
        programs=children,
        effective_lengths=child_lengths,
        crossed_children=crossed_children,
        sampled_segments=crossed_children * int(crossover_segments),
        copied_loci=int(np.count_nonzero(copy_mask)),
    )


def _history_from_lists(history: dict[str, list]) -> IslandGAHistory:
    return IslandGAHistory(
        generations=np.asarray(history["generations"], dtype=np.uint32),
        current_best_fitness=np.asarray(
            history["current_best_fitness"], dtype=np.float64
        ),
        global_best_fitness=np.asarray(
            history["global_best_fitness"], dtype=np.float64
        ),
        mean_fitness=np.asarray(history["mean_fitness"], dtype=np.float64),
        std_fitness=np.asarray(history["std_fitness"], dtype=np.float64),
        mean_island_best=np.asarray(
            history["mean_island_best"], dtype=np.float64
        ),
        std_island_best=np.asarray(
            history["std_island_best"], dtype=np.float64
        ),
        best_bit_accuracy=np.asarray(
            history["best_bit_accuracy"], dtype=np.float64
        ),
        best_invalid_fraction=np.asarray(
            history["best_invalid_fraction"], dtype=np.float64
        ),
        exact_programs=np.asarray(history["exact_programs"], dtype=np.uint32),
        island_best_fitness=np.asarray(
            history["island_best_fitness"], dtype=np.float64
        ),
        promoted_child_elites=np.asarray(
            history["promoted_child_elites"], dtype=np.uint32
        ),
        cross_island_per_island=np.asarray(
            history["cross_island_per_island"], dtype=np.uint32
        ),
        children_per_island=np.asarray(
            history["children_per_island"], dtype=np.uint32
        ),
        crossed_children=np.asarray(
            history["crossed_children"], dtype=np.uint32
        ),
        sampled_segments=np.asarray(
            history["sampled_segments"], dtype=np.uint64
        ),
        copied_loci=np.asarray(history["copied_loci"], dtype=np.uint64),
        insertions=np.asarray(history["insertions"], dtype=np.uint32),
        deletions=np.asarray(history["deletions"], dtype=np.uint32),
        flipped_bits=np.asarray(history["flipped_bits"], dtype=np.uint64),
    )


class TaichiIslandGA:
    """Independent islands with delta tournaments and neutral elitism."""

    def __init__(
        self,
        simulator: TaichiUTMSimulator,
        task: TaskCases,
        config: IslandGAConfig = IslandGAConfig(),
    ) -> None:
        if not isinstance(simulator, TaichiUTMSimulator):
            raise TypeError("simulator must be a TaichiUTMSimulator")
        if not isinstance(task, TaskCases):
            raise TypeError("task must be a TaskCases")
        if not isinstance(config, IslandGAConfig):
            raise TypeError("config must be an IslandGAConfig")
        task.validate_right_budget(simulator.config.right_budget)
        if simulator.batch_capacity < config.total_population:
            raise ValueError(
                "simulator batch_capacity must be at least the total "
                "island population"
            )
        self.simulator = simulator
        self.task = task
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.programs: np.ndarray | None = None
        self.lengths: np.ndarray | None = None
        self.is_child: np.ndarray | None = None
        self.fitness: PopulationFitness | None = None
        self.global_best: IslandChampion | None = None
        self._ran = False

    def _initialize(self, initial_programs: np.ndarray | None) -> None:
        config = self.config
        width = self.simulator.config.program_width
        shape = (config.islands, config.population_per_island, width)
        if initial_programs is None:
            flat_programs, flat_lengths = initialize_population(
                self.simulator, config.total_population, self.rng
            )
            self.programs = flat_programs.reshape(shape)
            self.lengths = flat_lengths.reshape(shape[:2])
        else:
            raw = np.asarray(initial_programs)
            if raw.shape != shape:
                raise ValueError(
                    "initial_programs must have shape "
                    "(islands, population_per_island, program_width)"
                )
            table = self.simulator.table
            flat = raw.reshape(config.total_population, width)
            lengths = effective_program_lengths(
                flat,
                blank_id=table.blank_id,
                zero_id=table.zero_id,
                one_id=table.one_id,
            ).astype(np.int32)
            self.programs = raw.astype(np.uint8, copy=True)
            self.lengths = lengths.reshape(shape[:2])
        self.is_child = np.zeros(shape[:2], dtype=np.bool_)

    def _evaluate(self) -> PopulationFitness:
        if self.programs is None:
            raise RuntimeError("population is not initialized")
        flat = self.programs.reshape(
            self.config.total_population,
            self.simulator.config.program_width,
        )
        evaluated = evaluate_population_fitness_taichi(
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

    def _champion(
        self,
        generation: int,
        fitness: PopulationFitness,
    ) -> IslandChampion:
        if self.programs is None or self.lengths is None:
            raise RuntimeError("population is not initialized")
        flat_index = int(np.argmax(fitness.fitness))
        island, island_index = np.unravel_index(
            flat_index, fitness.fitness.shape
        )
        symbols = self.programs[island, island_index].copy()
        table = self.simulator.table
        return IslandChampion(
            generation=generation,
            island=int(island),
            island_index=int(island_index),
            program=effective_program_to_string(
                symbols,
                blank_id=table.blank_id,
                zero_id=table.zero_id,
                one_id=table.one_id,
            ),
            padded_program=padded_program_to_string(
                symbols,
                blank_id=table.blank_id,
                zero_id=table.zero_id,
                one_id=table.one_id,
            ),
            symbols=symbols,
            effective_length=int(self.lengths[island, island_index]),
            fitness=float(fitness.fitness[island, island_index]),
            bit_accuracy=float(fitness.bit_accuracy[island, island_index]),
            invalid_fraction=float(
                fitness.invalid_fraction[island, island_index]
            ),
            exact=bool(fitness.exact[island, island_index]),
        )

    def _make_next_generation(
        self,
        generation: int,
        fitness: PopulationFitness,
    ) -> ReproductionStats:
        if (
            self.programs is None
            or self.lengths is None
            or self.is_child is None
        ):
            raise RuntimeError("population is not initialized")
        config = self.config
        islands = config.islands
        population = config.population_per_island
        width = self.simulator.config.program_width
        elite_count = config.elite_count
        random_count = config.random_immigrant_count
        cross_active = generation % config.cross_island_interval == 0
        cross_count = config.cross_island_count if cross_active else 0
        child_count = population - elite_count - random_count - cross_count

        order = neutral_elite_order(fitness.fitness, self.is_child)
        elite_indices = order[:, :elite_count]
        island_rows = np.arange(islands)[:, None]
        promoted_child_elites = int(
            np.count_nonzero(self.is_child[island_rows, elite_indices])
        )
        elite_programs = self.programs[island_rows, elite_indices].copy()
        elite_lengths = self.lengths[island_rows, elite_indices].copy()

        next_programs = np.empty_like(self.programs)
        next_lengths = np.empty_like(self.lengths)
        next_is_child = np.zeros_like(self.is_child)
        next_programs[:, :elite_count] = elite_programs
        next_lengths[:, :elite_count] = elite_lengths
        position = elite_count

        if cross_count:
            for destination in range(islands):
                source_draw = self.rng.integers(
                    0, islands - 1, size=cross_count
                )
                source_islands = source_draw + (source_draw >= destination)
                source_elites = self.rng.integers(
                    0, elite_count, size=cross_count
                )
                stop = position + cross_count
                next_programs[destination, position:stop] = elite_programs[
                    source_islands, source_elites
                ]
                next_lengths[destination, position:stop] = elite_lengths[
                    source_islands, source_elites
                ]
            position += cross_count

        if random_count:
            random_programs, random_lengths = initialize_population(
                self.simulator, islands * random_count, self.rng
            )
            stop = position + random_count
            next_programs[:, position:stop] = random_programs.reshape(
                islands, random_count, width
            )
            next_lengths[:, position:stop] = random_lengths.reshape(
                islands, random_count
            )
            position = stop

        crossed_children = 0
        sampled_segments = 0
        copied_loci = 0
        insertions = 0
        deletions = 0
        flipped_bits = 0
        if child_count:
            selection = {
                "thresholds": config.tournament_delta_thresholds,
                "probabilities": config.tournament_best_probabilities,
                "base_probability": config.tournament_base_probability,
            }
            parent1_indices = delta_tournament_select_islands(
                fitness.fitness, child_count, self.rng, **selection
            )
            parent2_indices = delta_tournament_select_islands(
                fitness.fitness, child_count, self.rng, **selection
            )
            parent_rows = np.arange(islands)[:, None]
            parent1 = self.programs[
                parent_rows, parent1_indices
            ].reshape(islands * child_count, width)
            parent2 = self.programs[
                parent_rows, parent2_indices
            ].reshape(islands * child_count, width)
            parent1_lengths = self.lengths[
                parent_rows, parent1_indices
            ].reshape(-1)
            parent2_lengths = self.lengths[
                parent_rows, parent2_indices
            ].reshape(-1)

            crossover = segment_crossover(
                self.simulator,
                parent1,
                parent1_lengths,
                parent2,
                parent2_lengths,
                self.rng,
                p_crossover=config.p_crossover,
                crossover_segments=config.crossover_segments,
                segment_min=config.crossover_segment_min,
                segment_max=config.crossover_segment_max,
            )
            mutation = mutate_population(
                self.simulator,
                crossover.programs,
                crossover.effective_lengths,
                self.rng,
                p_insert=config.p_insert,
                p_delete=config.p_delete,
                p_bit_flip=config.p_bit_flip,
            )
            next_programs[:, position:] = mutation.programs.reshape(
                islands, child_count, width
            )
            next_lengths[:, position:] = mutation.effective_lengths.reshape(
                islands, child_count
            )
            next_is_child[:, position:] = True
            crossed_children = crossover.crossed_children
            sampled_segments = crossover.sampled_segments
            copied_loci = crossover.copied_loci
            insertions = mutation.insertions
            deletions = mutation.deletions
            flipped_bits = mutation.flipped_bits

        self.programs = next_programs
        self.lengths = next_lengths
        self.is_child = next_is_child
        return ReproductionStats(
            elites_per_island=elite_count,
            promoted_child_elites=promoted_child_elites,
            cross_island_per_island=cross_count,
            random_immigrants_per_island=random_count,
            children_per_island=child_count,
            crossed_children=crossed_children,
            sampled_segments=sampled_segments,
            copied_loci=copied_loci,
            insertions=insertions,
            deletions=deletions,
            flipped_bits=flipped_bits,
        )

    def _record(
        self,
        history: dict[str, list],
        generation: int,
        fitness: PopulationFitness,
        reproduction: ReproductionStats,
        started: float,
        terminal: bool,
    ) -> None:
        current = self._champion(generation, fitness)
        if self.global_best is None or current.fitness > self.global_best.fitness:
            self.global_best = current
        if self.global_best is None:
            raise RuntimeError("global best was not initialized")

        island_best = np.max(fitness.fitness, axis=1)
        history["generations"].append(generation)
        history["current_best_fitness"].append(current.fitness)
        history["global_best_fitness"].append(self.global_best.fitness)
        history["mean_fitness"].append(float(np.mean(fitness.fitness)))
        history["std_fitness"].append(float(np.std(fitness.fitness)))
        history["mean_island_best"].append(float(np.mean(island_best)))
        history["std_island_best"].append(float(np.std(island_best)))
        history["best_bit_accuracy"].append(current.bit_accuracy)
        history["best_invalid_fraction"].append(current.invalid_fraction)
        history["exact_programs"].append(int(np.count_nonzero(fitness.exact)))
        history["island_best_fitness"].append(island_best.copy())
        history["promoted_child_elites"].append(
            reproduction.promoted_child_elites
        )
        history["cross_island_per_island"].append(
            reproduction.cross_island_per_island
        )
        history["children_per_island"].append(
            reproduction.children_per_island
        )
        history["crossed_children"].append(reproduction.crossed_children)
        history["sampled_segments"].append(reproduction.sampled_segments)
        history["copied_loci"].append(reproduction.copied_loci)
        history["insertions"].append(reproduction.insertions)
        history["deletions"].append(reproduction.deletions)
        history["flipped_bits"].append(reproduction.flipped_bits)

        if self.config.print_every is not None and (
            generation % self.config.print_every == 0 or terminal
        ):
            shown = " ".join(
                f"{index + 1}:{value:.4f}"
                for index, value in enumerate(island_best[:10])
            )
            print(
                f"generation={generation:>5} "
                f"current_best={current.fitness:.6f} "
                f"global_best={self.global_best.fitness:.6f} "
                f"mean_island_best={np.mean(island_best):.6f} "
                f"accuracy={current.bit_accuracy:.6f} "
                f"invalid={current.invalid_fraction:.3f} "
                f"exact={history['exact_programs'][-1]:,} "
                f"promoted_children={reproduction.promoted_child_elites:,} "
                f"crossed={reproduction.crossed_children:,} "
                f"program={self.global_best.program!r} "
                f"elapsed={perf_counter() - started:.3f}s",
                flush=True,
            )
            print(f"island_best -> {shown}", flush=True)

    def run(
        self,
        *,
        initial_programs: np.ndarray | None = None,
    ) -> IslandGAResult:
        """Run this trainer once and return its complete in-memory result."""

        if self._ran:
            raise RuntimeError("this trainer instance has already run")
        self._ran = True
        self._initialize(initial_programs)
        started = perf_counter()
        history: dict[str, list] = {
            "generations": [],
            "current_best_fitness": [],
            "global_best_fitness": [],
            "mean_fitness": [],
            "std_fitness": [],
            "mean_island_best": [],
            "std_island_best": [],
            "best_bit_accuracy": [],
            "best_invalid_fraction": [],
            "exact_programs": [],
            "island_best_fitness": [],
            "promoted_child_elites": [],
            "cross_island_per_island": [],
            "children_per_island": [],
            "crossed_children": [],
            "sampled_segments": [],
            "copied_loci": [],
            "insertions": [],
            "deletions": [],
            "flipped_bits": [],
        }
        zero_stats = ReproductionStats(
            elites_per_island=self.config.elite_count,
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

        generation = 0
        fitness = self._evaluate()
        self.fitness = fitness
        terminal = (
            self.config.stop_first_exact and np.any(fitness.exact)
        ) or self.config.generations == 0
        self._record(
            history, generation, fitness, zero_stats, started, terminal
        )

        while not terminal:
            generation += 1
            reproduction = self._make_next_generation(generation, fitness)
            fitness = self._evaluate()
            self.fitness = fitness
            terminal = (
                self.config.stop_first_exact and np.any(fitness.exact)
            ) or generation == self.config.generations
            self._record(
                history,
                generation,
                fitness,
                reproduction,
                started,
                terminal,
            )

        if (
            self.global_best is None
            or self.programs is None
            or self.lengths is None
        ):
            raise RuntimeError("trainer did not produce a result")
        return IslandGAResult(
            best=self.global_best,
            solved=self.global_best.exact,
            generations_completed=generation,
            evaluated_programs=self.config.total_population * (generation + 1),
            elapsed_seconds=perf_counter() - started,
            history=_history_from_lists(history),
            final_programs=self.programs.copy(),
            final_effective_lengths=self.lengths.copy(),
        )
