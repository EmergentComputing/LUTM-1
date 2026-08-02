"""Behavioral tests for the NumPy island GA and CUDA evaluation path."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from island_ga import (
    IslandGAConfig,
    TaichiIslandGA,
    delta_selection_probabilities,
    delta_tournament_select_islands,
    evaluate_population_fitness_taichi,
    initialize_population,
    mutate_population,
    neutral_elite_order,
    segment_crossover,
)
from lutm import TransitionTable
from programs import PROGRAMS
from taichi_backend import TaichiUTMSimulator, initialize_taichi_cuda
from utils import (
    SimulatorConfig,
    TaskCases,
    effective_program_lengths,
    effective_program_to_string,
    encode_padded_programs,
)


class IslandGATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        initialize_taichi_cuda()
        cls.table = TransitionTable()

    def simulator(
        self,
        width: int,
        capacity: int,
        *,
        right_budget: int = 16,
        t_max: int = 1_000,
        left_budget: int | None = None,
    ) -> TaichiUTMSimulator:
        left = width + 1 if left_budget is None else left_budget
        return TaichiUTMSimulator(
            SimulatorConfig(width, left, right_budget, t_max),
            capacity,
            self.table,
        )

    def encode(self, programs: list[str], width: int) -> np.ndarray:
        return encode_padded_programs(
            programs,
            width,
            blank_id=self.table.blank_id,
            zero_id=self.table.zero_id,
            one_id=self.table.one_id,
        )

    def strings(self, programs: np.ndarray) -> list[str]:
        return [
            effective_program_to_string(
                row,
                blank_id=self.table.blank_id,
                zero_id=self.table.zero_id,
                one_id=self.table.one_id,
            )
            for row in programs
        ]

    def lengths(self, programs: np.ndarray) -> np.ndarray:
        return effective_program_lengths(
            programs,
            blank_id=self.table.blank_id,
            zero_id=self.table.zero_id,
            one_id=self.table.one_id,
        )

    def test_config_validation_and_derived_counts(self) -> None:
        config = IslandGAConfig(
            islands=4,
            population_per_island=20,
            generations=7,
            elite_fraction=0.15,
            cross_island_fraction=0.10,
            random_immigrant_fraction=0.20,
            print_every=None,
        )
        self.assertEqual(config.total_population, 80)
        self.assertEqual(config.elite_count, 3)
        self.assertEqual(config.cross_island_count, 2)
        self.assertEqual(config.random_immigrant_count, 4)

        invalid = (
            dict(population_per_island=1),
            dict(p_insert=0.6, p_delete=0.5),
            dict(tournament_delta_thresholds=(0.2, 0.05)),
            dict(tournament_best_probabilities=(0.52,)),
            dict(tournament_base_probability=0.55),
            dict(islands=1, cross_island_fraction=0.1),
            dict(
                population_per_island=10,
                elite_fraction=0.6,
                cross_island_fraction=0.2,
                random_immigrant_fraction=0.3,
            ),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                IslandGAConfig(**arguments)

    def test_delta_schedule_boundaries_and_tie(self) -> None:
        just_above_first = np.nextafter(0.05, np.inf)
        just_above_second = np.nextafter(0.20, np.inf)
        deltas = np.array(
            [0.0, 1e-12, 0.05, just_above_first, 0.20, just_above_second]
        )
        actual = delta_selection_probabilities(
            deltas,
            thresholds=(0.05, 0.20),
            probabilities=(0.52, 0.60),
            base_probability=0.90,
        )
        np.testing.assert_array_equal(actual, [0.50, 0.52, 0.52, 0.60, 0.60, 0.90])

    def test_empirical_tournament_selection_rates(self) -> None:
        fitness = np.array(
            [[0.0, 0.0], [0.0, 0.01], [0.0, 0.10], [0.0, 0.50]],
            dtype=np.float64,
        )
        selected = delta_tournament_select_islands(
            fitness,
            250_000,
            np.random.default_rng(417),
            thresholds=(0.05, 0.20),
            probabilities=(0.52, 0.60),
            base_probability=0.90,
        )
        # With two uniformly sampled contenders, P(select index 1) is
        # 1/4 + P(select fitter)/2.  The equal-fitness island stays neutral.
        observed = np.mean(selected == 1, axis=1)
        np.testing.assert_allclose(
            observed,
            [0.50, 0.51, 0.55, 0.70],
            atol=0.004,
            rtol=0.0,
        )

    def test_padded_initialization_and_exact_mutation_geometry(self) -> None:
        simulator = self.simulator(6, 4)
        initialized, initialized_lengths = initialize_population(
            simulator, 2_000, np.random.default_rng(9)
        )
        np.testing.assert_array_equal(self.lengths(initialized), initialized_lengths)
        self.assertEqual(int(initialized_lengths.min()), 0)
        self.assertEqual(int(initialized_lengths.max()), 6)

        source = self.encode(["", "0", "0101", "101010"], 6)
        source_lengths = self.lengths(source)

        deleted = mutate_population(
            simulator,
            source,
            source_lengths,
            np.random.default_rng(10),
            p_insert=0.0,
            p_delete=1.0,
            p_bit_flip=0.0,
        )
        self.assertEqual(self.strings(deleted.programs), ["", "", "101", "01010"])
        np.testing.assert_array_equal(deleted.effective_lengths, [0, 0, 3, 5])
        self.assertEqual((deleted.insertions, deleted.deletions), (0, 3))

        inserted = mutate_population(
            simulator,
            source,
            source_lengths,
            np.random.default_rng(11),
            p_insert=1.0,
            p_delete=0.0,
            p_bit_flip=0.0,
        )
        inserted_strings = self.strings(inserted.programs)
        np.testing.assert_array_equal(inserted.effective_lengths, [1, 2, 5, 6])
        for old, new in zip(self.strings(source)[:3], inserted_strings[:3]):
            self.assertIn(new[0], "01")
            self.assertEqual(new[1:], old)
        self.assertEqual(inserted_strings[3], "101010")
        self.assertEqual((inserted.insertions, inserted.deletions), (3, 0))

        flipped = mutate_population(
            simulator,
            source,
            source_lengths,
            np.random.default_rng(12),
            p_insert=0.0,
            p_delete=0.0,
            p_bit_flip=1.0,
        )
        self.assertEqual(self.strings(flipped.programs), ["", "1", "1010", "010101"])
        np.testing.assert_array_equal(flipped.effective_lengths, source_lengths)
        self.assertEqual(flipped.flipped_bits, int(source_lengths.sum()))

    def test_segment_crossover_and_neutral_child_preference(self) -> None:
        simulator = self.simulator(8, 3)
        parent1 = self.encode(["", "00000000", "000"], 8)
        parent2 = self.encode(["11111111", "11111", "11111111"], 8)
        first_lengths = self.lengths(parent1)
        second_lengths = self.lengths(parent2)
        crossed = segment_crossover(
            simulator,
            parent1,
            first_lengths,
            parent2,
            second_lengths,
            np.random.default_rng(22),
            p_crossover=1.0,
            crossover_segments=2,
            segment_min=2,
            segment_max=3,
        )
        np.testing.assert_array_equal(crossed.effective_lengths, first_lengths)
        self.assertEqual(crossed.crossed_children, 2)
        self.assertEqual(crossed.sampled_segments, 4)
        child_strings = self.strings(crossed.programs)
        self.assertEqual(child_strings[0], "")
        for child, overlap in zip(child_strings[1:], (5, 3)):
            copied = child[-overlap:].count("1")
            self.assertGreaterEqual(copied, 2)
            self.assertLessEqual(copied, min(overlap, 6))
            self.assertNotIn("1", child[:-overlap])
        self.assertEqual(crossed.copied_loci, sum(s.count("1") for s in child_strings))

        unchanged = segment_crossover(
            simulator,
            parent1,
            first_lengths,
            parent2,
            second_lengths,
            np.random.default_rng(23),
            p_crossover=0.0,
            crossover_segments=2,
            segment_min=1,
            segment_max=3,
        )
        np.testing.assert_array_equal(unchanged.programs, parent1)
        self.assertEqual(unchanged.crossed_children, 0)

        order = neutral_elite_order(
            np.array([[0.8, 1.0, 1.0, 0.8, 1.0]]),
            np.array([[True, False, True, False, True]], dtype=np.bool_),
        )
        np.testing.assert_array_equal(order, [[2, 4, 1, 0, 3]])

    def test_cuda_fitness_on_verified_plus_one_and_times_two(self) -> None:
        plus_one = PROGRAMS["plus_one"]
        times_two = PROGRAMS["times_two"]
        width = max(len(plus_one.program), len(times_two.program))
        simulator = self.simulator(
            width,
            2,
            right_budget=32,
            t_max=1_000_000,
            left_budget=width + 2,
        )
        programs = self.encode([plus_one.program, times_two.program], width)
        inputs = [format(value, "b") for value in (0, 1, 2, 3, 15, 16, 255)]

        for index, registered in enumerate((plus_one, times_two)):
            task = TaskCases(inputs, [registered.target(bits) for bits in inputs])
            fitness = evaluate_population_fitness_taichi(
                simulator, programs, task, k_penalty=0.5
            )
            self.assertTrue(bool(fitness.exact[index]))
            self.assertEqual(float(fitness.bit_accuracy[index]), 1.0)
            self.assertEqual(float(fitness.invalid_fraction[index]), 0.0)
            self.assertEqual(float(fitness.fitness[index]), 1.0)

    def test_stop_first_exact_terminates_before_reproduction(self) -> None:
        simulator = self.simulator(5, 8, right_budget=8, t_max=250)
        identity = self.encode([PROGRAMS["identity_short"].program], 5)[0]
        initial = np.broadcast_to(identity, (2, 4, 5)).copy()
        trainer = TaichiIslandGA(
            simulator,
            TaskCases(["0", "101"], ["0", "101"]),
            IslandGAConfig(
                islands=2,
                population_per_island=4,
                generations=5,
                elite_fraction=0.25,
                cross_island_fraction=0.0,
                random_immigrant_fraction=0.0,
                seed=6,
                print_every=None,
                stop_first_exact=True,
            ),
        )
        result = trainer.run(initial_programs=initial)
        self.assertTrue(result.solved)
        self.assertEqual(result.best.program, PROGRAMS["identity_short"].program)
        self.assertEqual(result.generations_completed, 0)
        self.assertEqual(result.evaluated_programs, 8)
        np.testing.assert_array_equal(result.history.generations, [0])

    def test_migration_evolution_and_seed_reproducibility(self) -> None:
        simulator = self.simulator(8, 24, right_budget=8, t_max=250)
        task = TaskCases(["0", "1"], ["1", "0"])
        config = IslandGAConfig(
            islands=3,
            population_per_island=8,
            generations=2,
            elite_fraction=0.25,
            cross_island_fraction=0.25,
            cross_island_interval=1,
            random_immigrant_fraction=0.125,
            p_crossover=1.0,
            crossover_segments=2,
            crossover_segment_min=1,
            crossover_segment_max=3,
            p_insert=0.2,
            p_delete=0.3,
            p_bit_flip=0.1,
            seed=12345,
            print_every=None,
            stop_first_exact=False,
        )
        first = TaichiIslandGA(simulator, task, config).run()
        second = TaichiIslandGA(simulator, task, config).run()

        self.assertEqual(first.generations_completed, 2)
        self.assertEqual(first.evaluated_programs, 72)
        np.testing.assert_array_equal(first.final_programs, second.final_programs)
        np.testing.assert_array_equal(
            first.final_effective_lengths, second.final_effective_lengths
        )
        np.testing.assert_array_equal(
            first.history.global_best_fitness, second.history.global_best_fitness
        )
        np.testing.assert_array_equal(
            first.history.cross_island_per_island, [0, 2, 2]
        )
        np.testing.assert_array_equal(first.history.children_per_island, [0, 3, 3])
        self.assertTrue(np.all(np.diff(first.history.global_best_fitness) >= 0.0))
        flat = first.final_programs.reshape(24, 8)
        np.testing.assert_array_equal(
            self.lengths(flat), first.final_effective_lengths.reshape(-1)
        )

    def test_single_large_40000_population_cuda_stress(self) -> None:
        # Required stress dimensions: tape = 70 + 29 + 1 = 100, T_max = 8,000.
        simulator = self.simulator(
            64,
            40_000,
            left_budget=70,
            right_budget=29,
            t_max=8_000,
        )
        self.assertEqual(simulator.config.tape_size, 100)
        identity = self.encode([PROGRAMS["identity_short"].program], 64)[0]
        initial = np.broadcast_to(identity, (40, 1_000, 64)).copy()
        config = IslandGAConfig(
            islands=40,
            population_per_island=1_000,
            generations=1,
            elite_fraction=0.01,
            cross_island_fraction=0.0,
            random_immigrant_fraction=0.0,
            p_crossover=0.0,
            p_insert=0.0,
            p_delete=0.0,
            p_bit_flip=0.0,
            seed=70,
            print_every=None,
            stop_first_exact=False,
        )
        trainer = TaichiIslandGA(
            simulator, TaskCases(["1"], ["1"]), config
        )
        result = trainer.run(initial_programs=initial)

        self.assertEqual(config.total_population, 40_000)
        self.assertEqual(result.generations_completed, 1)
        self.assertEqual(result.evaluated_programs, 80_000)
        self.assertTrue(result.solved)
        np.testing.assert_array_equal(result.history.exact_programs, [40_000, 40_000])
        self.assertEqual(result.final_programs.shape, (40, 1_000, 64))
        self.assertTrue(np.all(result.final_effective_lengths == 5))
        self.assertIsNotNone(trainer.fitness)
        self.assertTrue(np.all(trainer.fitness.exact))
        self.assertTrue(np.all(trainer.fitness.invalid_fraction == 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
