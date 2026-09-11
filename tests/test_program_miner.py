"""Focused tests for the endless program miner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from program_miner import (
    Candidate,
    MinerSettings,
    MiningIslandGA,
    MiningLimits,
    MiningTask,
    ProgramMiner,
    ProgramRegistry,
    evaluate_miner_fitness,
    load_miner_setup,
    pack_program,
    unpack_program,
)
from island_ga import IslandGAConfig
from programs import get_program
from taichi_backend import TaichiUTMSimulator, initialize_taichi_cuda
from utils import SimulatorConfig, TaskCases


class FakeSimulator:
    def __init__(self) -> None:
        self.config = SimpleNamespace(program_width=2, right_budget=4)
        self.batch_capacity = 2

    def simulate_program_batch_for_fitness(
        self, programs: np.ndarray, input_bits: str
    ) -> SimpleNamespace:
        del programs, input_bits
        return SimpleNamespace(
            output_bits=np.asarray([[0, 2, 2, 2], [0, 2, 2, 2]], dtype=np.uint8),
            output_lengths=np.asarray([1, 1], dtype=np.uint32),
            invalid=np.asarray([False, True]),
        )


class ProgramMinerUnitTests(unittest.TestCase):
    def test_invalid_case_has_zero_bit_accuracy(self) -> None:
        result = evaluate_miner_fitness(
            FakeSimulator(),
            np.zeros((2, 2), dtype=np.uint8),
            TaskCases(["0"], ["0"]),
            k_penalty=0.5,
        )
        np.testing.assert_array_equal(result.bit_accuracy, [1.0, 0.0])
        np.testing.assert_array_equal(result.invalid_fraction, [0.0, 1.0])
        np.testing.assert_array_equal(result.fitness, [1.0, -0.5])
        np.testing.assert_array_equal(result.exact, [True, False])

    def test_program_packing_preserves_length_and_leading_zeros(self) -> None:
        for program in ("", "0", "001", "1" * 400, "0" * 399 + "1"):
            length, packed = pack_program(program)
            self.assertEqual(unpack_program(length, packed), program)
        self.assertNotEqual(pack_program("1"), pack_program("001"))

    def test_task_loading_and_database_deduplication(self) -> None:
        settings, tasks = load_miner_setup()
        self.assertGreaterEqual(len(tasks), 2)
        self.assertIsNone(tasks[0].ga.seed)
        self.assertFalse(tasks[0].ga.stop_first_exact)
        first_task = tasks[0].name
        second_task = tasks[1].name

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "programs.sqlite3"
            with ProgramRegistry(path) as registry:
                identity_id = registry.register_task(tasks[0])
                run = registry.start_run(identity_id)
                first = registry.archive(
                    run,
                    7,
                    [
                        Candidate("001", 0.9, False),
                        Candidate("001", 0.9, False),
                        Candidate("1", 1.0, True),
                    ],
                )
                self.assertEqual(first.distinct_in_batch, 2)
                self.assertEqual(first.distinct_in_run, 2)
                self.assertEqual(first.new_programs_added, 2)

                repeated = registry.archive(
                    run, 8, [Candidate("001", 0.9, False)]
                )
                self.assertEqual(repeated.distinct_in_run, 2)
                self.assertEqual(repeated.new_programs_added, 0)
                registry.finish_run(
                    run,
                    generations_completed=8,
                    programs_found=2,
                    new_programs_added=2,
                    best_accuracy=1.0,
                    first_exact_generation=7,
                    stop_reason="test",
                )

                append_id = registry.register_task(tasks[1])
                second_run = registry.start_run(append_id)
                association = registry.archive(
                    second_run, 2, [Candidate("001", 0.85, False)]
                )
                self.assertEqual(association.new_programs_added, 1)
                registry.finish_run(
                    second_run,
                    generations_completed=2,
                    programs_found=1,
                    new_programs_added=1,
                    best_accuracy=0.85,
                    first_exact_generation=None,
                    stop_reason="test",
                )

                origins = registry.lookup_program("001")
                self.assertEqual(
                    {item["task"] for item in origins},
                    {first_task, second_task},
                )
                program_count = registry.connection.execute(
                    "SELECT COUNT(*) FROM programs"
                ).fetchone()[0]
                self.assertEqual(program_count, 2)
                statistics = registry.statistics(recent=5)
                self.assertEqual(statistics["unique_programs"], 2)
                self.assertEqual(
                    statistics["task_program_associations"], 3
                )
                self.assertEqual(statistics["exact_associations"], 1)
                self.assertEqual(statistics["runs"], 2)
                self.assertEqual(statistics["completed_runs"], 2)
                self.assertEqual(
                    statistics["tasks_with_exact_programs"], [first_task]
                )
                self.assertEqual(len(statistics["recent_runs"]), 2)

            with ProgramRegistry(path, read_only=True) as registry:
                read_only_statistics = registry.statistics(recent=0)
                self.assertEqual(read_only_statistics["unique_programs"], 2)
                self.assertEqual(read_only_statistics["completed_runs"], 2)

        self.assertTrue(str(settings.database_path).endswith("programs.sqlite3"))


class ProgramMinerCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        initialize_taichi_cuda()

    def test_known_valid_and_invalid_programs(self) -> None:
        divergent = get_program("divergent_loop").program
        width = max(24, len(divergent))
        simulator = TaichiUTMSimulator(
            SimulatorConfig(
                program_width=width,
                left_budget=width + 8,
                right_budget=8,
                t_max=100,
            ),
            batch_capacity=2,
        )
        programs = simulator.encode_programs(
            [get_program("identity_short").program, divergent]
        )
        result = evaluate_miner_fitness(
            simulator,
            programs,
            TaskCases(["0"], ["0"]),
            k_penalty=0.5,
        )
        self.assertTrue(result.exact[0])
        self.assertEqual(result.bit_accuracy[0], 1.0)
        self.assertEqual(result.invalid_fraction[0], 0.0)
        self.assertFalse(result.exact[1])
        self.assertEqual(result.bit_accuracy[1], 0.0)
        self.assertEqual(result.invalid_fraction[1], 1.0)

    @staticmethod
    def _tiny_task() -> MiningTask:
        return MiningTask(
            name="tiny_identity",
            cases=TaskCases(["0", "1"], ["0", "1"]),
            simulator=SimulatorConfig(
                program_width=8,
                left_budget=10,
                right_budget=8,
                t_max=100,
            ),
            ga=IslandGAConfig(
                islands=1,
                population_per_island=4,
                generations=0,
                elite_fraction=0.25,
                cross_island_fraction=0.0,
                cross_island_interval=1,
                random_immigrant_fraction=0.0,
                k_penalty=0.5,
                seed=None,
                print_every=None,
                stop_first_exact=False,
            ),
            limits=MiningLimits(
                max_generations=0,
                post_perfect_generations=2,
                min_accuracy=0.99,
            ),
            definition_json='{"name":"tiny_identity"}',
            signature="tiny-identity-test",
        )

    def test_tiny_episode_and_clean_interrupt(self) -> None:
        task = self._tiny_task()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = MinerSettings(
                database_path=root / "programs.sqlite3",
                print_every_generations=1,
            )
            with ProgramRegistry(settings.database_path) as registry:
                ProgramMiner(settings, [task], registry).run_forever(1)
                completed = registry.connection.execute(
                    """
                    SELECT generations_completed, stop_reason, ended_at
                    FROM runs WHERE run_number = 1
                    """
                ).fetchone()
                self.assertEqual(completed[0], 0)
                self.assertEqual(completed[1], "generation-budget")
                self.assertIsNotNone(completed[2])

                miner = ProgramMiner(settings, [task], registry)
                with patch.object(
                    MiningIslandGA,
                    "_evaluate",
                    side_effect=KeyboardInterrupt,
                ):
                    miner.run_forever(1)
                interrupted = registry.connection.execute(
                    """
                    SELECT stop_reason, ended_at
                    FROM runs WHERE run_number = 2
                    """
                ).fetchone()
                self.assertEqual(interrupted[0], "interrupted")
                self.assertIsNotNone(interrupted[1])


if __name__ == "__main__":
    unittest.main()
