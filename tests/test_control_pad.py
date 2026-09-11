"""Focused tests for the live control-pad backend."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control_pad_backend import (
    ControlPadTaichiSimulator,
    LiveTrainingSession,
    evaluate_population_with_diagnostics,
    parse_new_run_payload,
)
from control_pad import ControlPadController
from island_ga import evaluate_population_fitness_taichi
from taichi_backend import initialize_taichi_cuda
from utils import SimulatorConfig, TaskCases


class ControlPadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        initialize_taichi_cuda()

    def test_light_diagnostics_equal_existing_results(self) -> None:
        simulator = ControlPadTaichiSimulator(
            SimulatorConfig(
                program_width=8,
                left_budget=10,
                right_budget=12,
                t_max=2_000,
            ),
            batch_capacity=3,
        )
        programs = simulator.encode_programs(["", "00100", "10101"])
        task = TaskCases(["0", "1", "101"], ["0", "1", "101"])
        baseline = evaluate_population_fitness_taichi(
            simulator, programs, task, k_penalty=0.5
        )
        diagnostic = evaluate_population_with_diagnostics(
            simulator, programs, task, k_penalty=0.5
        )
        np.testing.assert_array_equal(
            diagnostic.fitness.fitness, baseline.fitness
        )
        np.testing.assert_array_equal(
            diagnostic.fitness.bit_accuracy, baseline.bit_accuracy
        )
        np.testing.assert_array_equal(
            diagnostic.fitness.invalid_fraction,
            baseline.invalid_fraction,
        )
        np.testing.assert_array_equal(
            diagnostic.fitness.exact, baseline.exact
        )

        light = simulator.simulate_program_batch_for_control_pad(
            programs, "101"
        )
        complete = simulator.simulate_program_batch(programs, "101")
        np.testing.assert_array_equal(light.output_bits, complete.output_bits)
        np.testing.assert_array_equal(
            light.output_lengths, complete.output_lengths
        )
        np.testing.assert_array_equal(light.invalid, complete.invalid)
        np.testing.assert_array_equal(light.T, complete.T)
        np.testing.assert_array_equal(
            light.left_space_used, complete.left_space_used
        )
        np.testing.assert_array_equal(
            light.right_space_used, complete.right_space_used
        )
        np.testing.assert_array_equal(light.halted, complete.halted)

    def test_atomic_checkpoint_resumes_exact_rng_trajectory(self) -> None:
        payload = {
            "run_id": "checkpoint-test",
            "train_inputs": ["0", "1"],
            "train_targets": ["0", "1"],
            "test_inputs": [],
            "test_targets": [],
            "simulator": {
                "program_width": 8,
                "left_budget": 10,
                "right_budget": 8,
                "t_max": 300,
            },
            "structural": {
                "islands": 2,
                "population_per_island": 8,
                "seed": 91,
                "k_penalty": 0.5,
            },
            "mutable": {
                "generation_target": 3,
                "elite_fraction": 0.125,
                "cross_island_fraction": 0.0,
                "cross_island_interval": 2,
                "random_immigrant_fraction": 0.125,
                "tournament_delta_thresholds": [0.05, 0.20],
                "tournament_best_probabilities": [0.52, 0.60],
                "tournament_base_probability": 0.90,
                "p_crossover": 0.5,
                "crossover_segments": 2,
                "crossover_segment_min": 1,
                "crossover_segment_max": 3,
                "p_insert": 0.04,
                "p_delete": 0.06,
                "p_bit_flip": 0.02,
                "stop_first_exact": False,
                "diagnostic_interval": 2,
                "checkpoint_interval": 1,
            },
        }
        definition, settings = parse_new_run_payload(payload)
        uninterrupted = LiveTrainingSession.create(definition, settings)
        uninterrupted.step()
        self.assertEqual(len(uninterrupted.diagnostic_records), 1)
        json.dumps(
            uninterrupted.history_payload(0, [0, 1]),
            allow_nan=False,
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = uninterrupted.save_checkpoint(
                root, reason="test"
            )
            self.assertTrue(checkpoint.is_file())
            resumed = LiveTrainingSession.load_checkpoint(
                root, definition.run_id
            )
            self.assertEqual(resumed.generation, 1)
            self.assertEqual(resumed.definition.to_json(), definition.to_json())
            self.assertEqual(resumed.settings.to_json(), settings.to_json())
            np.testing.assert_array_equal(
                resumed.engine.programs, uninterrupted.engine.programs
            )
            np.testing.assert_array_equal(
                resumed.engine.lengths, uninterrupted.engine.lengths
            )
            np.testing.assert_array_equal(
                resumed.engine.is_child, uninterrupted.engine.is_child
            )

            uninterrupted.step()
            resumed.step()
            self.assertEqual(len(uninterrupted.diagnostic_records), 2)
            self.assertEqual(len(resumed.diagnostic_records), 2)
            np.testing.assert_array_equal(
                resumed.engine.programs, uninterrupted.engine.programs
            )
            np.testing.assert_array_equal(
                resumed.engine.lengths, uninterrupted.engine.lengths
            )
            np.testing.assert_array_equal(
                resumed.engine.is_child, uninterrupted.engine.is_child
            )
            np.testing.assert_array_equal(
                resumed.engine.fitness.fitness,
                uninterrupted.engine.fitness.fitness,
            )

    def test_new_run_requires_stopped_state_and_clears_session(self) -> None:
        with TemporaryDirectory() as temporary:
            controller = ControlPadController(Path(temporary))
            controller.status = "paused"
            with self.assertRaises(RuntimeError):
                controller.request_new_run()

            controller.status = "stopped"
            controller.session = object()  # type: ignore[assignment]
            controller.snapshot = {"run_id": "old"}
            controller.request_new_run()

            self.assertEqual(controller.status, "idle")
            self.assertIsNone(controller.session)
            self.assertIsNone(controller.snapshot)


if __name__ == "__main__":
    unittest.main()
