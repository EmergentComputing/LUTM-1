"""Standalone NumPy/Taichi differential and long arithmetic tests."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numba
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lutm import InvalidReason, ScalarUTMSimulator, TransitionTable
from numpy_backend import NumpyUTMSimulator, compile_dense_table
from programs import PROGRAMS
from taichi_backend import (
    TaichiUTMSimulator,
    evaluate_program_taichi,
    find_exact_program_taichi,
    initialize_taichi_cuda,
)
from utils import (
    SimulatorConfig,
    TaskCases,
    encode_padded_programs,
    ordinal_to_program,
    program_count,
)


def assert_batch_equal(test: unittest.TestCase, first, second) -> None:
    for field in (
        "output_bits",
        "output_lengths",
        "invalid",
        "invalid_reason",
        "T",
        "left_space_used",
        "right_space_used",
        "halted",
        "final_state_ids",
        "final_heads",
        "final_tape",
        "effective_program_lengths",
    ):
        np.testing.assert_array_equal(
            getattr(first, field),
            getattr(second, field),
            err_msg=field,
        )
    test.assertEqual(first.output_strings(), second.output_strings())
    test.assertEqual(first.invalid_reason_names(), second.invalid_reason_names())


@numba.njit
def _run_full_physical(
    program,
    input_symbols,
    valid_lut,
    write_lut,
    move_lut,
    next_lut,
    start_id,
    halt_id,
    blank_id,
    separator_id,
    left_budget,
    right_budget,
    t_max,
):
    origin = left_budget
    tape = np.full(left_budget + right_budget + 1, blank_id, dtype=np.uint8)
    tape[origin - program.size : origin] = program
    tape[origin] = separator_id
    tape[origin + 1 : origin + 1 + input_symbols.size] = input_symbols
    state = start_id
    head = 0
    steps = 0
    valid = True
    while state != halt_id and steps < t_max:
        tape_index = origin + head
        if tape_index < 0 or tape_index >= tape.size:
            valid = False
            break
        read = tape[tape_index]
        if not valid_lut[state, read]:
            valid = False
            break
        tape[tape_index] = write_lut[state, read]
        head += move_lut[state, read]
        state = next_lut[state, read]
        steps += 1
    return valid, state == halt_id, steps, state, head, tape


class BackendDifferentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        initialize_taichi_cuda()
        cls.table = TransitionTable()

    def encode(self, programs: list[str], width: int) -> np.ndarray:
        return encode_padded_programs(
            programs,
            width,
            blank_id=self.table.blank_id,
            zero_id=self.table.zero_id,
            one_id=self.table.one_id,
        )

    def test_all_programs_through_length_six_match_exactly(self) -> None:
        width = 6
        config = SimulatorConfig(width, 260, 260, 250)
        numpy_simulator = NumpyUTMSimulator(config, self.table)
        scalar_simulator = ScalarUTMSimulator(self.table)
        taichi_simulator = TaichiUTMSimulator(
            config, batch_capacity=127, table=self.table
        )
        programs = [ordinal_to_program(index, width) for index in range(127)]
        padded = self.encode(programs, width)
        for input_bits in ("", "0", "001", "1011"):
            expected = numpy_simulator.simulate(padded, input_bits)
            actual = taichi_simulator.simulate_program_batch(
                padded, input_bits
            )
            assert_batch_equal(self, actual, expected)
            for result in (expected, actual):
                self.assertEqual(result.T.dtype, np.dtype(np.uint32))
                self.assertEqual(result.final_heads.dtype, np.dtype(np.int32))
                self.assertEqual(
                    result.left_space_used.dtype, np.dtype(np.int32)
                )
                self.assertEqual(
                    result.right_space_used.dtype, np.dtype(np.int32)
                )
            for row, program in enumerate(programs):
                scalar = scalar_simulator.run(
                    program, input_bits, t_max=config.t_max
                )
                self.assertEqual(expected.output_strings()[row], scalar.output)
                self.assertEqual(bool(expected.invalid[row]), scalar.invalid)
                self.assertEqual(
                    int(expected.invalid_reason[row]), int(scalar.invalid_reason)
                )
                self.assertEqual(int(expected.T[row]), scalar.T)
                self.assertEqual(
                    int(expected.left_space_used[row]), scalar.left_space_used
                )
                self.assertEqual(
                    int(expected.right_space_used[row]), scalar.right_space_used
                )
                self.assertEqual(bool(expected.halted[row]), scalar.halted)
                self.assertEqual(
                    self.table.states[int(expected.final_state_ids[row])],
                    scalar.final_state,
                )
                self.assertEqual(
                    int(expected.final_heads[row]), scalar.final_head
                )
                scalar_tape = np.full(
                    config.tape_size, self.table.blank_id, dtype=np.uint8
                )
                for coordinate, symbol in scalar.final_tape.items():
                    scalar_tape[config.origin_index + coordinate] = (
                        self.table.symbol_ids[symbol]
                    )
                np.testing.assert_array_equal(
                    expected.final_tape[row], scalar_tape
                )

    def test_known_wide_programs_match_on_diverse_inputs(self) -> None:
        for name, values in (
            ("bit_not", ["0", "1", "001", "10110"]),
            (
                "plus_one",
                [format(value, "b") for value in (0, 1, 2, 3, 15, 16, 255, 1023)],
            ),
            (
                "times_two",
                [format(value, "b") for value in (0, 1, 2, 3, 15, 16, 255, 1023)],
            ),
        ):
            registered = PROGRAMS[name]
            width = len(registered.program)
            config = SimulatorConfig(width, width + 2, 40, 1_000_000)
            numpy_simulator = NumpyUTMSimulator(config, self.table)
            taichi_simulator = TaichiUTMSimulator(
                config, batch_capacity=1, table=self.table
            )
            padded = self.encode([registered.program], width)
            for input_bits in values:
                expected = numpy_simulator.simulate(padded, input_bits)
                actual = taichi_simulator.simulate_program_batch(
                    padded, input_bits
                )
                assert_batch_equal(self, actual, expected)
                self.assertFalse(bool(actual.invalid[0]))
                self.assertEqual(
                    actual.output_strings()[0], registered.target(input_bits)
                )

    def test_minimal_fitness_copy_matches_complete_result(self) -> None:
        config = SimulatorConfig(8, 10, 12, 250)
        simulator = TaichiUTMSimulator(config, 4, self.table)
        programs = self.encode(["", "0", "00100", "11111111"], 8)
        complete = simulator.simulate_program_batch(programs, "101")
        minimal = simulator.simulate_program_batch_for_fitness(programs, "101")
        for field in (
            "output_bits",
            "output_lengths",
            "invalid",
            "invalid_reason",
        ):
            np.testing.assert_array_equal(
                getattr(minimal, field), getattr(complete, field)
            )

    def test_exact_timeout_and_right_boundary_semantics_match(self) -> None:
        identity = PROGRAMS["identity_short"].program
        endpoint_config = SimulatorConfig(5, 5, 8, 100)
        endpoint_program = self.encode([identity], 5)
        endpoint_numpy = NumpyUTMSimulator(
            endpoint_config, self.table
        ).simulate(endpoint_program, "101")
        endpoint_taichi = TaichiUTMSimulator(
            endpoint_config, 1, self.table
        ).simulate_program_batch(endpoint_program, "101")
        assert_batch_equal(self, endpoint_taichi, endpoint_numpy)
        self.assertEqual(
            int(endpoint_taichi.invalid_reason[0]),
            int(InvalidReason.LEFT_SPACE_EXCEEDED),
        )
        self.assertEqual(int(endpoint_taichi.T[0]), 6)
        self.assertEqual(int(endpoint_taichi.final_heads[0]), -6)

        for t_max, expected_reason in (
            (24, InvalidReason.TIMEOUT),
            (25, InvalidReason.NONE),
        ):
            config = SimulatorConfig(5, 6, 8, t_max)
            padded = self.encode([identity], 5)
            numpy_result = NumpyUTMSimulator(config, self.table).simulate(
                padded, "101110"
            )
            taichi_result = TaichiUTMSimulator(
                config, 1, self.table
            ).simulate_program_batch(padded, "101110")
            assert_batch_equal(self, taichi_result, numpy_result)
            self.assertEqual(
                int(taichi_result.invalid_reason[0]), int(expected_reason)
            )
            self.assertEqual(int(taichi_result.T[0]), t_max)

        bit_not = PROGRAMS["bit_not"].program
        config = SimulatorConfig(80, 81, 1, 10_000)
        padded = self.encode([bit_not], 80)
        numpy_result = NumpyUTMSimulator(config, self.table).simulate(padded, "1")
        taichi_result = TaichiUTMSimulator(
            config, 1, self.table
        ).simulate_program_batch(padded, "1")
        assert_batch_equal(self, taichi_result, numpy_result)
        self.assertEqual(
            int(taichi_result.invalid_reason[0]),
            int(InvalidReason.RIGHT_SPACE_EXCEEDED),
        )
        self.assertEqual(int(taichi_result.final_heads[0]), 2)
        self.assertEqual(int(taichi_result.right_space_used[0]), 2)

    def test_explicit_width_above_ordinal_limit(self) -> None:
        config = SimulatorConfig(80, 82, 12, 20_000)
        program = PROGRAMS["bit_not"].program
        padded = self.encode([program], 80)
        numpy_simulator = NumpyUTMSimulator(config, self.table)
        numpy_result = numpy_simulator.simulate(padded, "10110")
        simulator = TaichiUTMSimulator(config, 1, self.table)
        taichi_result = simulator.simulate_program_batch(padded, "10110")
        assert_batch_equal(self, taichi_result, numpy_result)
        with self.assertRaises(ValueError):
            simulator.simulate_ordinal_batch(0, 1, "0")

        wrapped_unknown = np.full(
            (1, config.program_width), self.table.blank_id, dtype=np.int64
        )
        wrapped_unknown[0, -1] = 256 + self.table.blank_id
        with self.assertRaisesRegex(ValueError, "unknown symbol"):
            numpy_simulator.simulate(wrapped_unknown, "0")
        with self.assertRaisesRegex(ValueError, "unknown symbol"):
            simulator.simulate_program_batch(wrapped_unknown, "0")

    def test_uint64_ordinal_extremes_match_numpy(self) -> None:
        width = 62
        total = program_count(width)
        ordinals = (0, total - 2, total - 1)
        programs = [ordinal_to_program(value, width) for value in ordinals]
        padded = self.encode(programs, width)
        config = SimulatorConfig(width, 64, 12, 80)
        numpy_simulator = NumpyUTMSimulator(config, self.table)
        simulator = TaichiUTMSimulator(config, 2, self.table)
        for start, count, rows in ((0, 1, padded[:1]), (total - 2, 2, padded[1:])):
            expected = numpy_simulator.simulate(rows, "0")
            actual = simulator.simulate_ordinal_batch(start, count, "0")
            assert_batch_equal(self, actual, expected)

    def test_task_evaluator_and_first_exact_search(self) -> None:
        task = TaskCases(["0", "1", "101"], ["0", "1", "101"])
        config = SimulatorConfig(5, 6, 8, 200)
        simulator = TaichiUTMSimulator(config, 32, self.table)
        evaluations = evaluate_program_taichi(
            simulator, PROGRAMS["identity_short"].program, task
        )
        self.assertTrue(all(case.exact for case in evaluations))
        result = find_exact_program_taichi(
            simulator,
            task,
            batch_size=32,
            stop_first_exact=True,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.program, "00000")
        self.assertEqual(result.program_ordinal, 31)


class SquarePhysicalSeriesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.table = TransitionTable()
        cls.dense = compile_dense_table(cls.table)
        cls.program = encode_padded_programs(
            [PROGRAMS["square"].program],
            len(PROGRAMS["square"].program),
            blank_id=cls.table.blank_id,
            zero_id=cls.table.zero_id,
            one_id=cls.table.one_id,
        )[0]

    def test_square_physical_series_with_every_transition_counted(self) -> None:
        known = {
            0: ("0", 211_287),
            1: ("1", 100_360_363),
            2: ("100", 232_065_331),
            3: ("1001", 470_246_995),
            5: ("11001", 1_731_738_891),
        }
        for value, (expected_output, expected_steps) in known.items():
            input_bits = format(value, "b")
            input_symbols = np.fromiter(
                (
                    self.table.one_id if bit == "1" else self.table.zero_id
                    for bit in input_bits
                ),
                dtype=np.uint8,
            )
            left_budget = len(self.program) + 2
            right_budget = value * value + value + 2 * len(input_bits) + 32
            valid, halted, steps, state, _head, tape = _run_full_physical(
                self.program,
                input_symbols,
                self.dense.valid,
                self.dense.write,
                self.dense.move,
                self.dense.next_state,
                self.table.start_id,
                self.table.halt_id,
                self.table.blank_id,
                self.table.separator_id,
                left_budget,
                right_budget,
                2_000_000_000,
            )
            self.assertTrue(valid)
            self.assertTrue(halted)
            self.assertEqual(int(state), self.table.halt_id)
            self.assertEqual(int(steps), expected_steps)
            origin = left_budget
            self.assertEqual(int(tape[origin]), self.table.separator_id)
            output = []
            position = origin + 1
            while tape[position] in (self.table.zero_id, self.table.one_id):
                output.append(
                    "0" if tape[position] == self.table.zero_id else "1"
                )
                position += 1
            self.assertEqual("".join(output), expected_output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
