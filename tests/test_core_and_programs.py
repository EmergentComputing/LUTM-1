from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lutm  # noqa: E402
from lutm import (  # noqa: E402
    BLANK,
    HALT,
    SEPARATOR,
    START,
    SYMBOLS,
    UINT64_MAX,
    InvalidReason,
    ScalarUTMSimulator,
    Transition,
    TransitionTable,
)
from programs import PROGRAMS, get_program  # noqa: E402
from utils import (  # noqa: E402
    INT32_MAX,
    UINT32_MAX,
    SimulatorConfig,
    TaskCases,
    effective_program_lengths,
    effective_program_to_string,
    encode_padded_programs,
    enumerate_program_batches,
    iter_binary_inputs,
    ordinal_to_program,
    padded_program_to_string,
    program_count,
    program_ordinal,
    validate_binary,
)


class TransitionTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.table = TransitionTable()
        cls.csv_path = ROOT / "data" / "transition_table.csv"

    def test_canonical_hash_and_total_transition_relation(self) -> None:
        digest = hashlib.sha256(self.csv_path.read_bytes()).hexdigest()
        self.assertEqual(digest, lutm.CANONICAL_CSV_SHA256)
        self.assertEqual(len(self.table.states), 150)
        self.assertEqual(self.table.symbols, SYMBOLS)
        self.assertEqual(len(self.table.transitions), 149 * len(SYMBOLS))
        self.assertIn(START, self.table.states)
        self.assertIn(HALT, self.table.states)

        for state in self.table.states:
            for symbol in SYMBOLS:
                key = (state, symbol)
                if state == HALT:
                    self.assertNotIn(key, self.table.transitions)
                    continue
                self.assertIn(key, self.table.transitions)
                transition = self.table.transitions[key]
                self.assertIn(transition.write, SYMBOLS)
                self.assertIn(transition.move, (-1, 0, 1))
                self.assertIn(transition.next_state, self.table.states)

    def test_ids_and_transition_mappings_are_consistent_and_read_only(self) -> None:
        self.assertEqual(self.table.states[self.table.start_id], START)
        self.assertEqual(self.table.states[self.table.halt_id], HALT)
        for symbol_id, symbol in enumerate(SYMBOLS):
            self.assertEqual(self.table.symbol_ids[symbol], symbol_id)

        with self.assertRaises(TypeError):
            self.table.state_ids["new"] = 999  # type: ignore[index]
        with self.assertRaises(TypeError):
            self.table.transitions[(HALT, BLANK)] = Transition(  # type: ignore[index]
                BLANK, 0, HALT
            )

    def _load_modified_rows(
        self,
        rows: list[list[str]],
        fieldnames: list[str] | None = None,
    ) -> TransitionTable:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "table.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    fieldnames
                    or ["state", "read", "write", "move", "next_state"]
                )
                writer.writerows(rows)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(lutm, "CANONICAL_CSV_SHA256", digest):
                return TransitionTable(path)

    def _canonical_rows(self) -> list[list[str]]:
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.reader(handle))[1:]

    def test_missing_file_and_hash_mismatch_are_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            TransitionTable(ROOT / "data" / "does_not_exist.csv")
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "table.csv"
            path.write_text(self.csv_path.read_text(encoding="utf-8") + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                TransitionTable(path)

    def test_bad_schema_duplicate_missing_and_unknown_symbol_are_rejected(self) -> None:
        rows = self._canonical_rows()
        malformed = (
            (
                rows,
                ["state", "read", "write", "next_state", "move"],
                "unexpected transition-table columns",
            ),
            (rows + [rows[0]], None, "duplicate transition"),
            (rows[:-1], None, "expected 2384 transitions"),
            (
                [[row[0], "?", row[2], row[3], row[4]] if index == 0 else row
                 for index, row in enumerate(rows)],
                None,
                "unknown symbol",
            ),
        )
        for modified_rows, fieldnames, message in malformed:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._load_modified_rows(modified_rows, fieldnames)


class ScalarContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.simulator = ScalarUTMSimulator()

    def test_initial_tape_is_exactly_raw_program_separator_input(self) -> None:
        program = "00100"
        input_bits = "101110"
        tape = self.simulator.initial_tape(program, input_bits)
        self.assertEqual(
            "".join(tape[position] for position in range(-5, 0)),
            program,
        )
        self.assertEqual(tape[0], SEPARATOR)
        self.assertEqual(
            "".join(tape[position] for position in range(1, 7)),
            input_bits,
        )
        self.assertEqual(set(tape), set(range(-5, 7)))

    def test_valid_result_does_not_require_program_restoration(self) -> None:
        registered = PROGRAMS["plus_one"]
        initial = self.simulator.initial_tape(registered.program, "0")
        result = self.simulator.run(registered.program, "0", t_max=20_000)
        self.assertTrue(result.halted)
        self.assertFalse(result.invalid)
        self.assertEqual(result.output, "1")
        self.assertEqual(result.final_tape[0], SEPARATOR)
        self.assertTrue(
            any(
                result.final_tape.get(position, BLANK)
                != initial.get(position, BLANK)
                for position in range(-len(registered.program), 0)
            ),
            "this computation deliberately leaves target-state garbage left of #",
        )

    def test_decoder_stops_at_scratch_and_ignores_final_head(self) -> None:
        table = SimpleNamespace(
            transitions={
                (START, SEPARATOR): Transition(SEPARATOR, 1, "S1"),
                ("S1", "1"): Transition("1", 1, "S2"),
                ("S2", "0"): Transition("0", 1, "S3"),
                ("S3", BLANK): Transition("X", 1, HALT),
            }
        )
        result = ScalarUTMSimulator(table).run("", "10", t_max=4)  # type: ignore[arg-type]
        self.assertTrue(result.halted)
        self.assertFalse(result.invalid)
        self.assertEqual(result.output, "10")
        self.assertEqual(result.final_tape[3], "X")
        self.assertEqual(result.final_head, 4)

    def test_empty_output_missing_separator_and_invalid_transition_are_explicit(self) -> None:
        empty = self.simulator.run("00100", "", t_max=25)
        self.assertTrue(empty.halted)
        self.assertTrue(empty.invalid)
        self.assertEqual(empty.invalid_reason, InvalidReason.EMPTY_OUTPUT)
        self.assertEqual(empty.output, "")

        erasing_table = SimpleNamespace(
            transitions={
                (START, SEPARATOR): Transition(BLANK, 1, HALT),
            }
        )
        missing = ScalarUTMSimulator(erasing_table).run("", "1", t_max=1)  # type: ignore[arg-type]
        self.assertTrue(missing.halted)
        self.assertTrue(missing.invalid)
        self.assertEqual(missing.invalid_reason, InvalidReason.MISSING_SEPARATOR)

        no_rules = SimpleNamespace(transitions={})
        invalid = ScalarUTMSimulator(no_rules).run("", "1", t_max=1)  # type: ignore[arg-type]
        self.assertFalse(invalid.halted)
        self.assertEqual(invalid.T, 0)
        self.assertEqual(invalid.invalid_reason, InvalidReason.INVALID_TRANSITION)

    def test_timeout_boundary_is_exact(self) -> None:
        timed_out = self.simulator.run("00100", "101", t_max=24)
        exact = self.simulator.run("00100", "101", t_max=25)
        extra = self.simulator.run("00100", "101", t_max=26)

        self.assertFalse(timed_out.halted)
        self.assertEqual(timed_out.T, 24)
        self.assertEqual(timed_out.invalid_reason, InvalidReason.TIMEOUT)
        for result in (exact, extra):
            self.assertTrue(result.halted)
            self.assertFalse(result.invalid)
            self.assertEqual(result.T, 25)
            self.assertEqual(result.output, "101")
        self.assertEqual(dict(exact.final_tape), dict(extra.final_tape))

    def test_divergent_program_and_uint64_timeout_contract(self) -> None:
        loop = PROGRAMS["divergent_loop"]
        result = self.simulator.run(loop.program, "101", t_max=257)
        self.assertFalse(result.halted)
        self.assertEqual(result.T, 257)
        self.assertIs(type(result.T), int)
        self.assertEqual(result.invalid_reason, InvalidReason.TIMEOUT)

        quick = self.simulator.run("00100", "1", t_max=UINT64_MAX)
        self.assertEqual(quick.T, 25)
        with self.assertRaises(ValueError):
            self.simulator.run("00100", "1", t_max=UINT64_MAX + 1)
        with self.assertRaises(TypeError):
            self.simulator.run("00100", "1", t_max=True)

    def test_self_extending_tape_and_space_measurement(self) -> None:
        result = self.simulator.run(
            PROGRAMS["plus_one"].program,
            "1111111111111111",
            t_max=200_000,
        )
        self.assertFalse(result.invalid)
        self.assertEqual(result.output, "1" + "0" * 16)
        self.assertGreaterEqual(result.right_space_used, 17)
        self.assertEqual(
            result.span_used,
            result.left_space_used + result.right_space_used + 1,
        )
        self.assertEqual(
            result.max_space_used,
            max(result.left_space_used, result.right_space_used),
        )


class UtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.table = TransitionTable()

    def test_binary_and_task_validation(self) -> None:
        validate_binary("input", "")
        validate_binary("input", "0101", allow_empty=False)
        with self.assertRaises(ValueError):
            validate_binary("input", "", allow_empty=False)
        with self.assertRaises(ValueError):
            validate_binary("input", "012")
        with self.assertRaises(TypeError):
            validate_binary("input", b"01")  # type: ignore[arg-type]

        task = TaskCases(["", "0", "101"], ["0", "1", "11"])
        self.assertEqual(task.pairs(), (("", "0"), ("0", "1"), ("101", "11")))
        task.validate_right_budget(3)
        with self.assertRaises(ValueError):
            task.validate_right_budget(2)

        invalid_tasks = (
            ([], []),
            (["0"], []),
            (["0", "0"], ["1", "1"]),
            (["x"], ["1"]),
            (["0"], [""]),
        )
        for inputs, targets in invalid_tasks:
            with self.subTest(inputs=inputs, targets=targets):
                with self.assertRaises(ValueError):
                    TaskCases(inputs, targets)

    def test_simulator_config_bounds_and_exact_tape_size(self) -> None:
        config = SimulatorConfig(64, 70, 29, 8_000)
        self.assertEqual(config.tape_size, 100)
        self.assertEqual(config.origin_index, 70)
        endpoint_config = SimulatorConfig(64, 64, 29, 8_000)
        self.assertEqual(endpoint_config.origin_index, 64)
        self.assertEqual(endpoint_config.tape_size, 94)
        self.assertEqual(SimulatorConfig(1, 2, 1, UINT32_MAX).t_max, UINT32_MAX)
        with self.assertRaises(ValueError):
            SimulatorConfig(64, 63, 29, 8_000)
        with self.assertRaises(ValueError):
            SimulatorConfig(1, 2, 1, UINT32_MAX + 1)
        with self.assertRaises(ValueError):
            SimulatorConfig(1, INT32_MAX, 1, 1)
        with self.assertRaises(TypeError):
            SimulatorConfig(True, 2, 1, 1)  # type: ignore[arg-type]

    def test_binary_input_iteration(self) -> None:
        self.assertEqual(
            list(iter_binary_inputs(0, 2)),
            ["", "0", "1", "00", "01", "10", "11"],
        )
        with self.assertRaises(ValueError):
            list(iter_binary_inputs(2, 1))
        with self.assertRaises(TypeError):
            list(iter_binary_inputs(0, 2.0))  # type: ignore[arg-type]

    def test_program_ordinal_order_and_round_trip(self) -> None:
        expected = [
            "",
            "0",
            "1",
            "00",
            "01",
            "10",
            "11",
            "000",
            "001",
            "010",
            "011",
            "100",
            "101",
            "110",
            "111",
        ]
        self.assertEqual(program_count(3), len(expected))
        self.assertEqual(
            [ordinal_to_program(ordinal, 3) for ordinal in range(len(expected))],
            expected,
        )
        for ordinal, program in enumerate(expected):
            self.assertEqual(program_ordinal(program, 3), ordinal)
        self.assertNotEqual(program_ordinal("0", 3), program_ordinal("00", 3))
        with self.assertRaises(ValueError):
            program_count(63)
        with self.assertRaises(ValueError):
            ordinal_to_program(15, 3)

    def test_padding_lengths_strings_and_batches(self) -> None:
        blank = self.table.blank_id
        zero = self.table.zero_id
        one = self.table.one_id
        programs = ["", "0", "1", "00", "101"]
        encoded = encode_padded_programs(
            programs,
            3,
            blank_id=blank,
            zero_id=zero,
            one_id=one,
        )
        self.assertEqual(encoded.dtype, np.dtype(np.uint8))
        self.assertEqual(
            [
                padded_program_to_string(
                    row, blank_id=blank, zero_id=zero, one_id=one
                )
                for row in encoded
            ],
            ["BBB", "BB0", "BB1", "B00", "101"],
        )
        self.assertEqual(
            effective_program_lengths(
                encoded, blank_id=blank, zero_id=zero, one_id=one
            ).tolist(),
            [0, 1, 1, 2, 3],
        )
        self.assertEqual(
            effective_program_to_string(
                encoded[-1], blank_id=blank, zero_id=zero, one_id=one
            ),
            "101",
        )

        bad = np.asarray([[zero, blank, one]], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "blank-left-padded"):
            effective_program_lengths(
                bad, blank_id=blank, zero_id=zero, one_id=one
            )

        batches = list(
            enumerate_program_batches(
                3,
                4,
                blank_id=blank,
                zero_id=zero,
                one_id=one,
            )
        )
        self.assertEqual([batch.start_ordinal for batch in batches], [0, 4, 8, 12])
        self.assertEqual([batch.end_ordinal for batch in batches], [4, 8, 12, 15])
        self.assertEqual(
            np.concatenate([batch.effective_lengths for batch in batches]).tolist(),
            [len(program) for program in [
                "", "0", "1", "00", "01", "10", "11", "000", "001",
                "010", "011", "100", "101", "110", "111",
            ]],
        )


class ProgramRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.simulator = ScalarUTMSimulator()

    def assert_computes(
        self,
        name: str,
        inputs: list[str],
        *,
        t_max: int,
    ) -> None:
        registered = PROGRAMS[name]
        for input_bits in inputs:
            with self.subTest(program=name, input=input_bits):
                self.assertTrue(registered.accepts(input_bits))
                expected = registered.target(input_bits)
                result = self.simulator.run(
                    registered.program,
                    input_bits,
                    t_max=t_max,
                )
                self.assertTrue(result.halted)
                self.assertFalse(result.invalid, result.invalid_reason.name)
                self.assertEqual(result.invalid_reason, InvalidReason.NONE)
                self.assertEqual(result.output, expected)
                self.assertEqual(result.final_tape.get(0, BLANK), SEPARATOR)

    def test_registry_names_lengths_hashes_and_lookup(self) -> None:
        expected_lengths = {
            "identity_short": 5,
            "identity_complete": 23,
            "bit_not": 80,
            "jump_probe": 76,
            "jump_to_zero_probe": 76,
            "plus_one": 176,
            "times_two": 114,
            "square": 4_605,
            "divergent_loop": 24,
        }
        self.assertEqual(set(PROGRAMS), set(expected_lengths))
        for name, expected_length in expected_lengths.items():
            with self.subTest(program=name):
                registered = PROGRAMS[name]
                self.assertIs(get_program(name), registered)
                self.assertEqual(len(registered.program), expected_length)
                self.assertLessEqual(set(registered.program), {"0", "1"})
                self.assertEqual(
                    hashlib.sha256(registered.program.encode("ascii")).hexdigest(),
                    registered.logical_sha256,
                )
        with self.assertRaises(KeyError):
            get_program("does_not_exist")

    def test_oracle_contracts_and_length_independence(self) -> None:
        arithmetic = ("plus_one", "times_two", "square")
        for name in arithmetic:
            for invalid in ("", "00", "01", "0101", "x"):
                with self.subTest(program=name, invalid=invalid):
                    self.assertFalse(PROGRAMS[name].accepts(invalid))
                    with self.assertRaises(ValueError):
                        PROGRAMS[name].target(invalid)

        ones = "1" * 512
        self.assertEqual(PROGRAMS["plus_one"].target(ones), "1" + "0" * 512)
        long_value = "1" + "01" * 200
        self.assertEqual(PROGRAMS["times_two"].target(long_value), long_value + "0")
        square_input = "1" + "00101101" * 31
        self.assertEqual(
            PROGRAMS["square"].target(square_input),
            format(int(square_input, 2) ** 2, "b"),
        )

        loop = PROGRAMS["divergent_loop"]
        self.assertTrue(loop.accepts(""))
        with self.assertRaisesRegex(ValueError, "no oracle"):
            loop.target("")

    def test_short_and_complete_identity_exhaustively(self) -> None:
        short_inputs = list(iter_binary_inputs(1, 8))
        complete_inputs = list(iter_binary_inputs(1, 5))
        self.assert_computes("identity_short", short_inputs, t_max=25)
        self.assert_computes("identity_complete", complete_inputs, t_max=200)

        for input_bits in ("0", "101", "000101"):
            result = self.simulator.run("00100", input_bits, t_max=25)
            self.assertEqual(result.T, 25)

    def test_bit_not_exhaustively_through_seven_bits(self) -> None:
        self.assert_computes(
            "bit_not",
            list(iter_binary_inputs(1, 7)),
            t_max=25_000,
        )

    def test_jump_witnesses(self) -> None:
        inputs = ["0", "1", "001011", "101101"]
        self.assert_computes("jump_probe", inputs, t_max=5_000)
        self.assert_computes("jump_to_zero_probe", inputs, t_max=5_000)

    def test_plus_one_for_first_256_integers(self) -> None:
        inputs = [format(value, "b") for value in range(256)]
        self.assert_computes("plus_one", inputs, t_max=100_000)

    def test_times_two_for_first_256_integers(self) -> None:
        inputs = [format(value, "b") for value in range(256)]
        self.assert_computes("times_two", inputs, t_max=50_000)

    def test_square_zero_full_physical_scalar_execution(self) -> None:
        result = self.simulator.run(
            PROGRAMS["square"].program,
            "0",
            t_max=300_000,
        )
        self.assertTrue(result.halted)
        self.assertFalse(result.invalid)
        self.assertEqual(result.output, "0")
        self.assertEqual(result.T, 211_287)


if __name__ == "__main__":
    unittest.main(verbosity=2)
