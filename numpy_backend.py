"""Blank-padded NumPy LUTM simulator, evaluator, and enumerator."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np

from lutm import InvalidReason, TransitionTable
from utils import (
    SimulatorConfig,
    TaskCases,
    effective_program_lengths,
    effective_program_to_string,
    encode_padded_programs,
    enumerate_program_batches,
    padded_program_to_string,
    validate_binary,
)


OUTPUT_PAD = np.uint8(2)


@dataclass(frozen=True)
class DenseTransitionArrays:
    valid: np.ndarray
    write: np.ndarray
    move: np.ndarray
    next_state: np.ndarray


def compile_dense_table(table: TransitionTable) -> DenseTransitionArrays:
    shape = (len(table.states), len(table.symbols))
    valid = np.zeros(shape, dtype=np.bool_)
    write = np.full(shape, 255, dtype=np.uint8)
    move = np.full(shape, 127, dtype=np.int8)
    next_state = np.full(shape, 65535, dtype=np.uint16)
    for (state, read), transition in table.transitions.items():
        state_id = table.state_ids[state]
        read_id = table.symbol_ids[read]
        valid[state_id, read_id] = True
        write[state_id, read_id] = table.symbol_ids[transition.write]
        move[state_id, read_id] = transition.move
        next_state[state_id, read_id] = table.state_ids[transition.next_state]
    for array in (valid, write, move, next_state):
        array.setflags(write=False)
    return DenseTransitionArrays(valid, write, move, next_state)


@dataclass(frozen=True, eq=False)
class BatchResult:
    output_bits: np.ndarray
    output_lengths: np.ndarray
    invalid: np.ndarray
    invalid_reason: np.ndarray
    T: np.ndarray
    left_space_used: np.ndarray
    right_space_used: np.ndarray
    halted: np.ndarray
    final_state_ids: np.ndarray
    final_heads: np.ndarray
    final_tape: np.ndarray
    effective_program_lengths: np.ndarray

    @property
    def span_used(self) -> np.ndarray:
        return self.left_space_used + self.right_space_used + 1

    @property
    def max_space_used(self) -> np.ndarray:
        return np.maximum(self.left_space_used, self.right_space_used)

    def output_strings(self) -> list[str]:
        return [
            "".join(str(int(bit)) for bit in row[: int(length)])
            for row, length in zip(self.output_bits, self.output_lengths)
        ]

    def invalid_reason_names(self) -> list[str]:
        return [
            InvalidReason(int(reason)).name for reason in self.invalid_reason
        ]


@dataclass(frozen=True)
class CaseEvaluation:
    input_bits: str
    target: str
    output: str
    exact: bool
    invalid: bool
    invalid_reason: str
    T: int
    left_space_used: int
    right_space_used: int


@dataclass(frozen=True)
class EnumerationResult:
    program: str
    padded_program: str
    program_ordinal: int
    effective_length: int
    evaluated_programs: int
    elapsed_seconds: float


class NumpyUTMSimulator:
    """Simulate one input across a batch of blank-left-padded programs."""

    def __init__(
        self,
        config: SimulatorConfig,
        table: TransitionTable | None = None,
    ) -> None:
        self.config = config
        self.table = TransitionTable() if table is None else table
        self.dense = compile_dense_table(self.table)

    def encode_programs(self, programs: Sequence[str]) -> np.ndarray:
        return encode_padded_programs(
            programs,
            self.config.program_width,
            blank_id=self.table.blank_id,
            zero_id=self.table.zero_id,
            one_id=self.table.one_id,
        )

    def simulate_one(self, program: str, input_bits: str) -> BatchResult:
        return self.simulate(self.encode_programs([program]), input_bits)

    def simulate(self, programs, input_bits: str) -> BatchResult:
        raw = np.asarray(programs)
        if raw.ndim != 2 or raw.shape[0] == 0:
            raise ValueError("programs must be a nonempty two-dimensional array")
        if raw.shape[1] != self.config.program_width:
            raise ValueError("program width disagrees with the simulator")
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("program symbols must use an integer dtype")
        lengths = effective_program_lengths(
            raw,
            blank_id=self.table.blank_id,
            zero_id=self.table.zero_id,
            one_id=self.table.one_id,
        ).astype(np.int32)
        programs_array = raw.astype(np.uint8, copy=False)
        validate_binary("input_bits", input_bits)
        if len(input_bits) > self.config.right_budget:
            raise ValueError("input does not fit within right_budget")

        population = len(programs_array)
        origin = self.config.origin_index
        rows = np.arange(population, dtype=np.int64)
        tape = np.full(
            (population, self.config.tape_size),
            self.table.blank_id,
            dtype=np.uint8,
        )
        tape[:, origin - self.config.program_width : origin] = programs_array
        tape[:, origin] = self.table.separator_id
        if input_bits:
            encoded_input = np.fromiter(
                (
                    self.table.one_id if bit == "1" else self.table.zero_id
                    for bit in input_bits
                ),
                dtype=np.uint8,
                count=len(input_bits),
            )
            tape[:, origin + 1 : origin + 1 + len(input_bits)] = encoded_input

        states = np.full(population, self.table.start_id, dtype=np.uint16)
        heads = np.zeros(population, dtype=np.int32)
        steps = np.zeros(population, dtype=np.uint32)
        left_used = lengths.copy()
        right_used = np.full(population, len(input_bits), dtype=np.int32)
        done = np.zeros(population, dtype=np.bool_)
        halted = np.zeros(population, dtype=np.bool_)
        reasons = np.full(population, InvalidReason.NONE, dtype=np.uint8)

        for _ in range(self.config.t_max):
            active = rows[~done]
            if not active.size:
                break
            active_heads = heads[active]
            tape_indices = origin + active_heads
            if np.any((tape_indices < 0) | (tape_indices >= tape.shape[1])):
                raise RuntimeError("an active head is outside the allocated tape")
            active_states = states[active]
            read_symbols = tape[active, tape_indices]
            valid = self.dense.valid[active_states, read_symbols]

            invalid_rows = active[~valid]
            if invalid_rows.size:
                reasons[invalid_rows] = InvalidReason.INVALID_TRANSITION
                done[invalid_rows] = True

            stepping = active[valid]
            if not stepping.size:
                continue
            indices = tape_indices[valid]
            state_ids = active_states[valid]
            reads = read_symbols[valid]
            tape[stepping, indices] = self.dense.write[state_ids, reads]
            new_states = self.dense.next_state[state_ids, reads]
            new_heads = heads[stepping] + self.dense.move[
                state_ids, reads
            ].astype(np.int32)
            states[stepping] = new_states
            heads[stepping] = new_heads
            steps[stepping] += np.uint32(1)
            left_used[stepping] = np.maximum(left_used[stepping], -new_heads)
            right_used[stepping] = np.maximum(right_used[stepping], new_heads)

            left_failed = new_heads < -self.config.left_budget
            right_failed = new_heads > self.config.right_budget
            if np.any(left_failed):
                failed = stepping[left_failed]
                reasons[failed] = InvalidReason.LEFT_SPACE_EXCEEDED
                done[failed] = True
            if np.any(right_failed):
                failed = stepping[right_failed]
                reasons[failed] = InvalidReason.RIGHT_SPACE_EXCEEDED
                done[failed] = True
            in_bounds = ~(left_failed | right_failed)
            halted_now = in_bounds & (new_states == self.table.halt_id)
            if np.any(halted_now):
                finished = stepping[halted_now]
                halted[finished] = True
                done[finished] = True

        reasons[~done] = InvalidReason.TIMEOUT
        output_bits = np.full(
            (population, self.config.right_budget),
            OUTPUT_PAD,
            dtype=np.uint8,
        )
        output_lengths = np.zeros(population, dtype=np.uint32)
        decode_rows = rows[halted & (reasons == InvalidReason.NONE)]
        if decode_rows.size:
            separator_present = tape[decode_rows, origin] == self.table.separator_id
            missing = decode_rows[~separator_present]
            reasons[missing] = InvalidReason.MISSING_SEPARATOR
            decodable = decode_rows[separator_present]
            if decodable.size:
                right_tape = tape[decodable, origin + 1 :]
                binary = (right_tape == self.table.zero_id) | (
                    right_tape == self.table.one_id
                )
                prefix = np.logical_and.accumulate(binary, axis=1)
                decoded_lengths = prefix.sum(axis=1, dtype=np.uint32)
                logical = np.full(right_tape.shape, OUTPUT_PAD, dtype=np.uint8)
                logical[right_tape == self.table.zero_id] = 0
                logical[right_tape == self.table.one_id] = 1
                output_bits[decodable] = np.where(prefix, logical, OUTPUT_PAD)
                output_lengths[decodable] = decoded_lengths
                reasons[decodable[decoded_lengths == 0]] = (
                    InvalidReason.EMPTY_OUTPUT
                )

        return BatchResult(
            output_bits=output_bits,
            output_lengths=output_lengths,
            invalid=reasons != InvalidReason.NONE,
            invalid_reason=reasons,
            T=steps,
            left_space_used=left_used,
            right_space_used=right_used,
            halted=halted,
            final_state_ids=states,
            final_heads=heads,
            final_tape=tape,
            effective_program_lengths=lengths,
        )


def evaluate_program(
    simulator: NumpyUTMSimulator,
    program: str,
    task: TaskCases,
) -> tuple[CaseEvaluation, ...]:
    task.validate_right_budget(simulator.config.right_budget)
    padded = simulator.encode_programs([program])
    evaluations = []
    for input_bits, target in task.pairs():
        result = simulator.simulate(padded, input_bits)
        output = result.output_strings()[0]
        invalid = bool(result.invalid[0])
        evaluations.append(
            CaseEvaluation(
                input_bits=input_bits,
                target=target,
                output=output,
                exact=not invalid and output == target,
                invalid=invalid,
                invalid_reason=result.invalid_reason_names()[0],
                T=int(result.T[0]),
                left_space_used=int(result.left_space_used[0]),
                right_space_used=int(result.right_space_used[0]),
            )
        )
    return tuple(evaluations)


def find_exact_program(
    simulator: NumpyUTMSimulator,
    task: TaskCases,
    *,
    batch_size: int = 10_000,
    stop_first_exact: bool = True,
    print_every_batches: int | None = None,
) -> EnumerationResult | None:
    if not isinstance(stop_first_exact, bool):
        raise TypeError("stop_first_exact must be boolean")
    if print_every_batches is not None and print_every_batches < 1:
        raise ValueError("print_every_batches must be positive or None")
    task.validate_right_budget(simulator.config.right_budget)
    started = perf_counter()
    evaluated = 0
    first: EnumerationResult | None = None
    for batch_index, batch in enumerate(
        enumerate_program_batches(
            simulator.config.program_width,
            batch_size,
            blank_id=simulator.table.blank_id,
            zero_id=simulator.table.zero_id,
            one_id=simulator.table.one_id,
        ),
        start=1,
    ):
        exact = np.ones(len(batch.symbols), dtype=np.bool_)
        for input_bits, target in task.pairs():
            result = simulator.simulate(batch.symbols, input_bits)
            target_bits = np.fromiter(
                (int(bit) for bit in target),
                dtype=np.uint8,
                count=len(target),
            )
            exact &= (
                ~result.invalid
                & (result.output_lengths == len(target))
                & np.all(
                    result.output_bits[:, : len(target)]
                    == target_bits[None, :],
                    axis=1,
                )
            )
        evaluated += len(batch.symbols)
        matches = np.flatnonzero(exact)
        if matches.size and first is None:
            index = int(matches[0])
            symbols = batch.symbols[index]
            first = EnumerationResult(
                program=effective_program_to_string(
                    symbols,
                    blank_id=simulator.table.blank_id,
                    zero_id=simulator.table.zero_id,
                    one_id=simulator.table.one_id,
                ),
                padded_program=padded_program_to_string(
                    symbols,
                    blank_id=simulator.table.blank_id,
                    zero_id=simulator.table.zero_id,
                    one_id=simulator.table.one_id,
                ),
                program_ordinal=batch.start_ordinal + index,
                effective_length=int(batch.effective_lengths[index]),
                evaluated_programs=evaluated,
                elapsed_seconds=perf_counter() - started,
            )
            if stop_first_exact:
                return first
        if print_every_batches is not None and batch_index % print_every_batches == 0:
            elapsed = perf_counter() - started
            print(
                f"evaluated={evaluated:,} elapsed={elapsed:.3f}s "
                f"rate={evaluated / max(elapsed, 1e-12):,.0f}/s",
                flush=True,
            )
    return first
