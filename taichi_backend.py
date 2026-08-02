"""Strict-CUDA Taichi LUTM simulator, evaluator, and enumerator."""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from time import perf_counter
import tempfile
from typing import Sequence

import numpy as np
import taichi as ti

from lutm import InvalidReason, TransitionTable
from numpy_backend import (
    BatchResult,
    CaseEvaluation,
    EnumerationResult,
    OUTPUT_PAD,
    compile_dense_table,
)
from utils import (
    SimulatorConfig,
    TaskCases,
    effective_program_lengths,
    encode_padded_programs,
    ordinal_to_program,
    program_count,
    validate_binary,
)


_TAICHI_INITIALIZED = False


def initialize_taichi_cuda() -> None:
    """Initialize Taichi on CUDA, explicitly forbidding architecture fallback."""
    global _TAICHI_INITIALIZED
    if _TAICHI_INITIALIZED:
        return

    cache_path = Path(tempfile.gettempdir()) / "lutm_1_taichi_cache"
    ti.init(
        arch=ti.cuda,
        enable_fallback=False,
        offline_cache=False,
        offline_cache_file_path=str(cache_path),
    )
    if ti.lang.impl.current_cfg().arch != ti.cuda:
        raise RuntimeError("Taichi did not initialize on CUDA")
    _TAICHI_INITIALIZED = True


@dataclass(frozen=True)
class TaichiBatchSpec:
    """One valid interval in the length-first ordinal program ordering."""

    start_ordinal: int
    population: int


@dataclass(frozen=True, eq=False)
class TaichiFitnessBatch:
    """The minimal result copied from CUDA for NumPy-side fitness."""

    output_bits: np.ndarray
    output_lengths: np.ndarray
    invalid: np.ndarray
    invalid_reason: np.ndarray

    def output_strings(self) -> list[str]:
        return [
            "".join(str(int(bit)) for bit in row[: int(length)])
            for row, length in zip(self.output_bits, self.output_lengths)
        ]


@ti.data_oriented
class TaichiUTMSimulator:
    """Run independent blank-padded LUTM programs in parallel on CUDA."""

    def __init__(
        self,
        config: SimulatorConfig,
        batch_capacity: int,
        table: TransitionTable | None = None,
    ) -> None:
        if not _TAICHI_INITIALIZED:
            raise RuntimeError(
                "Call initialize_taichi_cuda() before constructing "
                "TaichiUTMSimulator"
            )
        if not isinstance(config, SimulatorConfig):
            raise TypeError("config must be a SimulatorConfig")
        if not isinstance(batch_capacity, Integral) or isinstance(
            batch_capacity, (bool, np.bool_)
        ):
            raise TypeError("batch_capacity must be an integer")
        if batch_capacity < 1:
            raise ValueError("batch_capacity must be at least 1")

        self.config = config
        self.batch_capacity = int(batch_capacity)
        self.table = TransitionTable() if table is None else table
        if not isinstance(self.table, TransitionTable):
            raise TypeError("table must be a TransitionTable")
        dense = compile_dense_table(self.table)

        self.program_width = int(config.program_width)
        self.left_budget = int(config.left_budget)
        self.right_budget = int(config.right_budget)
        self.t_max = int(config.t_max)
        self.tape_size = int(config.tape_size)
        self.origin = int(config.origin_index)

        self.blank_id = self.table.blank_id
        self.zero_id = self.table.zero_id
        self.one_id = self.table.one_id
        self.separator_id = self.table.separator_id
        self.start_state_id = self.table.start_id
        self.halt_state_id = self.table.halt_id

        lut_shape = (len(self.table.states), len(self.table.symbols))
        self.lut_valid = ti.field(dtype=ti.u8, shape=lut_shape)
        self.lut_write = ti.field(dtype=ti.u8, shape=lut_shape)
        self.lut_move = ti.field(dtype=ti.i8, shape=lut_shape)
        self.lut_next = ti.field(dtype=ti.u16, shape=lut_shape)

        self.input_symbols = ti.field(dtype=ti.u8, shape=self.right_budget)
        self.program_symbols = ti.field(
            dtype=ti.u8,
            shape=(self.batch_capacity, self.program_width),
        )
        self.tape = ti.field(
            dtype=ti.u8,
            shape=(self.batch_capacity, self.tape_size),
        )
        self.output_bits = ti.field(
            dtype=ti.u8,
            shape=(self.batch_capacity, self.right_budget),
        )
        self.output_lengths = ti.field(dtype=ti.u32, shape=self.batch_capacity)
        self.invalid_reasons = ti.field(dtype=ti.u8, shape=self.batch_capacity)
        self.transition_counts = ti.field(dtype=ti.u32, shape=self.batch_capacity)
        self.left_space_used = ti.field(dtype=ti.i32, shape=self.batch_capacity)
        self.right_space_used = ti.field(dtype=ti.i32, shape=self.batch_capacity)
        self.halted = ti.field(dtype=ti.u8, shape=self.batch_capacity)
        self.states = ti.field(dtype=ti.u16, shape=self.batch_capacity)
        self.heads = ti.field(dtype=ti.i32, shape=self.batch_capacity)
        self.effective_program_lengths = ti.field(
            dtype=ti.i32,
            shape=self.batch_capacity,
        )

        self.lut_valid.from_numpy(dense.valid.astype(np.uint8))
        self.lut_write.from_numpy(dense.write.astype(np.uint8))
        self.lut_move.from_numpy(dense.move.astype(np.int8))
        self.lut_next.from_numpy(dense.next_state.astype(np.uint16))

    def encode_programs(self, programs: Sequence[str]) -> np.ndarray:
        """Encode effective binary programs as blank-left-padded rows."""
        return encode_padded_programs(
            programs,
            self.program_width,
            blank_id=self.blank_id,
            zero_id=self.zero_id,
            one_id=self.one_id,
        )

    @ti.func
    def _simulate_loaded_program(
        self,
        program_index: ti.i32,
        input_length: ti.i32,
        effective_length: ti.i32,
    ):
        for tape_index in range(self.tape_size):
            self.tape[program_index, tape_index] = ti.cast(
                self.blank_id, ti.u8
            )
        for output_index in range(self.right_budget):
            self.output_bits[program_index, output_index] = ti.cast(
                int(OUTPUT_PAD), ti.u8
            )

        program_start = self.origin - self.program_width
        for column in range(self.program_width):
            self.tape[program_index, program_start + column] = (
                self.program_symbols[program_index, column]
            )

        self.tape[program_index, self.origin] = ti.cast(
            self.separator_id, ti.u8
        )
        for input_index in range(input_length):
            self.tape[program_index, self.origin + 1 + input_index] = (
                self.input_symbols[input_index]
            )

        state = ti.cast(self.start_state_id, ti.u16)
        head = ti.cast(0, ti.i32)
        steps = ti.cast(0, ti.u32)
        left_used = effective_length
        right_used = input_length
        reason = ti.cast(int(InvalidReason.NONE), ti.u8)
        is_halted = ti.cast(0, ti.u8)
        is_done = ti.cast(0, ti.u8)

        while (
            is_done == ti.cast(0, ti.u8)
            and steps < ti.cast(self.t_max, ti.u32)
        ):
            tape_index = self.origin + head
            read_symbol = self.tape[program_index, tape_index]
            lut_state = ti.cast(state, ti.i32)
            lut_symbol = ti.cast(read_symbol, ti.i32)

            if self.lut_valid[lut_state, lut_symbol] == 0:
                reason = ti.cast(
                    int(InvalidReason.INVALID_TRANSITION), ti.u8
                )
                is_done = ti.cast(1, ti.u8)
            else:
                write_symbol = self.lut_write[lut_state, lut_symbol]
                move = ti.cast(self.lut_move[lut_state, lut_symbol], ti.i32)
                next_state = self.lut_next[lut_state, lut_symbol]

                self.tape[program_index, tape_index] = write_symbol
                head += move
                state = next_state
                steps += ti.cast(1, ti.u32)

                if head < 0:
                    left_used = ti.max(left_used, -head)
                else:
                    right_used = ti.max(right_used, head)

                if head < -self.left_budget:
                    reason = ti.cast(
                        int(InvalidReason.LEFT_SPACE_EXCEEDED), ti.u8
                    )
                    is_done = ti.cast(1, ti.u8)
                elif head > self.right_budget:
                    reason = ti.cast(
                        int(InvalidReason.RIGHT_SPACE_EXCEEDED), ti.u8
                    )
                    is_done = ti.cast(1, ti.u8)
                elif state == self.halt_state_id:
                    is_halted = ti.cast(1, ti.u8)
                    is_done = ti.cast(1, ti.u8)

        if is_done == ti.cast(0, ti.u8):
            reason = ti.cast(int(InvalidReason.TIMEOUT), ti.u8)

        output_length = ti.cast(0, ti.u32)
        if (
            reason == ti.cast(int(InvalidReason.NONE), ti.u8)
            and is_halted != 0
        ):
            if self.tape[program_index, self.origin] != self.separator_id:
                reason = ti.cast(
                    int(InvalidReason.MISSING_SEPARATOR), ti.u8
                )
            else:
                reading_output = ti.cast(1, ti.u8)
                for output_index in range(self.right_budget):
                    if reading_output != 0:
                        symbol = self.tape[
                            program_index, self.origin + 1 + output_index
                        ]
                        if symbol == self.zero_id:
                            self.output_bits[
                                program_index, output_index
                            ] = ti.cast(0, ti.u8)
                            output_length += ti.cast(1, ti.u32)
                        elif symbol == self.one_id:
                            self.output_bits[
                                program_index, output_index
                            ] = ti.cast(1, ti.u8)
                            output_length += ti.cast(1, ti.u32)
                        else:
                            reading_output = ti.cast(0, ti.u8)

                if output_length == 0:
                    reason = ti.cast(int(InvalidReason.EMPTY_OUTPUT), ti.u8)

        self.output_lengths[program_index] = output_length
        self.invalid_reasons[program_index] = reason
        self.transition_counts[program_index] = steps
        self.left_space_used[program_index] = left_used
        self.right_space_used[program_index] = right_used
        self.halted[program_index] = is_halted
        self.states[program_index] = state
        self.heads[program_index] = head
        self.effective_program_lengths[program_index] = effective_length

    @ti.kernel
    def _enumerate_and_simulate(
        self,
        start_ordinal: ti.u64,
        population: ti.i32,
        input_length: ti.i32,
    ):
        ti.loop_config(block_dim=128)
        for program_index in range(population):
            ordinal = start_ordinal + ti.cast(program_index, ti.u64)

            n = ordinal + ti.cast(1, ti.u64)
            effective_length = ti.cast(0, ti.i32)
            while n > ti.cast(1, ti.u64):
                n = n >> 1
                effective_length += 1

            level_start = (
                ti.cast(1, ti.u64) << effective_length
            ) - ti.cast(1, ti.u64)
            value = ordinal - level_start
            suffix_start = self.program_width - effective_length
            for column in range(self.program_width):
                symbol = ti.cast(self.blank_id, ti.u8)
                if column >= suffix_start:
                    shift = self.program_width - 1 - column
                    bit = (value >> shift) & ti.cast(1, ti.u64)
                    symbol = ti.cast(self.zero_id, ti.u8)
                    if bit != ti.cast(0, ti.u64):
                        symbol = ti.cast(self.one_id, ti.u8)
                self.program_symbols[program_index, column] = symbol

            self._simulate_loaded_program(
                program_index,
                input_length,
                effective_length,
            )

    @ti.kernel
    def _simulate_explicit_programs(
        self,
        population: ti.i32,
        input_length: ti.i32,
    ):
        ti.loop_config(block_dim=128)
        for program_index in range(population):
            self._simulate_loaded_program(
                program_index,
                input_length,
                self.effective_program_lengths[program_index],
            )

    def _validate_ordinal_batch(
        self,
        start_ordinal: int,
        population: int,
    ) -> TaichiBatchSpec:
        if self.program_width > 62:
            raise ValueError(
                "ordinal enumeration supports program_width <= 62; "
                "use simulate_program_batch for wider explicit programs"
            )
        if not isinstance(start_ordinal, Integral) or isinstance(
            start_ordinal, (bool, np.bool_)
        ):
            raise TypeError("start_ordinal must be an integer")
        if not isinstance(population, Integral) or isinstance(
            population, (bool, np.bool_)
        ):
            raise TypeError("population must be an integer")
        start_ordinal = int(start_ordinal)
        population = int(population)
        if start_ordinal < 0:
            raise ValueError("start_ordinal must be non-negative")
        if population < 1:
            raise ValueError("population must be at least 1")
        if population > self.batch_capacity:
            raise ValueError(
                f"population {population} exceeds batch_capacity "
                f"{self.batch_capacity}"
            )
        if start_ordinal + population > program_count(self.program_width):
            raise ValueError(
                "ordinal batch exceeds the configured program-width search space"
            )
        return TaichiBatchSpec(start_ordinal, population)

    def _upload_input(self, input_bits: str) -> None:
        validate_binary("input_bits", input_bits)
        if len(input_bits) > self.right_budget:
            raise ValueError("input does not fit within right_budget")
        encoded = np.full(
            self.right_budget,
            self.blank_id,
            dtype=np.uint8,
        )
        if input_bits:
            encoded[: len(input_bits)] = np.fromiter(
                (
                    self.one_id if bit == "1" else self.zero_id
                    for bit in input_bits
                ),
                dtype=np.uint8,
                count=len(input_bits),
            )
        self.input_symbols.from_numpy(encoded)

    def _load_explicit_programs(self, programs) -> int:
        raw = np.asarray(programs)
        if raw.ndim != 2 or raw.shape[0] == 0:
            raise ValueError("programs must be a nonempty two-dimensional array")
        if raw.shape[1] != self.program_width:
            raise ValueError("program width disagrees with the simulator")
        if raw.shape[0] > self.batch_capacity:
            raise ValueError(
                f"population {raw.shape[0]} exceeds batch_capacity "
                f"{self.batch_capacity}"
            )
        lengths = effective_program_lengths(
            raw,
            blank_id=self.blank_id,
            zero_id=self.zero_id,
            one_id=self.one_id,
        ).astype(np.int32)
        population = int(raw.shape[0])

        program_buffer = np.full(
            (self.batch_capacity, self.program_width),
            self.blank_id,
            dtype=np.uint8,
        )
        program_buffer[:population] = raw.astype(np.uint8, copy=False)
        length_buffer = np.zeros(self.batch_capacity, dtype=np.int32)
        length_buffer[:population] = lengths
        self.program_symbols.from_numpy(program_buffer)
        self.effective_program_lengths.from_numpy(length_buffer)
        return population

    def _launch_ordinal(
        self,
        start_ordinal: int,
        population: int,
        input_bits: str,
    ) -> TaichiBatchSpec:
        spec = self._validate_ordinal_batch(start_ordinal, population)
        self._upload_input(input_bits)
        self._enumerate_and_simulate(
            spec.start_ordinal,
            spec.population,
            len(input_bits),
        )
        return spec

    def _launch_explicit(self, programs, input_bits: str) -> int:
        population = self._load_explicit_programs(programs)
        self._upload_input(input_bits)
        self._simulate_explicit_programs(population, len(input_bits))
        return population

    def _fitness_result(self, size: int) -> TaichiFitnessBatch:
        reasons = self.invalid_reasons.to_numpy()[:size].copy()
        return TaichiFitnessBatch(
            output_bits=self.output_bits.to_numpy()[:size].copy(),
            output_lengths=self.output_lengths.to_numpy()[:size].copy(),
            invalid=reasons != int(InvalidReason.NONE),
            invalid_reason=reasons,
        )

    def _complete_result(self, size: int) -> BatchResult:
        reasons = self.invalid_reasons.to_numpy()[:size].copy()
        return BatchResult(
            output_bits=self.output_bits.to_numpy()[:size].copy(),
            output_lengths=self.output_lengths.to_numpy()[:size].copy(),
            invalid=reasons != int(InvalidReason.NONE),
            invalid_reason=reasons,
            T=self.transition_counts.to_numpy()[:size].copy(),
            left_space_used=self.left_space_used.to_numpy()[:size].copy(),
            right_space_used=self.right_space_used.to_numpy()[:size].copy(),
            halted=self.halted.to_numpy()[:size].astype(bool, copy=True),
            final_state_ids=self.states.to_numpy()[:size].copy(),
            final_heads=self.heads.to_numpy()[:size].copy(),
            final_tape=self.tape.to_numpy()[:size].copy(),
            effective_program_lengths=(
                self.effective_program_lengths.to_numpy()[:size].copy()
            ),
        )

    def simulate_one(self, program: str, input_bits: str) -> BatchResult:
        """Simulate one effective binary program on one input."""
        return self.simulate_program_batch(
            self.encode_programs([program]),
            input_bits,
        )

    def simulate_ordinal_batch_for_fitness(
        self,
        start_ordinal: int,
        population: int,
        input_bits: str,
    ) -> TaichiFitnessBatch:
        """Enumerate and run a batch, copying only fitness fields to NumPy."""
        spec = self._launch_ordinal(start_ordinal, population, input_bits)
        return self._fitness_result(spec.population)

    def simulate_ordinal_batch(
        self,
        start_ordinal: int,
        population: int,
        input_bits: str,
    ) -> BatchResult:
        """Enumerate and run a batch, returning complete diagnostics."""
        spec = self._launch_ordinal(start_ordinal, population, input_bits)
        return self._complete_result(spec.population)

    def simulate_program_batch_for_fitness(
        self,
        programs,
        input_bits: str,
    ) -> TaichiFitnessBatch:
        """Run explicit padded programs, copying only fitness fields."""
        population = self._launch_explicit(programs, input_bits)
        return self._fitness_result(population)

    def simulate_program_batch(
        self,
        programs,
        input_bits: str,
    ) -> BatchResult:
        """Run an explicit fixed-width padded program batch."""
        population = self._launch_explicit(programs, input_bits)
        return self._complete_result(population)


def evaluate_program_taichi(
    simulator: TaichiUTMSimulator,
    program: str,
    task: TaskCases,
) -> tuple[CaseEvaluation, ...]:
    """Run one effective program and return readable task diagnostics."""
    task.validate_right_budget(simulator.config.right_budget)
    padded = simulator.encode_programs([program])
    evaluations: list[CaseEvaluation] = []
    for input_bits, target in task.pairs():
        result = simulator.simulate_program_batch(padded, input_bits)
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


def _exact_ordinal_mask(
    simulator: TaichiUTMSimulator,
    start_ordinal: int,
    population: int,
    task: TaskCases,
) -> np.ndarray:
    task.validate_right_budget(simulator.config.right_budget)
    exact = np.ones(population, dtype=np.bool_)
    for input_bits, target in task.pairs():
        result = simulator.simulate_ordinal_batch_for_fitness(
            start_ordinal,
            population,
            input_bits,
        )
        target_bits = np.fromiter(
            (int(bit) for bit in target),
            dtype=np.uint8,
            count=len(target),
        )
        exact &= (
            ~result.invalid
            & (result.output_lengths == len(target))
            & np.all(
                result.output_bits[:, : len(target)] == target_bits[None, :],
                axis=1,
            )
        )
        if not np.any(exact):
            break
    return exact


def find_exact_program_taichi(
    simulator: TaichiUTMSimulator,
    task: TaskCases,
    *,
    batch_size: int | None = None,
    max_programs: int | None = None,
    stop_first_exact: bool = True,
    print_every_batches: int | None = None,
) -> EnumerationResult | None:
    """Search the length-first program ordering with strict-CUDA simulation."""
    if not isinstance(stop_first_exact, bool):
        raise TypeError("stop_first_exact must be boolean")
    size = simulator.batch_capacity if batch_size is None else batch_size
    if not isinstance(size, Integral) or isinstance(size, (bool, np.bool_)):
        raise TypeError("batch_size must be an integer or None")
    if not 1 <= size <= simulator.batch_capacity:
        raise ValueError("batch_size must be within the GPU batch capacity")
    size = int(size)

    total = program_count(simulator.config.program_width)
    if max_programs is not None:
        if not isinstance(max_programs, Integral) or isinstance(
            max_programs, (bool, np.bool_)
        ):
            raise TypeError("max_programs must be an integer or None")
        if max_programs < 1:
            raise ValueError("max_programs must be positive")
        total = min(total, int(max_programs))
    if print_every_batches is not None:
        if not isinstance(print_every_batches, Integral) or isinstance(
            print_every_batches, (bool, np.bool_)
        ):
            raise TypeError("print_every_batches must be an integer or None")
        if print_every_batches < 1:
            raise ValueError("print_every_batches must be positive")

    task.validate_right_budget(simulator.config.right_budget)
    started = perf_counter()
    evaluated = 0
    first: EnumerationResult | None = None
    for batch_index, start in enumerate(range(0, total, size), start=1):
        population = min(size, total - start)
        exact = _exact_ordinal_mask(simulator, start, population, task)
        evaluated += population
        matches = np.flatnonzero(exact)
        if matches.size and first is None:
            row = int(matches[0])
            ordinal = start + row
            program = ordinal_to_program(
                ordinal,
                simulator.config.program_width,
            )
            first = EnumerationResult(
                program=program,
                padded_program="B" * (simulator.program_width - len(program))
                + program,
                program_ordinal=ordinal,
                effective_length=len(program),
                evaluated_programs=evaluated,
                elapsed_seconds=perf_counter() - started,
            )
            if stop_first_exact:
                return first
        terminal = evaluated == total
        if print_every_batches is not None and (
            batch_index % print_every_batches == 0 or terminal
        ):
            elapsed = perf_counter() - started
            print(
                f"batch={batch_index:>4} evaluated={evaluated:,}/{total:,} "
                f"rate={evaluated / max(elapsed, 1e-12):,.0f}/s "
                f"elapsed={elapsed:.3f}s",
                flush=True,
            )
    return first
