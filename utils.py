"""Shared validation, padding, task, and enumeration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterator, Sequence


UINT32_MAX = (1 << 32) - 1
INT32_MAX = (1 << 31) - 1


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def validate_binary(name: str, bits: str, *, allow_empty: bool = True) -> None:
    if not isinstance(bits, str):
        raise TypeError(f"{name} must be a string")
    if not allow_empty and not bits:
        raise ValueError(f"{name} must not be empty")
    if any(bit not in "01" for bit in bits):
        raise ValueError(f"{name} may contain only 0 and 1")


@dataclass(frozen=True)
class TaskCases:
    inputs: Sequence[str]
    targets: Sequence[str]

    def __post_init__(self) -> None:
        if isinstance(self.inputs, (str, bytes)) or not isinstance(
            self.inputs, Sequence
        ):
            raise TypeError("inputs must be a sequence of strings")
        if isinstance(self.targets, (str, bytes)) or not isinstance(
            self.targets, Sequence
        ):
            raise TypeError("targets must be a sequence of strings")
        inputs = tuple(self.inputs)
        targets = tuple(self.targets)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "targets", targets)
        if not inputs:
            raise ValueError("inputs and targets must not be empty")
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must have the same length")
        for bits in inputs:
            validate_binary("input", bits)
        for bits in targets:
            validate_binary("target", bits, allow_empty=False)
        if len(set(inputs)) != len(inputs):
            raise ValueError("duplicate inputs are not allowed")

    def __len__(self) -> int:
        return len(self.inputs)

    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.inputs, self.targets))

    def validate_right_budget(self, right_budget: int) -> None:
        if max(map(len, self.inputs)) > right_budget:
            raise ValueError("an input exceeds right_budget")
        if max(map(len, self.targets)) > right_budget:
            raise ValueError("a target exceeds right_budget")


@dataclass(frozen=True)
class SimulatorConfig:
    program_width: int
    left_budget: int
    right_budget: int
    t_max: int

    def __post_init__(self) -> None:
        values = (
            self.program_width,
            self.left_budget,
            self.right_budget,
            self.t_max,
        )
        if not all(_is_integer(value) for value in values):
            raise TypeError("all simulator bounds must be integers")
        if self.program_width <= 0:
            raise ValueError("program_width must be positive")
        if self.left_budget < self.program_width:
            raise ValueError(
                "left_budget must be at least program_width"
            )
        if self.right_budget <= 0:
            raise ValueError("right_budget must be positive")
        if not 1 <= self.t_max <= UINT32_MAX:
            raise ValueError(f"t_max must be between 1 and {UINT32_MAX:,}")
        if self.left_budget > INT32_MAX or self.right_budget > INT32_MAX:
            raise ValueError("space budgets must fit in int32")
        if self.tape_size > INT32_MAX:
            raise ValueError("total tape size must fit in int32")

    @property
    def tape_size(self) -> int:
        return self.left_budget + self.right_budget + 1

    @property
    def origin_index(self) -> int:
        return self.left_budget


@dataclass(frozen=True)
class ProgramBatch:
    start_ordinal: int
    symbols: object
    effective_lengths: object

    @property
    def end_ordinal(self) -> int:
        return self.start_ordinal + len(self.symbols)  # type: ignore[arg-type]


def iter_binary_inputs(min_length: int, max_length: int) -> Iterator[str]:
    if not _is_integer(min_length) or not _is_integer(max_length):
        raise TypeError("input lengths must be integers")
    if min_length < 0 or max_length < min_length:
        raise ValueError("require 0 <= min_length <= max_length")
    for length in range(min_length, max_length + 1):
        if length == 0:
            yield ""
            continue
        for value in range(1 << length):
            yield format(value, f"0{length}b")


def program_count(program_width: int) -> int:
    if not _is_integer(program_width):
        raise TypeError("program_width must be an integer")
    if not 1 <= program_width <= 62:
        raise ValueError("ordinal enumeration requires program_width in 1..62")
    return (1 << (program_width + 1)) - 1


def program_ordinal(program: str, program_width: int) -> int:
    total = program_count(program_width)
    del total
    validate_binary("program", program)
    if len(program) > program_width:
        raise ValueError("program exceeds program_width")
    value = 0 if not program else int(program, 2)
    return (1 << len(program)) - 1 + value


def ordinal_to_program(ordinal: int, program_width: int) -> str:
    if not _is_integer(ordinal):
        raise TypeError("ordinal must be an integer")
    total = program_count(program_width)
    if not 0 <= ordinal < total:
        raise ValueError("ordinal is outside the program space")
    length = (ordinal + 1).bit_length() - 1
    if length == 0:
        return ""
    return format(ordinal - ((1 << length) - 1), f"0{length}b")


def encode_padded_programs(
    programs: Sequence[str],
    program_width: int,
    *,
    blank_id: int,
    zero_id: int,
    one_id: int,
):
    import numpy as np

    if isinstance(programs, (str, bytes)) or not isinstance(programs, Sequence):
        raise TypeError("programs must be a sequence of strings")
    if not programs:
        raise ValueError("programs must not be empty")
    if not _is_integer(program_width) or program_width <= 0:
        raise ValueError("program_width must be a positive integer")
    identifiers = (blank_id, zero_id, one_id)
    if any(not _is_integer(value) for value in identifiers):
        raise TypeError("symbol IDs must be integers")
    if len(set(map(int, identifiers))) != 3:
        raise ValueError("blank, zero, and one IDs must be distinct")
    result = np.full((len(programs), program_width), blank_id, dtype=np.uint8)
    for row, program in enumerate(programs):
        validate_binary("program", program)
        if len(program) > program_width:
            raise ValueError("program exceeds program_width")
        if program:
            result[row, -len(program) :] = np.fromiter(
                (one_id if bit == "1" else zero_id for bit in program),
                dtype=np.uint8,
                count=len(program),
            )
    return result


def effective_program_lengths(
    programs,
    *,
    blank_id: int,
    zero_id: int,
    one_id: int,
):
    import numpy as np

    array = np.asarray(programs)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("programs must be a nonempty two-dimensional array")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("program symbols must use an integer dtype")
    allowed = (array == blank_id) | (array == zero_id) | (array == one_id)
    if not np.all(allowed):
        raise ValueError("programs contain an unknown symbol ID")
    binary = (array == zero_id) | (array == one_id)
    lengths = binary.sum(axis=1, dtype=np.int32)
    width = array.shape[1]
    expected_binary = np.arange(width)[None, :] >= (width - lengths)[:, None]
    if not np.array_equal(binary, expected_binary):
        raise ValueError("programs must be blank-left-padded binary suffixes")
    return lengths


def padded_program_to_string(
    symbols,
    *,
    blank_id: int,
    zero_id: int,
    one_id: int,
) -> str:
    mapping = {blank_id: "B", zero_id: "0", one_id: "1"}
    try:
        return "".join(mapping[int(value)] for value in symbols)
    except KeyError as error:
        raise ValueError("program contains an unknown symbol ID") from error


def effective_program_to_string(
    symbols,
    *,
    blank_id: int,
    zero_id: int,
    one_id: int,
) -> str:
    padded = padded_program_to_string(
        symbols,
        blank_id=blank_id,
        zero_id=zero_id,
        one_id=one_id,
    )
    return padded.lstrip("B")


def enumerate_program_batches(
    program_width: int,
    batch_size: int,
    *,
    blank_id: int,
    zero_id: int,
    one_id: int,
) -> Iterator[ProgramBatch]:
    import numpy as np

    total = program_count(program_width)
    if not _is_integer(batch_size) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        strings = [ordinal_to_program(value, program_width) for value in range(start, stop)]
        symbols = encode_padded_programs(
            strings,
            program_width,
            blank_id=blank_id,
            zero_id=zero_id,
            one_id=one_id,
        )
        lengths = np.fromiter(map(len, strings), dtype=np.int32)
        yield ProgramBatch(start, symbols, lengths)
