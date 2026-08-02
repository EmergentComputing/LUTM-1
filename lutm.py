"""Canonical LUTM transition table and self-extending scalar simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import csv
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


BLANK = "B"
ZERO = "0"
ONE = "1"
SEPARATOR = "#"
START = "START"
HALT = "HALT"
SYMBOLS = (
    "B",
    "0",
    "1",
    "#",
    "C0",
    "C1",
    "A0",
    "A1",
    "N",
    "X",
    "T",
    "Z",
    "hL",
    "hb",
    "h0",
    "h1",
)
CANONICAL_CSV_SHA256 = (
    "260fb1d15014523c6a46ede09f9fcd7cd7d591f668912842f1aae8cf993376f9"
)
UINT64_MAX = (1 << 64) - 1


class InvalidReason(IntEnum):
    NONE = 0
    TIMEOUT = 1
    LEFT_SPACE_EXCEEDED = 2
    RIGHT_SPACE_EXCEEDED = 3
    MISSING_SEPARATOR = 4
    EMPTY_OUTPUT = 5
    INVALID_TRANSITION = 6


@dataclass(frozen=True)
class Transition:
    write: str
    move: int
    next_state: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class TransitionTable:
    """Load and validate the one canonical CSV transition table."""

    def __init__(self, csv_path: Path | str | None = None) -> None:
        path = (
            Path(__file__).resolve().parent / "data" / "transition_table.csv"
            if csv_path is None
            else Path(csv_path).resolve()
        )
        if not path.is_file():
            raise FileNotFoundError(f"transition table not found: {path}")
        if _sha256(path) != CANONICAL_CSV_SHA256:
            raise ValueError("canonical transition-table CSV hash mismatch")

        rows: list[dict[str, str]] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected_fields = [
                "state",
                "read",
                "write",
                "move",
                "next_state",
            ]
            if reader.fieldnames != expected_fields:
                raise ValueError(
                    f"unexpected transition-table columns: {reader.fieldnames}"
                )
            rows.extend(reader)

        if not rows:
            raise ValueError("transition table is empty")
        state_names = {HALT}
        for row in rows:
            state_names.add(row["state"])
            state_names.add(row["next_state"])
        states = tuple(sorted(state_names))
        if len(states) != 150:
            raise ValueError(f"expected 150 states, found {len(states)}")
        if START not in state_names or HALT not in state_names:
            raise ValueError("transition table lacks START or HALT")

        moves = {"L": -1, "S": 0, "R": 1}
        transitions: dict[tuple[str, str], Transition] = {}
        for row in rows:
            state = row["state"]
            read = row["read"]
            write = row["write"]
            move_name = row["move"]
            next_state = row["next_state"]
            if read not in SYMBOLS or write not in SYMBOLS:
                raise ValueError(f"unknown symbol in transition row: {row}")
            if move_name not in moves:
                raise ValueError(f"unknown move in transition row: {row}")
            if next_state not in state_names:
                raise ValueError(f"unknown next state in transition row: {row}")
            key = (state, read)
            if key in transitions:
                raise ValueError(f"duplicate transition for {key}")
            transitions[key] = Transition(
                write=write,
                move=moves[move_name],
                next_state=next_state,
            )

        nonhalting = tuple(state for state in states if state != HALT)
        expected_rows = len(nonhalting) * len(SYMBOLS)
        if len(transitions) != expected_rows:
            raise ValueError(
                f"expected {expected_rows} transitions, found {len(transitions)}"
            )
        for state in nonhalting:
            missing = [
                symbol
                for symbol in SYMBOLS
                if (state, symbol) not in transitions
            ]
            if missing:
                raise ValueError(f"state {state} lacks transitions for {missing}")
        if any((HALT, symbol) in transitions for symbol in SYMBOLS):
            raise ValueError("HALT unexpectedly has outgoing transitions")

        self.path = path
        self.states = states
        self.symbols = SYMBOLS
        self.state_ids = MappingProxyType(
            {state: index for index, state in enumerate(states)}
        )
        self.symbol_ids = MappingProxyType(
            {symbol: index for index, symbol in enumerate(SYMBOLS)}
        )
        self.transitions: Mapping[tuple[str, str], Transition] = (
            MappingProxyType(transitions)
        )
        self.start_id = self.state_ids[START]
        self.halt_id = self.state_ids[HALT]
        self.blank_id = self.symbol_ids[BLANK]
        self.zero_id = self.symbol_ids[ZERO]
        self.one_id = self.symbol_ids[ONE]
        self.separator_id = self.symbol_ids[SEPARATOR]


@dataclass(frozen=True)
class ScalarResult:
    output: str
    invalid: bool
    invalid_reason: InvalidReason
    T: int
    left_space_used: int
    right_space_used: int
    halted: bool
    final_state: str
    final_head: int
    final_tape: Mapping[int, str]

    @property
    def span_used(self) -> int:
        return self.left_space_used + self.right_space_used + 1

    @property
    def max_space_used(self) -> int:
        return max(self.left_space_used, self.right_space_used)


def _validate_binary(name: str, bits: str) -> None:
    if not isinstance(bits, str):
        raise TypeError(f"{name} must be a string")
    if any(bit not in "01" for bit in bits):
        raise ValueError(f"{name} may contain only 0 and 1")


class ScalarUTMSimulator:
    """Exact, nonparallel LUTM with a self-extending dictionary tape."""

    def __init__(self, table: TransitionTable | None = None) -> None:
        self.table = TransitionTable() if table is None else table

    @staticmethod
    def initial_tape(program: str, input_bits: str) -> dict[int, str]:
        _validate_binary("program", program)
        _validate_binary("input_bits", input_bits)
        tape: dict[int, str] = {0: SEPARATOR}
        for offset, bit in enumerate(program, start=-len(program)):
            tape[offset] = bit
        for offset, bit in enumerate(input_bits, start=1):
            tape[offset] = bit
        return tape

    def run(
        self,
        program: str,
        input_bits: str,
        *,
        t_max: int = 10_000_000,
    ) -> ScalarResult:
        _validate_binary("program", program)
        _validate_binary("input_bits", input_bits)
        if isinstance(t_max, bool) or not isinstance(t_max, int):
            raise TypeError("t_max must be an integer")
        if not 1 <= t_max <= UINT64_MAX:
            raise ValueError(f"t_max must be between 1 and {UINT64_MAX:,}")

        tape = self.initial_tape(program, input_bits)
        state = START
        head = 0
        steps = 0
        left_used = len(program)
        right_used = len(input_bits)
        reason = InvalidReason.NONE

        while state != HALT and steps < t_max:
            read = tape.get(head, BLANK)
            transition = self.table.transitions.get((state, read))
            if transition is None:
                reason = InvalidReason.INVALID_TRANSITION
                break
            if transition.write == BLANK:
                tape.pop(head, None)
            else:
                tape[head] = transition.write
            head += transition.move
            state = transition.next_state
            steps += 1
            left_used = max(left_used, -head)
            right_used = max(right_used, head)

        halted = state == HALT
        if reason == InvalidReason.NONE and not halted:
            reason = InvalidReason.TIMEOUT

        output = ""
        if halted and reason == InvalidReason.NONE:
            if tape.get(0, BLANK) != SEPARATOR:
                reason = InvalidReason.MISSING_SEPARATOR
            else:
                position = 1
                bits: list[str] = []
                while tape.get(position, BLANK) in {ZERO, ONE}:
                    bits.append(tape[position])
                    position += 1
                output = "".join(bits)
                if not output:
                    reason = InvalidReason.EMPTY_OUTPUT

        return ScalarResult(
            output=output,
            invalid=reason != InvalidReason.NONE,
            invalid_reason=reason,
            T=steps,
            left_space_used=left_used,
            right_space_used=right_used,
            halted=halted,
            final_state=state,
            final_head=head,
            final_tape=MappingProxyType(dict(tape)),
        )
