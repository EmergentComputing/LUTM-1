"""Static register of verified LUTM programs and their mathematical oracles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable


Oracle = Callable[[str], str]
InputContract = Callable[[str], bool]


def _binary(bits: str) -> bool:
    return isinstance(bits, str) and all(bit in "01" for bit in bits)


def _nonempty_binary(bits: str) -> bool:
    return bool(bits) and _binary(bits)


def _canonical_integer(bits: str) -> bool:
    return bits == "0" or (
        bool(bits) and bits[0] == "1" and _binary(bits)
    )


def _identity(bits: str) -> str:
    return bits


def _bit_not(bits: str) -> str:
    return "".join("1" if bit == "0" else "0" for bit in bits)


def _plus_one(bits: str) -> str:
    return format(int(bits, 2) + 1, "b")


def _times_two(bits: str) -> str:
    return format(int(bits, 2) * 2, "b")


def _square(bits: str) -> str:
    value = int(bits, 2)
    return format(value * value, "b")


@dataclass(frozen=True)
class RegisteredProgram:
    name: str
    description: str
    program: str
    oracle: Oracle | None
    accepts: InputContract
    logical_sha256: str

    def __post_init__(self) -> None:
        if not self.program or any(bit not in "01" for bit in self.program):
            raise ValueError(f"registered program {self.name!r} is not binary")
        actual = hashlib.sha256(self.program.encode("ascii")).hexdigest()
        if actual != self.logical_sha256:
            raise ValueError(f"registered program {self.name!r} hash mismatch")

    def target(self, input_bits: str) -> str:
        if not self.accepts(input_bits):
            raise ValueError(
                f"input {input_bits!r} is outside {self.name!r}'s contract"
            )
        if self.oracle is None:
            raise ValueError(f"registered program {self.name!r} has no oracle")
        return self.oracle(input_bits)


IDENTITY_COMPLETE = "00100011010101010111010"
IDENTITY_SHORT = "00100"
BIT_NOT = (
    "0001110011010101010111010001010010011101101110100111000100011010"
    "1000111011001110"
)
JUMP_PROBE = (
    "0001111001101010101011101000100011010101010111010001010010011010"
    "001101100110"
)
JUMP_TO_ZERO_PROBE = (
    "0001111001001101000110110011000100011010101010111010001010011010"
    "101010111010"
)
DIVERGENT_LOOP = "001010011010101010111010"
PLUS_ONE = (
    "0001110011010101010111010001010010011101001110110111000011111100"
    "1101011001111010001110001000110101000111101100111100010100110101"
    "101111111011101000101010001111010011111110111010"
)
TIMES_TWO = (
    "0001110011010101010111010001010011010100011110110111100010101000"
    "11110100111101101111000100011010100011110110011110"
)


def _load_square() -> str:
    path = Path(__file__).resolve().parent / "data" / "square_program.txt"
    literal = path.read_text(encoding="ascii").strip()
    if len(literal) != 4_605:
        raise ValueError("square program length mismatch")
    return literal


SQUARE = _load_square()


PROGRAMS = {
    item.name: item
    for item in (
        RegisteredProgram(
            "identity_short",
            "Five-bit raw identity witness.",
            IDENTITY_SHORT,
            _identity,
            _nonempty_binary,
            "83dc8ab9acc8e45c2d2dec3e34a08e2c1e84c91cef7262fb086f0192040fa4cf",
        ),
        RegisteredProgram(
            "identity_complete",
            "Complete serialized identity target machine.",
            IDENTITY_COMPLETE,
            _identity,
            _nonempty_binary,
            "568e0e3e2ae08195f97cb8e98650cf41f6971ad3fe75846f8c71ce66ff95e8e9",
        ),
        RegisteredProgram(
            "bit_not",
            "Flip every input bit.",
            BIT_NOT,
            _bit_not,
            _nonempty_binary,
            "b90b4c9b53c00056c8e65c01076cbab1f629433074a60024830212a73bbe0144",
        ),
        RegisteredProgram(
            "jump_probe",
            "Identity witness exercising a forward target-state jump.",
            JUMP_PROBE,
            _identity,
            _nonempty_binary,
            "1dea123f123adef62ac617606d22ea7fe71ef91a3d7ab8c9a52128831462d08a",
        ),
        RegisteredProgram(
            "jump_to_zero_probe",
            "Identity witness exercising a jump back to target state zero.",
            JUMP_TO_ZERO_PROBE,
            _identity,
            _nonempty_binary,
            "20f33ed3917dd21b638b6ac4ebb72a116aae600bf244ed559b4d19602d106e8c",
        ),
        RegisteredProgram(
            "plus_one",
            "Canonical binary successor x -> x + 1.",
            PLUS_ONE,
            _plus_one,
            _canonical_integer,
            "12e4043e2a8331fe662ba1c1084aab18723e1b4954b81e16686e0e9dbe491c77",
        ),
        RegisteredProgram(
            "times_two",
            "Canonical binary doubling x -> 2*x.",
            TIMES_TWO,
            _times_two,
            _canonical_integer,
            "ffeb4ed574daa0a024af409da31aea00c70a9c3757b10108ed95397de9b7e6be",
        ),
        RegisteredProgram(
            "square",
            "Arbitrary-length canonical binary square x -> x^2.",
            SQUARE,
            _square,
            _canonical_integer,
            "8420dfe17f776a8c4eba33909bdce8fe7a339fcba89258b4512618851e484413",
        ),
        RegisteredProgram(
            "divergent_loop",
            "Deliberately divergent timeout diagnostic.",
            DIVERGENT_LOOP,
            None,
            _binary,
            "dbcc8dbc154f34da6397efde91c3c7c57953a8845fcc7a08a5612583621e3dd2",
        ),
    )
}


def get_program(name: str) -> RegisteredProgram:
    try:
        return PROGRAMS[name]
    except KeyError as error:
        choices = ", ".join(PROGRAMS)
        raise KeyError(f"unknown program {name!r}; choose from {choices}") from error
