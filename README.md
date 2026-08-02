# LUTM-1

LUTM-1 is a standalone research environment for a fixed **Latent Universal
Turing Machine**. The transition rule never changes: evaluation, exhaustive
enumeration, and evolutionary search vary only the literal binary program on
the tape.

It was developed alongside *Emergent Models: Intelligence from Tiny
Substrates* as a practical substrate for studying latent universality and
program synthesis under a fixed machine rule.

The repository contains one readable scalar reference simulator, matching
bounded NumPy and Taichi/CUDA backends, a length-first program enumerator, a
Taichi-evaluated island genetic algorithm, and a register of known programs.

## Machine interface

Every computation starts with the physical head at coordinate `0` in state
`START`:

```text
... B B p # x B B ...
          ^
     coordinate 0
```

`p` is the binary program on negative coordinates, `#` is fixed initially at
coordinate `0`, and `x` is the binary input on positive coordinates. The
machine performs the complete computation itself. There is no preprocessing
of the tape and no requirement that the program survive the computation.

A successful result has all of these properties:

- the physical machine reaches `HALT`;
- `#` is present at coordinate `0` when it halts;
- the maximal binary prefix beginning at coordinate `1` is nonempty.

That prefix is the output. The final head location, the entire left side, and
any symbols after the output prefix are intentionally ignored. A transition
writes, moves, changes state, and then increments `T`. A halt reached on
transition `T_max` succeeds; an unhalted machine after exactly `T_max`
transitions times out.

The scalar simulator uses a self-extending, bi-infinite dictionary tape. The
NumPy and Taichi backends use a finite tape with independently configurable
left and right budgets. Both endpoint cells are usable; crossing an endpoint
fails after the crossing transition has been counted. Space usage includes
the initial program and input as well as every position visited by the head,
including visited blank cells.

## Windows setup

From PowerShell, create an environment and install the pinned computational
dependencies:

```powershell
conda create -n lutm1 python=3.10 -y
conda activate lutm1
Set-Location LUTM-1
python -m pip install -r .\requirements.txt
```

The GPU backend is deliberately strict: it requires an NVIDIA CUDA device and
does not fall back to CPU execution. If you already use the prepared
`slackenv` environment, activate it instead.

Start Jupyter with:

```powershell
jupyter notebook
```

The four notebooks are clean, editable research entry points:

1. `01_python_evaluator.ipynb` — unbounded scalar execution and registered
   task verification;
2. `02_numpy_simulator_and_enumerator.ipynb` — bounded NumPy execution,
   input/target evaluation, and growing-length enumeration;
3. `03_taichi_simulator_and_enumerator.ipynb` — the same bounded workflow on
   CUDA;
4. `04_temperature_island_ga.ipynb` — the Taichi-evaluated, NumPy-evolved
   island GA with live training output.

Run notebooks from the repository root and select the environment containing
the installed requirements.

## Minimal scalar run

```python
from lutm import ScalarUTMSimulator
from programs import get_program

task = get_program("plus_one")
result = ScalarUTMSimulator().run(task.program, "1011", t_max=10_000_000)

print(result.output)          # 1100
print(result.T)
print(result.invalid_reason)
```

The scalar transition counter is a Python integer and accepts a timeout up to
`2**64 - 1`. It has no space-failure mode.

## NumPy batches and enumeration

Bounded backends use fixed-width arrays. An effective program is right-aligned
next to `#` and padded on the left with physical blanks. At width three the
length-first enumeration is:

```text
BBB, BB0, BB1, B00, B01, B10, B11,
000, 001, 010, 011, 100, 101, 110, 111
```

Leading zeroes and the empty program are distinct valid candidates.

```python
from numpy_backend import NumpyUTMSimulator, find_exact_program
from utils import SimulatorConfig, TaskCases

task = TaskCases(
    inputs=["0", "1", "01", "10"],
    targets=["0", "1", "01", "10"],
)
simulator = NumpyUTMSimulator(
    SimulatorConfig(
        program_width=5,
        left_budget=7,
        right_budget=8,
        t_max=1_000,
    )
)
solution = find_exact_program(
    simulator,
    task,
    batch_size=64,
    stop_first_exact=True,
    print_every_batches=1,
)
print(solution)
```

Ordinal enumeration is limited to program widths `1..62` so the entire
length-first ordinal space is representable safely. Explicit program batches
may be wider.

## Taichi/CUDA

Programs are parallelized on the GPU; task inputs are evaluated serially. A
single kernel launch runs each physical machine until halt, failure, or
`T_max`. There is no transition chunking.

```python
from taichi_backend import TaichiUTMSimulator, initialize_taichi_cuda
from utils import SimulatorConfig

initialize_taichi_cuda()
gpu = TaichiUTMSimulator(
    SimulatorConfig(80, 88, 24, 20_000),
    batch_capacity=40_000,
)
programs = gpu.encode_programs(["00100"])
result = gpu.simulate_program_batch(programs, "101110")
print(result.output_strings(), result.T)
```

The NumPy and Taichi transition counters use `uint32`; `T_max` may be at most
`4,294,967,295`. Head positions and the two directional space measurements
use `int32`.

## Island genetic algorithm

The GA evolves independent islands in vectorized NumPy and sends only physical
program evaluation to Taichi/CUDA. Fitness is

```text
normalized positional bit accuracy
    - k_penalty * invalid-case fraction
```

Missing and extra output bits are mismatches. Tournament size is always two.
The probability of choosing the fitter contender follows configurable
fitness-delta intervals; exact ties are sampled 50/50. Elitism prefers new
children on equal fitness, encouraging neutral movement without an additional
diversity metric.

Each child independently receives segment crossover. Its effective length is
inherited from parent 1, and one or more uniformly sampled parent-2 segments
are copied inside the shared binary suffix. Segments may overlap. Mutation
then draws one mutually exclusive structural action—insert, delete, or stay—
and independently flips every effective bit with the configured Bernoulli
probability. Islands periodically exchange elites and receive random
immigrants.

See `04_temperature_island_ga.ipynb` for the complete editable trainer.

## Registered programs

`programs.py` is the immutable program/task register. Each literal is checked
against its declared SHA-256 digest when imported, and each computational task
has an input contract and an oracle.

| Name | Program bits | Behavior | Input contract |
|---|---:|---|---|
| `identity_short` | 5 | identity | nonempty binary string |
| `identity_complete` | 23 | identity | nonempty binary string |
| `bit_not` | 80 | pointwise bit NOT | nonempty binary string |
| `jump_probe` | 76 | identity, forward-jump probe | nonempty binary string |
| `jump_to_zero_probe` | 76 | identity, backward-jump probe | nonempty binary string |
| `plus_one` | 176 | canonical binary `x + 1` | `0` or leading-one integer |
| `times_two` | 114 | canonical binary `2x` | `0` or leading-one integer |
| `square` | 4,605 | canonical binary `x²` | `0` or leading-one integer |
| `divergent_loop` | 24 | timeout diagnostic | any binary string |

The arithmetic literals are length-independent algorithms, not lookup tables.
The square program is computationally expensive and is verified only through
the scalar and NumPy implementations, never through the GPU test suite.

## Verification

Run the complete standalone suite from the repository root:

```powershell
conda activate slackenv
python .\run_tests.py
```

The suite validates the canonical CSV and program hashes, scalar semantics,
input and padding rules, growing-length enumeration, NumPy/Taichi differential
behavior, known programs, GA operators and reproducibility, bounded failures,
and one 40,000-program CUDA stress run. The long square verification counts
every physical transition and can take appreciable time.

The authoritative transition table is
`data/transition_table.csv`:

```text
SHA-256 260fb1d15014523c6a46ede09f9fcd7cd7d591f668912842f1aae8cf993376f9
150 states (including HALT), 16 symbols, 2,384 transition rows
```

Changing this file changes the machine. The loader rejects an altered table.

## Repository layout

```text
data/transition_table.csv       canonical fixed rule
data/square_program.txt         verified long square literal
lutm.py                         scalar simulator and table loader
programs.py                     known programs, contracts, and oracles
utils.py                        tasks, bounds, padding, and enumeration
numpy_backend.py                bounded NumPy simulator and enumerator
taichi_backend.py               strict-CUDA simulator and enumerator
island_ga.py                    NumPy island GA with Taichi evaluation
tests/                          standalone semantic and stress tests
run_tests.py                    complete test entry point
```

## License

[MIT](LICENSE), copyright 2026 emergentcomputing.
