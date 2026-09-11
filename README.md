# LUTM-1

LUTM-1 is a fixed one-tape Turing machine designed to be a **Latent Universal
Turing Machine**. Its central idea is simple: the machine and its input/output
interface stay the same for every task. Only the program changes.

```text
program p + input x  ->  p#x  ->  fixed LUTM  ->  residue#y
```

Here `p` specifies the algorithm, `x` is the input data, and `y` is the output.
All three are ordinary binary strings. A program for addition and a program
for a sequence operation therefore receive their inputs in exactly the same
way and use exactly the same output decoder.

This repository grew out of the latent-universality idea in
[*Emergent Models: Intelligence from Tiny Substrates*](https://arxiv.org/abs/2608.14019).
It is intended as a concrete machine for studying that idea, not as a claim
that universal programs are easy to find or that this implementation is
competitive with conventional programming or machine learning systems.

## Classical and latent universality

Classical universality says that one fixed Turing machine can simulate any
other Turing machine when given a description of that machine and its data. It
is primarily a statement about computational power: one machine can perform
every computable task by encoding together program AND input.

Latent universality keeps that goal but adds a practical interface principle:
the algorithm and the data protocol are separate.

- The program `p` is the part that selects the algorithm.
- The input `x` is always placed literally to the right of `#`.
- The output `y` is always read from that same location by the same rule.

Changing the task means changing `p`; it does not mean changing the physical
machine, the tape layout, or the decoder. In this sense, LUTM-1 provides a
universal binary interface. Any finite kind of data can be serialized as bits,
then supplied through the same protocol. Structured data still needs a binary
representation, but the LUTM itself does not need a new datatype-specific
input or output mechanism.

As explained in
[*Emergent Models: Intelligence from Tiny Substrates*](https://arxiv.org/abs/2608.14019),
the program carries the semantics of the algorithm: it specifies the general
transformation from the input space to the output space. The fixed transition
function and the encoding and decoding rules provide the syntax. They define
how programs and data are placed, executed, and read, but they do not encode
which task-specific transformation should be performed. In this precise sense,
all task-specific computational meaning resides in `p`.

The transition function still performs the physical computation step by step.
Calling it syntactic means that it is task-independent: the same rule executes
identity, arithmetic, and every other program. Only `p` determines which of
those transformations the run represents.

This separation is especially useful for program search. Enumeration or a
learning algorithm can try many programs against the same input-target format
without rebuilding the execution interface for each candidate or task. The
program varies; the substrate and the meaning of `p#x` do not.

Latent universality is not a stronger class of computability than classical
Turing universality. The term highlights this fixed-interface organization and
the fact that the algorithm is stored as a variable pattern in the initial
tape rather than in the transition table.

## Status of this machine

Latent universality itself is a proved general theorem. The unresolved question
in this repository is narrower: whether this particular transition table,
LUTM-1, is a correct concrete realization of that theorem.

LUTM-1 was constructed to be Turing-complete through the fixed interface above.
[`construction.md`](construction.md) gives an explicit simulation construction
and a detailed proof sketch. The machine has also executed many compiled test
cases, including length-independent arithmetic programs, with consistent
results across independent simulator implementations.

The specific LUTM-1 construction has not yet been established by a complete,
independently checked mathematical proof or a machine-checked formalization.
It should therefore be regarded as a strongly supported candidate latent
universal machine, rather than a formally certified instance. This qualification
concerns LUTM-1 only; it does not concern the validity of the general latent
universality theorem. Computational tests strengthen confidence in the
construction, but finitely many successful programs cannot prove that this
particular machine is universal.

## The `p#x` tape interface

Let `p` be a binary program and `x` a binary input. Every run begins as:

```text
... B B p # x B B ...
          ^
     coordinate 0
```

More explicitly:

- `p` occupies the negative tape coordinates;
- `#` is at coordinate `0`;
- `x` begins at coordinate `1`;
- every other cell is blank (`B`);
- the physical head begins on `#` in state `START`.

For example, program `00100` on input `1011` starts as:

```text
... B B 0 0 1 0 0 # 1 0 1 1 B B ...
```

The machine may overwrite the program, the separator, the input, or any other
visited tape cell while it computes. There is no external preprocessing and
no requirement that the program survive unchanged.

When the machine reaches `HALT`, the decoder checks that `#` is again at
coordinate `0`, then reads the consecutive `0` and `1` symbols immediately to
its right. It stops at the first other symbol. The left side of the tape, the
final head position, and later scratch symbols are ignored.

```text
... arbitrary residue # 1 0 0 1 X arbitrary residue ...
                        \_______/
                         output 1001
```

In the current research API, a missing separator or an empty output prefix is
reported as an invalid computation. Timeouts and bounded-tape overflows are
also explicit failures.

## Why this is useful for machine learning

Suppose we have examples of a target behavior:

```text
input  -> target
0      -> 1
1      -> 0
01     -> 10
10     -> 01
```

Because input placement, execution, and output decoding never change, a search
algorithm can evaluate many candidate programs under exactly the same
conditions. A program can be scored by how closely its outputs match the
targets, then changed by operations such as bit flips, insertion, deletion,
or crossover.

This turns learning into **program synthesis over initial tape states**:

```text
candidate program -> fixed LUTM -> outputs -> fitness
          ^                                  |
          |________ mutation/selection ______|
```

The universality result says that sufficiently expressive finite programs
exist in principle. It does **not** say that evolutionary search will find
them efficiently. Searchability is an empirical research question, and long
programs can produce extremely expensive physical computations.

## What is included

The repository provides three implementations of the same physical machine:

| Implementation | What it does | Main use |
|---|---|---|
| Scalar Python | Runs one program and one input, one transition at a time, on a self-extending tape | Readable reference and long computations |
| NumPy | Runs batches on a finite padded tape | CPU evaluation and exhaustive enumeration |
| Taichi/CUDA | Runs many independent programs in parallel on the GPU | Fast enumeration and GA fitness evaluation |

"Scalar", "NumPy", and "Taichi" describe the simulators, not different
Turing machines. They load the same fixed transition table and use the same
`p#x` semantics. The bounded implementations can additionally fail when their
configured tape or time budget is exceeded.

The repository also contains:

- growing-length enumeration of raw binary programs;
- a NumPy island genetic algorithm evaluated by Taichi/CUDA;
- known programs for identity, bitwise NOT, `+1`, multiplication by two, and
  squaring;
- differential, arithmetic, boundary, GA, and stress tests.

For the internal construction and the universality argument, see
[construction.md](construction.md).

## Start on Windows

The project was developed with Python 3.10. From PowerShell:

```powershell
conda create -n slackenv python=3.10 -y
conda activate slackenv
Set-Location LUTM-1
python -m pip install -r .\requirements.txt
```

The Taichi backend requires an NVIDIA CUDA device and intentionally has no CPU
fallback. The scalar and NumPy implementations do not require CUDA.

Start Jupyter with:

```powershell
jupyter notebook
```

Then open one of these notebooks:

1. [`01_python_evaluator.ipynb`](01_python_evaluator.ipynb) introduces the
   unbounded, nonparallel simulator and runs registered programs.
2. [`02_numpy_simulator_and_enumerator.ipynb`](02_numpy_simulator_and_enumerator.ipynb)
   runs padded NumPy batches and performs growing-length enumeration.
3. [`03_taichi_simulator_and_enumerator.ipynb`](03_taichi_simulator_and_enumerator.ipynb)
   provides the corresponding CUDA workflow.
4. [`04_temperature_island_ga.ipynb`](04_temperature_island_ga.ipynb) is the
   full island-GA research trainer with editable hyperparameters and live
   progress output.

Run the notebooks from the repository root and select the Python environment
in which the requirements were installed.

## Live training control pad

The control pad runs the same latest temperature island GA used by the fourth
notebook and the program miner; all three import the single implementation in
[`island_ga.py`](island_ga.py). Double-click [`run.bat`](run.bat) to activate
the documented `slackenv` Conda environment, start the local server, and open
the interface in the default browser. Equivalently, from an already activated
environment:

```powershell
python .\control_pad.py
```

Set the datasets and structural parameters before starting a run. Structural
parameters remain locked after initialization; editable training parameters
are applied together at the next generation boundary. Training can be paused,
continued, stopped, or saved manually. Reopening a run by its ID restores its
population, random-number state, configuration history, datasets, and recorded
diagnostics, then leaves it paused until explicitly continued.

After **Stop & save** completes, **New run** unloads the stopped session and
restores editable setup fields with a fresh run ID. The saved checkpoint is
retained and can still be loaded later.

The default diagnostic interval is 10 generations. Diagnostic generations
report global and per-island performance, Q20/Q80 bands across island
champions, and invalid rates for both the champions and the complete
population. Time and space statistics use halted island champions only, with
space defined as left space plus right space. Test cases are diagnostic only
and never affect selection.

Each run is stored as `runs/<run-id>/latest.npz`. The file is atomically
replaced at each checkpoint, so a run ID always refers to one latest complete
checkpoint. The `runs` directory is local research state and is ignored by
Git.

## Endless program miner

[`program_miner.py`](program_miner.py) repeatedly samples a task uniformly
with replacement, starts a fresh island population, and writes every valid
program above that task's accuracy threshold to a deduplicated SQLite
register. Mined programs are never used to seed later runs.

Edit [`miner_config.json`](miner_config.json) for global defaults and
[`miner_tasks.json`](miner_tasks.json) for input-target pairs and per-task
simulator, GA, generation, and threshold overrides. Then run:

```powershell
conda activate slackenv
python .\program_miner.py
```

Press `Ctrl+C` to finish the active run record, close SQLite, and exit cleanly.
Use `--max-runs 1` for one finite episode. To recover the task and first run
associated with a program:

```powershell
python .\program_miner.py --find 00100
```

Show global statistics, per-task program and exact-solution counts, tasks that
admit an exact mined program, and the ten most recent runs:

```powershell
python .\program_miner.py --stats
python .\program_miner.py --stats --recent 25
```

The default database is `mined_programs/programs.sqlite3`. Programs are
bit-packed while preserving their exact lengths and leading zeros; the local
database directory is ignored by Git.

## A minimal computation

The following runs the registered successor program on binary `1011` (decimal
11):

```python
from lutm import ScalarUTMSimulator
from programs import get_program

task = get_program("plus_one")
result = ScalarUTMSimulator().run(
    task.program,
    "1011",
    t_max=10_000_000,
)

print(result.output)          # 1100
print(result.T)               # physical transition count
print(result.invalid_reason)  # NONE on success
```

The scalar tape extends as needed in either direction. It has a time limit but
no artificial space limit.

## Program enumeration

For batching, programs occupy a fixed-width region and are padded on the left
with physical blanks. At width three, enumeration proceeds by increasing
effective length:

```text
BBB
BB0
BB1
B00
B01
B10
B11
000
001
010
011
100
101
110
111
```

Thus the empty program, leading-zero programs, and all ordinary binary strings
are distinct candidates. The number of candidates through width $L_p$ is:

$$
N(L_p) = \sum_{k=0}^{L_p} 2^k = 2^{L_p+1}-1.
$$

Ordinal enumeration is limited to width 62 by its integer representation.
Explicitly supplied program batches may be wider.

## Evolutionary search

The island GA keeps evolution on the CPU in vectorized NumPy and evaluates the
physical computations on the GPU. Each island performs two-member tournament
selection. The probability of choosing the fitter parent depends on the
fitness difference, so small differences permit more exploratory selection.

Children can receive:

- segment crossover while inheriting the first parent's effective length;
- insertion of a bit immediately to the left of the program;
- deletion of the program's leftmost effective bit;
- independent Bernoulli bit flips.

Equal-fitness children are preferred over older elites, allowing neutral
movement. Islands periodically exchange candidates and receive random
immigrants. The notebook exposes all probabilities and intervals directly.

Fitness is based on normalized positional bit accuracy with an additional
penalty for invalid computations. Missing and extra output positions count as
mismatches.

## Known programs

[`programs.py`](programs.py) contains an immutable register of verified binary
programs, input contracts, and ordinary Python functions that compute their
expected results.

| Name | Bits | Behavior |
|---|---:|---|
| `identity_short` | 5 | Return the input unchanged |
| `identity_complete` | 23 | Complete serialized identity witness |
| `bit_not` | 80 | Flip every input bit |
| `jump_probe` | 76 | Identity while exercising a forward state jump |
| `jump_to_zero_probe` | 76 | Identity while jumping back to state zero |
| `plus_one` | 176 | Canonical binary successor |
| `times_two` | 114 | Canonical binary multiplication by two |
| `square` | 4,605 | Canonical binary squaring |
| `divergent_loop` | 24 | Deliberate nonhalting diagnostic |

The arithmetic programs are length-independent algorithms rather than lookup
tables. Squaring is physically very expensive; the test suite evaluates it on
the scalar/CPU path and does not run it through Taichi.

## Tests and reproducibility

Run the complete suite from the repository root:

```powershell
conda activate slackenv
python .\run_tests.py
```

The suite checks:

- the transition-table and program hashes;
- the exact initial tape and decoder behavior;
- timeout and left/right boundary semantics;
- exhaustive short-program agreement between scalar Python, NumPy, and
  Taichi;
- identity, bitwise NOT, `+1`, multiplication by two, and square programs;
- enumeration order and exact-match stopping;
- GA mutation, crossover, selection, migration, neutrality, and
  reproducibility;
- one 40,000-program CUDA stress run.

The canonical physical rule is [`data/transition_table.csv`](data/transition_table.csv):

```text
SHA-256 260fb1d15014523c6a46ede09f9fcd7cd7d591f668912842f1aae8cf993376f9
150 states including HALT
16 tape symbols
2,384 transition rows
```

The CSV is intentionally preserved byte-for-byte by `.gitattributes` because
the loader checks its raw hash. Changing it means changing the machine.

## Repository map

```text
README.md                       introduction and usage
construction.md                 machine construction and proof sketch
data/transition_table.csv       canonical fixed machine rule
data/square_program.txt         verified long square program
lutm.py                         scalar simulator and table loader
programs.py                     known programs and expected functions
utils.py                        tasks, bounds, padding, and enumeration
numpy_backend.py                bounded NumPy simulator
taichi_backend.py               strict-CUDA simulator
island_ga.py                    NumPy island GA with Taichi evaluation
control_pad_backend.py          live diagnostics and resumable GA sessions
control_pad.py                  local control-pad server
control_pad.html                control-pad interface
run.bat                         Windows control-pad launcher
program_miner.py                endless independent island-GA miner
miner_config.json               miner defaults and database location
miner_tasks.json                tasks and per-task overrides
tests/                          verification and stress tests
run_tests.py                    complete test entry point
```

## Scope

The construction establishes an expressivity result: a fixed machine and
fixed interface can represent arbitrary computable behavior through finite
programs. It does not establish efficient simulation, efficient training,
robustness to malformed programs, or an absence of implementation bugs.

The current runtime also treats an empty raw output as invalid. Full
universality over binary strings including the empty string is recovered by a
fixed outer output code; this point is explained precisely in
[construction.md](construction.md#the-empty-output-convention).

## Development note

> [!NOTE]
> This project was developed with substantial AI assistance, particularly from
> GPT-5.6-Sol through Codex, and the construction document was produced by AI.
> The implementation has undergone extensive automated testing, including
> several nontrivial compiled programs such as arbitrary-length binary
> squaring, but bugs or unintended behavior may remain. The project was built
> and tested on Windows and has not yet been tested on Linux. Corrections and
> bug reports are welcome through
> [GitHub Issues](https://github.com/EmergentComputing/LUTM-1/issues).

## License

[MIT](LICENSE), copyright 2026 emergentcomputing.
