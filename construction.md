# LUTM-1 construction and latent-universality proof sketch

This document describes the fixed Turing machine implemented by LUTM-1, the
binary language used for its universal witness programs, and the reason the
same physical machine can represent arbitrary computable tasks while keeping
one input/output interface.

The transition table is the machine. Python, NumPy, and Taichi are three
simulators of that same table; they are not three different constructions.

## 1. What is being constructed?

Let

$$
\Sigma = \{0,1\}
$$

and let $U$ be the deterministic, bi-infinite, one-tape Turing machine stored
in `data/transition_table.csv`. It has:

```text
physical tape symbols:          16
states including HALT:         150
nonhalting states:             149
transition rows:             2,384 = 149 x 16
```

The transition function is total on every nonhalting state and every physical
symbol. `HALT` has no outgoing transition.

For each finite binary program $p$ and finite binary input $x$, the machine
starts from a literal, spatially separated configuration. If $|p|=m$ and
$|x|=n$, the initial tape $T_{p,x}:\mathbb Z\to\Gamma$ is:

$$
T_{p,x}(k)=
\begin{cases}
p[k+m], & -m\le k<0,\\
\#, & k=0,\\
x[k-1], & 1\le k\le n,\\
B, & \text{otherwise}.
\end{cases}
$$

The physical head starts at coordinate $0$ in state `START`:

```text
... B B p[0] ... p[m-1] # x[0] ... x[n-1] B B ...
                              ^
                      head, state START
```

Nothing outside the Turing machine translates the input into an internal
encoding. The input bits are placed directly on the tape, and every later
change is performed by the fixed transition table.

## 2. The external input/output contract

The initial separator is not a protected cell. During a valid simulated
computation it can temporarily become the auxiliary symbol `hL`, which means
that the simulated target-machine head is on its logical left-end cell. The
machine restores `#` on the canonical halting path.

The raw decoder $D_{\mathrm{raw}}$ reads the maximal binary prefix beginning
at coordinate $1$. If the final tape begins on the right as

```text
# 1 0 0 1 X 0 1 ...
```

then:

$$
D_{\mathrm{raw}}(T)=1001.
$$

The `X` and everything after it are ignored. Everything to the left of
coordinate $0$ is also ignored. The research simulators declare a computation
successful when:

1. the physical machine reaches `HALT`;
2. coordinate $0$ contains `#`;
3. the raw binary prefix is nonempty.

The final physical head position is deliberately not part of this runtime
contract. The canonical universal witnesses constructed below satisfy the
stronger property that the head returns to coordinate $0$, but arbitrary raw
programs do not need to do so.

For a raw program $p$, define its induced partial behavior by:

$$
\varphi_p(x)=D_{\mathrm{raw}}(U(p\#x))
$$

when the run satisfies the successful-halting conditions. Otherwise
$\varphi_p(x)$ is undefined.

The program is not required to remain unchanged. LUTM-1 is a single-tape
machine, so the negative program region is also ordinary writable memory.
Canonical simulation uses controlled temporary marks there, and arbitrary
programs may leave arbitrary residue.

## 3. Why an ordinary universal machine is not yet the whole story

Classical universality says that one machine can simulate every other Turing
machine after receiving a description of the simulated machine. For latent
universality, the separation of roles is also important:

- $U$ is fixed;
- $E(x)$ always writes the literal input in the same place;
- $D$ always reads output in the same way;
- the combining operation always forms $p\#x$;
- only $p$ changes with the desired task.

The relevant quantifier order is:

$$
\exists\,(U,E,D,\oplus)\;
\forall g\;
\exists p_g\;
\forall x.
$$

Here $g$ ranges over partial computable functions. The encoder and decoder are
chosen once, before the task and program. This prevents task-specific work
from being hidden in input preprocessing or output interpretation.

For machine learning, the separation means that $p$ can be treated as the
trainable variable. Enumeration or evolution may change program bits while
all examples continue to use exactly the same interface.

## 4. The canonical target machines

To establish universality, the physical machine only needs one universal
subset of its raw binary programs. LUTM-1 uses encodings of deterministic,
one-way, one-tape **target machines** with:

- logical alphabet $\{L,B,0,1\}$;
- logical cell $0$ permanently representing the left-end symbol $L$;
- initial logical tape $LxB^\infty$;
- target states $0,\ldots,n-1$, initially state $0$;
- target head initially on $L$;
- left, right, and stay moves;
- a distinguished halt destination.

A canonical target witness additionally obeys:

1. it preserves $L$;
2. it never writes $L$ on a positive data cell;
3. on successful halt, its tape is $LyB^\infty$;
4. on successful halt, its target head is on $L$.

These conditions are not checked by the physical LUTM. They define a
convenient class of programs used in the existence proof.

### Why this target class is universal

Start with any deterministic Turing machine computing a partial computable
binary function $g$.

Standard effective transformations can:

1. represent multiple tapes by tracks on one tape;
2. represent a two-sided tape on a one-way tape by interleaving negative and
   nonnegative coordinates;
3. encode every symbol of a finite work alphabet as a fixed-length binary
   block;
4. add a finite prologue that converts literal $LxB^\infty$ into the chosen
   internal work representation;
5. add a finite epilogue that, after simulated halting, extracts the answer,
   erases the finitely visited work region, writes $LyB^\infty$, and returns
   the head to $L$.

If the original computation diverges, the epilogue is never reached. All of
these transformations use only finite control. Therefore every partial
computable binary function has a canonical target-machine witness.

## 5. Binary serialization of a target machine

Each target state has four transition records, ordered by the symbol read:

```text
L, B, 0, 1
```

Each record is the concatenation:

```text
write-code  direction-code  next-code
```

The fields are encoded as follows:

| Field | Meaning | Binary code |
|---|---|---:|
| write | `L` | `00` |
| write | `B` | `01` |
| write | `0` | `10` |
| write | `1` | `11` |
| direction | left | `00` |
| direction | right | `01` |
| direction | stay | `10` |
| next | halt | `0` |
| next | state $r$ | $1^{r+1}0$ |

Direction code `11` is unused by canonical witnesses.

The write and direction fields have fixed length. The unary next-state field
ends at its first `0`. Four consecutive records form one state block. Thus a
finite target transition table has a finite, effectively constructible, and
uniquely parseable binary serialization:

$$
p_M=\operatorname{encode}(M).
$$

The physical machine does not begin by validating this grammar. Every finite
binary string is an operational raw program, although malformed programs may
time out or produce invalid output. This is useful for search: candidate
programs do not need to be generated by a compiler.

The five-bit identity program `00100` illustrates this relaxed behavior. Its
only reached path decodes as "write $L$, stay, halt". It does not contain all
four records of a complete state block, but it still returns every nonempty
input unchanged.

## 6. The physical alphabet and its markers

The physical alphabet is:

```text
B, 0, 1, #,
C0, C1, A0, A1,
N, X, T, Z,
hL, hb, h0, h1
```

The roles are:

| Symbols | Role |
|---|---|
| `B`, `0`, `1`, `#` | Literal blank, binary data, and separator |
| `C0`, `C1` | First bit of the currently active target-state block |
| `A0`, `A1` | Candidate state block during next-state resolution |
| `N`, `X`, `T`, `Z` | Temporary marks inside the unary next-state field |
| `hL` | Target head on the logical left-end cell |
| `hb`, `h0`, `h1` | Target head on a blank, zero, or one data cell |

The state marker preserves the bit it replaces: `C0` means that the original
program bit was `0`, and `C1` means it was `1`. The same principle applies to
`A0` and `A1`. This lets the interpreter use the program region as temporary
workspace without losing the serialized transition table along a normal
nonhalting macro-step.

## 7. The macro-step invariant

At the boundary between two simulated target transitions, the physical
configuration represents a target configuration $(q,h,t)$ when:

1. the physical state is `FIND_CURRENT_a`, where $a$ is the logical symbol
   currently under the target head;
2. the physical head is at coordinate $-1$;
3. exactly one `C0` or `C1` marks the first bit of target-state block $q$;
4. no temporary `A`, `N`, `X`, `T`, or `Z` marker remains;
5. if $h=0$, coordinate $0$ is `hL` and there is no positive head marker;
6. if $h>0$, coordinate $0$ is `#` and exactly coordinate $h$ contains
   `hb`, `h0`, or `h1`;
7. every other positive cell literally represents the target tape with `B`,
   `0`, or `1`.

This invariant is the bridge between one logical target step and many
physical LUTM transitions.

## 8. Initialization

Starting from $p\#x$:

1. `START` changes `#` into `hL`, marking the initial target-head position;
2. `INIT_FIND_PROGRAM_LEFT` crosses the finite program and finds the blank
   immediately to its left;
3. `INIT_MARK_STATE0` changes the first bit of state block $0$ into `C0` or
   `C1`;
4. `INIT_SEEK_HASH` returns to the logical boundary;
5. the machine enters `FIND_CURRENT_L` at coordinate $-1$.

The positive input cells are not moved or encoded. The resulting physical
configuration represents target state $0$, target head $0$, and target tape
$LxB^\infty$.

## 9. Simulating one target transition

One target transition is implemented by four conceptual phases.

### 9.1 Selecting the record

From coordinate $-1$, `FIND_CURRENT_a` moves left until it reaches the unique
`C0` or `C1` state marker. The read symbol $a\in\{L,B,0,1\}$ determines whether
zero, one, two, or three records must be skipped.

Each skipped record contains four fixed prefix bits followed by a unary field
ending in `0`. The `SKIP_*` states therefore cross exactly the required number
of complete records. `SELECT_W*` decodes the write symbol, and `SELECT_D*`
decodes the direction into finite control.

The next-state field is marked temporarily:

$$
\begin{aligned}
0 &\longrightarrow Z && \text{(halt)},\\
1^{r+1}0 &\longrightarrow N^{r+1}T && \text{(target state $r$)}.
\end{aligned}
$$

### 9.2 Applying the write and movement

If the target head is on $L$, the physical marker is `hL`. A canonical target
transition preserves $L$:

- a stay or left move leaves `hL` in place;
- a right move restores `#` and marks data cell $1$ as headed.

If the target head is on a data cell, `UPDATE_FIND_*` scans to the unique
`hb`, `h0`, or `h1` marker. It writes the decoded literal symbol and places a
new head marker on the left, same, or right cell. A left move from data cell
$1$ changes `#` into `hL`.

Thus the positive half-tape after this phase is exactly the target tape after
its write and move.

### 9.3 Resolving the next state

For a nonhalting next state $r$, the selected field contains $N^{r+1}T$. The
old `C0`/`C1` marker is restored, and target-state block $0$ becomes the
candidate `A0`/`A1`.

The resolver consumes `N` markers from right to left. Whenever it consumes an
`N` and another `N` remains, the `ADV_*` states move the candidate across
exactly four records, which is exactly one target-state block.

After the first $j$ marks have been consumed, with another mark still
remaining, the candidate is on state block $j$. After the final $(r+1)$-st
mark, no further advance occurs. The candidate is therefore exactly state
block $r$.

The machine then restores:

```text
X -> 1
T -> 0
A0/A1 -> C0/C1
```

and returns to the appropriate `FIND_CURRENT_a` state. The complete macro-step
invariant is re-established for the exact successor target configuration.

### 9.4 Why a macro-step is finite

Every sweep of the program crosses a finite binary program. Every unary field
is finite. A valid witness only names target states that exist. After any
finite number of target steps, the target head is at a finite coordinate.
Consequently each physical phase used to simulate one target step terminates
after finitely many physical transitions.

This construction is exact but not efficient. A logical transition can
require repeated scans across the program and the used data region. That is
why even simple arithmetic programs may take many physical transitions.

## 10. Halting and output

For a halt next-code, the selected `0` is marked `Z`. The interpreter applies
the target write and movement before resolving the halt.

By the canonical target convention, a successful target halt has:

```text
logical tape:  L y B B ...
target head:   on L
```

The physical tape therefore has `hL` at coordinate $0$ and literal output
$y$ on the positive side. `HALT_TO_BOUNDARY` returns through the residual
program region, changes `hL` back into `#`, and enters physical `HALT`. It does
not clean the negative tape.

The canonical path leaves the physical head at coordinate $0$, although the
repository's general raw-output contract does not require that head position.
The raw decoder returns exactly $y$.

For a nonhalting transition, the next-state field is restored and another
macro-step begins. Therefore an infinite target execution produces infinitely
many finite physical macro-steps and never reaches physical `HALT`.

## 11. Proof sketch of latent universality

We can now assemble the argument.

### Lemma A: canonical target witnesses are universal

For every partial computable binary function $g$, the standard normalization
described in Section 4 constructs a canonical target machine $M_g$ that:

- diverges when $g(x)$ is undefined;
- otherwise halts with logical tape $Lg(x)B^\infty$ and head on $L$.

### Lemma B: witness serialization is effective

Section 5 gives a finite, computable, uniquely parseable binary program:

$$
p_g=\operatorname{encode}(M_g).
$$

### Lemma C: initialization is correct

Section 8 establishes the macro-step invariant for the initial target
configuration of $M_g$ on $x$.

### Lemma D: one macro-step is exact

Assume the invariant represents a nonhalting target configuration $c$.
Record selection chooses exactly the transition for the state and scanned
symbol of $c$. The data phase applies exactly its write and movement. Unary
resolution selects exactly its successor state and restores the invariant.
Thus finitely many physical transitions produce the representation of the
next target configuration $c'$.

### Lemma E: halting and divergence are preserved

If $M_g(x)$ halts canonically with output $y$, the physical halt phase restores
`#` and the raw decoder returns $y$. If $M_g(x)$ runs forever, induction over
its target steps yields infinitely many finite physical macro-steps and no
physical halt.

### The result

Combining the lemmas, for every partial computable function in the represented
output convention there is a finite program $p_g$ such that:

$$
\varphi_{p_g}(x)=g(x)
$$

whenever $g(x)$ is defined, while both sides are undefined when it diverges.

Conversely, every $\varphi_p$ is partial computable: $U$ is one fixed finite
deterministic Turing machine, and the successful-halting test and prefix
decoder are computable. Therefore the program family does not exceed the
partial computable functions.

The crucial latent-universality property is that $U$, input placement,
separator placement, halting rule, and output decoding are fixed. Only the
finite program $p_g$ depends on $g$.

## 12. The empty-output convention

The conventional set $\Sigma^*$ contains the empty string $\epsilon$. The
current LUTM-1 runtime deliberately declares an empty raw output prefix
invalid. Its literal raw interface therefore directly represents partial
computable functions whose defined outputs lie in:

$$
\Sigma^+ = \Sigma^*\setminus\{\epsilon\}.
$$

This does not reduce computational universality under a fixed representation.
Use the total, injective output code:

$$
\eta(y)=1y.
$$

Every encoded output is nonempty, including:

$$
\eta(\epsilon)=1.
$$

A canonical target for $g$ writes $\eta(g(x))$ instead of $g(x)$. A fixed
outer decoder removes the initial `1`:

$$
D_{\mathrm{std}} = \eta^{-1}\circ D_{\mathrm{raw}}.
$$

The code and its inverse are independent of $g$, $p_g$, and $x$, so the
required latent-universality quantifier order is unchanged. The software in
this repository exposes $D_{\mathrm{raw}}$ directly; users who need semantic
empty outputs must apply a fixed convention of this kind to their targets and
decoded results.

## 13. What the proof does and does not establish

The construction establishes:

- one fixed finite transition table;
- literal, program-independent binary input placement;
- fixed halting and output conventions;
- a finite program for every partial computable task, under the fixed output
  representation;
- exact preservation of target halting and divergence.

It does not establish:

- that useful programs are short;
- that the simulation overhead is small;
- that enumeration or a genetic algorithm will find a witness in practical
  time;
- that malformed raw programs behave cleanly;
- that finite tests alone prove an unbounded theorem;
- that this software implementation is free from bugs.

The unbounded claim rests on the parametric simulation invariant and induction
over target steps. The repository tests provide implementation evidence: they
check the transition table, known computations, boundaries, enumeration,
independent simulator agreement, GA operators, and large batches. Those tests
are complementary to, but not a replacement for, the construction argument.

## 14. Canonical artifact

The fixed physical rule is:

```text
data/transition_table.csv
SHA-256 260fb1d15014523c6a46ede09f9fcd7cd7d591f668912842f1aae8cf993376f9
```

The CSV's exact bytes are preserved in Git as a binary artifact because its
raw hash identifies this particular machine. A different hash must be treated
as a different transition table and audited accordingly.
