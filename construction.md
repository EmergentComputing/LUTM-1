# How LUTM-1 works

This document explains the construction of LUTM-1 and gives a proof sketch of
its latent universality. It begins with the concrete execution model, then
describes the internal markers and, finally, states the formal argument.

> [!CAUTION]
> The construction presented here was produced entirely by AI and has not been
> formally verified. During development, the system successfully produced and
> executed several nontrivial programs—including arbitrary-length binary
> squaring—but the construction may still contain errors, and parts of its
> explanation may be difficult to understand. Please report suspected problems
> through [GitHub Issues](https://github.com/EmergentComputing/LUTM-1/issues).

The fixed transition table in
[`data/transition_table.csv`](data/transition_table.csv) is the actual
Turing machine. The Python, NumPy, and Taichi code only simulates that table;
it does not preprocess the tape, repair programs, or perform part of a
computation outside the machine.

## The construction in one paragraph

LUTM-1 is a fixed one-tape Turing machine, called $U$. A binary program
$p$ is placed to the left of a separator, and a literal binary input $x$
is placed to its right:

```text
... B B p # x B B ...
          ^
       physical head
```

For the canonical universal programs, $p$ is a serialized transition table
of another, simpler Turing machine $M$. LUTM-1 repeatedly scans $p$ to find
the transition of $M$ that should run next, updates the simulated tape on
the right, marks the new simulated state inside $p$, and repeats. When $M$
halts in the required output form, LUTM-1 restores the separator and halts.
The bits immediately to the right of the separator are then the output.

The construction therefore has three distinct layers:

| Layer | What it is |
|---|---|
| Physical machine | The fixed 150-state LUTM-1 transition table |
| Program | A finite binary description stored on the physical tape |
| Simulated machine | The target machine described by a canonical program |

Only the program changes from one task to another.

## 1. The fixed external interface

Let $B$ be the blank symbol and let $S$ denote the physical separator
written as `#`. For a program of length $m$ and an input of length $n$,
the initial tape is the function $T_{p,x}:\mathbb Z\to\Gamma$ defined by

$$
T_{p,x}(k)=
\begin{cases}
p[k+m], & -m\leq k<0,\\
S, & k=0,\\
x[k-1], & 1\leq k\leq n,\\
B, & \text{otherwise}.
\end{cases}
$$

The physical head starts at coordinate $0$, on the separator, in state
`START`. No external encoder transforms $x$: its bits are copied directly
onto the tape.

### Output decoding

After physical `HALT`, the runtime requires the separator to be back at
coordinate $0$. It starts at coordinate $1$ and reads the longest
consecutive sequence of `0` and `1` symbols. Decoding stops at the first
different symbol.

```text
... arbitrary residue # 1 0 0 1 X arbitrary residue ...
                        \_______/
                         output 1001
```

The final head position and everything to the left of the separator are
ignored. The program does **not** need to survive unchanged.

If $D_{\mathrm{raw}}$ denotes this decoder, the partial behavior of a raw
program is

$$
\varphi_p(x)=D_{\mathrm{raw}}\bigl(U(T_{p,x})\bigr),
$$

when the machine halts and decoding succeeds. Otherwise
$\varphi_p(x)$ is undefined. In the current software, an empty decoded
prefix is also considered invalid; the consequence is explained in
[The empty-output convention](#the-empty-output-convention).

## 2. What canonical programs describe

Every binary string is allowed as a raw LUTM-1 program. Many strings are
incomplete or malformed and may time out, but the machine still executes them
without an external validator.

The universality argument needs only a well-formed subset. A program in that
subset describes a deterministic, one-way, one-tape **target machine** with:

- logical tape alphabet $\{L,B,0,1\}$;
- a permanent left-end cell $L$ at logical coordinate $0$;
- initial logical tape $LxB^\infty$;
- states $0,\ldots,n-1$, starting in state $0$;
- left, right, and stationary head moves;
- one distinguished halt destination.

For a successful canonical computation, the target machine also follows this
output convention:

1. it never changes the left-end symbol $L$;
2. it never writes $L$ on a positive data cell;
3. it halts with logical tape $LyB^\infty$;
4. it halts with its head back on $L$.

These restrictions define the convenient machines used by the proof. LUTM-1
does not check them while executing an arbitrary program.

## 3. How a target transition table becomes bits

Each target state has four records, one for each symbol it could read, in this
fixed order:

```text
L, B, 0, 1
```

Each record contains three fields:

```text
write-code  direction-code  next-state-code
```

| Field | Meaning | Bits |
|---|---|---:|
| write | $L$ | `00` |
| write | $B$ | `01` |
| write | $0$ | `10` |
| write | $1$ | `11` |
| direction | left | `00` |
| direction | right | `01` |
| direction | stay | `10` |
| next state | halt | `0` |
| next state | state $r$ | $1^{r+1}0$ |

Direction code `11` is unused by canonical programs. The write and direction
fields always use two bits. The next-state field is unary and ends at its
first `0`. This makes every record self-delimiting, so four consecutive
records form one target-state block.

For a finite target machine $M$, concatenating its state blocks produces a
finite binary program

$$
p_M=\operatorname{encode}(M).
$$

The encoding is effective: an ordinary compiler can calculate $p_M$ from
the finite transition table of $M$. The compiler is used only to construct
a program beforehand. During execution, LUTM-1 itself reads and interprets
every field.

### Raw programs can be shorter

The five-bit identity program `00100` is an intentionally small example. Its
executed path means “write $L$, stay, halt.” It omits records that this path
never reaches, so it is a valid operational program but not a complete
four-record serialization. This distinction is useful in program search:
LUTM-1 can exploit short partial descriptions even though the universality
proof uses complete ones.

## 4. The physical machine and its marker symbols

LUTM-1 is a deterministic, bi-infinite, one-tape machine with:

```text
physical tape symbols:          16
states including HALT:         150
nonhalting states:             149
transition rows:             2,384 = 149 x 16
```

Its physical alphabet is

```text
B, 0, 1, #,
C0, C1, A0, A1,
N, X, T, Z,
hL, hb, h0, h1
```

The extra symbols are temporary annotations written by the machine:

| Symbols | Meaning |
|---|---|
| `B`, `0`, `1`, `#` | Literal blank, binary data, and separator |
| `C0`, `C1` | The first bit of the currently active target-state block |
| `A0`, `A1` | A candidate block while finding the next target state |
| `N`, `X`, `T`, `Z` | Marks used while reading a unary next-state field |
| `hL` | The simulated head is on logical $L$ |
| `hb`, `h0`, `h1` | The simulated head is on a blank, `0`, or `1` data cell |

`C0` and `C1` do not erase a program bit. They mean “marked `0`” and
“marked `1`,” respectively. The same is true of `A0` and `A1`. LUTM-1
can therefore use its program region as workspace and later recover the
original bit.

The separator is not protected. While the simulated head is on logical $L$,
LUTM-1 temporarily replaces `#` with `hL`. It restores `#` before a
canonical physical halt.

## 5. Initialization

Starting from the literal tape $T_{p,x}$, LUTM-1 performs four tasks:

1. `START` changes the separator into `hL`, marking the initial target-head
   position.
2. It scans left across the finite program to find the program's first bit.
3. It changes that bit into `C0` or `C1`, marking target state $0$ as
   active without losing the bit.
4. It returns to coordinate $-1$ and enters the state used to find the
   transition for logical symbol $L$.

At this point the input is still literal $x$, the simulated target tape is
$LxB^\infty$, the simulated head is on $L$, and target state $0$ is
active.

## 6. One simulated target step

One transition of the target machine takes many physical LUTM-1 transitions.
The work can be understood as four phases.

### Phase A: find the applicable record

The active target-state block is marked by `C0` or `C1`. LUTM-1 scans to
that marker. The symbol under the simulated head determines which record is
needed:

| Simulated symbol | Records skipped |
|---|---:|
| $L$ | 0 |
| $B$ | 1 |
| $0$ | 2 |
| $1$ | 3 |

To skip one record, the machine crosses its four fixed write/direction bits
and then scans to the terminating `0` of its unary next-state field. It can
therefore reach the correct record even though records have different
lengths.

LUTM-1 reads the selected write and direction bits into its finite control. It
temporarily changes the selected next-state field as follows:

$$
\begin{aligned}
0 &\longrightarrow Z
&& \text{for halt},\\
1^{r+1}0 &\longrightarrow N^{r+1}T
&& \text{for target state }r.
\end{aligned}
$$

These marks let the machine find the field again after scanning elsewhere.

### Phase B: write and move on the simulated tape

If the simulated head is on $L$, its physical marker is `hL` at coordinate
$0$. A canonical target transition leaves $L$ unchanged. A right move
restores the separator and marks data cell $1$; a stay or left move keeps
`hL`.

If the simulated head is on a positive data cell, LUTM-1 scans right to the
unique `hb`, `h0`, or `h1` marker. It writes the selected symbol and
places a new head marker on the left, same, or right data cell. Moving left
from data cell $1$ changes the separator into `hL`.

After this phase, the positive physical tape exactly represents the target
tape after its write and move.

### Phase C: select the next target state

For a nonhalting transition to state $r$, the program now contains
$N^{r+1}T$ in the selected record. LUTM-1 restores the old active marker and
marks state block $0$ as a candidate using `A0` or `A1`.

It then consumes the $N$ marks one at a time. If another $N$ remains after
one is consumed, the candidate advances by one complete four-record state
block. When no $N$ remains, it stops.

For example, state $2$ is encoded by three unary `1` bits. The temporary
field contains `NNNT`: the first consumed `N` advances the candidate to
block $1$, the second advances it to block $2$, and the last confirms the
selection without another advance.

Finally, LUTM-1 restores all temporary symbols:

```text
X       -> 1
T       -> 0
A0/A1   -> C0/C1
```

The marker now identifies exactly the next target state.

### Phase D: repeat or halt

For a nonhalting record, LUTM-1 returns to coordinate $-1$ and begins another
simulated step using the symbol under the new simulated head.

For a halt record, the selected `0` was marked `Z`. LUTM-1 first performs
the requested write and movement, then follows its halt path instead of
choosing another target state.

## 7. The precise simulation invariant

The following invariant makes the preceding description suitable for an
inductive proof. Between two simulated target steps, a target configuration
$(q,h,t)$ is represented when:

1. the physical control state records the logical symbol currently under the
   simulated head;
2. the physical head is at coordinate $-1$;
3. exactly one `C0` or `C1` marks the first bit of target-state block $q$;
4. no temporary `A`, `N`, `X`, `T`, or `Z` marker remains;
5. if $h=0$, coordinate $0$ is `hL` and no positive head marker exists;
6. if $h>0$, coordinate $0$ is `#` and coordinate $h$ contains exactly
   one of `hb`, `h0`, or `h1`;
7. every other positive cell is the corresponding literal target-tape symbol.

Initialization establishes this invariant for state $0$, head position
$0$, and tape $LxB^\infty$. Phases A through C preserve the represented
program bits and transform the rest of the representation into the exact
successor target configuration. This proves one simulation step by direct
inspection of the four phases.

Each simulated step finishes after finitely many physical transitions:

- the program is finite;
- every serialized record and unary field is finite;
- a well-formed program refers only to existing target states;
- after finitely many target steps, the target head is at a finite coordinate.

The construction is exact but intentionally not efficient. A single target
step may require several full scans of the program and of the used data
region.

## 8. Halting and decoded output

A canonical target machine halts with logical tape
$LyB^\infty$ and its head on $L$. At that moment, the physical tape has
`hL` at coordinate $0$ and literal $y$ on the positive side.

LUTM-1's halt path:

1. returns through the program region;
2. changes `hL` back into `#`;
3. enters physical `HALT`.

It does not clean the negative tape, because the decoder ignores it. The
canonical path also leaves the physical head at coordinate $0$, although
the general runtime contract does not require that.

If the target machine never halts, LUTM-1 performs an infinite sequence of
finite simulated steps and never reaches physical `HALT). Thus both halting
and divergence are preserved.

## 9. Why the target-machine class is universal

The target machines used above are deliberately simple, but they can compute
every partial computable binary function. Starting with any ordinary
deterministic Turing machine, standard finite transformations can:

1. encode multiple tapes as tracks on one tape;
2. encode a two-sided tape on a one-way tape by interleaving its coordinates;
3. encode any finite work alphabet with fixed-length binary blocks;
4. add a finite prologue that turns literal $LxB^\infty$ into the chosen
   work representation;
5. add a finite epilogue that extracts the answer, erases the finitely visited
   work region, writes $LyB^\infty$, and returns the head to $L$.

If the original computation diverges, the epilogue is never reached. Every
transformation changes only a finite transition table. Therefore every
partial computable binary function has a target machine satisfying the
canonical input and output conventions.

## 10. Latent-universality proof sketch

Let $g$ be any partial computable binary function.

1. **Normalize the target.** Section 9 provides a canonical target machine
   $M_g$ that computes $g$, returns its head to $L$, and leaves only the
   output on its logical tape.
2. **Serialize it.** Section 3 effectively constructs the finite program
   $p_g=\operatorname{encode}(M_g)$.
3. **Initialize the simulation.** Section 5 establishes the invariant for
   $M_g$ on the literal input $x$.
4. **Simulate by induction.** Assuming the invariant represents one target
   configuration, Section 6 produces the exact successor configuration after
   finitely many physical steps and re-establishes the invariant.
5. **Preserve termination.** Section 8 shows that canonical target halting
   produces the same decoded output, while target divergence produces physical
   divergence.

Consequently, whenever $g(x)$ is defined,

$$
\varphi_{p_g}(x)=g(x).
$$

When $g(x)$ is undefined, both computations diverge. Conversely, every
$\varphi_p$ is partial computable because LUTM-1 is one fixed finite
deterministic Turing machine and its output decoder is computable.

The important order of choices is

$$
\exists\,(U,E,D,C)\;
\forall g\;
\exists p_g\;
\forall x.
$$

Here $E$ is literal input placement, $D$ is fixed output decoding, and
$C$ combines a program and encoded input into the fixed initial tape
configuration. They are chosen once. Only $p_g$ depends on the desired task
$g$. This is the sense in which universality is *latent in the program
region* and useful for program search.

It is not a stronger computability class than ordinary Turing universality.
The emphasis is the fixed, task-independent input/output interface.

## The empty-output convention

The standard set $\Sigma^*$ contains the empty string $\epsilon$. The
current LUTM-1 runtime deliberately rejects an empty raw output prefix, so its
directly represented defined outputs lie in

$$
\Sigma^+=\Sigma^*\setminus\{\epsilon\}.
$$

A fixed output code recovers the conventional representation without changing
which functions are computable:

$$
\eta(y)=1y,
\qquad
\eta(\epsilon)=1.
$$

A canonical target writes $\eta(g(x))$, and one fixed outer decoder removes
the initial `1`:

$$
D_{\mathrm{std}}
=
\eta^{-1}\circ D_{\mathrm{raw}}.
$$

The code and its inverse do not depend on $g$, $p_g$, or $x$, so the
order of choices in the latent-universality statement is unchanged. The
repository exposes $D_{\mathrm{raw}}$ directly; users who need semantic
empty outputs must apply this fixed convention to both targets and decoded
results.

## What this argument does—and does not—establish

The construction argues for:

- one fixed finite transition table;
- literal, task-independent binary input placement;
- fixed halting and output conventions;
- a finite program for every partial computable binary task under the stated
  output representation;
- preservation of target halting and divergence.

It does **not** establish:

- that useful programs are short;
- that simulation is efficient;
- that enumeration or evolution will find a program in practical time;
- that arbitrary malformed programs behave cleanly;
- that finite tests prove an unbounded theorem;
- that the construction or software is free from errors.

The unbounded claim depends on the simulation invariant and induction over
target steps. Repository tests provide implementation evidence by checking
known computations, boundary behavior, simulator agreement, enumeration, GA
operators, and large batches. Tests support the argument but cannot replace a
formal proof.

## Canonical artifact

The fixed physical rule is:

```text
data/transition_table.csv
SHA-256 260fb1d15014523c6a46ede09f9fcd7cd7d591f668912842f1aae8cf993376f9
```

The CSV is preserved byte-for-byte as a binary Git artifact because this hash
identifies the particular machine described here. A different hash is a
different transition table and must be audited as a different machine.
