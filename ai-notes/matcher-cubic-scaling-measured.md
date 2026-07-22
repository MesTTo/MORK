
## CORRECTION (2026-07-22, later): the localization below was WRONG

The section below attributed the cube to the I/O `==` source join. That was a misread: there are
two near-identical bench functions, and the cubic was measured on `process_calculus_bench` (the
`(, ...)` conjunctive body, which is also what `mork bench process_calculus` ships), NOT on
`process_calculus_source_sink_bench` (the `(I (== ...))` variant at main.rs:311 whose source I
read). The shipped bench never calls `query_multi_i`. Verified on the routed binary
(ai/io-eq-leapfrog @ 65b878d): the original `,`-bench is byte-identical to stock under the new
route (slope 2.949 stock, 2.949 gated, 2.956 forced). The io-eq route is real for `(I (==))`
bodies but does not touch the shipped cubic.

What remains TRUE from the measurements: O(n^2.95) transitions vs O(n) output on the shipped
bench; unifications and writes linear; SNI is a 4.4x constant; leapfrog gated AND forced leave
the exponent unchanged. Facts about the `,`-reaction now in evidence: its join variables
($channel, $payload) sit INSIDE compound argument positions; accumulated receives are SCHEMATIC
(`(add $ret)` stores a variable); the add-only `,`-semantics never removes consumed pairs, so the
dish accumulates O(n) facts of O(n) bytes. Which rule actually produces the cubic transitions
(reaction join re-check, par rule, SNI not engaging on the respawned rule key, or other) is OPEN
and must be settled by per-exec-token transition attribution, not analysis. Two candidate levers
on file: the `retrieval_join` feature stub (Cargo.toml: "unifiability retrieval into stored
regions as a join-side operation") and extending seek to compound-nested columns.

## De-risk (2026-07-22): the cube is the I/O `==` source join, NOT the conjunctive matcher

Rebuilt `--features leapfrog,semi_naive_ic` and re-ran the pc scaling sweep. Leapfrog is the
existing indexed WCO join (#124), wired into metta_calculus. It does NOT fix pc:

| config | n=32 | n=64 | n=128 | n=256 | exp(128->256) |
|---|---:|---:|---:|---:|---:|
| sni-only baseline | 364,515 | 2,519,859 | 18,757,075 | 144,812,819 | 2.95 |
| leapfrog gated + sni | (identical to baseline) | | | | 2.95 |
| leapfrog FORCED + sni | 351,398 | 2,469,046 | 18,557,142 | 144,019,734 | 2.96 |

Gated leapfrog declines pc's bodies entirely (identical to baseline). Forced
(`MORK_LEAPFROG=all`) changes the count by ~1% and leaves the exponent at 2.96, still cubic. So
the cube is NOT in the plain conjunctive-join matching leapfrog handles.

Localization: pc's hot bodies are I/O `==` sources,
`(I (== (petri (? $ch $pl $body)) $recv) (== (petri (! $ch $pl)) $send))`, a join over petri
facts on a shared channel, which #124 explicitly leaves on the stock path ("interpreted sources
and sinks (I/O) keep the stock path"). That stock path scans the dish pairwise.

(A) vs (B), resolved to (A) fixable from the counters already in hand: writes (reactions) are
linear, transitions (finding the reaction) are cubic, so each of O(n) reactions costs O(dish) =
O(n^2) to locate a partner that a channel index would find in O(1). Small relevant set, scanned.
Not inherent. The fix is to make the `==` source join seek by key instead of scanning, which is
relational e-matching applied to the I/O source path specifically. Oracle: pc transitions
exponent must fall from ~2.95 toward ~2 or below, byte-identical output.

## Routed I-path result

This section measures a different workload from the cubic baseline tables above. The routed
fixture calls `process_calculus_source_sink_bench`, whose reaction body uses interpreted
`I (== ...)` sources. It also rebuilds the two generic schematic addition receivers as one
ground-channel receiver per recursive value because each matched receiver is removed. Those
ground channel facts are the selective columns that the leapfrog join can seek.

The release binary was built with `--features leapfrog,semi_naive_ic` and
`RUSTFLAGS="-C target-cpu=native -Awarnings --cfg gxhash"`. Each row below ran the separately
named `mork bench process_calculus_io_s{4n+1}_n{n}` selector in a fresh process.

| config | n=32 | n=64 | n=128 | n=256 | slope 128->256 |
|---|---:|---:|---:|---:|---:|
| stock, `MORK_LEAPFROG=0` | 986,181 | 7,142,389 | 54,363,989 | 424,234,517 | 2.964 |
| routed, default gate | 986,181 | 7,142,389 | 14,305,936 | 33,098,768 | 1.210 |
| routed, `MORK_LEAPFROG=all` | 27,236 | 87,140 | 305,252 | 1,134,692 | 1.894 |

All configurations produced the asserted Peano result at every size. Unifications were 227, 451,
899, and 1,795. Writes were 549, 1,093, 2,181, and 4,357. Their 128->256 slopes were 0.998 and
0.998 respectively.

The original `process_calculus_s{4n+1}_n{n}` selector again calls the shipped
`process_calculus_bench`, whose reaction body is the plain `(, ...)` transform path. A fresh sweep
on that selector produced the following transition counts. All three configurations produced
byte-identical Peano results. Stock and the default gate have identical counters. Forced leapfrog
changes the count slightly but leaves the exponent cubic.

| shipped config | n=32 | n=64 | n=128 | n=256 | slope 128->256 |
|---|---:|---:|---:|---:|---:|
| stock, `MORK_LEAPFROG=0` | 364,515 | 2,519,859 | 18,757,075 | 144,812,819 | 2.949 |
| routed, default gate | 364,515 | 2,519,859 | 18,757,075 | 144,812,819 | 2.949 |
| routed, `MORK_LEAPFROG=all` | 351,398 | 2,469,046 | 18,557,142 | 144,019,734 | 2.956 |

Verdict: interpreted all-`==` I-bodies now route through the existing leapfrog dispatch with
byte-identical effects. The selective ground-channel fixture demonstrates an asymptotic drop from
2.964 to 1.210 under the deterministic default gate. This is a dispatch-coverage extension, not a
fix for the recorded `process_calculus_bench` deficiency.

The shipped benchmark's cubic remains open. Its known instance has join variables inside compound
argument positions, schematic accumulated receives, and add-only accumulation. The route in this
branch does not change that workload's exponent.
