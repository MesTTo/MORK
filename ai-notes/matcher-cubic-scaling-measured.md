
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

Built the release binary with `--features leapfrog,semi_naive_ic` and
`RUSTFLAGS="-C target-cpu=native -Awarnings --cfg gxhash"`. Each row ran
`mork bench process_calculus_s{4n+1}_n{n}` in a fresh process.

| config | n=32 | n=64 | n=128 | n=256 | slope 128->256 |
|---|---:|---:|---:|---:|---:|
| stock, `MORK_LEAPFROG=0` | 986,181 | 7,142,389 | 54,363,989 | 424,234,517 | 2.964 |
| routed, default gate | 986,181 | 7,142,389 | 14,305,936 | 33,098,768 | 1.210 |
| routed, `MORK_LEAPFROG=all` | 27,236 | 87,140 | 305,252 | 1,134,692 | 1.894 |

All configurations produced the asserted Peano result at every size. Unifications were 227, 451,
899, and 1,795. Writes were 549, 1,093, 2,181, and 4,357. Their 128->256 slopes were 0.998 and
0.998 respectively.

The checked-in source/sink fixture previously had one generic recursive receiver, then removed it
on its first match. It therefore quiesced without an addition result for `n >= 2`. The scaling
wiring now registers one consumable receiver for every recursive value of `x`. The quoted `I`/`O`
reaction rule is unchanged, and the dish has the measured O(n) pending receives plus one live send.

Verdict: the routed default slope is 1.210, below the required 2.2 ceiling. The forced slope is
1.894. The differential oracle is byte-identical, and unifications and writes remain linear.
