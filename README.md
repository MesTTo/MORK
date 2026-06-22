
# MeTTa Optimal Reduction Kernel

**A blazing fast hypergraph processing kernel for Hyperon**

MORK seeks to retrofit Hyperon with a state-of-the-art graph database and a specialized zipper-based multi-threaded virtual machine to provide speedy MeTTa evaluation across the full range of Space sizes and topologies.

By rearchitecting certain Hyperon bottlenecks, MORK has the potential to accelerate important use cases by thousands to millions of times.  That kind of speedup represents a qualitative jump in capabilities.  It's the difference between running a training step vs. finishing the training in the same amount of time.  It's the difference between a thousand input samples vs. millions, or a crocodile's brain vs. a human's.  Deep learning has advanced due in part to the software platforms that exposed the full capabilities of underlying hardware, and we hope Hyperon + MORK can help do that for symbolic AI.

## This fork: optimized kernel + WILLIAM

This fork keeps the design above and rebuilds the parts that fall over under load. Measured against stock upstream MORK with `mork bench` on a Ryzen
9950X, same graphs and same results:

| benchmark | this fork vs upstream |
|-----------|-----------------------|
| clique, 5-clique join | **~1200×** (14.4 s → 12 ms, same 102 cliques found) |
| process-calculus rewriting | **2.0×** (byte-identical `(S^400 Z)` result) |
| transitive closure | **1.5×–2.4×** |
| finite-domain solving | **1.8×** |
| counter-machine | **1.35×** (same 1813 steps) |

What changed:

- A worst-case-optimal join for multi-pattern queries, so a conjunctive match costs what the
  output costs and not what the cross-product costs. This is the clique win.
- A compiled discrimination-trie matcher that walks the pattern and the trie together in one descent.
- A streamed, factorized emit that never materializes the full product.
- A single-factor fast path plus per-thread query metrics. Read-only point queries now scale
  1.9M → 17.6M → 26.6M per second across 1, 16, and 32 threads, where a global metrics mutex
  used to collapse them to 3.6M past eight threads. The change is byte-identical to the planned path.
- WILLIAM-on-MORK (whitepaper 5.12): a compression-gain weighted index that returns the
  heaviest, most compressible subpatterns from any prefix without a scan. `mork legacy william`
  finds the top 16 of 70,000 prefixes in about 20µs, against about 3.8ms for the full scan.
- A numeric layer that runs tensor operations on the same kernel as the symbolic side (sparse
  SpGEMM, block-sparse attention, and an einsum VM). See "Tensors on the same substrate" below.

Build with `RUSTFLAGS="-C target-cpu=native" cargo +nightly build --release -p mork --bin mork`.
The crate that runs a Hyperon atomspace on this kernel is
[metta-on-mork](https://github.com/MesTTo/metta-on-mork).

### Tensors on the same substrate

The fork carries a numeric half too. `linalg/` is a sparse-and-dense linear algebra engine: CSR
compressed-sparse-row matrices with sequential and rayon-parallel SpGEMM, a block-sparse tensor
with AVX2+FMA attention kernels, an OpenBLAS dense path, and a runtime einsum VM that composes
over all of them through one trait pair, so a spec like `ab,bc->ac` runs whether the inputs are
sparse, dense, or a mix of the two. `linalg/benches/crossover.rs` measures the density crossover,
the point where the sparse path stops beating dense OpenBLAS, across GPT-2 Small to XL attention
shapes.

Measured on a Ryzen 9950X against OpenBLAS 0.3.32 pinned to one thread, the sparse path is
numerically equal to dense BLAS (`max_rel < 1e-3` at every density) and wins below these crossover
densities:

| workload | sparse beats dense below |
|----------|--------------------------|
| SpGEMM, sequential CSR | ~2.8–6.2% density (n from 256 to 4096) |
| SpGEMM, parallel CSR | ~6.9–24.9% density, rising with n |
| attention, Blocked8 | ~0.54–0.70% density (GPT-2 Small to XL) |
| attention, Blocked16 | ~0.26–0.35% density (GPT-2 Small to XL) |

At extreme sparsity the gap is large: a 1024×1024 SpGEMM at 0.01% density runs in about 1µs against
OpenBLAS's 7ms, since the sparse kernel only touches the nonzeros while dense BLAS does the full
n³ work no matter how many entries are zero. Above the crossover dense BLAS wins for that same
reason, which is what the sweep is there to locate. Reproduce with `cargo bench -p linalg --bench
crossover --features blas`, and set `CROSSOVER_LARGE=1` to add the Large and XL configs.

The kernel wires that engine onto MORK's relations. `graph_tensor` materialises a binary relation
`(rel a b)` into a CSR adjacency, carrying a per-relation `symbol_bytes <-> dense u32` bijection
because MORK nodes are encoded symbol bytes while CSR needs a dense `0..n` numbering. It runs
graph-numeric sweeps over that adjacency (degrees, two-hop counts, message passing) and maps the
results back to symbols; for a ShardZipper shard the numbering is local and small, so this is the
numeric half of the cost-bounded materialise step. `tensor_ops` reads tensor specs out of MeTTa
expressions and runs them through the einsum JIT. The result is that numeric work, graph sweeps
and attention included, runs on the same kernel as the symbolic relations: the symbols become
sparse matrices, and the numbers become symbols again.

## Wiki
[The wiki](https://github.com/trueagi-io/MORK/wiki#where-to-start) is where you find examples, tutorials, and more info about both the formalism and implementation.

## Trying it out
If you're looking for the MORK server, use the [server branch](https://github.com/trueagi-io/MORK/tree/server).

If you're looking for the MORK command line utility, run `cargo build --release` in `/kernel`; you'll need a nightly compiler `rustup toolchain install nightly`.
