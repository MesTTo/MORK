# This branch: the MesTTo release lineage

`metta-on-mork-base` is the kernel lineage behind
[metta-on-mork](https://github.com/MesTTo/metta-on-mork), the in-process Hyperon atomspace
on MORK. It is upstream main plus the full open MesTTo PR set
([`upstream-plus-prs`](https://github.com/MesTTo/MORK/tree/upstream-plus-prs) is exactly
that merge) plus the features beyond it: `stratified_quiescence`, `guarded_emit`,
`retrieval_join`, `bulk_emit`, `witness_select`, and the einsum/tensor sink layer. Build the
demo kernel with
`--features semi_naive_ic,leapfrog,stratified_quiescence,guarded_emit,retrieval_join,witness_select`
(nightly, `RUSTFLAGS="-C target-cpu=native"`).

The tensor layer (`kernel/src/tensor_ops.rs`, `kernel/src/graph_tensor.rs`,
`kernel/tests/einsum_sink.rs`, and the `linalg/` crate) is the kernel-side source of the
in-store GPT-2 result: the full 12-layer GPT-2 (124M) forward pass executes natively in the
store, matched to the reference implementation at a maximum relative logit error of
6.06e-6 with argmax agreement, with exact incremental decode. The measurement record is the
[metta-quantimork-transformer report](https://github.com/MesTTo/metta-on-mork/blob/main/metta-quantimork-transformer.pdf);
the Python driving harness is not yet published. Everything below is upstream's README.

---

# MeTTa Optimal Reduction Kernel

**A blazing fast hypergraph processing kernel for Hyperon**

MORK seeks to retrofit Hyperon with a state-of-the-art graph database and a specialized zipper-based multi-threaded virtual machine to provide speedy MeTTa evaluation across the full range of Space sizes and topologies.

By rearchitecting certain Hyperon bottlenecks, MORK has the potential to accelerate important use cases by thousands to millions of times.  That kind of speedup represents a qualitative jump in capabilities.  It's the difference between running a training step vs. finishing the training in the same amount of time.  It's the difference between a thousand input samples vs. millions, or a crocodile's brain vs. a human's.  Deep learning has advanced due in part to the software platforms that exposed the full capabilities of underlying hardware, and we hope Hyperon + MORK can help do that for symbolic AI.

## Wiki
[The wiki](https://github.com/trueagi-io/MORK/wiki#where-to-start) is where you find examples, tutorials, and more info about both the formalism and implementation.

## Trying it out
If you're looking for the MORK server, use the [server branch](https://github.com/trueagi-io/MORK/tree/server).

If you're looking for the MORK command line utility, run `cargo build --release` in `/kernel`; you'll need a nightly compiler `rustup toolchain install nightly`.
