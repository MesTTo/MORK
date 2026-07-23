# This branch: GPT-2 runs natively in the store

This lineage carries the einsum/tensor sink layer that ran the full 12-layer GPT-2 (124M)
forward pass natively inside MORK: weights and activations as expressions in the space, the
forward pass as the kernel's own rule application through dense sinks, logits matched to the
reference implementation at a maximum relative error of 6.06e-6 with argmax agreement, and
exact incremental decode. The kernel-side source is `kernel/src/tensor_ops.rs`,
`kernel/src/graph_tensor.rs`, `kernel/tests/einsum_sink.rs`, and the `linalg/` crate; the
measurement record is the
[metta-quantimork-transformer report](https://github.com/MesTTo/metta-on-mork/blob/main/metta-quantimork-transformer.pdf).
The Python driving harness is not yet published.

`metta-on-mork-base` is also the kernel base behind
[metta-on-mork](https://github.com/MesTTo/metta-on-mork), the in-process Hyperon atomspace
on MORK: upstream main plus the full open MesTTo PR set
([`upstream-plus-prs`](https://github.com/MesTTo/MORK/tree/upstream-plus-prs) is exactly
that merge) plus the features beyond it: `stratified_quiescence`, `guarded_emit`,
`retrieval_join`, `bulk_emit`, `witness_select`, and the tensor layer above. Build the demo
kernel with
`--features semi_naive_ic,leapfrog,stratified_quiescence,guarded_emit,retrieval_join,witness_select`
(nightly, `RUSTFLAGS="-C target-cpu=native --cfg gxhash"`). Everything below is upstream's
README.

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
