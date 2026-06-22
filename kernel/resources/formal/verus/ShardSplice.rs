use vstd::prelude::*;

verus! {

// Abstract model of the ShardZipper capture/reintegrate laws (shard_zipper.rs),
// with a pathspace viewed as a set of byte-string keys (PathMap<()>). It
// certifies the splice contract the executable take_map / graft_map must meet:
// capturing the shard at a prefix and reintegrating it unchanged round-trips to
// the original space, and a sweep that replaces only the shard leaves every key
// outside the prefix exactly as it was. Goertzel, ShardZipper, 2025.

/// `p` is a prefix of `k`.
pub open spec fn is_prefix(p: Seq<u8>, k: Seq<u8>) -> bool {
    p.len() <= k.len() && k.subrange(0, p.len() as int) =~= p
}

/// The suffix of `k` after stripping the prefix `p` (used when is_prefix(p,k)).
pub open spec fn strip(p: Seq<u8>, k: Seq<u8>) -> Seq<u8> {
    k.subrange(p.len() as int, k.len() as int)
}

/// Capture C_k: the shard at prefix `p` is the set of suffixes `s` with `p + s`
/// a key. Keys inside the shard are relative to the shard root.
pub open spec fn shard_of(t: ISet<Seq<u8>>, p: Seq<u8>) -> ISet<Seq<u8>> {
    ISet::new(|s: Seq<u8>| t.contains(p + s))
}

/// What stays after capture: the keys not under `p`.
pub open spec fn rest_of(t: ISet<Seq<u8>>, p: Seq<u8>) -> ISet<Seq<u8>> {
    ISet::new(|k: Seq<u8>| t.contains(k) && !is_prefix(p, k))
}

/// Reintegrate Gamma_s: splice `shard` back in at `p` alongside `rest`. A key is
/// present if it is in the rest, or it lies under `p` and its suffix is in the
/// shard.
pub open spec fn reintegrate(rest: ISet<Seq<u8>>, p: Seq<u8>, shard: ISet<Seq<u8>>) -> ISet<Seq<u8>> {
    rest.union(ISet::new(|k: Seq<u8>| is_prefix(p, k) && shard.contains(strip(p, k))))
}

/// Stripping then re-prepending the prefix reconstructs the key.
pub proof fn prefix_strip_reconstructs(p: Seq<u8>, k: Seq<u8>)
    requires
        is_prefix(p, k),
    ensures
        p + strip(p, k) =~= k,
{
    assert(k.subrange(0, p.len() as int) + k.subrange(p.len() as int, k.len() as int) =~= k);
}

/// Capturing the shard at `p` and reintegrating it unchanged restores the
/// original space.
pub proof fn capture_reintegrate_round_trips(t: ISet<Seq<u8>>, p: Seq<u8>)
    ensures
        reintegrate(rest_of(t, p), p, shard_of(t, p)) =~= t,
{
    assert(reintegrate(rest_of(t, p), p, shard_of(t, p)) =~= t) by {
        assert forall|k: Seq<u8>| #![auto]
            reintegrate(rest_of(t, p), p, shard_of(t, p)).contains(k) <==> t.contains(k) by {
            if is_prefix(p, k) {
                prefix_strip_reconstructs(p, k);
            }
        }
    }
}

/// A sweep replaces the shard with the kernel's output `shard2`. Every key not
/// under `p` is unchanged: it is present after the sweep exactly when it was
/// present before. Edits are confined to the prefix.
pub proof fn sweep_preserves_outside(t: ISet<Seq<u8>>, p: Seq<u8>, shard2: ISet<Seq<u8>>)
    ensures
        forall|k: Seq<u8>| !is_prefix(p, k) ==> (#[trigger]
        reintegrate(rest_of(t, p), p, shard2).contains(k) <==> t.contains(k)),
{
}

fn main() {
}

}
