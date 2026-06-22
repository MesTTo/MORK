use vstd::prelude::*;

verus! {

pub fn rollback_to<T>(trail: &mut Vec<T>, mark: usize)
    requires
        mark <= old(trail).len(),
    ensures
        final(trail)@ == old(trail)@.subrange(0, mark as int),
{
    while trail.len() > mark
        invariant
            mark <= trail.len() <= old(trail).len(),
            trail@ == old(trail)@.subrange(0, trail.len() as int),
        decreases trail.len() - mark,
    {
        trail.pop();
    }
}

fn main() {}

}
