use log::trace;
use mork_expr::macros::SerializableExpr;
use mork_expr::{Expr, Tag, destruct, item_byte, serialize};
use pathmap::PathMap;
use pathmap::arena_compact::ACTMmapZipper;
use pathmap::zipper::*;

pub(crate) enum ResourceRequest {
    BTM(&'static [u8]),
    ACT(&'static str),
    #[cfg(feature = "z3")]
    Z3(&'static str),
}

pub(crate) enum Resource<'trie, 'path> {
    BTM(ReadZipperUntracked<'trie, 'path, ()>),
    ACT(ACTMmapZipper<'trie, ()>),
    #[cfg(feature = "z3")]
    Z3(ReadZipperOwned<()>),
}

fn btm_request() -> impl Iterator<Item = ResourceRequest> {
    std::iter::once(ResourceRequest::BTM(&[]))
}

fn next_btm_resource<'trie, 'path, It>(it: &mut It) -> ReadZipperUntracked<'trie, 'path, ()>
where
    It: Iterator<Item = Resource<'trie, 'path>>,
{
    let Resource::BTM(rz) = it.next().unwrap() else {
        unreachable!()
    };
    rz
}

pub(crate) trait Source {
    // step 1: parsing the source
    fn new(e: Expr) -> Self;
    // step 2: request access to resources before running
    fn request(&self) -> impl Iterator<Item = ResourceRequest>;
    // step 3: create the factor in the product/the (virtual) zipper for the source
    fn source<'trie, 'path, It: Iterator<Item = Resource<'trie, 'path>>>(
        &self,
        it: It,
    ) -> AFactor<'trie, ()>
    where
        'path: 'trie;
}

pub(crate) struct CompatSource;
impl Source for CompatSource {
    fn source<'trie, 'path, It: Iterator<Item = Resource<'trie, 'path>>>(
        &self,
        mut it: It,
    ) -> AFactor<'trie, ()>
    where
        'path: 'trie,
    {
        let rz = next_btm_resource(&mut it);
        AFactor::CompatSource(rz)
    }

    fn new(_e: Expr) -> Self {
        Self
    }

    fn request(&self) -> impl Iterator<Item = ResourceRequest> {
        btm_request()
    }
}

pub(crate) struct BTMSource;
impl Source for BTMSource {
    fn request(&self) -> impl Iterator<Item = ResourceRequest> {
        btm_request()
    }

    fn source<'trie, 'path, It: Iterator<Item = Resource<'trie, 'path>>>(
        &self,
        mut it: It,
    ) -> AFactor<'trie, ()>
    where
        'path: 'trie,
    {
        // (I (BTM <pat1>) (ACT <filename> <pat2>)
        //    --factor1--  -----factor2---------
        // prefix: '[2] BTM'
        static PREFIX: [u8; 5] = [
            item_byte(Tag::Arity(2)),
            item_byte(Tag::SymbolSize(3)),
            b'B',
            b'T',
            b'M',
        ];
        let rz = next_btm_resource(&mut it);
        let rz = PrefixZipper::new(&PREFIX[..], rz);
        AFactor::PosSource(rz)
    }

    fn new(_e: Expr) -> Self {
        Self
    }
}

pub(crate) struct ACTSource {
    act: &'static str,
}
impl Source for ACTSource {
    fn new(e: Expr) -> Self {
        destruct!(e, ("ACT" {act: &str} se), {
            return ACTSource { act }
        }, _err => { panic!("act not the right shape") });
    }

    fn request(&self) -> impl Iterator<Item = ResourceRequest> {
        std::iter::once(ResourceRequest::ACT(self.act))
    }

    fn source<'trie, 'path, It: Iterator<Item = Resource<'trie, 'path>>>(
        &self,
        mut it: It,
    ) -> AFactor<'trie, ()>
    where
        'path: 'trie,
    {
        // prefix: '[3] ACT <filename>'
        static CONSTANT_PREFIX: [u8; 5] = [
            item_byte(Tag::Arity(3)),
            item_byte(Tag::SymbolSize(3)),
            b'A',
            b'C',
            b'T',
        ];
        let Resource::ACT(rz) = it.next().unwrap() else {
            unreachable!()
        };
        let mut prefix = vec![];
        prefix.extend_from_slice(&CONSTANT_PREFIX[..]);
        prefix.push(item_byte(Tag::SymbolSize((self.act.size() as u8) - 1)));
        prefix.extend_from_slice(self.act.as_bytes());
        trace!(target: "source", "act prefix {}", serialize(&prefix[..]));
        let rz = PrefixZipper::new(prefix, rz);
        AFactor::ACTSource(rz)
    }
}

#[cfg(feature = "z3")]
pub(crate) struct Z3Source {
    ins: &'static str,
}
#[cfg(feature = "z3")]
impl Source for Z3Source {
    fn new(e: Expr) -> Self {
        destruct!(e, ("z3" {instance: &str} se), {
            return Z3Source { ins: instance }
        }, _err => { panic!("z3 not the right shape {:?}", e) });
    }

    fn request(&self) -> impl Iterator<Item = ResourceRequest> {
        std::iter::once(ResourceRequest::Z3(self.ins))
    }

    fn source<'trie, 'path, It: Iterator<Item = Resource<'trie, 'path>>>(
        &self,
        mut it: It,
    ) -> AFactor<'trie, ()>
    where
        'path: 'trie,
    {
        // prefix: '[3] z3 <instance name>'
        static CONSTANT_PREFIX: [u8; 4] = [
            item_byte(Tag::Arity(3)),
            item_byte(Tag::SymbolSize(2)),
            b'z',
            b'3',
        ];
        let Resource::Z3(rz) = it.next().unwrap() else {
            unreachable!()
        };
        let mut prefix = vec![];
        prefix.extend_from_slice(&CONSTANT_PREFIX[..]);
        prefix.push(item_byte(Tag::SymbolSize((self.ins.size() as u8) - 1)));
        prefix.extend_from_slice(self.ins.as_bytes());
        trace!(target: "source", "z3 prefix {}", serialize(&prefix[..]));
        let rz = PrefixZipper::new(prefix, rz);
        AFactor::Z3Source(rz)
    }
}

pub(crate) struct CmpSource {
    cmp: usize,
}

impl CmpSource {
    fn freshened_rhs_copy(p: &[u8]) -> Vec<u8> {
        let e = Expr {
            ptr: p.as_ptr().cast_mut(),
        };
        let introductions = e.newvars();
        debug_assert!(u8::try_from(introductions).is_ok());

        let mut rhs = p.to_vec();
        e.shift(
            introductions as u8,
            &mut mork_expr::ExprZipper::new(Expr {
                ptr: rhs.as_mut_ptr(),
            }),
        );
        rhs
    }

    fn exclude_rhs_path(map: &PathMap<()>, p: &[u8]) -> PathMap<()> {
        map.subtract(&PathMap::single(p, ()))
    }

    fn policy(
        ctx: (usize, PathMap<()>),
        p: &[u8],
        c: usize,
    ) -> ((usize, PathMap<()>), Option<ReadZipperOwned<()>>) {
        let (cmp, map) = ctx;
        if c == 0 {
            if cmp == 0 {
                trace!(target: "source", "== enrolling at {}", serialize(p));
                let qv = Self::freshened_rhs_copy(p);
                (
                    (cmp, map),
                    Some(PathMap::single(&qv[..], ()).into_read_zipper(&[])),
                )
            } else if cmp == 1 {
                let present = map.get_val_at(p).is_some();
                let filtered = Self::exclude_rhs_path(&map, p);
                trace!(target: "source", "!= enrolling (present {:?}) at {}", present, serialize(p));
                ((cmp, map), Some(filtered.into_read_zipper(&[])))
            } else {
                unreachable!()
            }
        } else {
            ((cmp, map), None)
        }
    }
}

impl Source for CmpSource {
    fn new(e: Expr) -> Self {
        let cmp = if unsafe { *e.ptr.offset(2) == b'=' } {
            assert!(unsafe { *e.ptr.offset(3) == b'=' });
            0
        } else if unsafe { *e.ptr.offset(2) == b'!' } {
            assert!(unsafe { *e.ptr.offset(3) == b'=' });
            1
        } else {
            // todo < <= #=
            panic!("comparator not implemented")
        };
        // trace!(target: "source", "cmp {cmp} source");
        CmpSource { cmp }
    }

    fn source<'trie, 'path, It: Iterator<Item = Resource<'trie, 'path>>>(
        &self,
        mut it: It,
    ) -> AFactor<'trie, ()>
    where
        'path: 'trie,
    {
        static EQ_PREFIX: [u8; 4] = [
            item_byte(Tag::Arity(3)),
            item_byte(Tag::SymbolSize(2)),
            b'=',
            b'=',
        ];
        static NE_PREFIX: [u8; 4] = [
            item_byte(Tag::Arity(3)),
            item_byte(Tag::SymbolSize(2)),
            b'!',
            b'=',
        ];
        let rz = next_btm_resource(&mut it);
        let map = rz.try_make_map().unwrap();
        let rz = DependentProductZipperG::new_enroll(
            rz,
            (self.cmp, map),
            CmpSource::policy
                as for<'a> fn(
                    (usize, PathMap<()>),
                    &'a [u8],
                    usize,
                )
                    -> ((usize, PathMap<()>), Option<ReadZipperOwned<()>>),
        );
        let rz = PrefixZipper::new(
            if self.cmp == 0 {
                &EQ_PREFIX[..]
            } else if self.cmp == 1 {
                &NE_PREFIX[..]
            } else {
                unreachable!()
            },
            rz,
        );
        AFactor::CmpSource(rz)
    }

    fn request(&self) -> impl Iterator<Item = ResourceRequest> {
        btm_request()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inequality_exclusion_uses_pathmap_subtraction() {
        let mut map = PathMap::new();
        map.insert(b"left", ());
        map.insert(b"right", ());
        map.insert(b"right/deep", ());

        let filtered = CmpSource::exclude_rhs_path(&map, b"right");

        assert_eq!(map.get_val_at(b"right"), Some(&()));
        assert_eq!(filtered.get_val_at(b"left"), Some(&()));
        assert_eq!(filtered.get_val_at(b"right"), None);
        assert_eq!(filtered.get_val_at(b"right/deep"), Some(&()));
    }
}

pub(crate) enum ASource {
    PosSource(BTMSource),
    ACTSource(ACTSource),
    CmpSource(CmpSource),
    CompatSource(CompatSource),
    #[cfg(feature = "z3")]
    Z3Source(Z3Source),
}

#[derive(PolyZipper)]
pub enum AFactor<'trie, V: Clone + Send + Sync + Unpin + 'static = ()> {
    CompatSource(ReadZipperUntracked<'trie, 'trie, V>),
    PosSource(PrefixZipper<'trie, ReadZipperUntracked<'trie, 'trie, V>>),
    ACTSource(PrefixZipper<'trie, ACTMmapZipper<'trie, V>>),
    CmpSource(
        PrefixZipper<
            'trie,
            DependentProductZipperG<
                'trie,
                ReadZipperUntracked<'trie, 'trie, V>,
                ReadZipperOwned<V>,
                V,
                (usize, PathMap<()>),
                for<'a> fn(
                    (usize, PathMap<()>),
                    &'a [u8],
                    usize,
                ) -> ((usize, PathMap<()>), Option<ReadZipperOwned<V>>),
            >,
        >,
    ),
    #[cfg(feature = "z3")]
    Z3Source(PrefixZipper<'trie, ReadZipperOwned<V>>),
}

impl ASource {
    pub fn compat(e: Expr) -> Self {
        ASource::CompatSource(CompatSource::new(e))
    }
}

impl Source for ASource {
    fn new(e: Expr) -> Self {
        if unsafe {
            *e.ptr == item_byte(Tag::Arity(2))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(3))
                && *e.ptr.offset(2) == b'B'
                && *e.ptr.offset(3) == b'T'
                && *e.ptr.offset(4) == b'M'
        } {
            ASource::PosSource(BTMSource::new(e))
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(3))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(3))
                && *e.ptr.offset(2) == b'A'
                && *e.ptr.offset(3) == b'C'
                && *e.ptr.offset(4) == b'T'
        } {
            ASource::ACTSource(ACTSource::new(e))
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(3))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(2))
                && *e.ptr.offset(2) == b'z'
                && *e.ptr.offset(3) == b'3'
        } {
            #[cfg(feature = "z3")]
            return ASource::Z3Source(Z3Source::new(e));
            #[cfg(not(feature = "z3"))]
            panic!(
                "MORK was not built with the z3 feature, yet trying to call {:?}",
                e
            );
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(3))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(2))
                && (*e.ptr.offset(2) == b'=' || *e.ptr.offset(2) == b'!')
                && *e.ptr.offset(3) == b'='
        } {
            ASource::CmpSource(CmpSource::new(e))
        } else {
            unreachable!()
        }
    }

    fn request(&self) -> impl Iterator<Item = ResourceRequest> {
        gen move {
            match self {
                ASource::PosSource(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASource::ACTSource(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASource::CmpSource(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASource::CompatSource(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                #[cfg(feature = "z3")]
                ASource::Z3Source(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
            }
        }
    }

    fn source<'trie, 'path, It: Iterator<Item = Resource<'trie, 'path>>>(
        &self,
        it: It,
    ) -> AFactor<'trie, ()>
    where
        'path: 'trie,
    {
        match self {
            ASource::PosSource(s) => s.source(it),
            ASource::ACTSource(s) => s.source(it),
            ASource::CmpSource(s) => s.source(it),
            ASource::CompatSource(s) => s.source(it),
            #[cfg(feature = "z3")]
            ASource::Z3Source(s) => s.source(it),
        }
    }
}
