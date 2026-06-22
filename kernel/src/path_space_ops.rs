use pathmap::PathMap;

/// Keeps accepted paths that have no accepted strict prefix.
///
/// This is a finite-pathset reference helper for the MORK `^space`
/// subsumption operation. It intentionally materializes path lists, so it is a
/// semantic target for future zipper/residual-DAG implementations rather than a
/// hot-path kernel.
pub fn prefix_minimal_values(map: &PathMap<()>) -> PathMap<()> {
    paths_to_map(&prefix_minimal_sorted_paths(sorted_paths(map)))
}

/// Keeps accepted paths that have no accepted strict extension.
///
/// This is a finite-pathset reference helper for the MORK `v space`
/// instantiation operation.
pub fn prefix_maximal_values(map: &PathMap<()>) -> PathMap<()> {
    paths_to_map(&prefix_maximal_sorted_paths(sorted_paths(map)))
}

/// Computes the prefix-maximal common-prefix witnesses between accepted paths.
///
/// This is the compact reference oracle for `sharing` semantics: collect
/// cross-space longest common prefixes, then keep only the prefix-maximal
/// witnesses. Lexicographic order makes the all-pairs scan unnecessary: a
/// non-adjacent cross-space pair's common prefix is also shared by every path
/// between them, so some adjacent cross-space boundary has an equal or longer
/// common prefix.
pub fn shared_prefix_witnesses(left: &PathMap<()>, right: &PathMap<()>) -> PathMap<()> {
    let left_paths = sorted_paths(left);
    let right_paths = sorted_paths(right);
    let mut witnesses = Vec::new();
    let mut left_index = 0;
    let mut right_index = 0;
    let mut previous: Option<TaggedPath<'_>> = None;

    while left_index < left_paths.len() || right_index < right_paths.len() {
        let current = if right_index == right_paths.len()
            || left_index < left_paths.len()
                && left_paths[left_index].as_slice() <= right_paths[right_index].as_slice()
        {
            let tagged = TaggedPath {
                side: PathSide::Left,
                path: left_paths[left_index].as_slice(),
            };
            left_index += 1;
            tagged
        } else {
            let tagged = TaggedPath {
                side: PathSide::Right,
                path: right_paths[right_index].as_slice(),
            };
            right_index += 1;
            tagged
        };

        if let Some(previous) = previous
            && previous.side != current.side
        {
            let len = common_prefix_len(previous.path, current.path);
            witnesses.push(previous.path[..len].to_vec());
        }
        previous = Some(current);
    }

    paths_to_map(&prefix_maximal_sorted_paths(witnesses))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PathSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug)]
struct TaggedPath<'a> {
    side: PathSide,
    path: &'a [u8],
}

fn sorted_paths(map: &PathMap<()>) -> Vec<Vec<u8>> {
    let mut paths = Vec::new();
    map.for_each_value(|path, _| paths.push(path.to_vec()));
    paths.sort();
    paths.dedup();
    paths
}

fn paths_to_map(paths: &[Vec<u8>]) -> PathMap<()> {
    let mut map = PathMap::new();
    for path in paths {
        map.insert(path, ());
    }
    map
}

fn prefix_minimal_sorted_paths(paths: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut minimal: Vec<Vec<u8>> = Vec::new();
    for path in paths {
        if !minimal
            .last()
            .is_some_and(|prefix| is_strict_prefix(prefix, &path))
        {
            minimal.push(path);
        }
    }
    minimal
}

fn prefix_maximal_sorted_paths(paths: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut paths = paths;
    paths.sort();
    paths.dedup();

    paths
        .iter()
        .enumerate()
        .filter_map(|(index, path)| {
            let has_extension = paths
                .get(index + 1)
                .is_some_and(|next| is_strict_prefix(path, next));
            (!has_extension).then_some(path.clone())
        })
        .collect()
}

fn is_strict_prefix(prefix: &[u8], path: &[u8]) -> bool {
    prefix.len() < path.len() && path.starts_with(prefix)
}

fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left, right)| left == right)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    type Paths = &'static [&'static [u8]];

    fn map(paths: Paths) -> PathMap<()> {
        PathMap::from_iter(paths.iter().copied())
    }

    fn paths(map: &PathMap<()>) -> Vec<Vec<u8>> {
        sorted_paths(map)
    }

    fn map_owned(paths: &[Vec<u8>]) -> PathMap<()> {
        let mut map = PathMap::new();
        for path in paths {
            map.insert(path, ());
        }
        map
    }

    fn brute_shared_prefix_witnesses(left: &PathMap<()>, right: &PathMap<()>) -> PathMap<()> {
        let left_paths = sorted_paths(left);
        let right_paths = sorted_paths(right);
        let mut witnesses = Vec::new();

        for left_path in &left_paths {
            for right_path in &right_paths {
                let len = common_prefix_len(left_path, right_path);
                witnesses.push(left_path[..len].to_vec());
            }
        }

        paths_to_map(&prefix_maximal_sorted_paths(witnesses))
    }

    #[test]
    fn prefix_minimal_values_stop_below_accepted_ancestor() {
        let source = map(&[b"foo", b"foo/bar", b"foo/bar/baz", b"other"]);

        assert_eq!(
            paths(&prefix_minimal_values(&source)),
            vec![b"foo".to_vec(), b"other".to_vec()]
        );
    }

    #[test]
    fn prefix_minimal_values_use_sorted_ancestor_run() {
        let source = map(&[b"a", b"a/1", b"a/2", b"b", b"b/1", b"c/d"]);

        assert_eq!(
            paths(&prefix_minimal_values(&source)),
            vec![b"a".to_vec(), b"b".to_vec(), b"c/d".to_vec()]
        );
    }

    #[test]
    fn prefix_maximal_values_keep_deepest_accepted_descendants() {
        let source = map(&[b"foo", b"foo/bar", b"foo/bar/baz", b"other"]);

        assert_eq!(
            paths(&prefix_maximal_values(&source)),
            vec![b"foo/bar/baz".to_vec(), b"other".to_vec()]
        );
    }

    #[test]
    fn empty_value_subsumes_and_is_not_instantiated_when_descendant_exists() {
        let source = map(&[b"", b"a", b"ab"]);

        assert_eq!(
            paths(&prefix_minimal_values(&source)),
            vec![Vec::<u8>::new()]
        );
        assert_eq!(paths(&prefix_maximal_values(&source)), vec![b"ab".to_vec()]);
    }

    #[test]
    fn shared_prefix_witnesses_keep_maximal_common_prefixes() {
        let left = map(&[b"alpha/red", b"beta"]);
        let right = map(&[b"alpha/blue", b"betamax"]);

        assert_eq!(
            paths(&shared_prefix_witnesses(&left, &right)),
            vec![b"alpha/".to_vec(), b"beta".to_vec()]
        );
    }

    #[test]
    fn shared_prefix_witnesses_can_return_empty_root_witness() {
        let left = map(&[b"alpha"]);
        let right = map(&[b"beta"]);

        assert_eq!(
            paths(&shared_prefix_witnesses(&left, &right)),
            vec![Vec::<u8>::new()]
        );
    }

    #[test]
    fn shared_prefix_witnesses_match_pairwise_oracle_on_small_pathsets() {
        let universe = vec![
            Vec::new(),
            b"a".to_vec(),
            b"aa".to_vec(),
            b"ab".to_vec(),
            b"b".to_vec(),
            b"ba".to_vec(),
            b"bb".to_vec(),
        ];

        for left_mask in 0usize..(1usize << universe.len()) {
            let left_paths = subset(&universe, left_mask);
            let left = map_owned(&left_paths);

            for right_mask in 0usize..(1usize << universe.len()) {
                let right_paths = subset(&universe, right_mask);
                let right = map_owned(&right_paths);

                assert_eq!(
                    paths(&shared_prefix_witnesses(&left, &right)),
                    paths(&brute_shared_prefix_witnesses(&left, &right)),
                    "left_mask={left_mask} right_mask={right_mask}"
                );
            }
        }
    }

    fn subset(universe: &[Vec<u8>], mask: usize) -> Vec<Vec<u8>> {
        universe
            .iter()
            .enumerate()
            .filter_map(|(index, path)| ((mask & (1usize << index)) != 0).then_some(path.clone()))
            .collect()
    }
}
