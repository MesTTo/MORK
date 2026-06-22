use std::collections::BTreeMap;

use pathmap::PathMap;
use pathmap::zipper::{ZipperAbsolutePath, ZipperMoving, ZipperValues, ZipperWriting};

fn sample_map() -> PathMap<u8> {
    let mut map = PathMap::new();
    map.insert(b"alpha/x", 1);
    map.insert(b"alpha/y", 2);
    map.insert(b"beta/x", 3);
    map.insert(b"beta/z/deep", 4);
    map.insert(b"beta/z/other", 5);
    map
}

fn value_paths(map: &PathMap<u8>) -> BTreeMap<Vec<u8>, u8> {
    let mut values = BTreeMap::new();
    map.for_each_value(|path, value| {
        values.insert(path.to_vec(), *value);
    });
    values
}

fn value_paths_u32(map: &PathMap<u32>) -> BTreeMap<Vec<u8>, u32> {
    let mut values = BTreeMap::new();
    map.for_each_value(|path, value| {
        values.insert(path.to_vec(), *value);
    });
    values
}

fn remove_value_at(map: &mut PathMap<u32>, path: &[u8]) -> Option<u32> {
    let mut zipper = map.write_zipper();
    zipper.move_to_path(path);
    zipper.remove_val(true)
}

fn next_xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn deterministic_key(state: &mut u64) -> Vec<u8> {
    let length = 1 + (next_xorshift64(state) as usize % 31);
    (0..length).map(|_| next_xorshift64(state) as u8).collect()
}

#[test]
fn read_zipper_origin_path_composes_root_prefix_and_relative_focus() {
    let map = sample_map();
    let mut zipper = map.read_zipper_at_path(b"alpha/");

    assert_eq!(zipper.root_prefix_path(), b"alpha/");
    assert_eq!(zipper.path(), b"");
    assert_eq!(zipper.origin_path(), b"alpha/");

    zipper.descend_to(b"x");

    assert_eq!(zipper.path(), b"x");
    assert_eq!(zipper.origin_path(), b"alpha/x");
    assert_eq!(zipper.val(), Some(&1));

    zipper.reset();

    assert_eq!(zipper.path(), b"");
    assert_eq!(zipper.origin_path(), zipper.root_prefix_path());
}

#[test]
fn read_zipper_move_to_path_reuses_common_context_and_preserves_values() {
    let map = sample_map();
    let mut zipper = map.read_zipper_at_path(b"beta/");

    zipper.descend_to(b"z/deep");
    assert_eq!(zipper.val(), Some(&4));

    let overlap = zipper.move_to_path(b"z/other");

    assert_eq!(overlap, b"z/".len());
    assert_eq!(zipper.path(), b"z/other");
    assert_eq!(zipper.origin_path(), b"beta/z/other");
    assert_eq!(zipper.val(), Some(&5));

    let overlap = zipper.move_to_path(b"x");

    assert_eq!(overlap, 0);
    assert_eq!(zipper.path(), b"x");
    assert_eq!(zipper.origin_path(), b"beta/x");
    assert_eq!(zipper.val(), Some(&3));
}

#[test]
fn write_zipper_focus_update_preserves_unrelated_context() {
    let mut map = sample_map();

    {
        let mut zipper = map.write_zipper_at_path(b"alpha/");
        assert_eq!(zipper.root_prefix_path(), b"alpha/");

        zipper.descend_to(b"new");
        assert_eq!(zipper.origin_path(), b"alpha/new");
        assert_eq!(zipper.set_val(9), None);

        assert_eq!(zipper.move_to_path(b"x"), 0);
        assert_eq!(zipper.val(), Some(&1));
        assert_eq!(zipper.set_val(7), Some(1));
    }

    assert_eq!(map.get_val_at(b"alpha/x"), Some(&7));
    assert_eq!(map.get_val_at(b"alpha/y"), Some(&2));
    assert_eq!(map.get_val_at(b"alpha/new"), Some(&9));
    assert_eq!(map.get_val_at(b"beta/x"), Some(&3));
    assert_eq!(map.get_val_at(b"beta/z/deep"), Some(&4));
    assert_eq!(map.get_val_at(b"beta/z/other"), Some(&5));
}

#[test]
fn deterministic_pathmap_mutation_model_matches_btree_map() {
    let mut map = PathMap::new();
    let mut model = BTreeMap::<Vec<u8>, u32>::new();
    let mut state = 0x243f_6a88_85a3_08d3u64;

    for step in 0..8_000u32 {
        let key = deterministic_key(&mut state);
        match step % 5 {
            0..=2 => {
                assert_eq!(map.insert(&key, step), model.insert(key, step));
            }
            3 => {
                assert_eq!(remove_value_at(&mut map, &key), model.remove(&key));
            }
            _ => {
                assert_eq!(map.get_val_at(&key).copied(), model.get(&key).copied());
            }
        }

        if step % 97 == 0 {
            assert_eq!(value_paths_u32(&map), model);
        }
    }

    assert_eq!(value_paths_u32(&map), model);
}

#[test]
fn cloned_pathmap_snapshot_is_not_changed_by_later_write_zipper_mutation() {
    let mut map = sample_map();
    let snapshot = map.clone();

    {
        let mut zipper = map.write_zipper();
        zipper.descend_to(b"alpha/x");
        assert_eq!(zipper.set_val(11), Some(1));
        zipper.move_to_path(b"gamma/root");
        assert_eq!(zipper.set_val(12), None);
    }

    assert_eq!(snapshot.get_val_at(b"alpha/x"), Some(&1));
    assert_eq!(snapshot.get_val_at(b"gamma/root"), None);
    assert_eq!(map.get_val_at(b"alpha/x"), Some(&11));
    assert_eq!(map.get_val_at(b"gamma/root"), Some(&12));
}

#[test]
fn write_zipper_take_and_graft_preserve_surrounding_context() {
    let mut map = sample_map();
    let alpha_subtrie = {
        let mut zipper = map.write_zipper_at_path(b"alpha/");
        zipper
            .take_map(true)
            .expect("alpha subtrie should exist and be extractable")
    };

    assert_eq!(
        value_paths(&alpha_subtrie),
        BTreeMap::from([(b"x".to_vec(), 1), (b"y".to_vec(), 2),])
    );
    assert_eq!(map.get_val_at(b"alpha/x"), None);
    assert_eq!(map.get_val_at(b"beta/x"), Some(&3));

    {
        let mut zipper = map.write_zipper_at_path(b"gamma/");
        zipper.graft_map(alpha_subtrie);
    }

    assert_eq!(map.get_val_at(b"gamma/x"), Some(&1));
    assert_eq!(map.get_val_at(b"gamma/y"), Some(&2));
    assert_eq!(map.get_val_at(b"beta/z/deep"), Some(&4));
    assert_eq!(map.get_val_at(b"beta/z/other"), Some(&5));
}
