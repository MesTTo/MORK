use std::collections::BTreeSet;

use mork::binding_plan::{BindingAccessPlan, BindingSidecarPlan, PatternProjection};
use mork::binding_space::BindingVar;
use mork::space::Space;
use mork::term_identity::TermIdentitySidecar;

#[test]
fn json_to_paths_round_trips_owned_streamed_paths() {
    let json = br#"{
        "name": "Ada",
        "active": true,
        "scores": [1, 2],
        "empty": {}
    }"#;

    let mut encoded = Vec::new();
    let mut source = Space::new();
    let written = source.json_to_paths(json, &mut encoded).unwrap();
    assert!(written > 0);

    let mut restored = Space::new();
    pathmap::paths_serialization::deserialize_paths(
        restored.btm.write_zipper(),
        std::io::Cursor::new(&encoded),
        (),
    )
    .unwrap();

    let mut output = Vec::new();
    restored.dump_all_sexpr(&mut output).unwrap();
    let output = String::from_utf8(output).unwrap();

    assert!(output.contains("(name Ada)"), "{output}");
    assert!(output.contains("(active true)"), "{output}");
    assert!(output.contains("(scores (0 1))"), "{output}");
    assert!(output.contains("(scores (1 2))"), "{output}");
}

#[test]
fn singular_json_path_value_pattern_queries_loaded_json_with_binding_sidecar() {
    let json = br#"{
        "profile": {
            "name": "Ada",
            "scores": [1, 2],
            "tags": ["engineer", "math"]
        },
        "active": true
    }"#;

    let mut space = Space::new();
    assert_eq!(space.load_json(json).unwrap(), 6);

    let pattern = space
        .singular_json_path_value_pattern("$.profile.scores[1]")
        .unwrap();
    let mut sidecar = TermIdentitySidecar::new();
    let pattern = sidecar.insert_term(&pattern).unwrap();
    sidecar.extend_from_pathmap(&space.btm).unwrap();

    let plan = BindingSidecarPlan::new(
        [BindingAccessPlan::Pattern {
            pattern,
            projection: PatternProjection::new([BindingVar(0)], [0]).unwrap(),
        }],
        [BindingVar(0)],
    );
    let result = plan.execute_trie_join(&sidecar).unwrap();
    let product_pattern = mork::expr!(space, "[2] , [2] profile [2] scores [2] 1 $");
    let product_count = Space::query_multi(&space.btm, product_pattern, |_, _| true);
    let expected_value = {
        let encoded = mork::expr!(space, "2");
        sidecar
            .term_id_for_encoded(unsafe { encoded.span().as_ref().unwrap() })
            .unwrap()
    };
    let values = result
        .relation
        .positive_rows()
        .map(|row| row[0])
        .collect::<BTreeSet<_>>();

    assert_eq!(product_count, 1);
    assert_eq!(values, BTreeSet::from([expected_value]));
    assert_eq!(result.relation.positive_rows().count(), product_count);
    assert_eq!(result.stats.expression_trie_builds, 1);
    assert_eq!(result.stats.expression_trie_candidates, 1);
    assert_eq!(result.stats.pattern_matches, 1);
    assert_eq!(result.trie_stats.output_rows, 1);
}
