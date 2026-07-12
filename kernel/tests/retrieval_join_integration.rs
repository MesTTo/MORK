#![cfg(all(feature = "retrieval_join", feature = "leapfrog"))]

use mork::space::Space;

fn run_program(program: &[u8], leapfrog: bool) -> String {
    mork::zipper_join::set_leapfrog_dispatch(leapfrog);
    let mut s = Space::new();
    s.add_all_sexpr(program).unwrap();
    s.metta_calculus(100);
    mork::zipper_join::set_leapfrog_dispatch(true);
    let mut out = Vec::new();
    s.dump_all_sexpr(&mut out).unwrap();
    let mut lines: Vec<&str> = std::str::from_utf8(&out).unwrap().lines().collect();
    lines.sort();
    lines.join("\n")
}

// The MITM meet shape in miniature: a large-ish factor whose compound column
// carries the join variable, a small schema-table factor keyed by the same
// variable interior, and an arithmetic guard table.
const MEET: &[u8] = b"(sol 9 (c: (i (n a) (i a b)) f1))
(sol 9 (c: (i c d) f2))
(sol 7 (c: (i $x (i $y $x)) f3))
(fwd 1 (i (n $p) (i $p $q)))
(fwd 1 (i $p (i $q $p)))
(lte 1 7)
(lte 1 9)
(exec 0
  (, (sol $s (c: $thm $f))
     (lte 1 $s)
     (fwd 1 $thm))
  (, (hit $s $thm $f)))
";

#[test]
fn meet_shape_retrieval_matches_product_path() {
    let dispatched = run_program(MEET, true);
    let product = run_program(MEET, false);
    let d_hits: Vec<&str> = dispatched.lines().filter(|l| l.starts_with("(hit")).collect();
    let p_hits: Vec<&str> = product.lines().filter(|l| l.starts_with("(hit")).collect();
    assert_eq!(d_hits, p_hits, "retrieval path diverges from product path\nDISPATCHED:\n{dispatched}\n\nPRODUCT:\n{product}");
    assert!(!p_hits.is_empty(), "reference must produce hits:\n{product}");
}

// The real MITM split-rule shape: multiple small arithmetic guard tables whose
// variables chain through each other and feed the template.
const SPLIT: &[u8] = b"(sol 9 1 (c: (-> (i p q) r) I))
(gtFn 9 6)
(decFn 9 8)
(incFn 1 2)
(lte 1 8)
(exec 0
  (, (sol $ski $hi (c: (-> $b $c) $f))
     (gtFn $ski 6)
     (decFn $ski $ki)
     (lte $hi $ki)
     (incFn $hi $shi))
  (, (sol $ki $shi (c: (-> (i $a $b) (-> $a $c)) (m $f)))))
";

#[test]
fn split_shape_retrieval_matches_product_path() {
    for cap in ["1", "2", "3", "4"] {
        unsafe { std::env::set_var("MORK_RETRIEVAL_MAX_REMOVED", cap) };
        let dispatched = run_program(SPLIT, true);
        unsafe { std::env::remove_var("MORK_RETRIEVAL_MAX_REMOVED") };
        let product = run_program(SPLIT, false);
        let d: Vec<&str> = dispatched.lines().filter(|l| l.starts_with("(sol 8")).collect();
        let p: Vec<&str> = product.lines().filter(|l| l.starts_with("(sol 8")).collect();
        assert!(!p.is_empty(), "reference must derive the split result:\n{product}");
        assert_eq!(d, p, "cap={cap} diverges\nDISPATCHED:\n{dispatched}\nPRODUCT:\n{product}");
    }
}
