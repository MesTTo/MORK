#![cfg(feature = "eqrel")]

use std::collections::{BTreeMap, BTreeSet};

use mork::__mork_expr::{Tag, item_byte};
use mork::eqrel::EqRel;
use mork::expr;
use mork::space::Space;

#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.0 = x;
        x
    }

    fn usize(&mut self, upper: usize) -> usize {
        (self.next() as usize) % upper
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            values.swap(i, self.usize(i + 1));
        }
    }
}

struct RefUf {
    parents: Vec<usize>,
    ranks: Vec<u8>,
    seen: Vec<bool>,
}

impl RefUf {
    fn new(len: usize) -> Self {
        Self {
            parents: (0..len).collect(),
            ranks: vec![0; len],
            seen: vec![false; len],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parents[value] != value {
            let root = self.find(self.parents[value]);
            self.parents[value] = root;
        }
        self.parents[value]
    }

    fn union(&mut self, left: usize, right: usize) {
        self.seen[left] = true;
        self.seen[right] = true;
        let mut left = self.find(left);
        let mut right = self.find(right);
        if left == right {
            return;
        }
        if self.ranks[left] < self.ranks[right] {
            std::mem::swap(&mut left, &mut right);
        }
        self.parents[right] = left;
        if self.ranks[left] == self.ranks[right] {
            self.ranks[left] += 1;
        }
    }

    fn same_class_pairs(&mut self) -> BTreeSet<(Vec<u8>, Vec<u8>)> {
        let mut out = BTreeSet::new();
        for left in 0..self.parents.len() {
            if !self.seen[left] {
                continue;
            }
            for right in 0..self.parents.len() {
                if self.seen[right] && self.find(left) == self.find(right) {
                    out.insert((symbol_bytes(left), symbol_bytes(right)));
                }
            }
        }
        out
    }
}

fn symbol_name(index: usize) -> String {
    format!("n{index:04}")
}

fn symbol_bytes(index: usize) -> Vec<u8> {
    let name = symbol_name(index);
    let mut out = vec![item_byte(Tag::SymbolSize(name.len() as u8))];
    out.extend_from_slice(name.as_bytes());
    out
}

fn generated_sequence(seed: u64) -> (usize, Vec<(usize, usize)>) {
    let mut rng = Rng::new(seed);
    let n = 24 + rng.usize(72);
    let mut unions = Vec::new();
    match seed % 6 {
        0 => {
            for i in 0..n - 1 {
                unions.push((i, i + 1));
            }
            for _ in 0..n / 2 {
                let a = rng.usize(n);
                unions.push((a, a));
                unions.push((a, (a + 1) % n));
            }
        }
        1 => {
            let center = rng.usize(n);
            for i in 0..n {
                unions.push((center, i));
            }
            for _ in 0..n {
                unions.push((center, rng.usize(n)));
            }
        }
        2 => {
            for _ in 0..n * 4 {
                let a = rng.usize(n);
                let b = if rng.usize(5) == 0 { a } else { rng.usize(n) };
                unions.push((a, b));
                if rng.usize(3) == 0 {
                    unions.push((a, b));
                }
            }
        }
        3 => {
            let split = n / 2;
            for i in 0..split - 1 {
                unions.push((i, i + 1));
                unions.push((split + i, split + i + 1));
            }
            for i in 0..split.min(n - split) {
                unions.push((i, split + i));
            }
        }
        4 => {
            let width = 5;
            for base in (0..n).step_by(width) {
                let end = (base + width).min(n);
                unions.push((base, base));
                for i in base + 1..end {
                    unions.push((base, i));
                }
            }
        }
        _ => {
            for _ in 0..n * 3 {
                unions.push((rng.usize(n), rng.usize(n)));
            }
        }
    }
    rng.shuffle(&mut unions);
    (n, unions)
}

fn same_class_from_quotient(quotient: &[(Vec<u8>, Vec<u8>)]) -> BTreeSet<(Vec<u8>, Vec<u8>)> {
    let mut by_rep = BTreeMap::<Vec<u8>, Vec<Vec<u8>>>::new();
    for (element, rep) in quotient {
        by_rep.entry(rep.clone()).or_default().push(element.clone());
    }
    let mut out = BTreeSet::new();
    for elements in by_rep.values() {
        for left in elements {
            for right in elements {
                out.insert((left.clone(), right.clone()));
            }
        }
    }
    out
}

#[test]
fn quotient_same_class_matches_naive_reference_for_seeded_sequences() {
    for seed in 0..1_200 {
        let (n, unions) = generated_sequence(seed);
        let mut eqrel = EqRel::new();
        let mut reference = RefUf::new(n);
        for (left, right) in unions {
            eqrel.union(&symbol_bytes(left), &symbol_bytes(right));
            reference.union(left, right);
        }

        let actual = same_class_from_quotient(&eqrel.quotient_pairs());
        let expected = reference.same_class_pairs();
        assert_eq!(actual, expected, "seed {seed}");
    }
}

fn run(program: &[u8], steps: usize) -> Space {
    let mut space = Space::new();
    space.add_all_sexpr(program).unwrap();
    space.metta_calculus(steps);
    space
}

fn dump_selection(space: &Space, query: &str, template: &str) -> String {
    let mut output = Vec::new();
    space.dump_sexpr(expr!(space, query), expr!(space, template), &mut output);
    String::from_utf8(output).unwrap()
}

fn eqrel_program(unions: &[(usize, usize)]) -> String {
    let mut program = String::new();
    for (left, right) in unions {
        program.push_str(&format!(
            "(u {} {})\n",
            symbol_name(*left),
            symbol_name(*right)
        ));
    }
    program.push_str("(exec 0 (, (u $x $y)) (O (eqrel $x $y (q $e $r))))\n");
    program
}

#[test]
fn sink_emits_identical_quotient_bytes_for_shuffled_unions() {
    let mut unions = Vec::new();
    for i in 0..47 {
        unions.push((i, i + 1));
    }
    for i in 0..24 {
        unions.push((i, 47 - i));
        unions.push((i, i));
        unions.push((i, 47 - i));
    }

    let mut expected = None;
    for seed in 0..6 {
        let mut shuffled = unions.clone();
        Rng::new(seed).shuffle(&mut shuffled);
        let program = eqrel_program(&shuffled);
        let space = run(program.as_bytes(), 20);
        let dump = dump_selection(&space, "[3] q $ $", "[3] q _1 _2");
        if let Some(expected) = &expected {
            assert_eq!(&dump, expected, "shuffle seed {seed}");
        } else {
            expected = Some(dump);
        }
    }
}

fn relation_pairs(output: &str, relation: &str) -> BTreeSet<(String, String)> {
    let mut pairs = BTreeSet::new();
    for line in output.lines() {
        let line = line
            .strip_prefix('(')
            .and_then(|line| line.strip_suffix(')'))
            .unwrap_or(line);
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(parts.len(), 3, "bad pair line: {line}");
        assert_eq!(parts[0], relation, "wrong relation in line: {line}");
        pairs.insert((parts[1].to_string(), parts[2].to_string()));
    }
    pairs
}

#[test]
fn quotient_join_matches_materialized_closure_same_class_queries() {
    let closure = run(
        br#"
(eq a b)
(eq b c)
(eq c d)
(exec 0 (, (eq $x $y))
        (, (same-closure $x $y)
           (same-closure $y $x)
           (same-closure $x $x)
           (same-closure $y $y)))
(exec 1 (, (same-closure $x $y) (same-closure $y $z))
        (, (same-closure $x $z)))
(exec 2 (, (same-closure $x $y) (same-closure $y $z))
        (, (same-closure $x $z)))
(exec 3 (, (same-closure $x $y) (same-closure $y $z))
        (, (same-closure $x $z)))
"#,
        200,
    );
    let closure_pairs = relation_pairs(
        &dump_selection(&closure, "[3] same-closure $ $", "[3] same-closure _1 _2"),
        "same-closure",
    );

    let quotient = run(
        br#"
(eq a b)
(eq b c)
(eq c d)
(exec 0 (, (eq $x $y)) (O (eqrel $x $y (quot $e $r))))
(exec 1 (, (quot $x $r) (quot $y $r))
        (, (same-quot $x $y)))
"#,
        200,
    );
    let quotient_pairs = relation_pairs(
        &dump_selection(&quotient, "[3] same-quot $ $", "[3] same-quot _1 _2"),
        "same-quot",
    );

    assert_eq!(quotient_pairs, closure_pairs);
    assert_eq!(closure_pairs.len(), 16);
}
