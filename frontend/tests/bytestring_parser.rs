use mork_expr::{Expr, ExprZipper, Tag, byte_item};
use mork_frontend::bytestring_parser::{Context, Parser};

struct IdentityParser;

impl Parser for IdentityParser {
    fn tokenizer<'r>(&'r mut self, symbol: &'r [u8]) -> &'r [u8] {
        symbol
    }
}

fn parse_one(input: &[u8]) -> Vec<u8> {
    let mut parser = IdentityParser;
    let mut context = Context::new(input);
    let mut output = vec![0_u8; 256];
    let mut zipper = ExprZipper::new(Expr {
        ptr: output.as_mut_ptr(),
    });

    parser.sexpr(&mut context, &mut zipper).unwrap();
    output.truncate(zipper.loc);
    output
}

fn parse_one_vec(input: &[u8]) -> Vec<u8> {
    let mut parser = IdentityParser;
    let mut context = Context::new(input);
    let mut output = Vec::new();

    parser.sexpr_to_vec(&mut context, &mut output).unwrap();
    output
}

fn render(bytes: &mut [u8]) -> String {
    format!(
        "{:?}",
        Expr {
            ptr: bytes.as_mut_ptr()
        }
    )
}

#[test]
fn vector_parser_matches_zipper_parser_for_named_variables() {
    let input = b"(Implies (Human $x) (Mortal $x))";

    assert_eq!(parse_one_vec(input), parse_one(input));
}

#[test]
fn vector_parser_grows_for_large_flat_expression() {
    let mut input = String::from("(");
    for i in 0..63 {
        if i > 0 {
            input.push(' ');
        }
        input.push_str(&format!("symbol{i:02}"));
    }
    input.push(')');

    let parsed = parse_one_vec(input.as_bytes());

    assert!(parsed.len() > 256, "encoded length was {}", parsed.len());
    assert_eq!(byte_item(parsed[0]), Tag::Arity(63));
}

#[test]
fn named_variables_lower_to_alpha_equivalent_nameless_bytes() {
    let mut first = parse_one(b"(Implies (Human $x) (Mortal $x))");
    let second = parse_one(b"(Implies (Human $subject) (Mortal $subject))");

    assert_eq!(first, second);
    assert_eq!(render(&mut first), "(Implies (Human $) (Mortal _1))");
}

#[test]
fn named_variables_preserve_distinct_repeated_slots() {
    let mut parsed = parse_one(b"(Pair $left $right $left $right)");

    assert_eq!(render(&mut parsed), "(Pair $ $ _1 _2)");
}
