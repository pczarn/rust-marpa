#![feature(plugin, unboxed_closures, associated_types)]

#[plugin]
extern crate "marpa-macros" as marpa_macros;

extern crate marpa;
extern crate regex;

#[plugin]
#[no_link]
extern crate regex_macros;

fn apply(op: &str, l: i32, r: i32) -> i32 {
    match op {
        "+" => l + r,
        "-" => l - r,
        "*" => l * r,
        "/" => l / r,
        _ => panic!(),
    }
}

fn main() {
    let mut simple = grammar! {
        expr ::=  l:expr op:r"[-+*/]" r:expr { apply(op, l, r) }
                | num:r"\d" { num.parse().unwrap() } ;
        discard ~ r"\s" ;
    };
    for result in simple.parses_iter("2 - 0 * 3 + 1") {
        println!("{}", result);
    }
}
