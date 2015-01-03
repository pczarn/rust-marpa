// #![crate_type = "bin"]
#![feature(phase, unboxed_closures)]

#[phase(plugin, link)]
extern crate "marpa-macros" as marpa_macros;

extern crate marpa;
extern crate regex;

#[phase(plugin, link)]
extern crate regex_macros;

fn apply(op: &str, l: uint, r: uint) -> uint {
    match op {
        "+" => l + r,
        "-" => l - r,
        "*" => l * r,
        "/" => l / r,
        _ => panic!(),
    }
}

fn main() {
    let n = &1u;
    let mut simple = grammar! {
        expr ::= expr op expr { *n } | number { 2u } ;
        // expr ::= expr op:r"[-+*/]" expr { apply(op, l, r) }
        //         | num:r"\d" { num.parse().unwrap() } ;
        // expr ::= number { 2u } | l:expr op:op r:expr { match op { "+" => l + r } } ;
        number ~ r"\d" ;
        op ~ r"[-+*/]" ;
        discard ~ r"\s" ;
    };
    for x in simple.parse("2 - 0 * 3 + 1") {
        println!("{}", x);
    }
}
