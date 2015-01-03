// #![crate_type = "bin"]
#![feature(phase, unboxed_closures)]

#[phase(plugin, link)]
extern crate "marpa-macros" as marpa_macros;

extern crate marpa;
extern crate regex;

#[phase(plugin, link)]
extern crate regex_macros;

fn main() {
    let mut simple = grammar! {
        expr ::= expr op expr { 1u } | number { 2u } ;
        number ~ r"\d" ;
        op ~ r"[-+*/]" ;
    };
    let x: uint = simple.parse("2 - 0 * 3 + 1");
    println!("{}", x);
}
