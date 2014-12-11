// #![crate_type = "bin"]
#![feature(phase, overloaded_calls)]

#[phase(plugin, link)]
extern crate "marpa-macros" as marpa_macros;

extern crate marpa;
extern crate regex;

#[phase(plugin, link)]
extern crate regex_macros;

fn main() {
    let g = grammar! {
        expr ::= expr op expr | number ;
        number ~ r"\d" ;
        op ~ r"[-+*/]" ;
    };
    g("2 - 0 * 3 + 1");
}
