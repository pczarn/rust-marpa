// #![crate_type = "dylib"]
#![feature(phase, overloaded_calls)]

#[phase(plugin, link)]
extern crate "marpa-macros" as marpa_macros;

extern crate marpa;
extern crate regex;

#[phase(plugin, link)]
extern crate regex_macros;

// #[export_lua_module]
// pub mod mylib {
//     static PI:f32 = 3.141592;

//     fn function1(a: int, b: int) -> int { a + b }

//     fn function2(a: int) -> int { a + 5 }

//     #[lua_module_init]
//     fn init() {
//         println!("mylib is now loaded!")
//     }
// }

fn main() {
    let g = grammar! {
        expr ::= expr op expr | number ;
        number ~ r"\d" ;
        op ~ r"[-+*/]" ;
    };
    g("2 - 0 * 3 + 1");
    // g("5*2");
    // println!("{}", g);
}
