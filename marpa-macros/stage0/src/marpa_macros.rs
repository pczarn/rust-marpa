#![crate_name = "marpa_macros"]
#![feature(plugin, plugin_registrar, unboxed_closures, quote, box_syntax,
    rustc_private, box_patterns, trace_macros, core, collections)]
#![plugin(marpa_macros)]

extern crate rustc;
extern crate syntax;
extern crate marpa;
#[macro_use] extern crate marpa_macros;

pub use foo::plugin_registrar;

stage0!();
