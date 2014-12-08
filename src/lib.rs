#![crate_name = "marpa"]

#![feature(plugin_registrar, macro_rules, unsafe_destructor)]

extern crate libc;

pub use ffi::{Config, Step};
pub use marpa::{Grammar, Bocage, Order, Recognizer, Tree, Value};

pub mod ffi;
pub mod marpa;

#[cfg(test)]
mod test;
