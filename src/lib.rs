#![crate_name = "marpa"]

#![feature(plugin_registrar, unsafe_destructor, libc)]

extern crate libc;

pub use ffi::{Config, Step, ErrorCode};
pub use marpa::{Grammar, Bocage, Order, Recognizer, Tree, Value, Symbol, Rule};

pub mod ffi;
pub mod marpa;

#[cfg(test)]
mod test;
