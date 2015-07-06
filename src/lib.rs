#![crate_name = "marpa"]

#![feature(plugin_registrar, libc)]

extern crate libc;
extern crate marpa_sys as ffi;

pub use ffi::{Config, Step, ErrorCode};
pub use marpa::{Grammar, Bocage, Order, Recognizer, Tree, Value, Symbol, Rule};

pub mod marpa;

#[cfg(test)]
mod test;
