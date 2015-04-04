// #![crate_name = "marpa_macros"]

// #[macro_export]
// macro_rules! grammar {
// 	( $($x:tt)* ) => (
// 		include!("../../stage-minus-one/src/grammar-expanded.rs")
// 	)
// }

#![crate_name = "marpa_macros"]
#![feature(plugin, plugin_registrar, unboxed_closures, quote, box_syntax,
    rustc_private, box_patterns, trace_macros, core, collections)]

extern crate rustc;
extern crate syntax;
extern crate marpa;

use syntax::ast;
use syntax::ast::TokenTree;
use syntax::codemap::{respan, DUMMY_SP};
use syntax::codemap::Span;
use syntax::ext::base::{ExtCtxt, MacResult, MacEager};
pub use syntax::parse::token;
use syntax::parse;
use syntax::ptr::P;
use syntax::ext::build::AstBuilder;
use syntax::print::pprust;
use syntax::parse::lexer;
use syntax::ext::tt::transcribe;
use syntax::ext::base;
pub use syntax::parse::token::*;
use syntax::owned_slice::OwnedSlice;
use syntax::codemap;

use syntax::util::small_vector::SmallVector;

use std::path::{Path, PathBuf};

use syntax::ast::{Ty_, Ty, PathParameters, AngleBracketedParameters, AngleBracketedParameterData, TyPath,
    PathSegment, Block, TtToken, Pat, Ident, TyTup, TyInfer};

use std::collections::HashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::HashSet;
use std::mem;
use std::fmt;
use std::iter;
use std::iter::AdditiveIterator;
use std::iter::Iterator;
use std::iter::IteratorExt;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut ::rustc::plugin::Registry) {
    reg.register_macro("stage0", expand_stage0);
}

fn expand_stage0(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree])
                  -> Box<MacResult+'static> {
    let sess = cx.parse_sess();

    let src = include_str!("../../src/marpa_macros.rs");
    let subst = include_str!("../src/grammar-expanded.rs");

    let begin = src.find("// EXPAND:").unwrap();
    let end   = src.find("// :EXPAND").unwrap();

    let src = format!("pub mod foo {{ {}{}{} }}", &src[..begin], subst, &src[end..]);

   	let mut expr =
   		parse::parse_item_from_source_str(
			"foo".to_string(),
			src,
			cx.cfg(),
   			sess,
		);
   	MacEager::items(SmallVector::one(expr.unwrap()))
}
