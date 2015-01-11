#![feature(plugin, unboxed_closures, associated_types)]

#[plugin]
extern crate "marpa-macros" as marpa_macros;

extern crate marpa;
extern crate regex;

#[plugin]
#[no_link]
extern crate regex_macros;

extern crate syntax;

use syntax::parse::token::str_to_ident;
use syntax::ast;
use syntax::codemap;
use syntax::ptr::P;

use syntax::ast::{Ident, Expr, Path, PathSegment, PathParameters};

fn psp<T: 'static>(arg: T) -> P<codemap::Spanned<T>> {
    P(codemap::dummy_spanned(arg))
}

fn sp<T>(arg: T) -> codemap::Spanned<T> {
    codemap::dummy_spanned(arg)
}

fn mk_expr(node: ast::Expr_) -> P<Expr> {
    P(Expr {
        id: ast::DUMMY_NODE_ID,
        node: node,
        span: codemap::DUMMY_SP,
    })
}

fn path_segment(i: Ident) -> PathSegment {
    ast::PathSegment { identifier: i, parameters: PathParameters::none() }
}

fn main() {
    let mut rust = grammar! {
        primary_expr ::=
              li:lit_integer -> P<Expr> { mk_expr(ast::ExprLit(li)) }
            | p:path -> _ { mk_expr(ast::ExprPath(p)) } ;

        path ::=
              "::" ps:path_segment
                -> Path { Path { segments: ps, global: true, span: codemap::DUMMY_SP } }
            | ps:path_segment
                -> _ { Path { segments: ps, global: false, span: codemap::DUMMY_SP } } ;

        path_segment ::=
              i:ident
                -> Vec<PathSegment> {
                    let mut v = Vec::new();
                    v.push(path_segment(i));
                    v
                }
            | (mut ps):path_segment "::" i:ident
                 -> _ { ps.push(path_segment(i)); ps } ;

        lit_integer ~ s:r"[0-9][0-9_]*" {
            // TODO suffixes
            psp(ast::LitInt(s.parse::<u64>().unwrap(), ast::SignedIntLit(ast::TyIs(false), ast::Plus)))
        } ;

        ident ~ s:r"[_a-zA-Z][_a-zA-Z0-9]*" -> Ident { str_to_ident(s) } ;

        discard ~ r"\s" ;
    };

    for ast in rust.parses_iter("::ab::cd::ef") {
        println!("{:?}", ast);
    }
}
