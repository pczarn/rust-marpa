#![crate_name = "marpa-macros"]

#![feature(plugin_registrar, quote, globs, macro_rules)]

extern crate rustc;
extern crate syntax;

use self::RuleRhs::*;
use self::KleeneOp::*;

use syntax::ast;
use syntax::ast::TokenTree;
use syntax::codemap::Span;
// use syntax::ext::base;
use syntax::ext::base::{ExtCtxt, MacResult, MacExpr};
use syntax::parse::token;
use syntax::parse::parser::Parser;
use syntax::parse;
// use syntax::ptr::P;

use std::collections::HashSet;

// #[macro_export]
// macro_rules! exported_macro (() => (2i))

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut ::rustc::plugin::Registry) {
    // reg.register_syntax_extension(token::intern("grammar"),
    //     base::Modifier(box expand_grammar));
    reg.register_macro("grammar", expand_grammar);
}

#[deriving(Show)]
enum RuleRhs {
    Alternative(Vec<RuleRhs>),
    Sequence(Vec<RuleRhs>),
    Ident(u32),
    Repeat(u32, KleeneOp),
    Empty,
}

#[deriving(Show)]
pub enum KleeneOp {
    ZeroOrMore,
    OneOrMore,
}

fn parse_name_or_repeat(parser: &mut Parser, syms: &mut HashSet<ast::Ident>, symvec: &mut Vec<ast::Ident>) -> RuleRhs {
    let mut seq = vec![];
    while parser.token.is_ident() &&
            !parser.token.is_strict_keyword() &&
            !parser.token.is_reserved_keyword() {
        let name = parser.parse_ident();
        syms.insert(name);
        symvec.push(name);
        let elem = match parser.token {
            token::BinOp(token::Star) => {
                parser.bump();
                Repeat(name, ZeroOrMore)
            }
            token::BinOp(token::Plus) => {
                parser.bump();
                Repeat(name, OneOrMore)
            }
            _ => Ident(name),
        };
        seq.push(elem);
    }

    if seq.is_empty() {
        Empty
    } else if seq.len() == 1 {
        seq.into_iter().next().unwrap()
    } else {
        Sequence(seq)
    }

    // else {
    //     parser.expect_one_of(&[], &[token::Semi, token::Eof]);
    //     Empty
    // }
}

fn parse_rhs(parser: &mut Parser, syms: &mut HashSet<ast::Ident>, symvec: &mut Vec<ast::Ident>) -> RuleRhs {
    let elem = parse_name_or_repeat(parser, syms, symvec);
    if parser.token == token::BinOp(token::Or) {
        parser.bump();
        // flattened alternative
        match parse_rhs(parser, syms, symvec) {
            Alternative(mut alt_seq) => {
                alt_seq.unshift(elem);
                Alternative(alt_seq)
            }
            name_or_rep => Alternative(vec![elem, name_or_rep])
        }
    } else {
        // complete this rule
        parser.expect_one_of(&[token::Semi], &[token::Eof]);
        elem
    }
}

fn expand_grammar(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree])
                  -> Box<MacResult+'static> {
    let mut parser = parse::new_parser_from_tts(cx.parse_sess(),
        cx.cfg(), tts.to_vec());

    let mut g1_rules = vec![];
    let mut l0_rules = vec![];

    let mut g1_syms = HashSet::new();
    let mut g1_symvec = vec![];

    while parser.token != token::Eof {
        let name = parser.parse_ident();

        if parser.token == token::Tilde {
            parser.bump();
            let (regex, _str_style) = parser.parse_str();
            l0_rules.push((name, regex));
            parser.expect(&token::Semi);
        } else if parser.token == token::ModSep && parser.look_ahead(1, |t| *t == token::Eq) {
            parser.bump();
            parser.bump();
            let rule_rhs = parse_rhs(&mut parser, &mut g1_syms, &mut g1_symvec);
            g1_rules.push((name, rule_rhs));
        } else {
            let sp = parser.span;
            parser.span_err(sp, "expected `::=` or `~`");
        }
    }

    println!("{} {} {}", g1_rules, l0_rules, g1_syms);
    // parser.expect(&token::Semi);

    // let expr = parser.parse_expr();

    let num_syms = g1_syms.len();

    g1_rules.map.

    let g_stmt = quote_stmt!(cx,
        let grammar = ::marpa::Grammar::new().unwrap();
        let syms: [::marpa::Symbol, ..$num_syms] = unsafe { ::std::mem::uninitialized() };
        for sym in syms.iter_mut() { *sym = grammar.symbol_new().unwrap(); }
    );

    // let sym_stmt = quote_stmt!(cx, let syms: [::marpa::Symbol, ..$num_syms] = unsafe { ::std::mem::uninitialized() };
    //     for sym in syms.iter_mut() { *sym = grammar.symbol_new().unwrap(); });

    MacExpr::new(quote_expr!(cx, { $g_stmt }))
}

// grammar! {
//     :start ::= expr ;
//     expr ::= expr op expr | number ;
//     number ~ r"\d" ;
//     op ~ r"[\*\+\-]" ;
// }

// grammar! {
//     ident ~ \w+ ;
//     rule ::= lhs '::=' rhs ';' ;
//     rhs ::= rhs '|' rhs | ident '+' | ident '*' | ident ;
//     lhs ::= ident ;
// }
