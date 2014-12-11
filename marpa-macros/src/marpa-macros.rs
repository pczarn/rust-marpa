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
use syntax::util::interner::StrInterner;
use syntax::ptr::P;
use syntax::ext::build::AstBuilder;

use std::collections::HashSet;
use std::collections::HashMap;
use std::mem;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut ::rustc::plugin::Registry) {
    reg.register_macro("grammar", expand_grammar);
}

#[deriving(Clone, Show)]
enum RuleRhs {
    Alternative(Vec<RuleRhs>),
    Sequence(Vec<RuleRhs>),
    Ident(ast::Name),
    Repeat(ast::Name, KleeneOp),
    Empty,
}

impl RuleRhs {
    fn create_seq_rules(&mut self, syms: &mut StrInterner, cont: &mut Vec<(ast::Name, RuleRhs)>) {
        match self {
            &Sequence(ref mut seq) | &Alternative(ref mut seq) => {
                for rule in seq.iter_mut() {
                    rule.create_seq_rules(syms, cont);
                }
            }
            mut rule @ &Repeat(..) => {
                let new_sym = syms.gensym("");
                let rule = mem::replace(rule, Ident(new_sym));
                match rule {
                    Repeat(repeated, op) => {
                        match op {
                            ZeroOrMore => {
                                cont.push((new_sym, Sequence(vec![])));
                            }
                            OneOrMore => {
                                cont.push((new_sym, Ident(repeated.clone())));
                            }
                        }
                        cont.push((new_sym, Sequence(vec![Ident(new_sym), Ident(repeated)])));
                    }
                    _ => unreachable!()
                }
            }
            &Ident(..) | &Empty => {}
        }
    }
}

#[deriving(Clone, Show)]
pub enum KleeneOp {
    ZeroOrMore,
    OneOrMore,
}

fn parse_name_or_repeat(parser: &mut Parser, syms: &mut StrInterner) -> RuleRhs {
    let mut seq = vec![];
    while parser.token.is_ident() &&
            !parser.token.is_strict_keyword() &&
            !parser.token.is_reserved_keyword() {
        let ident = parser.parse_ident();
        let new_name = syms.intern(ident.as_str());
        let elem = match parser.token {
            token::BinOp(token::Star) => {
                parser.bump();
                Repeat(new_name, ZeroOrMore)
            }
            token::BinOp(token::Plus) => {
                parser.bump();
                Repeat(new_name, OneOrMore)
            }
            _ => Ident(new_name),
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

fn parse_rhs(parser: &mut Parser, syms: &mut StrInterner) -> RuleRhs {
    let elem = parse_name_or_repeat(parser, syms);
    if parser.token == token::BinOp(token::Or) {
        parser.bump();
        // flattened alternative
        match parse_rhs(parser, syms) {
            Alternative(mut alt_seq) => {
                alt_seq.insert(0, elem);
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

    let mut g1_syms = StrInterner::new();
    let start_sym = g1_syms.gensym(":start");

    while parser.token != token::Eof {
        let ident = parser.parse_ident();
        let new_name = g1_syms.intern(ident.as_str());

        if parser.token == token::Tilde {
            parser.bump();
            let (regex, _str_style) = parser.parse_str();
            l0_rules.push((new_name, regex));
            parser.expect(&token::Semi);
        } else if parser.token == token::ModSep && parser.look_ahead(1, |t| *t == token::Eq) {
            parser.bump();
            parser.bump();
            let rule_rhs = parse_rhs(&mut parser, &mut g1_syms);
            g1_rules.push((new_name, rule_rhs));
        } else {
            let sp = parser.span;
            parser.span_err(sp, "expected `::=` or `~`");
        }
    }

    // prepare grammar rules

    let mut g1_rules_cont = vec![];
    for rule in g1_rules.into_iter() {
        match rule {
            (name, Alternative(alt_seq)) =>
                g1_rules_cont.extend(alt_seq.into_iter().map(|alt| (name, alt))),
            seq_or_name =>
                g1_rules_cont.push(seq_or_name),
        }
    }
    // rule :start
    let &(implicit_start, _) = &g1_rules_cont[0];
    g1_rules_cont.push((start_sym, Ident(implicit_start)));

    let mut g1_seq_rules = vec![];
    for &(_, ref mut rule) in g1_rules_cont.iter_mut() {
        rule.create_seq_rules(&mut g1_syms, &mut g1_seq_rules);
    }

    let num_syms = g1_syms.len();

    // LEXER -- L0
    let reg_alt = l0_rules.iter().map(|&(l0_sym_id, ref l0_reg)| l0_reg.get().into_string()).collect::<Vec<_>>().connect(")|(");
    let scan_syms = l0_rules.iter().map(|&(l0_sym_id, _)| l0_sym_id).collect::<Vec<_>>();
    let ast::Name(scan_syms_0) = scan_syms[0];
    let num_scan_syms = l0_rules.len();
    let reg_alt = format!("({})", reg_alt);

    // ``` let scan_syms = [syms[$l0_sym_id0], syms[$l0_sym_id1], ...]; ```
    let scan_syms_exprs = l0_rules.iter().map(|&(ast::Name(l0_sym_id), _)| quote_expr!(cx, syms[$l0_sym_id as uint])).collect::<Vec<_>>();
    let scan_syms_expr = cx.expr_vec(sp, scan_syms_exprs);
    let let_scan_syms_stmt = cx.stmt_let(sp, false, cx.ident_of("scan_syms"), scan_syms_expr);

    // RULES -- G1
    let mut rhs_exprs = vec![];
    let rule_exprs = g1_rules_cont.iter().chain(g1_seq_rules.iter()).map(|&(ast::Name(lhs), ref rhs)| {
        match rhs {
            &Sequence(ref seq) => {
                rhs_exprs.extend(seq.iter().map(|node|
                    match node {
                        &Ident(ast::Name(id)) => quote_expr!(cx, syms[$id as uint]),
                        _ => panic!()
                    }
                ));
            }
            &Ident(ast::Name(rhs_sym)) => {
                rhs_exprs.push(quote_expr!(cx, syms[$rhs_sym as uint]));
            }
            _ => panic!()
        }

        return cx.expr_tuple(sp, vec![
            quote_expr!(cx, syms[$lhs as uint]),
            cx.expr_vec_slice(sp, mem::replace(&mut rhs_exprs, vec![])),
        ])
    }).collect::<Vec<_>>();

    // ``` let rules = &[$rule_exprs]; ```
    let num_rules = rule_exprs.len();
    let rules_expr = cx.expr_vec(sp, rule_exprs);
    let let_rules_stmt =
        cx.stmt_let_typed(sp,
                          false,
                          cx.ident_of("rules"),
                          quote_ty!(cx, [(::marpa::Symbol, &[::marpa::Symbol]), ..$num_rules]),
                          rules_expr);

    let g_expr = quote_expr!(cx,
        {
            let mut cfg = ::marpa::Config::new();
            let grammar = ::marpa::Grammar::with_config(&mut cfg).unwrap();
            let mut syms: [::marpa::Symbol, ..$num_syms] = unsafe { ::std::mem::uninitialized() };
            for sym in syms.iter_mut() { *sym = grammar.symbol_new().unwrap(); }
            grammar.start_symbol_set(syms[0]);
            $let_rules_stmt
            $let_scan_syms_stmt
            let mut rule_ids: [::marpa::Rule, ..$num_rules] = unsafe { ::std::mem::uninitialized() };
            for (&(lhs, rhs), id) in rules.iter().zip(rule_ids.iter_mut()) {
                *id = grammar.rule_new(lhs, rhs).unwrap();
            }
            grammar.precompute();

            let recce = ::marpa::Recognizer::new(&grammar).unwrap();
            recce.start_input();

            let scanner = regex!($reg_alt);

            move |: input: &str| {
                for capture in scanner.captures_iter(input) {
                    let pos = capture.iter().skip(1).position(|s| !s.is_empty()).unwrap();
                    recce.alternative(scan_syms[pos], 1i32, 1);
                    recce.earleme_complete();
                }

                let latest_es = recce.latest_earley_set();
                let bocage = ::marpa::Bocage::new(&recce, latest_es).unwrap();
                let order = ::marpa::Order::new(&bocage).unwrap();
                let tree = ::marpa::Tree::new(&order).unwrap();

                for valuator in tree.values() {
                    for &rule in rule_ids.iter() {
                        valuator.rule_is_valued_set(rule, 1);
                    }

                    loop {
                        use marpa::Step;
                        match valuator.step() {
                            Step::StepToken => {
                                let tok_idx = valuator.token_value() as uint;
                                println!("tok {}", tok_idx);
                            }
                            Step::StepRule => {
                                let arg_0 = valuator.arg_0();
                                let arg_n = valuator.arg_n();
                                let rule = valuator.rule();
                                println!("rule {} {}", arg_0, arg_n);
                            }
                            Step::StepInactive | Step::StepNullingSymbol => {
                                println!("step");
                                break;
                            }
                            other => panic!("unexpected step {}", other),
                        }
                    }
                }
            }
        }
    );

    MacExpr::new(g_expr)
}
