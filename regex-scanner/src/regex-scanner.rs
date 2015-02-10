#![crate_name = "marpa-macros"]

#![feature(plugin_registrar, quote, globs, macro_rules, box_syntax, rustc_private)]

extern crate rustc;
extern crate syntax;
#[macro_use] extern crate log;

use self::RuleRhs::*;
use self::KleeneOp::*;
use self::InlineActionType::*;

use syntax::ast;
use syntax::ast::TokenTree;
use syntax::codemap::Span;
use syntax::ext::base::{ExtCtxt, MacResult, MacExpr};
use syntax::parse::token;
use syntax::parse::parser::Parser;
use syntax::parse;
use syntax::util::interner::StrInterner;
use syntax::ptr::P;
use syntax::ext::build::AstBuilder;
use syntax::print::pprust;

use std::collections::HashMap;
use std::collections::HashSet;
use std::mem;
use std::fmt;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut ::rustc::plugin::Registry) {
    reg.register_macro("regex_scanner", expand_regex_scanner);
}

fn expand_regex_scanner(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree])
                       -> Box<MacResult+'static> {
    let mut parser = parse::new_parser_from_tts(cx.parse_sess(),
                                                cx.cfg(), tts.to_vec());

    let mut regs = vec![];

    while !parser.is_eof() {
        let (reg, style) = parser.parse_str();
        let len = regs.len();
        regs.push((len, reg));
        parser.expect(&token::Comma);
    }

    regs[].sort_by(|(_n, s1), (_n, s2)| s1.cmp(s2));
    regs.dedup();

    l0_discard_rules.push(format!("({})", regs.connect(")|(")));
    let reg_alt_s = l0_discard_rules.connect("|");
    let reg_alt = &reg_alt_s[];
    debug!("Lexer regexstr: {}", reg_alt_s);

    quote_expr!(cx,
        struct Lexer {
            reg: ::regex::Regex,
        }

        struct LexerParse<'a> {
            input: &'a str,
            positions: Vec<(u32, u32, usize)>,
        }

        impl Lexer {
            fn new() -> Lexer {
                Lexer {
                    reg: regex!($reg_alt)
                }
            }

            fn accept_input(input: &str) -> LexerParse {
                LexerParse {
                    input: input,
                    positions: vec![],
                }
            }
        }

        impl<'a> LexerParse<'a> {
            fn tokens_iter(&mut self) -> FilterMap<u8> {
                self.reg.captures_iter(input).filter_map(|capture| {
                    capture.iter_pos().skip(1).enumerate().find(|(_n, (p1, p2))| p1 != p2)
                                                          .map(|(group, (start_pos, end_pos))| {
                        positions.push((start_pos as u32, end_pos as u32, group));
                        (positions.len() - 1, group)
                    })
                })
            }

            fn token(idx: usize) -> &'a str {
                let (start_pos, end_pos, tok_kind) = self.positions[idx];
                (self.input.slice(start_pos as usize, end_pos as usize),
                 tok_kind)
            }
        }
    )
}
