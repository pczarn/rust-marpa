#![crate_name = "regex_scanner"]

#![feature(plugin_registrar, quote, globs, macro_rules, box_syntax, rustc_private)]

extern crate rustc;
extern crate syntax;
#[macro_use] extern crate log;

use syntax::ast;
use syntax::ast::TokenTree;
use syntax::codemap::Span;
use syntax::ext::base::{ExtCtxt, MacResult, MacExpr, MacItems};
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
    let mut discard = None;

    while parser.token != token::Eof {
        if parser.eat(&token::Semi) {
            let (reg, _style) = parser.parse_str();
            discard = Some(reg.to_string());
        } else {
            let (reg, _style) = parser.parse_str();
            let len = regs.len();
            regs.push((len, reg));
        }
        parser.expect(&token::Comma);
    }

    let num_syms = regs.len();

    let (idxs, regs): (Vec<_>, Vec<_>) = regs.into_iter().map(|(n, s)| (n, s.to_string())).unzip();

    let num_ary = cx.expr_vec(sp, idxs.into_iter().map(|i| cx.expr_usize(sp, i)).collect());
    let reg_ary = cx.expr_vec(sp, regs.into_iter().map(|r| {
        let r = &r[];
        quote_expr!(cx, regex!(concat!(r"\A", $r)))
    }).collect());

    let discard = discard.as_ref().map(|s| &s[]);

    let lexer_mod = quote_item!(cx,
        pub mod regex_scanner {
            use regex::Regex;
            use std::vec::IntoIter;
            use std::iter::FilterMap;

            pub type Input<'a> = &'a str;
            pub type Output<'a> = &'a str;

            pub struct Lexer {
                reg: [Regex; $num_syms],
                discard: Regex,
            }

            pub struct LexerParse<'a, 'b> {
                reg: &'b [Regex; $num_syms],
                discard: &'b Regex,
                input: &'a str,
                offset: usize,
            }

            pub struct Token {
                group: u32,
                begin: u32,
                end: u32,
            }

            impl Token {
                pub fn sym(&self) -> usize {
                    self.group as usize
                }
            }

            impl Lexer {
                pub fn new() -> Lexer {
                    Lexer {
                        reg: $reg_ary,
                        discard: regex!(concat!(r"\A", $discard)),
                    }
                }

                pub fn new_parse<'a, 'b>(&'b self, input: &'a str) -> LexerParse<'a, 'b> {
                    LexerParse {
                        reg: &self.reg,
                        discard: &self.discard,
                        input: input,
                        // positions: vec![],
                        offset: 0,
                    }
                }
            }

            impl<'a, 'b> LexerParse<'a, 'b> {
                // for LATM
                pub fn longest_parses_iter(&mut self, accept: &[u32])
                                           -> Option<IntoIter<Token>> {
                    while let Some((begin, end)) = self.discard.find(&self.input[self.offset..]) {
                        debug_assert_eq!(begin, 0);
                        self.offset += end;
                    }

                    if self.is_empty() {
                        return None;
                    }

                    let input = &self.input[self.offset..];
                    let (mut max_end, mut num_max_end): (usize, usize) = (0, 0);
                    let parses = accept.iter().filter_map(|pos| {
                        self.reg[*pos as usize].find(input).and_then(|(begin, end)| {
                            debug_assert_eq!(begin, 0);
                            if end >= max_end {
                                max_end = end;
                                if end > max_end {
                                    num_max_end = 0;
                                }
                                num_max_end += 1;
                                Some((*pos as u32, end as u32))
                            } else {
                                None
                            }
                        })
                    }).collect::<Vec<_>>().into_iter().filter_map(|(id, end)|
                        if end as usize == max_end {
                            Some(Token {
                                group: id,
                                begin: self.offset as u32,
                                end: self.offset as u32 + end
                            })
                        } else {
                            None
                        }
                    ).collect::<Vec<_>>();

                    assert!(!parses.is_empty());

                    self.offset += max_end;

                    Some(parses.into_iter())
                }

                pub fn is_empty(&self) -> bool {
                    debug_assert!(self.offset <= self.input.len());
                    self.offset == self.input.len()
                }

                pub fn get(&self, toks: &[Token], idx: usize) -> (&'a str, usize) {
                    let tok = &toks[idx];
                    (&self.input[tok.begin as usize .. tok.end as usize], tok.sym())
                }
            }
        }
    ).unwrap();

    MacItems::new(Some(lexer_mod).into_iter())
}
