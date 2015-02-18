#![crate_name = "regex_scanner"]

#![feature(plugin_registrar, quote, globs, macro_rules, box_syntax, rustc_private)]

extern crate rustc;
extern crate syntax;
#[macro_use] extern crate log;

// extern crate regex;

// use self::RuleRhs::*;
// use self::KleeneOp::*;
// use self::InlineActionType::*;

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

    // an array [origin_idx] => 
    // let redir_ary = 

    // regs[].sort_by(|&(_n1, ref s1), &(_n2, ref s2)| s1.cmp(s2));
    // regs.dedup();

    let num_syms = regs.len();

    let (idxs, regs): (Vec<_>, Vec<_>) = regs.into_iter().map(|(n, s)| (n, s.to_string())).unzip();

    // array of [final group] => origin idx
    let num_ary = cx.expr_vec(sp, idxs.into_iter().map(|i| cx.expr_usize(sp, i)).collect());
    let reg_ary = cx.expr_vec(sp, regs.into_iter().map(|r| {
        let r = &r[];
        quote_expr!(cx, regex!(concat!(r"\A", $r)))
        // quote_expr!(cx, $r)
    }).collect());

    // let mut l0_discard_rules = vec![];
    // l0_discard_rules.push(format!("({})", regs.connect(")|(")));
    // let reg_alt_s = l0_discard_rules.connect("|");
    // let reg_alt = &reg_alt_s[];

    // debug!("Lexer regexstr: {}", reg_alt_s);

    // let num_syms: usize = 0;
    let discard = discard.as_ref().map(|s| &s[]);

    let lexer_mod = quote_item!(cx,
        mod lexer {
            use regex::Regex;
            use std::vec::IntoIter;
            use std::iter::FilterMap;

            pub type Input<'a> = &'a str;

            // pub static NUM_SYMS: usize = $num_syms;
            // pub static SYMS: &'static [usize] = &$num_ary;

            pub struct Lexer {
                reg: [Regex; $num_syms],
                discard: Regex,
            }

            pub struct LexerParse<'a, 'b> {
                reg: &'b [Regex; $num_syms],
                discard: &'b Regex,
                input: &'a str,
                offset: usize,
                // positions: Vec<(u32, u32, usize)>,
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
                    // println!("{:?}", $reg_ary);
                    Lexer {
                        reg: $reg_ary,
                        discard: regex!(concat!(r"\A", $discard)),
                    }
                }

                pub fn new_parse<'a, 'b>(&'b self, input: &'a str) -> LexerParse<'a, 'b> {
                    // for reg in self.reg.iter() {
                    //     println!("{}", reg.as_str());
                    // }
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
                // pub fn tokens_iter(&mut self) -> FilterMap<u8> {
                //     self.reg.captures_iter(input).filter_map(|capture| {
                //         capture.iter_pos().skip(1).enumerate().find(|(_n, (p1, p2))| p1 != p2)
                //                                               .map(|(group, (start_pos, end_pos))|
                //             (group as i32, (start_pos as u32, end_pos as u32))
                //         )
                //     })
                // }

                // LATM
                pub fn longest_parses_iter(&mut self, accept: &[u32])
                                           -> Option<IntoIter<Token>> {
                    // println!("{}", self.offset);
                    // println!("{:?}", accept);
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

                pub fn get(&self, tok: &Token) -> (&'a str, usize) {
                    (&self.input[tok.begin as usize .. tok.end as usize], tok.sym())
                }
            }
        }
    ).unwrap();

    MacItems::new(Some(lexer_mod).into_iter())
}
