#![crate_name = "enum_adaptor"]

#![feature(plugin_registrar, quote, globs, macro_rules, box_syntax, rustc_private)]

extern crate rustc;
extern crate syntax;
#[macro_use] extern crate log;

use syntax::ast;
use syntax::ast::TokenTree;
use syntax::codemap::Span;
use syntax::ext::base::{ExtCtxt, MacResult, MacEager};
use syntax::parse::token;
use syntax::parse::parser::Parser;
use syntax::parse;
use syntax::util::interner::StrInterner;
use syntax::ptr::P;
use syntax::ext::build::AstBuilder;
use syntax::print::pprust;
use syntax::util::small_vector::SmallVector;

use std::collections::HashMap;
use std::collections::HashSet;
use std::mem;
use std::fmt;
use std::option;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut ::rustc::plugin::Registry) {
    reg.register_macro("enum_adaptor", expand_enum_adaptor);
}

fn expand_enum_adaptor(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree])
                       -> Box<MacResult+'static> {
    let mut parser = parse::new_parser_from_tts(cx.parse_sess(),
                                                cx.cfg(), tts.to_vec());

    let mut pats = vec![];
    let mut discard = None;

    let Token = parser.parse_ident();
    let T = parser.parse_ty();

    if parser.eat(&token::Semi) {
        let pat = parser.parse_pat();
        discard = Some(pat);
    } 
    parser.expect(&token::Comma);

    while parser.token != token::Eof && parser.token != token::Semi {
        {
            let pat = parser.parse_pat();
            pats.push(pat);
        }
        parser.expect(&token::Comma);
    }

    parser.expect(&token::Semi);
    let grammar_expr = parser.parse_token_tree();

    debug!("{:?}", pats);

    let (matches_n, matches_pat): (Vec<u32>, Vec<P<ast::Pat>>) = pats.into_iter().enumerate().map(|(n, pat)| {
        (n as u32, pat)
    }).unzip();

    let enum_adaptor_expr = quote_expr!(cx, {
        use std::vec::IntoIter;
        use std::marker::PhantomData;

        type LInp<'a> = &'a [$T];
        type LOut<'a> = &'a $T;

        #[derive(Debug)]
        struct $Token {
            sym: u32,
            offset: u32,
        }

        struct LPar<'a, 'b> {
            input: LInp<'a>,
            offset: usize,
            marker: PhantomData<&'b ()>,
        }

        type Input<'a> = &'a [$T];
        type Output<'a> = &'a $T;

        struct EnumAdaptor;

        impl $Token {
            fn sym(&self) -> usize {
                self.sym as usize
            }
        }

        trait Lexer {
            fn new_parse<'a, 'b>(&self, input: Input<'a>) -> LPar<'a, 'b>;
        }

        impl Lexer for EnumAdaptor {
            fn new_parse<'a, 'b>(&self, input: Input<'a>) -> LPar<'a, 'b> {
                LPar {
                    input: input,
                    offset: 0,
                    marker: PhantomData,
                }
            }
        }

        impl<'a, 'b> LPar<'a, 'b> {
            // for LATM
            fn longest_parses_iter(&mut self, _accept: &[u32])
                                       -> Option<IntoIter<$Token>> {
                while !self.is_empty() {
                    // hack:
                    match (true, &self.input[self.offset]) {
                        (true, &$discard) => {
                            self.offset += 1;
                        }
                        _ => break
                    }
                }

                if self.is_empty() {
                    return None;
                }

                let mut syms = vec![];

                $(
                    match (true, &self.input[self.offset]) {
                        (true, &$matches_pat) => {
                            syms.push(($matches_n, self.offset as u32));
                        }
                        _ => ()
                    }
                )*

                assert!(!syms.is_empty());

                self.offset += 1;

                Some(syms.map_in_place(|(n, offset)| $Token { sym: n, offset: offset }).into_iter())
            }

            fn is_empty(&self) -> bool {
                debug_assert!(self.offset <= self.input.len());
                self.offset == self.input.len()
            }

            fn get(&self, toks: &[$Token], idx: usize) -> (&'a $T, usize) {
                (&self.input[toks[idx].offset as usize], toks[idx].sym())
            }
        }

        fn new_lexer() -> EnumAdaptor {
            EnumAdaptor
        }

        $grammar_expr
    });

    debug!("{}", pprust::expr_to_string(&enum_adaptor_expr));

    MacEager::expr(enum_adaptor_expr)
}
