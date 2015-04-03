#![crate_name = "marpa_macros"]
#![feature(plugin, plugin_registrar, unboxed_closures, quote, globs, macro_rules, box_syntax, rustc_private, box_patterns, trace_macros)]

extern crate rustc;
extern crate syntax;
extern crate marpa;
#[macro_use] extern crate log;

use self::Expr::*;
pub use self::RustToken::Tok;

use syntax::ast;
use syntax::ast::TokenTree;
use syntax::codemap::{respan, DUMMY_SP};
use syntax::codemap::Span;
use syntax::ext::base::{ExtCtxt, MacResult, MacEager};
pub use syntax::parse::token;
use syntax::parse::parser::Parser;
use syntax::parse;
use syntax::util::interner::StrInterner;
use syntax::ptr::P;
use syntax::ext::build::AstBuilder;
use syntax::print::pprust;
use syntax::parse::common::seq_sep_none;
use syntax::parse::lexer;
use syntax::ext::tt::transcribe;
pub use syntax::parse::token::*;
use syntax::owned_slice::OwnedSlice;

use syntax::ast::{Ty_, Ty, PathParameters, AngleBracketedParameters, AngleBracketedParameterData, TyPath,
    PathSegment, Block, TtToken, Pat, Ident, TyTup, TyInfer};
// use syntax::parse::token::{OpenDelim, CloseDelim};

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
    reg.register_macro("grammar", expand_grammar);
}

// Structures

#[derive(Debug)]
struct Context {
    options: Vec<Opt>,
    rules: Vec<Rule>,
    l0_rules: Vec<L0Rule>,
}

#[derive(Debug)]
struct Rule {
    name: ast::Name,
    ty: P<Ty>,
    rhs: Vec<Alternative>,
}

#[derive(Debug)]
struct L0Rule {
    name: ast::Name,
    ty: P<Ty>,
    pat: Option<P<Pat>>,
    rhs: Vec<Token>,
    action: InlineAction,
}

#[derive(Debug)]
struct Alternative {
    inner: Vec<Expr>,
    pats: Vec<(usize, P<Pat>)>,
    action: InlineAction,
}

#[derive(Debug)]
enum Expr {
    NameExpr(ast::Name),
    ExprOptional(Box<Expr>),
    ExprSeq(Vec<Expr>, Vec<Expr>, ast::KleeneOp),
}

#[derive(Debug)]
struct InlineAction {
    block: Option<P<Block>>,
}

#[derive(Debug)]
struct Opt {
    ident: ast::Ident,
    tokens: Vec<TokenTree>,
}

struct SeqRule {
    name: ast::Name,
    ty: P<Ty>,
    rhs: ast::Name,
    sep: ast::Name,
}

struct ExtractionContext<'a, 'b: 'a> {
    namespace: &'a mut HashMap<ast::Name, (u32, u32)>,
    l0_types: Vec<P<Ty>>,
    types: Vec<P<Ty>>,
    sp: Span,
    ext: &'a mut ExtCtxt<'b>,
}

impl Rule {
    fn extract(&mut self, cx: &mut ExtractionContext) -> (Vec<Rule>, Vec<SeqRule>) {
        let mut new_rules = vec![];
        let mut new_seq_rules = vec![];

        for alt in &mut self.rhs {
            for expr in &mut alt.inner {
                let (_, r, s) = expr.extract(cx);

                new_rules.extend(r.into_iter());
                new_seq_rules.extend(s.into_iter());
            }
        }

        (new_rules, new_seq_rules)
    }
}

impl Expr {
    // Transforms an expr into NameExpr
    fn extract(&mut self, cx: &mut ExtractionContext) -> (P<Ty>, Vec<Rule>, Vec<SeqRule>) {
        let mut replace_with = None;
        let mut v_rules = vec![];
        let mut v_seq_rules = vec![];

        let tup = match self {
            &mut NameExpr(name) => {
                let (kind, n) = cx.namespace[name];
                let n = n as usize;
                let ty = if kind == 1 { cx.types[n].clone() } else { cx.l0_types[n].clone() };
                (ty, vec![], vec![])
            }
            &mut ExprOptional(ref mut inner) => {
                let (inner_ty, new_r, new_seq) = inner.extract(cx);
                v_rules.extend(new_r.into_iter());
                v_seq_rules.extend(new_seq.into_iter());
                let pat_ident = gensym_ident("_pat_");
                let pat = cx.ext.pat_ident(cx.sp, pat_ident);
                let name = gensym_ident("_name_");
                let ty = quote_ty!(cx.ext, Option<$inner_ty>);

                v_rules.push(Rule {
                    name: name.name,
                    ty: ty.clone(),
                    rhs: vec![
                        Alternative {
                            inner: vec![NameExpr(inner.name())],
                            pats: vec![(0, pat)],
                            action: InlineAction {
                                block: Some(cx.ext.block_expr(cx.ext.expr_some(cx.sp, cx.ext.expr_ident(cx.sp, pat_ident)))),
                            }
                        },
                        Alternative {
                            inner: vec![],
                            pats: vec![],
                            action: InlineAction {
                                block: Some(cx.ext.block_expr(cx.ext.expr_none(cx.sp))),
                            }
                        }
                    ],
                });

                replace_with = Some(NameExpr(name.name));

                (ty, v_rules, v_seq_rules)
            }
            &mut ExprSeq(ref mut body, ref mut sep, _op) => {
                let mut v_body_ty = vec![];

                for e in body.iter_mut() {
                    let (ty, a, b) = e.extract(cx);
                    v_body_ty.push(ty);
                    v_rules.extend(a.into_iter());
                    v_seq_rules.extend(b.into_iter());
                }

                let name = gensym_ident("_name_");
                let name_body = gensym_ident("_name_body_");
                let name_sep = gensym_ident("_name_sep_");

                let mut v_body = mem::replace(body, vec![]);
                let mut v_pats = vec![];

                let (action, inner_ty) = if v_body.len() == 1 {
                    let pat_ident = gensym_ident("_pat_ident_");
                    v_pats.push((0, cx.ext.pat_ident(cx.sp, pat_ident)));
                    (InlineAction { block: Some(cx.ext.block_expr(cx.ext.expr_ident(cx.sp, pat_ident))) },
                     v_body_ty.pop().unwrap())
                } else {
                    unreachable!()
                };

                v_rules.push(Rule {
                    name: name_body.name,
                    ty: inner_ty.clone(),
                    rhs: vec![
                        Alternative {
                            inner: v_body,
                            pats: v_pats,
                            action: action,
                        }
                    ],
                });

                v_rules.push(Rule {
                    name: name_sep.name,
                    ty: quote_ty!(cx.ext, ()),
                    rhs: vec![
                        Alternative {
                            inner: mem::replace(sep, vec![]),
                            pats: vec![],
                            action: InlineAction { block: Some(cx.ext.block_expr(cx.ext.expr_tuple(cx.sp, vec![]))) },
                        }
                    ],
                });

                let ty = quote_ty!(cx.ext, Vec<$inner_ty>);

                v_seq_rules.push(SeqRule {
                    name: name.name,
                    ty: ty.clone(),
                    rhs: name_body.name,
                    sep: name_sep.name,
                });

                replace_with = Some(NameExpr(name.name));

                (ty, v_rules, v_seq_rules)
            }
        };

        if let Some(with) = replace_with {
            *self = with;
        }

        tup
    }

    fn name(&self) -> ast::Name {
        match self {
            &NameExpr(name) => name,
            _ => unreachable!()
        }
    }
}

fn repr_variant(n: u32) -> Ident {
    let variant_name = format!("Spec{}", n);
    token::str_to_ident(&variant_name[..])
}

fn quote_block(cx: &mut ExtCtxt, toks: Vec<Token>) -> P<Block> {
    let tts: Vec<TokenTree> = toks.into_iter().map(|t| TtToken(DUMMY_SP, t)).collect();
    cx.block_expr(quote_expr!(cx, { $tts }))
}

fn quote_pat(cx: &mut ExtCtxt, toks: Vec<Token>) -> P<Pat> {
    let tts: Vec<TokenTree> = toks.into_iter().map(|t| TtToken(DUMMY_SP, t)).collect();
    quote_pat!(cx, $tts)
}

fn quote_tokens(cx: &mut ExtCtxt, toks: Vec<Token>) -> Vec<TokenTree> {
    toks.into_iter().map(|t| TtToken(DUMMY_SP, t)).collect()
}

fn panic(arg: &RustToken) -> ! {
    panic!("{:?}", arg);
}

#[derive(Debug)]
pub enum RustToken {
    Delim(Token),
    Tok(Token)
}

fn parse_ast(cx: &mut ExtCtxt, tokens: &[RustToken]) -> (Context, HashMap<ast::Name, (u32, u32)>, Option<L0Rule>) {
    type Alt = (Vec<Expr>, Vec<(usize, P<Pat>)>);

    let mut grammar = {
    use std::vec::IntoIter;
    use std::marker::PhantomData;
    type LInp<'a> = &'a [RustToken];
    type LOut<'a> = &'a RustToken;
    #[derive(Debug)]
    struct Token_ {
        sym: u32,
        offset: u32,
    }
    struct LPar<'a, 'b> {
        input: LInp<'a>,
        offset: usize,
        marker: PhantomData<&'b ()>,
    }
    type Input<'a> = &'a [RustToken];
    type Output<'a> = &'a RustToken;
    struct EnumAdaptor;
    impl Token_ {
        fn sym(&self) -> usize { self.sym as usize }
    }
    trait Lexer {
        fn new_parse<'a, 'b>(&self, input: Input<'a>)
        -> LPar<'a, 'b>;
    }
    impl Lexer for EnumAdaptor {
        fn new_parse<'a, 'b>(&self, input: Input<'a>) -> LPar<'a, 'b> {
            LPar{input: input, offset: 0, marker: PhantomData,}
        }
    }
    impl <'a, 'b> LPar<'a, 'b> {
        fn longest_parses_iter(&mut self, _accept: &[u32])
         -> Option<IntoIter<Token_>> {
            while !self.is_empty() {
                match (true, &self.input[self.offset]) {
                    (true, &Tok(Whitespace)) => { self.offset += 1; }
                    _ => break ,
                }
            }
            if self.is_empty() { return None; }
            let mut syms = vec!();
            match (true, &self.input[self.offset]) {
                (true, &Tok(Tilde)) => {
                    syms.push((0u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(::syntax::parse::token::Ident(..))) => {
                    syms.push((1u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(::syntax::parse::token::Ident(..))) => {
                    syms.push((2u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(ModSep)) => {
                    syms.push((3u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(Not)) => {
                    syms.push((4u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(Eq)) => { syms.push((5u32, self.offset as u32)); }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(Lt)) => { syms.push((6u32, self.offset as u32)); }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(Gt)) => { syms.push((7u32, self.offset as u32)); }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(Semi)) => {
                    syms.push((8u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(RArrow)) => {
                    syms.push((9u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(Colon)) => {
                    syms.push((10u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(BinOp(Or))) => {
                    syms.push((11u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(Underscore)) => {
                    syms.push((12u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(Question)) => {
                    syms.push((13u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(BinOp(Star))) => {
                    syms.push((14u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &RustToken::Delim(OpenDelim(Brace))) => {
                    syms.push((15u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &RustToken::Delim(CloseDelim(Brace))) => {
                    syms.push((16u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &RustToken::Delim(OpenDelim(Paren))) => {
                    syms.push((17u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &RustToken::Delim(CloseDelim(Paren))) => {
                    syms.push((18u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &RustToken::Delim(OpenDelim(Bracket))) => {
                    syms.push((19u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &RustToken::Delim(CloseDelim(Bracket))) => {
                    syms.push((20u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(..)) => {
                    syms.push((21u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &_) => { syms.push((22u32, self.offset as u32)); }
                _ => (),
            }
            assert!(! syms . is_empty (  ));
            self.offset += 1;
            Some(syms.map_in_place(|(n, offset)|
                                       Token_{sym: n,
                                              offset: offset,}).into_iter())
        }
        fn is_empty(&self) -> bool {
            debug_assert!(self . offset <= self . input . len (  ));
            self.offset == self.input.len()
        }
        fn get(&self, toks: &[Token_], idx: usize) -> (&'a RustToken, usize) {
            (&self.input[toks[idx].offset as usize], toks[idx].sym())
        }
    }
    fn new_lexer() -> EnumAdaptor { EnumAdaptor }
    {
        enum Repr<'a> {
            Continue,
            Unused(PhantomData<&'a ()>),
            Spec0(()),
            Spec1(::syntax::ast::Ident),
            Spec2(::syntax::parse::token::Token),
            Spec3(()),
            Spec4(()),
            Spec5(()),
            Spec6(()),
            Spec7(()),
            Spec8(()),
            Spec9(()),
            Spec10(()),
            Spec11(()),
            Spec12(()),
            Spec13(()),
            Spec14(()),
            Spec15(()),
            Spec16(()),
            Spec17(()),
            Spec18(()),
            Spec19(()),
            Spec20(()),
            Spec21(::syntax::parse::token::Token),
            Spec22(::syntax::parse::token::Token),
            Spec23(Context),
            Spec24(Vec<Opt>),
            Spec25(Opt),
            Spec26(Vec<Rule>),
            Spec27(Rule),
            Spec28(Vec<Alternative>),
            Spec29(Alternative),
            Spec30(Alt),
            Spec31(Vec<Expr>),
            Spec32(Expr),
            Spec33(Option<P<Pat>>),
            Spec34(P<Ty>),
            Spec35(InlineAction),
            Spec36(P<Ty>),
            Spec37(Ty_),
            Spec38(ast::Path),
            Spec39(Vec<PathSegment>),
            Spec40(PathSegment),
            Spec41(PathParameters),
            Spec42(Vec<L0Rule>),
            Spec43(L0Rule),
            Spec44(P<Block>),
            Spec45(Vec<Token>),
            Spec46(Vec<Token>),
            Spec47(Vec<Token>),
            Spec48(Vec<Token>),
            Spec49(Vec<Token>),
            Spec50(Vec<Token>),
            Spec51(()),
        }
        struct Grammar<F, G, L> {
            grammar: ::marpa::Grammar,
            scan_syms: [::marpa::Symbol; 23usize],
            nulling_syms: [(::marpa::Symbol, u32); 0usize],
            rule_ids: [::marpa::Rule; 54usize],
            lexer: L,
            lex_closure: F,
            eval_closure: G,
        }
        struct Parses<'a, 'b, F: 'b, G: 'b, L: 'b> {
            tree: ::marpa::Tree,
            lex_parse: LPar<'a, 'b>,
            positions: Vec<Token_>,
            stack: Vec<Repr<'a>>,
            parent: &'b mut Grammar<F, G, L>,
        }
        impl <F, G, L: Lexer> Grammar<F, G, L> where
         F: for<'c>FnMut(LOut<'c>, usize) -> Repr<'c>,
         G: for<'c>FnMut(&mut [Repr<'c>], usize) -> Repr<'c> {
            fn new(lexer: L, lex_closure: F, eval_closure: G)
             -> Grammar<F, G, L> {
                use marpa::{Config, Symbol, Rule};
                let mut cfg = Config::new();
                let mut grammar =
                    ::marpa::Grammar::with_config(&mut cfg).unwrap();
                let mut syms: [Symbol; 52usize] =
                    unsafe { ::std::mem::uninitialized() };
                for s in syms.iter_mut() {
                    *s = grammar.symbol_new().unwrap();
                }
                grammar.start_symbol_set(syms[23usize]);
                let mut scan_syms: [Symbol; 23usize] =
                    unsafe { ::std::mem::uninitialized() };
                for (dst, src) in
                    scan_syms.iter_mut().zip(syms[..23usize].iter()) {
                    *dst = *src;
                }
                let rules: [(Symbol, &[Symbol]); 54usize] =
                    [(syms[23usize],
                      &[syms[24usize], syms[26usize], syms[42usize]]),
                     (syms[24usize], &[syms[24usize], syms[25usize]]),
                     (syms[24usize], &[syms[25usize]]),
                     (syms[25usize],
                      &[syms[1usize], syms[4usize], syms[17usize],
                        syms[48usize], syms[18usize], syms[8usize]]),
                     (syms[26usize], &[syms[26usize], syms[27usize]]),
                     (syms[26usize], &[syms[27usize]]),
                     (syms[27usize],
                      &[syms[1usize], syms[34usize], syms[51usize],
                        syms[28usize], syms[8usize]]),
                     (syms[28usize], &[syms[29usize]]),
                     (syms[28usize],
                      &[syms[28usize], syms[11usize], syms[29usize]]),
                     (syms[29usize], &[syms[30usize], syms[35usize]]),
                     (syms[30usize], &[syms[32usize]]),
                     (syms[30usize], &[syms[33usize], syms[32usize]]),
                     (syms[30usize], &[syms[30usize], syms[32usize]]),
                     (syms[30usize],
                      &[syms[30usize], syms[33usize], syms[32usize]]),
                     (syms[31usize], &[syms[32usize]]),
                     (syms[31usize], &[syms[32usize], syms[31usize]]),
                     (syms[32usize], &[syms[1usize]]),
                     (syms[32usize], &[syms[32usize], syms[13usize]]),
                     (syms[32usize],
                      &[syms[19usize], syms[31usize], syms[20usize],
                        syms[15usize], syms[31usize], syms[16usize],
                        syms[14usize]]),
                     (syms[33usize], &[syms[2usize], syms[10usize]]),
                     (syms[33usize],
                      &[syms[17usize], syms[48usize], syms[18usize],
                        syms[10usize]]),
                     (syms[34usize], &[syms[9usize], syms[36usize]]),
                     (syms[35usize], &[syms[44usize]]),
                     (syms[36usize], &[syms[37usize]]),
                     (syms[37usize], &[syms[38usize]]),
                     (syms[37usize], &[syms[17usize], syms[18usize]]),
                     (syms[37usize], &[syms[12usize]]),
                     (syms[38usize], &[syms[3usize], syms[39usize]]),
                     (syms[38usize], &[syms[39usize]]),
                     (syms[39usize],
                      &[syms[39usize], syms[3usize], syms[40usize]]),
                     (syms[39usize], &[syms[40usize]]),
                     (syms[40usize], &[syms[1usize]]),
                     (syms[40usize], &[syms[1usize], syms[41usize]]),
                     (syms[41usize],
                      &[syms[6usize], syms[36usize], syms[7usize]]),
                     (syms[42usize], &[syms[42usize], syms[43usize]]),
                     (syms[42usize], &[syms[43usize]]),
                     (syms[43usize],
                      &[syms[1usize], syms[34usize], syms[0usize],
                        syms[49usize], syms[35usize], syms[8usize]]),
                     (syms[44usize], &[syms[47usize]]),
                     (syms[45usize], &[syms[47usize]]),
                     (syms[45usize], &[syms[46usize]]),
                     (syms[46usize], &[syms[21usize]]),
                     (syms[46usize], &[syms[17usize], syms[18usize]]),
                     (syms[46usize],
                      &[syms[17usize], syms[48usize], syms[18usize]]),
                     (syms[46usize], &[syms[19usize], syms[20usize]]),
                     (syms[46usize],
                      &[syms[19usize], syms[48usize], syms[20usize]]),
                     (syms[47usize], &[syms[15usize], syms[16usize]]),
                     (syms[47usize],
                      &[syms[15usize], syms[48usize], syms[16usize]]),
                     (syms[48usize], &[syms[45usize]]),
                     (syms[48usize], &[syms[48usize], syms[45usize]]),
                     (syms[49usize], &[syms[46usize]]),
                     (syms[49usize], &[syms[49usize], syms[46usize]]),
                     (syms[50usize], &[syms[22usize]]),
                     (syms[50usize], &[syms[50usize], syms[22usize]]),
                     (syms[51usize], &[syms[3usize], syms[5usize]])];
                let seq_rules: [(Symbol, Symbol, Symbol); 0usize] = [];
                let mut rule_ids: [Rule; 54usize] =
                    unsafe { ::std::mem::uninitialized() };
                {
                    for (dst, &(lhs, rhs)) in
                        rule_ids.iter_mut().zip(rules.iter()) {
                        *dst = grammar.rule_new(lhs, rhs).unwrap();
                    }
                    for (dst, &(lhs, rhs, sep)) in
                        rule_ids.iter_mut().skip(54usize).zip(seq_rules.iter())
                        {
                        *dst = grammar.sequence_new(lhs, rhs, sep).unwrap();
                    }
                };
                let mut nulling_syms: [(Symbol, u32); 0usize] =
                    unsafe { ::std::mem::uninitialized() };
                let nulling_rule_id_n: &[usize] = &[];
                for (dst, &n) in
                    nulling_syms.iter_mut().zip(nulling_rule_id_n.iter()) {
                    *dst = (rules[n].0, n as u32);
                }
                grammar.precompute().unwrap();
                Grammar{lexer: lexer,
                        lex_closure: lex_closure,
                        eval_closure: eval_closure,
                        grammar: grammar,
                        scan_syms: scan_syms,
                        nulling_syms: nulling_syms,
                        rule_ids: rule_ids,}
            }
            #[inline]
            fn parses_iter<'a, 'b>(&'b mut self, input: LInp<'a>)
             -> Parses<'a, 'b, F, G, L> {
                use marpa::{Recognizer, Bocage, Order, Tree, Symbol,
                            ErrorCode};
                let mut recce = Recognizer::new(&mut self.grammar).unwrap();
                recce.start_input();
                let mut lex_parse = self.lexer.new_parse(input);
                let mut positions = vec!();
                let mut ith = 0;
                while !lex_parse.is_empty() {
                    let mut syms: [Symbol; 23usize] =
                        unsafe { mem::uninitialized() };
                    let mut terminals_expected: [u32; 23usize] =
                        unsafe { mem::uninitialized() };
                    let terminal_ids_expected =
                        unsafe { recce.terminals_expected(&mut syms[..]) };
                    let terminals_expected =
                        &mut terminals_expected[..terminal_ids_expected.len()];
                    for (terminal, &id) in
                        terminals_expected.iter_mut().zip(terminal_ids_expected.iter())
                        {
                        *terminal =
                            self.scan_syms.iter().position(|&sym|
                                                               sym ==
                                                                   id).unwrap()
                                as u32;
                    }
                    let mut iter =
                        match lex_parse.longest_parses_iter(terminals_expected)
                            {
                            Some(iter) => iter,
                            None => break ,
                        };
                    for token in iter {
                        recce.alternative(self.scan_syms[token.sym()],
                                          positions.len() as i32 + 1, 1);
                        positions.push(token);
                    }
                    recce.earleme_complete();
                    ith += 1;
                }
                let latest_es = recce.latest_earley_set();
                let mut tree =
                    Bocage::new(&mut recce,
                                latest_es).and_then(|mut bocage|
                                                        Order::new(&mut bocage)).and_then(|mut order|
                                                                                              Tree::new(&mut order));
                Parses{tree: tree.unwrap(),
                       lex_parse: lex_parse,
                       positions: positions,
                       stack: vec!(),
                       parent: self,}
            }
        }
        impl <'a, 'b, F, G, L> Iterator for Parses<'a, 'b, F, G, L> where
         F: for<'c>FnMut(LOut<'c>, usize) -> Repr<'c>,
         G: for<'c>FnMut(&mut [Repr<'c>], usize) -> Repr<'c> {
            type
            Item
            =
            Context;
            fn next(&mut self) -> Option<Context> {
                use marpa::{Step, Value};
                let mut valuator =
                    if self.tree.next() >= 0 {
                        Value::new(&mut self.tree).unwrap()
                    } else { return None; };
                for &rule in self.parent.rule_ids.iter() {
                    valuator.rule_is_valued_set(rule, 1);
                }
                loop  {
                    match valuator.step() {
                        Step::StepToken => {
                            let idx = valuator.token_value() as usize - 1;
                            let elem =
                                self.parent.lex_closure.call_mut(self.lex_parse.get(&self.positions[..],
                                                                                    idx));
                            self.stack_put(valuator.result() as usize, elem);
                        }
                        Step::StepRule => {
                            let rule = valuator.rule();
                            let arg_0 = valuator.arg_0() as usize;
                            let arg_n = valuator.arg_n() as usize;
                            let elem =
                                {
                                    let slice =
                                        self.stack.slice_mut(arg_0,
                                                             arg_n + 1);
                                    let choice =
                                        self.parent.rule_ids.iter().position(|r|
                                                                                 *r
                                                                                     ==
                                                                                     rule);
                                    self.parent.eval_closure.call_mut((slice,
                                                                       choice.expect("unknown rule")))
                                };
                            match elem {
                                Repr::Continue => { continue  }
                                other_elem => {
                                    self.stack_put(arg_0, other_elem);
                                }
                            }
                        }
                        Step::StepNullingSymbol => {
                            let sym = valuator.symbol();
                            let choice =
                                self.parent.nulling_syms.iter().find(|&&(s,
                                                                         _)|
                                                                         s ==
                                                                             sym).expect("unknown nulling sym").1;
                            let elem =
                                self.parent.eval_closure.call_mut((&mut [],
                                                                   choice as
                                                                       usize));
                            self.stack_put(valuator.result() as usize, elem);
                        }
                        Step::StepInactive => { break ; }
                        other => panic!("unexpected step {:?}" , other),
                    }
                }
                let result = self.stack.drain().next();
                match result {
                    Some(Repr::Spec23(val)) => Some(val),
                    _ => None,
                }
            }
        }
        impl <'a, 'b, F, G, L> Parses<'a, 'b, F, G, L> {
            fn stack_put(&mut self, idx: usize, elem: Repr<'a>) {
                if idx == self.stack.len() {
                    self.stack.push(elem);
                } else { self.stack[idx] = elem; }
            }
        }
        Grammar::new(new_lexer(), |arg, choice_| {
                     let r =
                         match (true, arg) {
                             (true, _) if choice_ == 0usize =>
                             Some(Repr::Spec0({ { { } } })),
                             (true, i) if choice_ == 1usize =>
                             Some(Repr::Spec1({
                                                  {
                                                      {
                                                          if let &Tok(::syntax::parse::token::Ident(i,
                                                                                                    _))
                                                                 = i {
                                                              i
                                                          } else {
                                                              unreachable!();
                                                          }
                                                      }
                                                  }
                                              })),
                             (true, i) if choice_ == 2usize =>
                             Some(Repr::Spec2({
                                                  {
                                                      {
                                                          if let &Tok(ref i) =
                                                                 i {
                                                              i.clone()
                                                          } else {
                                                              unreachable!()
                                                          }
                                                      }
                                                  }
                                              })),
                             (true, _) if choice_ == 3usize =>
                             Some(Repr::Spec3({ { { } } })),
                             (true, _) if choice_ == 4usize =>
                             Some(Repr::Spec4({ { { } } })),
                             (true, _) if choice_ == 5usize =>
                             Some(Repr::Spec5({ { { } } })),
                             (true, _) if choice_ == 6usize =>
                             Some(Repr::Spec6({ { { } } })),
                             (true, _) if choice_ == 7usize =>
                             Some(Repr::Spec7({ { { } } })),
                             (true, _) if choice_ == 8usize =>
                             Some(Repr::Spec8({ { { } } })),
                             (true, _) if choice_ == 9usize =>
                             Some(Repr::Spec9({ { { } } })),
                             (true, _) if choice_ == 10usize =>
                             Some(Repr::Spec10({ { { } } })),
                             (true, _) if choice_ == 11usize =>
                             Some(Repr::Spec11({ { { } } })),
                             (true, _) if choice_ == 12usize =>
                             Some(Repr::Spec12({ { { } } })),
                             (true, _) if choice_ == 13usize =>
                             Some(Repr::Spec13({ { { } } })),
                             (true, _) if choice_ == 14usize =>
                             Some(Repr::Spec14({ { { } } })),
                             (true, _) if choice_ == 15usize =>
                             Some(Repr::Spec15({ { { } } })),
                             (true, _) if choice_ == 16usize =>
                             Some(Repr::Spec16({ { { } } })),
                             (true, _) if choice_ == 17usize =>
                             Some(Repr::Spec17({ { { } } })),
                             (true, _) if choice_ == 18usize =>
                             Some(Repr::Spec18({ { { } } })),
                             (true, _) if choice_ == 19usize =>
                             Some(Repr::Spec19({ { { } } })),
                             (true, _) if choice_ == 20usize =>
                             Some(Repr::Spec20({ { { } } })),
                             (true, tok) if choice_ == 21usize =>
                             Some(Repr::Spec21({
                                                   {
                                                       {
                                                           if let &Tok(ref tok)
                                                                  = tok {
                                                               tok.clone()
                                                           } else {
                                                               unreachable!()
                                                           }
                                                       }
                                                   }
                                               })),
                             (true, tok) if choice_ == 22usize =>
                             Some(Repr::Spec22({
                                                   {
                                                       {
                                                           match tok {
                                                               &RustToken::Delim(ref tok)
                                                               => tok.clone(),
                                                               &Tok(ref tok)
                                                               => tok.clone(),
                                                           }
                                                       }
                                                   }
                                               })),
                             _ => None,
                         }; r.expect("marpa-macros: internal error: lexing")
                 }, |args, choice_| {
                     let r =
                         if choice_ == 0usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[2usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec24(opt), Repr::Spec26(rs),
                                  Repr::Spec42(rs0)) =>
                                 Some(Repr::Spec23({
                                                       {
                                                           {
                                                               Context{options:
                                                                           opt,
                                                                       rules:
                                                                           rs,
                                                                       l0_rules:
                                                                           rs0,}
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 1usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec24(mut opts),
                                  Repr::Spec25(o)) =>
                                 Some(Repr::Spec24({
                                                       {
                                                           {
                                                               opts.push(o);
                                                               opts
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 2usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec25(o)) =>
                                 Some(Repr::Spec24({
                                                       {
                                                           {
                                                               let mut opts =
                                                                   Vec::new();
                                                               opts.push(o);
                                                               opts
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 3usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[3usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec1(i), Repr::Spec48(tts)) =>
                                 Some(Repr::Spec25({
                                                       {
                                                           {
                                                               Opt{ident: i,
                                                                   tokens:
                                                                       quote_tokens(cx,
                                                                                    tts),}
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 4usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec26(mut rs), Repr::Spec27(r))
                                 =>
                                 Some(Repr::Spec26({
                                                       { { rs.push(r); rs } }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 5usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec27(r)) =>
                                 Some(Repr::Spec26({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(r);
                                                               v
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 6usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[3usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec1(i), Repr::Spec34(ty),
                                  Repr::Spec28(rhs)) =>
                                 Some(Repr::Spec27({
                                                       {
                                                           {
                                                               Rule{name:
                                                                        i.name,
                                                                    ty: ty,
                                                                    rhs: rhs,}
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 7usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec29(seq)) =>
                                 Some(Repr::Spec28({
                                                       {
                                                           {
                                                               let mut rhs =
                                                                   Vec::new();
                                                               rhs.push(seq);
                                                               rhs
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 8usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[2usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec28(mut rhs),
                                  Repr::Spec29(seq)) =>
                                 Some(Repr::Spec28({
                                                       {
                                                           {
                                                               rhs.push(seq);
                                                               rhs
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 9usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec30(v), Repr::Spec35(b)) =>
                                 Some(Repr::Spec29({
                                                       {
                                                           {
                                                               Alternative{inner:
                                                                               v.0,
                                                                           pats:
                                                                               v.1,
                                                                           action:
                                                                               b,}
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 10usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec32(a)) =>
                                 Some(Repr::Spec30({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(a);
                                                               (v, Vec::new())
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 11usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec33(pat), Repr::Spec32(a)) =>
                                 Some(Repr::Spec30({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(a);
                                                               let mut v2 =
                                                                   Vec::new();
                                                               v2.push((0,
                                                                        pat.unwrap()));
                                                               (v, v2)
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 12usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec30((mut v, mut v2)),
                                  Repr::Spec32(a)) =>
                                 Some(Repr::Spec30({
                                                       {
                                                           {
                                                               v.push(a);
                                                               (v, v2)
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 13usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[2usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec30((mut v, mut v2)),
                                  Repr::Spec33(pat), Repr::Spec32(a)) =>
                                 Some(Repr::Spec30({
                                                       {
                                                           {
                                                               v2.push((v.len(),
                                                                        pat.unwrap()));
                                                               v.push(a);
                                                               (v, v2)
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 14usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec32(a)) =>
                                 Some(Repr::Spec31({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(a);
                                                               v
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 15usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec32(a), Repr::Spec31(mut v))
                                 =>
                                 Some(Repr::Spec31({
                                                       {
                                                           {
                                                               v.insert(0, a);
                                                               v
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 16usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec1(i)) =>
                                 Some(Repr::Spec32({
                                                       {
                                                           {
                                                               NameExpr(i.name)
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 17usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec32(a)) =>
                                 Some(Repr::Spec32({
                                                       {
                                                           {
                                                               ExprOptional(box() a)
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 18usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[4usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec31(body), Repr::Spec31(sep))
                                 =>
                                 Some(Repr::Spec32({
                                                       {
                                                           {
                                                               ExprSeq(body,
                                                                       sep,
                                                                       ast::ZeroOrMore)
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 19usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec2(i)) =>
                                 Some(Repr::Spec33({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(i);
                                                               Some(quote_pat(cx,
                                                                              v))
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 20usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec48(tts)) =>
                                 Some(Repr::Spec33({
                                                       {
                                                           {
                                                               Some(quote_pat(cx,
                                                                              tts))
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 21usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec36(ty)) =>
                                 Some(Repr::Spec34({ { { ty } } })),
                                 _ => None,
                             }
                         } else if choice_ == 22usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec44(block)) =>
                                 Some(Repr::Spec35({
                                                       {
                                                           {
                                                               InlineAction{block:
                                                                                Some(block),}
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 23usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec37(t)) =>
                                 Some(Repr::Spec36({
                                                       {
                                                           {
                                                               P(Ty{id:
                                                                        ast::DUMMY_NODE_ID,
                                                                    node: t,
                                                                    span:
                                                                        DUMMY_SP,})
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 24usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec38(path)) =>
                                 Some(Repr::Spec37({
                                                       {
                                                           {
                                                               TyPath(None,
                                                                      path)
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 25usize {
                             match (true,) {
                                 (true,) =>
                                 Some(Repr::Spec37({
                                                       {
                                                           {
                                                               TyTup(Vec::new())
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 26usize {
                             match (true,) {
                                 (true,) =>
                                 Some(Repr::Spec37({ { { TyInfer } } })),
                                 _ => None,
                             }
                         } else if choice_ == 27usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec39(ps)) =>
                                 Some(Repr::Spec38({
                                                       {
                                                           {
                                                               ast::Path{global:
                                                                             true,
                                                                         segments:
                                                                             ps,
                                                                         span:
                                                                             DUMMY_SP,}
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 28usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec39(ps)) =>
                                 Some(Repr::Spec38({
                                                       {
                                                           {
                                                               ast::Path{global:
                                                                             false,
                                                                         segments:
                                                                             ps,
                                                                         span:
                                                                             DUMMY_SP,}
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 29usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[2usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec39(mut ps), Repr::Spec40(p))
                                 =>
                                 Some(Repr::Spec39({
                                                       { { ps.push(p); ps } }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 30usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec40(p)) =>
                                 Some(Repr::Spec39({
                                                       {
                                                           {
                                                               let mut ps =
                                                                   Vec::new();
                                                               ps.push(p);
                                                               ps
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 31usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec1(i)) =>
                                 Some(Repr::Spec40({
                                                       {
                                                           {
                                                               PathSegment{identifier:
                                                                               i,
                                                                           parameters:
                                                                               PathParameters::none(),}
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 32usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec1(i), Repr::Spec41(param))
                                 =>
                                 Some(Repr::Spec40({
                                                       {
                                                           {
                                                               PathSegment{identifier:
                                                                               i,
                                                                           parameters:
                                                                               param,}
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 33usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec36(t)) =>
                                 Some(Repr::Spec41({
                                                       {
                                                           {
                                                               let mut ts =
                                                                   Vec::new();
                                                               ts.push(t);
                                                               AngleBracketedParameters(AngleBracketedParameterData{lifetimes:
                                                                                                                        Vec::new(),
                                                                                                                    types:
                                                                                                                        OwnedSlice::from_vec(ts),
                                                                                                                    bindings:
                                                                                                                        OwnedSlice::empty(),})
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 34usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec42(mut rs), Repr::Spec43(r))
                                 =>
                                 Some(Repr::Spec42({
                                                       { { rs.push(r); rs } }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 35usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec43(r)) =>
                                 Some(Repr::Spec42({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(r);
                                                               v
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 36usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[3usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[4usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec1(i), Repr::Spec34(ty),
                                  Repr::Spec49(mut rhs), Repr::Spec35(b)) =>
                                 Some(Repr::Spec43({
                                                       {
                                                           {
                                                               let pat =
                                                                   if rhs[1]
                                                                          ==
                                                                          Colon
                                                                      {
                                                                       let p =
                                                                           quote_pat(cx,
                                                                                     rhs.clone());
                                                                       rhs.remove(0);
                                                                       rhs.remove(0);
                                                                       Some(p)
                                                                   } else {
                                                                       None
                                                                   };
                                                               L0Rule{name:
                                                                          i.name,
                                                                      ty: ty,
                                                                      pat:
                                                                          pat,
                                                                      rhs:
                                                                          rhs,
                                                                      action:
                                                                          b,}
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 37usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec47(tts)) =>
                                 Some(Repr::Spec44({
                                                       {
                                                           {
                                                               quote_block(cx,
                                                                           tts)
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 38usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec47(tt)) =>
                                 Some(Repr::Spec45({ { { tt } } })),
                                 _ => None,
                             }
                         } else if choice_ == 39usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec46(tt)) =>
                                 Some(Repr::Spec45({ { { tt } } })),
                                 _ => None,
                             }
                         } else if choice_ == 40usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec21(t)) =>
                                 Some(Repr::Spec46({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(t);
                                                               v
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 41usize {
                             match (true,) {
                                 (true,) =>
                                 Some(Repr::Spec46({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(OpenDelim(Paren));
                                                               v.push(CloseDelim(Paren));
                                                               v
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 42usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec48(tts)) =>
                                 Some(Repr::Spec46({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(OpenDelim(Paren));
                                                               v.extend(tts.into_iter());
                                                               v.push(CloseDelim(Paren));
                                                               v
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 43usize {
                             match (true,) {
                                 (true,) =>
                                 Some(Repr::Spec46({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(OpenDelim(Bracket));
                                                               v.push(CloseDelim(Bracket));
                                                               v
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 44usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec48(tts)) =>
                                 Some(Repr::Spec46({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(OpenDelim(Bracket));
                                                               v.extend(tts.into_iter());
                                                               v.push(CloseDelim(Bracket));
                                                               v
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 45usize {
                             match (true,) {
                                 (true,) =>
                                 Some(Repr::Spec47({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(OpenDelim(Brace));
                                                               v.push(CloseDelim(Brace));
                                                               v
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 46usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec48(tts)) =>
                                 Some(Repr::Spec47({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(OpenDelim(Brace));
                                                               v.extend(tts.into_iter());
                                                               v.push(CloseDelim(Brace));
                                                               v
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 47usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec45(tt)) =>
                                 Some(Repr::Spec48({ { { tt } } })),
                                 _ => None,
                             }
                         } else if choice_ == 48usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec48(mut tts),
                                  Repr::Spec45(tt)) =>
                                 Some(Repr::Spec48({
                                                       {
                                                           {
                                                               tts.extend(tt.into_iter());
                                                               tts
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 49usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec46(tt)) =>
                                 Some(Repr::Spec49({ { { tt } } })),
                                 _ => None,
                             }
                         } else if choice_ == 50usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec49(mut tts),
                                  Repr::Spec46(tt)) =>
                                 Some(Repr::Spec49({
                                                       {
                                                           {
                                                               tts.extend(tt.into_iter());
                                                               tts
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 51usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec22(a)) =>
                                 Some(Repr::Spec50({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(a);
                                                               v
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 52usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec50(mut ary),
                                  Repr::Spec22(a)) =>
                                 Some(Repr::Spec50({
                                                       {
                                                           {
                                                               ary.push(a);
                                                               ary
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 53usize {
                             match (true,) {
                                 (true,) => Some(Repr::Spec51({ { { } } })),
                                 _ => None,
                             }
                         } else { None };
                     r.expect("marpa-macros: internal error: eval") })
        }
    };

    let mut ast_ = None;
    let mut namespace_ = None;
    let mut discard_rule = None;

    for mut ast in grammar.parses_iter(&tokens[..]) {
        let mut namespace: HashMap<ast::Name, (u32, u32)> = HashMap::new();

        let mut rules = vec![];
        for (n, rule) in ast.rules.drain().enumerate() {
            match namespace.entry(rule.name) {
                Vacant(mut vacant) => {
                    vacant.insert((1, rules.len() as u32));
                    rules.push(rule);
                }
                Occupied(mut occupied) => {
                    let &(x, y) = occupied.get();
                    assert_eq!(x, 1);
                    rules[y as usize].rhs.extend(rule.rhs.into_iter());
                }
            }
        }
        ast.rules = rules;

        let mut rules = vec![];
        for (n, rule) in ast.l0_rules.drain().enumerate() {
            if rule.name.as_str() == "discard" {
                discard_rule = Some(rule);
                continue;
            }
            match namespace.entry(rule.name) {
                Vacant(mut vacant) => {
                    vacant.insert((0, rules.len() as u32));
                    rules.push(rule);
                }
                Occupied(mut occupied) => {
                    panic!("duplicate l0");
                }
            }
        }
        ast.l0_rules = rules;

        ast_ = Some(ast);
        namespace_ = Some(namespace);
    }

    (ast_.unwrap(), namespace_.unwrap(), discard_rule)
}

fn expand_grammar(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree])
                  -> Box<MacResult+'static> {
    let sess = cx.parse_sess();
    let mut trdr = lexer::new_tt_reader(&sess.span_diagnostic, None, None, tts.to_vec());

    let mut tokens = vec![];
    let mut tok = transcribe::tt_next_token(&mut trdr).tok;
    while tok != Eof {
        let t = mem::replace(&mut tok, transcribe::tt_next_token(&mut trdr).tok);
        match t {
            OpenDelim(..) | CloseDelim(..) => {
                tokens.push(RustToken::Delim(t));
            }
            _ => {
                tokens.push(RustToken::Tok(t));
            }
        }
    }

    let mac = MacEager::expr(quote_expr!(cx, unreachable!()));

    // call
    let (mut ast, mut namespace, discard_rule) = parse_ast(cx, &tokens[..]);

    let rules_offset = ast.rules.len();

    let mut new_rules_tmp = vec![];
    let mut seq_rules = vec![];

    {
        let mut cx = ExtractionContext {
            namespace: &mut namespace,
            l0_types: ast.l0_rules.iter().map(|r| r.ty.clone()).collect(),
            types: ast.rules.iter().map(|r| r.ty.clone()).collect(),
            sp: sp,
            ext: cx,
        };

        for rule in &mut ast.rules {
            let (new_rules, new_seqs) = rule.extract(&mut cx);
            for (n, new_rule) in new_rules.into_iter().enumerate() {
                cx.namespace.insert(new_rule.name, (1, rules_offset as u32 + new_rules_tmp.len() as u32));
                new_rules_tmp.push(new_rule);
            }

            seq_rules.extend(new_seqs.into_iter());
        }
    };

    ast.rules.extend(new_rules_tmp.into_iter());

    for (n, new_rule) in seq_rules.iter().enumerate() {
        namespace.insert(new_rule.name, (2, n as u32));
    }

    let (variant_names, variant_tys): (Vec<_>, Vec<_>) =
            ast.l0_rules.iter().map(|r| &r.ty)
                .chain(ast.rules.iter().map(|r| &r.ty))
                .chain(seq_rules.iter().map(|r| &r.ty))
                .enumerate().map(|(n, ty)| {
        (repr_variant(n as u32), ty.clone())
    }).unzip();

    let num_syms = ast.l0_rules.len() + ast.rules.len() + seq_rules.len();
    let num_scan_syms = ast.l0_rules.len();
    let num_rule_alts = ast.rules.iter().map(|rule| rule.rhs.len()).sum();
    let num_rules = ast.rules.len();
    let num_seq_rules = seq_rules.len();
    let num_rule_ids = num_rule_alts + num_seq_rules;
    let offset_rules = ast.l0_rules.len();
    let offset_seq_rules = ast.l0_rules.len() + ast.rules.len();

    // renumerate
    for (_, &mut (kind, ref mut nth)) in namespace.iter_mut() {
        if kind == 1 {
            *nth += offset_rules as u32;
        } else if kind == 2 {
            *nth += offset_seq_rules as u32;
        }
    }

    let mut seq_rule_names = vec![];
    let mut seq_rule_rhs = vec![];
    let mut seq_rule_separators = vec![];

    let mut rule_seq_cond_n = vec![];
    let mut rule_seq_action_c = vec![];
    let mut rule_seq_value_c = vec![];

    // all produced sequences
    for (n, rule) in seq_rules.iter().enumerate() {
        seq_rule_names.push(n + offset_seq_rules);
        seq_rule_rhs.push(namespace[rule.rhs].1 as usize);
        seq_rule_separators.push(namespace[rule.sep].1 as usize);
        rule_seq_cond_n.push(num_rule_alts + n);
        rule_seq_action_c.push(repr_variant((n + offset_seq_rules) as u32));
        rule_seq_value_c.push(repr_variant(namespace[rule.rhs].1));
    }

    let rules = ast.rules.iter().flat_map(|rule| {
        rule.rhs.iter().map(|alt| {
            alt.inner.iter().map(|expr| {
                namespace[expr.name()].1 as usize
            }).collect::<Vec<_>>()
        })
    }).collect::<Vec<_>>();

    let rule_names = ast.rules.iter().flat_map(|rule| {
        iter::repeat(namespace[rule.name].1 as usize).take(rule.rhs.len())
    }).collect::<Vec<_>>();

    let ty_return = ast.rules[0].ty.clone();

    let mut lex_cond_n = vec![];
    let mut lex_tup_pat = vec![];
    let mut lex_action = vec![];
    let mut lex_action_c = vec![];
    let mut lexer_tts = vec![];
    let mut discard_rule = discard_rule.map(|rule| quote_tokens(cx, rule.rhs))
                                       .into_iter()
                                       .collect::<Vec<_>>();

    for (nth, rule) in ast.l0_rules.iter().enumerate() {
        lex_cond_n.push(nth);
        lex_tup_pat.push(rule.pat.clone().unwrap_or_else(|| cx.pat_wild(sp)));
        lex_action.push(rule.action.block.clone().unwrap());
        lex_action_c.push(repr_variant(nth as u32));
        lexer_tts.push(quote_tokens(cx, rule.rhs.clone()));
    }

    let mut rule_cond_n = vec![];
    let mut rule_tup_nth = vec![];
    let mut rule_tup_variant = vec![];
    let mut rule_tup_pat = vec![];
    let mut rule_action = vec![];
    let mut rule_action_c = vec![];
    let mut rule_nulling_offset: usize = 0;

    let mut rule_nulling_rule_id = vec![];
    let mut rule_nulling_cond_n = vec![];
    let mut rule_nulling_action = vec![];
    let mut rule_nulling_action_c = vec![];

    let mut i_n: usize = 0;
    for (nr, rule) in ast.rules.iter().enumerate() {
        for subrule in rule.rhs.iter() {
            if subrule.inner.is_empty() {
                rule_nulling_cond_n.push(i_n);
                rule_nulling_rule_id.push(i_n);
                rule_nulling_action.push(subrule.action.block.clone().unwrap());
                rule_nulling_action_c.push(repr_variant((nr + offset_rules) as u32));
            } else {
                rule_cond_n.push(i_n);
                rule_action.push(subrule.action.block.clone().unwrap());
                rule_action_c.push(repr_variant((nr + offset_rules) as u32));

                let &Alternative {
                    ref inner,
                    ref pats,
                    ..
                } = subrule;

                let (mut tup_nth, mut tup_variant, mut tup_pat) = (vec![], vec![], vec![]);
                for &(nth, ref pat) in pats.iter() {
                    let variant_name = match &inner[nth] { &NameExpr(name) => name, _ => unreachable!() };
                    let variant_name = repr_variant(namespace[variant_name].1);
                    tup_nth.push(nth);
                    tup_variant.push(variant_name);
                    tup_pat.push(pat.clone());
                }
                rule_tup_nth.push(tup_nth);
                rule_tup_variant.push(tup_variant);
                rule_tup_pat.push(tup_pat);
            }
            i_n += 1;
        }
    }

    let num_nulling_syms = rule_nulling_rule_id.len();

    let (lexer, lexer_opt) = ast.options.iter().map(|o| (o.ident, o.tokens.clone()))
                                               .next()
                                               .unwrap_or_else(|| (token::str_to_ident("regex_scanner"), vec![]));

    let start_variant = repr_variant(offset_rules as u32);

    let Token = gensym_ident("Token_");

    // quote the code
    let grammar_expr = quote_tokens!(cx, {
        enum Repr<'a> {
            Continue, // also a placeholder
            Unused(PhantomData<&'a ()>),
            $( $variant_names($variant_tys), )*
        }

        struct Grammar<F, G, L> {
            grammar: ::marpa::Grammar,
            scan_syms: [::marpa::Symbol; $num_scan_syms],
            nulling_syms: [(::marpa::Symbol, u32); $num_nulling_syms],
            rule_ids: [::marpa::Rule; $num_rule_ids],
            lexer: L,
            lex_closure: F,
            eval_closure: G,
        }

        struct Parses<'a, 'b, F: 'b, G: 'b, L: 'b> {
            tree: ::marpa::Tree,
            lex_parse: LPar<'a, 'b>,
            positions: Vec<$Token>,
            stack: Vec<Repr<'a>>,
            parent: &'b mut Grammar<F, G, L>,
        }

        impl<F, G, L: Lexer> Grammar<F, G, L>
            where F: for<'c> FnMut(LOut<'c>, usize) -> Repr<'c>,
                  G: for<'c> FnMut(&mut [Repr<'c>], usize) -> Repr<'c>,
        {
            fn new(lexer: L, lex_closure: F, eval_closure: G) -> Grammar<F, G, L> {
                use marpa::{Config, Symbol, Rule};
                let mut cfg = Config::new();
                let mut grammar = ::marpa::Grammar::with_config(&mut cfg).unwrap();

                let mut syms: [Symbol; $num_syms] = unsafe { ::std::mem::uninitialized() };
                for s in syms.iter_mut() {
                    *s = grammar.symbol_new().unwrap();
                }
                grammar.start_symbol_set(syms[$offset_rules]);

                let mut scan_syms: [Symbol; $num_scan_syms] = unsafe { ::std::mem::uninitialized() };
                for (dst, src) in scan_syms.iter_mut().zip(syms[..$num_scan_syms].iter()) {
                    *dst = *src;
                }

                let rules: [(Symbol, &[Symbol]); $num_rule_alts] = [ $(
                    (syms[$rule_names],
                     &[ $(
                        syms[$rules],
                     )* ]),
                )* ];

                let seq_rules: [(Symbol, Symbol, Symbol); $num_seq_rules] = [ $(
                    (syms[$seq_rule_names],
                     syms[$seq_rule_rhs],
                     syms[$seq_rule_separators])
                )* ];

                let mut rule_ids: [Rule; $num_rule_ids] = unsafe { ::std::mem::uninitialized() };
                
                {
                    for (dst, &(lhs, rhs)) in rule_ids.iter_mut().zip(rules.iter()) {
                        *dst = grammar.rule_new(lhs, rhs).unwrap() ;
                    }
                    for (dst, &(lhs, rhs, sep)) in rule_ids.iter_mut().skip($num_rule_alts).zip(seq_rules.iter()) {
                        *dst = grammar.sequence_new(lhs, rhs, sep).unwrap();
                    }
                };

                let mut nulling_syms: [(Symbol, u32); $num_nulling_syms] = unsafe { ::std::mem::uninitialized() };
                let nulling_rule_id_n: &[usize] = &[$($rule_nulling_rule_id,)*];
                for (dst, &n) in nulling_syms.iter_mut().zip(nulling_rule_id_n.iter()) {
                    *dst = (rules[n].0, n as u32);
                }

                grammar.precompute().unwrap();

                Grammar {
                    lexer: lexer,
                    lex_closure: lex_closure,
                    eval_closure: eval_closure,
                    grammar: grammar,
                    scan_syms: scan_syms,
                    nulling_syms: nulling_syms,
                    rule_ids: rule_ids,
                }
            }

            #[inline]
            fn parses_iter<'a, 'b>(&'b mut self, input: LInp<'a>) -> Parses<'a, 'b, F, G, L> {
                use marpa::{Recognizer, Bocage, Order, Tree, Symbol, ErrorCode};
                let mut recce = Recognizer::new(&mut self.grammar).unwrap();
                recce.start_input();

                let mut lex_parse = self.lexer.new_parse(input);

                let mut positions = vec![];

                let mut ith = 0;

                while !lex_parse.is_empty() {
                    let mut syms: [Symbol; $num_scan_syms] = unsafe { mem::uninitialized() };
                    let mut terminals_expected: [u32; $num_scan_syms] = unsafe { mem::uninitialized() };
                    let terminal_ids_expected = unsafe {
                        recce.terminals_expected(&mut syms[..])
                    };
                    let terminals_expected = &mut terminals_expected[..terminal_ids_expected.len()];
                    for (terminal, &id) in terminals_expected.iter_mut().zip(terminal_ids_expected.iter()) {
                        // TODO optimize find
                        *terminal = self.scan_syms.iter().position(|&sym| sym == id).unwrap() as u32;
                    }

                    let mut iter = match lex_parse.longest_parses_iter(terminals_expected) {
                        Some(iter) => iter,
                        None => break
                    };

                    for token in iter {
                        // let expected: Vec<&str> = terminals_expected.iter().map(|&i| l0_names[i as usize]).collect();
                        recce.alternative(self.scan_syms[token.sym()], positions.len() as i32 + 1, 1);
                        positions.push(token); // TODO optimize
                    }
                    recce.earleme_complete();
                    ith += 1;
                }

                let latest_es = recce.latest_earley_set();

                let mut tree =
                    Bocage::new(&mut recce, latest_es)
                 .and_then(|mut bocage|
                    Order::new(&mut bocage)
                ).and_then(|mut order|
                    Tree::new(&mut order)
                );

                Parses {
                    tree: tree.unwrap(),
                    lex_parse: lex_parse,
                    positions: positions,
                    stack: vec![],
                    parent: self,
                }
            }
        }

        impl<'a, 'b, F, G, L> Iterator for Parses<'a, 'b, F, G, L>
            where F: for<'c> FnMut(LOut<'c>, usize) -> Repr<'c>,
                  G: for<'c> FnMut(&mut [Repr<'c>], usize) -> Repr<'c>,
        {
            type Item = $ty_return;

            fn next(&mut self) -> Option<$ty_return> {
                use marpa::{Step, Value};
                let mut valuator = if self.tree.next() >= 0 {
                    Value::new(&mut self.tree).unwrap()
                } else {
                    return None;
                };
                for &rule in self.parent.rule_ids.iter() {
                    valuator.rule_is_valued_set(rule, 1);
                }

                loop {
                    match valuator.step() {
                        Step::StepToken => {
                            let idx = valuator.token_value() as usize - 1;
                            let elem = self.parent.lex_closure.call_mut(self.lex_parse.get(&self.positions[..], idx));
                            self.stack_put(valuator.result() as usize, elem);
                        }
                        Step::StepRule => {
                            let rule = valuator.rule();
                            let arg_0 = valuator.arg_0() as usize;
                            let arg_n = valuator.arg_n() as usize;
                            let elem = {
                                let slice = self.stack.slice_mut(arg_0, arg_n + 1);
                                let choice = self.parent.rule_ids.iter().position(|r| *r == rule);
                                self.parent.eval_closure.call_mut((slice,
                                                                    choice.expect("unknown rule")))
                            };
                            match elem {
                                Repr::Continue => {
                                    continue
                                }
                                other_elem => {
                                    self.stack_put(arg_0, other_elem);
                                }
                            }
                        }
                        Step::StepNullingSymbol => {
                            let sym = valuator.symbol();
                            let choice = self.parent.nulling_syms.iter().find(|&&(s, _)| s == sym).expect("unknown nulling sym").1;
                            let elem = self.parent.eval_closure.call_mut((&mut [], choice as usize));
                            self.stack_put(valuator.result() as usize, elem);
                        }
                        Step::StepInactive => {
                            break;
                        }
                        other => panic!("unexpected step {:?}", other),
                    }
                }

                let result = self.stack.drain().next();

                match result {
                    Some(Repr::$start_variant(val)) =>
                        Some(val),
                    _ =>
                        None,
                }
            }
        }

        impl<'a, 'b, F, G, L> Parses<'a, 'b, F, G, L> {
            fn stack_put(&mut self, idx: usize, elem: Repr<'a>) {
                if idx == self.stack.len() {
                    self.stack.push(elem);
                } else {
                    self.stack[idx] = elem;
                }
            }
        }

        Grammar::new(
            new_lexer(),
            |arg, choice_| {
                let r = match (true, arg) {
                    $((true, $lex_tup_pat) if choice_ == $lex_cond_n => Some(Repr::$lex_action_c($lex_action)),)*
                    _ => None
                };
                r.expect("marpa-macros: internal error: lexing")
            },
            |args, choice_| {
                let r = $(
                    if choice_ == $rule_cond_n {
                        match ( true, $( mem::replace(&mut args[$rule_tup_nth], Repr::Continue), )* ) {
                            ( true, $( Repr::$rule_tup_variant($rule_tup_pat), )* ) => Some(Repr::$rule_action_c($rule_action)),
                            _ => None
                        }
                    } else
                )+
                $(
                    if choice_ == $rule_nulling_cond_n {
                        Some(Repr::$rule_nulling_action_c($rule_nulling_action))
                    } else
                )*
                $(
                    if choice_ == $rule_seq_cond_n {
                        let v = args.iter_mut().enumerate().filter_map(|(n, arg)|
                            if n & 1 == 0 {
                                match (true, mem::replace(arg, Repr::Continue)) {
                                    (true, Repr::$rule_seq_value_c(elem)) => Some(elem),
                                    _ => None
                                }
                            } else {
                                None
                            }
                        ).collect::<Vec<_>>();
                        if v.len() == args.len() / 2 {
                            Some(Repr::$rule_seq_action_c(v))
                        } else {
                            None
                        }
                    } else
                )* {
                    None
                };
                r.expect("marpa-macros: internal error: eval")
            },
        )
    });

    let grammar_expr = quote_expr!(cx, {
        $lexer!($Token $lexer_opt ; $($($discard_rule)*)*, $( $lexer_tts, )* ; $grammar_expr)
    });

    MacEager::expr(grammar_expr)
}
