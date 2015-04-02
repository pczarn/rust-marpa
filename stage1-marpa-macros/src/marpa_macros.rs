#![crate_name = "marpa_macros"]
#![feature(plugin, plugin_registrar, unboxed_closures, quote, globs, macro_rules, box_syntax, rustc_private, box_patterns, trace_macros)]
#![plugin(marpa_macros, enum_adaptor)]

extern crate rustc;
extern crate syntax;
extern crate marpa;
extern crate marpa_macros;
extern crate enum_adaptor;
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
                // match expr {
                //     &mut ExprOptional(inner) => {
                //         let (ty, n_r, n_seq) = inner.extract();
                //         let name = gensym_ident("");
                //         new_rules.push(Rule {
                //             name: name,
                //             ty: quote_ty!(cx, Option<$ty>),
                //             rhs: vec![

                //             ],
                //         });
                //     }
                // }
            }
        }

        (new_rules, new_seq_rules)
    }
}

impl Expr {
    // Transforms an expr into Tagged or Untagged
    fn extract(&mut self, cx: &mut ExtractionContext) -> (P<Ty>, Vec<Rule>, Vec<SeqRule>) {
        let mut replace_with = None;
        let mut v_rules = vec![];
        let mut v_seq_rules = vec![];

        let tup = match self {
            &mut Tagged(ref tagged) => {
                (cx.types[cx.namespace[tagged.name.ident.name].1 as usize].clone(), vec![], vec![])
            }
            &mut Untagged(ref name_expr) => {
                (cx.types[cx.namespace[name_expr.ident.name].1 as usize].clone(), vec![], vec![])
            }
            &mut ExprOptional(ref mut inner) => {
                let (inner_ty, new_r, new_seq) = inner.extract(cx);
                v_rules.extend(new_r.into_iter());
                v_seq_rules.extend(new_seq.into_iter());
                let pat_ident = gensym_ident("");
                let pat = cx.ext.pat_ident(cx.sp, pat_ident);
                let name = gensym_ident("");
                let ty = quote_ty!(cx.ext, Option<$inner_ty>);

                v_rules.push(Rule {
                    name: name.name,
                    ty: ty.clone(),
                    rhs: vec![
                        Alternative {
                            inner: vec![Tagged(TaggedExpr {
                                pat: pat,
                                name: NameExpr {
                                    ident: inner.ident(),
                                }
                            })],
                            action: InlineAction {
                                block: Some(cx.ext.block_expr(cx.ext.expr_some(cx.sp, cx.ext.expr_ident(cx.sp, pat_ident)))),
                            }
                        },
                        Alternative {
                            inner: vec![],
                            action: InlineAction {
                                block: Some(cx.ext.block_expr(cx.ext.expr_none(cx.sp))),
                            }
                        }
                    ],
                });

                replace_with = Some(Untagged(NameExpr { ident: name }));

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

                let name = gensym_ident("");
                let name_body = gensym_ident("");
                let name_sep = gensym_ident("");

                let v_body = mem::replace(body, vec![]);

                // make: action
                let num_tagged = v_body.iter().filter_map(|e|
                    if let &Tagged(_) = e {
                        Some(())
                    } else {
                        None
                    }
                ).count();

                let num_ignored = v_body.iter().filter_map(|e|
                    if let &Tagged(ref t) = e {
                        if t.pat.node == ast::PatWild(ast::PatWildSingle) {
                            Some(())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                ).count();

                let (v_body, action, inner_ty) =
                if v_body.len() == 1 && num_tagged == 0 && num_ignored == 0 {
                    let pat_ident = gensym_ident("");
                    let v_body = vec![Tagged(TaggedExpr {
                        pat: cx.ext.pat_ident(cx.sp, pat_ident),
                        name: NameExpr {
                            ident: v_body[0].ident(),
                        },
                    })];
                    (v_body, InlineAction { block: Some(cx.ext.block_expr(cx.ext.expr_ident(cx.sp, pat_ident))) }, v_body_ty.pop().unwrap())
                // } else if num_tagged > num_ignored {
                //     // struct

                } else {
                    // tuple
                    unreachable!()
                };

                v_rules.push(Rule {
                    name: name_body.name,
                    ty: inner_ty.clone(),
                    rhs: vec![
                        Alternative {
                            inner: v_body,
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

                replace_with = Some(Untagged(NameExpr { ident: name }));

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
            &Tagged(ref e) => e.name.ident.name,
            &Untagged(ref e) => e.ident.name,
            _ => unreachable!()
        }
    }

    fn ident(&self) -> ast::Ident {
        match self {
            &Tagged(ref e) => e.name.ident,
            &Untagged(ref e) => e.ident,
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

fn name(cx: &mut ExtCtxt, tokens: &[RustToken]) -> (Context, HashMap<ast::Name, (u32, u32)>, Option<L0Rule>) {
    let mut grammar = grammar! {
        enum_adaptor!(RustToken);

        start -> Context ::=
            opt:options rs:statements rs0:l0_statements {
                Context { options: opt, rules: rs, l0_rules: rs0 }
            } ;

        options -> Vec<Opt> ::=
            (mut opts):options o:option {
                opts.push(o);
                opts
            }
            | o:option {
                let mut opts = Vec::new();
                opts.push(o);
                opts
            } ;

        option -> Opt ::=
            i:ident not lparen tts:tts rparen semi {
                Opt { ident: i, tokens: quote_tokens(cx, tts) }
            } ;

        statements -> Vec<Rule> ::=
            (mut rs):statements r:stmt {
                rs.push(r);
                rs
            }
            | r:stmt {
                let mut v = Vec::new();
                v.push(r);
                v
            } ;

        stmt -> Rule ::=
            i:ident ty:rtype mod_eq rhs:alternative semi {
                Rule {
                    name: i.name,
                    ty: ty,
                    rhs: rhs,
                }
            } ;

        alternative -> Vec<Alternative> ::=
            seq:list {
                let mut rhs = Vec::new();
                rhs.push(seq);
                rhs
            }
            | (mut rhs):alternative pipe seq:list {
                rhs.push(seq);
                rhs
            } ;

        list -> Alternative ::=
            (v, v2):pat_expr_list b:inline_action {
                Alternative { inner: v, pats: v2, action: b, }
            } ;

        pat_expr_list -> (Vec<Expr>, Vec<(usize, P<Pat>)>) ::=
            a:atom {
                let mut v = Vec::new();
                v.push(a);
                v
            }
            | pat:bind_pat a:atom {
                let mut v = Vec::new();
                v.push(a);
                let mut v2 = Vec::new();
                v2.push((0, pat.unwrap()));
                (v, v2)
            }
            | (mut v, mut v2):pat_expr_list a:atom {
                v.push(a);
                (v, v2)
            }
            | (mut v, mut v2):pat_expr_list pat:bind_pat a:atom {
                v2.push((v.len(), pat.unwrap()));
                v.push(a);
                (v, v2)
            } ;

        expr_list -> Vec<Expr> ::=
            a:atom {
                let mut v = Vec::new();
                v.push(a);
                v
            }
            | a:atom (mut v):expr_list {
                v.insert(0, a);
                v
            } ;

        atom -> Expr ::=
            pat:bind_pat i:ident {
                match pat {
                    Some(pat) =>
                        Tagged(TaggedExpr {
                            pat: pat,
                            name: NameExpr { ident: i }
                        }),
                    None =>
                        Untagged(NameExpr { ident: i }),
                }
            }
            | i:ident {
                Untagged(NameExpr { ident: i })
            }
            | a:atom question {
                ExprOptional(box a)
            }
            | lbracket body:expr_list rbracket lbrace sep:expr_list rbrace star {
                ExprSeq(body, sep, ast::ZeroOrMore)
            } ;

        bind_pat -> Option<P<Pat> > ::=
            i:ident_tok colon {
                let mut v = Vec::new();
                v.push(i);
                Some(quote_pat(cx, v))
            } 
            | lparen tts:tts rparen colon {
                Some(quote_pat(cx, tts))
            } ;

        rtype -> P<Ty> ::= rarrow ty:ty { ty } ;

        inline_action -> InlineAction ::=
            block:block {
                InlineAction { block: Some(block) }
            } ;

        ty -> P<Ty> ::=
            t:ty_ {
                P(Ty { id: ast::DUMMY_NODE_ID, node: t, span: DUMMY_SP })
            } ;

        // Rust's type
        ty_ -> Ty_ ::=
            path:path {
                TyPath(None, path)
            }
            | lparen rparen {
                TyTup(Vec::new())
            }
            | underscore {
                TyInfer
            } ;

        path -> ast::Path ::=
            mod_sep ps:path_segments {
                ast::Path { global: true, segments: ps, span: DUMMY_SP }
            }
            | ps:path_segments {
                ast::Path { global: false, segments: ps, span: DUMMY_SP }
            } ;

        path_segments -> Vec<PathSegment> ::=
            (mut ps):path_segments mod_sep p:path_segment {
                ps.push(p);
                ps
            }
            | p:path_segment {
                let mut ps = Vec::new();
                ps.push(p);
                ps
            } ;

        path_segment -> PathSegment ::=
            i:ident {
                PathSegment { identifier: i, parameters: PathParameters::none() }
            }
            | i:ident param:path_param {
                PathSegment { identifier: i, parameters: param }
            } ;

        path_param -> PathParameters ::=
            lt t:ty gt {
                let mut ts = Vec::new();
                ts.push(t);
                AngleBracketedParameters(AngleBracketedParameterData {
                    lifetimes: Vec::new(),
                    types: OwnedSlice::from_vec(ts),
                    bindings: OwnedSlice::empty(),
                })
            } ;

        // TODO: handle empty rule list
        l0_statements -> Vec<L0Rule> ::=
            (mut rs):l0_statements r:l0_stmt {
                rs.push(r);
                rs
            }
            | r:l0_stmt {
                let mut v = Vec::new();
                v.push(r);
                v
            } ;

        l0_stmt -> L0Rule ::=
            i:ident ty:rtype squiggly (mut rhs):paren_bracket_tts b:inline_action semi {
                let pat = if rhs[1] == Colon {
                    let p = quote_pat(cx, rhs.clone());
                    rhs.remove(0);
                    rhs.remove(0);
                    Some(p)
                } else {
                    None
                };
                L0Rule { name: i.name, ty: ty, pat: pat, rhs: rhs, action: b }
            } ;

        // Rust's block
        block -> P<Block> ::=
            tts:brace_tt {
                quote_block(cx, tts)
            } ;

        token_tree -> Vec<Token> ::=
            tt:brace_tt {
                tt
            }
            | tt:paren_bracket_tt {
                tt
            } ;

        paren_bracket_tt -> Vec<Token> ::=
            t:nondelim {
                let mut v = Vec::new();
                v.push(t);
                v
            }
            | lparen rparen {
                let mut v = Vec::new();
                v.push(OpenDelim(Paren));
                v.push(CloseDelim(Paren));
                v
            }
            | lparen tts:tts rparen {
                let mut v = Vec::new();
                v.push(OpenDelim(Paren));
                v.extend(tts.into_iter());
                v.push(CloseDelim(Paren));
                v
            }
            | lbracket rbracket {
                let mut v = Vec::new();
                v.push(OpenDelim(Bracket));
                v.push(CloseDelim(Bracket));
                v
            }
            | lbracket tts:tts rbracket {
                let mut v = Vec::new();
                v.push(OpenDelim(Bracket));
                v.extend(tts.into_iter());
                v.push(CloseDelim(Bracket));
                v
            } ;

        brace_tt -> Vec<Token> ::=
            lbrace rbrace {
                let mut v = Vec::new();
                v.push(OpenDelim(Brace));
                v.push(CloseDelim(Brace));
                v
            }
            | lbrace tts:tts rbrace {
                let mut v = Vec::new();
                v.push(OpenDelim(Brace));
                v.extend(tts.into_iter());
                v.push(CloseDelim(Brace));
                v
            } ;

        tts -> Vec<Token> ::=
            tt:token_tree {
                tt
            }
            | (mut tts):tts tt:token_tree {
                tts.extend(tt.into_iter());
                tts
            } ;

        paren_bracket_tts -> Vec<Token> ::=
            tt:paren_bracket_tt {
                tt
            }
            | (mut tts):paren_bracket_tts tt:paren_bracket_tt {
                tts.extend(tt.into_iter());
                tts
            } ;

        // Rust's tokens
        any_toks -> Vec<Token> ::=
            a:any {
                let mut v = Vec::new();
                v.push(a);
                v
            }
            | (mut ary):any_toks a:any {
                ary.push(a);
                ary
            } ;

        // `::=`
        mod_eq -> () ::= mod_sep eq {};

        // Tokenization is performed by Rust's lexer. Use the enum adaptor to
        // read tokens.
        squiggly -> () ~ Tok(Tilde) {} ;
        ident -> ::syntax::ast::Ident ~
            i : Tok(::syntax::parse::token::Ident(..)) {
                if let &Tok(::syntax::parse::token::Ident(i, _)) = i {
                    i
                } else {
                    unreachable!();
                }
            } ;
        ident_tok -> ::syntax::parse::token::Token ~
            i : Tok(::syntax::parse::token::Ident(..)) {
                if let &Tok(ref i) = i {
                    i.clone()
                } else {
                    unreachable!()
                }
            } ;
        mod_sep -> () ~ Tok(ModSep) {} ;
        not -> () ~ Tok(Not) {} ;
        eq -> () ~ Tok(Eq) {} ;
        lt -> () ~ Tok(Lt) {} ;
        gt -> () ~ Tok(Gt) {} ;
        semi -> () ~ Tok(Semi) {} ;
        rarrow -> () ~ Tok(RArrow) {} ;
        colon -> () ~ Tok(Colon) {} ;
        pipe -> () ~ Tok(BinOp(Or)) {} ;
        underscore -> () ~ Tok(Underscore) {} ;
        question -> () ~ Tok(Question) {} ;
        star -> () ~ Tok(BinOp(Star)) {} ;

        lbrace -> () ~ RustToken::Delim(OpenDelim(Brace)) {} ;
        rbrace -> () ~ RustToken::Delim(CloseDelim(Brace)) {} ;
        lparen -> () ~ RustToken::Delim(OpenDelim(Paren)) {} ;
        rparen -> () ~ RustToken::Delim(CloseDelim(Paren)) {} ;
        lbracket -> () ~ RustToken::Delim(OpenDelim(Bracket)) {} ;
        rbracket -> () ~ RustToken::Delim(CloseDelim(Bracket)) {} ;
        nondelim -> ::syntax::parse::token::Token ~
            tok : Tok(..) {
                if let &Tok(ref tok) = tok {
                    tok.clone()
                } else {
                    unreachable!()
                }
            } ;
        any -> ::syntax::parse::token::Token ~
            tok:_ {
                match tok {
                    &RustToken::Delim(ref tok) => tok.clone(),
                    &Tok(ref tok) => tok.clone(),
                }
            } ;

        discard -> () ~ Tok(Whitespace) {} ;
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
                    vacant.insert((0, rules.len() as u32));
                    rules.push(rule);
                }
                Occupied(mut occupied) => {
                    let &(x, y) = occupied.get();
                    assert_eq!(x, 0);
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
                    vacant.insert((1, rules.len() as u32));
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

    let (mut ast, mut namespace, discard_rule) = name(cx, &tokens[..]);

    let rules_offset = ast.rules.len();

    let mut new_rules_tmp = vec![];
    let mut seq_rules = vec![];

    {
        let mut cx = ExtractionContext {
            namespace: &mut namespace,
            types: ast.rules.iter().map(|r| r.ty.clone()).collect(),
            sp: sp,
            ext: cx,
        };

        for rule in &mut ast.rules {
            let (new_rules, new_seqs) = rule.extract(&mut cx);
            for (n, new_rule) in new_rules.into_iter().enumerate() {
                cx.namespace.insert(new_rule.name, (0, rules_offset as u32 + new_rules_tmp.len() as u32));
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
            ast.rules.iter().map(|r| &r.ty)
                .chain(ast.l0_rules.iter().map(|r| &r.ty))
                .chain(seq_rules.iter().map(|r| &r.ty))
                .enumerate().map(|(n, ty)| {
        (repr_variant(n as u32), ty.clone())
    }).unzip();

    let num_syms = ast.rules.len() + ast.l0_rules.len() + seq_rules.len();
    let num_scan_syms = ast.l0_rules.len();
    let num_rules = ast.rules.iter().map(|rule| rule.rhs.len()).sum();
    let num_rules_offset = ast.rules.len();
    let num_seq_rules = seq_rules.len();
    let num_all_rules = num_rules + num_seq_rules;

    // renumerate
    for (_, &mut (kind, ref mut nth)) in namespace.iter_mut() {
        if kind == 1 {
            *nth += num_rules_offset as u32;
        } else if kind == 2 {
            *nth += (num_rules_offset + num_scan_syms) as u32;
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
        seq_rule_names.push(n);
        seq_rule_rhs.push(namespace[rule.rhs].1 as usize /*+ num_rules_offset*/);
        seq_rule_separators.push(namespace[rule.sep].1 as usize /*+ num_rules_offset*/);
        rule_seq_cond_n.push(num_rules_offset + n);
        rule_seq_action_c.push(repr_variant((n + num_rules_offset + num_scan_syms) as u32));
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
        lex_action_c.push(repr_variant(nth as u32 + num_rules_offset as u32));
        lexer_tts.push(quote_tokens(cx, rule.rhs.clone()));
    }

    let mut rule_cond_n = vec![];
    let mut rule_tup_nth = vec![];
    let mut rule_tup_pat = vec![];
    let mut rule_action = vec![];
    let mut rule_action_c = vec![];
    let mut rule_nulling_offset: usize = 0;

    let mut rule_nulling_rule_n = vec![];
    let mut rule_nulling_cond_n = vec![];
    let mut rule_nulling_action = vec![];
    let mut rule_nulling_action_c = vec![];

    for (nr, rule) in ast.rules.iter().enumerate() {
        for (ns, subrule) in rule.rhs.iter().enumerate() {
            let n = nr + ns;
            if subrule.inner.is_empty() {
                rule_nulling_rule_n.push(nr);
                rule_nulling_action.push(subrule.action.block.clone().unwrap());
                rule_nulling_action_c.push(repr_variant(nr as u32));
            } else {
                rule_cond_n.push(rule_nulling_offset);
                rule_nulling_offset += 1;
                let (tup_nth, tup_pat): (Vec<usize>, Vec<P<Pat>>) =
                subrule.inner.iter().enumerate().filter_map(|(nth, expr)| {
                    match expr {
                        &Tagged(ref t) => {
                            let variant_name = repr_variant(namespace[t.name.ident.name].1);
                            let pat = t.pat.clone();
                            let pat = quote_pat!(cx, Repr::$variant_name($pat));
                            Some((nth, pat))
                        }
                        &Untagged(ref t) =>
                            None,
                        _ => unreachable!()
                    }
                }).unzip();
                rule_tup_nth.push(tup_nth);
                rule_tup_pat.push(tup_pat);
                rule_action.push(subrule.action.block.clone().unwrap());
                rule_action_c.push(repr_variant(nr as u32));
            }
        }
    }

    let num_nulling_syms = rule_nulling_rule_n.len();
    rule_nulling_cond_n.extend(rule_nulling_offset .. rule_nulling_offset + num_nulling_syms);

    let (lexer, lexer_opt) = ast.options.iter().map(|o| (o.ident, o.tokens.clone()))
                                               .next()
                                               .unwrap_or_else(|| (token::str_to_ident("regex_scanner"), vec![]));

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
            nulling_syms: [::marpa::Symbol; $num_nulling_syms],
            rule_ids: [::marpa::Rule; $num_all_rules],
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
                grammar.start_symbol_set(syms[0]);

                let mut scan_syms: [Symbol; $num_scan_syms] = unsafe { ::std::mem::uninitialized() };
                for (dst, src) in scan_syms.iter_mut().zip(syms.iter().skip($num_rules_offset)) {
                    *dst = *src;
                }

                let mut nulling_syms: [Symbol; $num_nulling_syms] = unsafe { ::std::mem::uninitialized() };
                let nulling_rule_n: &[usize] = &[$($rule_nulling_rule_n,)*];
                for (dst, &n) in nulling_syms.iter_mut().zip(nulling_rule_n.iter()) {
                    *dst = syms[n];
                }

                let rules: [(Symbol, &[Symbol]); $num_rules] = [ $(
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

                let mut rule_ids: [Rule; $num_all_rules] = unsafe { ::std::mem::uninitialized() };
                
                {
                    let mut iter_dst = rule_ids.iter_mut();

                    for (dst, &(lhs, rhs)) in iter_dst.by_ref().zip(rules.iter()) {
                        *dst = grammar.rule_new(lhs, rhs).unwrap() ;
                    }
                    for (dst, &(lhs, rhs, sep)) in iter_dst.by_ref().zip(seq_rules.iter()) {
                        *dst = grammar.sequence_new(lhs, rhs, sep).unwrap();
                    }
                };

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
                    for (&id, terminal) in terminal_ids_expected.iter().zip(terminals_expected.iter_mut()) {
                        // TODO optimize find
                        *terminal = self.scan_syms.iter().position(|&sym| sym == id).unwrap() as u32;
                    }

                    let mut iter = match lex_parse.longest_parses_iter(terminals_expected) {
                        Some(iter) => iter,
                        None => break
                    };

                    // println!("#{}", iter.len());
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
                            let choice = self.parent.nulling_syms.iter().position(|s| *s == sym);
                            let elem = self.parent.eval_closure.call_mut((&mut [], choice.expect("unknown nulling sym")));
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
                    Some(Repr::Spec0(val)) =>
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
                r.unwrap()
            },
            |args, choice_| {
                let r = $(
                    if choice_ == $rule_cond_n {
                        match ( true, $( mem::replace(&mut args[$rule_tup_nth], Repr::Continue), )* ) {
                            ( true, $( $rule_tup_pat, )* ) => Some(Repr::$rule_action_c($rule_action)),
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
                r.unwrap()
            },
        )
    });

    let grammar_expr = quote_expr!(cx, {
        $lexer!($Token $lexer_opt ; $($($discard_rule)*)*, $( $lexer_tts, )* ; $grammar_expr)
    });

    MacEager::expr(grammar_expr)
}
