#![crate_name = "marpa_macros"]
#![feature(plugin, plugin_registrar, unboxed_closures, quote, box_syntax,
    rustc_private, box_patterns, trace_macros, core, collections)]
#![plugin(marpa_macros, enum_adaptor)]

extern crate rustc;
extern crate syntax;
extern crate marpa;
#[macro_use] extern crate marpa_macros;
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
use syntax::parse;
use syntax::ptr::P;
use syntax::ext::build::AstBuilder;
use syntax::print::pprust;
use syntax::parse::lexer;
use syntax::ext::tt::transcribe;
pub use syntax::parse::token::*;
use syntax::owned_slice::OwnedSlice;

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
    ExprSeq(Vec<Expr>, Option<Vec<Expr>>, ast::KleeneOp),
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
    sep: Option<ast::Name>,
    min: i32,
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
            &mut ExprSeq(ref mut body, ref mut sep, op) => {
                let mut v_body_ty = vec![];

                for e in body.iter_mut() {
                    let (ty, a, b) = e.extract(cx);
                    v_body_ty.push(ty);
                    v_rules.extend(a.into_iter());
                    v_seq_rules.extend(b.into_iter());
                }

                let name = gensym_ident("_name_");
                let name_body = gensym_ident("_name_body_");
                let mut name_sep = None;

                let v_body = mem::replace(body, vec![]);
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

                if let &mut Some(ref mut sep) = sep {
                    let name_sep_some = gensym_ident("_name_sep_").name;
                    name_sep = Some(name_sep_some);
                    v_rules.push(Rule {
                        name: name_sep_some,
                        ty: quote_ty!(cx.ext, ()),
                        rhs: vec![
                            Alternative {
                                inner: mem::replace(sep, vec![]),
                                pats: vec![],
                                action: InlineAction { block: Some(cx.ext.block_expr(cx.ext.expr_tuple(cx.sp, vec![]))) },
                            }
                        ],
                    });
                }

                let ty = quote_ty!(cx.ext, Vec<$inner_ty>);

                v_seq_rules.push(SeqRule {
                    name: name.name,
                    ty: ty.clone(),
                    rhs: name_body.name,
                    sep: name_sep,
                    min: if op == ast::OneOrMore { 1 } else { 0 },
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

fn quote_tokens(_: &mut ExtCtxt, toks: Vec<Token>) -> Vec<TokenTree> {
    toks.into_iter().map(|t| TtToken(DUMMY_SP, t)).collect()
}

#[derive(Debug)]
pub enum RustToken {
    Delim(Token),
    Tok(Token)
}

fn parse_ast(cx: &mut ExtCtxt, tokens: &[RustToken]) -> (Context, HashMap<ast::Name, (u32, u32)>, Option<L0Rule>) {
    type Alt = (Option<P<Pat> >, Expr);

    let mut grammar =
    // EXPAND:
    grammar! {
        enum_adaptor!(RustToken);

        start -> Context ::=
            opt:[option]* statements:[stmt]* l0_statements:[l0_stmt]* {
                Context {
                    options: opt,
                    rules: statements,
                    l0_rules: l0_statements
                }
            } ;

        option -> Opt ::=
            i:ident not lparen tts:tts rparen semi {
                Opt { ident: i, tokens: quote_tokens(cx, tts) }
            } ;

        l0_stmt -> L0Rule ::=
            i:ident rarrow ty:ty squiggly (mut rhs):paren_bracket_tts b:inline_action semi {
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

        stmt -> Rule ::=
            i:ident rarrow ty:ty mod_eq rhs:[list]{pipe}* semi {
                Rule {
                    name: i.name,
                    ty: ty,
                    rhs: rhs,
                }
            } ;

        list -> Alternative ::=
            v:[pat_expr]* action:inline_action {
                let mut pats = Vec::new();
                let inner = v.into_iter().enumerate().map(|(n, (pat, expr))| {
                    if let Some(pat) = pat {
                        pats.push((n, pat));
                    }
                    expr
                }).collect();
                Alternative { inner: inner, pats: pats, action: action, }
            } ;
 
        pat_expr -> Alt ::=
            pat:bind_pat? a:atom {
                (pat, a)
            } ;

        expr_list -> Vec<Expr> ::=
            a:[atom]* {
                a
            } ;

        atom -> Expr ::=
            i:ident {
                NameExpr(i.name)
            }
            | a:atom question {
                ExprOptional(box a)
            }
            | lbracket body:expr_list rbracket lbrace sep:expr_list rbrace op:kleene_op {
                ExprSeq(body, Some(sep), op)
            }
            | lbracket body:expr_list rbracket op:kleene_op {
                ExprSeq(body, None, op)
            } ;

        kleene_op -> ast::KleeneOp ::=
            star { ast::ZeroOrMore }
            | plus { ast::OneOrMore } ;

        bind_pat -> P<Pat> ::=
            i:ident_tok colon {
                let mut v = Vec::new();
                v.push(i);
                quote_pat(cx, v)
            } 
            | lparen tts:tts rparen colon {
                quote_pat(cx, tts)
            } ;

        inline_action -> InlineAction ::=
            block:block {
                InlineAction { block: Some(block) }
            } ;

        ty -> P<Ty> ::=
            t:ty_ {
                P(Ty {
                    id: ast::DUMMY_NODE_ID,
                    node: t,
                    span: DUMMY_SP
                })
            } ;

        // Rust's type
        ty_ -> Ty_ ::=
            path:path {
                TyPath(None, path)
            }
            | lparen tup:[ty]{comma}* rparen {
                TyTup(Vec::new())
            }
            | underscore {
                TyInfer
            } ;

        path -> ast::Path ::=
            global:mod_sep? ps:[path_segment]{mod_sep}+ {
                ast::Path {
                    global: global.is_some(),
                    segments: ps,
                    span: DUMMY_SP
                }
            } ;

        path_segment -> PathSegment ::=
            i:ident param:path_param? {
                PathSegment {
                    identifier: i,
                    parameters: param.unwrap_or_else(|| PathParameters::none())
                }
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
        comma -> () ~ Tok(Comma) {} ;
        semi -> () ~ Tok(Semi) {} ;
        rarrow -> () ~ Tok(RArrow) {} ;
        colon -> () ~ Tok(Colon) {} ;
        pipe -> () ~ Tok(BinOp(Or)) {} ;
        underscore -> () ~ Tok(Underscore) {} ;
        question -> () ~ Tok(Question) {} ;
        star -> () ~ Tok(BinOp(Star)) {} ;
        plus -> () ~ Tok(BinOp(Plus)) {} ;

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
    }
    // :EXPAND
    ;

    let mut ast_ = None;
    let mut namespace_ = None;
    let mut discard_rule = None;

    for mut ast in grammar.parses_iter(&tokens[..]) {
        let mut namespace: HashMap<ast::Name, (u32, u32)> = HashMap::new();

        let mut rules = vec![];
        for rule in ast.rules.drain() {
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
        for rule in ast.l0_rules.drain() {
            if rule.name.as_str() == "discard" {
                discard_rule = Some(rule);
                continue;
            }
            match namespace.entry(rule.name) {
                Vacant(mut vacant) => {
                    vacant.insert((0, rules.len() as u32));
                    rules.push(rule);
                }
                Occupied(_) => {
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
    // let num_rules = ast.rules.len();
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
    let mut seq_rule_min = vec![];
    let mut seq_rule_separators = vec![];
    let mut has_separator = vec![];

    let mut rule_seq_cond_n = vec![];
    let mut rule_seq_action_c = vec![];
    let mut rule_seq_value_c = vec![];

    // all produced sequences
    for (n, rule) in seq_rules.iter().enumerate() {
        seq_rule_names.push(n + offset_seq_rules);
        seq_rule_rhs.push(namespace[rule.rhs].1 as usize);
        seq_rule_min.push(rule.min);
        if let Some(sep) = rule.sep {
            seq_rule_separators.push(namespace[sep].1 as usize);
            has_separator.push(true);
        } else {
            seq_rule_separators.push(0);
            has_separator.push(false);
        };
        rule_seq_cond_n.push(num_rule_alts + n);
        rule_seq_action_c.push(repr_variant((n + offset_seq_rules) as u32));
        rule_seq_value_c.push(repr_variant(namespace[rule.rhs].1));
    }

    let has_separator2 = has_separator.clone();

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
    let discard_rule = discard_rule.map(|rule| quote_tokens(cx, rule.rhs))
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

                let seq_rules: [(Symbol, Symbol, Option<Symbol>, i32); $num_seq_rules] = [ $(
                    (syms[$seq_rule_names],
                     syms[$seq_rule_rhs],
                     if $has_separator { Some(syms[$seq_rule_separators as usize]) } else { None },
                     $seq_rule_min),
                )* ];

                let mut rule_ids: [Rule; $num_rule_ids] = unsafe { ::std::mem::uninitialized() };
                
                {
                    for (dst, &(lhs, rhs)) in rule_ids.iter_mut().zip(rules.iter()) {
                        *dst = grammar.rule_new(lhs, rhs).unwrap() ;
                    }
                    for (dst, &(lhs, rhs, sep, min)) in rule_ids.iter_mut().skip($num_rule_alts).zip(seq_rules.iter()) {
                        *dst = grammar.sequence_new(lhs, rhs, sep, min).unwrap();
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
                        let has_sep = $has_separator2;
                        let v = args.iter_mut().enumerate().filter_map(|(n, arg)|
                            if n & 1 == 0 || !has_sep {
                                match (true, mem::replace(arg, Repr::Continue)) {
                                    (true, Repr::$rule_seq_value_c(elem)) => Some(elem),
                                    _ => None
                                }
                            } else {
                                None
                            }
                        ).collect::<Vec<_>>();
                        if v.len() == (args.len() + 1) / 2 || !has_sep {
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
