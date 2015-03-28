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
    action: InlineAction,
}

#[derive(Debug)]
enum Expr {
    Tagged(TaggedExpr),
    Untagged(NameExpr),
}

#[derive(Debug)]
struct TaggedExpr {
    pat: P<Pat>,
    name: NameExpr,
}

#[derive(Debug)]
struct NameExpr {
    ident: ast::Ident,
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

// intermediate

#[derive(Debug)]
struct Seq {
    inner: Vec<Expr>,
    ty: Option<P<Ty>>,
    action: InlineAction,
}

impl Rule {
    fn new(name: ast::Name, rhs: Vec<Seq>) -> Rule {
        let mut ty = None;
        for seq in rhs.iter() {
            match (&ty, &seq.ty) {
                (&None, &Some(ref seq_ty)) => {
                    if seq_ty.node != ast::TyInfer {
                        ty = Some(seq_ty.clone());
                    }
                }
                (&Some(_), &Some(ref seq_ty)) => {
                    if seq_ty.node != ast::TyInfer {
                        panic!()
                    }
                }
                _ => {}
            }
        }
        Rule {
            name: name,
            ty: ty.expect("todo: parameterize Repr to allow this"),
            rhs: rhs.into_iter().map(|r| Alternative { inner: r.inner, action: r.action }).collect()
        }
    }
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

fn quote_tokens_p(cx: &mut ExtCtxt, toks: P<Ty>) -> Vec<TokenTree> {
    quote_tokens!(cx, $toks)
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

        start ::= opt:options rs:statements rs0:l0_statements -> Context {
            Context { options: opt, rules: rs, l0_rules: rs0 }
        } ;

        options ::=
            (mut opts):options o:option -> Vec<Opt> {
                opts.push(o);
                opts
            }
            | o:option -> _ {
                let mut opts = Vec::new();
                opts.push(o);
                opts
            } ;

        option ::=
            i:ident not lparen tts:tts rparen semi -> Opt {
                Opt { ident: i, tokens: quote_tokens(cx, tts) }
            } ;

        statements ::=
            (mut rs):statements r:stmt -> Vec<Rule> {
                rs.push(r);
                rs
            }
            | r:stmt -> _ {
                let mut v = Vec::new();
                v.push(r);
                v
            } ;

        stmt ::=
            i:ident mod_eq rhs:alternative semi -> Rule {
                Rule::new(i.name, rhs)
            } ;

        alternative ::=
            seq:list -> Vec<Seq> {
                let mut rhs = Vec::new();
                rhs.push(seq);
                rhs
            }
            | (mut rhs):alternative pipe seq:list -> _ {
                rhs.push(seq);
                rhs
            } ;

        list ::=
            ty:rtype b:inline_action -> Seq {
                Seq { inner: Vec::new(), ty: ty, action: b, }
            }
            | a:atom (mut seq):list -> _ {
                seq.inner.insert(0, a);
                seq
            } ;

        atom ::=
            pat:bind_pat i:ident -> Expr {
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
            | i:ident -> _ {
                Untagged(NameExpr { ident: i })
            } ;

        bind_pat ::=
            i:ident_tok colon -> Option<P<Pat> > {
                let mut v = Vec::new();
                v.push(i);
                Some(quote_pat(cx, v))
            } 
            | lparen tts:tts rparen colon -> _ { Some(quote_pat(cx, tts)) } ;

        rtype ::=
            rarrow ty:ty -> Option<P<Ty> > { Some(ty) } ;

        inline_action ::=
            block:block -> InlineAction {
                InlineAction { block: Some(block) }
            } ;

        ty ::=
            t:ty_ -> P<Ty> {
                P(Ty { id: ast::DUMMY_NODE_ID, node: t, span: DUMMY_SP })
            } ;

        // Rust's type
        ty_ ::=
            path:path -> Ty_ {
                TyPath(None, path)
            }
            | lparen rparen -> _ {
                TyTup(Vec::new())
            }
            | underscore -> _ {
                TyInfer
            } ;

        path ::=
            mod_sep ps:path_segments -> ast::Path {
                ast::Path { global: true, segments: ps, span: DUMMY_SP }
            }
            | ps:path_segments -> _ {
                ast::Path { global: false, segments: ps, span: DUMMY_SP }
            } ;

        path_segments ::=
            (mut ps):path_segments mod_sep p:path_segment -> Vec<PathSegment> {
                ps.push(p);
                ps
            }
            | p:path_segment -> _ {
                let mut ps = Vec::new();
                ps.push(p);
                ps
            } ;

        path_segment ::=
            i:ident -> PathSegment {
                PathSegment { identifier: i, parameters: PathParameters::none() }
            }
            | i:ident param:path_param -> _ {
                PathSegment { identifier: i, parameters: param }
            } ;

        path_param ::=
            lt t:ty gt -> PathParameters {
                let mut ts = Vec::new();
                ts.push(t);
                AngleBracketedParameters(AngleBracketedParameterData {
                    lifetimes: Vec::new(),
                    types: OwnedSlice::from_vec(ts),
                    bindings: OwnedSlice::empty(),
                })
            } ;

        // TODO: handle empty rule list
        l0_statements ::=
            (mut rs):l0_statements r:l0_stmt -> Vec<L0Rule> {
                rs.push(r);
                rs
            }
            | r:l0_stmt -> _ {
                let mut v = Vec::new();
                v.push(r);
                v
            } ;

        l0_stmt ::=
            i:ident squiggly (mut rhs):paren_bracket_tts ty:rtype b:inline_action semi -> L0Rule {
                let pat = if rhs[1] == Colon {
                    rhs.remove(0);
                    rhs.remove(0);
                    Some(quote_pat(cx, rhs.clone()))
                } else {
                    None
                };
                L0Rule { name: i.name, ty: ty.unwrap(), pat: pat, rhs: rhs, action: b }
            } ;

        // Rust's block
        block ::=
            tts:brace_tt -> P<Block> {
                quote_block(cx, tts)
            } ;

        token_tree ::=
            tt:brace_tt -> _ {
                tt
            }
            | tt:paren_bracket_tt -> Vec<Token> {
                tt
            } ;

        paren_bracket_tt ::=
            t:nondelim -> Vec<Token> {
                let mut v = Vec::new();
                v.push(t);
                v
            }
            | lparen rparen -> _ {
                let mut v = Vec::new();
                v.push(OpenDelim(Paren));
                v.push(CloseDelim(Paren));
                v
            }
            | lparen tts:tts rparen -> _ {
                let mut v = Vec::new();
                v.push(OpenDelim(Paren));
                v.extend(tts.into_iter());
                v.push(CloseDelim(Paren));
                v
            }
            | lbracket rbracket -> _ {
                let mut v = Vec::new();
                v.push(OpenDelim(Bracket));
                v.push(CloseDelim(Bracket));
                v
            }
            | lbracket tts:tts rbracket -> _ {
                let mut v = Vec::new();
                v.push(OpenDelim(Bracket));
                v.extend(tts.into_iter());
                v.push(CloseDelim(Bracket));
                v
            } ;

        brace_tt ::=
            lbrace rbrace -> Vec<Token> {
                let mut v = Vec::new();
                v.push(OpenDelim(Brace));
                v.push(CloseDelim(Brace));
                v
            }
            | lbrace tts:tts rbrace -> _ {
                let mut v = Vec::new();
                v.push(OpenDelim(Brace));
                v.extend(tts.into_iter());
                v.push(CloseDelim(Brace));
                v
            } ;

        tts ::=
            tt:token_tree -> Vec<Token> {
                tt
            }
            | (mut tts):tts tt:token_tree -> _ {
                tts.extend(tt.into_iter());
                tts
            } ;

        paren_bracket_tts ::=
            tt:paren_bracket_tt -> Vec<Token> {
                tt
            }
            | (mut tts):paren_bracket_tts tt:paren_bracket_tt -> _ {
                tts.extend(tt.into_iter());
                tts
            } ;

        // Rust's tokens
        any_toks ::=
            a:any -> Vec<Token> {
                let mut v = Vec::new();
                v.push(a);
                v
            }
            | (mut ary):any_toks a:any -> _ {
                ary.push(a);
                ary
            } ;

        // `::=`
        mod_eq ::= mod_sep eq -> () {};

        // Tokenization is performed by Rust's lexer. Use the enum adaptor to
        // read tokens.
        squiggly ~ Tok(Tilde) -> () {} ;
        ident ~ i : Tok(::syntax::parse::token::Ident(..)) -> ::syntax::ast::Ident {
            if let &Tok(::syntax::parse::token::Ident(i, _)) = i {
                i
            } else {
                unreachable!();
            }
        } ;
        ident_tok ~ i : Tok(::syntax::parse::token::Ident(..)) -> ::syntax::parse::token::Token {
            if let &Tok(ref i) = i {
                i.clone()
            } else {
                unreachable!()
            }
        } ;
        mod_sep ~ Tok(ModSep) -> () {} ;
        not ~ Tok(Not) -> () {} ;
        eq ~ Tok(Eq) -> () {} ;
        lt ~ Tok(Lt) -> () {} ;
        gt ~ Tok(Gt) -> () {} ;
        semi ~ Tok(Semi) -> () {} ;
        rarrow ~ Tok(RArrow) -> () {} ;
        colon ~ Tok(Colon) -> () {} ;
        pipe ~ Tok(BinOp(Or)) -> () {} ;
        underscore ~ Tok(Underscore) -> () {} ;

        lbrace ~ RustToken::Delim(OpenDelim(Brace)) -> () {} ;
        rbrace ~ RustToken::Delim(CloseDelim(Brace)) -> () {} ;
        lparen ~ RustToken::Delim(OpenDelim(Paren)) -> () {} ;
        rparen ~ RustToken::Delim(CloseDelim(Paren)) -> () {} ;
        lbracket ~ RustToken::Delim(OpenDelim(Bracket)) -> () {} ;
        rbracket ~ RustToken::Delim(CloseDelim(Bracket)) -> () {} ;
        nondelim ~ tok : Tok(..) -> ::syntax::parse::token::Token {
            if let &Tok(ref tok) = tok {
                tok.clone()
            } else {
                unreachable!()
            }
        } ;
        any ~ tok:_ -> ::syntax::parse::token::Token {
            match tok {
                &RustToken::Delim(ref tok) => tok.clone(),
                &Tok(ref tok) => tok.clone(),
            }
        } ;

        discard ~ Tok(Whitespace) -> () {} ;
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

        // for  {
            
        // }

        // println!("{:?}", ast);
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

    let (ast, namespace, discard_rule) = name(cx, &tokens[..]);

    let (variant_names, variant_tys): (Vec<_>, Vec<_>) =
            ast.rules.iter().map(|r| &r.ty)
                .chain(ast.l0_rules.iter().map(|r| &r.ty))
                .enumerate().map(|(n, ty)| {
        let variant_name = format!("Spec{}", n);
        (token::str_to_ident(&variant_name[..]), ty.clone())
    }).unzip();

    let num_syms = ast.rules.len() + ast.l0_rules.len();
    let num_scan_syms = ast.l0_rules.len();
    let num_rules = ast.rules.iter().map(|rule| rule.rhs.len()).sum();

    let rules = ast.rules.iter().flat_map(|rule| {
        rule.rhs.iter().map(|alt| {
            alt.inner.iter().map(|expr| {
                let name = match expr {
                    &Tagged(ref t) => t.name.ident.name,
                    &Untagged(ref t) => t.ident.name,
                };

                let (k, n) = namespace[name];
                if k == 1 { n + num_rules as u32 } else { n }
            }).collect::<Vec<_>>()
        })
    }).collect::<Vec<_>>();

    let rule_names = ast.rules.iter().flat_map(|rule| {
        iter::repeat(namespace[rule.name].1).take(rule.rhs.len())
    }).collect::<Vec<_>>();

    let ty_return = ast.rules[0].ty.clone();

    let mut lex_cond_n = vec![];
    // let mut lex_tup_arg = vec![];
    let mut lex_tup_pat = vec![];
    let mut lex_action = vec![];
    let mut lexer_tts = vec![];
    let mut discard_rule = discard_rule.map(|rule| quote_tokens(cx, rule.rhs));

    for (nth, rule) in ast.l0_rules.iter().enumerate() {
        lex_cond_n.push(nth);
        // if rule.pat.is_some() {
        //     lex_tup_arg.push(vec![quote_expr!(cx, args)]);
        // } else {
        //     lex_tup_arg.push(vec![]);
        // }
        // lex_tup_pat.push(rule.pat.iter().cloned().collect());
        lex_tup_pat.push(rule.pat.clone().unwrap_or_else(|| cx.pat_wild(sp)));
        lex_action.push(rule.action.block.clone().unwrap());
        lexer_tts.push(quote_tokens(cx, rule.rhs.clone()));
    }

    let mut rule_cond_n = vec![];
    let mut rule_tup_nth = vec![];
    let mut rule_tup_pat = vec![];
    let mut rule_action = vec![];

    for (n, subrule) in ast.rules.iter().flat_map(|r| r.rhs.iter()).enumerate() {
        rule_cond_n.push(n);
        let (tup_nth, tup_pat): (Vec<usize>, Vec<P<Pat>>) =
        subrule.inner.iter().enumerate().filter_map(|(nth, expr)| {
            match expr {
                &Tagged(ref t) =>
                    Some((nth, t.pat.clone())),
                &Untagged(ref t) =>
                    None,
            }
        }).unzip();
        rule_tup_nth.push(tup_nth);
        rule_tup_pat.push(tup_pat);
        rule_action.push(subrule.action.block.clone().unwrap());
    }

    // let (lexer, lexer_opt) = ast.options.iter().find(|opt| opt.ident.as_str() == "").unwrap_or_else(|| (token::str_to_ident("regex_scanner"), vec![]));
    let (lexer, lexer_opt) = ast.options.iter().map(|o| (o.ident, o.tokens.clone())).next().unwrap_or_else(|| (token::str_to_ident("regex_scanner"), vec![]));
    // let lexer_opt = 

    // quote the code
    let grammar_expr = quote_expr!(cx, {
        trait Lexer {
            type Par;
            type Tok;
            type Inp;
            type Out;
        }

        enum Repr<'a> {
            Continue, // also a placeholder
            $( $variant_names($variant_tys), )*
        }

        struct Grammar<F, G, L: Lexer> {
            grammar: ::marpa::Grammar,
            scan_syms: [::marpa::Symbol; $num_scan_syms],
            rule_ids: [::marpa::Rule; $num_rules],
            lexer: L,
            lex_closure: F,
            parse_closure: G,
        }

        struct Parses<'a, 'b, F: 'b, G: 'b, L: Lexer + 'b> {
            tree: ::marpa::Tree,
            lex_parse: L::Par,
            positions: Vec<L::Tok>,
            stack: Vec<Repr<'a>>,
            parent: &'b mut Grammar<F, G, L>,
        }

        impl<F, G, L: Lexer> Grammar<F, G, L> {
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
                for (dst, src) in scan_syms.iter_mut().zip(syms.iter().skip($num_rules)) {
                    *dst = *src;
                }

                let rules: [(Symbol, &[Symbol]); $num_rules] = [ $(
                    ($rule_names,
                     &[ $(
                        syms[$rules],
                     )* ]),
                )* ];

                let mut rule_ids: [Rule; $num_rules] = unsafe { ::std::mem::uninitialized() };
                for (id, &(lhs, rhs)) in rule_ids.iter_mut().zip(rules.iter()) {
                    *id = grammar.rule_new(lhs, rhs).unwrap();
                }

                grammar.precompute().unwrap();

                Grammar {
                    lexer: lexer,
                    lex_closure: lex_closure,
                    eval_closure: eval_closure,
                    grammar: grammar,
                    scan_syms: scan_syms,
                    rule_ids: rule_ids,
                }
            }

            #[inline]
            fn parses_iter<'a, 'b>(&'b mut self, input: L::Inp) -> Parses<'a, 'b, F, G> {
                use marpa::{Recognizer, Bocage, Order, Tree};
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
                        match recce.alternative(self.scan_syms[token.sym()], positions.len() as i32 + 1, 1) {
                            ErrorCode::UnexpectedTokenId => {
                                // println!("{}: expected {:?}, but found {}", ith, expected, l0_names[token.sym()]);
                            }
                            _ => {
                                // println!("{}: expected {:?}, ACCEPTED {}", ith, expected, l0_names[token.sym()]);
                            }
                        }
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

        impl<'a, 'b, F, G, L: Lexer> Parses<'a, 'b, F, G, L>
            where F: for<'c> FnMut(L::Out<'c>, usize) -> Repr<'c>,
                  G: for<'c> FnMut(&mut [Repr<'c>], usize) -> Repr<'c>,
        {
            fn next(&mut self) -> Option<$ty_return> {
                let mut valuator = if self.tree.next() >= 0 {
                    Value::new(&mut self.tree).unwrap()
                } else {
                    return None;
                };
                for &rule in &self.parent.rule_ids {
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
                                self.parent.parse_closure.call_mut((slice,
                                                                    choice.expect("unknown rule")))
                            };
                            match elem {
                                SlifRepr::Continue => {
                                    continue
                                }
                                other_elem => {
                                    self.stack_put(arg_0, other_elem);
                                }
                            }
                        }
                        Step::StepInactive => {
                            break;
                        }
                        other => panic!("unexpected step {:?}", other),
                    }
                }

                let result = self.stack.drain().next();

                match result {
                    Some(Spec0(val)) =>
                        Some(val),
                    _ =>
                        None,
                }
            }
        }

        impl<'a, 'b, F, G> Parses<'a, 'b, F, G> {
            fn stack_put(&mut self, idx: usize, elem: Repr<'a>) {
                if idx == self.stack.len() {
                    self.stack.push(elem);
                } else {
                    self.stack[idx] = elem;
                }
            }
        }

        Grammar::new(
            $lexer!($lexer_opt ; $($($discard_rule)*)*, $( $lexer_tts, )*),
            |arg_, choice_| {
                let args: &[_] = &[arg_];
                $(if choice_ == $lex_cond_n {
                    match (args,) {
                        ($lex_tup_pat,) => Some({ $lex_action }),
                        _ => None
                    }
                })else+
            },
            |args, choice_| {
                $(if choice_ == $rule_cond_n {
                    match ( $( mem::replace(&mut args[$rule_tup_nth], Repr::Continue), )* ) {
                        ( $( $rule_tup_pat, )* ) => Some({ $rule_action }),
                        _ => None
                    }
                })else+
            },
        )
    });

        // enum SlifRepr<'a, $T> {
        //     Continue, // also a placeholder
        //     $variants
        // }

    // unreachable!();
    MacEager::expr(grammar_expr)
    // return mac;
}
