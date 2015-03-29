#![crate_name = "marpa_macros"]

#![feature(plugin_registrar, quote, globs, macro_rules, box_syntax, rustc_private, box_patterns)]

extern crate rustc;
extern crate syntax;
#[macro_use] extern crate log;

use self::RuleRhs::*;
use self::KleeneOp::*;
use self::InlineActionType::*;

use syntax::ast;
use syntax::ast::TokenTree;
use syntax::codemap::respan;
use syntax::codemap::Span;
use syntax::ext::base::{ExtCtxt, MacResult, MacEager};
use syntax::parse::token;
use syntax::parse::parser::Parser;
use syntax::parse;
use syntax::util::interner::{StrInterner, RcStr};
use syntax::ptr::P;
use syntax::ext::build::AstBuilder;
use syntax::print::pprust;
use syntax::parse::common::seq_sep_none;

use std::collections::HashMap;
use std::collections::HashSet;
use std::mem;
use std::fmt;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut ::rustc::plugin::Registry) {
    reg.register_macro("grammar", expand_grammar);
}

macro_rules! rule {
    ($lhs:ident ::= $rhs:expr) => (
        Rule {
            name: $lhs,
            rhs: $rhs,
        }
    )
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
enum InlineActionType {
    InferT,
    StrSlice,
    Continue,
    DirectStr,
    Explicit(u32),
    InferFromRule,
}

#[derive(Clone)]
struct InlineAction {
    block: P<ast::Block>,
    ty_return: InlineActionType,
}

#[derive(Clone, Debug)]
struct InlineBind {
    pat: P<ast::Pat>,
    ty_param: InlineActionType,
}

impl InlineAction {
    fn new(block: P<ast::Block>) -> InlineAction {
        InlineAction {
            block: block,
            ty_return: InferT
        }
    }
}

impl InlineBind {
    fn new(pat: P<ast::Pat>) -> InlineBind {
        InlineBind {
            pat: pat,
            ty_param: InferT
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum KleeneOp {
    ZeroOrMore,
    OneOrMore,
}

#[derive(Clone, Debug)]
struct LexemeRhs {
    tts: Vec<TokenTree>,
    bind: Option<InlineBind>,
    action: Option<InlineAction>,
}

#[derive(Clone, Debug)]
enum RuleRhs {
    Alternative(Vec<RuleRhs>),
    Sequence(Vec<RuleRhs>, Option<InlineAction>),
    Ident(ast::Name, Option<InlineBind>),
    Repeat(Box<RuleRhs>, KleeneOp),
    Lexeme(LexemeRhs),
}

#[derive(Debug)]
struct Rule {
    name: ast::Name,
    rhs: RuleRhs,
}

impl InlineActionType {
    fn pat_match(&self, cx: &mut Context, pat: P<ast::Pat>) -> P<ast::Pat> {
        match *self {
            InferT =>
                quote_pat!(cx.ext, SlifRepr::ValInfer($pat)),
            StrSlice =>
                quote_pat!(cx.ext, SlifRepr::ValLexed($pat)),
            DirectStr =>
                pat,
            Explicit(n) => {
                let n_s = format!("Spec{}", n);
                let ident = token::str_to_ident(n_s.as_slice());
                quote_pat!(cx.ext, SlifRepr::$ident($pat))
            }
            Continue | InferFromRule =>
                unreachable!(),
        }
    }
}

impl InlineAction {
    fn expr_wrap(self, cx: &mut Context) -> P<ast::Expr> {
        let InlineAction { ty_return, block } = self;
        match ty_return {
            InferT =>
                quote_expr!(cx.ext, SlifRepr::ValInfer($block)),
            StrSlice =>
                quote_expr!(cx.ext, SlifRepr::ValLexed($block)),
            Continue =>
                quote_expr!(cx.ext, SlifRepr::Continue),
            Explicit(n) => {
                let n_s = format!("Spec{}", n);
                let ident = token::str_to_ident(n_s.as_slice());
                quote_expr!(cx.ext, SlifRepr::$ident($block))
            }
            DirectStr | InferFromRule =>
                unreachable!(),
        }
    }
}

impl InlineBind {
    fn pat_match(self, cx: &mut Context, nth: usize)
                -> (P<ast::Pat>, Option<(P<ast::Expr>, P<ast::Pat>)>) {
        let InlineBind { ty_param, pat } = self;

        let by_val_ident = match pat.node {
            ast::PatIdent(ast::BindByValue(..), ident, _) if ty_param != DirectStr =>
                Some(ident),
            _ =>
                None
        };

        let mut b_pat = ty_param.pat_match(cx, pat);

        let mut r_stmt = None;

        if let Some(ident) = by_val_ident {
            r_stmt = Some((
                quote_expr!(cx.ext,
                    mem::replace(&mut args[$nth], SlifRepr::Continue)
                ),
                b_pat.clone()));
            b_pat = cx.ext.pat_wild(cx.sp);
        }

        (b_pat, r_stmt)
    }
}

impl RuleRhs {
    fn extract_sub_rules(&mut self, cx: &mut Context, cont: &mut Vec<Rule>, top: bool) {
        match self {
            &mut Lexeme(..) if !top => {
                // displace the lexical rule
                let new_sym = cx.syms.gensym("");
                let ty_pass = StrSlice;
                cx.val_reprs.insert(ty_pass);
                let new_bind;

                match self {
                    &mut Lexeme(ref mut lex) => {
                        lex.action = Some(InlineAction {
                            block: cx.ext.block_expr(quote_expr!(cx.ext, arg_)),
                            ty_return: ty_pass,
                        });
                        new_bind = lex.bind.take().map(|b| InlineBind { pat: b.pat, ty_param: ty_pass });
                    }
                    _ => unreachable!()
                }

                let this = Ident(new_sym, new_bind);
                let displaced_lex = mem::replace(self, this);

                cont.push(rule!(new_sym ::= displaced_lex));
            }
            &mut Repeat(box ref mut sub_rule, _) => match sub_rule {
                &mut Ident(..) => {},
                other_sub_rule => {
                    other_sub_rule.extract_sub_rules(cx, cont, false);
                    let new_sym = cx.syms.gensym("");
                    let other = mem::replace(other_sub_rule, Ident(new_sym, None)); // None??
                    cont.push(rule!(new_sym ::= other));
                }
            },
            &mut Sequence(ref mut seq, _) | &mut Alternative(ref mut seq) => {
                for rule in seq.iter_mut() {
                    rule.extract_sub_rules(cx, cont, false);
                }
            }
            _ => {}
        }
    }

    fn create_seq_rules(&mut self, syms: &mut StrInterner, cont: &mut Vec<Rule>) {
        match self {
            &mut Sequence(ref mut seq, _) | &mut Alternative(ref mut seq) => {
                for rule in seq.iter_mut() {
                    rule.create_seq_rules(syms, cont);
                }
            }
            &mut Repeat(box ref mut repeated, op) => {
                let new_sym = syms.gensym("");
                let repeated = mem::replace(repeated, Ident(new_sym, None));
                match op {
                    ZeroOrMore => {
                        cont.push(rule!(new_sym ::= Sequence(vec![], None)));
                    }
                    OneOrMore => {
                        cont.push(rule!(new_sym ::= repeated.clone()));
                    }
                }
                cont.push(rule!(
                    new_sym ::= Sequence(vec![Ident(new_sym, None), repeated], None)
                ));
            }
            &mut Ident(..) | &mut Lexeme(..) => {}
        }
    }

    fn alternative(self, new: RuleRhs) -> RuleRhs {
        // flattened alternative
        match self {
            Alternative(mut alt_seq) => {
                alt_seq.push(new);
                Alternative(alt_seq)
            }
            other => Alternative(vec![new, other]),
        }
    }
}

impl Rule {
    fn ident(left: ast::Name, right: ast::Name) -> Rule {
        Rule { name: left, rhs: Ident(right, None) }
    }

    fn extract_sub_rules(&mut self, cx: &mut Context, cont: &mut Vec<Rule>) {
        self.rhs.extract_sub_rules(cx, cont, true);
    }

    fn create_seq_rules(&mut self, syms: &mut StrInterner, cont: &mut Vec<Rule>) {
        self.rhs.create_seq_rules(syms, cont);
    }

    fn join_action_ty(&self, tys: &mut HashMap<ast::Name, Option<InlineActionType>>) {
        let ty_return = match self.rhs {
            Sequence(_, ref action) => {
                action.as_ref().map(|a| a.ty_return)
            }
            Lexeme(LexemeRhs { ref action, .. }) => {
                Some(action.as_ref().map(|a| a.ty_return).unwrap_or(StrSlice))
            }
            _ => { return; }
        };

        let elem = tys.entry(self.name).get().unwrap_or_else(|ve| ve.insert(ty_return));

        if *elem == Some(InferFromRule) {
            *elem = ty_return;
        } else if *elem != ty_return && ty_return != Some(InferFromRule) {
            panic!("cannot infer: expected {:?}, found {:?}", elem, ty_return);
        }
    }

    fn infer_action_ty(&mut self, tys: &HashMap<ast::Name, Option<InlineActionType>>) {
        match self.rhs {
            Sequence(ref mut inner, Some(InlineAction { ref mut ty_return, .. })) => {
                if let Some(&Some(val)) = tys.get(&self.name) {
                    *ty_return = val;
                }

                for rhs in inner.iter_mut() {
                    if let &mut Ident(bind_name, Some(InlineBind { ref mut ty_param, .. })) = rhs {
                        if let Some(&Some(val)) = tys.get(&bind_name) {
                            *ty_param = val;
                        }
                    }
                }
            }
            Ident(bind_name, Some(InlineBind { ref mut ty_param, .. })) => {
                if let Some(&Some(val)) = tys.get(&bind_name) {
                    *ty_param = val;
                }
            }
            _ => {}
        }
    }

    fn alternatives(self) -> Vec<Rule> {
        match self {
            Rule { rhs: Alternative(alt_seq), name } =>
                alt_seq.into_iter().map(|alt| Rule { name: name, rhs: alt }).collect(),
            Rule { rhs, name } =>
                vec![rhs].into_iter().map(|alt| Rule { name: name, rhs: alt }).collect(),
        }
    }
}

struct Context<'a, 'b: 'a> {
    ext: &'a mut ExtCtxt<'b>,
    sp: Span,
    parser: Parser<'a>,
    val_reprs: HashSet<InlineActionType>,
    explicit_tys: HashMap<P<ast::Ty>, u32>,
    syms: StrInterner,
    lexer: Option<(ast::Ident, Vec<TokenTree>)>,
}

impl<'a, 'b: 'a> Context<'a, 'b> {
    fn parse_bound(&mut self) -> Option<P<ast::Pat>> {
        if self.parser.token.is_ident() &&
                !self.parser.token.is_strict_keyword() &&
                !self.parser.token.is_reserved_keyword() &&
                self.parser.look_ahead(1, |t| *t == token::Colon) ||
                self.parser.token == token::OpenDelim(token::Paren) {
            let parenthesized = self.parser.eat(&token::OpenDelim(token::Paren));
            let bound_with = self.parser.parse_pat();
            if parenthesized {
                self.parser.expect(&token::CloseDelim(token::Paren));
            }
            self.parser.expect(&token::Colon);
            Some(bound_with)
        } else {
            None
        }
    }

    fn parse_name_or_repeat(&mut self) -> RuleRhs {
        let mut seq = vec![];
        loop {
            let bound_with = self.parse_bound();
            let elem = if self.parser.token.is_ident() &&
                    !self.parser.token.is_strict_keyword() &&
                    !self.parser.token.is_reserved_keyword() {
                let ident = self.parser.parse_ident();
                let new_name = self.syms.intern(ident.as_str());
                // Give a better type?
                Ident(new_name, bound_with.map(|b| InlineBind::new(b)))
            } else {
                match self.parser.token {
                    token::Literal(token::Str_(_), _) | token::Literal(token::StrRaw(..), _) => {
                        Lexeme(LexemeRhs {
                            tts: vec![self.parser.parse_token_tree()],
                            bind: bound_with.map(|b| InlineBind { pat: b, ty_param: DirectStr }),
                            action: None,
                        })
                    }
                    _ => break
                }
                // match self.parser.parse_optional_str() {
                //     Some((s, _, suf)) => {
                //         let sp = self.parser.last_span;
                //         self.parser.expect_no_suffix(sp, "str literal", suf);
                //         Lexeme(LexemeRhs {
                //             // regstr: s,
                //             tts: 
                //             bind: bound_with.map(|b| InlineBind { pat: b, ty_param: DirectStr }),
                //             action: None,
                //         })
                //     }
                //     _ => break
                // }
            };
            let elem_or_rep = match self.parser.token {
                token::BinOp(token::Star) => {
                    self.parser.bump();
                    Repeat(box elem, ZeroOrMore)
                }
                token::BinOp(token::Plus) => {
                    self.parser.bump();
                    Repeat(box elem, OneOrMore)
                }
                _ => elem,
            };
            seq.push(elem_or_rep);
        }

        let blk = if self.parser.token == token::OpenDelim(token::Brace) ||
                     self.parser.token == token::RArrow {
            let output_ty = if self.parser.eat(&token::RArrow) {
                let ty = self.parser.parse_ty();

                if let ast::TyInfer = ty.node {
                    // -> _ {}
                    InferFromRule
                } else {
                    // -> Foo {}
                    let len = self.explicit_tys.len() as u32;
                    let n_ty = self.explicit_tys.entry(ty).get().unwrap_or_else(|ve| ve.insert(len));

                    Explicit(*n_ty)
                }
            } else {
                self.val_reprs.insert(InferT);
                // {}
                InferT
            };

            let block = self.parser.parse_block();
            Some(InlineAction { ty_return: output_ty, block: block })
        } else {
            None
        };

        return Sequence(seq, blk);
    }

    fn parse_rhs(&mut self) -> RuleRhs {
        let elem = self.parse_name_or_repeat();
        if self.parser.eat(&token::BinOp(token::Or)) {
            // flattened alternative
            self.parse_rhs().alternative(elem)
        } else {
            // complete this rule
            self.parser.expect_one_of(&[token::Semi], &[token::Eof]);
            elem
        }
    }

    fn parse_rule(&mut self) -> Option<Rule> {
        let ident = self.parser.parse_ident();

        if self.parser.eat(&token::Not) {
            let tts = match self.parser.token {
                token::OpenDelim(delim) => {
                    self.parser.bump();

                    // Parse the token trees within the delimiters
                    let tts = self.parser.parse_seq_to_before_end(
                        &token::CloseDelim(delim),
                        seq_sep_none(),
                        |p| p.parse_token_tree()
                    );

                    self.parser.bump();

                    // if tts.is_empty() {
                    //     None
                    // } else {
                    //     Some(tts)
                    // }
                    tts
                }
                _ => {
                    let sp = self.parser.span;
                    self.parser.span_err(sp, "expected macro-like invocation");
                    return None;
                }
            };
            self.lexer = Some((ident, tts));
            self.parser.expect(&token::Semi);
            return None;
        }

        let new_name = self.syms.intern(ident.as_str());

        if self.parser.eat(&token::Tilde) {
            // assertion
            // if let Sequence(seq, blk) = self.parse_rhs() {
            //     if let Some(Lexeme(LexemeRhs { regstr, bind: bound_with, action: None }))
            //             = seq.into_iter().next() {
            //         return Some(Rule {
            //             name: new_name,
            //             rhs: Lexeme(LexemeRhs { regstr: regstr, bind: bound_with, action: blk }),
            //         });
            //     }
            //     // next?
            // }
            let bind = self.parse_bound().map(|b| InlineBind { pat: b, ty_param: DirectStr });
            let mut tts = vec![];
            while self.parser.token != token::Semi
                    && self.parser.token != token::OpenDelim(token::Brace)
                    && self.parser.token != token::RArrow {
                tts.push(self.parser.parse_token_tree());
            }
            if let Sequence(seq, blk) = self.parse_rhs() {
                assert!(seq.is_empty());
                return Some(Rule {
                    name: new_name,
                    rhs: Lexeme(LexemeRhs { tts: tts, bind: bind, action: blk }),
                });
            }

            let sp = self.parser.span;
            self.parser.span_err(sp, "expected a lexical rule");
            None
        } else if self.parser.eat(&token::ModSep) && self.parser.eat(&token::Eq) {
            Some(Rule {
                name: new_name,
                rhs: self.parse_rhs(),
            })
        } else {
            let sp = self.parser.span;
            self.parser.span_err(sp, "expected `::=` or `~`");
            None
        }
    }

    fn parse_rules_to_end(&mut self) -> Vec<Rule> {
        let mut rules = vec![];
        while self.parser.token != token::Eof {
            match self.parse_rule() {
                Some(rule) => rules.push(rule),
                None => ()
            }
        }
        rules
    }

    fn build_conditional(&mut self,
                         arg: Vec<(Option<InlineAction>, Vec<Option<InlineBind>>)>)
                         -> P<ast::Expr> {
        let mut cond_expr = quote_expr!(self.ext, panic!("unknown choice"));

        for (n, (action, bounds)) in arg.into_iter().enumerate().rev() {
            let (bounds_pat, stmts): (Vec<_>, Vec<_>) =
            bounds.into_iter().enumerate().map(|(n, bind_opt)| {
                if let Some(bind) = bind_opt {
                    bind.pat_match(self, n)
                } else {
                    (self.ext.pat_wild(self.sp), None)
                }
            }).unzip();

            let (mut exprs, mut pats): (Vec<_>, Vec<_>) = stmts.into_iter().filter_map(|t| t).unzip();

            let arg_pat = self.ext.pat(self.sp, ast::PatVec(
                bounds_pat,
                None,
                vec![]
            ));

            pats.push(arg_pat);

            let tup_pat = self.ext.pat_tuple(self.sp, pats);

            exprs.push(quote_expr!(self.ext, args));

            let tup_arg = self.ext.expr_tuple(self.sp, exprs);

            let action_expr = if let Some(a) = action {
                a.expr_wrap(self)
            } else {
                quote_expr!(self.ext, SlifRepr::ValLexed(arg_))
            };

            cond_expr = self.ext.expr_if(self.sp, quote_expr!(self.ext, choice_ == $n),
                                       quote_expr!(self.ext, {
                                            // $stmts
                                            match $tup_arg {
                                                $tup_pat => { $action_expr },
                                                _ => unreachable!()
                                            }
                                       }),
                                       Some(cond_expr));
        }

        cond_expr
    }
}

fn expand_grammar(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree])
                  -> Box<MacResult+'static> {
    let mut parser = parse::new_parser_from_tts(cx.parse_sess(),
                                                cx.cfg(), tts.to_vec());

    let mut ctxt = Context {
        ext: &mut *cx,
        sp: sp,
        parser: parser,
        val_reprs: HashSet::new(),
        explicit_tys: HashMap::new(),
        syms: StrInterner::new(),
        lexer: None,
    };

    let mut g1_rules = ctxt.parse_rules_to_end();

    debug!("Parsed: {:?}", g1_rules.iter().fold(String::new(), |s, r| format!("{}\n{:?}", s, r)));

    // prepare grammar rules

    // pass 1: extract sub-rules
    let mut g1_extracted_rules = vec![];
    for rule in g1_rules.iter_mut() {
        rule.extract_sub_rules(&mut ctxt, &mut g1_extracted_rules);
    }
    g1_rules.extend(g1_extracted_rules.into_iter());

    // pass 2: extract l0
    let mut l0_rules = vec![];
    let mut l0_discard_rules = vec![];

    let mut g1_rules = g1_rules.into_iter().filter_map(|rule| {
        match rule {
            Rule { rhs: Lexeme(rhs), name } => {
                if &*ctxt.syms.get(name) == "discard" {
                    l0_discard_rules.push(rhs.tts);
                } else {
                    ctxt.val_reprs.insert(StrSlice);
                    l0_rules.push(Rule { rhs: Lexeme(rhs), name: name });
                }

                None
            }
            other => Some(other)
        }
    }).flat_map(|rule| {
        // pass 3: flatten alternatives
        rule.alternatives().into_iter()
    }).collect::<Vec<_>>();

    debug!("For lexing: {}", l0_rules.iter().fold(String::new(), |s, r| format!("{}\n{:?}", s, r)));

    // pass 4: sequences
    let mut g1_seq_rules = vec![];
    for rule in g1_rules.iter_mut() {
        rule.create_seq_rules(&mut ctxt.syms, &mut g1_seq_rules);
    }
    debug!("Sequence rules: {}", g1_seq_rules.iter().fold(String::new(), |s, r| format!("{}\n{:?}", s, r)));

    let match_ty_return_val;

    let ty_return = match &g1_rules[0].rhs {
        &Sequence(_, ref action) => {
            let ty_return = action.as_ref().map(|a| a.ty_return).unwrap_or(InferT);
            // if ty_return

            let pat_val = ctxt.ext.pat_ident(ctxt.sp, token::str_to_ident("val"));
            match_ty_return_val = ty_return.pat_match(&mut ctxt, pat_val);

            match ty_return {
                Explicit(n) => ctxt.explicit_tys.iter().find(|&(ty, v)| *v == n).map(|(ty, v)| ty.clone()).unwrap(),
                StrSlice => quote_ty!(ctxt.ext, &'a str),
                InferT => quote_ty!(ctxt.ext, T),
                _ => unreachable!()
            }
        }
        _ => { unreachable!() }
    };

    let mut rule_alt_tys = HashMap::new();

    for rule in g1_rules.iter().chain(g1_seq_rules.iter()).chain(l0_rules.iter()) {
        rule.join_action_ty(&mut rule_alt_tys);
    }

    // LEXER -- L0
    // Lexer regexstr

    let mut l0_inline_actions = vec![];
    let mut l0_names = vec![];
    let (scan_id_exprs, reg_alts): (Vec<P<ast::Expr>>, Vec<Vec<ast::TokenTree>>) =
    l0_rules.into_iter().map(|rule| {
        if let Rule { rhs: Lexeme(mut lex), name } = rule {
            l0_inline_actions.push((lex.action.take(), vec![lex.bind.take()]));

            l0_names.push(ctxt.syms.get(name));
            // let regstr = &lex.regstr.to_string()[];
            // let regstr = ctxt.ext.expr_lit(sp, ast::LitStr(token::intern_and_get_ident(regstr), ast::RawStr(2)));
            let tts = lex.tts;
            (ctxt.ext.expr_usize(sp, name.usize()), quote_tokens!(ctxt.ext, $tts,))
        } else {
            unreachable!()
        }
    }).unzip();

    let l0_names: Vec<P<ast::Expr>> = l0_names.into_iter().map(|rcstr| {
        let s = &rcstr[..];
        ctxt.ext.expr_str(sp, token::intern_and_get_ident(s))
    }).collect();
    let l0_names = ctxt.ext.expr_vec(sp, l0_names);

    let num_scan_syms = scan_id_exprs.len();
    // ``` [$l0_sym_id0, $l0_sym_id1, ...] ```
    let scan_id_ary = ctxt.ext.expr_vec(sp, scan_id_exprs);
    // let discard_rule = ctxt.ext.expr_lit(sp, ast::LitStr(token::intern_and_get_ident(&l0_discard_rules[0][]), ast::RawStr(2)));
    let discard_rule = l0_discard_rules;

    let num_syms = ctxt.syms.len();

    // ``` if tok_kind == 0 { $block } else if ... ```
    let lex_cond_expr = ctxt.build_conditional(l0_inline_actions);

    let (lexer, lexer_opt) =
        ctxt.lexer.take().unwrap_or_else(|| (token::str_to_ident("regex_scanner"), vec![]));

    // RULES -- G1
    let mut rhs_exprs = vec![];

    for rule in g1_rules.iter_mut().chain(g1_seq_rules.iter_mut()) {
        rule.infer_action_ty(&rule_alt_tys);
    }

    debug!("G1 rules inferred: {}", g1_rules.iter().fold(String::new(), |s, r| format!("{}\n{:?}", s, r)));
    debug!("Sequence rules inferred: {}", g1_seq_rules.iter().fold(String::new(), |s, r| format!("{}\n{:?}", s, r)));

    let (rule_exprs, rule_actions): (Vec<_>, Vec<_>) =
    g1_rules.into_iter().chain(g1_seq_rules.into_iter()).map(|rule| {
        let rblk = match rule.rhs {
            Sequence(seq, action) => {
                let mut bounds = vec![];
                rhs_exprs.extend(seq.into_iter().map(|node|
                    match node {
                        Ident(name, bind) => {
                            bounds.push(bind);
                            let id = name.usize();
                            quote_expr!(ctxt.ext, syms[$id])
                        }
                        _ => panic!("not an ident in a sequence")
                    }
                ));
                (action, bounds)
            }
            Ident(rhs_sym, bound_with) => {
                let rhs_id = rhs_sym.usize();
                rhs_exprs.push(quote_expr!(ctxt.ext, syms[$rhs_id]));

                (Some(InlineAction {
                    block: ctxt.ext.block_expr(quote_expr!(ctxt.ext, ())),
                    ty_return: Continue,
                 }),
                 vec![bound_with])
            }
            // &Lexeme(ref s)
            _ => panic!("not an ident or seq or lexeme")
        };

        let lhs = rule.name.usize();

        return (
            ctxt.ext.expr_tuple(sp, vec![
                quote_expr!(ctxt.ext, syms[$lhs]),
                ctxt.ext.expr_vec_slice(sp, mem::replace(&mut rhs_exprs, vec![])),
            ]),
            rblk
        );
    }).unzip();

    // ``` [$rule_exprs] ```
    let num_rules = rule_exprs.len();
    let rules_expr = ctxt.ext.expr_vec(sp, rule_exprs);

    // ``` if rule == self.rule_ids[$x] { $block } ```
    let rule_cond_expr = ctxt.build_conditional(rule_actions);

    // Generated code

    debug!("Reprs {:?}, tys:", ctxt.val_reprs);
    for (ty, n_ty) in ctxt.explicit_tys.iter() {
        debug!("{} => {}", n_ty, pprust::ty_to_string(&**ty));
    }

    let mut variants = ctxt.explicit_tys.iter().map(|(ty, n_ty)| {
        let n_ty_s = format!("Spec{}", n_ty);
        let variant_name = token::str_to_ident(&n_ty_s[..]);
        quote_tokens!(ctxt.ext, $variant_name($ty),)
    }).collect::<Vec<_>>();

    let mut T = None;

    if ctxt.val_reprs.contains(&InferT) {
        variants.push(quote_tokens!(ctxt.ext, ValInfer(T),));
        T = Some(quote_tokens!(ctxt.ext, T,));
    }

    if ctxt.val_reprs.contains(&StrSlice) {
        variants.push(quote_tokens!(ctxt.ext, ValLexed(lexer::$lexer::Output<'a>),));
    }

    let fn_parse_expr = quote_expr!(ctxt.ext, {
        let mut recce = ::marpa::Recognizer::new(&mut self.grammar).unwrap();
        recce.start_input();

        let mut lex_parse = self.lexer.new_parse(input);

        let mut positions = vec![];
        let l0_names = $l0_names;

        let mut ith = 0;

        while !lex_parse.is_empty() {
            let mut syms: [Symbol; $num_scan_syms] = unsafe { mem::uninitialized() };
            let mut terminals_expected: [u32; $num_scan_syms] = unsafe { mem::uninitialized() };
            let terminal_ids_expected = unsafe {
                recce.terminals_expected(&mut syms[])
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
                let expected: Vec<&str> = terminals_expected.iter().map(|&i| l0_names[i as usize]).collect();
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

        // println!("{:?}", positions.len());

        let latest_es = recce.latest_earley_set();

        let mut tree =
            ::marpa::Bocage::new(&mut recce, latest_es)
         .and_then(|mut bocage|
            ::marpa::Order::new(&mut bocage)
        ).and_then(|mut order|
            ::marpa::Tree::new(&mut order)
        );

        SlifParse {
            tree: tree.unwrap(),
            lex_parse: lex_parse,
            positions: positions,
            stack: vec![],
            parent: self,
        }
    });

    let slif_iter_next_expr = quote_expr!(ctxt.ext, {
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
                    let elem = self.parent.lex_closure.call_mut(self.lex_parse.get(&self.positions[], idx));
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
                // Step::StepNullingSymbol => {
                //     // let rule = valuator.rule();
                //     // let elem = {
                //     //     let slice: &mut [SlifRepr] = &mut [];
                //     //     let choice = self.parent.rule_ids.iter().position(|r| *r == rule);
                //     //     self.parent.parse_closure.call_mut((slice, choice.expect("unknown rule")))
                //     // };
                //     // match elem {
                //     //     SlifRepr::Continue => {
                //     //         continue
                //     //     }
                //     //     other_elem => {
                //     //         self.stack_put(valuator.result() as usize, other_elem);
                //     //     }
                //     // }
                //     break;
                // }
                Step::StepInactive => {
                    break;
                }
                other => panic!("unexpected step {:?}", other),
            }
        }

        // println!("{:?}", self.stack.capacity());

        let result = self.stack.drain().next();

        match result {
            Some($match_ty_return_val) => Some(val),
            _ => {
                None
            }
        }
    });

    let slif_expr = quote_expr!(ctxt.ext, {
        use marpa::{Tree, Symbol, Value, Step, ErrorCode};
        use marpa::marpa::Values;
        use std::mem;
        // pub use self::*;

        mod lexer {
            // pub use self::lexer_::$lexer::*;
            // mod lexer_ {
                // pub use super::super::Token2;
                pub use super::*;
                $lexer!($lexer_opt ; $discard_rule, $reg_alts);
            // }
        }

        struct SlifGrammar<C, D, $T> {
            grammar: ::marpa::Grammar,
            scan_syms: [Symbol; $num_scan_syms],
            rule_ids: [marpa::Rule; $num_rules],
            lexer: lexer::$lexer::Lexer,
            lex_closure: C,
            parse_closure: D,
        }

        struct SlifParse<'a, 'b, C: 'b, D: 'b, $T> {
            tree: Tree,
            lex_parse: lexer::$lexer::LexerParse<'a, 'b>,
            positions: Vec<lexer::$lexer::Token>,
            stack: Vec<SlifRepr<'a, $T>>,
            parent: &'b mut SlifGrammar<C, D, $T>,
        }

        enum SlifRepr<'a, $T> {
            Continue, // also a placeholder
            $variants
        }

        impl<C, D, $T> SlifGrammar<C, D, $T>
                where C: for<'c> FnMut(lexer::$lexer::Output<'c>, usize) -> SlifRepr<'c, $T>,
                      D: for<'c> FnMut(&mut [SlifRepr<'c, $T>], usize) -> SlifRepr<'c, $T>,
        {
            #[inline]
            fn new(closures: (C, D)) -> SlifGrammar<C, D, $T> {
                let mut cfg = ::marpa::Config::new();
                let mut grammar = ::marpa::Grammar::with_config(&mut cfg).unwrap();

                let mut syms: [::marpa::Symbol; $num_syms] = unsafe { ::std::mem::uninitialized() };
                for sym in syms.iter_mut() { *sym = grammar.symbol_new().unwrap(); }
                grammar.start_symbol_set(syms[0]);

                let rules: [(::marpa::Symbol, &[::marpa::Symbol]); $num_rules] = $rules_expr;

                let scan_id_ary = $scan_id_ary;
                let mut scan_syms: [Symbol; $num_scan_syms] = unsafe { ::std::mem::uninitialized() };
                for (sym, idx) in scan_syms.iter_mut().zip(scan_id_ary.iter()) {
                    *sym = syms[*idx];
                }

                let mut rule_ids: [::marpa::marpa::Rule; $num_rules] = unsafe { ::std::mem::uninitialized() };
                for (&(lhs, rhs), id) in rules.iter().zip(rule_ids.iter_mut()) {
                    *id = grammar.rule_new(lhs, rhs).unwrap();
                }
                grammar.precompute().unwrap();

                SlifGrammar {
                    grammar: grammar,
                    scan_syms: scan_syms,
                    rule_ids: rule_ids,
                    lexer: lexer::$lexer::Lexer::new(),
                    lex_closure: closures.0,
                    parse_closure: closures.1,
                }
            }

            #[inline]
            fn parses_iter<'a, 'b>(&'b mut self, input: lexer::$lexer::Input<'a>) -> SlifParse<'a, 'b, C, D, $T> {
                $fn_parse_expr
            }
        }

        impl<'a, 'b, C, D, $T> Iterator for SlifParse<'a, 'b, C, D, $T>
                where C: for<'c> FnMut(lexer::$lexer::Output<'c>, usize) -> SlifRepr<'c, $T>,
                      D: for<'c> FnMut(&mut [SlifRepr<'c, $T>], usize) -> SlifRepr<'c, $T>,
        {
            type Item = $ty_return;

            fn next(&mut self) -> Option<$ty_return> {
                $slif_iter_next_expr
            }
        }

        impl<'a, 'b, C, D, $T> SlifParse<'a, 'b, C, D, $T> {
            fn stack_put(&mut self, idx: usize, elem: SlifRepr<'a, $T>) {
                if idx == self.stack.len() {
                    self.stack.push(elem);
                } else {
                    self.stack[idx] = elem;
                }
            }
        }

        SlifGrammar::new((
            |arg_, choice_| {
                let args: &[_] = &[arg_];
                $lex_cond_expr
            },
            |args, choice_| {
                $rule_cond_expr
            },
        ))
    });

    debug!("{}", pprust::expr_to_string(&*slif_expr));

    MacEager::expr(slif_expr)
}

// Debugging

impl fmt::Debug for InlineAction {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        try!(fmt.write_fmt(format_args!("InlineAction {{ block: {}, ty_return: ",
                           pprust::block_to_string(&*self.block))));
        try!(self.ty_return.fmt(fmt));
        try!(fmt.write_str(" }}"));
        Ok(())
    }
}
