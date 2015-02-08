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
    regstr: token::InternedString,
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
                quote_pat!(cx.ext, SlifRepr::ValStrSlice($pat)),
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
                quote_expr!(cx.ext, SlifRepr::ValStrSlice($block)),
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
}

impl<'a, 'b: 'a> Context<'a, 'b> {
    fn parse_name_or_repeat(&mut self) -> RuleRhs {
        let mut seq = vec![];
        loop {
            let bound_with = if self.parser.token.is_ident() &&
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
            };
            let elem = if self.parser.token.is_ident() &&
                    !self.parser.token.is_strict_keyword() &&
                    !self.parser.token.is_reserved_keyword() {
                let ident = self.parser.parse_ident();
                let new_name = self.syms.intern(ident.as_str());
                // Give a better type?
                Ident(new_name, bound_with.map(|b| InlineBind::new(b)))
            } else {
                match self.parser.parse_optional_str() {
                    Some((s, _, suf)) => {
                        let sp = self.parser.last_span;
                        self.parser.expect_no_suffix(sp, "str literal", suf);
                        Lexeme(LexemeRhs {
                            regstr: s,
                            bind: bound_with.map(|b| InlineBind { pat: b, ty_param: DirectStr }),
                            action: None,
                        })
                    }
                    _ => break
                }
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
        let new_name = self.syms.intern(ident.as_str());

        if self.parser.eat(&token::Tilde) {
            // assertion
            if let Sequence(seq, blk) = self.parse_rhs() {
                if let Some(Lexeme(LexemeRhs { regstr, bind: bound_with, action: None }))
                        = seq.into_iter().next() {
                    return Some(Rule {
                        name: new_name,
                        rhs: Lexeme(LexemeRhs { regstr: regstr, bind: bound_with, action: blk }),
                    });
                }
                // next?
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
                quote_expr!(self.ext, SlifRepr::ValStrSlice(arg_))
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
    };

    // let start_sym = ctxt.syms.gensym(":start");

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
                    l0_discard_rules.push(rhs.regstr.to_string());
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

    // rule :start
    // let start_rule = Rule::ident(start_sym, implicit_start);
    // g1_rules.push(start_rule);

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

    let mut reg_dedup: HashMap<String, Rule> = HashMap::new();
    let mut reg_dups = HashMap::new();

    // LEXER -- L0
    // Lexer regexstr
    for rule in l0_rules.into_iter() {
        let regstr = if let Rule { rhs: Lexeme(LexemeRhs { ref regstr, .. }), name } = rule {
            regstr.to_string()
        } else {
            unreachable!()
        };

        match reg_dedup.entry(regstr).get() {
            Ok(uniq_rule) => {
                // if has action: error
                // redirect
                reg_dups.insert(rule.name, uniq_rule.name);
            },
            Err(ve) => {
                ve.insert(rule);
            }
        }
    }

    let num_scan_syms = reg_dedup.len();

    let scan_syms_exprs = reg_dedup.values().map(|rule| {
        let symid = rule.name.usize();
        quote_expr!(ctxt.ext, syms[$symid])
    }).collect::<Vec<_>>();

    let (reg_alt, l0_rules): (Vec<_>, Vec<_>) = reg_dedup.into_iter().unzip();

    l0_discard_rules.push(format!("({})", reg_alt.connect(")|(")));
    let reg_alt_s = l0_discard_rules.connect("|");
    let reg_alt = reg_alt_s.as_slice();
    debug!("Lexer regexstr: {}", reg_alt_s);

    let num_syms = ctxt.syms.len();

    // ``` [syms[$l0_sym_id0], syms[$l0_sym_id1], ...] ```
    let scan_syms_expr = ctxt.ext.expr_vec(sp, scan_syms_exprs);

    let mut l0_inline_actions = vec![];

    let l0_rules = l0_rules.into_iter().map(|rule| {
        if let Rule { rhs: Lexeme(mut lex), name } = rule {
            l0_inline_actions.push((
                lex.action.take(),
                vec![lex.bind.take()]
            ));

            lex
        } else {
            unreachable!()
        }
    }).collect::<Vec<_>>();

    // ``` if tok_kind == 0 { $block } ... ```
    let lex_cond_expr = ctxt.build_conditional(l0_inline_actions);

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
                            let id = if let Some(replacement) = reg_dups.get(&name) {
                                replacement.usize()
                            } else {
                                name.usize()
                            };
                            quote_expr!(ctxt.ext, syms[$id])
                        }
                        _ => panic!("not an ident in a sequence")
                    }
                ));
                (action, bounds)
            }
            Ident(rhs_sym, bound_with) => {
                let rhs_id = if let Some(replacement) = reg_dups.get(&rhs_sym) {
                    replacement.usize()
                } else {
                    rhs_sym.usize()
                };
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

    let fn_new_expr = quote_expr!(ctxt.ext, {
        let mut cfg = ::marpa::Config::new();
        let mut grammar = ::marpa::Grammar::with_config(&mut cfg).unwrap();
        let mut syms: [::marpa::Symbol; $num_syms] = unsafe { ::std::mem::uninitialized() };
        for sym in syms.iter_mut() { *sym = grammar.symbol_new().unwrap(); }
        grammar.start_symbol_set(syms[0]);
        let rules: [(::marpa::Symbol, &[::marpa::Symbol]); $num_rules] = $rules_expr;
        let scan_syms = $scan_syms_expr;
        let mut rule_ids: [::marpa::Rule; $num_rules] = unsafe { ::std::mem::uninitialized() };
        for (&(lhs, rhs), id) in rules.iter().zip(rule_ids.iter_mut()) {
            *id = grammar.rule_new(lhs, rhs).unwrap();
        }
        grammar.precompute().unwrap();

        let scanner = regex!($reg_alt);
        (grammar, scan_syms, rule_ids, scanner)
    });

    let fn_parse_expr = quote_expr!(ctxt.ext, {
        let mut recce = ::marpa::Recognizer::new(&mut self.grammar).unwrap();
        recce.start_input();

        let ns = precise_time_ns();

        let mut positions = vec![];

        for capture in self.scanner.captures_iter(input) {
            let pos = match capture.iter().skip(1).position(|s| !s.is_empty()) {
                Some(pos) => pos,
                _ => continue
            };
            let (start_pos, end_pos) = capture.pos(pos + 1).expect("pos panicked");
            positions.push((start_pos as u32, end_pos as u32, pos));
        }

        println!("regex: elapsed {}", (precise_time_ns() - ns) as f64 / 1_000_000_000f64);

        let ns = precise_time_ns();

        for (nth, &(start_pos, end_pos, pos)) in positions.iter().enumerate() {
            recce.alternative(self.scan_syms[pos], nth as i32 + 1, 1); // 0 is reserved
            recce.earleme_complete();
        }

        let latest_es = recce.latest_earley_set();

        let mut tree = ::marpa::Bocage::new(&mut recce, latest_es).and_then(|mut bocage|
            ::marpa::Order::new(&mut bocage)
        ).and_then(|mut order|
            ::marpa::Tree::new(&mut order)
        );

        println!("marpa: elapsed {}", (precise_time_ns() - ns) as f64 / 1_000_000_000f64);

        SlifParse {
            tree: tree.unwrap(),
            positions: positions,
            input: input,
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
            let (i, el) = match valuator.step() {
                Step::StepToken => {
                    let idx = valuator.token_value() as usize - 1;
                    // let (end_pos, tok_kind) = if start_pos == 0 {
                    //     self.positions[0]
                    // } else {
                        // println!("{:?} in {:?}", start_pos, self.positions);
                        // let end_idx = self.positions.binary_search_by(|&(el, _)| el.cmp(&start_pos))
                        //                             .unwrap_or_else(|x| x - 1) + 1;
                    //     self.positions[end_idx]
                    // };
                    let (start_pos, end_pos, tok_kind) = self.positions[idx];
                    (valuator.result() as usize,
                     self.parent.lex_closure.call((self.input.slice(start_pos as usize,
                                                                    end_pos as usize),
                                                   tok_kind)))
                }
                Step::StepRule => {
                    let rule = valuator.rule();
                    let arg_0 = valuator.arg_0() as usize;
                    let arg_n = valuator.arg_n() as usize;
                    let slice = self.stack.slice_mut(arg_0, arg_n + 1);
                    let choice = self.parent.rule_ids.iter().position(|r| *r == rule);
                    let elem = self.parent.rules_closure.call((slice,
                                                               choice.expect("unknown rule")));
                    match elem {
                        SlifRepr::Continue => continue,
                        other_elem => (arg_0, other_elem),
                    }
                }
                Step::StepInactive | Step::StepNullingSymbol => {
                    break;
                }
                other => panic!("unexpected step {:?}", other),
            };

            if i == self.stack.len() {
                self.stack.push(el);
            } else {
                self.stack[i] = el;
            }
        }

        let result = self.stack.swap_remove(0);
        self.stack.clear();
        match result {
            $match_ty_return_val => Some(val),
            _ => None
        }
    });

    debug!("Reprs {:?}, tys:", ctxt.val_reprs);
    for (ty, n_ty) in ctxt.explicit_tys.iter() {
        debug!("{} => {}", n_ty, pprust::ty_to_string(&**ty));
    }

    let mut variants = ctxt.explicit_tys.iter().map(|(ty, n_ty)| {
        let n_ty_s = format!("Spec{}", n_ty);
        let variant_name = token::str_to_ident(n_ty_s.as_slice());
        quote_tokens!(ctxt.ext, $variant_name($ty),)
    }).collect::<Vec<_>>();

    let mut T = None;

    if ctxt.val_reprs.contains(&InferT) {
        variants.push(quote_tokens!(ctxt.ext, ValInfer(T),));
        T = Some(quote_tokens!(ctxt.ext, T,));
    }

    if ctxt.val_reprs.contains(&StrSlice) {
        variants.push(quote_tokens!(ctxt.ext, ValStrSlice(&'a str),));
    }

    let repr_enum = quote_item!(ctxt.ext,
        enum SlifRepr<'a, $T> {
            Continue, // also a placeholder
            $variants
        }
    );

    // let test_pat = quote_pat!(ctxt.ext, foo);
    // let test_pat = quote_block!(ctxt.ext, foo);

    let slif_expr = quote_expr!(ctxt.ext, {
        use marpa::{Tree, Rule, Symbol, Value, Step};
        use marpa::marpa::Values;
        use std::mem;

        struct SlifGrammar<'a, C, D, $T> {
            grammar: ::marpa::Grammar,
            scan_syms: [Symbol; $num_scan_syms],
            rule_ids: [Rule; $num_rules],
            scanner: ::regex::Regex,
            rules_closure: C,
            lex_closure: D,
        }

        struct SlifParse<'a, 'b, 'l, C: 'b, D: 'b, $T> {
            tree: Tree,
            positions: Vec<(u32, u32, usize)>,
            input: &'a str,
            stack: Vec<SlifRepr<'a, $T>>,
            parent: &'b SlifGrammar<'l, C, D, $T>,
        }

        $repr_enum

        impl<'a, C, D, $T> SlifGrammar<'a, C, D, $T>
                where C: for<'c> Fn(&mut [SlifRepr<'c, $T>], usize) -> SlifRepr<'c, $T>,
                      D: for<'c> Fn(&'c str, usize) -> SlifRepr<'c, $T> {
            #[inline]
            fn new(rules_closure: C, lex_closure: D) -> SlifGrammar<'a, C, D, $T> {
                // $test

                let (grammar, ssym, rid, scanner) = $fn_new_expr;
                SlifGrammar {
                    grammar: grammar,
                    scan_syms: ssym,
                    rule_ids: rid,
                    scanner: scanner,
                    rules_closure: rules_closure,
                    lex_closure: lex_closure,
                }
            }
            #[inline]
            fn parses_iter<'b, 'c>(&'c mut self, input: &'b str) -> SlifParse<'b, 'c, 'a, C, D, $T> {
                $fn_parse_expr
            }
        }
        impl<'a, 'b, 'l, C, D, $T> Iterator for SlifParse<'a, 'b, 'l, C, D, $T>
                where C: for<'c> Fn(&mut [SlifRepr<'c, $T>], usize) -> SlifRepr<'c, $T>,
                      D: for<'c> Fn(&'c str, usize) -> SlifRepr<'c, $T> {
            type Item = $ty_return;

            fn next(&mut self) -> Option<$ty_return> {
                $slif_iter_next_expr
            }
        }
        SlifGrammar::new(
            |args, choice_| {
                $rule_cond_expr
            },
            |arg_, choice_| {
                let args: &[_] = &[arg_];
                $lex_cond_expr
            }
        )
    });

    debug!("{}", pprust::expr_to_string(&*slif_expr));

    MacExpr::new(slif_expr)
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
