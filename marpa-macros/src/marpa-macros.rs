#![crate_name = "marpa-macros"]

#![feature(plugin_registrar, quote, globs, macro_rules)]

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

#[derive(Copy, Clone, Show, Hash, PartialEq, Eq)]
enum InlineActionType {
    // Unknown,
    InferT,
    StrSlice,
    Continue,
    DirectStr,
}

#[derive(Clone)]
struct InlineAction {
    block: P<ast::Block>,
    ty_return: InlineActionType,
}

#[derive(Clone, Show)]
struct InlineBind {
    name: ast::Ident,
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
    fn new(name: ast::Ident) -> InlineBind {
        InlineBind {
            name: name,
            ty_param: InferT
        }
    }
}

#[derive(Copy, Clone, Show)]
pub enum KleeneOp {
    ZeroOrMore,
    OneOrMore,
}

#[derive(Clone, Show)]
struct LexemeRhs {
    regstr: token::InternedString,
    bind: Option<InlineBind>,
    action: Option<InlineAction>,
}

#[derive(Clone, Show)]
enum RuleRhs {
    Alternative(Vec<RuleRhs>),
    Sequence(Vec<RuleRhs>, Option<InlineAction>),
    Ident(ast::Name, Option<InlineBind>),
    Repeat(Box<RuleRhs>, KleeneOp),
    Lexeme(LexemeRhs),
}

#[derive(Show)]
struct Rule {
    name: ast::Name,
    rhs: RuleRhs,
}

impl RuleRhs {
    fn extract_sub_rules(&mut self, cx: &mut ExtCtxt, syms: &mut StrInterner,
                                                      reprs: &mut HashSet<InlineActionType>,
                                                      cont: &mut Vec<Rule>,
                                                      top: bool) {
        match self {
            &Lexeme(..) if !top => {
                // displace the lexical rule
                let new_sym = syms.gensym("");
                let ty_pass = StrSlice;
                reprs.insert(ty_pass);
                let new_bind;

                match self {
                    &Lexeme(ref mut lex) => {
                        lex.action = Some(InlineAction {
                            block: cx.block_expr(quote_expr!(cx, arg_)),
                            ty_return: ty_pass,
                        });
                        new_bind = lex.bind.as_ref().map(|b| InlineBind { name: b.name, ty_param: ty_pass });
                        lex.bind = None;
                    }
                    _ => unreachable!()
                }

                let this = Ident(new_sym, new_bind);
                let displaced_lex = mem::replace(self, this);

                cont.push(rule!(new_sym ::= displaced_lex));
            }
            &Repeat(box ref mut sub_rule, _) => match sub_rule {
                &Ident(..) => {},
                other_sub_rule => {
                    other_sub_rule.extract_sub_rules(cx, syms, reprs, cont, false);
                    let new_sym = syms.gensym("");
                    let other = mem::replace(other_sub_rule, Ident(new_sym, None)); // None??
                    cont.push(rule!(new_sym ::= other));
                }
            },
            &Sequence(ref mut seq, _) | &Alternative(ref mut seq) => {
                for rule in seq.iter_mut() {
                    rule.extract_sub_rules(cx, syms, reprs, cont, false);
                }
            }
            _ => {}
        }
    }

    fn create_seq_rules(&mut self, syms: &mut StrInterner, cont: &mut Vec<Rule>) {
        match self {
            &Sequence(ref mut seq, _) | &Alternative(ref mut seq) => {
                for rule in seq.iter_mut() {
                    rule.create_seq_rules(syms, cont);
                }
            }
            &Repeat(box ref mut repeated, op) => {
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
            &Ident(..) | &Lexeme(..) => {}
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

    fn extract_sub_rules(&mut self, cx: &mut ExtCtxt, syms: &mut StrInterner,
                                                      reprs: &mut HashSet<InlineActionType>,
                                                      cont: &mut Vec<Rule>) {
        self.rhs.extract_sub_rules(cx, syms, reprs, cont, true);
    }

    fn create_seq_rules(&mut self, syms: &mut StrInterner, cont: &mut Vec<Rule>) {
        self.rhs.create_seq_rules(syms, cont);
    }

    fn alternatives_iter(self) -> Vec<Rule> {
        match self {
            Rule { rhs: Alternative(alt_seq), name } =>
                alt_seq.into_iter().map(|alt| Rule { name: name, rhs: alt }).collect(),
            Rule { rhs, name } =>
                vec![rhs].into_iter().map(|alt| Rule { name: name, rhs: alt }).collect(),
        }
    }
}

fn parse_name_or_repeat(parser: &mut Parser, syms: &mut StrInterner) -> RuleRhs {
    let mut seq = vec![];
    loop {
        let bound_with = if parser.token.is_ident() &&
                !parser.token.is_strict_keyword() &&
                !parser.token.is_reserved_keyword() &&
                parser.look_ahead(1, |t| *t == token::Colon) {
            let bound_with = parser.parse_ident();
            parser.expect(&token::Colon);
            Some(bound_with)
        } else {
            None
        };
        let elem = if parser.token.is_ident() &&
                !parser.token.is_strict_keyword() &&
                !parser.token.is_reserved_keyword() {
            let ident = parser.parse_ident();
            let new_name = syms.intern(ident.as_str());
            // Give a better type?
            Ident(new_name, bound_with.map(|b| InlineBind::new(b)))
        } else {
            match parser.parse_optional_str() {
                Some((s, _, suf)) => {
                    let sp = parser.last_span;
                    parser.expect_no_suffix(sp, "str literal", suf);
                    Lexeme(LexemeRhs {
                        regstr: s,
                        bind: bound_with.map(|b| InlineBind { name: b, ty_param: DirectStr }),
                        action: None,
                    })
                }
                _ => break
            }
        };
        let elem_or_rep = match parser.token {
            token::BinOp(token::Star) => {
                parser.bump();
                Repeat(box elem, ZeroOrMore)
            }
            token::BinOp(token::Plus) => {
                parser.bump();
                Repeat(box elem, OneOrMore)
            }
            _ => elem,
        };
        seq.push(elem_or_rep);
    }

    let blk = if parser.token == token::OpenDelim(token::Brace) {
        Some(InlineAction::new(parser.parse_block()))
    } else {
        None
    };

    return Sequence(seq, blk);
}

fn parse_rhs(parser: &mut Parser, syms: &mut StrInterner) -> RuleRhs {
    let elem = parse_name_or_repeat(parser, syms);
    if parser.eat(&token::BinOp(token::Or)) {
        // flattened alternative
        parse_rhs(parser, syms).alternative(elem)
    } else {
        // complete this rule
        parser.expect_one_of(&[token::Semi], &[token::Eof]);
        elem
    }
}

fn parse_rule(parser: &mut Parser, syms: &mut StrInterner) -> Option<Rule> {
    let ident = parser.parse_ident();
    let new_name = syms.intern(ident.as_str());

    if parser.eat(&token::Tilde) {
        // assertion
        if let Sequence(seq, blk) = parse_rhs(parser, syms) {
            if let Some(Lexeme(LexemeRhs { regstr, bind: bound_with, action: None }))
                    = seq.into_iter().next() {
                return Some(Rule {
                    name: new_name,
                    rhs: Lexeme(LexemeRhs { regstr: regstr, bind: bound_with, action: blk }),
                });
            }
            // next?
        }

        let sp = parser.span;
        parser.span_err(sp, "expected a lexical rule");
        None

    } else if parser.eat(&token::ModSep) && parser.eat(&token::Eq) {
        Some(Rule {
            name: new_name,
            rhs: parse_rhs(parser, syms),
        })
    } else {
        let sp = parser.span;
        parser.span_err(sp, "expected `::=` or `~`");
        None
    }
}

fn build_conditional(cx: &mut ExtCtxt,
                     sp: Span,
                     arg: Vec<(Option<InlineAction>, Vec<Option<InlineBind>>)>) -> P<ast::Expr> {
    let mut cond_expr = quote_expr!(cx, panic!("unknown choice"));

    for (n, (action, bounds)) in arg.into_iter().enumerate().rev() {
        let arg_pat = cx.pat(sp, ast::PatVec(
            bounds.iter().map(|bind| {
                let pat = bind.as_ref().map(|b| cx.pat_ident(sp, b.name))
                                       .unwrap_or_else(|| cx.pat_wild(sp));

                match bind {
                    &Some(InlineBind { ty_param: StrSlice, .. }) =>
                        quote_pat!(cx, SlifRepr::ValStrSlice($pat)),
                    &Some(InlineBind { ty_param: DirectStr, name }) =>
                        cx.pat_ident(sp, name),
                    &None =>
                        pat,
                    &Some(InlineBind { ty_param: Continue, .. }) =>
                        unreachable!(),
                    _ =>
                        quote_pat!(cx, SlifRepr::ValInfer($pat)),
                }
            }).collect(),
            None,
            vec![]
        ));

        let action_expr = if let Some(a) = action {
            match a {
                InlineAction { ty_return: InferT, block } =>
                    quote_expr!(cx, SlifRepr::ValInfer($block)),
                InlineAction { ty_return: StrSlice, block } =>
                    quote_expr!(cx, SlifRepr::ValStrSlice($block)),
                InlineAction { ty_return: Continue, .. } =>
                    quote_expr!(cx, SlifRepr::Continue),
                InlineAction { ty_return: DirectStr, .. } =>
                    unreachable!(),
            }
        } else {
            quote_expr!(cx, SlifRepr::ValStrSlice(arg_))
        };

        cond_expr = cx.expr_if(sp, quote_expr!(cx, choice_ == $n),
                                   quote_expr!(cx,
                                        match args {
                                            $arg_pat => $action_expr,
                                            _ => unreachable!()
                                        }
                                   ),
                                   Some(cond_expr));
    }

    cond_expr
}

fn expand_grammar(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree])
                  -> Box<MacResult+'static> {
    let mut parser = parse::new_parser_from_tts(cx.parse_sess(),
        cx.cfg(), tts.to_vec());

    let mut g1_rules = vec![];

    let mut g1_syms = StrInterner::new();
    let start_sym = g1_syms.gensym(":start");

    while parser.token != token::Eof {
        match parse_rule(&mut parser, &mut g1_syms) {
            Some(rule) => g1_rules.push(rule),
            None => ()
        }
    }

    debug!("Parsed: {:?}", g1_rules.iter().fold(String::new(), |s, r| format!("{}\n{:?}", s, r)));

    // prepare grammar rules

    let mut val_reprs = HashSet::new();

    // pass 1: extract sub-rules
    let mut g1_extracted_rules = vec![];
    for rule in g1_rules.iter_mut() {
        rule.extract_sub_rules(cx, &mut g1_syms, &mut val_reprs, &mut g1_extracted_rules);
    }
    g1_rules.extend(g1_extracted_rules.into_iter());

    // pass 2: extract l0
    let mut l0_rules = vec![];
    let mut l0_discard_rules = vec![];

    let mut g1_rules = g1_rules.into_iter().filter_map(|rule| {
        match rule {
            Rule { rhs: Lexeme(rhs), name } => {
                if &*g1_syms.get(name) == "discard" {
                    l0_discard_rules.push(rhs.regstr.get().to_string());
                } else {
                    val_reprs.insert(StrSlice);
                    l0_rules.push(Rule { rhs: Lexeme(rhs), name: name });
                }

                None
            }
            other => Some(other)
        }
    }).flat_map(|rule| {
        // pass 3: flatten alternatives
        rule.alternatives_iter().into_iter()
    }).collect::<Vec<_>>();

    debug!("For lexing: {}", l0_rules.iter().fold(String::new(), |s, r| format!("{}\n{:?}", s, r)));

    // rule :start
    let &Rule { name: implicit_start, .. } = &g1_rules[0];
    let start_rule = Rule::ident(start_sym, implicit_start);
    g1_rules.push(start_rule);

    // pass 4: sequences
    let mut g1_seq_rules = vec![];
    for rule in g1_rules.iter_mut() {
        rule.create_seq_rules(&mut g1_syms, &mut g1_seq_rules);
    }
    debug!("Sequence rules: {}", g1_seq_rules.iter().fold(String::new(), |s, r| format!("{}\n{:?}", s, r)));

    let num_syms = g1_syms.len();

    // LEXER -- L0
    let num_scan_syms = l0_rules.len();

    // ``` [syms[$l0_sym_id0], syms[$l0_sym_id1], ...] ```
    let scan_syms_exprs = l0_rules.iter().map(|rule| {
        let symid = rule.name.uint();
        quote_expr!(cx, syms[$symid])
    }).collect::<Vec<_>>();
    let scan_syms_expr = cx.expr_vec(sp, scan_syms_exprs);

    let mut l0_inline_actions = vec![];

    let l0_rules = l0_rules.into_iter().map(|rule| {
        if let Rule { rhs: Lexeme(mut lex), .. } = rule {
            l0_inline_actions.push((
                lex.action.take(),
                vec![lex.bind.take()]
            ));

            lex
        } else {
            unreachable!()
        }
    }).collect::<Vec<_>>();

    // Lexer regexstr
    let reg_alt = l0_rules.iter().map(|rule|
        rule.regstr.get().to_string()
    ).collect::<Vec<_>>();

    l0_discard_rules.push(format!("({})", reg_alt.connect(")|(")));
    let reg_alt_s = l0_discard_rules.connect("|");
    let reg_alt = reg_alt_s.as_slice();

    // ``` if tok_kind == 0 { $block } ... ```
    let lex_cond_expr = build_conditional(cx, sp, l0_inline_actions);

    // RULES -- G1
    let mut rhs_exprs = vec![];

    let (rule_exprs, rule_actions): (Vec<_>, Vec<_>) =
    g1_rules.into_iter().chain(g1_seq_rules.into_iter()).map(|rule| {
        let rblk = match rule.rhs {
            Sequence(seq, action) => {
                let mut bounds = vec![];
                rhs_exprs.extend(seq.into_iter().map(|node|
                    match node {
                        Ident(name, bind) => {
                            bounds.push(bind);
                            let id = name.uint();
                            quote_expr!(cx, syms[$id])
                        }
                        _ => panic!("not an ident in a sequence")
                    }
                ));
                (action, bounds)
            }
            Ident(rhs_sym, bound_with) => {
                let rhs_sym = rhs_sym.uint();
                rhs_exprs.push(quote_expr!(cx, syms[$rhs_sym]));

                (Some(InlineAction {
                    block: cx.block_expr(quote_expr!(cx, ())),
                    ty_return: Continue,
                 }),
                 vec![bound_with])
            }
            // &Lexeme(ref s)
            _ => panic!("not an ident or seq or lexeme")
        };
        let lhs = rule.name.uint();

        return (
            cx.expr_tuple(sp, vec![
                quote_expr!(cx, syms[$lhs]),
                cx.expr_vec_slice(sp, mem::replace(&mut rhs_exprs, vec![])),
            ]),
            rblk
        );
    }).unzip();

    // ``` [$rule_exprs] ```
    let num_rules = rule_exprs.len();
    let rules_expr = cx.expr_vec(sp, rule_exprs);

    // ``` if rule == self.rule_ids[$x] { $block } ```
    let rule_cond_expr = build_conditional(cx, sp, rule_actions);

    // Generated code

    let fn_new_expr = quote_expr!(cx, {
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

    let fn_parse_expr = quote_expr!(cx, {
        let mut recce = ::marpa::Recognizer::new(&mut self.grammar).unwrap();
        recce.start_input();

        let mut positions = vec![];

        for capture in self.scanner.captures_iter(input) {
            let pos = match capture.iter().skip(1).position(|s| !s.is_empty()) {
                Some(pos) => pos,
                _ => continue
            };
            let (start_pos, end_pos) = capture.pos(pos + 1).expect("pos panicked");
            positions.push((end_pos, pos));
            recce.alternative(self.scan_syms[pos], (start_pos + 1) as i32, 1); // 0 is reserved
            recce.earleme_complete();
        }

        let latest_es = recce.latest_earley_set();
        let mut bocage = ::marpa::Bocage::new(&mut recce, latest_es).unwrap();
        let mut order = ::marpa::Order::new(&mut bocage).unwrap();
        let mut tree = ::marpa::Tree::new(&mut order).unwrap();

        SlifParse {
            tree: tree,
            positions: positions,
            input: input,
            stack: vec![],
            parent: self,
        }
    });

    let slif_iter_next_expr = quote_expr!(cx, {
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
                    let start_pos = valuator.token_value() as uint - 1;
                    let (end_pos, tok_kind) = if start_pos == 0 {
                        self.positions[0]
                    } else {
                        let end_idx = self.positions.binary_search_by(|&(el, _)| el.cmp(&start_pos))
                                                    .err()
                                                    .expect("binary search panicked");
                        self.positions[end_idx]
                    };
                    (valuator.result() as uint,
                     self.parent.lex_closure.call((self.input.slice(start_pos, end_pos),
                                                   tok_kind)))
                }
                Step::StepRule => {
                    let rule = valuator.rule();
                    let arg_0 = valuator.arg_0() as uint;
                    let arg_n = valuator.arg_n() as uint;
                    let slice = self.stack.slice(arg_0, arg_n + 1);
                    let choice = self.parent.rule_ids.iter().position(|&mut: r| *r == rule);
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
            SlifRepr::ValInfer(val) => Some(val),
            _ => None
        }
    });

    let repr_enum = if val_reprs.is_empty() {
        quote_item!(cx,
            enum SlifRepr<'a, T> {
                ValInfer(T),
                Continue,
            }
        )
    } else {
        quote_item!(cx,
            enum SlifRepr<'a, T> {
                ValInfer(T),
                ValStrSlice(&'a str),
                Continue,
            }
        )
    };

    let slif_expr = quote_expr!(cx, {
        use marpa::{Tree, Rule, Symbol, Value, Step};
        use marpa::marpa::Values;

        struct SlifGrammar<'a, T, C, D> {
            grammar: ::marpa::Grammar,
            scan_syms: [Symbol; $num_scan_syms],
            rule_ids: [Rule; $num_rules],
            scanner: ::regex::Regex,
            rules_closure: C,
            lex_closure: D,
        }

        struct SlifParse<'a, 'b, 'l, T, C: 'b, D: 'b> {
            tree: Tree,
            positions: Vec<(uint, uint)>,
            input: &'a str,
            stack: Vec<SlifRepr<'a, T>>,
            parent: &'b SlifGrammar<'l, T, C, D>,
        }

        $repr_enum

        impl<'a, T, C, D> SlifGrammar<'a, T, C, D>
                where C: for<'c> Fn(&[SlifRepr<'c, T>], uint) -> SlifRepr<'c, T>,
                      D: for<'c> Fn(&'c str, uint) -> SlifRepr<'c, T> {
            #[inline]
            fn new(rules_closure: C, lex_closure: D) -> SlifGrammar<'a, T, C, D> {
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
            fn parse<'b, 'c>(&'c mut self, input: &'b str) -> SlifParse<'b, 'c, 'a, T, C, D> {
                $fn_parse_expr
            }
        }
        impl<'a, 'b, 'l, T, C, D> Iterator for SlifParse<'a, 'b, 'l, T, C, D>
                where C: for<'c> Fn(&[SlifRepr<'c, T>], uint) -> SlifRepr<'c, T>,
                      D: for<'c> Fn(&'c str, uint) -> SlifRepr<'c, T> {
            type Item = T;

            fn next(&mut self) -> Option<T> {
                $slif_iter_next_expr
            }
        }
        SlifGrammar::new(
            |&: args, choice_| {
                $rule_cond_expr
            },
            |&: arg_, choice_| {
                let args: &[_] = &[arg_];
                $lex_cond_expr
            }
        )
    });

    debug!("{}", pprust::expr_to_string(&*slif_expr));

    MacExpr::new(slif_expr)
}

// Debugging

impl fmt::Show for InlineAction {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        try!(fmt.write_fmt(format_args!("InlineAction {{ block: {}, ty_return: ",
                           pprust::block_to_string(&*self.block))));
        try!(self.ty_return.fmt(fmt));
        try!(fmt.write_str(" }}"));
        Ok(())
    }
}
