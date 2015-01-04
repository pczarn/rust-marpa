#![crate_name = "marpa-macros"]

#![feature(plugin_registrar, quote, globs, macro_rules)]

extern crate rustc;
extern crate syntax;

use self::RuleRhs::*;
use self::KleeneOp::*;
use self::InlineActionType::*;

use syntax::ast;
use syntax::ast::TokenTree;
use syntax::codemap::Span;
// use syntax::ext::base;
use syntax::ext::base::{ExtCtxt, MacResult, MacExpr};
use syntax::parse::token;
use syntax::parse::parser::Parser;
use syntax::parse;
use syntax::util::interner::StrInterner;
use syntax::ptr::P;
use syntax::ext::build::AstBuilder;
use syntax::print::pprust;

use std::collections::HashSet;
use std::iter::DoubleEndedIteratorExt;
use std::mem;
use std::vec;

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

#[deriving(Copy, Clone, Show, Hash, PartialEq, Eq)]
enum InlineActionType {
    Infer,
    StrSlice,
}

#[deriving(Clone, Show)]
struct InlineAction {
    block: P<ast::Block>,
    ty_return: InlineActionType,
}

#[deriving(Clone, Show)]
struct InlineBind {
    name: ast::Ident,
    ty_param: InlineActionType,
}

impl InlineAction {
    fn new(block: P<ast::Block>) -> InlineAction {
        InlineAction {
            block: block,
            ty_return: Infer
        }
    }
}

impl InlineBind {
    fn new(name: ast::Ident) -> InlineBind {
        InlineBind {
            name: name,
            ty_param: Infer
        }
    }
}

#[deriving(Clone, Show)]
enum RuleRhs {
    Alternative(Vec<RuleRhs>),
    Sequence(Vec<RuleRhs>, Option<InlineAction>),
    Ident(ast::Name, Option<InlineBind>),
    Repeat(Box<RuleRhs>, KleeneOp),
    Lexeme(token::InternedString, Option<InlineAction>, Option<InlineBind>),
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
                let mut new_bind = None;

                match self {
                    &Lexeme(_, ref mut action, ref mut bind) => {
                        *action = Some(InlineAction {
                            block: cx.block_expr(quote_expr!(cx, arg)),
                            ty_return: ty_pass,
                        });
                        new_bind = bind.as_ref().map(|b| InlineBind { name: b.name, ty_param: ty_pass });
                    }
                    _ => unreachable!()
                }

                let this = Ident(new_sym, new_bind);
                let displaced_lex = mem::replace(self, this);

                cont.push(rule!(new_sym ::= displaced_lex));
            }
            &Repeat(box ref mut sub_rule, op) => match sub_rule {
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

    fn l0(rules: &[Rule], syms: &StrInterner) -> String {
        let reg_alt = rules.iter().map(|rule|
            match &rule.rhs {
                &Lexeme(ref s, _, _) => {
                    s.get().into_string()
                }
                _ => unreachable!()
            }
        ).collect::<Vec<_>>();

        format!("({})", reg_alt.connect(")|("))
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

#[deriving(Copy, Clone, Show)]
pub enum KleeneOp {
    ZeroOrMore,
    OneOrMore,
}

#[deriving(Show)]
struct Rule {
    name: ast::Name,
    rhs: RuleRhs,
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
            Some(InlineBind::new(bound_with))
        } else {
            None
        };
        let elem = if parser.token.is_ident() &&
                !parser.token.is_strict_keyword() &&
                !parser.token.is_reserved_keyword() {
            let ident = parser.parse_ident();
            let new_name = syms.intern(ident.as_str());
            Ident(new_name, bound_with)
        } else {
            match parser.parse_optional_str() {
                Some((s, style, suf)) => {
                    let sp = parser.last_span;
                    parser.expect_no_suffix(sp, "str literal", suf);
                    Lexeme(s, None, bound_with)
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
            if let Some(Lexeme(s, None, bound_with)) = seq.into_iter().next() {
                return Some(Rule {
                    name: new_name,
                    rhs: Lexeme(s, blk, bound_with),
                });
            }
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
    let mut l0_inline_actions = vec![];

    let mut g1_rules = g1_rules.into_iter().filter_map(|rule| {
        match rule {
            Rule { rhs: Lexeme(s, action, bind), name } => {
                if &*g1_syms.get(name) == "discard" {
                    l0_discard_rules.push(s.get().into_string());
                    return None;
                }
                l0_inline_actions.push((
                    action.clone(),
                    bind
                ));
                l0_rules.push(Rule { rhs: Lexeme(s, action, None), name: name });
                None
            }
            other => Some(other)
        }
    }).collect::<Vec<_>>();

    // pass 3: flatten alternatives
    let mut g1_rules = g1_rules.into_iter().flat_map(|rule| {
        rule.alternatives_iter().into_iter()
    }).collect::<Vec<_>>();

    // rule :start
    let &Rule { name: implicit_start, .. } = &g1_rules[0];
    g1_rules.push(Rule::ident(start_sym, implicit_start));

    let mut g1_seq_rules = vec![];
    for rule in g1_rules.iter_mut() {
        rule.create_seq_rules(&mut g1_syms, &mut g1_seq_rules);
    }

    let num_syms = g1_syms.len();

    // LEXER -- L0
    let num_scan_syms = l0_rules.len();

    let reg_alt_s = Rule::l0(l0_rules.as_slice(), &g1_syms);
    l0_discard_rules.push(reg_alt_s);
    let reg_alt_s = l0_discard_rules.connect("|");
    let reg_alt = reg_alt_s.as_slice();

    // ``` if tok_kind == 0 { $block } ```
    let mut lex_cond_expr = quote_expr!(cx, panic!("unknown rule"));
    for (n, (action, bind)) in l0_inline_actions.into_iter().enumerate().rev() {
        let action_expr = action.map(|a| cx.expr_block(a.block))
                                .unwrap_or(quote_expr!(cx, {7u}));
        let action_expr = quote_expr!(cx, SlifRepr::ValStrSlice($action_expr));
        let arg_pat = cx.pat_ident(sp, bind.map(|b| b.name)
                                           .unwrap_or_else(|| token::gensym_ident("_"))); // str_to_ident
        lex_cond_expr = cx.expr_if(sp, quote_expr!(cx, tok_kind == $n),
                                        quote_expr!(cx, {
                                            let $arg_pat = arg;
                                            $action_expr
                                        }),
                                        Some(lex_cond_expr));
    }

    // ``` let scan_syms = [syms[$l0_sym_id0], syms[$l0_sym_id1], ...]; ```
    let scan_syms_exprs = l0_rules.iter().map(|rule| {
        let symid = rule.name.uint();
        quote_expr!(cx, syms[$symid])
    }).collect::<Vec<_>>();
    let scan_syms_expr = cx.expr_vec(sp, scan_syms_exprs);
    let let_scan_syms_stmt = cx.stmt_let(sp, false, cx.ident_of("scan_syms"), scan_syms_expr);

    // RULES -- G1
    let mut rhs_exprs = vec![];

    let rules =
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
                    block: cx.block_expr(quote_expr!(cx, unsafe { return ::std::ptr::read(&args[0]) })),
                    ty_return: bound_with.as_ref().map(|b| b.ty_param).unwrap_or(Infer),
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
            // quote_expr!(cx, box()(|&: args| $rblk))
            rblk
        );
    });
    let (rule_exprs, rule_actions) = vec::unzip(rules);

    // ``` let rules = &[$rule_exprs]; ```
    let num_rules = rule_exprs.len();
    let rules_expr = cx.expr_vec(sp, rule_exprs);
    let let_rules_stmt =
        cx.stmt_let_typed(sp,
                          false,
                          cx.ident_of("rules"),
                          quote_ty!(cx, [(::marpa::Symbol, &[::marpa::Symbol]); $num_rules]),
                          rules_expr);

    // ``` if rule == self.rule_ids[$x] { $block } ```
    let mut rule_cond_expr = quote_expr!(cx, panic!("unknown rule"));
    for (n, (action, bounds)) in rule_actions.into_iter().enumerate().rev() {
        let action_expr = action.map(|a| {
            match a {
                InlineAction { ty_return: Infer, block } =>
                    quote_expr!(cx, SlifRepr::ValInfer($block)),
                InlineAction { ty_return: StrSlice, block } =>
                    quote_expr!(cx, SlifRepr::ValInfer($block)),
            }
        }).unwrap_or(quote_expr!(cx, SlifRepr::ValInfer(7u)));

        let args_pat = cx.pat(sp, ast::PatVec(
            bounds.into_iter().map(|bind| {
                // bind to an ident or ignore
                let bind_name = bind.as_ref().map(|b| b.name).unwrap_or_else(|| token::gensym_ident("_"));
                let pat = cx.pat_ident(sp, bind_name); // str_to_ident?
                match bind {
                    Some(InlineBind { ty_param: StrSlice, .. }) =>
                        quote_pat!(cx, SlifRepr::ValStrSlice($pat)),
                    // Some(InlineBind { ty_param: Infer, }) | None =>
                    _ => quote_pat!(cx, SlifRepr::ValInfer($pat)),
                }
            }).collect(),
            None,
            vec![]
        ));



        rule_cond_expr = cx.expr_if(sp, quote_expr!(cx, rule == rules[$n]),
                                        quote_expr!(cx,
                                            match args {
                                                $args_pat => $action_expr,
                                                _ => unreachable!()
                                            }
                                        ),
                                        Some(rule_cond_expr));
    }

    // Generated code

    let fn_new_expr = quote_expr!(cx, {
        let mut cfg = ::marpa::Config::new();
        let mut grammar = ::marpa::Grammar::with_config(&mut cfg).unwrap();
        let mut syms: [::marpa::Symbol; $num_syms] = unsafe { ::std::mem::uninitialized() };
        for sym in syms.iter_mut() { *sym = grammar.symbol_new().unwrap(); }
        grammar.start_symbol_set(syms[0]);
        $let_rules_stmt
        $let_scan_syms_stmt
        let mut rule_ids: [::marpa::Rule; $num_rules] = unsafe { ::std::mem::uninitialized() };
        for (&(lhs, rhs), id) in rules.iter().zip(rule_ids.iter_mut()) {
            *id = grammar.rule_new(lhs, rhs).unwrap();
        }
        grammar.precompute();

        let scanner = regex!($reg_alt);
        (grammar, scan_syms, rule_ids, scanner)
    });

    let fn_parse_expr = quote_expr!(cx, {
        let mut recce = ::marpa::Recognizer::new(&mut self.grammar).unwrap();
        recce.start_input();

        // let mut stack: Vec<T> = vec![];

        // let scanner = regex!($reg_alt);
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
                        self.positions[self.positions.binary_search_by(|&(el, _)| el.cmp(&start_pos)).err().expect("binary search panicked")]
                    };
                    (valuator.result() as uint,
                     self.parent.lex_closure.call((self.input.slice(start_pos, end_pos), tok_kind)))
                }
                Step::StepRule => {
                    let rule = valuator.rule();
                    let arg_0 = valuator.arg_0() as uint;
                    let arg_n = valuator.arg_n() as uint;
                    let elem = self.parent.rules_closure.call((self.stack.slice(arg_0, arg_n + 1), rule, &self.parent.rule_ids));
                    // println!("rule {} {} => {}", arg_0, arg_n, elem);
                    (arg_0, elem)
                }
                Step::StepInactive | Step::StepNullingSymbol => {
                    break;
                }
                other => panic!("unexpected step {}", other),
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
            struct SlifRepr<'a, T>(T);
        )
    } else {
        quote_item!(cx,
             enum SlifRepr<'a, T> {
                 ValInfer(T),
                 ValStrSlice(&'a str),
             }
         )
    };

    let slif_expr = quote_expr!(cx, {
        use std::iter::{Map, Iterator};
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
                where C: for<'c> Fn(&[SlifRepr<'c, T>], Rule, &[Rule; $num_rules]) -> SlifRepr<'c, T>,
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
        impl<'a, 'b, 'l, T, C, D> Iterator<T> for SlifParse<'a, 'b, 'l, T, C, D>
                where C: for<'c> Fn(&[SlifRepr<'c, T>], Rule, &[Rule; $num_rules]) -> SlifRepr<'c, T>,
                      D: for<'c> Fn(&'c str, uint) -> SlifRepr<'c, T> {
            fn next(&mut self) -> Option<T> {
                $slif_iter_next_expr
            }
        }
        SlifGrammar::new(|&: args, rule, rules| {
            $rule_cond_expr
        },
        |&: arg, tok_kind| {
            $lex_cond_expr
        })
    });

    MacExpr::new(slif_expr)
}
