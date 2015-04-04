{
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
                (true, &Tok(Comma)) => {
                    syms.push((8u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(Semi)) => {
                    syms.push((9u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(RArrow)) => {
                    syms.push((10u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(Colon)) => {
                    syms.push((11u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(BinOp(Or))) => {
                    syms.push((12u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(Underscore)) => {
                    syms.push((13u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(Question)) => {
                    syms.push((14u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(BinOp(Star))) => {
                    syms.push((15u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(BinOp(Plus))) => {
                    syms.push((16u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &RustToken::Delim(OpenDelim(Brace))) => {
                    syms.push((17u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &RustToken::Delim(CloseDelim(Brace))) => {
                    syms.push((18u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &RustToken::Delim(OpenDelim(Paren))) => {
                    syms.push((19u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &RustToken::Delim(CloseDelim(Paren))) => {
                    syms.push((20u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &RustToken::Delim(OpenDelim(Bracket))) => {
                    syms.push((21u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &RustToken::Delim(CloseDelim(Bracket))) => {
                    syms.push((22u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &Tok(..)) => {
                    syms.push((23u32, self.offset as u32));
                }
                _ => (),
            }
            match (true, &self.input[self.offset]) {
                (true, &_) => { syms.push((24u32, self.offset as u32)); }
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
            Spec21(()),
            Spec22(()),
            Spec23(::syntax::parse::token::Token),
            Spec24(::syntax::parse::token::Token),
            Spec25(Context),
            Spec26(Opt),
            Spec27(LexRule),
            Spec28(Rule),
            Spec29(Alternative),
            Spec30(Alt),
            Spec31(Vec<Expr>),
            Spec32(Expr),
            Spec33(ast::KleeneOp),
            Spec34(P<Pat>),
            Spec35(InlineAction),
            Spec36(P<Ty>),
            Spec37(Ty_),
            Spec38(ast::Path),
            Spec39(PathSegment),
            Spec40(PathParameters),
            Spec41(P<Block>),
            Spec42(Vec<Token>),
            Spec43(Vec<Token>),
            Spec44(Vec<Token>),
            Spec45(Vec<Token>),
            Spec46(Vec<Token>),
            Spec47(()),
            Spec48(Opt),
            Spec49(Rule),
            Spec50(LexRule),
            Spec51(Alternative),
            Spec52(()),
            Spec53(Alt),
            Spec54(Option<P<Pat>>),
            Spec55(Expr),
            Spec56(Option<()>),
            Spec57(PathSegment),
            Spec58(()),
            Spec59(Option<PathParameters>),
            Spec60(Vec<Opt>),
            Spec61(Vec<Rule>),
            Spec62(Vec<LexRule>),
            Spec63(Vec<Alternative>),
            Spec64(Vec<Alt>),
            Spec65(Vec<Expr>),
            Spec66(Vec<PathSegment>),
        }
        struct Grammar<F, G, L> {
            grammar: ::marpa::Grammar,
            scan_syms: [::marpa::Symbol; 25usize],
            nulling_syms: [(::marpa::Symbol, u32); 3usize],
            rule_ids: [::marpa::Rule; 60usize],
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
                let mut syms: [Symbol; 67usize] =
                    unsafe { ::std::mem::uninitialized() };
                for s in syms.iter_mut() {
                    *s = grammar.symbol_new().unwrap();
                }
                grammar.start_symbol_set(syms[25usize]);
                let mut scan_syms: [Symbol; 25usize] =
                    unsafe { ::std::mem::uninitialized() };
                for (dst, src) in
                    scan_syms.iter_mut().zip(syms[..25usize].iter()) {
                    *dst = *src;
                }
                let rules: [(Symbol, &[Symbol]); 53usize] =
                    [(syms[25usize],
                      &[syms[60usize], syms[61usize], syms[62usize]]),
                     (syms[26usize],
                      &[syms[1usize], syms[4usize], syms[19usize],
                        syms[45usize], syms[20usize], syms[9usize]]),
                     (syms[27usize],
                      &[syms[1usize], syms[10usize], syms[36usize],
                        syms[0usize], syms[46usize], syms[35usize],
                        syms[9usize]]),
                     (syms[28usize],
                      &[syms[1usize], syms[10usize], syms[36usize],
                        syms[47usize], syms[63usize], syms[9usize]]),
                     (syms[29usize], &[syms[64usize], syms[35usize]]),
                     (syms[30usize], &[syms[54usize], syms[32usize]]),
                     (syms[31usize], &[syms[65usize]]),
                     (syms[32usize], &[syms[1usize]]),
                     (syms[32usize], &[syms[32usize], syms[14usize]]),
                     (syms[32usize],
                      &[syms[21usize], syms[31usize], syms[22usize],
                        syms[17usize], syms[31usize], syms[18usize],
                        syms[33usize]]),
                     (syms[32usize],
                      &[syms[21usize], syms[31usize], syms[22usize],
                        syms[33usize]]), (syms[33usize], &[syms[15usize]]),
                     (syms[33usize], &[syms[16usize]]),
                     (syms[34usize], &[syms[2usize], syms[11usize]]),
                     (syms[34usize],
                      &[syms[19usize], syms[45usize], syms[20usize],
                        syms[11usize]]), (syms[35usize], &[syms[41usize]]),
                     (syms[36usize], &[syms[37usize]]),
                     (syms[37usize], &[syms[38usize]]),
                     (syms[37usize], &[syms[19usize], syms[20usize]]),
                     (syms[37usize], &[syms[13usize]]),
                     (syms[38usize], &[syms[56usize], syms[66usize]]),
                     (syms[39usize], &[syms[1usize], syms[59usize]]),
                     (syms[40usize],
                      &[syms[6usize], syms[36usize], syms[7usize]]),
                     (syms[41usize], &[syms[44usize]]),
                     (syms[42usize], &[syms[44usize]]),
                     (syms[42usize], &[syms[43usize]]),
                     (syms[43usize], &[syms[23usize]]),
                     (syms[43usize], &[syms[19usize], syms[20usize]]),
                     (syms[43usize],
                      &[syms[19usize], syms[45usize], syms[20usize]]),
                     (syms[43usize], &[syms[21usize], syms[22usize]]),
                     (syms[43usize],
                      &[syms[21usize], syms[45usize], syms[22usize]]),
                     (syms[44usize], &[syms[17usize], syms[18usize]]),
                     (syms[44usize],
                      &[syms[17usize], syms[45usize], syms[18usize]]),
                     (syms[45usize], &[syms[42usize]]),
                     (syms[45usize], &[syms[45usize], syms[42usize]]),
                     (syms[46usize], &[syms[43usize]]),
                     (syms[46usize], &[syms[46usize], syms[43usize]]),
                     (syms[47usize], &[syms[3usize], syms[5usize]]),
                     (syms[48usize], &[syms[26usize]]),
                     (syms[49usize], &[syms[28usize]]),
                     (syms[50usize], &[syms[27usize]]),
                     (syms[51usize], &[syms[29usize]]),
                     (syms[52usize], &[syms[12usize]]),
                     (syms[53usize], &[syms[30usize]]),
                     (syms[54usize], &[syms[34usize]]), (syms[54usize], &[]),
                     (syms[55usize], &[syms[32usize]]),
                     (syms[56usize], &[syms[3usize]]), (syms[56usize], &[]),
                     (syms[57usize], &[syms[39usize]]),
                     (syms[58usize], &[syms[3usize]]),
                     (syms[59usize], &[syms[40usize]]), (syms[59usize], &[])];
                let seq_rules:
                        [(Symbol, Symbol, Option<Symbol>, i32); 7usize] =
                    [(syms[60usize], syms[48usize],
                      if false { Some(syms[0usize as usize]) } else { None },
                      0i32),
                     (syms[61usize], syms[49usize],
                      if false { Some(syms[0usize as usize]) } else { None },
                      0i32),
                     (syms[62usize], syms[50usize],
                      if false { Some(syms[0usize as usize]) } else { None },
                      0i32),
                     (syms[63usize], syms[51usize],
                      if true { Some(syms[52usize as usize]) } else { None },
                      0i32),
                     (syms[64usize], syms[53usize],
                      if false { Some(syms[0usize as usize]) } else { None },
                      0i32),
                     (syms[65usize], syms[55usize],
                      if false { Some(syms[0usize as usize]) } else { None },
                      0i32),
                     (syms[66usize], syms[57usize],
                      if true { Some(syms[58usize as usize]) } else { None },
                      0i32)];
                let mut rule_ids: [Rule; 60usize] =
                    unsafe { ::std::mem::uninitialized() };
                {
                    for (dst, &(lhs, rhs)) in
                        rule_ids.iter_mut().zip(rules.iter()) {
                        *dst = grammar.rule_new(lhs, rhs).unwrap();
                    }
                    for (dst, &(lhs, rhs, sep, min)) in
                        rule_ids.iter_mut().skip(53usize).zip(seq_rules.iter())
                        {
                        *dst =
                            grammar.sequence_new(lhs, rhs, sep, min).unwrap();
                    }
                };
                let mut nulling_syms: [(Symbol, u32); 3usize] =
                    unsafe { ::std::mem::uninitialized() };
                let nulling_rule_id_n: &[usize] =
                    &[45usize, 48usize, 52usize];
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
                while !lex_parse.is_empty() {
                    let mut syms: [Symbol; 25usize] =
                        unsafe { mem::uninitialized() };
                    let mut terminals_expected: [u32; 25usize] =
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
                    Some(Repr::Spec25(val)) => Some(val),
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
                             (true, _) if choice_ == 21usize =>
                             Some(Repr::Spec21({ { { } } })),
                             (true, _) if choice_ == 22usize =>
                             Some(Repr::Spec22({ { { } } })),
                             (true, tok) if choice_ == 23usize =>
                             Some(Repr::Spec23({
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
                             (true, tok) if choice_ == 24usize =>
                             Some(Repr::Spec24({
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
                                 (true, Repr::Spec60(opt),
                                  Repr::Spec61(statements),
                                  Repr::Spec62(l0_statements)) =>
                                 Some(Repr::Spec25({
                                                       {
                                                           {
                                                               Context{options:
                                                                           opt,
                                                                       rules:
                                                                           statements,
                                                                       lex_rules:
                                                                           l0_statements,}
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 1usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[3usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec1(i), Repr::Spec45(tts)) =>
                                 Some(Repr::Spec26({
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
                         } else if choice_ == 2usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[2usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[4usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[5usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec1(i), Repr::Spec36(ty),
                                  Repr::Spec46(mut rhs), Repr::Spec35(b)) =>
                                 Some(Repr::Spec27({
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
                                                               LexRule{name:
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
                         } else if choice_ == 3usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[2usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[4usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec1(i), Repr::Spec36(ty),
                                  Repr::Spec63(rhs)) =>
                                 Some(Repr::Spec28({
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
                         } else if choice_ == 4usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec64(v), Repr::Spec35(action))
                                 =>
                                 Some(Repr::Spec29({
                                                       {
                                                           {
                                                               let mut pats =
                                                                   Vec::new();
                                                               let inner =
                                                                   v.into_iter().enumerate().map(|(n,
                                                                                                   (pat,
                                                                                                    expr))|
                                                                                                     {
                                                                                                 if let Some(pat)
                                                                                                        =
                                                                                                        pat
                                                                                                        {
                                                                                                     pats.push((n,
                                                                                                                pat));
                                                                                                 }
                                                                                                 expr
                                                                                             }).collect();
                                                               Alternative{inner:
                                                                               inner,
                                                                           pats:
                                                                               pats,
                                                                           action:
                                                                               action,}
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 5usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec54(pat), Repr::Spec32(a)) =>
                                 Some(Repr::Spec30({ { { (pat, a) } } })),
                                 _ => None,
                             }
                         } else if choice_ == 6usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec65(a)) =>
                                 Some(Repr::Spec31({ { { a } } })),
                                 _ => None,
                             }
                         } else if choice_ == 7usize {
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
                         } else if choice_ == 8usize {
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
                         } else if choice_ == 9usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[4usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[6usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec31(body), Repr::Spec31(sep),
                                  Repr::Spec33(op)) =>
                                 Some(Repr::Spec32({
                                                       {
                                                           {
                                                               ExprSeq(body,
                                                                       Some(sep),
                                                                       op)
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 10usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[3usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec31(body), Repr::Spec33(op))
                                 =>
                                 Some(Repr::Spec32({
                                                       {
                                                           {
                                                               ExprSeq(body,
                                                                       None,
                                                                       op)
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 11usize {
                             match (true,) {
                                 (true,) =>
                                 Some(Repr::Spec33({
                                                       { { ast::ZeroOrMore } }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 12usize {
                             match (true,) {
                                 (true,) =>
                                 Some(Repr::Spec33({
                                                       { { ast::OneOrMore } }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 13usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec2(i)) =>
                                 Some(Repr::Spec34({
                                                       {
                                                           {
                                                               let mut v =
                                                                   Vec::new();
                                                               v.push(i);
                                                               quote_pat(cx,
                                                                         v)
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 14usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec45(tts)) =>
                                 Some(Repr::Spec34({
                                                       {
                                                           {
                                                               quote_pat(cx,
                                                                         tts)
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 15usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec41(block)) =>
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
                         } else if choice_ == 16usize {
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
                         } else if choice_ == 17usize {
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
                         } else if choice_ == 18usize {
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
                         } else if choice_ == 19usize {
                             match (true,) {
                                 (true,) =>
                                 Some(Repr::Spec37({ { { TyInfer } } })),
                                 _ => None,
                             }
                         } else if choice_ == 20usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec56(global),
                                  Repr::Spec66(ps)) =>
                                 Some(Repr::Spec38({
                                                       {
                                                           {
                                                               ast::Path{global:
                                                                             global.is_some(),
                                                                         segments:
                                                                             ps,
                                                                         span:
                                                                             DUMMY_SP,}
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 21usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec1(i), Repr::Spec59(param))
                                 =>
                                 Some(Repr::Spec39({
                                                       {
                                                           {
                                                               PathSegment{identifier:
                                                                               i,
                                                                           parameters:
                                                                               param.unwrap_or_else(||
                                                                                                        PathParameters::none()),}
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 22usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec36(t)) =>
                                 Some(Repr::Spec40({
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
                         } else if choice_ == 23usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec44(tts)) =>
                                 Some(Repr::Spec41({
                                                       {
                                                           {
                                                               quote_block(cx,
                                                                           tts)
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 24usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec44(tt)) =>
                                 Some(Repr::Spec42({ { { tt } } })),
                                 _ => None,
                             }
                         } else if choice_ == 25usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec43(tt)) =>
                                 Some(Repr::Spec42({ { { tt } } })),
                                 _ => None,
                             }
                         } else if choice_ == 26usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec23(t)) =>
                                 Some(Repr::Spec43({
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
                         } else if choice_ == 27usize {
                             match (true,) {
                                 (true,) =>
                                 Some(Repr::Spec43({
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
                         } else if choice_ == 28usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec45(tts)) =>
                                 Some(Repr::Spec43({
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
                         } else if choice_ == 29usize {
                             match (true,) {
                                 (true,) =>
                                 Some(Repr::Spec43({
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
                         } else if choice_ == 30usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec45(tts)) =>
                                 Some(Repr::Spec43({
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
                         } else if choice_ == 31usize {
                             match (true,) {
                                 (true,) =>
                                 Some(Repr::Spec44({
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
                         } else if choice_ == 32usize {
                             match (true,
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec45(tts)) =>
                                 Some(Repr::Spec44({
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
                         } else if choice_ == 33usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec42(tt)) =>
                                 Some(Repr::Spec45({ { { tt } } })),
                                 _ => None,
                             }
                         } else if choice_ == 34usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec45(mut tts),
                                  Repr::Spec42(tt)) =>
                                 Some(Repr::Spec45({
                                                       {
                                                           {
                                                               tts.extend(tt.into_iter());
                                                               tts
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 35usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec43(tt)) =>
                                 Some(Repr::Spec46({ { { tt } } })),
                                 _ => None,
                             }
                         } else if choice_ == 36usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue),
                                    mem::replace(&mut args[1usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec46(mut tts),
                                  Repr::Spec43(tt)) =>
                                 Some(Repr::Spec46({
                                                       {
                                                           {
                                                               tts.extend(tt.into_iter());
                                                               tts
                                                           }
                                                       }
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 37usize {
                             match (true,) {
                                 (true,) => Some(Repr::Spec47({ { { } } })),
                                 _ => None,
                             }
                         } else if choice_ == 38usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec26(_pat_ident_)) =>
                                 Some(Repr::Spec48({ _pat_ident_ })),
                                 _ => None,
                             }
                         } else if choice_ == 39usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec28(_pat_ident_)) =>
                                 Some(Repr::Spec49({ _pat_ident_ })),
                                 _ => None,
                             }
                         } else if choice_ == 40usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec27(_pat_ident_)) =>
                                 Some(Repr::Spec50({ _pat_ident_ })),
                                 _ => None,
                             }
                         } else if choice_ == 41usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec29(_pat_ident_)) =>
                                 Some(Repr::Spec51({ _pat_ident_ })),
                                 _ => None,
                             }
                         } else if choice_ == 42usize {
                             match (true,) {
                                 (true,) => Some(Repr::Spec52({ () })),
                                 _ => None,
                             }
                         } else if choice_ == 43usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec30(_pat_ident_)) =>
                                 Some(Repr::Spec53({ _pat_ident_ })),
                                 _ => None,
                             }
                         } else if choice_ == 44usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec34(_pat_)) =>
                                 Some(Repr::Spec54({
                                                       ::std::option::Option::Some(_pat_)
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 46usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec32(_pat_ident_)) =>
                                 Some(Repr::Spec55({ _pat_ident_ })),
                                 _ => None,
                             }
                         } else if choice_ == 47usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec3(_pat_)) =>
                                 Some(Repr::Spec56({
                                                       ::std::option::Option::Some(_pat_)
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 49usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec39(_pat_ident_)) =>
                                 Some(Repr::Spec57({ _pat_ident_ })),
                                 _ => None,
                             }
                         } else if choice_ == 50usize {
                             match (true,) {
                                 (true,) => Some(Repr::Spec58({ () })),
                                 _ => None,
                             }
                         } else if choice_ == 51usize {
                             match (true,
                                    mem::replace(&mut args[0usize],
                                                 Repr::Continue)) {
                                 (true, Repr::Spec40(_pat_)) =>
                                 Some(Repr::Spec59({
                                                       ::std::option::Option::Some(_pat_)
                                                   })),
                                 _ => None,
                             }
                         } else if choice_ == 45usize {
                             Some(Repr::Spec54({
                                                   ::std::option::Option::None
                                               }))
                         } else if choice_ == 48usize {
                             Some(Repr::Spec56({
                                                   ::std::option::Option::None
                                               }))
                         } else if choice_ == 52usize {
                             Some(Repr::Spec59({
                                                   ::std::option::Option::None
                                               }))
                         } else if choice_ == 53usize {
                             let has_sep = false;
                             let v =
                                 args.iter_mut().enumerate().filter_map(|(n,
                                                                          arg)|
                                                                            if n
                                                                                   &
                                                                                   1
                                                                                   ==
                                                                                   0
                                                                                   ||
                                                                                   !has_sep
                                                                               {
                                                                                match (true,
                                                                                       mem::replace(arg,
                                                                                                    Repr::Continue))
                                                                                    {
                                                                                    (true,
                                                                                     Repr::Spec48(elem))
                                                                                    =>
                                                                                    Some(elem),
                                                                                    _
                                                                                    =>
                                                                                    None,
                                                                                }
                                                                            } else {
                                                                                None
                                                                            }).collect::<Vec<_>>();
                             if v.len() == (args.len() + 1) / 2 || !has_sep {
                                 Some(Repr::Spec60(v))
                             } else { None }
                         } else if choice_ == 54usize {
                             let has_sep = false;
                             let v =
                                 args.iter_mut().enumerate().filter_map(|(n,
                                                                          arg)|
                                                                            if n
                                                                                   &
                                                                                   1
                                                                                   ==
                                                                                   0
                                                                                   ||
                                                                                   !has_sep
                                                                               {
                                                                                match (true,
                                                                                       mem::replace(arg,
                                                                                                    Repr::Continue))
                                                                                    {
                                                                                    (true,
                                                                                     Repr::Spec49(elem))
                                                                                    =>
                                                                                    Some(elem),
                                                                                    _
                                                                                    =>
                                                                                    None,
                                                                                }
                                                                            } else {
                                                                                None
                                                                            }).collect::<Vec<_>>();
                             if v.len() == (args.len() + 1) / 2 || !has_sep {
                                 Some(Repr::Spec61(v))
                             } else { None }
                         } else if choice_ == 55usize {
                             let has_sep = false;
                             let v =
                                 args.iter_mut().enumerate().filter_map(|(n,
                                                                          arg)|
                                                                            if n
                                                                                   &
                                                                                   1
                                                                                   ==
                                                                                   0
                                                                                   ||
                                                                                   !has_sep
                                                                               {
                                                                                match (true,
                                                                                       mem::replace(arg,
                                                                                                    Repr::Continue))
                                                                                    {
                                                                                    (true,
                                                                                     Repr::Spec50(elem))
                                                                                    =>
                                                                                    Some(elem),
                                                                                    _
                                                                                    =>
                                                                                    None,
                                                                                }
                                                                            } else {
                                                                                None
                                                                            }).collect::<Vec<_>>();
                             if v.len() == (args.len() + 1) / 2 || !has_sep {
                                 Some(Repr::Spec62(v))
                             } else { None }
                         } else if choice_ == 56usize {
                             let has_sep = true;
                             let v =
                                 args.iter_mut().enumerate().filter_map(|(n,
                                                                          arg)|
                                                                            if n
                                                                                   &
                                                                                   1
                                                                                   ==
                                                                                   0
                                                                                   ||
                                                                                   !has_sep
                                                                               {
                                                                                match (true,
                                                                                       mem::replace(arg,
                                                                                                    Repr::Continue))
                                                                                    {
                                                                                    (true,
                                                                                     Repr::Spec51(elem))
                                                                                    =>
                                                                                    Some(elem),
                                                                                    _
                                                                                    =>
                                                                                    None,
                                                                                }
                                                                            } else {
                                                                                None
                                                                            }).collect::<Vec<_>>();
                             if v.len() == (args.len() + 1) / 2 || !has_sep {
                                 Some(Repr::Spec63(v))
                             } else { None }
                         } else if choice_ == 57usize {
                             let has_sep = false;
                             let v =
                                 args.iter_mut().enumerate().filter_map(|(n,
                                                                          arg)|
                                                                            if n
                                                                                   &
                                                                                   1
                                                                                   ==
                                                                                   0
                                                                                   ||
                                                                                   !has_sep
                                                                               {
                                                                                match (true,
                                                                                       mem::replace(arg,
                                                                                                    Repr::Continue))
                                                                                    {
                                                                                    (true,
                                                                                     Repr::Spec53(elem))
                                                                                    =>
                                                                                    Some(elem),
                                                                                    _
                                                                                    =>
                                                                                    None,
                                                                                }
                                                                            } else {
                                                                                None
                                                                            }).collect::<Vec<_>>();
                             if v.len() == (args.len() + 1) / 2 || !has_sep {
                                 Some(Repr::Spec64(v))
                             } else { None }
                         } else if choice_ == 58usize {
                             let has_sep = false;
                             let v =
                                 args.iter_mut().enumerate().filter_map(|(n,
                                                                          arg)|
                                                                            if n
                                                                                   &
                                                                                   1
                                                                                   ==
                                                                                   0
                                                                                   ||
                                                                                   !has_sep
                                                                               {
                                                                                match (true,
                                                                                       mem::replace(arg,
                                                                                                    Repr::Continue))
                                                                                    {
                                                                                    (true,
                                                                                     Repr::Spec55(elem))
                                                                                    =>
                                                                                    Some(elem),
                                                                                    _
                                                                                    =>
                                                                                    None,
                                                                                }
                                                                            } else {
                                                                                None
                                                                            }).collect::<Vec<_>>();
                             if v.len() == (args.len() + 1) / 2 || !has_sep {
                                 Some(Repr::Spec65(v))
                             } else { None }
                         } else if choice_ == 59usize {
                             let has_sep = true;
                             let v =
                                 args.iter_mut().enumerate().filter_map(|(n,
                                                                          arg)|
                                                                            if n
                                                                                   &
                                                                                   1
                                                                                   ==
                                                                                   0
                                                                                   ||
                                                                                   !has_sep
                                                                               {
                                                                                match (true,
                                                                                       mem::replace(arg,
                                                                                                    Repr::Continue))
                                                                                    {
                                                                                    (true,
                                                                                     Repr::Spec57(elem))
                                                                                    =>
                                                                                    Some(elem),
                                                                                    _
                                                                                    =>
                                                                                    None,
                                                                                }
                                                                            } else {
                                                                                None
                                                                            }).collect::<Vec<_>>();
                             if v.len() == (args.len() + 1) / 2 || !has_sep {
                                 Some(Repr::Spec66(v))
                             } else { None }
                         } else { None };
                     r.expect("marpa-macros: internal error: eval") })
    }
}
