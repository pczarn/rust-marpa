use {Config, Bocage, Grammar, Order, Recognizer, Step, Tree};

use std::borrow::Cow;

#[test]
fn test_simple_with_cfg() {
    let mut cfg = Config::new();
    Grammar::with_config(&mut cfg).unwrap();
}

#[derive(Clone, Debug)]
struct Node {
    formatted: Cow<'static, str>,
    value: u32,
}

#[test]
fn test_ambiguous_parse() {
    let tok_strings = [".", "1", "2", "3", "0", "-", "+", "*"];
    let tok_value = [0, 1, 2, 3, 0, 0, 0, 0];
    let expected = [
        ("(2-(0*(3+1))) == 2", 2),
        ("(((2-0)*3)+1) == 7", 7),
        ("((2-(0*3))+1) == 3", 3),
        ("((2-0)*(3+1)) == 8", 8),
        ("(2-((0*3)+1)) == 1", 1),
    ];

    let mut cfg = Config::new();
    let mut g = Grammar::with_config(&mut cfg).unwrap();

    let s = g.symbol_new().unwrap();
    let e = g.symbol_new().unwrap();
    let op = g.symbol_new().unwrap();
    let number = g.symbol_new().unwrap();
    g.start_symbol_set(s);
    let start_rule  = g.rule_new(s, &[e]).unwrap();
    let op_rule     = g.rule_new(e, &[e, op, e]).unwrap();
    let number_rule = g.rule_new(e, &[number]).unwrap();
    g.precompute().unwrap();

    let mut r = Recognizer::new(&mut g).unwrap();
    r.start_input();

    let tok_symbols = [number, number, number, number, number, op, op, op];

    for ch in "2 - 0 * 3 + 1".chars() {
        match tok_strings.iter().position(|&tok| tok.find(ch) == Some(0)) {
            Some(pos) => {
                r.alternative(tok_symbols[pos], pos as i32, 1);
                r.earleme_complete();
            }
            None => {}
        }
    }

    let latest_es = r.latest_earley_set();
    let mut bocage = Bocage::new(&mut r, latest_es).unwrap();
    let mut order = Order::new(&mut bocage).unwrap();
    let mut tree = Tree::new(&mut order).unwrap();

    let mut stack: Vec<Node> = vec![];

    for mut valuator in tree.values() {
        valuator.rule_is_valued_set(op_rule, 1);
        valuator.rule_is_valued_set(start_rule, 1);
        valuator.rule_is_valued_set(number_rule, 1);

        loop {
            let (idx, elem) = match valuator.step() {
                Step::StepToken => {
                    let tok_idx = valuator.token_value() as usize;
                    (valuator.result() as usize,
                     Node {
                        formatted: tok_strings[tok_idx].into(),
                        value: tok_value[tok_idx],
                     })
                }
                Step::StepRule => {
                    let rule = valuator.rule();
                    let arg_0 = valuator.arg_0() as usize;
                    let arg_n = valuator.arg_n() as usize;
                    let &Node { formatted: ref left_str,  value: lval } = &stack[arg_0];
                    let &Node { formatted: ref right_str, value: rval } = &stack[arg_n];

                    let elem = if start_rule == rule {
                        Node {
                            formatted: format!("{} == {}", right_str, rval).into(),
                            value: rval,
                        }
                    } else if number_rule == rule {
                        Node {
                            formatted: lval.to_string().into(),
                            value: lval,
                        }
                    } else if op_rule == rule {
                        let op = &stack[arg_0 + 1].formatted[..];
                        Node {
                            formatted: format!("({}{}{})", left_str, op, right_str).into(),
                            value: match op {
                                "+" => lval + rval,
                                "-" => lval - rval,
                                "*" => lval * rval,
                                _ => panic!("unknown op"),
                            },
                        }
                    } else {
                        panic!("unknown rule")
                    };

                    (arg_0, elem)
                }
                Step::StepInactive => {
                    break;
                }
                other => panic!("unexpected step {:?}", other),
            };

            if idx == stack.len() {
                stack.push(elem);
            } else {
                stack[idx] = elem;
            }
        }

        {
            let &Node { formatted: ref result_str, value: result_val } = &stack[0];
            match expected.iter().find(|&&(ref s, _)| **s == **result_str) {
                Some(&(_, val)) => {
                    if val != result_val {
                        panic!("expected {:?}, but found {:?}", val, stack[0]);
                    }
                }
                None => panic!("totally unexpected {:?}", stack[0])
            }
        };
        stack.clear();
    }
}
