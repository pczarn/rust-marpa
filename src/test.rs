use {Config, Bocage, Grammar, Order, Recognizer, Step, Tree, Value};

use std::str::CowString;

#[test]
fn test_simple_create() {
    let mut cfg = Config::new();
    Grammar::new(&mut cfg);
}

#[test]
fn test_ambiguous_parse() {
    let tok_strings = ["0", "1", "2", "3", "0", "-", "+", "*"];
    let tok_value = [0, 1, 2, 3, 0, 0, 0, 0];
    let expected = [
        ("(2-(0*(3+1))) == 2", 2),
        ("(((2-0)*3)+1) == 7", 7),
        ("((2-(0*3))+1) == 3", 3),
        ("((2-0)*(3+1)) == 8", 8),
        ("(2-((0*3)+1)) == 1", 1),
    ];

    let mut cfg = Config::new();
    let g = Grammar::new(&mut cfg);

    let s = g.add_symbol();
    let e = g.add_symbol();
    let op = g.add_symbol();
    let number = g.add_symbol();
    g.set_start_symbol(s);
    let start_rule  = g.add_rule(s, &[e]);
    let op_rule     = g.add_rule(e, &[e, op, e]);
    let number_rule = g.add_rule(e, &[number]);
    g.precompute();

    let r = Recognizer::new(&g);
    r.start_input();

    let (zero, minus_tok, plus_tok, mul_tok) = (4, 5, 6, 7);

    // 2 - 0 * 3 + 1
    r.alternative(number, 2, 1);
    r.earleme_complete();
    r.alternative(op, minus_tok, 1);
    r.earleme_complete();
    r.alternative(number, zero, 1);
    r.earleme_complete();
    r.alternative(op, mul_tok, 1);
    r.earleme_complete();
    r.alternative(number, 3, 1);
    r.earleme_complete();
    r.alternative(op, plus_tok, 1);
    r.earleme_complete();
    r.alternative(number, 1, 1);
    r.earleme_complete();

    let latest_es = r.latest_earley_set();
    let bocage = Bocage::new(&r, latest_es);
    let order = Order::new(&bocage);
    let tree = Tree::new(&order);

    let mut stack: Vec<(CowString, uint)> = vec![];

    while tree.next() >= 0 {
        let valuator = Value::new(&tree);
        valuator.rule_is_valued_set(op_rule, 1);
        valuator.rule_is_valued_set(start_rule, 1);
        valuator.rule_is_valued_set(number_rule, 1);

        loop {
            let (idx, elem) = match valuator.step() {
                Step::StepToken => {
                    let tok_idx = valuator.token_value() as uint;
                    (valuator.result() as uint, (tok_strings[tok_idx].into_cow(),
                                                 tok_value[tok_idx]))
                }
                Step::StepRule => {
                    let arg_0 = valuator.arg_0() as uint;
                    let arg_n = valuator.arg_n() as uint;
                    let &(ref left_str, val) = &stack[arg_0];
                    let &(ref right_str, val2) = &stack[arg_n];
                    let rule = valuator.rule();

                    let elem = if start_rule == rule {
                        (format!("{} == {}", right_str, val2).into_cow(), val2)
                    } else if number_rule == rule {
                        (val.to_string().into_cow(), val)
                    } else if op_rule == rule {
                        let &(ref op, _) = &stack[arg_0 + 1];
                        let result = match op.as_slice() {
                            "+" => val + val2,
                            "-" => val - val2,
                            "*" => val * val2,
                            _ => panic!("unknown op"),
                        };
                        (format!("({}{}{})", left_str, op, right_str).into_cow(), result)
                    } else {
                        panic!("unknown rule")
                    };

                    (arg_0, elem)
                }
                Step::StepInactive => {
                    break;
                }
                other => panic!("unexpected step {}", other),
            };

            if idx == stack.len() {
                stack.push(elem);
            } else {
                stack[idx] = elem;
            }
        }

        {
            let &(ref result_str, result_val) = &stack[0];
            match expected.iter().find(|&&(ref s, _)| **s == **result_str) {
                Some(&(_, val)) => {
                    if val != result_val {
                        panic!("expected {}, but found {}", val, stack[0]);
                    }
                }
                None => panic!("totally unexpected {}", stack[0])
            }
        };
        stack.clear();
    }
}
