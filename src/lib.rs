#![crate_name = "marpa"]

extern crate libc;

use std::borrow::Cow;

#[repr(C)]
pub struct Config {
    t_is_ok: i32,
    t_error: i32,
    t_error_str: *const u8,
}

type SymbolId = i32;
type RuleId = i32;
type EarleySetId = i32;
type EarleyItemId = i32;
type Earleme = i32;
type StepType = i32;

pub struct Symbol {
    sym: SymbolId,
}

// impl Symbol {
//     pub fn new() -> Symbol {
//         Symbol {
//             sym: unsafe { marpa_g_symbol_new() },
//         }
//     }
// }

#[deriving(PartialEq, Eq)]
pub struct Rule {
    id: RuleId,
}

pub struct MarpaGrammar {
    p: *const MarpaG,
}

struct MarpaG;

struct Grammar {
    g: MarpaGrammar
}

impl Grammar {
    fn new(config: Config) -> Grammar {
        let g = unsafe {
            marpa_g_new(&config)
        };
        Grammar { g: g }
    }

    fn add_symbol(&self) -> Symbol {
        Symbol {
            sym: unsafe { marpa_g_symbol_new(self.g) },
        }
    }

    fn add_rule(&self, lhs: Symbol, rhs: &[Symbol]) -> Rule {
        unimplemented!()
    }

    fn set_start_symbol(&self, sym: Symbol) {
        unsafe {
            marpa_g_start_symbol_set(self.g, sym.sym); // -> ?
        }
    }

    fn precompute(&self) {
        unimplemented!();
        // add code here
    }

    fn recognizer(&self) -> Recognizer {
        unimplemented!();
        // add code here
    }
}

impl Drop for Grammar {
    fn drop(&mut self) {
        unsafe {
            marpa_g_unref(self.g);
        }
    }
}

enum MarpaR {}

struct Recognizer {
    recce: *mut MarpaR,
}

impl Recognizer {
    fn start_input(&self) {
        unimplemented!();
    }

    fn alternative(&self, token_id: Symbol, value: i32, length: i32) {
        unimplemented!()
    }

    fn earleme_complete(&self) {
        unimplemented!()
    }

    fn latest_earley_set(&self) -> EarleySetId {
        unimplemented!()
    }
}

enum MarpaBocage {}

struct Bocage {
    bocage: *mut MarpaBocage,
}

impl Bocage {
    fn new(recce: Recognizer, earley_set_id: EarleySetId) -> Bocage {
        Bocage {
            bocage: unsafe { marpa_b_new(recce.recce, earley_set_id) },
        }
    }

    fn order(&self) -> Order {
        unimplemented!()
    }
}

enum MarpaOrder {}

struct Order {
    order: *mut MarpaOrder,
}

impl Order {
    fn tree(&self) -> Tree {
        unimplemented!()
    }
}

enum MarpaTree {}

struct Tree {
    tree: *mut MarpaTree,
}

impl Tree {
    fn next(&self) -> i32 {
        unimplemented!()
    }

    fn value(&self) -> Value {
        unimplemented!()
    }
}

enum MarpaValue {}

struct Value {
    value: *mut MarpaValue,
}

impl Value {
    fn step(&self) -> Step {
        unimplemented!()
    }

    fn symbol_is_valued_set(&self, rule: Rule, n: i32) {
        unimplemented!()
    }

    fn rule(&self) -> Rule {
        unimplemented!()
    }

    fn token_value(&self) -> i32 {
        unimplemented!()
    }

    fn result(&self) -> i32 {
        unimplemented!()
    }

    fn arg_0(&self) -> i32 {
        unimplemented!()
    }

    fn arg_n(&self) -> i32 {
        unimplemented!()
    }
}

#[repr(C)]
enum Step {
    StepInternal1 = 0,
    StepRule = 1,
    StepToken = 2,
    StepNullingSymbol = 3,
    StepTrace = 4,
    StepInactive = 5,
    StepInternal2 = 6,
    StepInitial = 7,
    StepCount = 8,
}

#[link(name = "marpa")]
extern {
    fn marpa_c_init(config: *mut Config) -> i32;

    fn marpa_g_new(config: *const Config) -> MarpaGrammar;
    fn marpa_g_ref(grammar: MarpaGrammar) -> MarpaGrammar;
    fn marpa_g_unref(grammar: MarpaGrammar);
    fn marpa_g_precompute(grammar: MarpaGrammar) -> i32;
    fn marpa_g_is_precomputed(grammar: MarpaGrammar) -> i32;

    fn marpa_g_symbol_new(grammar: MarpaGrammar) -> SymbolId;
    fn marpa_g_start_symbol_set(grammar: MarpaGrammar, sym: SymbolId) -> SymbolId;

    fn marpa_r_start_input(recce: *mut MarpaR) -> i32;
    fn marpa_r_alternative(recce: *mut MarpaR, token_id: SymbolId, value: i32, length: i32) -> i32;

    fn marpa_b_new(recce: *mut MarpaR, earley_set_id: EarleySetId) -> *mut MarpaBocage;


 
    static marpa_major_version: i32;
    static marpa_minor_version: i32;
    static marpa_micro_version: i32;
}

#[test]
fn test_simple_first() {
    let mut cfg = Config { t_is_ok: 0, t_error: 0, t_error_str: 0u as *const _ };
    unsafe {
        marpa_c_init(&mut cfg);
        Grammar::new(cfg);
    }
}

#[test]
fn test_simple_parse() {
    let tok_strings = ["0", "1", "2", "3", "0", "-", "+", "*"];
    let expected = [
        ("(2-(0*(3+1))) == 2", 2),
        ("(((2-0)*3)+1) == 7", 7),
        ("((2-(0*3))+1) == 3", 3),
        ("((2-0)*(3+1)) == 8", 8),
        ("(2-((0*3)+1)) == 1", 1),
    ];
    let mut cfg = Config { t_is_ok: 0, t_error: 0, t_error_str: 0u as *const _ };
    let mut g = unsafe {
        marpa_c_init(&mut cfg);
        Grammar::new(cfg)
    };
    //[].iter().map(|name| (name, g.new_symbol())).
    let s = g.add_symbol();
    let e = g.add_symbol();
    let op = g.add_symbol();
    let number = g.add_symbol();
    g.set_start_symbol(s);
    let start_rule  = g.add_rule(s, &[e]);
    let op_rule     = g.add_rule(e, &[e, op, e]);
    let number_rule = g.add_rule(e, &[number]);
    g.precompute();

    let r = g.recognizer();
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
    let bocage = Bocage::new(r, latest_es);
    let order = bocage.order();
    let tree = order.tree();

    let mut stack = vec![];

    while tree.next() >= 0 {
        let mut nextok = true;
        let valuator = tree.value();
        valuator.symbol_is_valued_set(op_rule, 1);
        valuator.symbol_is_valued_set(start_rule, 1);
        valuator.symbol_is_valued_set(number_rule, 1);

        loop {
            match valuator.step() {
                Step::StepToken => {
                    let tok_value_idx = valuator.token_value() as uint;
                    stack[valuator.result() as uint] = (
                        Cow::Borrowed(tok_strings[tok_value_idx]), tok_value_idx
                    );
                }
                Step::StepRule => {
                    let arg_0 = valuator.arg_0() as uint;
                    let arg_n = valuator.arg_n() as uint;
                    let &(_, val) = &stack[arg_0];
                    match valuator.rule() {
                        rule if start_rule == rule => {
                            stack[arg_0] = {
                                let &(ref s, val) = &stack[arg_n];
                                (Cow::Owned(format!("{} == {}", s, val)), val)
                            };
                        }
                        rule if op_rule == rule => {
                            let result = match stack[arg_0 + 1] {
                                (Cow::Borrowed("+"), val2) => val + val2,
                                (Cow::Borrowed("-"), val2) => val - val2,
                                (Cow::Borrowed("*"), val2) => val * val2,
                                _ => panic!("unknown op"),
                            };
                            stack[arg_0] = {
                                let &(ref s, val) = &stack[arg_0];
                                let &(ref op, _) = &stack[arg_0 + 1];
                                let &(ref right_str, _) = &stack[arg_n];
                                (Cow::Owned(format!("({}{}{})", s, op, right_str)), result)
                            };
                        }
                        rule if number_rule == rule => {
                            stack[arg_0] = (Cow::Owned(val.to_string()), val);
                        }
                        _ => panic!("unknown rule"),
                    }
                }
                Step::StepInactive => {
                    break;
                }
                _ => panic!("unexpected step"),
            }
        }

        {
            let &(ref result_str, result_val) = &stack[0];
            match expected.iter().find(|&&(s, _)| *s == **result_str) {
                Some(&(_, val)) => {
                    if val == result_val {
                        println!("found {}", stack[0]);
                    } else {
                        println!("expected {}, but found {}", val, stack[0]);
                    }
                }
                None => {
                    println!("totally unexpected {}", stack[0]);
                }
            }
        };
        stack.clear();
    }
}
