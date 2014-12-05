use ffi;

#[repr(packed)]
pub struct Symbol {
    sym: ffi::SymbolId,
}

#[deriving(PartialEq, Eq)]
pub struct Rule {
    id: ffi::RuleId,
}

pub struct EarleySet {
    id: ffi::EarleySetId,
}

pub struct Grammar {
    g: *mut ffi::MarpaGrammar,
}

impl Grammar {
    pub fn new(config: &mut ffi::Config) -> Grammar {
        let g = unsafe {
            ffi::marpa_g_new(&*config)
        };
        Grammar { g: g }
    }

    pub fn add_symbol(&self) -> Symbol {
        Symbol {
            sym: unsafe { ffi::marpa_g_symbol_new(self.g) },
        }
    }

    pub fn add_rule(&self, lhs: Symbol, rhs: &[Symbol]) -> Rule {
        Rule {
            id: unsafe {
                ffi::marpa_g_rule_new(self.g, lhs.sym, rhs.as_ptr() as *const _, rhs.len() as i32)
            }
        }
    }

    pub fn set_start_symbol(&self, sym: Symbol) {
        unsafe {
            ffi::marpa_g_start_symbol_set(self.g, sym.sym); // -> ?
        }
    }

    pub fn precompute(&self) {
        unsafe {
            ffi::marpa_g_precompute(self.g);
        }
    }
}

impl Drop for Grammar {
    fn drop(&mut self) {
        unsafe {
            ffi::marpa_g_unref(self.g);
        }
    }
}

pub struct Recognizer {
    recce: *mut ffi::MarpaRecce,
}

impl Recognizer {
    pub fn new(grammar: &Grammar) -> Recognizer {
        Recognizer {
            recce: unsafe {
                ffi::marpa_r_new(grammar.g)
            }
        }
    }

    pub fn start_input(&self) {
        unsafe {
            ffi::marpa_r_start_input(self.recce);
        }
    }

    pub fn alternative(&self, token_id: Symbol, value: i32, length: i32) {
        unsafe {
            ffi::marpa_r_alternative(self.recce, token_id.sym, value, length);
        }
    }

    pub fn earleme_complete(&self) {
        unsafe {
            ffi::marpa_r_earleme_complete(self.recce);
        }
    }

    pub fn latest_earley_set(&self) -> EarleySet {
        EarleySet {
            id: unsafe {
                ffi::marpa_r_latest_earley_set(self.recce)
            }
        }
    }
}

impl Drop for Recognizer {
    fn drop(&mut self) {
        unsafe {
            ffi::marpa_r_unref(self.recce);
        }
    }
}

pub struct Bocage {
    bocage: *mut ffi::MarpaBocage,
}

impl Bocage {
    pub fn new(recce: &Recognizer, earley_set: EarleySet) -> Bocage {
        Bocage {
            bocage: unsafe { ffi::marpa_b_new(recce.recce, earley_set.id) },
        }
    }
}

impl Drop for Bocage {
    fn drop(&mut self) {
        unsafe {
            ffi::marpa_b_unref(self.bocage);
        }
    }
}

pub struct Order {
    order: *mut ffi::MarpaOrder,
}

impl Order {
    pub fn new(bocage: &Bocage) -> Order {
        Order {
            order: unsafe {
                ffi::marpa_o_new(bocage.bocage)
            }
        }
    }
}

impl Drop for Order {
    fn drop(&mut self) {
        unsafe {
            ffi::marpa_o_unref(self.order);
        }
    }
}

pub struct Tree {
    tree: *mut ffi::MarpaTree,
}

impl Tree {
    pub fn new(order: &Order) -> Tree {
        Tree {
            tree: unsafe {
                ffi::marpa_t_new(order.order)
            }
        }
    }

    pub fn next(&self) -> i32 {
        unsafe {
            ffi::marpa_t_next(self.tree)
        }
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        unsafe {
            ffi::marpa_t_unref(self.tree);
        }
    }
}

pub struct Value {
    value: *mut ffi::MarpaValue,
}

impl Value {
    pub fn new(tree: &Tree) -> Value {
        Value {
            value: unsafe {
                ffi::marpa_v_new(tree.tree)
            }
        }
    }

    pub fn step(&self) -> ffi::Step {
        unsafe {
            ffi::marpa_v_step(self.value)
        }
    }

    pub fn rule_is_valued_set(&self, rule: Rule, n: i32) {
        unsafe {
            ffi::marpa_v_rule_is_valued_set(self.value, rule.id, n);
        }
    }

    pub fn symbol_is_valued_set(&self, sym: Symbol, n: i32) {
        unsafe {
            ffi::marpa_v_symbol_is_valued_set(self.value, sym.sym, n);
        }
    }

    pub fn rule(&self) -> Rule {
        Rule {
            id: unsafe { (*self.value).t_rule_id }
        }
    }

    pub fn token_value(&self) -> i32 {
        unsafe { (*self.value).t_token_value }
    }

    pub fn result(&self) -> i32 {
        unsafe { (*self.value).t_result }
    }

    pub fn arg_0(&self) -> i32 {
        unsafe { (*self.value).t_arg_0 }
    }

    pub fn arg_n(&self) -> i32 {
        unsafe { (*self.value).t_arg_n }
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        unsafe {
            ffi::marpa_v_unref(self.value);
        }
    }
}
