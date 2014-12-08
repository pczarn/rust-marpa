use ffi;

use std::kinds::marker;
use std::ptr;

macro_rules! numbered_object_type (
    ($Ty:ident, $Id:ty) => (
        #[repr(packed)]
        #[deriving(PartialEq, Eq)]
        pub struct $Ty {
            id: $Id,
        }
        impl $Ty {
            fn new(id: $Id) -> Option<$Ty> {
                if id < 0 {
                    None
                } else {
                    Some($Ty { id: id })
                }
            }
        }
    )
)

numbered_object_type!(Earleme, ffi::EarlemeId)
numbered_object_type!(EarleyItem, ffi::EarleyItemId)
numbered_object_type!(EarleySet, ffi::EarleySetId)
numbered_object_type!(Symbol, ffi::SymbolId)
numbered_object_type!(Rule, ffi::RuleId)

pub struct Grammar<'a> {
    grammar: *mut ffi::MarpaGrammar,
    marker: marker::ContravariantLifetime<'a>,
}

impl<'a> Grammar<'a> {
    pub fn new() -> Option<Grammar<'static>> {
        let g_ptr = unsafe {
            ffi::marpa_g_new(ptr::null_mut())
        };
        if g_ptr.is_null() {
            None
        } else {
            Some(Grammar { grammar: g_ptr, marker: marker::ContravariantLifetime })
        }
    }

    pub fn with_config(config: &mut ffi::Config) -> Option<Grammar> {
        let g_ptr = unsafe {
            ffi::marpa_g_new(config)
        };
        if g_ptr.is_null() {
            None
        } else {
            Some(Grammar { grammar: g_ptr, marker: marker::ContravariantLifetime })
        }
    }

    pub fn symbol_new(&self) -> Option<Symbol> {
        Symbol::new(unsafe {
            ffi::marpa_g_symbol_new(self.grammar)
        })
    }

    pub fn rule_new(&self, lhs: Symbol, rhs: &[Symbol]) -> Option<Rule> {
        Rule::new(unsafe {
            ffi::marpa_g_rule_new(self.grammar, lhs.id, rhs.as_ptr() as *const _, rhs.len() as i32)
        })
    }

    pub fn start_symbol_set(&self, sym: Symbol) -> Option<Symbol> {
        Symbol::new(unsafe {
            ffi::marpa_g_start_symbol_set(self.grammar, sym.id)
        })
    }

    pub fn start_symbol(&self) -> Option<Symbol> {
        Symbol::new(unsafe {
            ffi::marpa_g_start_symbol(self.grammar)
        })
    }

    pub fn precompute(&self) {
        unsafe {
            ffi::marpa_g_precompute(self.grammar);
        }
    }

    pub fn is_precomputed(&self) -> bool {
        unsafe {
            ffi::marpa_g_is_precomputed(self.grammar) == 1
        }
    }
}

#[unsafe_destructor]
impl<'a> Drop for Grammar<'a> {
    fn drop(&mut self) {
        unsafe {
            ffi::marpa_g_unref(self.grammar);
        }
    }
}

pub struct Recognizer {
    recce: *mut ffi::MarpaRecce,
}

impl Recognizer {
    pub fn new(grammar: &Grammar) -> Option<Recognizer> {
        let recce_ptr = unsafe {
            ffi::marpa_r_new(grammar.grammar)
        };
        if recce_ptr.is_null() {
            None
        } else {
            Some(Recognizer { recce: recce_ptr })
        }
    }

    pub fn start_input(&self) -> bool {
        unsafe {
            ffi::marpa_r_start_input(self.recce) >= 0
        }
    }

    pub fn alternative(&self, token_id: Symbol, value: i32, length: i32) {
        // TODO: return value
        unsafe {
            ffi::marpa_r_alternative(self.recce, token_id.id, value, length);
        }
    }

    /// An exhausted parse may cause a failure.
    pub fn earleme_complete(&self) -> Option<Earleme> {
        Earleme::new(unsafe {
            ffi::marpa_r_earleme_complete(self.recce)
        })
    }

    pub fn latest_earley_set(&self) -> EarleySet {
        EarleySet::new(unsafe {
            ffi::marpa_r_latest_earley_set(self.recce)
        }).unwrap()
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
    pub fn new(recce: &Recognizer, earley_set: EarleySet) -> Option<Bocage> {
        let bocage_ptr = unsafe {
            ffi::marpa_b_new(recce.recce, earley_set.id)
        };
        if bocage_ptr.is_null() {
            None
        } else {
            Some(Bocage { bocage: bocage_ptr })
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
    pub fn new(bocage: &Bocage) -> Option<Order> {
        let order_ptr = unsafe {
            ffi::marpa_o_new(bocage.bocage)
        };
        if order_ptr.is_null() {
            None
        } else {
            Some(Order { order: order_ptr })
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
    pub fn new(order: &Order) -> Option<Tree> {
        let tree_ptr = unsafe {
            ffi::marpa_t_new(order.order)
        };
        if tree_ptr.is_null() {
            None
        } else {
            Some(Tree { tree: tree_ptr })
        }
    }

    pub fn next(&self) -> i32 {
        unsafe {
            ffi::marpa_t_next(self.tree)
        }
    }

    pub fn values(&self) -> Values {
        Values {
            tree: self,
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
    pub fn new(tree: &Tree) -> Option<Value> {
        let val_ptr = unsafe {
            ffi::marpa_v_new(tree.tree)
        };
        if val_ptr.is_null() {
            None
        } else {
            Some(Value { value: val_ptr })
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
            ffi::marpa_v_symbol_is_valued_set(self.value, sym.id, n);
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

pub struct Values<'a> {
    tree: &'a Tree,
}

impl<'a> Iterator<Value> for Values<'a> {
    fn next(&mut self) -> Option<Value> {
        if self.tree.next() >= 0 {
            Value::new(self.tree)
        } else {
            None
        }
    }
}
