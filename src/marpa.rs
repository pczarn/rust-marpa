use ffi;
use ffi::ErrorCode;

use std::ptr;

macro_rules! numbered_object_type {
    ($Ty:ident, $Id:ty) => (
        #[repr(packed)]
        #[derive(Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $Ty {
            id: $Id,
        }
        impl $Ty {
            #[allow(dead_code)]
            fn new(id: $Id) -> Option<$Ty> {
                if id < 0 {
                    None
                } else {
                    Some($Ty { id: id })
                }
            }
        }
    )
}

numbered_object_type!(Earleme, ffi::EarlemeId);
numbered_object_type!(EarleyItem, ffi::EarleyItemId);
numbered_object_type!(EarleySet, ffi::EarleySetId);
numbered_object_type!(Symbol, ffi::SymbolId);
numbered_object_type!(Rule, ffi::RuleId);

const PROPER_SEPARATION: i32 = 2;

/// A grammar. Grammars have no parent objects.
pub struct Grammar {
    grammar: *mut ffi::MarpaGrammar
}

impl Grammar {
    /// Constructs a new grammar. The returned grammar object is not yet precomputed, and will
    /// have no symbols and rules. Its reference count will be 1.
    pub fn new() -> Option<Grammar> {
        let g_ptr = unsafe {
            ffi::marpa_g_new(ptr::null_mut())
        };
        if g_ptr.is_null() {
            None
        } else {
            Some(Grammar { grammar: g_ptr })
        }
    }

    /// Constructs a new grammar.
    ///
    /// # Errors
    ///
    /// If an error happens during grammar construction, the `ErrorCode` is returned as an `Err`.
    pub fn with_config(config: &mut ffi::Config) -> Result<Grammar, ErrorCode> {
        let g_ptr = unsafe {
            ffi::marpa_g_new(config)
        };
        if g_ptr.is_null() {
            Err(config.error_code())
        } else {
            Ok(Grammar { grammar: g_ptr })
        }
    }

    /// Creates a new symbol. Returns `None` on failure.
    pub fn symbol_new(&mut self) -> Option<Symbol> {
        Symbol::new(unsafe {
            ffi::marpa_g_symbol_new(self.grammar)
        })
    }

    /// Creates a new rule. Returns `None` on failure.
    pub fn rule_new(&mut self, lhs: Symbol, rhs: &[Symbol]) -> Option<Rule> {
        Rule::new(unsafe {
            ffi::marpa_g_rule_new(self.grammar, lhs.id, rhs.as_ptr() as *const _, rhs.len() as i32)
        })
    }

    /// Creates a new sequence rule. Returns `None` on failure.
    pub fn sequence_new(&mut self, lhs: Symbol, rhs: Symbol, sep: Option<Symbol>, min: i32)
                       -> Option<Rule> {
        Rule::new(unsafe {
            let sep = sep.unwrap_or(Symbol { id: -1 });
            ffi::marpa_g_sequence_new(self.grammar, lhs.id, rhs.id, sep.id, min, PROPER_SEPARATION)
        })
    }

    /// Returns current value of the start symbol of the grammar, or `None` if one hasn't been set
    /// with `start_symbol_set`.
    pub fn start_symbol(&self) -> Option<Symbol> {
        Symbol::new(unsafe {
            ffi::marpa_g_start_symbol(self.grammar)
        })
    }

    /// Sets the start symbol of the grammar to `sym`. Returns the value of the new start symbol,
    /// or `None` if `sym` is well-formed, but there is no such symbol.
    pub fn start_symbol_set(&mut self, sym: Symbol) -> Option<Symbol> {
        Symbol::new(unsafe {
            ffi::marpa_g_start_symbol_set(self.grammar, sym.id)
        })
    }

    /// Precomputation involves freezing and then thoroughly checking the grammar.
    ///
    /// # Errors
    ///
    /// If precomputation fails, the `ErrorCode` is returned as an `Err`. Among the reasons
    /// for precomputation to fail are the following:
    ///
    /// * `NoRules`: The grammar has no rules.
    /// * `NoStartSymbol`: No start symbol was specified.
    /// * `InvalidStartSymbol`: A start symbol ID was specified,
    ///   but it is not the ID of a valid symbol.
    /// * `StartNotLhs`: The start symbol is not on the LHS of any rule.
    /// * `UnproductiveStart`: The start symbol is not productive.
    /// * `CountedNullable`: A symbol on the RHS of a sequence rule is
    ///   nullable. Libmarpa does not allow this.
    /// * `NullingTerminal`: A terminal is also a nulling symbol.
    ///   Libmarpa does not allow this.
    pub fn precompute(&mut self) -> Result<(), ErrorCode> {
        unsafe {
            if ffi::marpa_g_precompute(self.grammar) >= 0 {
                Ok(())
            } else {
                Err(ffi::marpa_g_error(self.grammar, ptr::null()))
            }
        }
    }

    /// Determines if the grammar is precomputed.
    pub fn is_precomputed(&self) -> bool {
        unsafe {
            ffi::marpa_g_is_precomputed(self.grammar) == 1
        }
    }
}

impl Drop for Grammar {
    fn drop(&mut self) {
        unsafe {
            ffi::marpa_g_unref(self.grammar);
        }
    }
}

pub struct Recognizer {
    recce: *mut ffi::MarpaRecce,
    grammar: *mut ffi::MarpaGrammar,
}

impl Recognizer {
    /// Constructs a new recognizer. Returns `None` if `grammar` is not precomputed, or on other
    /// failure.
    pub fn new(grammar: &mut Grammar) -> Result<Recognizer, ErrorCode> {
        let recce_ptr = unsafe {
            ffi::marpa_r_new(grammar.grammar)
        };
        if recce_ptr.is_null() {
            Err(unsafe { ffi::marpa_g_error(grammar.grammar, ptr::null()) })
        } else {
            Ok(Recognizer { recce: recce_ptr, grammar: grammar.grammar })
        }
    }

    /// Makes the recognizer ready for input. Returns `true` on success.
    pub fn start_input(&mut self) -> bool {
        unsafe {
            ffi::marpa_r_start_input(self.recce) >= 0
        }
    }

    pub fn alternative(&mut self, token_id: Symbol, value: i32, length: i32) -> ErrorCode {
        debug_assert!(value != 0);
        // TODO: return value
        unsafe {
            ffi::marpa_r_alternative(self.recce, token_id.id, value, length)
        }
    }

    /// Finalizes processing of the current earleme. An exhausted parse may cause a failure.
    pub fn earleme_complete(&mut self) -> Option<Earleme> {
        Earleme::new(unsafe {
            ffi::marpa_r_earleme_complete(self.recce)
        })
    }

    /// Returns the latest Earley set.
    pub fn latest_earley_set(&self) -> EarleySet {
        // always succeeds
        EarleySet::new(unsafe {
            ffi::marpa_r_latest_earley_set(self.recce)
        }).unwrap()
    }

    pub unsafe fn terminals_expected<'a>(&self, ary: &'a mut [Symbol]) -> &'a [Symbol] {
        let n = ffi::marpa_r_terminals_expected(self.recce, ary.as_mut_ptr() as *mut _);
        assert!(n >= 0);
        &ary[..n as usize]
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
    grammar: *mut ffi::MarpaGrammar,
}

impl Bocage {
    /// Constructs a new bocage. Returns `None` on failure.
    pub fn new(recce: &mut Recognizer, earley_set: EarleySet) -> Result<Bocage, ErrorCode> {
        let bocage_ptr = unsafe {
            ffi::marpa_b_new(recce.recce, earley_set.id)
        };
        if bocage_ptr.is_null() {
            Err(unsafe { ffi::marpa_g_error(recce.grammar, ptr::null()) })
        } else {
            Ok(Bocage { bocage: bocage_ptr, grammar: recce.grammar })
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
    grammar: *mut ffi::MarpaGrammar,
}

impl Order {
    /// Constructs a new order. Returns `None` on failure.
    pub fn new(bocage: &mut Bocage) -> Result<Order, ErrorCode> {
        let order_ptr = unsafe {
            ffi::marpa_o_new(bocage.bocage)
        };
        if order_ptr.is_null() {
            Err(unsafe { ffi::marpa_g_error(bocage.grammar, ptr::null()) })
        } else {
            Ok(Order { order: order_ptr, grammar: bocage.grammar })
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
    grammar: *mut ffi::MarpaGrammar,
}

impl Tree {
    pub fn new(order: &mut Order) -> Result<Tree, ErrorCode> {
        let tree_ptr = unsafe {
            ffi::marpa_t_new(order.order)
        };
        if tree_ptr.is_null() {
            Err(unsafe { ffi::marpa_g_error(order.grammar, ptr::null()) })
        } else {
            Ok(Tree { tree: tree_ptr, grammar: order.grammar })
        }
    }

    pub fn next(&mut self) -> i32 {
        unsafe {
            ffi::marpa_t_next(self.tree)
        }
    }

    pub fn values(&mut self) -> Values {
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
    pub fn new(tree: &mut Tree) -> Result<Value, ErrorCode> {
        let val_ptr = unsafe {
            ffi::marpa_v_new(tree.tree)
        };
        if val_ptr.is_null() {
            Err(unsafe { ffi::marpa_g_error(tree.grammar, ptr::null()) })
        } else {
            Ok(Value { value: val_ptr })
        }
    }

    pub fn step(&mut self) -> ffi::Step {
        unsafe {
            ffi::marpa_v_step(self.value)
        }
    }

    pub fn rule_is_valued_set(&mut self, rule: Rule, n: i32) {
        unsafe {
            ffi::marpa_v_rule_is_valued_set(self.value, rule.id, n);
        }
    }

    pub fn symbol_is_valued_set(&mut self, sym: Symbol, n: i32) {
        unsafe {
            ffi::marpa_v_symbol_is_valued_set(self.value, sym.id, n);
        }
    }

    pub fn symbol(&self) -> Symbol {
        unsafe { Symbol::new((*self.value).t_token_id).unwrap() }
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
    tree: &'a mut Tree,
}

impl<'a> Iterator for Values<'a> {
    type Item = Value;

    fn next(&mut self) -> Option<Value> {
        if self.tree.next() >= 0 {
            Value::new(self.tree).ok()
        } else {
            None
        }
    }
}
