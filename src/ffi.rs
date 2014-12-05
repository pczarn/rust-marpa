pub type SymbolId = i32;
pub type RuleId = i32;
pub type EarleySetId = i32;
pub type EarleyItemId = i32;
pub type Earleme = i32;
pub type StepType = i32;

pub enum MarpaGrammar {}
pub enum MarpaRecce {}
pub enum MarpaBocage {}
pub enum MarpaOrder {}
pub enum MarpaTree {}

#[repr(C)]
pub struct MarpaValue {
    pub t_step_type: Step,
    pub t_token_id: SymbolId,
    pub t_token_value: i32,
    pub t_rule_id: RuleId,
    pub t_arg_0: i32,
    pub t_arg_n: i32,
    pub t_result: i32,
    pub t_token_start_ys_id: EarleySetId,
    pub t_rule_start_ys_id: EarleySetId,
    pub t_ys_id: EarleySetId,
}

#[repr(C)]
pub struct Config {
    t_is_ok: i32,
    t_error: i32,
    t_error_str: *const u8,
}

impl Config {
    pub fn new() -> Config {
        let mut cfg = Config { t_is_ok: 0, t_error: 0, t_error_str: 0u as *const _ };
        unsafe { marpa_c_init(&mut cfg); }
        cfg
    }
}

#[deriving(Show)]
#[repr(C)]
pub enum Step {
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
    pub fn marpa_c_init(config: *mut Config) -> i32;

    pub fn marpa_g_new(config: *const Config) -> *mut MarpaGrammar;
    pub fn marpa_g_ref(grammar: *mut MarpaGrammar) -> *mut MarpaGrammar;

    pub fn marpa_g_unref(grammar: *mut MarpaGrammar);
    pub fn marpa_g_precompute(grammar: *mut MarpaGrammar) -> i32;
    pub fn marpa_g_is_precomputed(grammar: *mut MarpaGrammar) -> i32;
    pub fn marpa_g_symbol_new(grammar: *mut MarpaGrammar) -> SymbolId;
    pub fn marpa_g_start_symbol_set(grammar: *mut MarpaGrammar, sym: SymbolId) -> SymbolId;
    pub fn marpa_g_rule_new(g: *mut MarpaGrammar, lhs_id: SymbolId, rhs_ids: *const SymbolId,
                                                                    length: i32) -> RuleId;

    pub fn marpa_r_new(g: *mut MarpaGrammar) -> *mut MarpaRecce;
    pub fn marpa_r_unref(r: *mut MarpaRecce);

    pub fn marpa_r_start_input(recce: *mut MarpaRecce) -> i32;
    pub fn marpa_r_alternative(recce: *mut MarpaRecce, token_id: SymbolId, value: i32,
                                                                           length: i32) -> i32;
    pub fn marpa_r_earleme_complete(recce: *mut MarpaRecce) -> Earleme;
    pub fn marpa_r_latest_earley_set(recce: *mut MarpaRecce) -> EarleySetId;

    pub fn marpa_b_new(recce: *mut MarpaRecce, earley_set_id: EarleySetId) -> *mut MarpaBocage;
    pub fn marpa_b_unref(r: *mut MarpaBocage);

    pub fn marpa_o_new(b: *mut MarpaBocage) -> *mut MarpaOrder;
    pub fn marpa_o_unref(r: *mut MarpaOrder);

    pub fn marpa_t_new(o: *mut MarpaOrder) -> *mut MarpaTree;
    pub fn marpa_t_unref(r: *mut MarpaTree);
    pub fn marpa_t_next(t: *mut MarpaTree) -> i32;

    pub fn marpa_v_new(t: *mut MarpaTree) -> *mut MarpaValue;
    pub fn marpa_v_unref(v: *mut MarpaValue);

    pub fn marpa_v_step(v: *mut MarpaValue) -> Step;
    pub fn marpa_v_rule_is_valued_set(v: *mut MarpaValue, rule_id: RuleId, value: i32) -> i32;
    pub fn marpa_v_symbol_is_valued_set(v: *mut MarpaValue, sym_id: SymbolId, value: i32) -> i32;
 
    // static marpa_major_version: i32;
    // static marpa_minor_version: i32;
    // static marpa_micro_version: i32;
}