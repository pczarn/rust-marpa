pub type SymbolId = i32;
pub type RuleId = i32;
pub type EarleySetId = i32;
pub type EarleyItemId = i32;
pub type EarlemeId = i32;

#[derive(Copy)] pub enum MarpaGrammar {}
#[derive(Copy)] pub enum MarpaRecce {}
#[derive(Copy)] pub enum MarpaBocage {}
#[derive(Copy)] pub enum MarpaOrder {}
#[derive(Copy)] pub enum MarpaTree {}

#[derive(Copy)]
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

/// The configuration structure is intended for future extensions. Currently, the only function
/// of the config is to give `Grammar::with_config` a place to put its error code.
#[allow(raw_pointer_derive)]
#[derive(Copy)]
#[repr(C)]
pub struct Config {
    t_is_ok: i32,
    t_error: ErrorCode,
    t_error_str: *const u8,
}

impl Config {
    /// Creates a config initialized to default values.
    pub fn new() -> Config {
        let mut cfg = Config {
            t_is_ok: 0,
            t_error: ErrorCode::ErrNone,
            t_error_str: 0usize as *const _
        };
        unsafe { marpa_c_init(&mut cfg); }
        cfg
    }

    pub fn error_code(&self) -> ErrorCode {
        self.t_error
    }
}

#[derive(Copy, Debug)]
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

#[derive(Copy, Debug)]
#[repr(i32)]
pub enum ErrorCode {
    ErrNone = 0,
    AhfaIxNegative = 1,
    AhfaIxOob = 2,
    AndidNegative = 3,
    AndidNotInOr = 4,
    AndixNegative = 5,
    BadSeparator = 6,
    BocageIterationExhausted = 7,
    CountedNullable = 8,
    Development = 9,
    DuplicateAndNode = 10,
    DuplicateRule = 11,
    DuplicateToken = 12,
    YimCount = 13,
    YimIdInvalid = 14,
    EventIxNegative = 15,
    EventIxOob = 16,
    GrammarHasCycle = 17,
    InaccessibleToken = 18,
    Internal = 19,
    InvalidAhfaId = 20,
    InvalidAimid = 21,
    InvalidBoolean = 22,
    InvalidIrlid = 23,
    InvalidNsyid = 24,
    InvalidLocation = 25,
    InvalidRuleId = 26,
    InvalidStartSymbol = 27,
    InvalidSymbolId = 28,
    IAmNotOk = 29,
    MajorVersionMismatch = 30,
    MicroVersionMismatch = 31,
    MinorVersionMismatch = 32,
    NookidNegative = 33,
    NotPrecomputed = 34,
    NotTracingCompletionLinks = 35,
    NotTracingLeoLinks = 36,
    NotTracingTokenLinks = 37,
    NoAndNodes = 38,
    NoEarleySetAtLocation = 39,
    NoOrNodes = 40,
    NoParse = 41,
    NoRules = 42,
    NoStartSymbol = 43,
    NoTokenExpectedHere = 44,
    NoTraceYim = 45,
    NoTraceYs = 46,
    NoTracePim = 47,
    NoTraceSrcl = 48,
    NullingTerminal = 49,
    OrderFrozen = 50,
    OridNegative = 51,
    OrAlreadyOrdered = 52,
    ParseExhausted = 53,
    ParseTooLong = 54,
    PimIsNotLim = 55,
    PointerArgNull = 56,
    Precomputed = 57,
    ProgressReportExhausted = 58,
    ProgressReportNotStarted = 59,
    RecceNotAcceptingInput = 60,
    RecceNotStarted = 61,
    RecceStarted = 62,
    RhsIxNegative = 63,
    RhsIxOob = 64,
    RhsTooLong = 65,
    SequenceLhsNotUnique = 66,
    SourceTypeIsAmbiguous = 67,
    SourceTypeIsCompletion = 68,
    SourceTypeIsLeo = 69,
    SourceTypeIsNone = 70,
    SourceTypeIsToken = 71,
    SourceTypeIsUnknown = 72,
    StartNotLhs = 73,
    SymbolValuedConflict = 74,
    TerminalIsLocked = 75,
    TokenIsNotTerminal = 76,
    TokenLengthLeZero = 77,
    TokenTooLong = 78,
    TreeExhausted = 79,
    TreePaused = 80,
    UnexpectedTokenId = 81,
    UnproductiveStart = 82,
    ValuatorInactive = 83,
    ValuedIsLocked = 84,
    RankTooLow = 85,
    RankTooHigh = 86,
    SymbolIsNulling = 87,
    SymbolIsUnused = 88,
    NoSuchRuleId = 89,
    NoSuchSymbolId = 90,
    BeforeFirstTree = 91,
    SymbolIsNotCompletionEvent = 92,
    SymbolIsNotNulledEvent = 93,
    SymbolIsNotPredictionEvent = 94,
    RecceIsInconsistent = 95,
    InvalidAssertionId = 96,
    NoSuchAssertionId = 97,
    HeadersDoNotMatch = 98,
}

#[link(name = "marpa")]
extern {
    pub fn marpa_c_init(config: *mut Config) -> i32;

    pub fn marpa_g_new(config: *mut Config) -> *mut MarpaGrammar;
    pub fn marpa_g_ref(grammar: *mut MarpaGrammar) -> *mut MarpaGrammar;

    pub fn marpa_g_unref(grammar: *mut MarpaGrammar);
    pub fn marpa_g_precompute(grammar: *mut MarpaGrammar) -> i32;
    pub fn marpa_g_is_precomputed(grammar: *mut MarpaGrammar) -> i32;
    pub fn marpa_g_symbol_new(grammar: *mut MarpaGrammar) -> SymbolId;
    pub fn marpa_g_start_symbol(grammar: *mut MarpaGrammar) -> SymbolId;
    pub fn marpa_g_start_symbol_set(grammar: *mut MarpaGrammar, sym: SymbolId) -> SymbolId;
    pub fn marpa_g_rule_new(g: *mut MarpaGrammar, lhs_id: SymbolId, rhs_ids: *const SymbolId,
                                                                    length: i32) -> RuleId;
    pub fn marpa_g_sequence_new(g: *mut MarpaGrammar, lhs_id: SymbolId, rhs_id: SymbolId,
                                                                        sep_id: SymbolId,
                                                                        min: i32,
                                                                        flags: i32) -> RuleId;

    pub fn marpa_g_error(grammar: *mut MarpaGrammar, p_error_string: *const *const u8) -> ErrorCode;
    pub fn marpa_g_error_clear(grammar: *mut MarpaGrammar) -> ErrorCode;

    pub fn marpa_r_new(g: *mut MarpaGrammar) -> *mut MarpaRecce;
    pub fn marpa_r_unref(r: *mut MarpaRecce);

    pub fn marpa_r_start_input(recce: *mut MarpaRecce) -> i32;
    pub fn marpa_r_alternative(recce: *mut MarpaRecce, token_id: SymbolId,
                                                       value: i32,
                                                       length: i32) -> ErrorCode;
    pub fn marpa_r_earleme_complete(recce: *mut MarpaRecce) -> EarlemeId;
    pub fn marpa_r_latest_earley_set(recce: *mut MarpaRecce) -> EarleySetId;
    pub fn marpa_r_terminals_expected(recce: *mut MarpaRecce, ary: *mut SymbolId) -> i32;

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