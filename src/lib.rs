#![crate_name = "marpa"]

extern crate libc;

pub struct Config {
    t_is_ok: i32,
    t_error: i32,
    t_error_str: *const u8,
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
}

impl Drop for Grammar {
    fn drop(&mut self) {
        unsafe {
            marpa_g_unref(self.g);
        }
    }
}

#[link(name = "marpa")]
extern {
    pub fn marpa_c_init(config: *const Config);

    pub fn marpa_g_new(config: *const Config) -> MarpaGrammar;
    pub fn marpa_g_ref(grammar: MarpaGrammar) -> MarpaGrammar;
    pub fn marpa_g_unref(grammar: MarpaGrammar);
    //pub fn 
 
    static marpa_major_version: i32;
    static marpa_minor_version: i32;
    static marpa_micro_version: i32;
}

#[test]
fn test_simple_first() {
    let cfg = Config { t_is_ok: 0, t_error: 0, t_error_str: 0u as *const _ };
    unsafe {
        marpa_c_init(&cfg);
        Grammar::new(cfg);
    }
}
