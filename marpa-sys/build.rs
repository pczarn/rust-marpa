extern crate pkg_config;
extern crate gcc;

use std::env;
// use std::fs;
// use std::path::Path;
use std::process::Command;

const LIBMARPA_REPO: &'static str = "git@github.com:jeffreykegler/libmarpa.git";

fn fail(s: &str) -> ! {
    println!("\n\n{}\n\n", s);
    panic!()
}

fn getenv(v: &str) -> Option<String> {
    let r = env::var(v).ok();
    println!("{} = {:?}", v, r);
    r
}

fn getenv_unwrap(v: &str) -> String {
    match getenv(v) {
        Some(s) => s,
        None => fail(&format!("environment variable `{}` not defined", v)),
    }
}

fn main() {
    match pkg_config::find_library("marpa") {
        Ok(_) => return,
        Err(..) => {}
    };

    // let out_dir = getenv_unwrap("OUT_DIR");

    Command::new("git").args(&["clone", "--depth", "5", LIBMARPA_REPO]).status().unwrap();
    env::set_current_dir(&env::current_dir().unwrap().as_path().join("libmarpa")).unwrap();
    Command::new("make").arg("dist").status().unwrap();
    env::set_current_dir(&env::current_dir().unwrap().as_path().join("dist")).unwrap();
    Command::new("./configure").arg("--enable-shared").status().unwrap();
    Command::new("make").status().unwrap();
    // let dst = Path::new(&out_dir[..]);
    // fs::copy(".libs/libmarpa.a", dst.join("libmarpa_c.a")).unwrap();
    // fs::copy(".libs/libmarpa-8.3.0.so", dst.join("libmarpa_c.so")).unwrap();

    // println!("cargo:rustc-link-search=native={}", dst.display());
    // println!("cargo:rustc-link-search=native=marpa-sys/libmarpa/dist/.libs");
    // println!("cargo:rustc-link-lib=dylib=marpa");

    gcc::Config::new().object(".libs/marpa.o")
                      .object(".libs/marpa_obs.o")
                      .object(".libs/marpa_avl.o")
                      .object(".libs/marpa_tavl.o")
                      .object(".libs/marpa_ami.o")
                      .object(".libs/marpa_codes.o")
                      .compile("libmarpa.a");
}
