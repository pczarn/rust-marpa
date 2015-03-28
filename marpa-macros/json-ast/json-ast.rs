#![feature(plugin, unboxed_closures, associated_types)]
#![plugin(regex_macros, marpa_macros, regex_scanner)]

extern crate marpa_macros;
extern crate regex_scanner;

extern crate marpa;
extern crate regex;

#[no_link]
extern crate regex_macros;

extern crate time;

use std::collections::BTreeMap;
use std::old_io::File;
use std::os::args;

use time::precise_time_ns;

use self::Json::*;

#[derive(Debug)]
enum Json {
    I64(i64),
    U64(u64),
    F64(f64),
    JsonString(String),
    Boolean(bool),
    Array(Vec<Json>),
    Object(BTreeMap<String, Json>),
    Null,
}

fn main() {
    let mut json = grammar! {
        start ::=
            o:object -> Json { o }
            | a:array -> _ { a } ;

        object ::=
            r"\{" m:members r"\}" -> Json { Object(m) }
            | r"\{" r"\}" -> _ { Object(BTreeMap::new()) } ;

        members ::=
            s:string ":" v:value -> BTreeMap<String, Json> {
                let mut map = BTreeMap::new();
                map.insert(s, v);
                map
            }
            | (mut map):members "," s:string ":" v:value -> _ {
                map.insert(s, v);
                map
            } ;


        array ::=
            r"\[" m:array_members r"\]" -> Json { Array(m) }
            | r"\[" r"\]" -> _ { Array(vec![]) } ;

        array_members ::=
            v:value -> Vec<Json> {
                let mut values = Vec::new();
                values.push(v); // macros won't work here
                values
            }
            | (mut values):array_members "," v:value -> _ {
                values.push(v);
                values
            } ;

        value ::=
            s:string -> Json { JsonString(s) }
            | n:number -> _ { n }
            | o:object -> _ { o }
            | a:array -> _ { a }
            | "true" -> _ { Boolean(true) }
            | "false" -> _ { Boolean(false) }
            | "null" -> _ { Null } ;

        number ::=
            "-" i:integer fp:frac_part -> Json {
                F64(-(i as f64 + fp))
            }
            | "-" i:integer -> _ {
                I64(-(i as i64))
            }
            | i:integer fp:frac_part -> _ {
                F64(i as f64 + fp)
            }
            | i:integer -> _ {
                U64(i)
            } ;

        frac_part ::= f:r"\.\d+(?:[eE][-+]?)?\d*" -> f64 { f.parse().unwrap() } ;

        integer ::= i:r"\d+" -> u64 { i.parse().unwrap() } ;

        string ~ s:r#""(?:\\"|[^"])*""# -> String { s.slice(1, s.len() - 1).to_string() } ;

        discard ~ r"\s" ;
    };

    for json_file in args().iter().skip(1) {
        let contents = File::open(&Path::new(json_file.as_slice())).read_to_string().unwrap();

        let mut iter = json.parses_iter(&contents[]);

        let ns = precise_time_ns();

        for ast in iter {

            // println!("{:?}", ast);
        }
        println!("elapsed {}", (precise_time_ns() - ns) as f64 / 1_000_000_000f64);
    }

    // for ast in json.parses_iter(r#"{ "a": [123.123E-100, { "b\"c": null }] }"#) {
    //     println!("{:?}", ast);
    // }
}

        // start ::= collection ;

        // collection ::=
        //     r"\{" map:[ string ":" value ]{","}* r"\}" -> Json {
        //         Object(map)
        //     }
        //     | r"\[" values:[ value ]{","}* r"\]" -> Json {
        //         Array(values)
        //     } ;

        // value ::=
        //     s:string -> Json { JsonString(s) }
        //     | "true" -> _ { Boolean(true) }
        //     | "false" -> _ { Boolean(false) }
        //     | "null" -> _ { Null }
        //     | collection
        //     | signed
        //     | unsigned ;

        // signed ::=
        //     "-" F64(fp):unsigned -> Json {
        //         F64(-fp)
        //     }
        //     | "-" U64(i):unsigned -> _ {
        //         I64(-i as i64)
        //     } ;

        // unsigned ::=
        //     Some(i):integer Some(fp):frac_part -> Json {
        //         F64(i as f64 + fp)
        //     }
        //     | Some(i):integer -> _ {
        //         U64(i)
        //     } ;

        // frac_part ::= f:r"\.\d+(?:[eE][-+]?)?\d*" -> Option<f64> { f.parse() } ;

        // integer ::= i:r"\d+" -> Option<u64> { i.parse() } ;

        // string ~ "\"" s:r#"(?:\\"|[^"])*"# "\"" -> String { s.to_string() } ;

        // discard ~ r"\s" ;
