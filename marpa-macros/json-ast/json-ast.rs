#![feature(plugin, unboxed_closures, associated_types)]

#[plugin]
extern crate "marpa-macros" as marpa_macros;

extern crate marpa;
extern crate regex;

#[plugin]
#[no_link]
extern crate regex_macros;

use std::collections::BTreeMap;

use self::Json::*;

#[derive(Show)]
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

        integer ::= i:r"[1-9]*\d" -> u64 { i.parse().unwrap() } ;

        string ~ s:r#""(?:\\"|[^"])*""# -> String { s.slice(1, s.len() - 1).to_string() } ;

        discard ~ r"\s" ;
    };

    for ast in json.parses_iter(r#"{ "a": [123.123E-100, { "b\"c": null }] }"#) {
        println!("{:?}", ast);
    }
}
