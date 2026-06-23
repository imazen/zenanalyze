//! Coverage tripwire: every public API identifier in the source must be mentioned
//! (as a whole word) in `README.md` — which IS the crate docs (`#![doc =
//! include_str!("../README.md")]`). Adding or renaming a `pub` item without
//! updating the README fails the build.
//!
//! This is the completeness half of "check for oopsies"; the README's `rust`
//! examples being doctests is the other half (they catch a documented example
//! that no longer compiles against the API).

/// Leading identifier of `s` (`[A-Za-z0-9_]*` from the start).
fn lead_ident(s: &str) -> String {
    s.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

/// Public-API identifiers in a source file: `pub fn` / `pub const fn` names,
/// `pub struct` / `pub enum` / `pub mod` names, `pub` field names, and the bare
/// variant names inside a `pub enum` body. Deliberately simple line parsing — the
/// contract crate is tiny and frozen, so this stays trivially auditable.
fn public_idents(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut enum_depth: Option<i32> = None; // brace depth of the current `pub enum` body
    let mut pending_enum = false; // saw `pub enum X` header, awaiting its `{`
    let mut depth: i32 = 0;
    for raw in src.lines() {
        let line = raw.trim();
        let doc_or_attr =
            line.starts_with("//") || line.starts_with("#[") || line.starts_with("#!");

        if let Some(rest) = line.strip_prefix("pub ") {
            let ident = if let Some(r) = rest.strip_prefix("const fn ") {
                Some(lead_ident(r))
            } else if let Some(r) = rest.strip_prefix("fn ") {
                Some(lead_ident(r))
            } else if let Some(r) = rest.strip_prefix("struct ") {
                Some(lead_ident(r))
            } else if let Some(r) = rest.strip_prefix("enum ") {
                pending_enum = true;
                Some(lead_ident(r))
            } else if let Some(r) = rest.strip_prefix("mod ") {
                Some(lead_ident(r))
            } else if rest.starts_with("use ") || rest.starts_with("trait ") {
                None // no re-exports / traits today; surface them in the README if added
            } else {
                Some(lead_ident(rest)) // `pub <field>: Type,`
            };
            if let Some(id) = ident
                && !id.is_empty()
            {
                out.push(id);
            }
        } else if let Some(ed) = enum_depth
            && depth >= ed
            && !doc_or_attr
        {
            let id = lead_ident(line);
            if id.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
                out.push(id); // a variant
            }
        }

        let opens = line.matches('{').count() as i32;
        let closes = line.matches('}').count() as i32;
        if pending_enum && opens > 0 {
            enum_depth = Some(depth + 1);
            pending_enum = false;
        }
        depth += opens - closes;
        if enum_depth.is_some_and(|ed| depth < ed) {
            enum_depth = None;
        }
    }
    out
}

/// Whole-word membership: `ident` occurs in `haystack` flanked by non-identifier
/// characters (so `get` matches `.get(` but not `target`).
fn mentions(haystack: &str, ident: &str) -> bool {
    haystack.match_indices(ident).any(|(i, _)| {
        let before = haystack[..i].chars().next_back();
        let after = haystack[i + ident.len()..].chars().next();
        let boundary = |c: Option<char>| c.is_none_or(|c| !c.is_alphanumeric() && c != '_');
        boundary(before) && boundary(after)
    })
}

#[test]
fn readme_documents_every_public_item() {
    let readme = include_str!("../README.md");

    let mut idents = Vec::new();
    idents.extend(public_idents(include_str!("../src/lib.rs")));
    idents.sort();
    idents.dedup();

    // Sanity: the parser must find the surface, else it's silently broken.
    assert!(
        idents.len() >= 20,
        "parser found only {} public idents — it's broken: {idents:?}",
        idents.len()
    );

    let missing: Vec<&String> = idents.iter().filter(|id| !mentions(readme, id)).collect();
    assert!(
        missing.is_empty(),
        "README.md (the crate docs) does not mention these public API identifiers \
         as whole words:\n  {missing:?}\nDocument them in zenanalyze-api/README.md."
    );
}
