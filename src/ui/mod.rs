//! Embedded HTML/JS frontend.

/// Inlined as a `&'static str` so Cargo recompiles when `index.html` changes.
pub const INDEX_HTML: &str = include_str!("index.html");
