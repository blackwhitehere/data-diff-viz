//! data-diff-viz: visual cell-by-cell diff for two parquet dataframes.
//!
//! The pipeline is:
//! `load` → `align` → `compare` → `grid` (pyramid) → `server` (Axum + Canvas frontend).

pub mod align;
pub mod compare;
pub mod dataset;
pub mod grid;
pub mod load;
pub mod logger;
pub mod server;
pub mod ui;
