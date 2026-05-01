//! A `Dataset` bundles the inputs and pre-computed views for a single
//! left/right pair: the (possibly aligned) dataframes, display columns,
//! pyramid, and alignment report. The server holds one or more of these
//! and the UI lets the user switch between them.

use std::sync::Arc;

use anyhow::Result;
use polars::prelude::*;
use serde::Serialize;

use crate::align::{detect_and_align, AlignmentReport};
use crate::compare::{build_status_grid, Tolerance};
use crate::grid::{Pyramid, StatusCounts};

#[derive(Clone)]
pub struct Dataset {
    pub id: String,
    pub label: String,
    pub left: Arc<DataFrame>,
    pub right: Arc<DataFrame>,
    pub display_columns: Arc<Vec<String>>,
    pub pyramid: Arc<Pyramid>,
    pub status_counts: StatusCounts,
    pub report: Arc<AlignmentReport>,
}

#[derive(Serialize, Clone)]
pub struct DatasetInfo {
    pub id: String,
    pub label: String,
    pub rows: usize,
    pub cols: usize,
}

impl Dataset {
    pub fn info(&self) -> DatasetInfo {
        let base = self.pyramid.level(0).expect("level 0 exists");
        DatasetInfo {
            id: self.id.clone(),
            label: self.label.clone(),
            rows: base.rows,
            cols: base.cols,
        }
    }
}

/// Build a `Dataset` from raw left/right dataframes by running
/// `detect_and_align` → `build_status_grid` → `Pyramid::build`.
pub fn build_dataset(
    id: impl Into<String>,
    label: impl Into<String>,
    left: DataFrame,
    right: DataFrame,
    tol: Tolerance,
    tile_size: usize,
) -> Result<Dataset> {
    let id = id.into();
    let label = label.into();
    let (left, right, report) = detect_and_align(left, right, tol)?;
    let display_columns = report.display_columns.clone();
    let grid = build_status_grid(&left, &right, &display_columns, tol);
    let status_counts = grid.status_counts();
    let pyramid = Pyramid::build(grid, tile_size);
    Ok(Dataset {
        id,
        label,
        left: Arc::new(left),
        right: Arc::new(right),
        display_columns: Arc::new(display_columns),
        pyramid: Arc::new(pyramid),
        status_counts,
        report: Arc::new(report),
    })
}
