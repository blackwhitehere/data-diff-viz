//! Axum router that serves the visualization. Routes:
//! - `GET /`              — embedded HTML/JS frontend
//! - `GET /api/datasets`  — list of available datasets + default id
//! - `GET /api/meta`      — grid + alignment metadata for ?dataset=ID
//! - `GET /api/tile`      — tile bytes (one byte per cell, status code)
//! - `GET /api/cell`      — JSON pair of values from both dataframes

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Json, Router,
};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use crate::align::AlignmentReport;
use crate::compare::{compare_f64, Tolerance};
use crate::dataset::{Dataset, DatasetInfo};
use crate::grid::{CellStatus, StatusCounts};
use crate::ui::INDEX_HTML;

#[derive(Clone)]
pub struct AppState {
    pub datasets: Arc<Vec<Dataset>>,
    pub default_id: String,
    pub tol: Tolerance,
}

impl AppState {
    pub fn dataset(&self, id: &str) -> Option<&Dataset> {
        self.datasets.iter().find(|d| d.id == id)
    }
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/datasets", get(api_datasets))
        .route("/api/meta", get(api_meta))
        .route("/api/tile", get(api_tile))
        .route("/api/cell", get(api_cell))
        .with_state(state)
}

pub async fn serve(addr: SocketAddr, state: AppState) -> Result<()> {
    let app = build_router(state);
    let listener = TcpListener::bind(addr).await?;
    log::info!("listening on http://{}", addr);
    println!("data-diff-viz listening on http://{addr}");
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

async fn index() -> impl IntoResponse {
    Html(INDEX_HTML)
}

#[derive(Serialize)]
struct DatasetsResponse {
    datasets: Vec<DatasetInfo>,
    default_id: String,
}

async fn api_datasets(State(state): State<AppState>) -> Json<DatasetsResponse> {
    Json(DatasetsResponse {
        datasets: state.datasets.iter().map(|d| d.info()).collect(),
        default_id: state.default_id.clone(),
    })
}

#[derive(Deserialize)]
struct DatasetSelect {
    #[serde(default)]
    dataset: Option<String>,
}

fn pick_dataset<'a>(
    state: &'a AppState,
    sel: &DatasetSelect,
) -> Result<&'a Dataset, (StatusCode, String)> {
    let id = sel.dataset.as_deref().unwrap_or(&state.default_id);
    state
        .dataset(id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("unknown dataset {id}")))
}

#[derive(Serialize)]
struct MetaResponse {
    dataset_id: String,
    rows: usize,
    cols: usize,
    tile_size: usize,
    levels: usize,
    column_names: Vec<String>,
    status_counts: StatusCounts,
    alignment: AlignmentReport,
    rel_tol: f64,
    abs_tol: f64,
}

async fn api_meta(
    State(state): State<AppState>,
    Query(sel): Query<DatasetSelect>,
) -> Result<Json<MetaResponse>, (StatusCode, String)> {
    let ds = pick_dataset(&state, &sel)?;
    let base = ds.pyramid.level(0).expect("level 0 exists");
    Ok(Json(MetaResponse {
        dataset_id: ds.id.clone(),
        rows: base.rows,
        cols: base.cols,
        tile_size: ds.pyramid.tile_size,
        levels: ds.pyramid.num_levels(),
        column_names: ds.display_columns.as_ref().clone(),
        status_counts: ds.status_counts,
        alignment: ds.report.as_ref().clone(),
        rel_tol: state.tol.rel_tol,
        abs_tol: state.tol.abs_tol,
    }))
}

#[derive(Deserialize)]
struct TileQuery {
    #[serde(default)]
    dataset: Option<String>,
    level: usize,
    tx: usize,
    ty: usize,
}

async fn api_tile(
    State(state): State<AppState>,
    Query(q): Query<TileQuery>,
) -> Result<Response, (StatusCode, String)> {
    let sel = DatasetSelect { dataset: q.dataset };
    let ds = pick_dataset(&state, &sel)?;
    let bytes = ds.pyramid.tile_bytes(q.level, q.tx, q.ty);
    Ok(([(header::CONTENT_TYPE, "application/octet-stream")], bytes).into_response())
}

#[derive(Deserialize)]
struct CellQuery {
    #[serde(default)]
    dataset: Option<String>,
    row: usize,
    col: usize,
}

#[derive(Serialize)]
struct CellResponse {
    row: usize,
    col: usize,
    column_name: String,
    status: &'static str,
    left: serde_json::Value,
    right: serde_json::Value,
    abs_diff: Option<f64>,
    rel_diff: Option<f64>,
}

async fn api_cell(
    State(state): State<AppState>,
    Query(q): Query<CellQuery>,
) -> Result<Json<CellResponse>, (StatusCode, String)> {
    let sel = DatasetSelect { dataset: q.dataset };
    let ds = pick_dataset(&state, &sel)?;
    let columns = ds.display_columns.as_ref();
    if q.col >= columns.len() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("col {} out of range", q.col),
        ));
    }
    let column_name = columns[q.col].clone();

    let left_val = fetch_cell(&ds.left, &column_name, q.row);
    let right_val = fetch_cell(&ds.right, &column_name, q.row);

    let (status, abs_diff, rel_diff) = classify_cell(&left_val, &right_val, state.tol);

    Ok(Json(CellResponse {
        row: q.row,
        col: q.col,
        column_name,
        status: status.as_str(),
        left: any_to_json(left_val),
        right: any_to_json(right_val),
        abs_diff,
        rel_diff,
    }))
}

/// Fetch an `AnyValue` for a cell, or `AnyValue::Null` if the row or column
/// is out of range.
fn fetch_cell<'a>(df: &'a DataFrame, column: &str, row: usize) -> AnyValue<'a> {
    let Ok(col) = df.column(column) else {
        return AnyValue::Null;
    };
    let series = col.as_materialized_series();
    if row >= series.len() {
        return AnyValue::Null;
    }
    series.get(row).unwrap_or(AnyValue::Null)
}

fn classify_cell(
    left: &AnyValue<'_>,
    right: &AnyValue<'_>,
    tol: Tolerance,
) -> (CellStatus, Option<f64>, Option<f64>) {
    let l_null = matches!(left, AnyValue::Null);
    let r_null = matches!(right, AnyValue::Null);
    if l_null && r_null {
        return (CellStatus::Equal, None, None);
    }
    if l_null != r_null {
        return (CellStatus::Diff, None, None);
    }

    if let (Some(a), Some(b)) = (any_to_f64(left), any_to_f64(right)) {
        let status = compare_f64(Some(a), Some(b), tol);
        let abs = if a.is_finite() && b.is_finite() {
            Some((a - b).abs())
        } else {
            None
        };
        let rel = match abs {
            Some(d) if b.abs() > 0.0 => Some(d / b.abs()),
            _ => None,
        };
        return (status, abs, rel);
    }

    if left == right {
        (CellStatus::Equal, None, None)
    } else {
        (CellStatus::Diff, None, None)
    }
}

fn any_to_f64(v: &AnyValue<'_>) -> Option<f64> {
    match v {
        AnyValue::Float64(x) => Some(*x),
        AnyValue::Float32(x) => Some(*x as f64),
        AnyValue::Int8(x) => Some(*x as f64),
        AnyValue::Int16(x) => Some(*x as f64),
        AnyValue::Int32(x) => Some(*x as f64),
        AnyValue::Int64(x) => Some(*x as f64),
        AnyValue::UInt8(x) => Some(*x as f64),
        AnyValue::UInt16(x) => Some(*x as f64),
        AnyValue::UInt32(x) => Some(*x as f64),
        AnyValue::UInt64(x) => Some(*x as f64),
        _ => None,
    }
}

fn any_to_json(v: AnyValue<'_>) -> serde_json::Value {
    use serde_json::Value;
    match v {
        AnyValue::Null => Value::Null,
        AnyValue::Boolean(b) => Value::Bool(b),
        AnyValue::Int8(n) => Value::from(n),
        AnyValue::Int16(n) => Value::from(n),
        AnyValue::Int32(n) => Value::from(n),
        AnyValue::Int64(n) => Value::from(n),
        AnyValue::UInt8(n) => Value::from(n),
        AnyValue::UInt16(n) => Value::from(n),
        AnyValue::UInt32(n) => Value::from(n),
        AnyValue::UInt64(n) => Value::from(n),
        AnyValue::Float32(f) => json_from_f64(f as f64),
        AnyValue::Float64(f) => json_from_f64(f),
        AnyValue::String(s) => Value::String(s.to_string()),
        AnyValue::StringOwned(s) => Value::String(s.to_string()),
        other => Value::String(format!("{other}")),
    }
}

fn json_from_f64(f: f64) -> serde_json::Value {
    serde_json::Number::from_f64(f)
        .map(serde_json::Value::Number)
        .unwrap_or_else(|| serde_json::Value::String(f.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tol() -> Tolerance {
        Tolerance::default()
    }

    #[test]
    fn classify_equal_floats() {
        let (s, abs, _) = classify_cell(&AnyValue::Float64(1.0), &AnyValue::Float64(1.0), tol());
        assert_eq!(s, CellStatus::Equal);
        assert_eq!(abs, Some(0.0));
    }

    #[test]
    fn classify_close_floats() {
        let (s, _, _) = classify_cell(
            &AnyValue::Float64(1_000_000.0),
            &AnyValue::Float64(1_000_000.5),
            tol(),
        );
        assert_eq!(s, CellStatus::Close);
    }

    #[test]
    fn classify_null_vs_null_equal() {
        let (s, _, _) = classify_cell(&AnyValue::Null, &AnyValue::Null, tol());
        assert_eq!(s, CellStatus::Equal);
    }

    #[test]
    fn classify_null_vs_value_diff() {
        let (s, _, _) = classify_cell(&AnyValue::Null, &AnyValue::Int64(5), tol());
        assert_eq!(s, CellStatus::Diff);
    }

    #[test]
    fn classify_string_equality() {
        let (eq, _, _) = classify_cell(&AnyValue::String("a"), &AnyValue::String("a"), tol());
        assert_eq!(eq, CellStatus::Equal);
        let (df, _, _) = classify_cell(&AnyValue::String("a"), &AnyValue::String("b"), tol());
        assert_eq!(df, CellStatus::Diff);
    }
}
