//! End-to-end test exercising the full pipeline against an in-process
//! Axum router. We construct small parquets, run load → align → compare →
//! pyramid → server, and assert the visible HTTP responses.

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use polars::prelude::*;
use tempfile::TempDir;
use tower::util::ServiceExt;

use data_diff_viz::compare::Tolerance;
use data_diff_viz::dataset::build_dataset;
use data_diff_viz::grid::CellStatus;
use data_diff_viz::load::read_parquet;
use data_diff_viz::server::{build_router, AppState};

fn write_parquet(path: &PathBuf, mut df: DataFrame) {
    let file = File::create(path).unwrap();
    ParquetWriter::new(file).finish(&mut df).unwrap();
}

fn single_state(left: DataFrame, right: DataFrame, tol: Tolerance) -> AppState {
    let ds = build_dataset("default", "default", left, right, tol, 4).unwrap();
    AppState {
        datasets: Arc::new(vec![ds]),
        default_id: "default".to_string(),
        tol,
    }
}

async fn body_bytes(resp: axum::response::Response) -> Vec<u8> {
    resp.into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec()
}

#[tokio::test]
async fn full_pipeline_meta_tile_cell() {
    let dir = TempDir::new().unwrap();
    let left_path = dir.path().to_path_buf().join("left.parquet");
    let right_path = dir.path().to_path_buf().join("right.parquet");

    let left = df! {
        "id" => [1i64, 2, 3, 4],
        "v"  => [1.0f64, 2.0, 3.0, 4.0],
    }
    .unwrap();
    let right = df! {
        "id" => [1i64, 2, 3, 4],
        "v"  => [1.0f64, 2.0 + 1e-9, 99.0, 4.0],
    }
    .unwrap();
    write_parquet(&left_path, left);
    write_parquet(&right_path, right);

    let left = read_parquet(&left_path).unwrap();
    let right = read_parquet(&right_path).unwrap();
    let tol = Tolerance::default();
    let state = single_state(left, right, tol);
    let app = build_router(state);

    // /api/meta (with explicit dataset)
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/meta?dataset=default")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = body_bytes(resp).await;
    let meta: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(meta["rows"], 4);
    assert_eq!(meta["cols"], 2);
    assert_eq!(meta["dataset_id"], "default");
    assert!(meta["levels"].as_u64().unwrap() >= 1);
    assert_eq!(meta["column_names"][0], "id");
    assert_eq!(meta["column_names"][1], "v");
    assert_eq!(meta["status_counts"]["equal"], 6);
    assert_eq!(meta["status_counts"]["close"], 1);
    assert_eq!(meta["status_counts"]["diff"], 1);
    assert_eq!(meta["status_counts"]["missing"], 0);

    // /api/meta (no dataset query — should fall back to default)
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/meta")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // /api/tile
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/tile?dataset=default&level=0&tx=0&ty=0")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = body_bytes(resp).await;
    assert_eq!(bytes.len(), 16);
    let at = |row: usize, col: usize| bytes[row * 4 + col];
    assert_eq!(at(0, 0), CellStatus::Equal as u8);
    assert_eq!(at(1, 1), CellStatus::Close as u8);
    assert_eq!(at(2, 1), CellStatus::Diff as u8);
    assert_eq!(at(3, 0), CellStatus::Equal as u8);
    assert_eq!(at(0, 2), CellStatus::Missing as u8);
    assert_eq!(at(0, 3), CellStatus::Missing as u8);

    // /api/cell — drill into the Diff at (row=2, col=1)
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/cell?dataset=default&row=2&col=1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = body_bytes(resp).await;
    let cell: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(cell["column_name"], "v");
    assert_eq!(cell["status"], "diff");
    assert_eq!(cell["left"], 3.0);
    assert_eq!(cell["right"], 99.0);
}

#[tokio::test]
async fn datasets_endpoint_lists_all_with_default() {
    let tol = Tolerance::default();
    let identical = df! { "x" => [1i64, 2, 3] }.unwrap();
    let identical2 = identical.clone();
    let differing_left = df! { "x" => [1i64, 2, 3] }.unwrap();
    let differing_right = df! { "x" => [1i64, 2, 99] }.unwrap();

    let ds1 = build_dataset("matching", "Matching", identical, identical2, tol, 4).unwrap();
    let ds2 = build_dataset(
        "diffs",
        "With diffs",
        differing_left,
        differing_right,
        tol,
        4,
    )
    .unwrap();
    let state = AppState {
        datasets: Arc::new(vec![ds1, ds2]),
        default_id: "matching".to_string(),
        tol,
    };
    let app = build_router(state);

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/datasets")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = body_bytes(resp).await;
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["default_id"], "matching");
    let ids: Vec<String> = v["datasets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(ids, vec!["matching", "diffs"]);

    // Switching datasets via the query param returns different tile bytes:
    // matching dataset → all Equal (0); diffs dataset → tail cell is Diff (2).
    let matching_tile = body_bytes(
        app.clone()
            .oneshot(
                Request::builder()
                    .uri("/api/tile?dataset=matching&level=0&tx=0&ty=0")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap(),
    )
    .await;
    let diffs_tile = body_bytes(
        app.clone()
            .oneshot(
                Request::builder()
                    .uri("/api/tile?dataset=diffs&level=0&tx=0&ty=0")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap(),
    )
    .await;
    // tile_size=4, layout row-major. With a 3-row × 1-col dataset, cells at
    // (row=R, col=0) live at bytes[R*4+0]; col >= 1 are Missing on both.
    let at = |bytes: &[u8], row: usize, col: usize| bytes[row * 4 + col];
    assert_eq!(at(&matching_tile, 0, 0), CellStatus::Equal as u8);
    assert_eq!(at(&matching_tile, 1, 0), CellStatus::Equal as u8);
    assert_eq!(at(&matching_tile, 2, 0), CellStatus::Equal as u8);
    assert_eq!(at(&diffs_tile, 0, 0), CellStatus::Equal as u8);
    assert_eq!(at(&diffs_tile, 1, 0), CellStatus::Equal as u8);
    assert_eq!(at(&diffs_tile, 2, 0), CellStatus::Diff as u8);
}

#[tokio::test]
async fn unknown_dataset_returns_404() {
    let tol = Tolerance::default();
    let l = df! { "x" => [1i64] }.unwrap();
    let r = df! { "x" => [1i64] }.unwrap();
    let state = single_state(l, r, tol);
    let app = build_router(state);

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/meta?dataset=does-not-exist")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn alignment_warning_round_trips_to_meta() {
    let left = df! {
        "id" => [1i64, 2, 3, 4, 5],
        "v"  => [10.0f64, 20.0, 30.0, 40.0, 50.0],
    }
    .unwrap();
    let right = df! {
        "id" => [3i64, 1, 5, 2, 4],
        "v"  => [30.0f64, 10.0, 50.0, 20.0, 40.0],
    }
    .unwrap();
    let tol = Tolerance::default();
    let state = single_state(left, right, tol);
    let app = build_router(state);

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/meta")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = body_bytes(resp).await;
    let meta: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let sorted_by = meta["alignment"]["rows_sorted_by"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    assert!(sorted_by.contains(&"id".to_string()));
    assert!(meta["alignment"]["final_match_rate"].as_f64().unwrap() >= 0.95);
}

#[tokio::test]
async fn cell_endpoint_returns_400_for_out_of_range_col() {
    let left = df! { "x" => [1i64, 2, 3] }.unwrap();
    let right = df! { "x" => [1i64, 2, 3] }.unwrap();
    let state = single_state(left, right, Tolerance::default());
    let app = build_router(state);

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/cell?row=0&col=99")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn index_html_served_at_root() {
    let left = df! { "x" => [1i64] }.unwrap();
    let right = df! { "x" => [1i64] }.unwrap();
    let state = single_state(left, right, Tolerance::default());
    let app = build_router(state);

    let resp = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(body.contains("data-diff-viz"));
    assert!(body.contains("/api/datasets"));
    assert!(body.contains("/api/meta"));
}
