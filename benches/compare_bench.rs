//! Microbenchmark for the comparison + pyramid pipeline on a synthetic
//! dataframe. Run with `cargo bench`.

use criterion::{criterion_group, criterion_main, Criterion};
use polars::prelude::*;

use data_diff_viz::compare::{build_status_grid, Tolerance};
use data_diff_viz::grid::Pyramid;

fn make_df(rows: usize, cols: usize) -> DataFrame {
    let mut columns: Vec<Column> = Vec::with_capacity(cols);
    for c in 0..cols {
        let v: Vec<f64> = (0..rows).map(|r| (r as f64) + (c as f64) * 0.1).collect();
        columns.push(Column::new(format!("c{c}").as_str().into(), v));
    }
    DataFrame::new(columns).unwrap()
}

fn perturb(df: &DataFrame) -> DataFrame {
    // Add 1e-9 to every cell — keeps everything within close-tolerance.
    let mut out = df.clone();
    let names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    for name in &names {
        let s = out.column(name).unwrap().as_materialized_series().clone();
        let chunked = s.f64().unwrap();
        let new: Vec<f64> = chunked
            .into_iter()
            .map(|v| v.unwrap_or(0.0) + 1e-9)
            .collect();
        out.with_column(Column::new(name.as_str().into(), new))
            .unwrap();
    }
    out
}

fn bench_pipeline(c: &mut Criterion) {
    let rows = 100_000;
    let cols = 8;
    let left = make_df(rows, cols);
    let right = perturb(&left);
    let columns: Vec<String> = (0..cols).map(|i| format!("c{i}")).collect();
    let tol = Tolerance::default();

    c.bench_function("build_status_grid 100k x 8", |b| {
        b.iter(|| {
            let g = build_status_grid(&left, &right, &columns, tol);
            criterion::black_box(g);
        });
    });

    let grid = build_status_grid(&left, &right, &columns, tol);
    c.bench_function("build_pyramid 100k x 8 tile=256", |b| {
        b.iter(|| {
            let p = Pyramid::build(grid.clone(), 256);
            criterion::black_box(p);
        });
    });
}

criterion_group!(benches, bench_pipeline);
criterion_main!(benches);
