//! Generate a directory of demo dataset pairs for `just demo`. Each
//! dataset is written to its own subdirectory of the target directory:
//!
//! - `01-matching/`         — identical frames (default selection)
//! - `02-small-diffs/`      — ~1% close + ~0.1% diff perturbations
//! - `03-shuffled/`         — right is left with rows shuffled
//! - `04-extra-columns/`    — each side has one column the other lacks
//! - `05-size-mismatch/`    — right has fewer rows than left

use std::fs::{create_dir_all, File};
use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;
use polars::prelude::*;

#[path = "rand_lite.rs"]
mod rand_lite;

#[derive(Parser, Debug)]
struct Args {
    /// Output directory; one subdir is written per dataset.
    out_dir: PathBuf,
    #[arg(long, default_value_t = 50_000)]
    rows: usize,
    #[arg(long, default_value_t = 8)]
    cols: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    create_dir_all(&args.out_dir)?;

    let n = args.rows;
    let c = args.cols;

    let mut rng = rand_lite::SimpleRng::new(42);
    let base = build_base(n, c, &mut rng);

    write_pair(&args.out_dir.join("01-matching"), &base, &base)?;

    let small_diffs = build_small_diffs(&base, &mut rng)?;
    write_pair(&args.out_dir.join("02-small-diffs"), &base, &small_diffs)?;

    let shuffled = build_shuffled(&base, &mut rng)?;
    write_pair(&args.out_dir.join("03-shuffled"), &base, &shuffled)?;

    let (left_xc, right_xc) = build_extra_columns(&base)?;
    write_pair(&args.out_dir.join("04-extra-columns"), &left_xc, &right_xc)?;

    let size_mismatch = base.head(Some(n.saturating_sub(n / 5)));
    write_pair(
        &args.out_dir.join("05-size-mismatch"),
        &base,
        &size_mismatch,
    )?;

    println!(
        "wrote 5 demo datasets ({} × {} cells base) to {}",
        n,
        c + 1,
        args.out_dir.display()
    );
    Ok(())
}

fn build_base(rows: usize, cols: usize, rng: &mut rand_lite::SimpleRng) -> DataFrame {
    let ids: Vec<i64> = (0..rows as i64).collect();
    let mut columns: Vec<Column> = vec![Column::new("id".into(), &ids)];
    for ci in 0..cols {
        let v: Vec<f64> = (0..rows).map(|_| rng.next_f64() * 1000.0).collect();
        columns.push(Column::new(format!("v{ci}").as_str().into(), v));
    }
    DataFrame::new(columns).expect("base df build")
}

fn build_small_diffs(base: &DataFrame, rng: &mut rand_lite::SimpleRng) -> Result<DataFrame> {
    let mut out = base.clone();
    let names: Vec<String> = base
        .get_column_names()
        .iter()
        .filter(|n| n.as_str() != "id")
        .map(|n| n.to_string())
        .collect();
    for name in &names {
        let s = out.column(name)?.as_materialized_series().clone();
        let mut values: Vec<Option<f64>> = s.f64()?.clone().into_iter().collect();
        for x in values.iter_mut().flatten() {
            let r = rng.next_f64();
            if r < 0.01 {
                *x += 1e-9; // close
            } else if r < 0.011 {
                *x += 100.0; // diff
            }
        }
        out.with_column(Column::new(name.as_str().into(), values))?;
    }
    Ok(out)
}

fn build_shuffled(base: &DataFrame, rng: &mut rand_lite::SimpleRng) -> Result<DataFrame> {
    let n = base.height();
    let mut order: Vec<u32> = (0..n as u32).collect();
    for i in (1..order.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        order.swap(i, j);
    }
    let idx = IdxCa::from_vec(
        "idx".into(),
        order.into_iter().map(|x| x as IdxSize).collect(),
    );
    Ok(base.take(&idx)?)
}

fn build_extra_columns(base: &DataFrame) -> Result<(DataFrame, DataFrame)> {
    let mut left = base.clone();
    let mut right = base.clone();
    let n = base.height();

    let only_left: Vec<i64> = (0..n as i64).map(|i| i * 2).collect();
    let only_right: Vec<f64> = (0..n).map(|i| i as f64 / 7.0).collect();

    left.with_column(Column::new("only_left".into(), only_left))?;
    right.with_column(Column::new("only_right".into(), only_right))?;
    Ok((left, right))
}

fn write_pair(dir: &Path, left: &DataFrame, right: &DataFrame) -> Result<()> {
    create_dir_all(dir)?;
    write_parquet(&dir.join("left.parquet"), &mut left.clone())?;
    write_parquet(&dir.join("right.parquet"), &mut right.clone())?;
    Ok(())
}

fn write_parquet(path: &Path, df: &mut DataFrame) -> Result<()> {
    let file = File::create(path)?;
    ParquetWriter::new(file).finish(df)?;
    Ok(())
}
