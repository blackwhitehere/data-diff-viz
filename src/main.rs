use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use log::info;

use data_diff_viz::compare::Tolerance;
use data_diff_viz::dataset::{build_dataset, Dataset};
use data_diff_viz::load::read_parquet;
use data_diff_viz::server::{serve, AppState};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Visual cell-by-cell diff for two parquet dataframes"
)]
struct Args {
    /// Path to the left parquet file (positional, single-pair mode).
    left: Option<PathBuf>,

    /// Path to the right parquet file (positional, single-pair mode).
    right: Option<PathBuf>,

    /// Demo mode: directory containing one subdirectory per dataset, each
    /// with `left.parquet` and `right.parquet` files. The UI will show a
    /// dropdown to switch between datasets.
    #[arg(long, conflicts_with_all = ["left", "right"])]
    demo_dir: Option<PathBuf>,

    /// Default dataset id to show first when in --demo-dir mode (defaults
    /// to "matching" if present, otherwise the first by alphabetical order).
    #[arg(long)]
    default_dataset: Option<String>,

    /// Relative tolerance for the numeric "close" check.
    #[arg(long, default_value_t = 1e-5)]
    rel_tol: f64,

    /// Absolute tolerance for the numeric "close" check.
    #[arg(long, default_value_t = 1e-8)]
    abs_tol: f64,

    /// Bind address for the web server.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,

    /// Cells per tile edge.
    #[arg(long, default_value_t = 256)]
    tile_size: usize,
}

fn main() -> Result<()> {
    if let Err(e) = data_diff_viz::logger::init() {
        eprintln!("logger init failed: {e}");
    }
    let args = Args::parse();

    let tol = Tolerance {
        rel_tol: args.rel_tol,
        abs_tol: args.abs_tol,
    };

    let (datasets, default_id) = if let Some(dir) = args.demo_dir.as_ref() {
        load_demo_dir(dir, tol, args.tile_size, args.default_dataset.as_deref())?
    } else {
        let left = args
            .left
            .as_ref()
            .ok_or_else(|| anyhow!("expected LEFT and RIGHT positional args, or --demo-dir"))?;
        let right = args
            .right
            .as_ref()
            .ok_or_else(|| anyhow!("expected RIGHT positional arg"))?;
        let ds = load_single_pair(left, right, tol, args.tile_size)?;
        let id = ds.id.clone();
        (vec![ds], id)
    };

    for ds in &datasets {
        emit_warnings_for_dataset(ds);
    }

    let state = AppState {
        datasets: Arc::new(datasets),
        default_id,
        tol,
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(serve(args.bind, state))
}

fn load_single_pair(
    left_path: &Path,
    right_path: &Path,
    tol: Tolerance,
    tile_size: usize,
) -> Result<Dataset> {
    info!(
        "loading left={} right={}",
        left_path.display(),
        right_path.display()
    );
    let left = read_parquet(left_path).context("reading left parquet")?;
    let right = read_parquet(right_path).context("reading right parquet")?;
    info!(
        "left shape {:?}, right shape {:?}",
        left.shape(),
        right.shape()
    );
    let label = format!(
        "{} vs {}",
        left_path.file_stem().unwrap_or_default().to_string_lossy(),
        right_path.file_stem().unwrap_or_default().to_string_lossy(),
    );
    build_dataset("default", label, left, right, tol, tile_size)
}

/// Scan `dir` for subdirectories that each contain a `left.parquet` and
/// `right.parquet` and build one dataset per subdir.
fn load_demo_dir(
    dir: &Path,
    tol: Tolerance,
    tile_size: usize,
    requested_default: Option<&str>,
) -> Result<(Vec<Dataset>, String)> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading demo directory {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();

    let mut datasets = Vec::new();
    for sub in entries {
        let left_path = sub.join("left.parquet");
        let right_path = sub.join("right.parquet");
        if !left_path.exists() || !right_path.exists() {
            continue;
        }
        let dir_name = sub
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unnamed".to_string());
        let id = dir_name.clone();
        let label = pretty_label(&dir_name);
        info!("loading demo dataset '{}' from {}", id, sub.display());
        let left =
            read_parquet(&left_path).with_context(|| format!("reading {}", left_path.display()))?;
        let right = read_parquet(&right_path)
            .with_context(|| format!("reading {}", right_path.display()))?;
        let ds = build_dataset(id, label, left, right, tol, tile_size)?;
        datasets.push(ds);
    }

    if datasets.is_empty() {
        return Err(anyhow!(
            "no datasets found in {}: each dataset must be a subdir with left.parquet and right.parquet",
            dir.display()
        ));
    }

    let default_id = pick_default_id(&datasets, requested_default);
    Ok((datasets, default_id))
}

/// Strip a leading "NN-" sort prefix and replace separators with spaces.
fn pretty_label(dir_name: &str) -> String {
    let stripped = if dir_name.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        dir_name.split_once('-').map(|x| x.1).unwrap_or(dir_name)
    } else {
        dir_name
    };
    stripped.replace(['-', '_'], " ")
}

fn pick_default_id(datasets: &[Dataset], requested: Option<&str>) -> String {
    if let Some(req) = requested {
        if datasets.iter().any(|d| d.id == req) {
            return req.to_string();
        }
    }
    if let Some(matching) = datasets.iter().find(|d| d.id.contains("matching")) {
        return matching.id.clone();
    }
    datasets[0].id.clone()
}

fn emit_warnings_for_dataset(ds: &Dataset) {
    let report = ds.report.as_ref();
    if report.columns_reordered {
        eprintln!("[{}] warning: columns were reordered to align", ds.id);
    }
    if !report.rows_sorted_by.is_empty() {
        eprintln!(
            "[{}] warning: rows were sorted by [{}] to align (match rate {:.1}% → {:.1}%)",
            ds.id,
            report.rows_sorted_by.join(", "),
            report.initial_match_rate * 100.0,
            report.final_match_rate * 100.0,
        );
    }
    if !report.columns_only_in_left.is_empty() {
        eprintln!(
            "[{}] warning: columns only in left: {}",
            ds.id,
            report.columns_only_in_left.join(", ")
        );
    }
    if !report.columns_only_in_right.is_empty() {
        eprintln!(
            "[{}] warning: columns only in right: {}",
            ds.id,
            report.columns_only_in_right.join(", ")
        );
    }
    if report.row_count_diff != 0 {
        eprintln!(
            "[{}] warning: row counts differ: left - right = {}",
            ds.id, report.row_count_diff
        );
    }
}
