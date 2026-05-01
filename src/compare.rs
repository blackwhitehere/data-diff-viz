//! Cell-by-cell comparison of two aligned dataframes.

use polars::prelude::*;
use rayon::prelude::*;

use crate::grid::{CellStatus, StatusGrid};

/// Tolerances for the numeric "close" check. Follows numpy.isclose semantics:
/// `|a - b| <= abs_tol + rel_tol * |b|`.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    pub rel_tol: f64,
    pub abs_tol: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self {
            rel_tol: 1e-5,
            abs_tol: 1e-8,
        }
    }
}

/// Build the cell-status grid for two dataframes that have already been
/// aligned (column order matches; row order matches by position). Both
/// dataframes are expected to have the same set of columns in the same
/// order — pass the column names explicitly to make that contract explicit
/// and to drive the grid layout.
///
/// Per-column work is parallelized via rayon.
pub fn build_status_grid(
    left: &DataFrame,
    right: &DataFrame,
    columns: &[String],
    tol: Tolerance,
) -> StatusGrid {
    let n_rows = left.height().max(right.height());
    let n_cols = columns.len();

    if n_rows == 0 || n_cols == 0 {
        return StatusGrid::new(n_rows, n_cols);
    }

    let col_statuses: Vec<Vec<CellStatus>> = columns
        .par_iter()
        .map(|name| compare_column_by_name(left, right, name, n_rows, tol))
        .collect();

    let mut flat = vec![CellStatus::Missing; n_rows * n_cols];
    for (c, col) in col_statuses.iter().enumerate() {
        for (r, s) in col.iter().enumerate() {
            flat[r * n_cols + c] = *s;
        }
    }
    StatusGrid::from_flat(n_rows, n_cols, &flat)
}

fn compare_column_by_name(
    left: &DataFrame,
    right: &DataFrame,
    name: &str,
    n_rows: usize,
    tol: Tolerance,
) -> Vec<CellStatus> {
    let l = left.column(name).ok();
    let r = right.column(name).ok();
    match (l, r) {
        (Some(l), Some(r)) => {
            let ls = l.as_materialized_series();
            let rs = r.as_materialized_series();
            compare_series(ls, rs, n_rows, tol)
        }
        _ => vec![CellStatus::Missing; n_rows],
    }
}

fn compare_series(l: &Series, r: &Series, n_rows: usize, tol: Tolerance) -> Vec<CellStatus> {
    let l_len = l.len();
    let r_len = r.len();
    let common = l_len.min(r_len);

    if l.dtype() != r.dtype() {
        // Type mismatch — flag entire column as Diff (not Missing — values
        // do exist on both sides; they're just incomparable).
        return vec![CellStatus::Diff; n_rows];
    }

    let mut out = vec![CellStatus::Missing; n_rows];

    let dtype = l.dtype().clone();
    match dtype {
        DataType::Float64 => fill_floats(
            l.f64().expect("dtype-checked"),
            r.f64().expect("dtype-checked"),
            common,
            &mut out,
            tol,
        ),
        DataType::Float32 => fill_floats_32(
            l.f32().expect("dtype-checked"),
            r.f32().expect("dtype-checked"),
            common,
            &mut out,
            tol,
        ),
        _ => {
            // Exact-equality fallback for everything else (ints, bool,
            // strings, dates, etc.). AnyValue::eq treats nulls as not-equal,
            // so we handle (null, null) explicitly.
            for (i, slot) in out.iter_mut().enumerate().take(common) {
                let a = l.get(i).ok();
                let b = r.get(i).ok();
                *slot = match (a, b) {
                    (None, None) => CellStatus::Equal,
                    (Some(av), Some(bv)) => {
                        if av_is_null(&av) && av_is_null(&bv) {
                            CellStatus::Equal
                        } else if av_is_null(&av) || av_is_null(&bv) {
                            CellStatus::Diff
                        } else if av == bv {
                            CellStatus::Equal
                        } else {
                            CellStatus::Diff
                        }
                    }
                    _ => CellStatus::Diff,
                };
            }
        }
    }
    out
}

fn av_is_null(v: &AnyValue<'_>) -> bool {
    matches!(v, AnyValue::Null)
}

fn fill_floats(
    l: &Float64Chunked,
    r: &Float64Chunked,
    common: usize,
    out: &mut [CellStatus],
    tol: Tolerance,
) {
    for (i, slot) in out.iter_mut().enumerate().take(common) {
        *slot = compare_f64(l.get(i), r.get(i), tol);
    }
}

fn fill_floats_32(
    l: &Float32Chunked,
    r: &Float32Chunked,
    common: usize,
    out: &mut [CellStatus],
    tol: Tolerance,
) {
    for (i, slot) in out.iter_mut().enumerate().take(common) {
        let a = l.get(i).map(|v| v as f64);
        let b = r.get(i).map(|v| v as f64);
        *slot = compare_f64(a, b, tol);
    }
}

pub fn compare_f64(a: Option<f64>, b: Option<f64>, tol: Tolerance) -> CellStatus {
    match (a, b) {
        (None, None) => CellStatus::Equal,
        (None, _) | (_, None) => CellStatus::Diff,
        (Some(x), Some(y)) => {
            if x.to_bits() == y.to_bits() {
                return CellStatus::Equal;
            }
            if x.is_nan() && y.is_nan() {
                return CellStatus::Equal;
            }
            if x.is_nan() || y.is_nan() {
                return CellStatus::Diff;
            }
            if x.is_infinite() || y.is_infinite() {
                return if x == y {
                    CellStatus::Equal
                } else {
                    CellStatus::Diff
                };
            }
            if x == y {
                return CellStatus::Equal;
            }
            let diff = (x - y).abs();
            let bound = tol.abs_tol + tol.rel_tol * y.abs();
            if diff <= bound {
                CellStatus::Close
            } else {
                CellStatus::Diff
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t() -> Tolerance {
        Tolerance {
            rel_tol: 1e-5,
            abs_tol: 1e-8,
        }
    }

    #[test]
    fn equal_floats() {
        assert_eq!(compare_f64(Some(1.0), Some(1.0), t()), CellStatus::Equal);
    }

    #[test]
    fn close_within_abs_tol() {
        // Very small numbers within abs_tol.
        assert_eq!(compare_f64(Some(0.0), Some(1e-9), t()), CellStatus::Close);
    }

    #[test]
    fn close_within_rel_tol() {
        // Large numbers within rel_tol (1e-5).
        let a = 1_000_000.0;
        let b = a + 1.0; // rel diff 1e-6 < 1e-5
        assert_eq!(compare_f64(Some(a), Some(b), t()), CellStatus::Close);
    }

    #[test]
    fn diff_outside_tol() {
        assert_eq!(compare_f64(Some(1.0), Some(2.0), t()), CellStatus::Diff);
    }

    #[test]
    fn nan_vs_nan_is_equal() {
        assert_eq!(
            compare_f64(Some(f64::NAN), Some(f64::NAN), t()),
            CellStatus::Equal
        );
    }

    #[test]
    fn nan_vs_number_is_diff() {
        assert_eq!(
            compare_f64(Some(f64::NAN), Some(1.0), t()),
            CellStatus::Diff
        );
    }

    #[test]
    fn null_pair_is_equal_null_unilateral_is_diff() {
        assert_eq!(compare_f64(None, None, t()), CellStatus::Equal);
        assert_eq!(compare_f64(None, Some(1.0), t()), CellStatus::Diff);
        assert_eq!(compare_f64(Some(1.0), None, t()), CellStatus::Diff);
    }

    #[test]
    fn pos_inf_equal_neg_inf_diff() {
        assert_eq!(
            compare_f64(Some(f64::INFINITY), Some(f64::INFINITY), t()),
            CellStatus::Equal
        );
        assert_eq!(
            compare_f64(Some(f64::INFINITY), Some(f64::NEG_INFINITY), t()),
            CellStatus::Diff
        );
    }

    #[test]
    fn build_grid_mixed_dtype() {
        let left = df! {
            "x" => [1i64, 2, 3, 4],
            "y" => [1.0f64, 2.0, 3.0, 4.0],
            "s" => ["a", "b", "c", "d"],
        }
        .unwrap();
        let right = df! {
            "x" => [1i64, 2, 99, 4],            // diff at row 2
            "y" => [1.0f64, 2.0 + 1e-9, 3.0, 5.0], // close at row 1, diff at row 3
            "s" => ["a", "b", "c", "X"],         // diff at row 3
        }
        .unwrap();
        let cols = vec!["x".to_string(), "y".to_string(), "s".to_string()];
        let g = build_status_grid(&left, &right, &cols, t());
        assert_eq!(g.rows, 4);
        assert_eq!(g.cols, 3);
        assert_eq!(g.get(0, 0), CellStatus::Equal);
        assert_eq!(g.get(0, 1), CellStatus::Equal);
        assert_eq!(g.get(0, 2), CellStatus::Equal);
        assert_eq!(g.get(1, 1), CellStatus::Close);
        assert_eq!(g.get(2, 0), CellStatus::Diff);
        assert_eq!(g.get(3, 1), CellStatus::Diff);
        assert_eq!(g.get(3, 2), CellStatus::Diff);
    }

    #[test]
    fn type_mismatch_marks_column_diff() {
        let left = df! { "x" => [1i64, 2, 3] }.unwrap();
        let right = df! { "x" => [1.0f64, 2.0, 3.0] }.unwrap();
        let cols = vec!["x".to_string()];
        let g = build_status_grid(&left, &right, &cols, t());
        assert_eq!(g.get(0, 0), CellStatus::Diff);
        assert_eq!(g.get(1, 0), CellStatus::Diff);
        assert_eq!(g.get(2, 0), CellStatus::Diff);
    }

    #[test]
    fn unequal_lengths_pad_with_missing() {
        let left = df! { "x" => [1i64, 2, 3, 4] }.unwrap();
        let right = df! { "x" => [1i64, 2] }.unwrap();
        let cols = vec!["x".to_string()];
        let g = build_status_grid(&left, &right, &cols, t());
        assert_eq!(g.rows, 4);
        assert_eq!(g.get(0, 0), CellStatus::Equal);
        assert_eq!(g.get(1, 0), CellStatus::Equal);
        assert_eq!(g.get(2, 0), CellStatus::Missing);
        assert_eq!(g.get(3, 0), CellStatus::Missing);
    }

    #[test]
    fn missing_column_one_side_is_missing_column() {
        let left = df! { "x" => [1i64, 2], "y" => [1i64, 2] }.unwrap();
        let right = df! { "x" => [1i64, 2] }.unwrap();
        let cols = vec!["x".to_string(), "y".to_string()];
        let g = build_status_grid(&left, &right, &cols, t());
        assert_eq!(g.get(0, 0), CellStatus::Equal);
        assert_eq!(g.get(0, 1), CellStatus::Missing);
        assert_eq!(g.get(1, 1), CellStatus::Missing);
    }
}
