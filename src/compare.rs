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
/// Work is parallelized over the packed row-major byte buffer.
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

    let comparers: Vec<ColumnCompare<'_>> = columns
        .iter()
        .map(|name| ColumnCompare::new(left, right, name))
        .collect();

    let total = n_rows * n_cols;
    let mut packed = vec![0u8; total.div_ceil(4)];
    packed
        .par_iter_mut()
        .enumerate()
        .for_each(|(byte_idx, byte)| {
            let mut value = 0u8;
            let first_idx = byte_idx * 4;
            for offset in 0..4 {
                let idx = first_idx + offset;
                if idx >= total {
                    break;
                }
                let row = idx / n_cols;
                let col = idx % n_cols;
                let status = comparers[col].status(row, tol) as u8;
                value |= status << (offset * 2);
            }
            *byte = value;
        });

    StatusGrid::from_packed(n_rows, n_cols, packed)
}

enum ColumnCompare<'a> {
    Missing,
    TypeMismatch,
    Float64 {
        left: &'a Float64Chunked,
        right: &'a Float64Chunked,
        common: usize,
    },
    Float32 {
        left: &'a Float32Chunked,
        right: &'a Float32Chunked,
        common: usize,
    },
    Other {
        left: &'a Series,
        right: &'a Series,
        common: usize,
    },
}

impl<'a> ColumnCompare<'a> {
    fn new(left: &'a DataFrame, right: &'a DataFrame, name: &str) -> Self {
        let Some(left) = left.column(name).ok().map(|c| c.as_materialized_series()) else {
            return Self::Missing;
        };
        let Some(right) = right.column(name).ok().map(|c| c.as_materialized_series()) else {
            return Self::Missing;
        };

        if left.dtype() != right.dtype() {
            return Self::TypeMismatch;
        }

        let common = left.len().min(right.len());
        match left.dtype() {
            DataType::Float64 => Self::Float64 {
                left: left.f64().expect("dtype-checked"),
                right: right.f64().expect("dtype-checked"),
                common,
            },
            DataType::Float32 => Self::Float32 {
                left: left.f32().expect("dtype-checked"),
                right: right.f32().expect("dtype-checked"),
                common,
            },
            _ => Self::Other {
                left,
                right,
                common,
            },
        }
    }

    fn status(&self, row: usize, tol: Tolerance) -> CellStatus {
        match self {
            Self::Missing => CellStatus::Missing,
            // Type mismatch flags the whole column as Diff: values exist on
            // both sides, but are incomparable.
            Self::TypeMismatch => CellStatus::Diff,
            Self::Float64 {
                left,
                right,
                common,
            } => {
                if row >= *common {
                    CellStatus::Missing
                } else {
                    compare_f64(left.get(row), right.get(row), tol)
                }
            }
            Self::Float32 {
                left,
                right,
                common,
            } => {
                if row >= *common {
                    CellStatus::Missing
                } else {
                    let a = left.get(row).map(|v| v as f64);
                    let b = right.get(row).map(|v| v as f64);
                    compare_f64(a, b, tol)
                }
            }
            Self::Other {
                left,
                right,
                common,
            } => {
                if row >= *common {
                    CellStatus::Missing
                } else {
                    compare_any_values(left.get(row).ok(), right.get(row).ok())
                }
            }
        }
    }
}

fn compare_any_values(a: Option<AnyValue<'_>>, b: Option<AnyValue<'_>>) -> CellStatus {
    match (a, b) {
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
    }
}

fn av_is_null(v: &AnyValue<'_>) -> bool {
    matches!(v, AnyValue::Null)
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

    #[test]
    fn direct_packing_handles_partial_final_byte() {
        let left = df! {
            "a" => [1i64, 2, 3],
            "b" => [10i64, 20, 30],
            "c" => [1.0f64, 2.0, 3.0],
        }
        .unwrap();
        let right = df! {
            "a" => [1i64, 99, 3],
            "b" => [10i64, 20, 31],
            "c" => [1.0f64, 2.0 + 1e-9, 4.0],
        }
        .unwrap();
        let cols = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let g = build_status_grid(&left, &right, &cols, t());

        assert_eq!(g.rows, 3);
        assert_eq!(g.cols, 3);
        assert_eq!(g.packed.len(), 3);
        assert_eq!(
            g.to_byte_grid(),
            vec![
                CellStatus::Equal as u8,
                CellStatus::Equal as u8,
                CellStatus::Equal as u8,
                CellStatus::Diff as u8,
                CellStatus::Equal as u8,
                CellStatus::Close as u8,
                CellStatus::Equal as u8,
                CellStatus::Diff as u8,
                CellStatus::Diff as u8,
            ]
        );
    }

    #[test]
    fn direct_packing_preserves_missing_and_type_mismatch_edges() {
        let left = df! {
            "same" => [Some(1i64), None, Some(3)],
            "type_mismatch" => [1i64, 2, 3],
            "left_only" => ["a", "b", "c"],
            "short" => [1.0f64, 2.0, 3.0],
        }
        .unwrap();
        let right = df! {
            "same" => [Some(1i64), None],
            "type_mismatch" => [1.0f64, 2.0],
            "short" => [1.0f64, 9.0],
        }
        .unwrap();
        let cols = vec![
            "same".to_string(),
            "type_mismatch".to_string(),
            "left_only".to_string(),
            "short".to_string(),
        ];
        let g = build_status_grid(&left, &right, &cols, t());

        assert_eq!(g.get(0, 0), CellStatus::Equal);
        assert_eq!(g.get(1, 0), CellStatus::Equal);
        assert_eq!(g.get(2, 0), CellStatus::Missing);
        assert_eq!(g.get(0, 1), CellStatus::Diff);
        assert_eq!(g.get(2, 1), CellStatus::Diff);
        assert_eq!(g.get(0, 2), CellStatus::Missing);
        assert_eq!(g.get(2, 2), CellStatus::Missing);
        assert_eq!(g.get(0, 3), CellStatus::Equal);
        assert_eq!(g.get(1, 3), CellStatus::Diff);
        assert_eq!(g.get(2, 3), CellStatus::Missing);
    }
}
