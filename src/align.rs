//! Alignment detection. Determines whether the two dataframes are already
//! aligned column-for-column and row-for-row. If not, reorders columns to a
//! shared lexicographic order and (optionally) sorts rows by all common
//! columns. Emits a report describing what was changed.

use anyhow::Result;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::compare::{build_status_grid, Tolerance};
use crate::grid::CellStatus;

const SAMPLE_ROWS: usize = 1024;
/// If the as-given match rate is below this, we attempt a sort.
const SORT_TRIGGER_RATE: f64 = 0.95;
/// After sorting, we only keep the sort if the new match rate exceeds this.
const SORT_KEEP_RATE: f64 = 0.95;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentReport {
    /// Column names actually compared, in display order (lex-sorted union).
    pub display_columns: Vec<String>,
    /// `true` if column order had to be changed to align names.
    pub columns_reordered: bool,
    /// Names present in left but missing in right.
    pub columns_only_in_left: Vec<String>,
    /// Names present in right but missing in left.
    pub columns_only_in_right: Vec<String>,
    /// If non-empty, the rows were sorted by these columns to align.
    pub rows_sorted_by: Vec<String>,
    /// Sample match rate (Equal+Close fraction) under the original order.
    pub initial_match_rate: f64,
    /// Sample match rate after any sort that was applied.
    pub final_match_rate: f64,
    /// Difference in row counts between the two sides (left - right).
    pub row_count_diff: i64,
}

/// Detect whether the two dataframes need column / row reordering to align.
/// Returns the (possibly modified) dataframes plus a report. Both returned
/// dataframes have the same column set in the same order.
pub fn detect_and_align(
    left: DataFrame,
    right: DataFrame,
    tol: Tolerance,
) -> Result<(DataFrame, DataFrame, AlignmentReport)> {
    let left_cols = column_names(&left);
    let right_cols = column_names(&right);

    let mut report = AlignmentReport {
        display_columns: vec![],
        columns_reordered: false,
        columns_only_in_left: diff_set(&left_cols, &right_cols),
        columns_only_in_right: diff_set(&right_cols, &left_cols),
        rows_sorted_by: vec![],
        initial_match_rate: 0.0,
        final_match_rate: 0.0,
        row_count_diff: left.height() as i64 - right.height() as i64,
    };

    // Build display column list: lex-sorted union.
    let mut display_cols: Vec<String> =
        left_cols.iter().chain(right_cols.iter()).cloned().collect();
    display_cols.sort();
    display_cols.dedup();
    report.display_columns = display_cols.clone();

    // Did column order change vs. either source?
    if left_cols != display_cols || right_cols != display_cols {
        report.columns_reordered = true;
    }

    // Reorder dataframes to display column order (skipping columns that
    // don't exist on that side; we still want the column slot to render
    // as Missing in compare.rs).
    let left = select_existing(&left, &display_cols)?;
    let right = select_existing(&right, &display_cols)?;

    // Sample match rate as-given.
    let common_cols: Vec<String> = display_cols
        .iter()
        .filter(|c| {
            left.get_column_names()
                .iter()
                .any(|n| n.as_str() == c.as_str())
                && right
                    .get_column_names()
                    .iter()
                    .any(|n| n.as_str() == c.as_str())
        })
        .cloned()
        .collect();

    let initial_rate = sample_match_rate(&left, &right, &common_cols, tol);
    report.initial_match_rate = initial_rate;
    report.final_match_rate = initial_rate;

    if initial_rate >= SORT_TRIGGER_RATE || common_cols.is_empty() {
        return Ok((left, right, report));
    }

    // Try sorting by all common columns in their natural order.
    let sort_opts = SortMultipleOptions::default()
        .with_order_descending(false)
        .with_nulls_last(true)
        .with_multithreaded(true);

    let left_sorted = left.sort(&common_cols, sort_opts.clone())?;
    let right_sorted = right.sort(&common_cols, sort_opts)?;

    let sorted_rate = sample_match_rate(&left_sorted, &right_sorted, &common_cols, tol);

    if sorted_rate >= SORT_KEEP_RATE && sorted_rate > initial_rate {
        report.rows_sorted_by = common_cols;
        report.final_match_rate = sorted_rate;
        Ok((left_sorted, right_sorted, report))
    } else {
        // Sort didn't help — keep the original order.
        Ok((left, right, report))
    }
}

fn column_names(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .map(|n| n.to_string())
        .collect()
}

fn diff_set(a: &[String], b: &[String]) -> Vec<String> {
    a.iter().filter(|x| !b.contains(x)).cloned().collect()
}

/// Select columns that exist in `df`, in the order given. Columns absent
/// from `df` are silently skipped — compare.rs will then see them as
/// missing on this side and render the slot as `CellStatus::Missing`.
fn select_existing(df: &DataFrame, columns: &[String]) -> Result<DataFrame> {
    let existing: Vec<&str> = columns
        .iter()
        .filter(|c| {
            df.get_column_names()
                .iter()
                .any(|n| n.as_str() == c.as_str())
        })
        .map(|c| c.as_str())
        .collect();
    Ok(df.select(existing)?)
}

fn sample_match_rate(
    left: &DataFrame,
    right: &DataFrame,
    columns: &[String],
    tol: Tolerance,
) -> f64 {
    if columns.is_empty() {
        return 1.0;
    }
    let n = SAMPLE_ROWS.min(left.height()).min(right.height());
    if n == 0 {
        return 1.0;
    }
    let l_head = left.head(Some(n));
    let r_head = right.head(Some(n));
    let g = build_status_grid(&l_head, &r_head, columns, tol);
    let total = n * columns.len();
    if total == 0 {
        return 1.0;
    }
    let mut hits = 0usize;
    for i in 0..total {
        match g.get_index(i) {
            CellStatus::Equal | CellStatus::Close => hits += 1,
            _ => {}
        }
    }
    hits as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t() -> Tolerance {
        Tolerance::default()
    }

    #[test]
    fn already_aligned_no_changes() {
        let left = df! {
            "a" => [1i64, 2, 3],
            "b" => [10i64, 20, 30],
        }
        .unwrap();
        let right = left.clone();
        let (_l, _r, report) = detect_and_align(left, right, t()).unwrap();
        assert!(!report.columns_reordered);
        assert!(report.rows_sorted_by.is_empty());
        assert!(report.columns_only_in_left.is_empty());
        assert!(report.columns_only_in_right.is_empty());
        assert!(report.initial_match_rate >= 0.99);
    }

    #[test]
    fn reordered_columns_get_realigned() {
        let left = df! {
            "a" => [1i64, 2, 3],
            "b" => [10i64, 20, 30],
        }
        .unwrap();
        let right = df! {
            "b" => [10i64, 20, 30],
            "a" => [1i64, 2, 3],
        }
        .unwrap();
        let (l, r, report) = detect_and_align(left, right, t()).unwrap();
        assert!(report.columns_reordered);
        // Both sides reordered to lex order: a, b.
        let l_names: Vec<String> = l.get_column_names().iter().map(|s| s.to_string()).collect();
        let r_names: Vec<String> = r.get_column_names().iter().map(|s| s.to_string()).collect();
        assert_eq!(l_names, vec!["a", "b"]);
        assert_eq!(r_names, vec!["a", "b"]);
        assert!(report.rows_sorted_by.is_empty());
    }

    #[test]
    fn shuffled_rows_get_sorted() {
        let left = df! {
            "id" => [1i64, 2, 3, 4, 5],
            "v" => [10i64, 20, 30, 40, 50],
        }
        .unwrap();
        let right = df! {
            "id" => [3i64, 1, 5, 2, 4],
            "v" => [30i64, 10, 50, 20, 40],
        }
        .unwrap();
        let (_l, _r, report) = detect_and_align(left, right, t()).unwrap();
        assert!(!report.rows_sorted_by.is_empty(), "expected a sort");
        assert!(report.rows_sorted_by.contains(&"id".to_string()));
        assert!(report.final_match_rate > report.initial_match_rate);
        assert!(report.final_match_rate >= 0.95);
    }

    #[test]
    fn extra_columns_recorded_in_report() {
        let left = df! {
            "a" => [1i64, 2, 3],
            "extra_left" => [9i64, 9, 9],
        }
        .unwrap();
        let right = df! {
            "a" => [1i64, 2, 3],
            "extra_right" => [8i64, 8, 8],
        }
        .unwrap();
        let (_l, _r, report) = detect_and_align(left, right, t()).unwrap();
        assert_eq!(report.columns_only_in_left, vec!["extra_left".to_string()]);
        assert_eq!(
            report.columns_only_in_right,
            vec!["extra_right".to_string()]
        );
        assert!(report.display_columns.contains(&"a".to_string()));
        assert!(report.display_columns.contains(&"extra_left".to_string()));
        assert!(report.display_columns.contains(&"extra_right".to_string()));
    }

    #[test]
    fn unequal_lengths_reported() {
        let left = df! { "a" => [1i64, 2, 3, 4] }.unwrap();
        let right = df! { "a" => [1i64, 2] }.unwrap();
        let (_l, _r, report) = detect_and_align(left, right, t()).unwrap();
        assert_eq!(report.row_count_diff, 2);
    }
}
