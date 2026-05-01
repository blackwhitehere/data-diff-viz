//! Parquet I/O. Thin wrapper around `polars::ParquetReader` with the
//! pandas-written `__index_level_0__` column dropped automatically.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use polars::prelude::*;

const PANDAS_INDEX_COL_PREFIX: &str = "__index_level_";

/// Read a parquet file into a Polars `DataFrame`. Columns matching
/// `__index_level_*__` (written by pandas) are dropped; they are not part
/// of the user-facing data.
pub fn read_parquet<P: AsRef<Path>>(path: P) -> Result<DataFrame> {
    let path = path.as_ref();
    let file =
        File::open(path).with_context(|| format!("opening parquet file {}", path.display()))?;
    let df = ParquetReader::new(file)
        .finish()
        .with_context(|| format!("reading parquet file {}", path.display()))?;
    Ok(drop_pandas_index_cols(df))
}

fn drop_pandas_index_cols(df: DataFrame) -> DataFrame {
    let to_drop: Vec<String> = df
        .get_column_names()
        .iter()
        .filter(|n| n.starts_with(PANDAS_INDEX_COL_PREFIX))
        .map(|n| n.to_string())
        .collect();
    if to_drop.is_empty() {
        return df;
    }
    let mut out = df;
    for name in &to_drop {
        if let Ok(d) = out.drop(name) {
            out = d;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_parquet(path: &Path, mut df: DataFrame) -> Result<()> {
        let file = File::create(path)?;
        ParquetWriter::new(file).finish(&mut df)?;
        Ok(())
    }

    #[test]
    fn round_trip_small_parquet() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("x.parquet");
        let df = df! {
            "a" => [1i64, 2, 3],
            "b" => [1.0f64, 2.0, 3.0],
        }
        .unwrap();
        write_parquet(&path, df.clone()).unwrap();
        let read = read_parquet(&path).unwrap();
        assert_eq!(read.shape(), (3, 2));
        assert_eq!(read.get_column_names(), vec!["a", "b"]);
    }

    #[test]
    fn drops_pandas_index_col() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("x.parquet");
        let df = df! {
            "__index_level_0__" => [10i64, 11, 12],
            "value" => [1.0f64, 2.0, 3.0],
        }
        .unwrap();
        write_parquet(&path, df).unwrap();
        let read = read_parquet(&path).unwrap();
        assert_eq!(read.get_column_names(), vec!["value"]);
        assert_eq!(read.shape(), (3, 1));
    }
}
