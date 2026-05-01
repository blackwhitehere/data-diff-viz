//! Compact 2-bit packed status grid + tile pyramid for the visualization.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatusCounts {
    pub equal: usize,
    pub close: usize,
    pub diff: usize,
    pub missing: usize,
}

impl StatusCounts {
    pub fn total(self) -> usize {
        self.equal + self.close + self.diff + self.missing
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellStatus {
    Equal = 0,
    Close = 1,
    Diff = 2,
    Missing = 3,
}

impl CellStatus {
    pub fn from_u8(v: u8) -> Self {
        match v & 0b11 {
            0 => Self::Equal,
            1 => Self::Close,
            2 => Self::Diff,
            _ => Self::Missing,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Equal => "equal",
            Self::Close => "close",
            Self::Diff => "diff",
            Self::Missing => "missing",
        }
    }
}

/// Worst-status reduction. Used by the pyramid to summarize a 2x2 block.
/// Order: Missing > Diff > Close > Equal.
pub fn worst_of(a: CellStatus, b: CellStatus) -> CellStatus {
    if (a as u8) >= (b as u8) {
        a
    } else {
        b
    }
}

pub fn worst_of_4(a: CellStatus, b: CellStatus, c: CellStatus, d: CellStatus) -> CellStatus {
    worst_of(worst_of(a, b), worst_of(c, d))
}

/// 2-bit packed status grid, row-major. 4 cells per byte.
#[derive(Debug, Clone)]
pub struct StatusGrid {
    pub rows: usize,
    pub cols: usize,
    pub packed: Vec<u8>,
}

impl StatusGrid {
    pub fn new(rows: usize, cols: usize) -> Self {
        let total = rows.saturating_mul(cols);
        let bytes = total.div_ceil(4);
        Self {
            rows,
            cols,
            packed: vec![0u8; bytes],
        }
    }

    pub fn from_packed(rows: usize, cols: usize, packed: Vec<u8>) -> Self {
        let total = rows.saturating_mul(cols);
        assert_eq!(packed.len(), total.div_ceil(4), "packed length mismatch");
        Self { rows, cols, packed }
    }

    /// Build a grid from a flat row-major slice of statuses.
    pub fn from_flat(rows: usize, cols: usize, flat: &[CellStatus]) -> Self {
        assert_eq!(flat.len(), rows * cols, "flat length mismatch");
        let mut g = Self::new(rows, cols);
        for (i, s) in flat.iter().enumerate() {
            g.set_index(i, *s);
        }
        g
    }

    fn linear_index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    pub fn set(&mut self, row: usize, col: usize, status: CellStatus) {
        self.set_index(self.linear_index(row, col), status);
    }

    pub fn get(&self, row: usize, col: usize) -> CellStatus {
        self.get_index(self.linear_index(row, col))
    }

    pub fn set_index(&mut self, idx: usize, status: CellStatus) {
        let byte = idx >> 2;
        let shift = ((idx & 0b11) * 2) as u32;
        let mask = 0b11u8 << shift;
        self.packed[byte] = (self.packed[byte] & !mask) | (((status as u8) << shift) & mask);
    }

    pub fn get_index(&self, idx: usize) -> CellStatus {
        let byte = idx >> 2;
        let shift = ((idx & 0b11) * 2) as u32;
        CellStatus::from_u8((self.packed[byte] >> shift) & 0b11)
    }

    /// Flatten to one byte per cell (row-major). Useful for tests and tile
    /// rendering at the full-resolution level.
    pub fn to_byte_grid(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.rows * self.cols];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.get_index(i) as u8;
        }
        out
    }

    pub fn status_counts(&self) -> StatusCounts {
        let mut counts = StatusCounts {
            equal: 0,
            close: 0,
            diff: 0,
            missing: 0,
        };
        for idx in 0..self.rows * self.cols {
            match self.get_index(idx) {
                CellStatus::Equal => counts.equal += 1,
                CellStatus::Close => counts.close += 1,
                CellStatus::Diff => counts.diff += 1,
                CellStatus::Missing => counts.missing += 1,
            }
        }
        counts
    }
}

/// Image-pyramid of status grids. Level 0 is full resolution; each subsequent
/// level halves both dimensions, taking the worst status of each 2x2 block.
#[derive(Debug, Clone)]
pub struct Pyramid {
    pub tile_size: usize,
    pub levels: Vec<StatusGrid>,
}

impl Pyramid {
    pub fn build(base: StatusGrid, tile_size: usize) -> Self {
        assert!(tile_size > 0);
        let mut levels = vec![base];
        loop {
            let last = levels.last().unwrap();
            if last.rows <= tile_size && last.cols <= tile_size {
                break;
            }
            if last.rows <= 1 && last.cols <= 1 {
                break;
            }
            levels.push(downsample(last));
        }
        Self { tile_size, levels }
    }

    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    pub fn level(&self, level: usize) -> Option<&StatusGrid> {
        self.levels.get(level)
    }

    /// Extract a tile-sized buffer (one byte per cell). Cells outside the grid
    /// are filled with `CellStatus::Missing`.
    pub fn tile_bytes(&self, level: usize, tile_x: usize, tile_y: usize) -> Vec<u8> {
        let g = match self.levels.get(level) {
            Some(g) => g,
            None => return vec![CellStatus::Missing as u8; self.tile_size * self.tile_size],
        };
        let row0 = tile_y * self.tile_size;
        let col0 = tile_x * self.tile_size;
        let mut out = vec![CellStatus::Missing as u8; self.tile_size * self.tile_size];
        for ty in 0..self.tile_size {
            let r = row0 + ty;
            if r >= g.rows {
                break;
            }
            for tx in 0..self.tile_size {
                let c = col0 + tx;
                if c >= g.cols {
                    break;
                }
                out[ty * self.tile_size + tx] = g.get(r, c) as u8;
            }
        }
        out
    }
}

fn downsample(src: &StatusGrid) -> StatusGrid {
    let rows = src.rows.div_ceil(2);
    let cols = src.cols.div_ceil(2);
    let mut out = StatusGrid::new(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            let r0 = r * 2;
            let c0 = c * 2;
            let a = src.get(r0, c0);
            let b = if c0 + 1 < src.cols {
                src.get(r0, c0 + 1)
            } else {
                a
            };
            let cc = if r0 + 1 < src.rows {
                src.get(r0 + 1, c0)
            } else {
                a
            };
            let d = if r0 + 1 < src.rows && c0 + 1 < src.cols {
                src.get(r0 + 1, c0 + 1)
            } else {
                a
            };
            out.set(r, c, worst_of_4(a, b, cc, d));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let cases = [
            CellStatus::Equal,
            CellStatus::Close,
            CellStatus::Diff,
            CellStatus::Missing,
            CellStatus::Equal,
            CellStatus::Diff,
            CellStatus::Close,
            CellStatus::Missing,
            CellStatus::Equal,
        ];
        let mut g = StatusGrid::new(3, 3);
        for (i, s) in cases.iter().enumerate() {
            g.set_index(i, *s);
        }
        for (i, s) in cases.iter().enumerate() {
            assert_eq!(g.get_index(i), *s, "mismatch at {i}");
        }
    }

    #[test]
    fn worst_of_precedence() {
        assert_eq!(
            worst_of(CellStatus::Equal, CellStatus::Close),
            CellStatus::Close
        );
        assert_eq!(
            worst_of(CellStatus::Close, CellStatus::Diff),
            CellStatus::Diff
        );
        assert_eq!(
            worst_of(CellStatus::Diff, CellStatus::Missing),
            CellStatus::Missing
        );
        assert_eq!(
            worst_of_4(
                CellStatus::Equal,
                CellStatus::Equal,
                CellStatus::Close,
                CellStatus::Equal,
            ),
            CellStatus::Close
        );
    }

    #[test]
    fn pyramid_4x4_one_diff() {
        // Build 4x4 with a single Diff cell at (0, 0); rest Equal.
        let flat = {
            let mut v = vec![CellStatus::Equal; 16];
            v[0] = CellStatus::Diff;
            v
        };
        let base = StatusGrid::from_flat(4, 4, &flat);
        let p = Pyramid::build(base, 1);
        // tile_size=1 forces full pyramid down to 1x1.
        assert!(p.num_levels() >= 3, "expected >=3 levels");
        // Level 1 (2x2): top-left should be Diff (from worst-of {Diff,Eq,Eq,Eq}).
        let l1 = p.level(1).unwrap();
        assert_eq!(l1.rows, 2);
        assert_eq!(l1.cols, 2);
        assert_eq!(l1.get(0, 0), CellStatus::Diff);
        assert_eq!(l1.get(0, 1), CellStatus::Equal);
        assert_eq!(l1.get(1, 0), CellStatus::Equal);
        assert_eq!(l1.get(1, 1), CellStatus::Equal);
        // Level 2 (1x1): single cell propagates worst-status.
        let l2 = p.level(2).unwrap();
        assert_eq!(l2.rows, 1);
        assert_eq!(l2.cols, 1);
        assert_eq!(l2.get(0, 0), CellStatus::Diff);
    }

    #[test]
    fn pyramid_odd_dims_uses_top_left_as_padding() {
        // 3x3, all Equal except (2,2) Diff. After downsample, the 2x2 block
        // at (1,1) covers cells (2,2)..(3,3) — out-of-range cells are padded
        // with the top-left's value (CellStatus::Diff at (2,2) here).
        let flat = {
            let mut v = vec![CellStatus::Equal; 9];
            v[8] = CellStatus::Diff;
            v
        };
        let base = StatusGrid::from_flat(3, 3, &flat);
        let p = Pyramid::build(base, 1);
        let l1 = p.level(1).unwrap();
        assert_eq!(l1.rows, 2);
        assert_eq!(l1.cols, 2);
        assert_eq!(l1.get(1, 1), CellStatus::Diff);
    }

    #[test]
    fn tile_bytes_pads_with_missing_outside_grid() {
        let base = StatusGrid::from_flat(2, 2, &[CellStatus::Equal; 4]);
        let p = Pyramid::build(base, 4);
        let bytes = p.tile_bytes(0, 0, 0);
        assert_eq!(bytes.len(), 16);
        // Top-left 2x2 is Equal (=0), rest is Missing (=3).
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[1], 0);
        assert_eq!(bytes[4], 0);
        assert_eq!(bytes[5], 0);
        assert_eq!(bytes[2], 3);
        assert_eq!(bytes[6], 3);
        assert_eq!(bytes[8], 3);
        assert_eq!(bytes[15], 3);
    }

    #[test]
    fn tile_bytes_out_of_range_level_returns_all_missing() {
        let base = StatusGrid::from_flat(1, 1, &[CellStatus::Equal]);
        let p = Pyramid::build(base, 4);
        let bytes = p.tile_bytes(99, 0, 0);
        assert!(bytes.iter().all(|b| *b == CellStatus::Missing as u8));
    }
}
