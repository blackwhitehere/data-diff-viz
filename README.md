# data-diff-viz

Visual cell-by-cell diff for two parquet dataframes. Reads two parquet files,
compares every cell with a numeric tolerance, and serves a zoomable web
visualization where each cell is colored:

- **green** — values are equal
- **orange** — values are close (numeric, within `--rel-tol`/`--abs-tol`)
- **red** — values differ
- **dark grey** — cell is missing on one side (different shapes)

Click any cell to inspect both values side-by-side. Designed to scale to
dataframes with millions of rows via a server-side tile pyramid (worst-status
downsample) and Canvas2D rendering.

## Usage

```sh
data-diff-viz LEFT.parquet RIGHT.parquet
```

Then open the printed URL (default `http://127.0.0.1:8080`).

### Options

```
--rel-tol <F>     Relative tolerance for numeric "close" check (default 1e-5)
--abs-tol <F>     Absolute tolerance for numeric "close" check (default 1e-8)
--bind <ADDR>     Bind address (default 127.0.0.1:8080)
--tile-size <N>   Cells per tile edge (default 256)
```

The "close" check follows `numpy.isclose`:
`|a - b| <= abs_tol + rel_tol * |b|`.

## Alignment

The tool first checks whether the two dataframes are already aligned. If not:

- **Columns** with the same set in different order are reordered to lexicographic.
- **Rows** are sorted by the natural order of their dtypes across all common
  columns when sample-match-rate is low under the given order.

A warning naming the reordered dimension is printed to stderr and shown as a
banner in the browser.

## Development

```sh
just test       # unit + integration
just lint       # clippy with -D warnings
just fmt        # format
just bench      # criterion benchmarks
just demo       # generate fixture parquets and launch
```
