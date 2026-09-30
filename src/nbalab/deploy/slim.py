"""Write the small copy of ``data/processed`` that the public app ships with.

For each table in :data:`nbalab.deploy.manifest.TABLES`:

1. keep only the manifest columns,
2. keep only seasons from the configured start (``BuildConfig.start_season``)
   and drop rows the loader would discard (``keep_if_true``),
3. apply the manifest's downcasts, sort (sorted data compresses better),
4. write parquet with zstd at a high level. Readers decompress zstd quickly
   at any level, so the extra cost is paid once, at write time.

A ``manifest.json`` with row counts and sizes is written next to the files,
and the app checks it on start.

Usage::

    python -m nbalab.deploy.slim                 # -> data/deploy/
    python -m nbalab.deploy.slim --start-season 2005
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nbalab.data.config import DATA_DIR, BuildConfig
from nbalab.deploy.manifest import TABLES, TableSpec

DEPLOY_DIR = DATA_DIR / "deploy"
ZSTD_LEVEL = 19
MANIFEST_FILE = "manifest.json"


def slim_table(table: pa.Table, spec: TableSpec, start_season: int) -> pa.Table:
    """Select, filter, downcast, and sort one table. Raises if a manifest column is missing."""
    missing = [c for c in spec.columns if c not in table.column_names]
    if missing:
        raise KeyError(f"columns in manifest but not in processed table: {missing}")
    out = table.select(list(spec.columns))
    if "season" in out.column_names:
        out = out.filter(pc.greater_equal(out["season"], start_season))
    if spec.keep_if_true:
        out = out.filter(pc.fill_null(out[spec.keep_if_true], False))
    for name, dtype in spec.downcast.items():
        out = out.set_column(out.schema.get_field_index(name), name, pc.cast(out[name], dtype))
    if spec.sort_by:
        out = out.sort_by([(c, "ascending") for c in spec.sort_by])
    return out


def write_slim(
    processed_dir: Path = BuildConfig().processed_dir,
    out_dir: Path = DEPLOY_DIR,
    start_season: int = BuildConfig().start_season,
) -> dict[str, dict[str, int]]:
    """Slim every manifest table present in ``processed_dir``. Returns per-table stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {}
    for name, spec in TABLES.items():
        src = processed_dir / f"{name}.parquet"
        if not src.exists():
            continue
        slim = slim_table(pq.read_table(src), spec, start_season)
        dest = out_dir / f"{name}.parquet"
        pq.write_table(slim, dest, compression="zstd", compression_level=ZSTD_LEVEL)
        stats[name] = {
            "rows": slim.num_rows, "columns": slim.num_columns,
            "source_bytes": src.stat().st_size, "bytes": dest.stat().st_size,
        }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start_season": start_season,
        "tables": stats,
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=BuildConfig().processed_dir)
    parser.add_argument("--out-dir", type=Path, default=DEPLOY_DIR)
    parser.add_argument("--start-season", type=int, default=BuildConfig().start_season)
    args = parser.parse_args()
    stats = write_slim(args.processed_dir, args.out_dir, args.start_season)
    total_src = sum(s["source_bytes"] for s in stats.values())
    total = sum(s["bytes"] for s in stats.values())
    for name, s in stats.items():
        print(f"{name:14s} {s['rows']:>10,} rows {s['columns']:>3} cols "
              f"{s['source_bytes'] / 1e6:7.1f} MB -> {s['bytes'] / 1e6:6.1f} MB")
    print(f"{'total':14s} {'':>24} {total_src / 1e6:7.1f} MB -> {total / 1e6:6.1f} MB  ({args.out_dir})")


if __name__ == "__main__":
    main()
