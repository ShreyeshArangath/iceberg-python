# Sub-File Scan Task Splitting for iceberg-python

## What This Is

Add sub-file read support to iceberg-python's scan planning and execution pipeline. Currently, `FileScanTask` represents an entire data file — one task per file, no way to read a portion. This project extends `FileScanTask` with byte-range fields (`start`/`length`), adds a `plan_splits()` method to `DataScan` that breaks large files into row-group-aligned sub-file tasks targeting a configurable byte size, and modifies `ArrowScan` to read only the specified row groups. This enables downstream consumers (like OpenHouse's dataloader) to distribute sub-file work units across workers.

## Core Value

Large Parquet files can be split into smaller, distributable read tasks at row-group boundaries — enabling parallelism within a single file, not just across files.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Extend `FileScanTask` with optional `start` (byte offset) and `length` (byte count) fields, defaulting to whole-file behavior when unset
- [ ] Add `DataScan.plan_splits(target_split_size: int)` method that returns `Iterable[FileScanTask]` with `start`/`length` populated
- [ ] `plan_splits()` uses `DataFile.split_offsets` (row group byte boundaries) to group consecutive row groups into splits that approximate the target byte size
- [ ] `plan_splits()` falls back to one task per file when `split_offsets` is `None` or the file is smaller than `target_split_size`
- [ ] Modify `ArrowScan._task_to_record_batches` to read only the row groups within the `start`/`length` byte range when those fields are set
- [ ] Position deletes are correctly scoped to the split's row range — only delete positions that fall within the split's row group range are applied
- [ ] Backward compatible: existing `FileScanTask` usage (no `start`/`length`) continues to read entire files with no behavior change
- [ ] Unit tests for `plan_splits()` with various file sizes and row group layouts
- [ ] Unit tests for `ArrowScan` partial reads — verify only specified row groups are read
- [ ] Integration test: `plan_splits()` → `ArrowScan` end-to-end produces correct data

### Out of Scope

- OpenHouse dataloader integration — separate effort that will consume this API
- ORC/Avro sub-file splitting — Parquet only for now (row groups are Parquet-specific)
- Server-side split planning via REST catalog — local planning only
- CombinedScanTask (grouping multiple small files into one task) — different problem
- V3 `content_offset`/`content_size_in_bytes` field exposure — related but separate concern

## Context

- **Branch:** `feat/sub-FST-read` on iceberg-python
- **Java Iceberg precedent:** Java's `FileScanTask` has `start()` and `length()` methods. This aligns iceberg-python with that model.
- **Existing metadata:** `DataFile.split_offsets` (`manifest.py:522`) already stores row group byte offsets from Parquet file writes. This metadata is written but never used during reads — this project makes it useful.
- **ArrowScan architecture:** `_task_to_record_batches` at `pyarrow.py:1586` opens the full file via `io.new_input(path).open()` and creates an Arrow fragment. PyArrow's `ParquetFileFormat.make_fragment()` and `Scanner.from_fragment()` support row-group-level reads via `fragment_scan_options` or `read_row_groups()`.
- **Delete file handling:** Position deletes carry `(file_path, row_position)` pairs. When splitting, row positions must be offset-adjusted to determine which deletes apply to which split. The split's row range is computable from row group record counts.
- **Downstream consumer:** OpenHouse dataloader (`data_loader_split.py`) calls `ArrowScan.to_record_batches([file_scan_task])` — once splits work, it can call `plan_splits()` instead of `plan_files()` and distribute tasks.

## Constraints

- **Backward compatibility**: Must not break any existing `plan_files()` or `ArrowScan` behavior. `start`/`length` default to whole-file semantics.
- **Split boundary**: Row groups are the atomic unit. Splits align to row group boundaries — no mid-row-group reads.
- **Parquet only**: ORC and Avro files don't have an equivalent of row groups with byte offsets. `plan_splits()` should pass them through unsplit.
- **No Parquet metadata fetch during planning**: `plan_splits()` must work with the metadata already in `DataFile` (`split_offsets`, `file_size_in_bytes`, `record_count`). It should not open files to read Parquet footer metadata during planning.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Extend FileScanTask (not new type) | Matches Java Iceberg's model. Single type for consumers. Backward compatible with defaults. | — Pending |
| Row-group-aligned target byte size | Row groups are the natural Parquet split boundary. Target size allows flexible configuration while respecting physical layout. | — Pending |
| New plan_splits() method (not extend plan_files()) | Keeps plan_files() unchanged. Consumers explicitly opt in to splitting. No surprise behavior changes. | — Pending |
| Position deletes handled correctly | Incorrect delete scoping would produce wrong results silently. Must be correct from the start. | — Pending |
| Parquet only | ORC/Avro lack row-group-level byte offset metadata. Simpler to scope to Parquet and add others later. | — Pending |

---
*Last updated: 2026-02-13 after initialization*
