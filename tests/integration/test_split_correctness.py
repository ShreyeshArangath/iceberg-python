# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name
"""Parametrized correctness tests validating split read equivalence.

These tests validate the fundamental correctness property:
    concat(all_splits) == read_whole_file

Across multiple scenarios:
- Various file sizes (small, medium, large)
- Position delete patterns (every_10th, first_100, last_100, scattered)
- Edge cases (all rows deleted, partitioned tables)
"""

import pyarrow as pa
import pytest
from pyspark.sql import SparkSession

from pyiceberg.catalog.rest import RestCatalog
from pyiceberg.io.pyarrow import ArrowScan
from pyiceberg.table import Table


def run_spark_commands(spark: SparkSession, sqls: list[str]) -> None:
    """Execute a list of SQL commands via Spark."""
    for sql in sqls:
        spark.sql(sql)


def has_orc_fragment_support() -> bool:
    """Check if PyArrow has ORC fragment subsetting support."""
    try:
        import pyarrow.dataset as ds

        # Check if OrcFileFragment has subset method
        return hasattr(ds.OrcFileFragment, 'subset')
    except (ImportError, AttributeError):
        return False


def _read_splits_concat(tbl: Table, target_split_size: int) -> pa.Table:
    """Helper that reads all splits and concatenates into a single PyArrow Table.

    Args:
        tbl: Iceberg table to read
        target_split_size: Target size in bytes for each split

    Returns:
        PyArrow Table with all split data concatenated and sorted by 'id'
    """
    # Plan splits
    split_tasks = list(tbl.scan().plan_splits(target_split_size=target_split_size))

    # If no splits (empty table), return empty table with schema
    if not split_tasks:
        from pyiceberg.io.pyarrow import schema_to_pyarrow
        arrow_schema = schema_to_pyarrow(tbl.schema(), include_field_ids=False)
        return arrow_schema.empty_table()

    # Use ArrowScan to read split tasks
    arrow_scan = ArrowScan(
        table_metadata=tbl.metadata,
        io=tbl.io,
        projected_schema=tbl.schema(),
        row_filter=tbl.scan().row_filter,
        case_sensitive=tbl.scan().case_sensitive,
        limit=None,
    )

    # Collect all record batches from split tasks
    batches = []
    for task in split_tasks:
        # Use to_record_batches method which properly handles splits
        task_table = arrow_scan.to_table([task])
        if len(task_table) > 0:
            batches.append(task_table)

    # If no data, return empty table
    if not batches:
        from pyiceberg.io.pyarrow import schema_to_pyarrow
        arrow_schema = schema_to_pyarrow(tbl.schema(), include_field_ids=False)
        return arrow_schema.empty_table()

    # Concatenate and sort by id
    result = pa.concat_tables(batches)
    return result.sort_by("id")


@pytest.mark.integration
@pytest.mark.parametrize("num_rows,target_split_mb", [
    (100, 32),       # Small table, may not split
    (50000, 32),     # Medium table, should split with 32MB target
    (100000, 64),    # Larger table, should produce multiple splits
])
def test_concat_splits_equals_whole_file_no_deletes(
    spark: SparkSession, session_catalog: RestCatalog, num_rows: int, target_split_mb: int
) -> None:
    """Test that concat(all_splits) == read_whole_file for tables without deletes.

    Validates the fundamental correctness property across different file sizes.
    """
    identifier = f"default.test_split_correctness_no_deletes_{num_rows}"

    # Force smaller row groups (8MB) to ensure multiple row groups even with moderate data
    spark.conf.set("parquet.block.size", str(8 * 1024 * 1024))

    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (
                id BIGINT,
                data STRING
            )
            USING iceberg
            TBLPROPERTIES('write.parquet.row-group-size-bytes' = '8388608')
            """,
            # Insert rows with padding to create significant data size
            f"""
            INSERT INTO {identifier}
            SELECT id, repeat('x', 500) as data
            FROM range({num_rows})
            """,
        ],
    )

    tbl = session_catalog.load_table(identifier)

    # Read whole file using standard to_arrow()
    whole = tbl.scan().to_arrow().sort_by("id")

    # Read splits and concatenate
    split_result = _read_splits_concat(tbl, target_split_size=target_split_mb * 1024 * 1024)

    # Assert equality
    assert whole.equals(split_result), f"Split reads must equal whole file read for {num_rows} rows"
    assert len(whole) == len(split_result), f"Row count mismatch: {len(whole)} != {len(split_result)}"


@pytest.mark.integration
@pytest.mark.parametrize("delete_pattern", [
    "every_10th",     # DELETE WHERE id % 10 = 0
    "first_100",      # DELETE WHERE id < 100
    "last_100",       # DELETE WHERE id >= (total - 100)
    "scattered",      # DELETE WHERE id % 1000 = 0 OR id % 1000 = 500
])
def test_concat_splits_equals_whole_file_with_position_deletes(
    spark: SparkSession, session_catalog: RestCatalog, delete_pattern: str
) -> None:
    """Test that concat(all_splits) == read_whole_file with position deletes.

    Validates correctness property with various delete patterns that test
    position delete scoping at split boundaries.
    """
    identifier = f"default.test_split_correctness_deletes_{delete_pattern}"

    # Force smaller row groups (8MB) to ensure splits
    spark.conf.set("parquet.block.size", str(8 * 1024 * 1024))

    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (
                id BIGINT,
                data STRING
            )
            USING iceberg
            TBLPROPERTIES(
                'format-version' = 2,
                'write.parquet.row-group-size-bytes' = '8388608',
                'write.delete.mode' = 'merge-on-read',
                'write.update.mode' = 'merge-on-read',
                'write.merge.mode' = 'merge-on-read'
            )
            """,
            # Insert 50k rows with padding
            f"""
            INSERT INTO {identifier}
            SELECT id, repeat('x', 500) as data
            FROM range(50000)
            """,
        ],
    )

    # Apply delete pattern
    if delete_pattern == "every_10th":
        delete_sql = f"DELETE FROM {identifier} WHERE id % 10 = 0"
    elif delete_pattern == "first_100":
        delete_sql = f"DELETE FROM {identifier} WHERE id < 100"
    elif delete_pattern == "last_100":
        delete_sql = f"DELETE FROM {identifier} WHERE id >= 49900"
    elif delete_pattern == "scattered":
        delete_sql = f"DELETE FROM {identifier} WHERE id % 1000 = 0 OR id % 1000 = 500"
    else:
        raise ValueError(f"Unknown delete pattern: {delete_pattern}")

    run_spark_commands(spark, [delete_sql])

    tbl = session_catalog.load_table(identifier)

    # Read whole file
    whole = tbl.scan().to_arrow().sort_by("id")

    # Read splits and concatenate
    split_result = _read_splits_concat(tbl, target_split_size=32 * 1024 * 1024)

    # Assert equality
    assert whole.equals(split_result), f"Split reads must equal whole file read with {delete_pattern} deletes"
    assert len(whole) == len(split_result), f"Row count mismatch: {len(whole)} != {len(split_result)}"


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:Merge on read is not yet supported, falling back to copy-on-write")
def test_concat_splits_with_all_rows_deleted(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Test that split reads correctly handle case where all rows are deleted.

    Both whole file read and split reads should return empty table.
    """
    identifier = "default.test_split_correctness_all_deleted"

    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (
                id BIGINT,
                data STRING
            )
            USING iceberg
            TBLPROPERTIES(
                'format-version' = 2,
                'write.delete.mode' = 'merge-on-read',
                'write.update.mode' = 'merge-on-read',
                'write.merge.mode' = 'merge-on-read'
            )
            """,
            f"""
            INSERT INTO {identifier}
            SELECT id, repeat('x', 100) as data
            FROM range(1000)
            """,
            # Delete all rows
            f"DELETE FROM {identifier} WHERE TRUE",
        ],
    )

    tbl = session_catalog.load_table(identifier)

    # Read whole file
    whole = tbl.scan().to_arrow()

    # Read splits
    split_result = _read_splits_concat(tbl, target_split_size=32 * 1024 * 1024)

    # Both should be empty
    assert len(whole) == 0, f"Expected 0 rows in whole file read, got {len(whole)}"
    assert len(split_result) == 0, f"Expected 0 rows in split reads, got {len(split_result)}"


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:Merge on read is not yet supported, falling back to copy-on-write")
def test_concat_splits_partitioned_table_with_deletes(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Test that split reads work correctly with partitioned tables and position deletes.

    Validates that split planning respects partition boundaries and position deletes
    are applied correctly across partitions.
    """
    identifier = "default.test_split_correctness_partitioned"

    spark.conf.set("parquet.block.size", str(8 * 1024 * 1024))

    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (
                id BIGINT,
                part_key INT,
                data STRING
            )
            USING iceberg
            PARTITIONED BY (part_key)
            TBLPROPERTIES(
                'format-version' = 2,
                'write.parquet.row-group-size-bytes' = '8388608',
                'write.delete.mode' = 'merge-on-read',
                'write.update.mode' = 'merge-on-read',
                'write.merge.mode' = 'merge-on-read'
            )
            """,
            # Insert data into 3 partitions
            f"""
            INSERT INTO {identifier}
            SELECT id, CAST((id % 3) AS INT) as part_key, repeat('x', 500) as data
            FROM range(30000)
            """,
        ],
    )

    # Delete some rows from each partition
    run_spark_commands(
        spark,
        [
            # Delete rows from partition 0
            f"DELETE FROM {identifier} WHERE part_key = 0 AND id % 100 = 0",
            # Delete rows from partition 1
            f"DELETE FROM {identifier} WHERE part_key = 1 AND id % 200 = 0",
            # Delete rows from partition 2
            f"DELETE FROM {identifier} WHERE part_key = 2 AND id < 100",
        ],
    )

    tbl = session_catalog.load_table(identifier)

    # Read whole file
    whole = tbl.scan().to_arrow().sort_by("id")

    # Read splits
    split_result = _read_splits_concat(tbl, target_split_size=32 * 1024 * 1024)

    # Assert equality
    assert whole.equals(split_result), "Split reads must equal whole file read for partitioned table with deletes"
    assert len(whole) == len(split_result), f"Row count mismatch: {len(whole)} != {len(split_result)}"


@pytest.mark.integration
@pytest.mark.skipif(not has_orc_fragment_support(), reason="ORC fragment subsetting not available")
def test_concat_splits_equals_whole_file_orc(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Test that concat(all_splits) == read_whole_file for ORC files.

    Validates correctness property for ORC format (when fragment support available).
    """
    identifier = "default.test_split_correctness_orc"

    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (
                id BIGINT,
                data STRING
            )
            USING iceberg
            TBLPROPERTIES(
                'write.format.default' = 'orc',
                'write.orc.stripe.size-bytes' = '8388608'
            )
            """,
            # Insert enough data to create multiple stripes
            f"""
            INSERT INTO {identifier}
            SELECT id, repeat('x', 500) as data
            FROM range(50000)
            """,
        ],
    )

    tbl = session_catalog.load_table(identifier)

    # Read whole file
    whole = tbl.scan().to_arrow().sort_by("id")

    # Read splits
    split_result = _read_splits_concat(tbl, target_split_size=32 * 1024 * 1024)

    # Assert equality
    assert whole.equals(split_result), "ORC split reads must equal whole file read"
    assert len(whole) == len(split_result), f"Row count mismatch: {len(whole)} != {len(split_result)}"
