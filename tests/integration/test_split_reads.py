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
"""Integration tests for plan_splits() end-to-end execution and backward compatibility."""

import pyarrow as pa
import pytest
from pyspark.sql import SparkSession

from pyiceberg.catalog.rest import RestCatalog
from pyiceberg.io.pyarrow import ArrowScan


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


@pytest.mark.integration
def test_plan_splits_produces_multiple_tasks_parquet(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Test that plan_splits() produces multiple FileScanTasks from a single large Parquet file."""
    identifier = "default.test_plan_splits_multiple_tasks"

    # Force smaller row groups (32MB) so the file has multiple row groups
    spark.conf.set("parquet.block.size", str(32 * 1024 * 1024))

    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (
                id BIGINT,
                padding STRING
            )
            USING iceberg
            TBLPROPERTIES('write.parquet.row-group-size-bytes' = '33554432')
            """,
            # Insert 100k rows with 1KB padding each = ~100MB of data
            # This should create multiple 32MB row groups
            f"""
            INSERT INTO {identifier}
            SELECT id, repeat('x', 1000) as padding
            FROM range(100000)
            """,
        ],
    )

    tbl = session_catalog.load_table(identifier)

    # Plan splits with 64MB target - should split the file
    split_tasks = list(tbl.scan().plan_splits(target_split_size=64 * 1024 * 1024))

    # Assert multiple splits were created
    assert len(split_tasks) > 1, f"Expected multiple splits, got {len(split_tasks)}"

    # Assert all tasks have start and length populated
    for task in split_tasks:
        assert task.start is not None, "Split task should have start offset"
        assert task.length is not None, "Split task should have length"

    # First task should start at offset 0
    assert split_tasks[0].start == 0, f"First split should start at 0, got {split_tasks[0].start}"


@pytest.mark.integration
def test_plan_splits_reads_correct_data_parquet(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Test that plan_splits() → ArrowScan reads correct data end-to-end for Parquet files."""
    identifier = "default.test_plan_splits_correct_data"

    # Force smaller row groups to create splits
    spark.conf.set("parquet.block.size", str(32 * 1024 * 1024))

    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (
                id BIGINT,
                value STRING
            )
            USING iceberg
            TBLPROPERTIES('write.parquet.row-group-size-bytes' = '33554432')
            """,
            # Insert enough data to create multiple row groups
            f"""
            INSERT INTO {identifier}
            SELECT id, CONCAT('value_', CAST(id AS STRING), repeat('x', 1000)) as value
            FROM range(50000)
            """,
        ],
    )

    tbl = session_catalog.load_table(identifier)

    # Read whole file using standard to_arrow()
    whole = tbl.scan().to_arrow().sort_by("id")

    # Read via plan_splits and concatenate results
    split_tasks = list(tbl.scan().plan_splits(target_split_size=64 * 1024 * 1024))

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
        task_batches = list(arrow_scan._task_to_record_batches(task))
        batches.extend(task_batches)

    # Concatenate and sort
    split_result = pa.Table.from_batches(batches).sort_by("id")

    # Assert data matches
    assert whole.equals(split_result), "Split reads should produce identical data to whole file read"


@pytest.mark.integration
def test_plan_files_backward_compatible(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Test that existing plan_files() → ArrowScan path works unchanged (backward compatibility)."""
    identifier = "default.test_plan_files_backward_compat"

    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (
                id BIGINT,
                name STRING
            )
            USING iceberg
            """,
            f"""
            INSERT INTO {identifier}
            VALUES (1, 'alice'), (2, 'bob'), (3, 'charlie')
            """,
        ],
    )

    tbl = session_catalog.load_table(identifier)

    # Use plan_files() - the existing code path
    tasks = list(tbl.scan().plan_files())

    # Assert tasks have no start/length (whole file tasks)
    for task in tasks:
        assert task.start is None, "plan_files() tasks should not have start offset"
        assert task.length is None, "plan_files() tasks should not have length"

    # Read to_arrow() and verify data
    result = tbl.scan().to_arrow().sort_by("id")

    expected = pa.Table.from_pydict({
        "id": [1, 2, 3],
        "name": ["alice", "bob", "charlie"],
    })

    assert result.equals(expected), "plan_files() should read correct data unchanged"


@pytest.mark.integration
def test_plan_splits_single_small_file_returns_one_task(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Test that plan_splits() returns one task for files smaller than target split size."""
    identifier = "default.test_plan_splits_small_file"

    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (
                id BIGINT,
                value STRING
            )
            USING iceberg
            """,
            # Small data - 5 rows
            f"""
            INSERT INTO {identifier}
            VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')
            """,
        ],
    )

    tbl = session_catalog.load_table(identifier)

    # Plan splits with default 128MB target - file too small to split
    split_tasks = list(tbl.scan().plan_splits())

    # Should return exactly one task
    assert len(split_tasks) == 1, f"Small file should produce 1 task, got {len(split_tasks)}"

    # Task should still have start and length populated (just one split covering whole file)
    task = split_tasks[0]
    assert task.start is not None, "Task should have start offset"
    assert task.length is not None, "Task should have length"

    # Verify data reads correctly
    result = tbl.scan().to_arrow().sort_by("id")
    assert len(result) == 5, "Should read all 5 rows"


@pytest.mark.integration
@pytest.mark.skipif(not has_orc_fragment_support(), reason="ORC fragment subsetting not available")
def test_plan_splits_reads_correct_data_orc(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Test that plan_splits() → ArrowScan reads correct data end-to-end for ORC files."""
    identifier = "default.test_plan_splits_correct_data_orc"

    # Configure Spark to write ORC format
    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (
                id BIGINT,
                value STRING
            )
            USING iceberg
            TBLPROPERTIES(
                'write.format.default' = 'orc',
                'write.orc.stripe.size-bytes' = '33554432'
            )
            """,
            # Insert enough data to create multiple stripes
            f"""
            INSERT INTO {identifier}
            SELECT id, CONCAT('value_', CAST(id AS STRING), repeat('x', 1000)) as value
            FROM range(50000)
            """,
        ],
    )

    tbl = session_catalog.load_table(identifier)

    # Read whole file using standard to_arrow()
    whole = tbl.scan().to_arrow().sort_by("id")

    # Read via plan_splits and concatenate results
    split_tasks = list(tbl.scan().plan_splits(target_split_size=64 * 1024 * 1024))

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
        task_batches = list(arrow_scan._task_to_record_batches(task))
        batches.extend(task_batches)

    # Concatenate and sort
    split_result = pa.Table.from_batches(batches).sort_by("id")

    # Assert data matches
    assert whole.equals(split_result), "ORC split reads should produce identical data to whole file read"
