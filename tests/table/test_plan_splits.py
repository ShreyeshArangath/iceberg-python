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
"""Tests for plan_splits() and _split_file_scan_task()."""

from pyiceberg.expressions import AlwaysTrue, EqualTo
from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
from pyiceberg.table import FileScanTask, _split_file_scan_task


def _make_data_file(
    path: str = "s3://bucket/table/data/file.parquet",
    file_format: FileFormat = FileFormat.PARQUET,
    file_size: int = 512 * 1024 * 1024,  # 512 MB
    split_offsets: list[int] | None = None,
) -> DataFile:
    """Create a DataFile for testing with configurable split metadata."""
    return DataFile.from_args(
        content=DataFileContent.DATA,
        file_path=path,
        file_format=file_format,
        partition={},
        record_count=10000,
        file_size_in_bytes=file_size,
        split_offsets=split_offsets,
    )


def test_split_file_no_split_offsets():
    """Test _split_file_scan_task with no split_offsets yields whole-file task."""
    data_file = _make_data_file(split_offsets=None)
    task = FileScanTask(data_file=data_file)

    splits = list(_split_file_scan_task(task, target_split_size=128 * 1024 * 1024))

    assert len(splits) == 1
    assert splits[0].start is None
    assert splits[0].length is None
    assert splits[0].file == data_file


def test_split_file_smaller_than_target():
    """Test _split_file_scan_task with file smaller than target yields whole-file task."""
    # 64 MB file with target of 128 MB
    data_file = _make_data_file(
        file_size=64 * 1024 * 1024,
        split_offsets=[0, 32 * 1024 * 1024],
    )
    task = FileScanTask(data_file=data_file)

    splits = list(_split_file_scan_task(task, target_split_size=128 * 1024 * 1024))

    assert len(splits) == 1
    assert splits[0].start is None
    assert splits[0].length is None


def test_split_file_unsupported_format():
    """Test _split_file_scan_task with AVRO format yields whole-file task."""
    data_file = _make_data_file(
        file_format=FileFormat.AVRO,
        split_offsets=[0, 128 * 1024 * 1024, 256 * 1024 * 1024],
    )
    task = FileScanTask(data_file=data_file)

    splits = list(_split_file_scan_task(task, target_split_size=128 * 1024 * 1024))

    assert len(splits) == 1
    assert splits[0].start is None
    assert splits[0].length is None


def test_split_file_basic_parquet():
    """Test _split_file_scan_task splits Parquet file at row group boundaries."""
    # 512 MB file with 4 row groups of 128 MB each, target = 128 MB
    MB_128 = 128 * 1024 * 1024
    data_file = _make_data_file(
        file_format=FileFormat.PARQUET,
        file_size=512 * 1024 * 1024,
        split_offsets=[0, MB_128, MB_128 * 2, MB_128 * 3],
    )
    task = FileScanTask(data_file=data_file)

    splits = list(_split_file_scan_task(task, target_split_size=MB_128))

    assert len(splits) == 4
    # First split: [0, 128MB)
    assert splits[0].start == 0
    assert splits[0].length == MB_128
    # Second split: [128MB, 256MB)
    assert splits[1].start == MB_128
    assert splits[1].length == MB_128
    # Third split: [256MB, 384MB)
    assert splits[2].start == MB_128 * 2
    assert splits[2].length == MB_128
    # Fourth split: [384MB, 512MB)
    assert splits[3].start == MB_128 * 3
    assert splits[3].length == MB_128


def test_split_file_basic_orc():
    """Test _split_file_scan_task splits ORC file at stripe boundaries."""
    # 512 MB file with 4 stripes of 128 MB each, target = 128 MB
    MB_128 = 128 * 1024 * 1024
    data_file = _make_data_file(
        file_format=FileFormat.ORC,
        file_size=512 * 1024 * 1024,
        split_offsets=[0, MB_128, MB_128 * 2, MB_128 * 3],
    )
    task = FileScanTask(data_file=data_file)

    splits = list(_split_file_scan_task(task, target_split_size=MB_128))

    assert len(splits) == 4
    # Verify same split structure as Parquet
    assert splits[0].start == 0
    assert splits[0].length == MB_128
    assert splits[1].start == MB_128
    assert splits[1].length == MB_128


def test_split_file_groups_small_row_groups():
    """Test _split_file_scan_task groups small row groups together."""
    # 200 MB file with 10 row groups of ~20 MB each, target = 128 MB
    # Expected: group first 7 offsets (~140MB), then remaining 3 (~60MB)
    MB_20 = 20 * 1024 * 1024
    offsets = [i * MB_20 for i in range(10)]
    data_file = _make_data_file(
        file_size=200 * 1024 * 1024,
        split_offsets=offsets,
    )
    task = FileScanTask(data_file=data_file)

    splits = list(_split_file_scan_task(task, target_split_size=128 * 1024 * 1024))

    # Greedy algorithm: accumulate until >= 128MB, then emit
    # [0, 140MB): contains offsets 0-6 (7 row groups)
    # [140MB, 200MB): contains offsets 7-9 (3 row groups)
    assert len(splits) == 2
    assert splits[0].start == 0
    assert splits[0].length == 7 * MB_20  # 140 MB
    assert splits[1].start == 7 * MB_20
    assert splits[1].length == 3 * MB_20  # 60 MB


def test_split_file_single_row_group():
    """Test _split_file_scan_task with single row group yields one split."""
    data_file = _make_data_file(
        file_size=512 * 1024 * 1024,
        split_offsets=[0],
    )
    task = FileScanTask(data_file=data_file)

    splits = list(_split_file_scan_task(task, target_split_size=128 * 1024 * 1024))

    assert len(splits) == 1
    assert splits[0].start == 0
    assert splits[0].length == 512 * 1024 * 1024


def test_split_file_preserves_delete_files():
    """Test _split_file_scan_task preserves delete_files in splits."""
    MB_128 = 128 * 1024 * 1024
    data_file = _make_data_file(
        file_size=256 * 1024 * 1024,
        split_offsets=[0, MB_128],
    )
    delete_file = _make_data_file(
        path="s3://bucket/table/deletes/delete.parquet",
        file_size=1024,
    )
    task = FileScanTask(
        data_file=data_file,
        delete_files={delete_file},
    )

    splits = list(_split_file_scan_task(task, target_split_size=MB_128))

    assert len(splits) == 2
    for split in splits:
        assert split.delete_files == {delete_file}


def test_split_file_preserves_residual():
    """Test _split_file_scan_task preserves residual expression in splits."""
    MB_128 = 128 * 1024 * 1024
    data_file = _make_data_file(
        file_size=256 * 1024 * 1024,
        split_offsets=[0, MB_128],
    )
    residual = EqualTo("id", 42)
    task = FileScanTask(
        data_file=data_file,
        residual=residual,
    )

    splits = list(_split_file_scan_task(task, target_split_size=MB_128))

    assert len(splits) == 2
    for split in splits:
        assert split.residual == residual


def test_split_file_coverage():
    """Test _split_file_scan_task splits cover entire file without gaps or overlaps."""
    MB_128 = 128 * 1024 * 1024
    file_size = 512 * 1024 * 1024
    data_file = _make_data_file(
        file_size=file_size,
        split_offsets=[0, MB_128, MB_128 * 2, MB_128 * 3],
    )
    task = FileScanTask(data_file=data_file)

    splits = list(_split_file_scan_task(task, target_split_size=MB_128))

    # First split starts at 0
    assert splits[0].start == 0

    # Last split ends at file size
    last_split = splits[-1]
    assert last_split.start + last_split.length == file_size

    # No gaps or overlaps: each split starts where previous ended
    for i in range(1, len(splits)):
        prev_split = splits[i - 1]
        curr_split = splits[i]
        assert curr_split.start == prev_split.start + prev_split.length


def test_split_file_empty_split_offsets():
    """Test _split_file_scan_task with empty split_offsets list yields whole-file task."""
    data_file = _make_data_file(
        split_offsets=[],
    )
    task = FileScanTask(data_file=data_file)

    splits = list(_split_file_scan_task(task, target_split_size=128 * 1024 * 1024))

    # Empty split_offsets means no row group metadata, should return whole file
    # Greedy algorithm will yield one split covering [0, file_size)
    assert len(splits) == 1
    assert splits[0].start == 0
    assert splits[0].length == data_file.file_size_in_bytes


def test_split_file_unsorted_offsets():
    """Test _split_file_scan_task handles unsorted split_offsets correctly."""
    MB_128 = 128 * 1024 * 1024
    # Provide offsets in unsorted order
    data_file = _make_data_file(
        file_size=512 * 1024 * 1024,
        split_offsets=[MB_128 * 2, 0, MB_128 * 3, MB_128],  # Unsorted!
    )
    task = FileScanTask(data_file=data_file)

    splits = list(_split_file_scan_task(task, target_split_size=MB_128))

    # Should sort internally and produce correct splits
    assert len(splits) == 4
    assert splits[0].start == 0
    assert splits[0].length == MB_128
    assert splits[1].start == MB_128
    assert splits[2].start == MB_128 * 2
    assert splits[3].start == MB_128 * 3
