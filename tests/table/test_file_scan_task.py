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
"""Tests for FileScanTask byte range fields."""

from pyiceberg.expressions import AlwaysTrue
from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
from pyiceberg.table import FileScanTask


def _make_data_file(path: str = "s3://bucket/table/data/file.parquet") -> DataFile:
    """Create a minimal DataFile for testing."""
    return DataFile.from_args(
        content=DataFileContent.DATA,
        file_path=path,
        file_format=FileFormat.PARQUET,
        partition={},
        record_count=100,
        file_size_in_bytes=1024,
    )


def test_file_scan_task_default_start_length():
    """Test FileScanTask defaults start and length to None for backward compatibility."""
    data_file = _make_data_file()
    task = FileScanTask(data_file=data_file)

    assert task.start is None
    assert task.length is None
    assert task.file == data_file


def test_file_scan_task_with_byte_range():
    """Test FileScanTask can be constructed with explicit byte range."""
    data_file = _make_data_file()
    task = FileScanTask(data_file=data_file, start=1024, length=4096)

    assert task.start == 1024
    assert task.length == 4096
    assert task.file == data_file


def test_file_scan_task_with_start_only():
    """Test FileScanTask with start but no length."""
    data_file = _make_data_file()
    task = FileScanTask(data_file=data_file, start=512)

    assert task.start == 512
    assert task.length is None


def test_file_scan_task_with_length_only():
    """Test FileScanTask with length but no start."""
    data_file = _make_data_file()
    task = FileScanTask(data_file=data_file, length=2048)

    assert task.start is None
    assert task.length == 2048


def test_file_scan_task_with_zero_start():
    """Test FileScanTask with start=0 (not confused with None)."""
    data_file = _make_data_file()
    task = FileScanTask(data_file=data_file, start=0, length=1024)

    # Critical: 0 is a valid start value, not None
    assert task.start is not None
    assert task.start == 0
    assert task.length == 1024


def test_file_scan_task_preserves_existing_fields():
    """Test FileScanTask preserves all fields including byte range."""
    data_file = _make_data_file()
    delete_file = _make_data_file("s3://bucket/table/deletes/delete.parquet")
    delete_files = {delete_file}
    residual = AlwaysTrue()

    task = FileScanTask(
        data_file=data_file,
        delete_files=delete_files,
        residual=residual,
        start=2048,
        length=8192,
    )

    assert task.file == data_file
    assert task.delete_files == delete_files
    assert task.residual == residual
    assert task.start == 2048
    assert task.length == 8192
