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
"""Tests for fragment subsetting in sub-file split reads."""

from unittest.mock import Mock

import pytest

from pyiceberg.io.pyarrow import _subset_fragment_by_byte_range
from pyiceberg.manifest import DataFile, FileFormat
from pyiceberg.table import FileScanTask


def test_subset_fragment_parquet_basic():
    """Test subsetting Parquet fragment with single row group in byte range."""
    # Create a FileScanTask with byte range covering first row group
    # split_offsets=[0, 128MB, 256MB] means 3 row groups starting at these bytes
    # byte range [0, 128MB) should include only row group 0
    task = FileScanTask(
        data_file=DataFile.from_args(
            content=1,  # DATA
            file_path="/test.parquet",
            file_format=FileFormat.PARQUET,
            partition={},
            record_count=1000,
            file_size_in_bytes=256 * 1024 * 1024,
            split_offsets=[0, 128 * 1024 * 1024, 256 * 1024 * 1024],
        ),
        delete_files=set(),
        start=0,
        length=128 * 1024 * 1024,
    )

    # Create a mock fragment
    fragment = Mock()
    subsetted_fragment = Mock()
    fragment.subset = Mock(return_value=subsetted_fragment)

    # Call the function
    result = _subset_fragment_by_byte_range(fragment, task)

    # Verify subset was called with correct row_group_ids
    fragment.subset.assert_called_once_with(row_group_ids=[0])
    assert result == subsetted_fragment


def test_subset_fragment_parquet_middle_split():
    """Test subsetting Parquet fragment with middle row group."""
    # byte range [128MB, 256MB) should include only row group 1 (offset 128MB)
    task = FileScanTask(
        data_file=DataFile.from_args(
            content=1,
            file_path="/test.parquet",
            file_format=FileFormat.PARQUET,
            partition={},
            record_count=1000,
            file_size_in_bytes=256 * 1024 * 1024,
            split_offsets=[0, 128 * 1024 * 1024, 256 * 1024 * 1024],
        ),
        delete_files=set(),
        start=128 * 1024 * 1024,
        length=128 * 1024 * 1024,
    )

    fragment = Mock()
    subsetted_fragment = Mock()
    fragment.subset = Mock(return_value=subsetted_fragment)

    result = _subset_fragment_by_byte_range(fragment, task)

    # Should include row group 1 (offset 128MB is in range [128MB, 256MB))
    fragment.subset.assert_called_once_with(row_group_ids=[1])
    assert result == subsetted_fragment


def test_subset_fragment_orc_basic():
    """Test subsetting ORC fragment with stripe_ids parameter."""
    task = FileScanTask(
        data_file=DataFile.from_args(
            content=1,
            file_path="/test.orc",
            file_format=FileFormat.ORC,
            partition={},
            record_count=1000,
            file_size_in_bytes=256 * 1024 * 1024,
            split_offsets=[0, 128 * 1024 * 1024, 256 * 1024 * 1024],
        ),
        delete_files=set(),
        start=0,
        length=128 * 1024 * 1024,
    )

    fragment = Mock()
    subsetted_fragment = Mock()
    fragment.subset = Mock(return_value=subsetted_fragment)

    result = _subset_fragment_by_byte_range(fragment, task)

    # Should use stripe_ids for ORC
    fragment.subset.assert_called_once_with(stripe_ids=[0])
    assert result == subsetted_fragment


def test_subset_fragment_no_split_offsets():
    """Test that fragment is returned unchanged when split_offsets is None."""
    task = FileScanTask(
        data_file=DataFile.from_args(
            content=1,
            file_path="/test.parquet",
            file_format=FileFormat.PARQUET,
            partition={},
            record_count=1000,
            file_size_in_bytes=256 * 1024 * 1024,
            split_offsets=None,  # No metadata
        ),
        delete_files=set(),
        start=0,
        length=128 * 1024 * 1024,
    )

    fragment = Mock()
    fragment.subset = Mock()

    result = _subset_fragment_by_byte_range(fragment, task)

    # Should return original fragment without calling subset
    fragment.subset.assert_not_called()
    assert result == fragment


def test_subset_fragment_multiple_row_groups_in_range():
    """Test subsetting when byte range covers multiple row groups."""
    # split_offsets=[0, 50MB, 100MB, 150MB, 200MB]
    # byte range [0, 150MB) should include row groups 0, 1, 2
    # (offsets 0, 50MB, 100MB are all in [0, 150MB))
    # offset 150MB is NOT included (exclusive end)
    task = FileScanTask(
        data_file=DataFile.from_args(
            content=1,
            file_path="/test.parquet",
            file_format=FileFormat.PARQUET,
            partition={},
            record_count=2000,
            file_size_in_bytes=200 * 1024 * 1024,
            split_offsets=[
                0,
                50 * 1024 * 1024,
                100 * 1024 * 1024,
                150 * 1024 * 1024,
                200 * 1024 * 1024,
            ],
        ),
        delete_files=set(),
        start=0,
        length=150 * 1024 * 1024,
    )

    fragment = Mock()
    subsetted_fragment = Mock()
    fragment.subset = Mock(return_value=subsetted_fragment)

    result = _subset_fragment_by_byte_range(fragment, task)

    # Should include row groups 0, 1, 2 (offsets < 150MB)
    fragment.subset.assert_called_once_with(row_group_ids=[0, 1, 2])
    assert result == subsetted_fragment


def test_subset_fragment_empty_id_list_fallback():
    """Test fallback to full read when no row groups match byte range."""
    # This edge case: byte range doesn't align with any split offset
    # In practice this shouldn't happen with proper planning, but test defensive behavior
    task = FileScanTask(
        data_file=DataFile.from_args(
            content=1,
            file_path="/test.parquet",
            file_format=FileFormat.PARQUET,
            partition={},
            record_count=1000,
            file_size_in_bytes=256 * 1024 * 1024,
            split_offsets=[0, 128 * 1024 * 1024, 256 * 1024 * 1024],
        ),
        delete_files=set(),
        start=64 * 1024 * 1024,  # Between offsets 0 and 128MB
        length=32 * 1024 * 1024,  # Ends before 128MB
    )

    fragment = Mock()
    fragment.subset = Mock()

    result = _subset_fragment_by_byte_range(fragment, task)

    # Should return original fragment without calling subset (fallback)
    fragment.subset.assert_not_called()
    assert result == fragment


def test_subset_fragment_unsorted_split_offsets():
    """Test that function handles unsorted split_offsets by sorting them."""
    # split_offsets may not be guaranteed sorted in metadata
    task = FileScanTask(
        data_file=DataFile.from_args(
            content=1,
            file_path="/test.parquet",
            file_format=FileFormat.PARQUET,
            partition={},
            record_count=1000,
            file_size_in_bytes=256 * 1024 * 1024,
            split_offsets=[128 * 1024 * 1024, 0, 256 * 1024 * 1024],  # Unsorted
        ),
        delete_files=set(),
        start=0,
        length=128 * 1024 * 1024,
    )

    fragment = Mock()
    subsetted_fragment = Mock()
    fragment.subset = Mock(return_value=subsetted_fragment)

    result = _subset_fragment_by_byte_range(fragment, task)

    # Should sort offsets and correctly identify row group 0
    # After sorting: [0, 128MB, 256MB], so offset 0 is at index 0
    fragment.subset.assert_called_once_with(row_group_ids=[0])
    assert result == subsetted_fragment


def test_subset_fragment_avro_unsupported_format():
    """Test that unsupported file formats return fragment unchanged."""
    task = FileScanTask(
        data_file=DataFile.from_args(
            content=1,
            file_path="/test.avro",
            file_format=FileFormat.AVRO,
            partition={},
            record_count=1000,
            file_size_in_bytes=256 * 1024 * 1024,
            split_offsets=[0, 128 * 1024 * 1024, 256 * 1024 * 1024],
        ),
        delete_files=set(),
        start=0,
        length=128 * 1024 * 1024,
    )

    fragment = Mock()
    fragment.subset = Mock()

    result = _subset_fragment_by_byte_range(fragment, task)

    # Should return original fragment without calling subset (unsupported format)
    fragment.subset.assert_not_called()
    assert result == fragment
