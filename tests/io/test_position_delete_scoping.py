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
"""Tests for position delete scoping in sub-file splits."""

from unittest.mock import Mock

import pyarrow.dataset as ds
import pytest

from pyiceberg.io.pyarrow import _compute_split_row_range
from pyiceberg.manifest import DataFile, FileFormat


@pytest.fixture
def mock_parquet_fragment_3rg():
    """Mock Parquet fragment with 3 row groups of 500, 500, 500 rows."""
    fragment = Mock(spec=ds.ParquetFileFragment)
    metadata = Mock()

    # Mock 3 row groups
    row_group_mocks = []
    for i in range(3):
        rg = Mock()
        rg.num_rows = 500
        row_group_mocks.append(rg)

    metadata.num_row_groups = 3
    metadata.row_group = lambda i: row_group_mocks[i]
    fragment.metadata = metadata

    return fragment


@pytest.fixture
def mock_parquet_fragment_2rg_uneven():
    """Mock Parquet fragment with 2 row groups: 500 rows, 500 rows."""
    fragment = Mock(spec=ds.ParquetFileFragment)
    metadata = Mock()

    row_group_mocks = [Mock(), Mock()]
    row_group_mocks[0].num_rows = 500
    row_group_mocks[1].num_rows = 500

    metadata.num_row_groups = 2
    metadata.row_group = lambda i: row_group_mocks[i]
    fragment.metadata = metadata

    return fragment


def test_compute_split_row_range_first_split(mock_parquet_fragment_3rg):
    """Test split row range computation for first split of 3-row-group file."""
    task = Mock()
    task.file.split_offsets = [0, 100_000_000, 200_000_000]
    task.file.file_format = FileFormat.PARQUET
    task.file.record_count = 1500
    task.start = 0
    task.length = 100_000_000

    split_start, split_end = _compute_split_row_range(mock_parquet_fragment_3rg, task)

    assert split_start == 0
    assert split_end == 500


def test_compute_split_row_range_second_split(mock_parquet_fragment_3rg):
    """Test split row range computation for second split of 3-row-group file."""
    task = Mock()
    task.file.split_offsets = [0, 100_000_000, 200_000_000]
    task.file.file_format = FileFormat.PARQUET
    task.file.record_count = 1500
    task.start = 100_000_000
    task.length = 100_000_000

    split_start, split_end = _compute_split_row_range(mock_parquet_fragment_3rg, task)

    assert split_start == 500
    assert split_end == 1000


def test_compute_split_row_range_last_split(mock_parquet_fragment_3rg):
    """Test split row range computation for last split of 3-row-group file."""
    task = Mock()
    task.file.split_offsets = [0, 100_000_000, 200_000_000]
    task.file.file_format = FileFormat.PARQUET
    task.file.record_count = 1500
    task.start = 200_000_000
    task.length = 100_000_000

    split_start, split_end = _compute_split_row_range(mock_parquet_fragment_3rg, task)

    assert split_start == 1000
    assert split_end == 1500


def test_compute_split_row_range_whole_file():
    """Test that whole-file read returns (0, record_count) range."""
    fragment = Mock()
    task = Mock()
    task.file.split_offsets = None
    task.file.record_count = 1000
    task.start = None
    task.length = None

    split_start, split_end = _compute_split_row_range(fragment, task)

    assert split_start == 0
    assert split_end == 1000


def test_compute_split_row_range_no_split_offsets():
    """Test that missing split_offsets returns whole-file range."""
    fragment = Mock()
    task = Mock()
    task.file.split_offsets = None
    task.file.record_count = 1500
    task.start = 100_000_000
    task.length = 100_000_000

    split_start, split_end = _compute_split_row_range(fragment, task)

    assert split_start == 0
    assert split_end == 1500


def test_compute_split_row_range_orc():
    """Test split row range computation for ORC files using even distribution."""
    fragment = Mock(spec=ds.OrcFileFragment)
    # Don't set num_stripes attribute to force fallback to len(split_offsets)

    task = Mock()
    task.file.split_offsets = [0, 100_000_000, 200_000_000]
    task.file.file_format = FileFormat.ORC
    task.file.record_count = 1500
    task.start = 100_000_000
    task.length = 100_000_000

    # Remove num_stripes attribute to avoid Mock interference
    if hasattr(fragment, 'num_stripes'):
        delattr(fragment, 'num_stripes')

    split_start, split_end = _compute_split_row_range(fragment, task)

    # Even distribution: 1500 / 3 = 500 per stripe
    assert split_start == 500
    assert split_end == 1000


def test_compute_split_row_range_unsorted_offsets(mock_parquet_fragment_3rg):
    """Test that unsorted split_offsets are handled correctly."""
    task = Mock()
    # Deliberately unsorted
    task.file.split_offsets = [200_000_000, 0, 100_000_000]
    task.file.file_format = FileFormat.PARQUET
    task.file.record_count = 1500
    task.start = 100_000_000
    task.length = 100_000_000

    split_start, split_end = _compute_split_row_range(mock_parquet_fragment_3rg, task)

    # After sorting, offset at index 1 should be 100MB
    assert split_start == 500
    assert split_end == 1000


def test_position_delete_at_split_boundary(mock_parquet_fragment_2rg_uneven):
    """Test that position deletes at split boundary are handled correctly.

    File with 1000 rows (2 row groups: 0-499, 500-999).
    Split covers row group 1 (rows 500-999).
    Delete at position 750 should map to row 250 within split.
    Delete at position 250 should NOT be applied (outside split).
    """
    task = Mock()
    task.file.split_offsets = [0, 100_000_000]
    task.file.file_format = FileFormat.PARQUET
    task.file.record_count = 1000
    task.start = 100_000_000
    task.length = 100_000_000

    split_start, split_end = _compute_split_row_range(mock_parquet_fragment_2rg_uneven, task)

    # Split should cover rows 500-1000
    assert split_start == 500
    assert split_end == 1000

    # Simulate position deletes at 250 and 750
    delete_positions = [250, 750]

    # Only 750 falls in [500, 1000)
    deletes_in_split = [pos for pos in delete_positions if split_start <= pos < split_end]
    assert deletes_in_split == [750]

    # After adjustment to split-relative coordinates:
    # 750 - 500 = 250
    split_relative_deletes = [pos - split_start for pos in deletes_in_split]
    assert split_relative_deletes == [250]


def test_position_delete_outside_split_not_applied():
    """Test that position deletes outside split range are not applied.

    File with 1500 rows split into 3 parts.
    Reading middle split (rows 500-1000).
    Deletes at rows 100 and 1200 should not affect this split.
    """
    fragment = Mock()
    task = Mock()
    task.file.split_offsets = [0, 100_000_000, 200_000_000]
    task.file.file_format = FileFormat.PARQUET
    task.file.record_count = 1500
    task.start = 100_000_000
    task.length = 100_000_000

    # Mock fragment with even 500-row row groups
    metadata = Mock()
    row_group_mocks = []
    for i in range(3):
        rg = Mock()
        rg.num_rows = 500
        row_group_mocks.append(rg)
    metadata.num_row_groups = 3
    metadata.row_group = lambda i: row_group_mocks[i]
    fragment.metadata = metadata

    split_start, split_end = _compute_split_row_range(fragment, task)

    assert split_start == 500
    assert split_end == 1000

    # Deletes outside the split range
    delete_positions = [100, 1200]

    deletes_in_split = [pos for pos in delete_positions if split_start <= pos < split_end]
    assert deletes_in_split == []  # No deletes should apply


def test_position_delete_whole_file_unchanged():
    """Test that position deletes with no start/length work as before.

    Whole-file reads should have split_start_row = 0, so no adjustment.
    """
    fragment = Mock()
    task = Mock()
    task.file.split_offsets = None
    task.file.record_count = 1000
    task.start = None
    task.length = None

    split_start, split_end = _compute_split_row_range(fragment, task)

    assert split_start == 0
    assert split_end == 1000

    # For whole file, file-absolute positions are the same as split-relative
    delete_positions = [100, 500, 900]

    # No adjustment needed - positions can be used directly
    adjusted_positions = [pos - split_start for pos in delete_positions]
    assert adjusted_positions == [100, 500, 900]
