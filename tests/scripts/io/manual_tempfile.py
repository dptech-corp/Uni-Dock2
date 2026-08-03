"""Characterization tests for `unidock2.io.tempfile.TemporaryDirectory`.

`8b121ae`: vendoring `tempfile` for the `delete` keyword on Python 3.10.
It was replaced by a simple implementation for `TemporaryDirectory`.
"""

import gc
import inspect
import os
import sys
import tempfile as stdlib_tempfile

import pytest

from unidock2.io.get_temp_dir_prefix import get_temp_dir_prefix
from unidock2.io.tempfile import TemporaryDirectory

@pytest.fixture
def temp_dir_prefix(tmp_path):
    """An absolute prefix, mirroring how the CLI drives this module.

    `cli/docking.py` joins the user's `temp_dir_name` onto the generated
    prefix, so `prefix` arrives as an absolute path and `dir` is never passed.
    """

    return os.path.join(str(tmp_path), get_temp_dir_prefix('docking'))

def test_signature_matches_the_vendored_stdlib_copy():
    parameter_dict = inspect.signature(TemporaryDirectory.__init__).parameters

    assert list(parameter_dict) == [
        'self',
        'suffix',
        'prefix',
        'dir',
        'ignore_cleanup_errors',
        'delete',
    ]

    assert parameter_dict['delete'].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter_dict['delete'].default is True

def test_stdlib_still_lacks_delete_on_the_oldest_supported_python():
    """The reason this module exists at all.

    `delete` landed in stdlib 3.12. Once Uni-Dock2 drops Python 3.10 and 3.11
    this assertion starts failing, which is the signal to delete this module
    and import `TemporaryDirectory` straight from `tempfile`.
    """

    has_delete = 'delete' in inspect.signature(
        stdlib_tempfile.TemporaryDirectory.__init__
    ).parameters

    assert has_delete == (sys.version_info >= (3, 12))

def test_absolute_prefix_places_the_directory_under_that_parent(temp_dir_prefix):
    with TemporaryDirectory(prefix=temp_dir_prefix, delete=True) as temp_dir_name:
        assert os.path.isdir(temp_dir_name)
        assert os.path.dirname(temp_dir_name) == os.path.dirname(temp_dir_prefix)
        assert os.path.basename(temp_dir_name).startswith(
            os.path.basename(temp_dir_prefix)
        )

def test_repeated_calls_with_one_prefix_never_collide(temp_dir_prefix):
    temp_dir_name_set = {
        TemporaryDirectory(prefix=temp_dir_prefix, delete=False).name
        for _ in range(5)
    }

    assert len(temp_dir_name_set) == 5

def test_delete_true_removes_the_directory_and_its_contents(temp_dir_prefix):
    with TemporaryDirectory(prefix=temp_dir_prefix, delete=True) as temp_dir_name:
        with open(os.path.join(temp_dir_name, 'intermediate.sdf'), 'w') as f:
            f.write('placeholder')

    assert not os.path.exists(temp_dir_name)

def test_delete_false_keeps_the_directory_and_its_contents(temp_dir_prefix):
    with TemporaryDirectory(prefix=temp_dir_prefix, delete=False) as temp_dir_name:
        intermediate_file_name = os.path.join(temp_dir_name, 'intermediate.sdf')
        with open(intermediate_file_name, 'w') as f:
            f.write('placeholder')

    assert os.path.isdir(temp_dir_name)
    assert os.path.isfile(intermediate_file_name)

def test_directory_is_removed_even_when_the_body_raises(temp_dir_prefix):
    with pytest.raises(RuntimeError):
        with TemporaryDirectory(prefix=temp_dir_prefix, delete=True) as temp_dir_name:
            escaped_temp_dir_name = temp_dir_name
            raise RuntimeError('docking failed')

    assert not os.path.exists(escaped_temp_dir_name)

def test_explicit_cleanup_removes_the_directory(temp_dir_prefix):
    temporary_directory = TemporaryDirectory(prefix=temp_dir_prefix, delete=True)
    temp_dir_name = temporary_directory.name

    assert os.path.isdir(temp_dir_name)
    assert repr(temporary_directory) == f'<TemporaryDirectory {temp_dir_name!r}>'

    temporary_directory.cleanup()

    assert not os.path.exists(temp_dir_name)

def test_garbage_collection_cleans_up_when_the_context_manager_is_skipped(
    temp_dir_prefix,
):
    temporary_directory = TemporaryDirectory(prefix=temp_dir_prefix, delete=True)
    temp_dir_name = temporary_directory.name

    del temporary_directory
    gc.collect()

    assert not os.path.exists(temp_dir_name)

def test_garbage_collection_respects_delete_false(temp_dir_prefix):
    temporary_directory = TemporaryDirectory(prefix=temp_dir_prefix, delete=False)
    temp_dir_name = temporary_directory.name

    del temporary_directory
    gc.collect()

    assert os.path.isdir(temp_dir_name)

def test_cli_temp_dir_policy_round_trip(tmp_path):
    """`/tmp` runs are discarded, user-chosen directories are kept.
    """

    for root_temp_dir_name, remove_temp_dir in (
        ('/tmp', True),
        (str(tmp_path), False),
    ):
        temp_dir_prefix = os.path.join(
            root_temp_dir_name, get_temp_dir_prefix('docking')
        )

        with TemporaryDirectory(
            prefix=temp_dir_prefix, delete=remove_temp_dir
        ) as temp_dir_name:
            assert os.path.isdir(temp_dir_name)

        assert os.path.exists(temp_dir_name) is not remove_temp_dir
