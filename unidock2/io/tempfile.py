"""Temporary directory helper that supports opting out of cleanup.

``tempfile.TemporaryDirectory`` only gained the ``delete`` keyword in Python
3.12, while Uni-Dock2 supports Python 3.10 onwards. Only that one behaviour is
reimplemented here, on top of ``tempfile.mkdtemp``, so that callers get the
same API on every supported interpreter.

``delete=False`` keeps the intermediate docking files behind for inspection
when the user points ``temp_dir_name`` at a directory of their own.
"""

import os
import shutil
import tempfile
import weakref

__all__ = ['TemporaryDirectory']

class TemporaryDirectory:
    """Create and optionally clean up a temporary directory.

    Optional Arguments:
        suffix - A str suffix for the directory name.  (see mkdtemp)
        prefix - A str prefix for the directory name.  (see mkdtemp)
        dir - A directory to create this temp dir in.  (see mkdtemp)
        ignore_cleanup_errors - False; ignore exceptions during cleanup?
        delete - True; whether the directory is automatically deleted.
    """

    def __init__(self, suffix=None, prefix=None, dir=None,
                 ignore_cleanup_errors=False, *, delete=True):
        self.name = tempfile.mkdtemp(suffix, prefix, dir)
        self._delete = delete
        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._finalizer = weakref.finalize(
            self, self._rmtree, self.name, ignore_cleanup_errors, delete
        )

    @staticmethod
    def _rmtree(name, ignore_errors, delete):
        if delete:
            shutil.rmtree(name, ignore_errors=ignore_errors)

    def cleanup(self):
        if self._finalizer.detach() or os.path.exists(self.name):
            shutil.rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)

    def __enter__(self):
        return self.name

    def __exit__(self, exc, value, tb):
        if self._delete:
            self.cleanup()

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.name!r}>'
