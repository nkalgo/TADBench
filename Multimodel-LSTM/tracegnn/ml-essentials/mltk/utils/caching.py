import os
import shutil
import sys
from contextlib import contextmanager
from typing import *
from urllib.parse import urlparse

import requests
from filelock import FileLock
from tqdm import tqdm

from .archive_utils import Extractor

__all__ = ['CacheDir']

_cache_root: str = None


@contextmanager
def _maybe_tqdm(tqdm_enabled: bool, **kwargs
                ) -> Generator[Optional[tqdm], None, None]:
    if tqdm_enabled:
        with tqdm(**kwargs) as t:
            yield t
    else:
        yield None


def _guess_show_progress_arg(progress_file,
                             show_progress: bool) -> bool:  # pragma: no cover
    if show_progress is None:
        if hasattr(progress_file, 'isatty'):
            return progress_file.isatty()
        else:
            return False
    else:
        return show_progress


def _guess_filename_from_uri(uri: str) -> str:
    parsed_uri = urlparse(uri)
    filename = parsed_uri.path.rsplit('/', 1)[-1]
    if not filename:  # pragma: no cover
        raise ValueError('`filename` cannot be inferred.')
    return filename


def _guess_extract_dir_from_filename(filename: str) -> str:
    extract_dir = filename.split('.', 1)[0]
    if not extract_dir:  # pragma: no cover
        raise ValueError('`extract_dir` cannot be inferred.')
    return extract_dir


class CacheDir(object):
    """Class to manipulate a cache directory."""

    def __init__(self, name: str, cache_root: Optional[str] = None):
        """
        Construct a new :class:`CacheDir`.

        Args:
            name: The name of the sub-directory under `cache_root`.
            cache_root: The cache root directory.  If not specified,
                use ``mltk.settings.cache_root``.
        """
        from mltk import settings

        if not name:
            raise ValueError('`name` is required.')
        if cache_root is None:
            cache_root = settings.cache_root
        self._name = name
        self._cache_root = os.path.abspath(cache_root)
        self._path = os.path.abspath(os.path.join(self._cache_root, name))

    @property
    def name(self) -> str:
        """Get the name of this cache directory under `cache_root`."""
        return self._name

    @property
    def cache_root(self) -> str:
        """Get the cache root directory."""
        return self._cache_root

    @property
    def path(self) -> str:
        """Get the absolute path of this cache directory."""
        return self._path

    def resolve(self, sub_path: str) -> str:
        """
        Resolve a sub path relative to ``self.path``.

        Args:
            sub_path: The sub path to resolve.

        Returns:
            The resolved absolute path of `sub_path`.
        """
        return os.path.join(self.path, sub_path)

    @contextmanager
    def _lock_file(self, file_path):
        lock_file = file_path + '.lock'
        parent_dir = os.path.split(lock_file)[0]
        os.makedirs(parent_dir, exist_ok=True)
        with FileLock(lock_file):
            yield

    def _download(self,
                  uri: str,
                  file_path: str,
                  show_progress: bool,
                  progress_file,
                  hasher=None,
                  expected_hash: Optional[str] = None) -> str:
        from mltk import settings

        if os.path.isfile(file_path):
            if settings.file_cache_checksum and hasher is not None:
                with open(file_path, 'rb') as f:
                    chunk = bytearray(8192)
                    n_bytes = f.readinto(chunk)
                    while n_bytes > 0:
                        hasher.update(chunk[:n_bytes])
                        n_bytes = f.readinto(chunk)

                got_hash = hasher.hexdigest()
                if got_hash != expected_hash:
                    os.remove(file_path)
                    raise IOError(
                        f'Hash not match for cached file {file_path}: '
                        f'{got_hash} vs expected {expected_hash}'
                    )

        else:
            temp_file = file_path + '._downloading_'
            try:
                if not show_progress:
                    progress_file.write('Downloading {} ... '.format(uri))
                    progress_file.flush()
                with _maybe_tqdm(tqdm_enabled=show_progress,
                                 desc='Downloading {}'.format(uri),
                                 unit='B', unit_scale=True, unit_divisor=1024,
                                 miniters=1, file=progress_file) as t, \
                        open(temp_file, 'wb') as f:
                    req = requests.get(uri, stream=True)
                    if req.status_code != 200:
                        raise IOError('HTTP Error {}: {}'.
                                      format(req.status_code, req.content))

                    # detect the total length
                    if t is not None:
                        cont_length = req.headers.get('Content-Length')
                        if cont_length is not None:
                            try:
                                t.total = int(cont_length)
                            except ValueError:  # pragma: no cover
                                pass

                    # do download the content
                    for chunk in req.iter_content(8192):
                        if chunk:
                            f.write(chunk)
                            if hasher is not None:
                                hasher.update(chunk)
                            if t is not None:
                                t.update(len(chunk))

                    if hasher is not None:
                        got_hash = hasher.hexdigest()
                        if got_hash != expected_hash:
                            raise IOError(
                                'Hash not match for file downloaded from {}: '
                                '{} vs expected {}'.
                                format(uri, got_hash, expected_hash)
                            )

            except BaseException:
                if not show_progress:
                    progress_file.write('error\n')
                    progress_file.flush()
                if os.path.isfile(temp_file):  # pragma: no cover
                    os.remove(temp_file)
                raise
            else:
                if not show_progress:
                    progress_file.write('ok\n')
                    progress_file.flush()
                os.rename(temp_file, file_path)
        return file_path

    def download(self,
                 uri: str,
                 filename: Optional[str] = None,
                 show_progress: Optional[bool] = None,
                 progress_file=sys.stderr,
                 hasher=None,
                 expected_hash: Optional[str] = None) -> str:
        """
        Download a file into this :class:`CacheDir`.

        Args:
            uri: The URI to be retrieved.
            filename: The filename to use as the downloaded file.
                If `filename` already exists in this :class:`CacheDir`,
                will not download `uri`.  Default :obj:`None`, will
                automatically infer `filename` according to `uri`.
            show_progress: Whether or not to show interactive progress bar?
                If not specified, will show progress only when `progress_file`
                is `std.stdout` or `std.stderr`, and `progress_file.isatty()`
                is :obj:`True`.
            progress_file: The file object where to write the progress.
                (default :obj:`sys.stderr`)
            hasher: A hasher algorithm instance from `hashlib`.
                If specified, will compute the hash of downloaded content,
                and validate against `expected_hash`.
            expected_hash: The expected hash of downloaded content.

        Returns:
            The absolute path of the downloaded file.

        Raises:
            ValueError: If `filename` cannot be inferred.
        """
        # check the arguments
        show_progress = _guess_show_progress_arg(progress_file, show_progress)

        if filename is None:
            filename = _guess_filename_from_uri(uri)
        file_path = os.path.abspath(os.path.join(self.path, filename))

        # download the file
        with self._lock_file(file_path):
            return self._download(
                uri, file_path, show_progress=show_progress,
                progress_file=progress_file, hasher=hasher,
                expected_hash=expected_hash
            )

    def _extract_file(self,
                      archive_file,
                      extract_path: str,
                      show_progress: bool,
                      progress_file) -> str:
        if not os.path.isdir(extract_path):
            temp_path = extract_path + '._extracting_'
            progress_file.write('Extracting {} ... '.format(archive_file))
            progress_file.flush()
            try:
                with Extractor.open(archive_file) as extractor:
                    for name, file_obj in extractor:
                        file_path = os.path.join(temp_path, name)
                        file_dir = os.path.split(file_path)[0]
                        os.makedirs(file_dir, exist_ok=True)
                        with open(file_path, 'wb') as dst_obj:
                            shutil.copyfileobj(file_obj, dst_obj)
            except BaseException:
                progress_file.write('error\n')
                progress_file.flush()
                if os.path.isdir(temp_path):  # pragma: no cover
                    shutil.rmtree(temp_path)
                raise
            else:
                progress_file.write('ok\n')
                progress_file.flush()
                os.rename(temp_path, extract_path)
        return extract_path

    def extract_file(self,
                     archive_file,
                     extract_dir: Optional[str] = None,
                     show_progress: Optional[bool] = None,
                     progress_file=sys.stderr) -> str:
        """
        Extract an archive file into this :class:`CacheDir`.

        Args:
            archive_file: The path of the archive file.
            extract_dir: The name to use as the extracted directory.
                If `extract_dir` already exists in this :class:`CacheDir`,
                will not extract `archive_file`.  Default :obj:`None`, will
                automatically infer `extract_dir` according to `archive_file`.
            show_progress: Whether or not to show interactive progress bar?
                If not specified, will show progress only when `progress_file`
                is `std.stdout` or `std.stderr`, and `progress_file.isatty()`
                is :obj:`True`.
            progress_file: The file object where to write the progress.
                (default :obj:`sys.stderr`)

        Returns:
            The absolute path of the extracted directory.

        Raises:
            ValueError: If `extract_dir` cannot be inferred.
        """
        # check the arguments
        show_progress = _guess_show_progress_arg(progress_file, show_progress)

        archive_file = os.path.abspath(archive_file)
        filename = os.path.split(archive_file)[-1]
        if extract_dir is None:
            extract_dir = _guess_extract_dir_from_filename(filename)
        extract_path = os.path.abspath(os.path.join(self.path, extract_dir))

        # extract the file
        with self._lock_file(archive_file):
            return self._extract_file(
                archive_file, extract_path, show_progress=show_progress,
                progress_file=progress_file
            )

    def download_and_extract(self,
                             uri: str,
                             filename: Optional[str] = None,
                             extract_dir: Optional[str] = None,
                             show_progress: Optional[bool] = None,
                             progress_file=sys.stderr,
                             hasher=None,
                             expected_hash: Optional[str] = None) -> str:
        """
        Download a file into this :class:`CacheDir`, and extract it.

        Args:
            uri: The URI to be retrieved.
            filename: The filename to use as the downloaded file.
                If `filename` already exists in this :class:`CacheDir`,
                will not download `uri`.  Default :obj:`None`, will
                automatically infer `filename` according to `uri`.
            extract_dir: The name to use as the extracted directory.
                If `extract_dir` already exists in this :class:`CacheDir`,
                will not extract `archive_file`.  Default :obj:`None`, will
                automatically infer `extract_dir` according to `archive_file`.
            show_progress: Whether or not to show interactive progress bar?
                If not specified, will show progress only when `progress_file`
                is `std.stdout` or `std.stderr`, and `progress_file.isatty()`
                is :obj:`True`.
            progress_file: The file object where to write the progress.
                (default :obj:`sys.stderr`)
            hasher: A hasher algorithm instance from `hashlib`.
                If specified, will compute the hash of downloaded content,
                and validate against `expected_hash`.
            expected_hash: The expected hash of downloaded content.

        Returns:
            The absolute path of the extracted directory.

        Raises:
            ValueError: If `filename` or `extract_dir` cannot be inferred.
        """
        # check the arguments
        show_progress = _guess_show_progress_arg(progress_file, show_progress)

        if filename is None:
            filename = _guess_filename_from_uri(uri)
        file_path = os.path.abspath(os.path.join(self.path, filename))

        if extract_dir is None:
            extract_dir = _guess_extract_dir_from_filename(filename)
        extract_path = os.path.abspath(os.path.join(self.path, extract_dir))

        # download and extract the file
        with self._lock_file(file_path):
            if not os.path.isdir(extract_path):
                archive_file = self._download(
                    uri, file_path, show_progress=show_progress,
                    progress_file=progress_file, hasher=hasher,
                    expected_hash=expected_hash
                )
                self._extract_file(
                    archive_file, extract_path, show_progress=show_progress,
                    progress_file=progress_file
                )
                # delete the archive file if we successfully extracted it.
                os.remove(file_path)
            return extract_path

    def purge_all(self):
        """Delete everything in this :class:`CacheDir`."""
        shutil.rmtree(self.path)
