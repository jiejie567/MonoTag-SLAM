"""Content identities for read-only adapters; unresolved loaders disable reuse."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


@lru_cache(maxsize=64)
def _digest(path, size, mtime, ctime, inode):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def content_identity(path):
    path = Path(path).resolve(strict=True)
    s = path.stat()
    return dict(path=str(path), sha256=_digest(str(path), s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino))


def adapter_identity(binary, vocabulary):
    """Resolve the actual ORB library under the current loader environment.

    Linux ldd reports LD_LIBRARY_PATH resolution. On macOS, these read-only
    adapters exit on missing arguments, allowing dyld to report its actual
    library without reading a video or opening an Atlas. Unknown loaders
    conservatively recompute instead of guessing a library.
    """
    try:
        if sys.platform.startswith('linux'):
            loaded = subprocess.run(['ldd', str(Path(binary).resolve())], capture_output=True,
                                    text=True, check=True, timeout=10).stdout
            libraries = re.findall(r'^\s*libORB_SLAM3[^\s]*\s+=>\s+(\S+)\s+\(', loaded, re.M)
            vision = re.findall(r'^\s*libopencv_[^\s]*\s+=>\s+(\S+)\s+\(', loaded, re.M)
            if 'not found' in loaded:return None
        elif sys.platform == 'darwin':
            env=dict(os.environ,DYLD_PRINT_LIBRARIES='1')
            loaded=subprocess.run([str(Path(binary).resolve())],env=env,capture_output=True,
                                  text=True,timeout=10).stderr
            libraries=re.findall(r'(/[^\n]*libORB_SLAM3[^/\n]*\.dylib)\s*$',loaded,re.M)
            vision=re.findall(r'(/[^\n]*libopencv_[^/\n]*\.dylib)\s*$',loaded,re.M)
        else:
            return None
        if len(libraries) != 1:
            return None
        return dict(binary=content_identity(binary), vocabulary=content_identity(vocabulary),
                    orb_library=content_identity(libraries[0]),
                    opencv_libraries=[content_identity(p) for p in sorted(set(vision))],
                    decoder=content_identity(shutil.which('ffmpeg')))
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return None


def video_identity(path):
    path = Path(path).resolve(strict=True)
    s = path.stat()
    return dict(path=str(path), size=s.st_size, mtime_ns=s.st_mtime_ns,
                ctime_ns=s.st_ctime_ns, inode=s.st_ino, device=s.st_dev)
