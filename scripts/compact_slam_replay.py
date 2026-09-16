"""Upgrade existing replays with a verified browser timeline AND player; no SLAM rerun."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from itertools import zip_longest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aruco_track.replay_browser import (
    FILENAME, iter_browser_timeline_rows, iter_timeline_rows, write_browser_timeline,
)


def refresh_player(directory, original_text):
    """Keep one recoverable copy, then atomically deploy the matching decoder."""
    target = directory / 'index.html'
    template = (ROOT / 'aruco_track/slam_replay.html').read_text(encoding='utf-8')
    if target.read_text(encoding='utf-8') != original_text:
        raise ValueError('player changed during compaction; player not overwritten')
    if original_text == template:
        return False
    backup = directory / 'index.pre-browser-upgrade.html'
    if not backup.exists():
        with backup.open('x', encoding='utf-8') as stream:
            stream.write(original_text)
    descriptor, temporary = tempfile.mkstemp(prefix='.index.html-', dir=directory)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(template)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return True


def compact_replay(directory):
    directory = Path(directory)
    original = directory / 'timeline.json.gz'
    manifest_path = directory / 'manifest.json'
    manifest_text = manifest_path.read_text(encoding='utf-8')
    player_text = (directory / 'index.html').read_text(encoding='utf-8')
    manifest = json.loads(manifest_text)
    if not isinstance(manifest, dict):
        raise ValueError('replay manifest must be an object')
    original_stat = original.stat()
    with tempfile.TemporaryDirectory(prefix='.browser-timeline-', dir=directory) as temporary_directory:
        staged = Path(temporary_directory)
        stats = write_browser_timeline(iter_timeline_rows(original), staged)
        missing = object()
        verified = 0
        originals = iter_timeline_rows(original)
        decoded = iter_browser_timeline_rows(staged / FILENAME)
        try:
            for index, (before, after) in enumerate(zip_longest(originals, decoded, fillvalue=missing)):
                if before is missing or after is missing or before != after:
                    raise ValueError(f'browser timeline differs from original at row {index}')
                verified += 1
        finally:
            originals.close()
            decoded.close()
        current_stat = original.stat()
        if (current_stat.st_size, current_stat.st_mtime_ns) != (original_stat.st_size, original_stat.st_mtime_ns):
            raise ValueError('legacy timeline changed during compaction; manifest not updated')
        if manifest_path.read_text(encoding='utf-8') != manifest_text:
            raise ValueError('manifest changed during compaction; manifest not updated')
        if (directory / 'index.html').read_text(encoding='utf-8') != player_text:
            raise ValueError('player changed during compaction; package not updated')
        os.replace(staged / FILENAME, directory / FILENAME)

    manifest['browser_timeline'] = FILENAME
    descriptor, temporary = tempfile.mkstemp(prefix='.manifest.json-', dir=directory)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    updated = refresh_player(directory, player_text)
    return dict(stats, verified_rows=verified, exact_round_trip=True, player_updated=updated,
                original_preserved=str(original), original_compressed_bytes=original_stat.st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='+', help='existing actions_replay directories')
    args = parser.parse_args()
    for directory in args.directory:
        print(f'Upgrading browser data and player: {directory}', file=sys.stderr, flush=True)
        print(json.dumps(compact_replay(directory), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
