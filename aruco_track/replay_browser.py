"""Exact shared-value encoding for browser replay timelines.

The legacy timeline remains the source record. This companion encoding shares
complete, identical map snapshots and event journals without dropping frames,
rounding coordinates, or relying on revision numbers to infer equality.
"""
import gzip
import json
import os
from pathlib import Path
import tempfile


SCHEMA = 'native-orb-shared-timeline/v1'
FILENAME = 'timeline.browser.json.gz'


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'), sort_keys=True)


def _reference(pool, index, name):
    if type(index) is not int or not 0 <= index < len(pool):
        raise ValueError(f'invalid {name} reference: {index!r}')
    return pool[index]


def _decode_row(row, maps, events):
    if not isinstance(row, dict) or not isinstance(row.get('maps'), list):
        raise ValueError('timeline row must contain a maps array')
    decoded = dict(row)
    decoded['maps'] = [_reference(maps, index, 'map') for index in row['maps']]
    if 'marker_graph_events' in row:
        decoded['marker_graph_events'] = _reference(
            events, row['marker_graph_events'], 'marker_graph_events')
    return decoded


def _validate_pools(maps, events):
    if not isinstance(maps, list) or not all(isinstance(m, dict) for m in maps):
        raise ValueError('shared maps must be an array of objects')
    if not isinstance(events, list) or not all(isinstance(e, list) for e in events):
        raise ValueError('shared marker_graph_events must be an array of arrays')


def decode_timeline(value):
    """Restore legacy rows; shared values are immutable replay data, not copies."""
    if isinstance(value, list):
        return value
    if not isinstance(value, dict) or value.get('schema') != SCHEMA:
        raise ValueError('unsupported browser timeline schema')
    maps, events = value.get('maps'), value.get('marker_graph_events')
    _validate_pools(maps, events)
    if not isinstance(value.get('rows'), list):
        raise ValueError('shared timeline rows must be an array')
    return [_decode_row(row, maps, events) for row in value['rows']]


def write_browser_timeline(rows, directory):
    """Atomically write a companion timeline from any iterable of legacy rows.

    Only unique shared values stay in memory. Encoded rows are staged in an
    automatically cleaned temporary file so the shared pools can precede rows;
    this also permits streaming verification of the finished gzip file.
    """
    directory = Path(directory)
    maps, events, map_lookup, event_lookup = [], [], {}, {}
    row_count = 0

    def share(value, pool, lookup):
        encoded = _json(value)
        index = lookup.get(encoded)
        if index is None:
            index = len(pool)
            lookup[encoded] = index
            pool.append(encoded)
        return index

    with tempfile.TemporaryFile(mode='w+b', dir=directory) as staged_rows:
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('maps'), list):
                raise ValueError('timeline row must contain a maps array')
            if not all(isinstance(mapping, dict) for mapping in row['maps']):
                raise ValueError('timeline maps must be objects')
            encoded = dict(row)
            encoded['maps'] = [share(mapping, maps, map_lookup)
                               for mapping in row['maps']]
            if 'marker_graph_events' in row:
                if not isinstance(row['marker_graph_events'], list):
                    raise ValueError('marker_graph_events must be an array')
                encoded['marker_graph_events'] = share(
                    row['marker_graph_events'], events, event_lookup)
            if row_count:
                staged_rows.write(b',')
            staged_rows.write(_json(encoded).encode('utf-8'))
            row_count += 1

        target = directory / FILENAME
        descriptor, temporary = tempfile.mkstemp(prefix='.' + FILENAME + '-', dir=directory)
        uncompressed_bytes = 0
        try:
            with os.fdopen(descriptor, 'wb') as raw:
                with gzip.GzipFile(filename='', fileobj=raw, mode='wb',
                                   compresslevel=6, mtime=0) as output:
                    def write(value):
                        nonlocal uncompressed_bytes
                        data = value.encode('utf-8') if isinstance(value, str) else value
                        output.write(data)
                        uncompressed_bytes += len(data)

                    write('{"schema":' + _json(SCHEMA) + ',"maps":[')
                    for index, value in enumerate(maps):
                        write((',' if index else '') + value)
                    write('],"marker_graph_events":[')
                    for index, value in enumerate(events):
                        write((',' if index else '') + value)
                    write('],"rows":[')
                    staged_rows.seek(0)
                    while chunk := staged_rows.read(1 << 20):
                        write(chunk)
                    write(']}')
                raw.flush()
                os.fsync(raw.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    return dict(schema=SCHEMA, filename=FILENAME, row_count=row_count,
                map_pool_count=len(maps), event_pool_count=len(events),
                compressed_bytes=target.stat().st_size,
                uncompressed_bytes=uncompressed_bytes)


class _JSONStream:
    """Small buffered JSON reader; array items need not be held together."""
    def __init__(self, stream, chunk_size=1 << 20):
        self.stream, self.chunk_size = stream, chunk_size
        self.buffer, self.position, self.eof = '', 0, False
        self.decoder = json.JSONDecoder()

    def _fill(self):
        chunk = self.stream.read(self.chunk_size)
        self.buffer = self.buffer[self.position:] + chunk
        self.position = 0
        self.eof = not chunk

    def peek(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ''
            self._fill()

    def expect(self, token):
        if self.peek() != token:
            raise ValueError(f'expected {token!r} in timeline JSON')
        self.position += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as error:
                if self.eof:
                    raise ValueError('incomplete or malformed timeline JSON') from error
                self._fill()
                continue
            # A primitive at a chunk boundary may continue in the next chunk.
            if end == len(self.buffer) and not self.eof:
                self._fill()
                continue
            self.position = end
            return value

    def array(self):
        self.expect('[')
        if self.peek() == ']':
            self.position += 1
            return
        while True:
            yield self.value()
            token = self.peek()
            if token == ']':
                self.position += 1
                return
            self.expect(',')

    def finish(self):
        if self.peek():
            raise ValueError('unexpected content after timeline JSON')


def iter_timeline_rows(path):
    """Read a legacy gzip JSON array, retaining only one publication at a time."""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        reader = _JSONStream(stream)
        for row in reader.array():
            if not isinstance(row, dict):
                raise ValueError('timeline rows must be objects')
            yield row
        reader.finish()


def iter_browser_timeline_rows(path):
    """Decode our pools-first companion file without loading its rows array."""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        reader = _JSONStream(stream)
        reader.expect('{')
        if reader.value() != 'schema':
            raise ValueError('expected browser timeline schema field')
        reader.expect(':')
        if reader.value() != SCHEMA:
            raise ValueError('unsupported browser timeline schema')
        pools = []
        for field in ('maps', 'marker_graph_events'):
            reader.expect(',')
            if reader.value() != field:
                raise ValueError(f'expected shared {field} field')
            reader.expect(':')
            pools.append(list(reader.array()))
        _validate_pools(*pools)
        reader.expect(',')
        if reader.value() != 'rows':
            raise ValueError('expected shared timeline rows field')
        reader.expect(':')
        for row in reader.array():
            yield _decode_row(row, *pools)
        reader.expect('}')
        reader.finish()
