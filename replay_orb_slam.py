#!/usr/bin/env python3
"""Serve a completed native replay; never run detection or SLAM here."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote
import webbrowser


class ReplayHandler(SimpleHTTPRequestHandler):
    def list_directory(self, path):
        self.send_error(403, "directory listing disabled")
        return None

    def send_head(self):
        root = Path(self.directory).resolve()
        path = Path(self.translate_path(self.path)).resolve()
        if not path.is_relative_to(root):
            self.send_error(403)
            return None
        if path.is_dir():
            path = path / "index.html"
        path = path.resolve()
        if not path.is_relative_to(root):
            self.send_error(403)
            return None
        if not path.is_file():
            self.send_error(404)
            return None
        size = path.stat().st_size
        start, stop = 0, size - 1
        partial_response = False
        header = self.headers.get("Range")
        if header:
            try:
                unit, value = header.split("=", 1)
                first, last = value.split("-", 1)
                if unit != "bytes" or "," in value:
                    raise ValueError()
                start = int(first) if first else max(0, size - int(last))
                stop = min(size - 1, int(last)) if first and last else size - 1
                if not 0 <= start <= stop < size:
                    raise ValueError()
                partial_response = True
            except ValueError:
                self.send_error(416)
                return None
        stream = path.open("rb")
        stream.seek(start)
        self._remaining = stop - start + 1
        self.send_response(206 if partial_response else 200)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(self._remaining))
        self.send_header("Accept-Ranges", "bytes")
        if partial_response:
            self.send_header("Content-Range", f"bytes {start}-{stop}/{size}")
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        remaining = self._remaining
        try:
            while remaining:
                block = source.read(min(1 << 20, remaining))
                if not block:
                    break
                outputfile.write(block)
                remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, help="completed replay directory or index.html")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--root", type=Path,
                        help="serve a catalog and all replay packages below this root")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    target = args.replay.resolve()
    index = target / "index.html" if target.is_dir() else target
    root = args.root.resolve() if args.root else index.parent
    if not index.is_file() or not index.is_relative_to(root):
        parser.error("replay index must exist below the served root")
    if args.root is None and not (root / "manifest.json").is_file():
        parser.error("completed native replay package required; raw video is not accepted")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(ReplayHandler, directory=str(root)))
    relative_index = quote(index.relative_to(root).as_posix(), safe="/")
    url = f"http://127.0.0.1:{server.server_port}/{relative_index}"
    print(url, flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
