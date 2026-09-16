import functools
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from replay_orb_slam import ReplayHandler


class ReplayServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.package = self.root/'package'; self.package.mkdir()
        (self.package/'process.mp4').write_bytes(bytes(range(100)))
        (self.root/'outside.txt').write_text('not a replay resource')
        (self.package/'index.html').symlink_to(self.root/'outside.txt')
        self.server = ThreadingHTTPServer(('127.0.0.1', 0),
                functools.partial(ReplayHandler, directory=str(self.package)))
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.worker.join()
        self.temporary.cleanup()

    def test_range_for_video_seek(self):
        with urlopen(Request(self.url+'/process.mp4', headers={'Range': 'bytes=10-19'})) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers['Content-Range'], 'bytes 10-19/100')
            self.assertEqual(response.read(), bytes(range(10, 20)))
        with urlopen(Request(self.url+'/process.mp4', headers={'Range': 'bytes=-5'})) as response:
            self.assertEqual(response.read(), bytes(range(95, 100)))

    def test_index_symlink_cannot_leave_replay_package(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self.url+'/')
        self.assertEqual(error.exception.code, 403)

    def test_invalid_range_rejected(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.url+'/process.mp4', headers={'Range': 'bytes=200-300'}))
        self.assertEqual(error.exception.code, 416)
