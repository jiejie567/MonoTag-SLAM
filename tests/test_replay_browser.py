"""Lossless browser timeline sharing; no video encoding or SLAM run."""
import copy
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from aruco_track.replay_browser import decode_timeline, write_browser_timeline
from scripts.compact_slam_replay import compact_replay


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'native-orb-shared-timeline/v1'


def publications():
    first_map = {'id': 0, 'revision': 7, 'metric': True, 'point_count': 1,
                 'keyframes': [[10, 0.0, [0, 0, 0, 0, 0, 0, 1]]],
                 'markers': {'20': [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]},
                 'loops': [], 'merges': [], 'custom': {'retained': True}}
    row = {'sequence': 0, 'timestamp': 0.0, 'active_map': 0, 'state': 2,
           'checkpoint': True, 'offset': 0, 'count': 1, 'deleted': [],
           'final': False, 'maps': [first_map], 'events': ['地图创建'],
           'custom_row': {'values': [None, False, '完整保留']}}
    rows = [copy.deepcopy(row) for _ in range(6)]
    for index, value in enumerate(rows):
        value.update(sequence=index, timestamp=index / 30, checkpoint=index == 0,
                     offset=28 * index, count=index % 2, events=[])
    rows[1]['marker_graph_events'] = []
    # The revision is unchanged, but the keyframe's measured pose is not.
    rows[2]['maps'][0]['keyframes'][0][2][0] = 0.125
    rows[3]['maps'] = copy.deepcopy(rows[2]['maps'])
    event = {'sequence': 7, 'type': 'scale_reanchor', 'status': 'accepted',
             'scale': 1.01, 'affected_keyframes': [10]}
    rows[2]['marker_graph_events'] = [copy.deepcopy(event)]
    rows[3]['marker_graph_events'] = [copy.deepcopy(event), dict(event, status='rejected')]
    for value in rows[4:]:
        value['maps'][0].update(id=9, revision=0, metric=False)
        value['active_map'] = 9
        value['marker_graph_events'] = copy.deepcopy(rows[3]['marker_graph_events'])
    rows[4]['deleted'] = [[0, 1]]
    rows[5].update(final=True, checkpoint=True, events=['最终发布'])
    return rows


class BrowserTimelineTests(unittest.TestCase):
    def encode(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            stats = write_browser_timeline(rows, Path(directory))
            with gzip.open(Path(directory) / stats['filename'], 'rt') as stream:
                data = json.load(stream)
            return data, stats

    def test_all_publications_epochs_checkpoints_offsets_and_events_round_trip(self):
        rows = publications()
        original = copy.deepcopy(rows)
        encoded, stats = self.encode(rows)
        encoded_before = copy.deepcopy(encoded)
        self.assertEqual(encoded['schema'], SCHEMA)
        self.assertEqual(decode_timeline(encoded), original)
        self.assertEqual(rows, original)
        self.assertEqual(encoded, encoded_before)
        self.assertEqual(stats['row_count'], len(rows))
        self.assertNotIn('marker_graph_events', encoded['rows'][0])
        self.assertEqual(encoded['rows'][4]['deleted'], [[0, 1]])

    def test_identical_full_maps_share_but_same_revision_changed_pose_does_not(self):
        encoded, stats = self.encode(publications())
        refs = [row['maps'][0] for row in encoded['rows']]
        self.assertEqual(refs[0], refs[1])
        self.assertEqual(refs[2], refs[3])
        self.assertEqual(refs[4], refs[5])
        self.assertNotEqual(refs[0], refs[2])
        self.assertNotEqual(refs[0], refs[4])
        self.assertEqual(len(encoded['maps']), 3)
        self.assertEqual(stats['map_pool_count'], 3)
        decoded = decode_timeline(encoded)
        self.assertIs(decoded[0]['maps'][0], decoded[1]['maps'][0])

    def test_cumulative_events_share_exactly_without_sequence_only_deduplication(self):
        rows = publications()
        encoded, stats = self.encode(rows)
        refs = [row.get('marker_graph_events') for row in encoded['rows']]
        self.assertNotEqual(refs[1], refs[2])
        self.assertNotEqual(refs[2], refs[3])
        self.assertEqual(refs[3:], [refs[3]] * 3)
        self.assertEqual(stats['event_pool_count'], 3)
        self.assertEqual(decode_timeline(encoded)[3]['marker_graph_events'],
                         rows[3]['marker_graph_events'])

    def test_legacy_array_remains_supported_without_mutation(self):
        rows = publications()
        original = copy.deepcopy(rows)
        self.assertEqual(decode_timeline(rows), original)
        self.assertEqual(rows, original)

    def test_repetition_is_compacted_without_dropping_rows(self):
        row = publications()[0]
        row['maps'][0]['keyframes'] *= 200
        rows = [dict(row, sequence=index) for index in range(200)]
        encoded, stats = self.encode(rows)
        self.assertEqual(len(encoded['rows']), 200)
        self.assertEqual(len(encoded['maps']), 1)
        self.assertLess(stats['uncompressed_bytes'],
                        len(json.dumps(rows, separators=(',', ':')).encode()) / 10)
        self.assertEqual(decode_timeline(encoded), rows)

    def test_malformed_pool_references_are_rejected(self):
        encoded, _ = self.encode(publications())
        for field, invalid in (
            ('maps', -1), ('maps', True), ('maps', '0'), ('maps', 0.0),
            ('maps', len(encoded['maps'])),
            ('marker_graph_events', -1), ('marker_graph_events', True),
            ('marker_graph_events', '0'), ('marker_graph_events', 0.0),
            ('marker_graph_events', len(encoded['marker_graph_events'])),
        ):
            with self.subTest(field=field, invalid=invalid):
                bad = copy.deepcopy(encoded)
                bad['rows'][1][field] = [invalid] if field == 'maps' else invalid
                with self.assertRaises(ValueError):
                    decode_timeline(bad)

    def test_malformed_schema_and_container_types_are_rejected(self):
        encoded, _ = self.encode(publications())
        changes = [('schema', 'unknown/v1'), ('rows', {}), ('maps', {}),
                   ('marker_graph_events', {}), ('rows', [None]), ('maps', [None]),
                   ('marker_graph_events', [None])]
        for key, invalid in changes:
            with self.subTest(key=key, invalid=invalid):
                bad = copy.deepcopy(encoded)
                bad[key] = invalid
                with self.assertRaises(ValueError):
                    decode_timeline(bad)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for the actual browser decoder')
    def test_actual_browser_decoder_preserves_shared_and_legacy_publications(self):
        rows = publications()
        encoded, _ = self.encode(rows)
        html = (ROOT / 'aruco_track/slam_replay.html').read_text()
        self.assertIn("gz(m.browser_timeline||'timeline.json.gz').then(decodeTimeline)", html)
        start = html.index('function decodeTimeline(data)')
        opening = html.index('{', start)
        depth = 1
        end = opening + 1
        while depth:
            depth += (html[end] == '{') - (html[end] == '}')
            end += 1
        decoder = html[start:end]
        harness = "const assert=require('node:assert/strict');\n" + decoder + '\n'
        harness += 'const rows=' + json.dumps(rows) + ';\n'
        harness += 'const encoded=' + json.dumps(encoded) + ';\n'
        harness += '''
const before=JSON.stringify(encoded);
assert.deepStrictEqual(decodeTimeline(rows),rows);
const decoded=decodeTimeline(encoded);
assert.deepStrictEqual(decoded,rows);
assert.equal(JSON.stringify(encoded),before);
assert.strictEqual(decoded[0].maps[0],decoded[1].maps[0]);
for(const invalid of [-1,true,'0',encoded.maps.length]){
 const bad=JSON.parse(before);bad.rows[0].maps=[invalid];
 assert.throws(()=>decodeTimeline(bad));
}
'''
        result = subprocess.run([shutil.which('node'), '-e', harness],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


class ReplayUpgradeTests(unittest.TestCase):
    def package(self, directory):
        with gzip.open(directory / 'timeline.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(publications(), stream)
        (directory / 'manifest.json').write_text(json.dumps({'frames': 6, 'video': 'process.mp4'}))
        (directory / 'index.html').write_text('old player')
        for name in ('process.mp4', 'points.bin.gz', 'video_frames.json.gz', 'atlas.osa'):
            (directory / name).write_bytes(b'preserve this existing resource')

    def test_upgrade_updates_decoder_and_preserves_all_source_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.package(directory)
            source_names = ('timeline.json.gz', 'process.mp4', 'points.bin.gz',
                            'video_frames.json.gz', 'atlas.osa')
            before = {name: (directory / name).read_bytes() for name in source_names}
            result = compact_replay(directory)
            self.assertTrue(result['exact_round_trip'])
            self.assertTrue(result['player_updated'])
            self.assertEqual(result['verified_rows'], 6)
            self.assertEqual((directory / 'index.html').read_bytes(),
                             (ROOT / 'aruco_track/slam_replay.html').read_bytes())
            self.assertEqual((directory / 'index.pre-browser-upgrade.html').read_text(), 'old player')
            manifest = json.loads((directory / 'manifest.json').read_text())
            self.assertEqual(manifest['frames'], 6)
            with gzip.open(directory / manifest['browser_timeline'], 'rt') as stream:
                self.assertEqual(decode_timeline(json.load(stream)), publications())
            self.assertEqual(before, {name: (directory / name).read_bytes() for name in source_names})
            again = compact_replay(directory)
            self.assertFalse(again['player_updated'])
            self.assertEqual((directory / 'index.pre-browser-upgrade.html').read_text(), 'old player')

    def test_failed_verification_does_not_publish_or_upgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.package(directory)
            manifest = (directory / 'manifest.json').read_bytes()
            with patch('scripts.compact_slam_replay.iter_browser_timeline_rows',
                       side_effect=lambda path: (row for row in [])):
                with self.assertRaisesRegex(ValueError, 'differs from original'):
                    compact_replay(directory)
            self.assertEqual((directory / 'manifest.json').read_bytes(), manifest)
            self.assertEqual((directory / 'index.html').read_text(), 'old player')
            self.assertFalse((directory / 'timeline.browser.json.gz').exists())

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for the browser stream reader')
    def test_browser_json_limit_cancels_stream_and_preserves_unicode(self):
        html = (ROOT / 'aruco_track/slam_replay.html').read_text()
        start = html.index('async function readReplayJSON(')
        end = html.index('async function gz(', start)
        harness = "const assert=require('node:assert/strict');\n" + html[start:end]
        harness += '''
(async()=>{
 const data=new TextEncoder().encode(JSON.stringify({message:'地图与轨迹',rows:[1,2,3]}));
 let at=0;
 const input=new ReadableStream({pull(c){if(at===data.length)c.close();else c.enqueue(data.slice(at,++at));}});
 assert.deepStrictEqual(await readReplayJSON(input,'test.json',data.length),{message:'地图与轨迹',rows:[1,2,3]});
 let cancelled=false,pulls=0;
 const oversized=new ReadableStream({pull(c){pulls++;c.enqueue(new Uint8Array(4));},cancel(){cancelled=true;}});
 await assert.rejects(readReplayJSON(oversized,'timeline.json.gz',8),/compact_slam_replay/);
 assert.equal(cancelled,true);assert.ok(pulls<=4);assert.equal(oversized.locked,false);
 const malformed=new ReadableStream({start(c){c.enqueue(new TextEncoder().encode('{bad'));c.close();}});
 await assert.rejects(readReplayJSON(malformed,'bad.json'),SyntaxError);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
        result = subprocess.run([shutil.which('node'), '-e', harness],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
