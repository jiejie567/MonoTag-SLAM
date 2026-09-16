"""Execute the shipped viewer's map selection; never run SLAM or encode video."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


def evaluate_select_map(html, *, mode, requested, maps, active_map, frame):
    functions = []
    for name in ('finalMapReplay', 'selectMap'):
        match = re.search(r'^function ' + name + r'\(\)\{.*?(?=\nfunction )',
                          html, flags=re.MULTILINE | re.DOTALL)
        if match is None:
            raise AssertionError('Missing viewer function: ' + name)
        functions.append(match.group(0))
    program = """
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
const manifest={replay_mode:input.mode};
const revision={maps:input.maps,active_map:input.active_map};
const frames=[input.frame],frameIndex=0;
function $(id){if(id!=='maps')throw Error('Unexpected UI lookup');return {value:input.requested};}
const before=JSON.stringify({revision,frames});
""" + '\n'.join(functions) + """
const selected=selectMap();
process.stdout.write(JSON.stringify({selected,unchanged:before===JSON.stringify({revision,frames})}));
"""
    completed = subprocess.run([shutil.which('node'), '-e', program],
                               input=json.dumps(dict(mode=mode, requested=requested,
                                                     maps=maps, active_map=active_map, frame=frame)),
                               text=True, capture_output=True, check=True)
    return json.loads(completed.stdout)


@unittest.skipUnless(shutil.which('node'), 'Node.js is needed to execute viewer JavaScript')
class FinalMapSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / 'aruco_track/slam_replay.html').read_text()

    def assert_selection(self, expected, *, maps=(7,), active_map=7, mode='final-map',
                         requested='auto', frame=None):
        if frame is None:
            frame = {'map_id': None, 'camera': None, 'source': 'invalid',
                     'hands': {}, 'trails': {}, 'metric': False}
        result = evaluate_select_map(self.html, mode=mode, requested=requested,
                                     maps=[{'id': identifier} for identifier in maps],
                                     active_map=active_map, frame=frame)
        self.assertEqual(result['selected'], expected)
        self.assertTrue(result['unchanged'], 'Selection must not fabricate camera/map ownership')

    def test_unlocalized_final_frame_can_show_only_available_map(self):
        self.assert_selection(7)

    def test_unlocalized_final_frame_does_not_guess_between_maps(self):
        self.assert_selection(None, maps=(7, 9), active_map=9)

    def test_unlocalized_final_frame_with_no_map_stays_empty(self):
        self.assert_selection(None, maps=())

    def test_unknown_nonnull_world_ownership_is_not_reassigned(self):
        self.assert_selection(None, frame={'map_id': 'unknown_world', 'camera': None,
                                           'source': 'invalid', 'hands': {}, 'trails': {}})

    def test_final_camera_owned_map_wins_over_active_map(self):
        self.assert_selection(7, maps=(7, 9), active_map=9,
                              frame={'map_id': 'atlas_7', 'camera': None})

    def test_manual_map_selection_wins_for_unlocalized_frame(self):
        self.assert_selection(9, maps=(7, 9), requested='9')

    def test_manual_marker_world_selection_is_preserved(self):
        self.assert_selection('marker_world', requested='marker_world')

    def test_independent_marker_world_view_is_preserved(self):
        self.assert_selection('marker_world', frame={'map_id': 'marker_world',
                                                     'marker_world_view': {'id': 'marker_world'}})

    def test_process_replay_still_uses_historical_active_map(self):
        self.assert_selection(9, maps=(7, 9), active_map=9, mode='process')


if __name__ == '__main__':
    unittest.main()
