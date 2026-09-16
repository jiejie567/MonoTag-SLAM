import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from aruco_track.models import BandLayout,Calibration,Pose
from aruco_track.orbslam3_backend import write_tag_observation_hints
from aruco_track.marker_candidate_hints import write_candidate_hints
from aruco_track import offline_gaps
from process_monotag import validate_arguments,merge_candidate_lines,validate_runtime,digest,install_replay_display_assets


class MonoTagProfileTests(unittest.TestCase):
    def test_frozen_replay_gets_current_display_assets_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'manifest.json').write_text('{}')
            (root/'index.html').write_text('<body>frozen runtime</body>')
            (root/'actions.jsonl').write_text('unchanged labels')
            result=(root/'process.mp4',root/'index.html')
            module=SimpleNamespace(write_slam_replay=lambda:result)
            install_replay_display_assets(module)
            self.assertEqual(module.write_slam_replay(),result)
            self.assertEqual(module.write_slam_replay(),result)
            self.assertEqual((root/'index.html').read_text().count('replay_event_overlay.js'),1)
            self.assertTrue((root/'replay_event_overlay.js').is_file())
            self.assertEqual((root/'actions.jsonl').read_text(),'unchanged labels')

    def test_actual_driver_must_match_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=dict(schema='monotag-runtime/v1',native_project=str(root))
            for name in ('library','driver','gap_adapter'):
                p=root/name;p.write_bytes(name.encode());config[name]=str(p);config[name+'_sha256']=digest(p)
            actual=root/'third_party/ORB_SLAM3/Examples/Monocular/mono_tum_headless'
            actual.parent.mkdir(parents=True);actual.write_bytes(b'old driver')
            with self.assertRaisesRegex(ValueError,'does not match'):validate_runtime(config)
            actual.write_bytes(b'driver');validate_runtime(config)
            actual.unlink();actual.symlink_to(root/'driver');validate_runtime(config)

    def hints(self,confidence=.2,error=.5,weight=1.,component='m20'):
        square=np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
        arguments=([Pose(np.zeros(3),np.array([0,0,.5]),error)], [confidence],
            [{20:square[:,:2]}],[(20,)],BandLayout('world','DICT_4X4_50',{20:square}),
            Calibration(np.eye(3),np.zeros(5),(640,480)),60.,0)
        kwargs=dict(marker_weights=[{20:weight}],include_ids=True,marker_component_ids=[component])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'hints.txt';write_tag_observation_hints(p,*arguments,**kwargs)
            old=p.read_text().splitlines();write_candidate_hints(p,*arguments,**kwargs)
            return old,p.read_text().splitlines()

    def test_strong_hint_is_byte_identical(self):
        old,new=self.hints(confidence=.9)
        self.assertEqual(old,new)

    def test_candidate_retained_but_not_promoted_to_strong(self):
        old,new=self.hints()
        self.assertEqual(old[1].split()[1],'0')
        self.assertEqual(new[1].split()[2],'0.2')
        self.assertTrue(new[1].endswith('candidate 1'))
        self.assertEqual(merge_candidate_lines(old,new),new)

    def test_bad_error_weak_border_low_confidence_unknown_component_excluded(self):
        for kwargs in (dict(error=3),dict(weight=.25),dict(confidence=.1),dict(component=None)):
            old,new=self.hints(**kwargs)
            self.assertEqual(old,new)
            self.assertEqual(new[1].split()[1],'0')

    def test_existing_partial_and_full_hints_always_win(self):
        old=['# header','0 1 existing-full','1 1 existing-partial','2 0','3 0']
        new=['# other','0 1 altered','1 1 candidate 1','2 1 candidate 1','3 1 other']
        self.assertEqual(merge_candidate_lines(old,new),old[:3]+[new[3],old[4]])
        with self.assertRaises(ValueError):merge_candidate_lines(old,new[:-1])

    def test_no_overwrite_and_execution_local_only(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'actions.jsonl'
            self.assertEqual(validate_arguments(['--output='+str(out),'--execution=local']),out.resolve())
            with self.assertRaises(SystemExit):validate_arguments(['--output',str(out),'--execution','server'])
            out.touch()
            with self.assertRaises(ValueError):validate_arguments(['--output',str(out)])

    def test_hook_is_explicit_not_source_rewriting(self):
        exporter=SimpleNamespace(_MONOTAG_GAP_RECOVERY=None)
        old=offline_gaps.GAP_ADAPTER
        try:
            offline_gaps.install(exporter,Path('/tmp/test-adapter'))
            self.assertIs(exporter._MONOTAG_GAP_RECOVERY,offline_gaps.recover_short_gaps)
            self.assertEqual(offline_gaps.GAP_ADAPTER,Path('/tmp/test-adapter').resolve())
        finally:offline_gaps.GAP_ADAPTER=old


if __name__=='__main__':unittest.main()
