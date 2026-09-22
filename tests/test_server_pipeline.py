import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from aruco_track.server_pipeline import digest, extract_results, verify_release, default_execution, json_argument
from scripts.server_worker import arguments
from scripts.package_release import source_files

class ServerPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def test_source_mutation_fails(self):
        p=self.root/'code.py';p.write_text('a=1')
        manifest={'files':{'code.py':digest(p)}};verify_release(self.root,manifest)
        p.write_text('a=2')
        with self.assertRaisesRegex(ValueError,'source mismatch'):verify_release(self.root,manifest)
    def test_release_path_escape_fails(self):
        with self.assertRaises(ValueError):verify_release(self.root,{'files':{'../other':'abc'}})
    def test_remote_false_flags_and_csv_preserved(self):
        args=arguments(dict(video='video ; literal.mp4',execution='local',band=['a b','r'],
                            static_marker_ids=[20,35,49],hand_joints=False,slam_replay=False,
                            auto_marker_map=True,open_replay=False,slam_debug_video=''))
        self.assertEqual(args[0],'video ; literal.mp4')
        self.assertIn('--no-hand-joints',args);self.assertIn('--no-slam-replay',args)
        self.assertIn('20,35,49',args);self.assertNotIn('--open-replay',args)
        self.assertEqual(args[args.index('--execution')+1],'local')
    def test_real_parser_marker_set_and_paths_survive_json(self):
        value=json.loads(json.dumps(dict(video=Path('x.mp4'),static_marker_ids={49,20}),default=json_argument))
        result=arguments(value)
        self.assertEqual(result,['x.mp4','--static-marker-ids','20,49'])
    def test_returned_archive_rejects_traversal_and_symlinks(self):
        for name,kind in [('../escape',tarfile.REGTYPE),('link',tarfile.SYMTYPE)]:
            archive=self.root/'bad.tar'
            with tarfile.open(archive,'w') as tar:
                member=tarfile.TarInfo(name);member.type=kind;member.linkname='/etc/passwd';tar.addfile(member)
            with self.assertRaises(ValueError):extract_results(archive,self.root/'out')
    def test_valid_archive(self):
        archive=self.root/'good.tar'
        with tarfile.open(archive,'w') as tar:
            member=tarfile.TarInfo('actions.jsonl');member.size=3;tar.addfile(member,io.BytesIO(b'{}\n'))
        extract_results(archive,self.root/'out')
        self.assertEqual((self.root/'out/actions.jsonl').read_bytes(),b'{}\n')
    def test_explicit_promotion_controls_default(self):
        p=self.root/'server.json'
        with patch('aruco_track.server_pipeline.DEFAULT_CONFIG',p):
            self.assertEqual(default_execution(),'local')
            p.write_text(json.dumps(dict(enabled=False)));self.assertEqual(default_execution(),'local')
            p.write_text(json.dumps(dict(enabled=True)));self.assertEqual(default_execution(),'server')
    def test_source_package_excludes_private_and_build_data(self):
        names=['tools/export_action_labels.py','aruco_track/a.py','config/production.json','config/server.local.json',
               '.local/secret.json','models/MANO_RIGHT.pkl','input/video.mp4','output/results.json',
               'third_party/ORB_SLAM3/src/Tracking.cc','third_party/ORB_SLAM3/CMakeLists.txt',
               'third_party/ORB_SLAM3/Examples/timestamps.txt','third_party/ORB_SLAM3/lib/libORB_SLAM3.so']
        for name in names:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
        got={p.relative_to(self.root).as_posix() for p in source_files(self.root)}
        self.assertEqual(got,set(names[:3]+[names[8],names[9]]))

if __name__=='__main__':unittest.main()
