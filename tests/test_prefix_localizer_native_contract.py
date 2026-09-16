"""Fast contracts; optional real-Atlas negative tests live beside the experiment."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'third_party/ORB_SLAM3/Examples/Monocular/relocalize_prefix_readonly'


class PrefixReadOnlyContract(unittest.TestCase):
    def test_no_mapping_or_serialization_writes(self):
        source = (ROOT / 'third_party/ORB_SLAM3/Examples/Monocular/relocalize_prefix_readonly.cc').read_text()
        for forbidden in ('new System(', 'binary_oarchive', 'SaveAtlas(', 'TrackMonocular(', 'CreateMapInAtlas('):
            self.assertNotIn(forbidden, source)
        self.assertIn('output_aliases_immutable_input', source)
        self.assertIn('requested_map_not_metric', source)
        self.assertIn('final_map_revision_mismatch', source)

    def test_pose_requires_geometry_and_temporal_anchor(self):
        source = (ROOT / 'third_party/ORB_SLAM3/Examples/Monocular/relocalize_prefix_readonly.cc').read_text()
        self.assertIn('cv::solvePnPRansac', source)
        self.assertIn('cv::solvePnPRefineLM', source)
        self.assertIn('connected>=2', source)
        self.assertIn('anchorTranslation<.05&&anchorRotation<5.', source)
        self.assertIn('query_outside_initial_prefix_5s', source)
        self.assertIn('refusing_existing_pose', source)
        self.assertIn('if(!result.accepted)out<<"null"', source)

    def test_temporal_support_does_not_publish_or_relax_anchor_checks(self):
        source = BINARY.with_suffix('.cc').read_text()
        for required in (
            'support_outside_anchor_window_2s',
            'query.supportOnly&&query.frame>=control.frame',
            'dt>0&&dt<=.25',
            'result.accepted=!result.supportOnly&&result.candidate&&result.connected&&connected>=2',
            'if(!it->supportOnly)++connected',
            'offline-prefix-temporal-support',
        ):
            self.assertIn(required, source)

    def test_correspondence_mode_preserves_observation_identity_and_frozen_pose(self):
        source = BINARY.with_suffix('.cc').read_text()
        for required in (
            'replay-feature-correspondence', 'offline-final-map-correspondence',
            'readonly-final-map-correspondence/v1', 'invalid_frozen_camera_se3',
            'matcher.knnMatch(descriptors,queryDescriptors,backward,2)',
            'pair[0].distance<.75*pair[1].distance', 'pointIds.push_back(item.first)',
            'matchedPointIds.push_back(pointIds[a.trainIdx])',
            'translationDistance(result.Twc,query.frozenTwc)>.05',
            'rotationDistance(result.Twc,query.frozenTwc)>5.',
            'error<=3.&&depth>0.', 'mask.at<unsigned char>(v,u)!=0',
            'result.candidate=result.ratio>=.45&&result.cells>=5&&result.hull>=.06',
            'result.inliers>=30&&result.cells>=5&&result.hull>=.06',
            'if(independentInliers[i]&&std::isfinite(error)&&error<=3.',
            'std::map<std::pair<double,double>,FeatureMatch> uniquePixels',
            'if(old==uniquePixels.end()||error<old->second.error)',
            'uniquePixels[key]={pixels[i],matchedPointIds[i],error}',
            'result.matchedFeatures.push_back(match)',
            'if(result.accepted) for(size_t i=0;i<result.matchedFeatures.size();++i)',
            'anchor_exclusion_mask_missing',
        ):
            self.assertIn(required, source)

    @unittest.skipUnless(BINARY.is_file(), 'build the standalone native adapter first')
    def test_native_input_guards_do_not_open_atlas_or_create_output(self):
        identity = [[1., 0., 0., 0.], [0., 1., 0., 0.],
                    [0., 0., 1., 0.], [0., 0., 0., 1.]]
        base = dict(camera_matrix=[[100., 0., 32.], [0., 100., 24.], [0., 0., 1.]],
                    dist_coeffs=[0.] * 5, image_width=64, image_height=48,
                    map_id=0, map_revision=1, boundary_frame=90,
                    boundary_time_s=1., video='absent-video.mp4')
        cases = []
        cases.append((None, dict(frame=0, timestamp_s=0., excluded_polygons=[],
                                 original_pose_valid=True), 'refusing_existing_pose'))
        cases.append(('replay-feature-correspondence',
                      dict(frame=0, timestamp_s=0., excluded_polygons=[]), 'invalid_matrix_shape'))
        reflection = [row[:] for row in identity]
        reflection[0][0] = -1.
        cases.append(('replay-feature-correspondence',
                      dict(frame=0, timestamp_s=0., excluded_polygons=[],
                           T_world_camera=reflection), 'invalid_frozen_camera_se3'))
        nonrigid = [row[:] for row in identity]
        nonrigid[3][0] = .1
        cases.append(('replay-feature-correspondence',
                      dict(frame=0, timestamp_s=0., excluded_polygons=[],
                           T_world_camera=nonrigid), 'invalid_frozen_camera_se3'))
        cases.append(('replay-feature-correspondence',
                      dict(frame=90, timestamp_s=1., excluded_polygons=[],
                           T_world_camera=identity), 'query_outside_initial_prefix_5s'))
        for purpose, query, expected in cases:
            with self.subTest(reason=expected), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                request, output = directory / 'request.json', directory / 'output.jsonl'
                manifest = dict(base, queries=[query])
                if purpose is not None:
                    manifest['purpose'] = purpose
                request.write_text(json.dumps(manifest))
                result = subprocess.run([str(BINARY), str(directory / 'absent-vocabulary'),
                                         str(directory / 'absent-atlas'), str(request), str(output)],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)
                self.assertFalse(output.exists())

    @unittest.skipUnless(BINARY.is_file(), 'build the standalone native adapter first')
    def test_native_support_input_bounds_and_existing_pose_policy(self):
        base = dict(camera_matrix=[[100., 0., 32.], [0., 100., 24.], [0., 0., 1.]],
                    dist_coeffs=[0.] * 5, image_width=64, image_height=48,
                    map_id=0, map_revision=1, boundary_frame=90,
                    boundary_time_s=1., video='absent-video.mp4',
                    queries=[dict(frame=0, timestamp_s=0., excluded_polygons=[])])
        cases = [
            (dict(frame=89, timestamp_s=.99), 'support_outside_anchor_window_2s'),
            (dict(frame=91, timestamp_s=3.01), 'support_outside_anchor_window_2s'),
            # A support image may already have a measured pose. This does not
            # make it a recovery target or trigger the existing-pose refusal.
            (dict(frame=90, timestamp_s=1., original_pose_valid=True), 'atlas_unreadable'),
        ]
        for support, expected in cases:
            with self.subTest(reason=expected), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                request, output = directory/'request.json', directory/'output.jsonl'
                manifest = dict(base, support_queries=[dict(support, excluded_polygons=[])])
                request.write_text(json.dumps(manifest))
                completed = subprocess.run([str(BINARY), str(directory/'absent-vocabulary'),
                    str(directory/'absent-atlas'), str(request), str(output)],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(completed.returncode, 2)
                self.assertIn(expected, completed.stderr)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
