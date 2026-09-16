import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from aruco_track.models import Calibration

from aruco_track import offline_gaps as gaps
spec=importlib.util.spec_from_file_location('short_gap_fixture',Path(__file__).with_name('test_offline_gap_recovery.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


class ShortGapJobsTests(unittest.TestCase):
    def test_bad_file_does_not_prevent_next_job(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);bad=p/'bad';good=p/'good';bad.write_text('{broken')
            good.write_text('\n'.join(json.dumps(r) for r in fixture.rows()))
            result=fixture.fixture();q=fixture.request(result);details=[]
            for path in (bad,good):
                detail=dict(accepted=0);result,_=gaps.consume_candidate_file(result,q,detail,path);details.append(detail)
            self.assertEqual(details[0]['status'],'adapter_failed')
            self.assertEqual(details[1]['accepted'],1)

    def test_positive_and_negative_cache_reuse_and_mask_invalidation(self):
        for positive in (True,False):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);atlas=p/'atlas';video=p/'video';records=p/'rows'
                atlas.write_bytes(b'fixed atlas');video.write_bytes(b'video')
                raw=[dict(frame=i,hands={},detected_marker_corners={}) for i in range(3)]
                records.write_text('\n'.join(json.dumps(r) for r in raw))
                def adapter(cmd,**kwargs):
                    rows=fixture.rows()
                    if not positive:
                        rows[0]['chain_valid']=False;rows[1]['accepted']=False
                    Path(cmd[-1]).write_text('\n'.join(json.dumps(r) for r in rows))
                    return SimpleNamespace(returncode=0,stdout='',stderr='')
                cal=Calibration(np.array([[500,0,320],[0,500,240],[0,0,1]],float),np.zeros(5),(640,480))
                with patch.object(gaps,'adapter_identity',return_value={'actual_library':'fixed'}),patch.object(gaps.subprocess,'run',side_effect=adapter) as run:
                    for n in range(2):
                        result=gaps.recover_short_gaps(fixture.fixture(),p,video,atlas,records,cal,10,p/str(n))
                        self.assertEqual(result.frames[1].pose is not None,positive)
                    self.assertEqual(run.call_count,1)
                    raw[1]['detected_marker_corners']={'20':[[0,0],[1,0],[1,1],[0,1]]}
                    records.write_text('\n'.join(json.dumps(r) for r in raw))
                    gaps.recover_short_gaps(fixture.fixture(),p,video,atlas,records,cal,10,p/'changed')
                    self.assertEqual(run.call_count,2)
