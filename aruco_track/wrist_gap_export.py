"""Derived continuous wrist poses; measurements and joint labels stay untouched."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def complete_wrists(rows, max_gap_s=0.5):
    if not math.isfinite(max_gap_s) or max_gap_s < 0:
        raise ValueError('max_gap_s must be finite and nonnegative')
    names = sorted({name for r in rows for name in r.get('hands', {})})
    result = [dict(frame=r['frame'], timestamp_s=r['timestamp_s'],
                   world_frame_id=r.get('world_frame_id'), map_revision=r.get('map_revision'),
                   wrists={}) for r in rows]
    def world(r):
        if (r.get('camera_world_source') in (None, 'invalid') or
            r.get('camera_world_pose_fused') is None or r.get('scale_status') != 'metric' or
            r.get('world_frame_id') is None or r.get('map_revision') is None):
            return None
        return r['world_frame_id'], r['map_revision']
    def valid_pose(p):
        if not isinstance(p, dict): return False
        t=np.asarray(p.get('translation_m', []), float)
        q=np.asarray(p.get('quaternion_wxyz', []), float)
        return t.shape==(3,) and q.shape==(4,) and np.isfinite(t).all() and np.isfinite(q).all() and abs(np.linalg.norm(q)-1)<.01
    for name in names:
        previous = None
        for i,r in enumerate(rows):
            entry = dict(pose=None, source='invalid', measurement_valid=False, estimate_valid=False)
            result[i]['wrists'][name] = entry
            key=world(r); timestamp=r['timestamp_s']
            if (key is None or not math.isfinite(timestamp) or
                (i and (not math.isfinite(rows[i-1]['timestamp_s']) or timestamp<=rows[i-1]['timestamp_s'] or key!=world(rows[i-1])))):
                previous=None
                if key is None or not math.isfinite(timestamp): continue
            hand=r.get('hands', {}).get(name, {})
            pose=hand.get('wrist_world_graph')
            if hand.get('world_submap_id')!=key[0] or not valid_pose(pose): continue
            entry.update(pose={k:pose[k] for k in ('translation_m','quaternion_wxyz')}, source='observed', measurement_valid=True, estimate_valid=True)
            if previous is not None and i>previous+1:
                duration=timestamp-rows[previous]['timestamp_s']
                if 0<duration<=max_gap_s+1e-9:
                    start=result[previous]['wrists'][name]['pose']
                    q=np.asarray([start['quaternion_wxyz'],pose['quaternion_wxyz']])[:,[1,2,3,0]]
                    rotation=Slerp([0,1],Rotation.from_quat(q))
                    for j in range(previous+1,i):
                        alpha=(rows[j]['timestamp_s']-rows[previous]['timestamp_s'])/duration
                        t=(1-alpha)*np.asarray(start['translation_m'])+alpha*np.asarray(pose['translation_m'])
                        quat=rotation([alpha]).as_quat()[0][[3,0,1,2]].tolist()
                        result[j]['wrists'][name].update(pose=dict(translation_m=t.tolist(),quaternion_wxyz=quat),
                            source='interpolated',estimate_valid=True,measurement_valid=False,
                            support_frames=[rows[previous]['frame'],r['frame']],support_span_s=duration)
            previous=i
    return result


class CompletedReplayTrails:
    """Render exactly the derived positions, retaining invalid/map boundaries."""
    def __init__(self, completed):
        self.by_frame={r['frame']:r for r in completed}
        self.by_time={round(r['timestamp_s'],9):r for r in completed}

    def apply(self, frame):
        row=self.by_frame.get(frame['source_frame'])
        if row is None: raise ValueError('Missing wrist completion frame')
        if frame.get('camera') is None or not frame.get('metric'):
            frame.update(trails={},trail_sample_sources={},trail_display_interpolated={},
                         wrist_completion_source='derived-wrist-completion-v1')
            return frame
        names=row['wrists']
        trails={name:[] for name in names};sources={name:[] for name in names}
        for timestamp in frame.get('trail_timestamps_s', []):
            previous=self.by_time.get(round(timestamp,9))
            same=(frame.get('camera') is not None and frame.get('metric') is True and previous is not None and
                  previous['world_frame_id']==frame.get('map_id') and previous['map_revision']==frame.get('map_revision'))
            for name in names:
                wrist=previous['wrists'].get(name,{}) if same else {}
                pose=wrist.get('pose') if wrist.get('estimate_valid') else None
                trails[name].append(pose['translation_m'] if pose else None)
                sources[name].append(wrist.get('source','invalid') if pose else 'invalid')
        frame['trails']=trails
        frame['trail_sample_sources']=sources
        frame['trail_display_interpolated']={n:[i for i,s in enumerate(ss) if s=='interpolated'] for n,ss in sources.items()}
        frame['wrist_completion_source']='derived-wrist-completion-v1'
        return frame


def export_wrist_completion(actions_path, max_gap_s=0.5):
    path=Path(actions_path)
    with path.open() as stream: rows=[json.loads(line) for line in stream if line.strip()]
    output=path.with_suffix('.wrists_completed.jsonl')
    if output.exists(): raise FileExistsError(output)
    completed=complete_wrists(rows,max_gap_s)
    with output.open('x') as stream:
        for row in completed: stream.write(json.dumps(row,allow_nan=False)+'\n')
    count=sum(w['source']=='interpolated' for r in completed for w in r['wrists'].values())
    return dict(enabled=True,path=str(output),max_support_span_s=max_gap_s,
                interpolated_wrist_frames=count,raw_labels_modified=False,
                policy='linear world position + quaternion SLERP; same metric map revision and valid camera throughout; no joint filling')
