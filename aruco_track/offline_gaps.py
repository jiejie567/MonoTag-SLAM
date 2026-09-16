"""Read-only final-map recovery of short, bracketed native tracking failures."""
from __future__ import annotations
from collections import Counter
from dataclasses import replace
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import cv2
from aruco_track.camera_state import observed_hands_for_mask
from aruco_track.orbslam3_backend import pose_from_native
from aruco_track.native_cache_identity import adapter_identity,content_identity,video_identity
from aruco_track.pose_geometry_diagnostics import pose_geometry_diagnostics


def read_candidate_rows(path):
    def reject_nonfinite(value):
        raise ValueError('Non-finite adapter JSON: '+value)
    rows=[json.loads(l,parse_constant=reject_nonfinite) for l in path.read_text().splitlines() if l.strip()]
    if not rows or any(not isinstance(r,dict) for r in rows):
        raise ValueError('Adapter rows must be nonempty objects')
    return rows


def consume_candidate_file(result,req,detail,output):
    """An unreadable candidate only fails its own gap, never later jobs."""
    try:
        rows=read_candidate_rows(output)
        if any(r.get('type')=='error' for r in rows):
            detail.update(status='adapter_failed',evidence=rows)
            return result,None
        updated=apply_gap_candidates(result,req,rows)
        detail.update(status='completed',accepted=sum(updated.frames[i].pose is not None for i in req['frames']),
            evidence=[{k:v for k,v in r.items() if k not in ('matched_features','T_world_camera')} for r in rows])
        return updated,rows
    except (OSError,ValueError,TypeError,KeyError,OverflowError) as error:
        detail.update(status='adapter_failed',reason=str(error))
        return result,None


def cacheable_rows(rows,req,accepted):
    if rows is None:return False
    meta=[r for r in rows if r.get('type')=='metadata']
    frames=[r for r in rows if r.get('type')=='frame']
    return (len(meta)==1 and meta[0].get('schema')=='readonly-short-gap/v1'
        and meta[0].get('atlas_modified') is False and meta[0].get('map_id')==req['map_id']
        and meta[0].get('map_revision')==req['map_revision']
        and len(frames)==len(req['frames']) and all(type(r.get('frame')) is int for r in frames)
        and {r['frame'] for r in frames}==set(req['frames'])
        and (accepted==len(req['frames']) or all(r.get('accepted') is False for r in frames)))


GAP_ADAPTER = Path(__file__).resolve().parent / 'relocalize_gap'


def install(exporter, adapter):
    """Explicit opt-in hook; no source inspection or runtime code rewriting."""
    global GAP_ADAPTER
    GAP_ADAPTER = Path(adapter).resolve()
    exporter._MONOTAG_GAP_RECOVERY = recover_short_gaps


def gap_requests(result,fps,max_gap_s=.5):
    if fps<=0 or not np.isfinite(fps) or not result.history or not result.history[-1].get('final'):
        return []
    states={round(h['timestamp']*fps):h for h in result.history if not h.get('final')}
    eligible=[i for i,f in enumerate(result.frames) if f.pose is None and states.get(i,{}).get('pose') is None
              and states.get(i,{}).get('state') in (3,4)]
    groups=[]
    for i in eligible:
        if groups and i==groups[-1][-1]+1:groups[-1].append(i)
        else:groups.append([i])
    requests=[]
    for indices in groups:
        a,b=indices[0]-1,indices[-1]+1
        if a<0 or b>=len(result.frames) or any(i not in states for i in range(a,b+1)):continue
        before,after=result.frames[a],result.frames[b]
        if any(f.pose is None or not f.metric or not f.background_ready or f.localization_recovery for f in (before,after)):continue
        if before.map_id!=after.map_id or before.revision!=after.revision:continue
        mapping=result.maps.get(before.map_id,{})
        if not mapping.get('metric') or not mapping.get('background') or mapping.get('revision')!=before.revision:continue
        if states[b]['timestamp']-states[a]['timestamp']>max_gap_s+1e-6:continue
        if any(states[i].get('active_map')!=states[a].get('active_map') for i in range(a,b+1)):continue
        if any(states[i].get('state') not in (2,6) or states[i].get('pose') is None for i in (a,b)):continue
        requests.append(dict(map_id=mapping['id'],map_revision=before.revision,frames=indices,
            before=a,after=b,timestamps={i:states[i]['timestamp'] for i in range(a,b+1)}))
    return requests


def apply_gap_candidates(result,request,rows):
    frames=list(result.frames)
    if not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows):return result
    meta=[r for r in rows if r.get('type')=='metadata']
    if len(meta)!=1 or meta[0].get('controls_valid') is not True or meta[0].get('chain_valid') is not True:return result
    if meta[0].get('map_id')!=request['map_id'] or meta[0].get('map_revision')!=request['map_revision']:return result
    if meta[0].get('schema')!='readonly-short-gap/v1' or meta[0].get('atlas_modified') is not False:return result
    candidates=[r for r in rows if r.get('type')=='frame']
    if any(type(r.get('frame')) is not int for r in candidates):return result
    counts=Counter(r['frame'] for r in candidates)
    mapping=result.maps.get(f"atlas_{request['map_id']}",{})
    point_lookup={int(p[0]):p[1:] for p in mapping.get('points',[]) if len(p)==4}
    # Atomic segment acceptance. Never cascade recovered frames as controls.
    validated={}
    for r in candidates:
        i=r.get('frame')
        if type(i) is not int or i not in request['frames'] or counts[i]!=1:continue
        if r.get('accepted') is not True or r.get('source')!='offline-short-gap-relocalization' or r.get('support_only') is not False:continue
        if r.get('map_id')!=request['map_id'] or r.get('map_revision')!=request['map_revision'] or frames[i].pose is not None:continue
        try:
            p=np.asarray(r['pose'],dtype=float);rms=float(r['rms_px']);n=r['inliers']
            if p.shape!=(7,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[3:])-1)>1e-4:continue
            if type(n) is not int or n<30 or not 0<=rms<=3 or not np.isfinite(rms):continue
            if not .45<=r['inlier_fraction']<=1 or not 5<=r['occupied_cells']<=12 or not .06<=r['hull_fraction']<=1:continue
            if not np.isfinite(r['timestamp_s']) or abs(r['timestamp_s']-request['timestamps'][i])>1e-4:continue
            effective=r['validation_effective_time_s']
            if (not np.isfinite(effective) or abs(effective-result.history[-1]['timestamp'])>1e-4
                or r.get('validated_after_final_atlas') is not True or r.get('gauge')!='final_metric_atlas'
                or r.get('connected') is not True or r.get('candidate') is not True):continue
            matches=r['matched_features']
            if len(matches)!=n or r['matched_feature_count']!=n or type(r['matches']) is not int or r['matches']<n:continue
            if abs(r['inlier_fraction']-n/r['matches'])>1e-5:continue
            if any(len(f)!=4 or type(f[2]) is not int or f[2] not in point_lookup for f in matches):continue
            features=np.asarray(matches,float)
            if features.shape!=(n,4) or not np.isfinite(features).all():continue
            u,v,ids,errors=features.T;w,h=request['image_width'],request['image_height']
            if (len(set(ids))!=n or len(set(zip(u,v)))!=n or np.any(u<0) or np.any(u>=w)
                or np.any(v<0) or np.any(v>=h) or np.any(errors<0) or np.any(errors>3.)):continue
            cells=len(set(zip((u*4/w).astype(int),(v*3/h).astype(int))))
            hull=cv2.contourArea(cv2.convexHull(features[:,:2].astype(np.float32)))/(w*h)
            if cells!=r['occupied_cells'] or cells<5 or hull<.06 or abs(hull-r['hull_fraction'])>1e-4:continue
            if abs(np.sqrt(np.mean(errors**2))-rms)>1e-4:continue
            pose=pose_from_native(p)
            if pose is None:continue
            T=np.asarray(r['T_world_camera'],float);expected=np.eye(4)
            expected[:3,:3]=pose.rotation_matrix;expected[:3,3]=pose.tvec.ravel()
            if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T,expected,atol=1e-5,rtol=0):continue
            world=np.asarray([point_lookup[f[2]] for f in matches],float)
            R=pose.rotation_matrix.T;t=-R@pose.tvec
            if not np.isfinite(world).all() or np.any((R@world.T+t)[2]<=0):continue
            pixels=cv2.projectPoints(world,cv2.Rodrigues(R)[0],t,np.asarray(request['camera_matrix'],float),
                                     np.asarray(request['dist_coeffs'],float))[0].reshape(-1,2)
            measured=np.linalg.norm(pixels-features[:,:2],axis=1)
            if np.any(measured>3.01) or np.max(np.abs(measured-errors))>.01:continue
        except (KeyError,ValueError,TypeError,OverflowError,cv2.error):continue
        diagnostics=pose_geometry_diagnostics((R@world.T+t).T,np.asarray(request['camera_matrix'],float),
                                              np.asarray(request['dist_coeffs'],float),rms)
        validated[i]=(pose,n,rms,r,diagnostics)
    if set(validated)!=set(request['frames']):return result
    anchor=frames[request['before']]
    for i,(pose,n,rms,r,diagnostics) in validated.items():
        frames[i]=replace(anchor,pose=pose,source='head-slam',confidence=float(np.clip(n/100,.35,.9)),
            slam_inliers=n,graph_reprojection_error_px=rms,anchor_consistency=None,
            localization_recovery=dict(method='native-orb-final-atlas-short-gap-pnp',accepted=True,
                original_tracking_valid=False,map_id=request['map_id'],map_revision=request['map_revision'],
                available_after_timestamp_s=result.history[-1]['timestamp'],inliers=n,rms_px=rms,
                controls=[request['before'],request['after']],interpolated=False,
                frame=i,timestamp_s=r['timestamp_s'],matched_features=r['matched_features'],
                occupied_cells=r['occupied_cells'],hull_fraction=r['hull_fraction'],geometry_diagnostics=diagnostics))
    return replace(result,frames=frames)


def recover_short_gaps(result,project_dir,video_path,atlas_path,records_path,calibration,fps,diagnostics_dir):
    started=time.monotonic();requests=gap_requests(result,fps)
    if not requests:return result
    binary=GAP_ADAPTER
    diagnostics=Path(diagnostics_dir)/'short_gap_recovery';diagnostics.mkdir(parents=True,exist_ok=True)
    needed={i for req in requests for i in range(req['before'],req['after']+1)};observations={}
    with Path(records_path).open() as stream:
        for line in stream:
            row=json.loads(line);i=int(row['frame'])
            if i not in needed:continue
            corners=dict(row.get('boundary_rejected_marker_corners',{}));corners.update(row.get('detected_marker_corners',{}))
            polygons=list(corners.values())
            for hand in observed_hands_for_mask(row).values():
                points=np.asarray(hand.get('image_landmarks_normalized',[]),dtype=float)
                if points.ndim==2 and len(points)>=3:polygons.append((points[:,:2]*calibration.image_size).tolist())
            observations[i]=polygons
    initial=result;attempts=[];jobs=[];cache_paths={}
    vocabulary=Path(project_dir)/'third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt'
    identity_start=time.monotonic()
    runtime=adapter_identity(binary,vocabulary) if os.environ.get('ORB_SLAM3_GAP_CACHE','1')!='0' else None
    binding=(dict(runtime=runtime,atlas=content_identity(atlas_path),video=video_identity(video_path),
                  validator=content_identity(__file__)) if runtime is not None else None)
    cache_dir=Path(os.environ.get('ORB_SLAM3_GAP_CACHE_DIR',str(Path(project_dir)/'output/.short_gap_cache')))
    identity_seconds=time.monotonic()-identity_start
    for req in requests:
        detail=dict(frames=req['frames'],status='not_run',accepted=0)
        if any(i not in observations for i in range(req['before'],req['after']+1)):
            detail['status']='missing_exclusion_observations';attempts.append(detail);continue
        queries=[]
        for i in range(req['before'],req['after']+1):
            f=initial.frames[i];support=i in (req['before'],req['after'])
            q=dict(frame=i,timestamp_s=req['timestamps'][i],excluded_polygons=observations[i],
                support_only=int(support),original_pose_valid=int(f.pose is not None))
            if support:
                T=np.eye(4);T[:3,:3]=f.pose.rotation_matrix;T[:3,3]=np.asarray(f.pose.tvec).reshape(3)
                q['T_world_camera']=T.tolist()
            queries.append(q)
        manifest=dict(video=str(Path(video_path).resolve()),camera_matrix=calibration.camera_matrix.tolist(),
            dist_coeffs=calibration.dist_coeffs.reshape(-1).tolist(),image_width=calibration.image_size[0],
            image_height=calibration.image_size[1],map_id=req['map_id'],map_revision=req['map_revision'],
            validation_effective_time_s=result.history[-1]['timestamp'],queries=queries)
        stem=diagnostics/str(req['frames'][0]);request=stem.with_suffix('.request.json');output=stem.with_suffix('.candidates.jsonl')
        request.write_text(json.dumps(manifest,allow_nan=False))
        req.update({k:manifest[k] for k in ('camera_matrix','dist_coeffs','image_width','image_height')})
        detail['cache_hit']=False
        if binding is not None:
            key=hashlib.sha256(json.dumps(dict(binding=binding,request=manifest),sort_keys=True,allow_nan=False).encode()).hexdigest()
            entry=cache_dir/(key+'.json')
            cache_paths[output]=(entry,key)
            try:
                cached=json.loads(entry.read_text())
                payload=cached['payload']
                if cached['key']==key and hashlib.sha256(payload.encode()).hexdigest()==cached['payload_sha256']:
                    output.write_text(payload)
                    cached_rows=read_candidate_rows(output)
                    # All positive poses are revalidated below against the current final map.
                    if cacheable_rows(cached_rows,req,len(req['frames'])):
                        detail['cache_hit']=True
            except (OSError,ValueError,KeyError,TypeError):pass
        jobs.append((req,detail,request,output))
    if jobs:
        pending=[j for j in jobs if not j[1]['cache_hit']]
        returncode=0
        try:
            if pending:
                command=[str(binary),str(vocabulary),str(atlas_path),'--batch']
                command.extend(str(p) for _,_,req,out in pending for p in (req,out))
                p=subprocess.run(command,capture_output=True,text=True,timeout=180+10*len(pending))
                returncode=p.returncode
                (diagnostics/'batch.log').write_text(p.stdout+p.stderr)
        except (OSError,subprocess.TimeoutExpired) as error:
            returncode=-1
            for _,detail,_,_ in pending:detail['reason']=str(error)
        for req,detail,_,output in jobs:
            if not detail['cache_hit'] and returncode:
                detail.update(status='adapter_failed',returncode=returncode);attempts.append(detail);continue
            result,rows=consume_candidate_file(result,req,detail,output)
            if output in cache_paths and not detail['cache_hit'] and cacheable_rows(rows,req,detail['accepted']):
                entry,key=cache_paths[output]
                try:
                    payload=output.read_text();entry.parent.mkdir(parents=True,exist_ok=True)
                    staging=entry.with_suffix('.'+str(os.getpid())+'.tmp')
                    staging.write_text(json.dumps(dict(key=key,payload=payload,payload_sha256=hashlib.sha256(payload.encode()).hexdigest()),allow_nan=False))
                    staging.replace(entry)
                except OSError as error:detail['cache_write_error']=str(error)
            attempts.append(detail)
    count=sum(a['accepted'] for a in attempts);seconds=time.monotonic()-started
    (diagnostics/'report.json').write_text(json.dumps(dict(attempts=attempts,recovered_frames=count,seconds=seconds,
        cache_identity_seconds=identity_seconds,cache_hits=sum(a.get('cache_hit',False) for a in attempts),
        history_modified=False,odometry_used=False),indent=2,allow_nan=False))
    print(f'Offline short-gap recovery: {count} frames, {seconds:.2f}s',flush=True)
    return replace(result,timing=dict(result.timing,short_gap_recovery_seconds=seconds,short_gap_recovered_frames=count))
