"""Validated Ubuntu MonoTag SLAM entry; ordinary exporter/Mac defaults unchanged.

Use --runtime-config PATH to select an installed, hash-verified native runtime.
All other arguments are normal export_action_labels.py arguments.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

STATIC_CORNER_SCHEMA = 'aruco-april-lines/pixel-center-stable-mask/v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_runtime(config):
    if config.get('schema')!='monotag-runtime/v1':raise ValueError('Unsupported native runtime schema')
    for key in ('library','driver','gap_adapter'):
        if digest(config[key])!=config[key+'_sha256']:raise ValueError('Runtime changed: '+key)
    if config.get('final_frame_adapter'):
        if digest(config['final_frame_adapter'])!=config['final_frame_adapter_sha256']:
            raise ValueError('Runtime changed: final_frame_adapter')
    # The backend launches this path, not the descriptive `driver` field.
    # Reject a mixed runtime before spending time preparing a full sequence.
    actual=Path(config['native_project'])/'third_party/ORB_SLAM3/Examples/Monocular/mono_tum_headless'
    if digest(actual)!=config['driver_sha256']:
        raise ValueError('Runtime driver does not match the executable under native_project')


def validate_arguments(arguments):
    # Explicit output avoids accidental replacement of an earlier experiment.
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--output',required=True)
    parser.add_argument('--execution',choices=['local'],default='local')
    options,_=parser.parse_known_args(arguments)
    output=Path(options.output).expanduser().resolve()
    if output.exists() or (output.parent/'runtime.json').exists():
        raise ValueError('Choose a fresh output: '+str(output))
    return output


def merge_candidate_lines(old,new):
    if len(old)!=len(new):raise ValueError('Marker hint stream length changed')
    return [b if not a.startswith('#') and a.split()[1]=='0' and 'candidate 1' in b else a
            for a,b in zip(old,new)]


def install_replay_display_assets(replay_module):
    """Keep display assets current even when the validated runtime is frozen."""
    from functools import wraps
    from scripts.refresh_replay_events import refresh
    original = replay_module.write_slam_replay
    @wraps(original)
    def write(*args, **kwargs):
        result = original(*args, **kwargs)
        refresh(Path(result[1]).parent)
        return result
    replay_module.write_slam_replay = write


def install_static_corner_refinement(pipeline_module, policy):
    """Explicit runtime policy; ordinary exporter/Mac defaults are unchanged."""
    if policy not in ('subpix', 'apriltag'):
        raise ValueError('Unknown static corner refinement: '+str(policy))
    if policy == 'apriltag':
        from functools import partial
        pipeline_module.ArucoDetector = partial(
            pipeline_module.ArucoDetector, static_corner_refinement=True)


def validate_cached_corner_policy(arguments, policy):
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--reuse-observations',type=Path)
    args,_=parser.parse_known_args(arguments)
    if args.reuse_observations is None:
        return
    provenance=args.reuse_observations.resolve().parent/'runtime.json'
    cached='subpix'
    metadata={}
    if provenance.exists():
        metadata=json.loads(provenance.read_text())
        cached=metadata.get('runtime_config',{}).get(
            'static_corner_refinement','subpix')
    if cached!=policy:
        raise ValueError('Cached marker corners use '+cached+', requested '+policy+
                         '; remove --reuse-observations to remeasure the raw video.')
    if policy=='apriltag' and metadata.get('static_corner_schema')!=STATIC_CORNER_SCHEMA:
        raise ValueError('Cached line-refined corners lack the current pixel/mask contract; '
                         'remove --reuse-observations to remeasure the raw video.')


def main():
    project=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--runtime-config',type=Path,default=project/'.monotag/runtime.json')
    options,arguments=parser.parse_known_args()
    if '--help' in arguments:
        print(__doc__+'\nExample: python3 process_monotag.py VIDEO --calib JSON --head-slam --output NEW/actions.jsonl')
        return
    output=validate_arguments(arguments)
    if not any(a=='--execution' or a.startswith('--execution=') for a in arguments):
        arguments+=['--execution','local']
    if sys.platform!='linux':raise SystemExit('Processing runs on Ubuntu only. Mac records, transfers and views results; no local fallback.')
    config=json.loads(options.runtime_config.read_text())
    validate_runtime(config)
    corner_policy=config.get('static_corner_refinement','subpix')
    validate_cached_corner_policy(arguments,corner_policy)
    profile=json.loads((project/'config/monotag_ubuntu_profile.json').read_text())
    environment=dict(profile['environment'])
    for key in environment:
        if key in os.environ:
            if os.environ[key] not in ('0','1'):raise ValueError('Expected 0 or 1 for '+key)
            environment[key]=os.environ[key]
    environment['LD_LIBRARY_PATH']=':'.join([str(Path(config['library']).parent),*config['library_paths']])
    os.environ.update(environment)
    for path in reversed(config.get('python_paths',[])):sys.path.insert(0,path)
    import aruco_track
    if config.get('replay_module_path'):aruco_track.__path__.insert(0,config['replay_module_path'])
    import aruco_track.orbslam3_backend as backend
    from aruco_track.marker_candidate_hints import write_candidate_hints
    baseline_writer=backend.write_tag_observation_hints
    output.parent.mkdir(parents=True,exist_ok=True)
    def writer(path,*args,**kwargs):
        baseline_writer(path,*args,**kwargs);old=path.read_text().splitlines()
        write_candidate_hints(path,*args,**kwargs);new=path.read_text().splitlines()
        merged=merge_candidate_lines(old,new)
        path.write_text('\n'.join(merged)+'\n')
        (output.parent/'hint_preservation.json').write_text(json.dumps(dict(
            preserved_valid=sum(not l.startswith('#') and l.split()[1]=='1' for l in old),
            added_candidates=sum(a!=b for a,b in zip(old,merged)),changed_existing_valid=0),indent=2))
    backend.write_tag_observation_hints=writer
    native=backend.run_orbslam3_sequence
    def run_native(*args,**kwargs):
        env=dict(kwargs.get('environment_overrides') or {})
        for key in ('LD_LIBRARY_PATH','ORB_SLAM3_WINDOW_LOOP','ORB_SLAM3_MARKER_REVISIT_LOOP','ORB_SLAM3_POSE_FLOW_VETO','MONOTAG_ABLATE_INTERVAL'):
            env[key]=os.environ[key]
        kwargs['environment_overrides']=env
        if args:args=(Path(config['native_project']),)+args[1:]
        else:kwargs['project_dir']=Path(config['native_project'])
        return native(*args,**kwargs)
    backend.run_orbslam3_sequence=run_native
    import aruco_track.slam_replay as replay_module
    install_replay_display_assets(replay_module)
    from aruco_track import pipeline as pipeline_module
    install_static_corner_refinement(pipeline_module,corner_policy)
    import export_action_labels as exporter
    exporter._MONOTAG_FINAL_FRAME_ADAPTER = config.get('final_frame_adapter')
    from aruco_track.offline_gaps import install
    install(exporter,config['gap_adapter'])
    original_cache_key=exporter.sequence_cache_key
    binding=json.dumps(dict(profile=environment,library=config['library_sha256'],driver=config['driver_sha256'],
        hints=digest(project/'aruco_track/marker_candidate_hints.py'),
        pose=digest(project/'aruco_track/pose.py'),
        bandsolve=digest(project/'aruco_track/bandsolve.py'),
        tag_graph=digest(project/'aruco_track/tag_graph.py'),
        marker_corners=digest(project/'aruco_track/marker_corners.py'),
        static_corner_refinement=corner_policy,
        detector=digest(project/'aruco_track/detector.py'),
        final_frame_adapter=config.get('final_frame_adapter_sha256')),sort_keys=True)
    exporter.sequence_cache_key=lambda *a,**kw:hashlib.sha256((original_cache_key(*a,**kw)+binding).encode()).hexdigest()
    report=dict(status='running',argv=arguments,environment=environment,library=config['library'],
        runtime_config=config,odometry_input_to_slam=False,profile=profile['profile'])
    report['static_corner_schema']=STATIC_CORNER_SCHEMA if corner_policy=='apriltag' else None
    start=time.monotonic()
    try:
        sys.argv=['export_action_labels.py',*arguments]
        exporter.main();report['status']='complete'
    except SystemExit as error:
        report['status']='complete' if error.code in (None,0) else 'failed'
        raise
    except BaseException:
        report['status']='failed';raise
    finally:
        report['wall_s']=time.monotonic()-start
        (output.parent/'runtime.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
