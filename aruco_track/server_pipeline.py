"""Server deployment transport; the computation remains export_action_labels.py."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT/'.local/server.json'

def json_argument(value):
    if isinstance(value,Path): return str(value)
    if isinstance(value,set): return sorted(value)
    raise TypeError(f'Unsupported argument type: {type(value).__name__}')

def default_execution():
    if DEFAULT_CONFIG.is_file() and json.loads(DEFAULT_CONFIG.read_text()).get('enabled'):
        return 'server'
    return 'local'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def verify_release(root, manifest):
    root=Path(root).resolve()
    for rel, expected in manifest['files'].items():
        path=(root/rel).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path)!=expected:
            raise ValueError(f'Release source mismatch: {rel}; redeploy before running')

def extract_results(archive, target):
    target=Path(target).resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if (not (member.isfile() or member.isdir()) or
                    not (target/member.name).resolve().is_relative_to(target)):
                raise ValueError('Unsafe server result archive member')
        tar.extractall(target)

def run_remote_pipeline(args):
    if args.reuse_observations:
        raise ValueError('Server entry currently accepts fresh video jobs; use --execution local '
                         'for historical local --reuse-observations caches (no silent path rewriting)')
    if args.lerobot_output:
        raise ValueError('Run the optional training export separately on the returned labels')
    for key in ('save_atlas','marker_map_output','graph_diagnostics','slam_debug_video'):
        requested=getattr(args,key,None)
        if requested and Path(requested).exists():
            raise FileExistsError(f'Refusing to overwrite {key}: {requested}')
    for key in ('save_atlas','load_atlas'):
        requested=getattr(args,key,None)
        if requested and Path(requested).suffix!='.osa':
            raise ValueError(f'{key} must end in .osa')
    config_path=Path(args.server_config).resolve()
    cfg=json.loads(config_path.read_text())
    manifest=json.loads(Path(cfg['manifest']).read_text())
    verify_release(ROOT,manifest)
    from .hawor_remote import _remote_path
    import re
    host=cfg['host']
    if not re.fullmatch(r'(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9_][A-Za-z0-9_.-]*',host):
        raise ValueError('Invalid SSH host')
    port=int(cfg['port'])
    if not 1<=port<=65535: raise ValueError('Invalid SSH port')
    remote_root=_remote_path(cfg['root']); release=_remote_path(cfg['release_dir'])
    python=_remote_path(cfg['python'])
    video=Path(args.video).resolve()
    if not video.is_file(): raise FileNotFoundError(video)
    output=Path(args.output).resolve() if args.output else video.with_name(video.stem+'_actions.jsonl')
    package=output.parent/(output.stem+'_server_result')
    if output.exists() or output.with_suffix('.meta.json').exists() or package.exists():
        raise FileExistsError('Choose a new output path; previous results are preserved')
    output.parent.mkdir(parents=True,exist_ok=True)
    job=uuid.uuid4().hex
    staging=output.parent/('.server-'+job);staging.mkdir()
    remote_job=remote_root+'/jobs/'+job
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=30',
         '-o','ServerAliveCountMax=3','-p',str(port),host]
    scp=['scp','-s','-o','BatchMode=yes','-P',str(port)]
    def remote(argv):
        return subprocess.check_output(ssh+[shlex.join(list(map(str,argv)))],text=True)
    print('Server pipeline: verifying deployed sources; local SLAM will not run',flush=True)
    remote([python,str(Path(release)/'scripts/server_worker.py'),'--verify',cfg['remote_manifest']])
    remote(['mkdir','-p',remote_job,remote_root+'/inputs'])
    uploads={}
    def upload(path, name):
        path=Path(path).resolve();sha=digest(path)
        target=remote_root+'/inputs/'+sha+path.suffix
        seed=cfg.get('seed_inputs',{}).get(sha)
        if seed: target=_remote_path(seed)
        check='import hashlib,pathlib,sys;p=pathlib.Path(sys.argv[1]);h=hashlib.sha256();\nif p.is_file():\n with p.open("rb") as f:\n  for b in iter(lambda:f.read(1048576),b""):h.update(b)\n print(h.hexdigest())\nelse:print("missing")'
        actual=remote([python,'-c',check,target]).strip()
        if actual=='missing':
            target=remote_root+'/inputs/'+sha+path.suffix
            pending=remote_job+'/'+name+'.upload'
            print(f'Server pipeline: uploading {name} ({path.stat().st_size/1e6:.1f} MB)',flush=True)
            subprocess.run(scp+[str(path),host+':'+pending],check=True)
            from .hawor_remote import _PUBLISH
            remote([python,'-c',_PUBLISH,pending,target,sha])
        elif actual!=sha:
            raise ValueError(f'Remote input hash mismatch: {name}')
        uploads[name]=dict(local=str(path),remote=target,sha256=sha)
        return target
    mapped=dict(vars(args));mapped['execution']='local'
    mapped.pop('server_config',None);mapped.pop('open_replay',None)
    mapped['video']=upload(video,'video')
    mapped['calib']=upload(args.calib,'calibration')
    mapped['band']=[upload(p,'band_'+str(i)) for i,p in enumerate(args.band)]
    if args.world_board: mapped['world_board']=upload(args.world_board,'world_board')
    if args.load_atlas: mapped['load_atlas']=upload(args.load_atlas,'load_atlas')
    if args.hand_joints and args.hand_backend=='mediapipe': mapped['hand_model']=upload(args.hand_model,'hand_model')
    else: mapped.pop('hand_model',None)
    mapped['hawor_config']=cfg['hawor_config'];mapped['hawor_device']='cuda'
    mapped['output']=remote_job+'/output/actions.jsonl'
    for key in ('save_atlas','marker_map_output','graph_diagnostics','slam_debug_video'):
        value=mapped.get(key)
        if value:
            suffix=Path(value).suffix
            mapped[key]=remote_job+'/output/'+key+(suffix or ('.osa' if key=='save_atlas' else '.jsonl'))
    request=dict(job=job,release=manifest['release'],arguments=mapped,inputs=uploads,
                 remote_job=remote_job,manifest=cfg['remote_manifest'],dependencies=cfg['dependencies'],
                 cuda_device=str(cfg.get('cuda_device','2')))
    request_path=staging/'request.json';request_path.write_text(json.dumps(request,default=json_argument,indent=2)+'\n')
    subprocess.run(scp+[str(request_path),host+':'+remote_job+'/request.json'],check=True)
    with (staging/'run.log').open('w') as log:
        command=ssh+[shlex.join([python,release+'/scripts/server_worker.py',remote_job+'/request.json'])]
        with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as proc:
            for line in proc.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            if proc.wait(): raise RuntimeError(f'Server job failed; original input preserved. See {staging}/run.log')
    subprocess.run(scp+[host+':'+remote_job+'/result.tar.gz',str(staging/'result.tar.gz')],check=True)
    subprocess.run(scp+[host+':'+remote_job+'/status.json',str(staging/'status.json')],check=True)
    status=json.loads((staging/'status.json').read_text())
    if status.get('status')!='complete' or status.get('release')!=manifest['release'] or digest(staging/'result.tar.gz')!=status.get('archive_sha256'):
        raise ValueError('Server result integrity/identity check failed')
    unpack=staging/'unpacked';unpack.mkdir();extract_results(staging/'result.tar.gz',unpack)
    if not (unpack/'actions.jsonl').is_file(): raise ValueError('Missing final action labels')
    unpack.rename(package)
    # Keep the returned package immutable. Links preserve all remote provenance.
    output.symlink_to(package/'actions.jsonl')
    output.with_suffix('.meta.json').symlink_to(package/'actions.meta.json')
    for key in ('save_atlas','marker_map_output','graph_diagnostics','slam_debug_video'):
        requested=getattr(args,key,None)
        if requested:
            source=package/Path(mapped[key]).name
            target=Path(requested).resolve()
            if source.exists() and not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source)
    (package/'local_paths.json').write_text(json.dumps(dict(inputs=uploads,server_status=status),indent=2)+'\n')
    print(f'Server complete: {output}\nReplay: {package / "actions_replay/index.html"}',flush=True)
    if args.open_replay and (package/'actions_replay/index.html').exists():
        subprocess.Popen([sys.executable,str(ROOT/'replay_orb_slam.py'),str(package/'actions_replay')],
                         stdout=(staging/'replay_server.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    return package
