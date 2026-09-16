"""Deploy to an inactive candidate, never mutate the active production release."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aruco_track.server_pipeline import digest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--skip-build',action='store_true',help='only when all native source hashes are unchanged')
    args=p.parse_args();cfg=json.loads(args.config.read_text())
    if cfg.get('enabled'):raise ValueError('Deploy into a disabled candidate, not the active release')
    # A candidate can be updated during preparation; production remains on its previous version.
    packaged=json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/package_release.py'),
                           '--output',str(ROOT/'output/server_migration_20260911/releases')],text=True))
    manifest=json.loads(Path(packaged['manifest']).read_text())
    if args.skip_build:
        old=json.loads(Path(cfg['manifest']).read_text())['files']
        native=lambda x:x.startswith('third_party/ORB_SLAM3/') or (x.startswith('tests/') and Path(x).suffix in {'.cc','.h','.cpp'})
        if {k:v for k,v in old.items() if native(k)}!={k:v for k,v in manifest['files'].items() if native(k)}:
            raise ValueError('Native sources changed: a rebuild is required')
    host=cfg['host'];ssh=['ssh','-o','BatchMode=yes','-p',str(cfg['port']),host]
    def remote(argv):subprocess.run(ssh+[shlex.join(list(map(str,argv)))],check=True)
    archive=cfg['root']+'/archives/'+Path(packaged['archive']).name
    remote_manifest=cfg['root']+'/archives/'+Path(packaged['manifest']).name
    remote(['mkdir','-p',cfg['root']+'/archives',cfg['release_dir']])
    subprocess.run(['scp','-s','-P',str(cfg['port']),packaged['archive'],packaged['manifest'],host+':'+cfg['root']+'/archives/'],check=True)
    check='import hashlib,sys;assert hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()==sys.argv[2]'
    remote([cfg['python'],'-c',check,archive,manifest['archive_sha256']])
    remote(['tar','-xzf',archive,'-C',cfg['release_dir']])
    if not args.skip_build:remote(['bash',cfg['release_dir']+'/scripts/build_linux.sh',cfg['dependencies'],'6'])
    cfg['manifest']=str(Path(packaged['manifest']).resolve());cfg['remote_manifest']=remote_manifest
    args.config.write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(dict(release=manifest['release'],config=str(args.config),activated=False)))

if __name__=='__main__':main()
