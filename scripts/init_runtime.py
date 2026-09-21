"""Bind a fresh Ubuntu source build to process_monotag.py, without hand weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deps-prefix', type=Path, required=True)
    p.add_argument('--vocabulary', type=Path, required=True)
    args = p.parse_args()
    if sys.platform != 'linux':
        p.error('The validated processing platform is Ubuntu 24.04.')
    native = ROOT / 'third_party/ORB_SLAM3'
    binaries = {'library': native / 'lib/libORB_SLAM3.so',
                'driver': native / 'Examples/Monocular/mono_tum_headless',
                'gap_adapter': native / 'Examples/Monocular/relocalize_gap',
                'final_frame_adapter': native / 'Examples/Monocular/relocalize_prefix_readonly'}
    deps = args.deps_prefix.resolve()
    libraries = [deps / 'opencv-4.10/lib', deps / 'pangolin/lib',
                 native / 'Thirdparty/DBoW2/lib', native / 'Thirdparty/g2o/lib']
    vocabulary = args.vocabulary.resolve()
    for path in [*binaries.values(), vocabulary]:
        if not path.is_file() or path.stat().st_size == 0:
            p.error(f'Missing or empty resource: {path}')
    for path in libraries:
        if not path.is_dir():
            p.error(f'Missing library directory: {path}')
    target = ROOT / '.monotag/runtime.json'
    if target.exists():
        p.error('Runtime already exists; retain it or explicitly move it before rebinding.')
    link = native / 'Vocabulary/ORBvoc.txt'
    if link.exists() or link.is_symlink():
        if link.resolve() != vocabulary:
            p.error('Vocabulary path already contains a different resource.')
    else:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(vocabulary)
    cfg = {'schema': 'monotag-runtime/v1', 'native_project': str(ROOT),
           'python_paths': [], 'library_paths': [str(x) for x in libraries],
           'static_corner_refinement': 'apriltag'}
    for key, path in binaries.items():
        cfg[key] = str(path)
        cfg[key + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(cfg, stream, indent=2)
    print(target)


if __name__ == '__main__':
    main()
