"""Build the read-only prefix localizer without touching CMake or shared objects."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def build_commands(root, native_root, output, build_dir=None):
    """Reuse the built frontend's compiler, ABI flags and platform link recipe."""
    build = build_dir if build_dir is not None else root / 'third_party/ORB_SLAM3/build'
    flags = {}
    for line in (build / 'CMakeFiles/mono_tum_headless.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            key, value = line.split(' = ', 1)
            flags[key] = shlex.split(value)
    source = root / 'third_party/ORB_SLAM3/Examples/Monocular/relocalize_prefix_readonly.cc'
    obj, binary = output / 'relocalize_prefix_readonly.o', output / 'relocalize_prefix_readonly'
    link = shlex.split((build / 'CMakeFiles/mono_tum_headless.dir/link.txt').read_text())
    objects = [i for i, token in enumerate(link) if token.endswith('mono_tum_headless.cc.o')]
    libraries = [i for i, token in enumerate(link)
                 if Path(token).name in ('libORB_SLAM3.so', 'libORB_SLAM3.dylib')]
    if len(objects) != 1 or len(libraries) != 1 or '-o' not in link:
        raise ValueError('Cached mono_tum_headless link recipe must identify one object and one ORB shared library')
    old_library = build / link[libraries[0]]
    native = native_root / 'third_party/ORB_SLAM3'
    library = native / 'lib' / old_library.name
    for required in (source, library):
        if not required.is_file():
            raise FileNotFoundError(f'Prefix adapter build prerequisite is missing: {required}')
    for i, token in enumerate(link):
        if i == objects[0]:
            link[i] = str(obj)
        elif i == libraries[0]:
            link[i] = str(library)
        elif token.startswith('-Wl,-rpath,'):
            # Linux CMake combines directories with colons; retain dependency paths.
            paths = token[len('-Wl,-rpath,'):].split(':')
            link[i] = '-Wl,-rpath,' + ':'.join(
                str(native / 'lib') if (build / path).resolve() == old_library.parent.resolve() else path
                for path in paths)
    link[link.index('-o') + 1] = str(binary)
    compile_command = [link[0], *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'],
                       *flags['CXX_FLAGS'], '-c', str(source), '-o', str(obj)]
    return compile_command, link, library


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root', type=Path, default=root)
    parser.add_argument('--output-dir', type=Path, default=root / 'third_party/ORB_SLAM3/Examples/Monocular')
    parser.add_argument('--build-dir', type=Path, default=root / 'third_party/ORB_SLAM3/build',
                        help='existing CMake build containing the frozen mono_tum_headless flags/link recipe')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    build = args.build_dir.resolve()
    compile_command, link, library = build_commands(root, args.native_root.resolve(), output, build)
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(compile_command, cwd=build, check=True)
    subprocess.run(link, cwd=build, check=True)
    binary = output / 'relocalize_prefix_readonly'
    print(json.dumps({'binary': str(binary), 'library': str(library)}, indent=2))


if __name__ == '__main__':
    main()
