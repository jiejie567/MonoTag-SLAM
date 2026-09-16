"""Upgrade one completed replay's display only, without re-running SLAM."""
import argparse
from pathlib import Path
import shutil

def refresh(directory):
    directory=Path(directory).resolve()
    index=directory/'index.html'
    if not (directory/'manifest.json').is_file():
        raise ValueError('Expected a completed replay directory with manifest.json')
    text=index.read_text()
    tag='<script src="replay_event_overlay.js"></script>'
    if tag not in text:
        if '</body>' not in text:raise ValueError('Replay HTML has no body closing tag')
        backup=directory/'index.before-event-overlay.html'
        if not backup.exists():shutil.copy2(index,backup)
        text=text.replace('</body>',tag+'</body>')
        index.write_text(text)
    shutil.copy2(Path(__file__).resolve().parents[1]/'aruco_track/replay_event_overlay.js',directory/'replay_event_overlay.js')
    print(f'Updated display only: {index}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay',type=Path)
    refresh(parser.parse_args().replay)
