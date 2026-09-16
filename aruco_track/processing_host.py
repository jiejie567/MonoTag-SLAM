"""Fail closed rather than accidentally starting offline processing on Mac."""
import json
import sys
from pathlib import Path


def require_processing_host(config_path=None, platform=None):
    path = Path(config_path) if config_path else Path(__file__).resolve().parents[1]/'config/processing_host.json'
    if not path.is_file():
        return
    policy = json.loads(path.read_text())
    if (platform or sys.platform) == 'darwin' and policy.get('allow_macos_processing') is False:
        raise RuntimeError(
            'Mac processing is disabled by the user. Run '+policy['entrypoint']+
            ' on '+policy['ssh_host']+' in '+policy['project']+
            '. Mac is for recording/transfers/replay only; no local fallback.')
