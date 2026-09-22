#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"

result_root="output/current_best_global_shutter_20260905"
mkdir -p "$result_root"

left_band="output/calibrated/strap_band_L_calibrated.json"
right_band="output/calibrated/strap_band_R_calibrated.json"
default_calibration="calib/camera_usb_1920x1080.json"

run_one() {
    local key="$1" video="$2" calibration="$3" cache="$4"
    local directory="$result_root/$key"
    local output="$directory/actions.jsonl"
    local replay="$directory/actions_replay"
    mkdir -p "$directory"
    if [[ -f "$output" && -f "${output%.jsonl}.meta.json" \
          && -f "$replay/index.html" && -f "$replay/process.mp4" \
          && -f "$replay/manifest.json" ]]; then
        echo "SKIP complete $key"
        return
    fi
    if [[ -e "$output" || -e "$replay/atlas.osa" ]]; then
        echo "ERROR partial output requires inspection: $directory" >&2
        return 1
    fi
    local command=(
        .venv/bin/python tools/export_action_labels.py "$video"
        --calib "$calibration"
        --band "$left_band" --band "$right_band"
        --head-slam --auto-marker-map
        --output "$output"
    )
    if [[ "$cache" != "-" ]]; then
        local cache_meta="${cache%.jsonl}.meta.json"
        if python3 - "$cache_meta" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    c = json.loads(p.read_text()).get("observation_cache_contract", {})
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if (
    c.get("schema") == "aruco-image-hand-observations/v1"
    and c.get("algorithm_version") == 3
) else 1)
PY
        then
            command+=(--reuse-observations "$cache")
        else
            echo "CACHE stale; fresh observations $key"
        fi
    fi
    echo "START $key $(date '+%F %T')"
    if ! /usr/bin/time -lp "${command[@]}" >"$directory/run.log" 2>&1; then
        if [[ " ${command[*]} " == *" --reuse-observations "* ]] \
                && grep -Eq 'observation cache|cached observation' "$directory/run.log"; then
            echo "CACHE incompatible; retry fresh observations $key"
            printf '\n--- cache rejected; rerunning from raw RGB ---\n' >>"$directory/run.log"
            local fresh_command=()
            local skip_next=0
            local argument
            for argument in "${command[@]}"; do
                if (( skip_next )); then
                    skip_next=0
                elif [[ "$argument" == "--reuse-observations" ]]; then
                    skip_next=1
                else
                    fresh_command+=("$argument")
                fi
            done
            /usr/bin/time -lp "${fresh_command[@]}" >>"$directory/run.log" 2>&1
        else
            return 1
        fi
    fi
    echo "DONE  $key $(date '+%F %T')"
}

run_one "ips_20260901_120049" \
    "recordings/IPS_2026-09-01.12.00.49.7830.mp4" "$default_calibration" \
    "output/IPS_2026-09-01_120049_current_20260905/actions.jsonl"
run_one "cross_room_20260903_183009" \
    "recordings/global_shutter_cross_room_20260903_183009.mp4" "$default_calibration" \
    "output/cross_room_global_shutter_20260904_temporal_prior/actions.jsonl"
run_one "raw_20260827_225919" \
    "recordings/raw_20260827_225919_588422.avi" "$default_calibration" \
    "recordings/raw_20260827_225919_588422_full_session_actions.jsonl"
run_one "raw_20260831_111003" \
    "recordings/raw_20260831_111003_047657.avi" "$default_calibration" \
    "recordings/raw_20260831_111003_047657_actions.jsonl"
run_one "raw_20260901_151529" \
    "recordings/raw_20260901_151529_316732.avi" "$default_calibration" \
    "recordings/raw_20260901_151529_316732_refined_v4_actions.jsonl"
run_one "raw_20260902_230109" \
    "recordings/raw_20260902_230109_762196.avi" "$default_calibration" \
    "recordings/raw_20260902_230109_762196_actions.jsonl"
run_one "raw_20260903_095622" \
    "recordings/raw_20260903_095622_622496.avi" "$default_calibration" \
    "recordings/raw_20260903_095622_622496_actions.jsonl"
run_one "raw_20260903_100543" \
    "recordings/raw_20260903_100543_112848.avi" "$default_calibration" \
    "output/independent_marker_validation_20260903/rigid_coplanar/actions.jsonl"
run_one "raw_20260904_100634" \
    "recordings/raw_20260904_100634_316877.avi" \
    "calib/camera_usb_1bcf_28c4_1920x1080_v2.json" \
    "recordings/raw_20260904_100634_316877_actions.jsonl"
run_one "uvc90_20260904_201356" \
    "recordings/samsung_global_shutter/UVC90_20260904_201356.mp4" \
    "calib/camera_usb_1bcf_28c4_1920x1080_v2.json" \
    "output/samsung_global_loop_ba_final_v2_20260905/actions.jsonl"

# Confirmed USB global-shutter rate-comparison sources. These have no prior
# action cache, so the current detector and hand model run once from raw RGB.
run_one "fps_1080p60_20260827_225130" \
    "recordings/fps_compare_1080p60_20260827_225130.avi" "$default_calibration" "-"
run_one "fps_120_20260827_224742" \
    "recordings/fps_compare_120_20260827_224742.avi" "$default_calibration" "-"
run_one "fps_120_20260827_224855" \
    "recordings/fps_compare_120_20260827_224855.avi" "$default_calibration" "-"
run_one "fps_60_20260827_224742" \
    "recordings/fps_compare_60_20260827_224742.avi" "$default_calibration" "-"
run_one "fps_720p120_20260827_225130" \
    "recordings/fps_compare_720p120_20260827_225130.avi" "$default_calibration" "-"

echo "ALL COMPLETE $(date '+%F %T')"
