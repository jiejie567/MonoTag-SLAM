from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


DEFAULT_UVC_DEVICE = "1d6b:0102"
DEFAULT_EXPOSURE_US = 8_000
DEFAULT_GAIN = 160
DEFAULT_WHITE_BALANCE = 128


class UVCControlError(RuntimeError):
    pass


class UVCController:
    def __init__(self, device: str = DEFAULT_UVC_DEVICE, executable: Path | None = None):
        project_root = Path(__file__).resolve().parents[1]
        self.executable = executable or project_root / "third_party/camtint/bin/uvcctl"
        self.device = device

    def _run(self, command: str, *arguments: str) -> str:
        if not self.executable.is_file():
            raise UVCControlError(
                f"UVC control helper is missing: {self.executable}; run setup.sh"
            )
        completed = subprocess.run(
            [str(self.executable), command, *arguments, "-d", self.device],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise UVCControlError(message or f"uvcctl {command} failed")
        return completed.stdout.strip()

    def capabilities(self) -> dict[str, dict]:
        payload = json.loads(self._run("caps"))
        return {control["name"]: control for control in payload["controls"]}

    def _set_verified(self, name: str, value: int) -> int:
        self._run("set", name, str(value))
        actual = int(self._run("get", name))
        if actual != value:
            raise UVCControlError(f"{name}: requested {value}, camera reports {actual}")
        return actual

    @staticmethod
    def _require_value(controls: dict[str, dict], name: str, value: int) -> None:
        control = controls.get(name)
        if control is None or not control.get("writable", False):
            raise UVCControlError(f"camera does not expose writable UVC control: {name}")
        if not int(control["min"]) <= value <= int(control["max"]):
            raise UVCControlError(
                f"{name}={value} is outside camera range "
                f"{control['min']}..{control['max']}"
            )

    def apply_auto(self) -> dict[str, int | str]:
        controls = self.capabilities()
        for name, value in (
            ("white_balance_temperature_auto", 1),
            ("exposure_auto", 8),
        ):
            self._require_value(controls, name, value)
            self._set_verified(name, value)
        return {"mode": "auto", "exposure_auto": 8, "white_balance_auto": 1}

    def apply_locked(
        self,
        exposure_us: int = DEFAULT_EXPOSURE_US,
        gain: int = DEFAULT_GAIN,
        white_balance: int = DEFAULT_WHITE_BALANCE,
        power_line_frequency_hz: int = 50,
        fps: float = 60.0,
    ) -> dict[str, int | str]:
        if exposure_us <= 0 or exposure_us % 100 != 0:
            raise UVCControlError("--exposure-us must be a positive multiple of 100")
        if fps > 0 and exposure_us > 1_000_000.0 / fps:
            raise UVCControlError(
                f"{exposure_us} us exposure is longer than one {fps:g} FPS frame"
            )
        exposure_units = exposure_us // 100
        frequency_value = {0: 0, 50: 1, 60: 2}.get(power_line_frequency_hz)
        if frequency_value is None:
            raise UVCControlError("power-line frequency must be 0, 50, or 60 Hz")
        requested = [
            ("exposure_auto", 1),
            ("white_balance_temperature_auto", 0),
            ("power_line_frequency", frequency_value),
            ("exposure_time_absolute", exposure_units),
            ("white_balance_temperature", white_balance),
        ]
        controls = self.capabilities()
        if "gain" in controls:
            requested.insert(4, ("gain", gain))
        for name, value in requested:
            self._require_value(controls, name, value)
        for name, value in requested:
            self._set_verified(name, value)
        return {
            "mode": "locked",
            "exposure_us": exposure_us,
            "gain": gain if "gain" in controls else "unsupported",
            "white_balance": white_balance,
            "power_line_frequency_hz": power_line_frequency_hz,
        }


def add_camera_control_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--camera-controls",
        choices=("locked", "auto"),
        default="locked",
        help="lock exposure/WB for tracking (default) or use camera auto controls",
    )
    parser.add_argument("--exposure-us", type=int, default=DEFAULT_EXPOSURE_US)
    parser.add_argument("--gain", type=int, default=DEFAULT_GAIN)
    parser.add_argument(
        "--white-balance",
        type=int,
        default=DEFAULT_WHITE_BALANCE,
        help="camera-native white-balance value (this model exposes 0..255, not Kelvin)",
    )
    parser.add_argument(
        "--power-line-frequency",
        type=int,
        choices=(0, 50, 60),
        default=50,
    )
    parser.add_argument("--uvc-device", default=DEFAULT_UVC_DEVICE, metavar="VID:PID")


def apply_camera_control_arguments(args, fps: float) -> dict[str, int | str]:
    controller = UVCController(args.uvc_device)
    if args.camera_controls == "auto":
        return controller.apply_auto()
    return controller.apply_locked(
        exposure_us=args.exposure_us,
        gain=args.gain,
        white_balance=args.white_balance,
        power_line_frequency_hz=args.power_line_frequency,
        fps=fps,
    )


def describe_camera_controls(settings: dict[str, int | str]) -> str:
    if settings["mode"] == "auto":
        return "CAMERA AUTO EXP/WB"
    gain = settings["gain"]
    gain_text = f"G{gain}" if gain != "unsupported" else "GAIN N/A"
    return (
        f"CAMERA LOCK EXP {int(settings['exposure_us']) / 1000.0:.1f}ms  "
        f"{gain_text}  WB{settings['white_balance']}  "
        f"{settings['power_line_frequency_hz']}Hz"
    )
