#!/usr/bin/env python3
import unittest

from aruco_track.camera_controls import UVCControlError, UVCController


class _FakeController(UVCController):
    def __init__(self):
        self.values = {}

    def capabilities(self):
        names = (
            "exposure_auto",
            "white_balance_temperature_auto",
            "power_line_frequency",
            "exposure_time_absolute",
            "gain",
            "white_balance_temperature",
        )
        return {
            name: {"writable": True, "min": 0, "max": 8192}
            for name in names
        }

    def _set_verified(self, name, value):
        self.values[name] = value
        return value


class UVCControllerTests(unittest.TestCase):
    def test_locked_settings_convert_microseconds_and_disable_auto(self):
        controller = _FakeController()
        result = controller.apply_locked(
            exposure_us=8_000,
            gain=160,
            white_balance=128,
            power_line_frequency_hz=50,
            fps=120,
        )
        self.assertEqual(result["mode"], "locked")
        self.assertEqual(controller.values["exposure_auto"], 1)
        self.assertEqual(controller.values["exposure_time_absolute"], 80)
        self.assertEqual(controller.values["gain"], 160)
        self.assertEqual(controller.values["white_balance_temperature_auto"], 0)
        self.assertEqual(controller.values["power_line_frequency"], 1)

    def test_locked_settings_allow_camera_without_gain_control(self):
        controller = _FakeController()
        original_capabilities = controller.capabilities
        controller.capabilities = lambda: {
            name: control
            for name, control in original_capabilities().items()
            if name != "gain"
        }
        result = controller.apply_locked(white_balance=4600)
        self.assertEqual(result["gain"], "unsupported")
        self.assertNotIn("gain", controller.values)

    def test_rejects_exposure_longer_than_frame(self):
        controller = _FakeController()
        with self.assertRaises(UVCControlError):
            controller.apply_locked(exposure_us=20_000, fps=60)


if __name__ == "__main__":
    unittest.main()
