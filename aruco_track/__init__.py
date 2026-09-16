"""Small, testable building blocks for ArUco wrist tracking."""

# OpenCV 4.6 names the bitmap helper ``drawMarker``; OpenCV 4.7+ added the
# non-drawing ``generateImageMarker`` spelling.  Keep one compatibility alias
# at package import so detector, marker-cover and test paths use identical
# marker templates on Ubuntu 24.04 and newer OpenCV builds.
import cv2

if (hasattr(cv2, "aruco") and not hasattr(cv2.aruco, "generateImageMarker")
        and hasattr(cv2.aruco, "drawMarker")):
    cv2.aruco.generateImageMarker = cv2.aruco.drawMarker

from .models import BandLayout, Calibration, Pose

__all__ = ["BandLayout", "Calibration", "Pose"]
