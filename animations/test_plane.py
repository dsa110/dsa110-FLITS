from manim import *

class TestPlane(ThreeDScene):
    def construct(self):
        # This script tests if the 'Plane' class is working in your Manim installation.
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        try:
            # Attempt to create a Plane
            test_plane = Plane()
            info_text = Text("Plane class works!", color=GREEN)
            self.add(test_plane)
        except NameError:
            # This will run if 'Plane' is not defined
            info_text = Text("NameError: 'Plane' is not defined.", color=RED)

        self.add(info_text)
        self.wait(3)
