# Final Script with Explicit Imports to guarantee Plane is loaded
import numpy as np
from manim import (
    ThreeDScene,
    Sphere,
    Dot3D,
    Plane,
    Create,
    DEGREES,
    ORIGIN,
    BLUE,
    YELLOW,
    RED,
    CYAN,
    UP,
    RIGHT,
    Text,
    linear,
)


class FRBTiming(ThreeDScene):
    def construct(self):
        # Set up the 3D camera for a good view of the Earth and the incoming wave
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.1)

        # --- 1. Create the Earth and Telescope Locations ---
        earth = Sphere(center=ORIGIN, radius=1.0, resolution=(50, 50))
        earth.set_color(BLUE)
        earth.set_opacity(0.8)
        self.add(earth)

        # Define telescope locations in latitude and longitude
        so_cal_lat, so_cal_lon = 34.0, -118.0
        bc_lat, bc_lon = 51.0, -123.0

        # Helper function to convert spherical (lat, lon) to 3D Cartesian coordinates
        def lat_lon_to_cartesian(lat, lon, radius=1.0):
            lat_rad = np.deg2rad(lat)
            lon_rad = np.deg2rad(lon)
            x = radius * np.cos(lat_rad) * np.cos(lon_rad)
            y = radius * np.cos(lat_rad) * np.sin(lon_rad)
            z = radius * np.sin(lat_rad)
            return np.array([x, y, z])

        # Calculate 3D positions for the telescopes on the Earth's surface
        so_cal_pos = lat_lon_to_cartesian(so_cal_lat, so_cal_lon, 1.01)
        bc_pos = lat_lon_to_cartesian(bc_lat, bc_lon, 1.01)

        # Create 3D dots to represent the telescopes
        so_cal_telescope = Dot3D(so_cal_pos, color=YELLOW, radius=0.05)
        bc_telescope = Dot3D(bc_pos, color=RED, radius=0.05)

        self.add(so_cal_telescope, bc_telescope)

        # --- 2. Create and Animate the FRB Wavefront ---
        dec = 70 * DEGREES
        frb_direction = np.array([0, -np.cos(dec), np.sin(dec)])

        # Create the FRB wavefront as a transparent plane
        wavefront = Plane(
            normal_vector=frb_direction,
            center=4 * -frb_direction,  # Start 4 units away
            width=8,
            height=8,
        )
        wavefront.set_color(CYAN)
        wavefront.set_opacity(0.5)

        # --- 3. Run the Animation ---
        self.play(Create(wavefront))
        self.wait(1)

        # Animate the wavefront moving towards and past Earth
        self.play(
            wavefront.animate.shift(8 * frb_direction),
            run_time=6,
            rate_func=linear,
        )
        self.wait(2)
