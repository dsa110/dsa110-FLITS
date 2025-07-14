# Artistic FRB Animation - Corrected Version
from manim import *
import numpy as np
from manim.utils.space_ops import rotation_between_vectors

class ArtisticFRB(ThreeDScene):
    def construct(self):
        # --- 1. Setup the Scene ---
        self.set_camera_orientation(phi=60 * DEGREES, theta=-100 * DEGREES, zoom=1.2)

        # Add a starry background image
        background = ImageMobject("stars.jpg")
        background.scale_to_fit_height(config.frame_height * 2)
        self.add(background)

        # --- 2. Create the Earth and Telescopes ---
        # Create a textured Earth sphere
        earth = Sphere(radius=1.5, resolution=(100, 100))
        # FIX: Updated syntax for applying textures
        earth.set_sheen(0.1, UL)
        earth.set_shader_unlit(texture_path="earth_night.jpg")

        # Rotate Earth to show the Americas
        earth.rotate(90 * DEGREES, axis=UP)
        earth.rotate(10 * DEGREES, axis=RIGHT)
        self.add(earth)

        # Function to place objects on the Earth's surface
        def place_on_surface(lat, lon, radius=1.51):
            lat_rad = np.deg2rad(lat)
            # Adjust for the initial rotation of the globe
            lon_rad = np.deg2rad(lon - 90)
            x = radius * np.cos(lat_rad) * np.cos(lon_rad)
            y = radius * np.cos(lat_rad) * np.sin(lon_rad)
            z = radius * np.sin(lat_rad)
            return np.array([x, y, z])

        # Telescope locations
        so_cal_pos = place_on_surface(34, -118)
        bc_pos = place_on_surface(51, -123)

        # Import telescope SVG
        telescope_svg = SVGMobject("telescope.svg").scale(0.2).set_color(WHITE)

        # Create two telescopes
        bc_telescope = telescope_svg.copy()
        so_cal_telescope = telescope_svg.copy()

        # FIX: Orient the 2D SVGs by calculating and applying the correct 3D rotation
        # This aligns the telescope's "up" direction with the normal of the sphere's surface.
        bc_rotation = rotation_between_vectors(UP, bc_pos)
        bc_telescope.apply_matrix(bc_rotation).move_to(bc_pos)

        so_cal_rotation = rotation_between_vectors(UP, so_cal_pos)
        so_cal_telescope.apply_matrix(so_cal_rotation).move_to(so_cal_pos)

        self.add(bc_telescope, so_cal_telescope)

        # --- 3. Create and Animate the FRB ---
        # FRB starting and ending points
        frb_start = np.array([5, 4, 3])
        frb_target = bc_pos # The beam will target the northern telescope first

        # Create the FRB beam as a glowing line
        frb_beam = Line(
            frb_start,
            frb_target,
            stroke_width=10,
            color=CYAN
        ).add_glow_effect(color=BLUE)

        # Add text labels
        explanation_text = Text("FRB signal arrives from deep space...", font_size=32).to_corner(UL)
        self.add_fixed_in_frame_mobjects(explanation_text)

        # Animate the beam traveling to the first telescope
        self.play(Create(frb_beam), run_time=2)
        self.play(Flash(bc_telescope, color=CYAN, flash_radius=0.3))

        # --- 4. Show the Time Delay ---
        # Animate a connecting line to show the baseline
        baseline_arc = ArcBetweenPoints(bc_pos, so_cal_pos, color=ORANGE, stroke_width=3)

        self.play(
            ReplacementTransform(explanation_text, Text("Signal detection is offset by light-travel time.", font_size=32).to_corner(UL))
        )
        self.play(ShowPassingFlash(baseline_arc, time_width=0.5, run_time=1.5))
        self.play(Flash(so_cal_telescope, color=ORANGE, flash_radius=0.3))

        self.wait(3)