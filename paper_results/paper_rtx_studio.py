"""Shared Newton RTX studio for reproducible paper renderings."""

import newton.viewer
import numpy as np
import warp as wp

from paper_render_style import PAPER_STUDIO, rgb

STUDIO_COLOR = rgb(PAPER_STUDIO.backdrop, linear=True)
STUDIO_AMBIENT = PAPER_STUDIO.rtx_ambient
STUDIO_KEY = PAPER_STUDIO.rtx_key
STUDIO_FILL = PAPER_STUDIO.rtx_fill
STUDIO_RIM = PAPER_STUDIO.rtx_rim
STUDIO_KEY_ANGLE = PAPER_STUDIO.key_angle_degrees
STUDIO_BACKDROP_START = PAPER_STUDIO.backdrop_start
STUDIO_BACKDROP_RADIUS = PAPER_STUDIO.backdrop_radius
STUDIO_BACKDROP_EMISSION = PAPER_STUDIO.rtx_backdrop_emission
STUDIO_ROD_ROUGHNESS = PAPER_STUDIO.robot_roughness
STUDIO_FLOOR_HEIGHT = -0.05


def look_at(viewer, eye, target) -> None:
    """Aim a Z-up viewer camera at a world-space point."""
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    direction = target - eye
    direction /= np.linalg.norm(direction)
    viewer.set_camera(
        wp.vec3(*eye),
        pitch=float(np.rad2deg(np.arcsin(direction[2]))),
        yaw=float(np.rad2deg(np.arctan2(direction[1], direction[0]))),
    )


class PaperViewerRTX(newton.viewer.ViewerRTX):
    """RTX viewer with a neutral studio backdrop and lights, and opt-in shadow-link exclusion."""

    shadow_excluded_mesh_names: tuple[str, ...] = ()

    def __init__(self, *args, samples_per_pixel: int, **kwargs):
        """Configure the path-tracer sample budget.

        Args:
            samples_per_pixel: Path-tracer sample count per frame.
        """
        if samples_per_pixel < 1:
            raise ValueError("samples_per_pixel must be positive")
        self.samples_per_pixel = samples_per_pixel
        super().__init__(*args, environment="none", async_rendering=False, **kwargs)
        self.show_ui = False
        self.show_joints = False
        self.show_contacts = False
        self.show_com = False
        self.show_inertia_boxes = False
        self.show_collision = False

    def _add_camera_lights_and_render_product(self) -> None:
        """Configure the path-tracer render product, then add the studio."""
        from pxr import Sdf

        super()._add_camera_lights_and_render_product()
        product = self.stage.GetPrimAtPath(self._render_product_path)
        product.GetAttribute("omni:rtx:rendermode").Set("PathTracing")
        product.GetAttribute("omni:rtx:quality").Set(1)
        product.GetAttribute("omni:rtx:pt:samplesPerPixel").Set(self.samples_per_pixel)
        product.CreateAttribute("omni:rtx:pt:samplesPerIteration", Sdf.ValueTypeNames.Int).Set(
            min(32, self.samples_per_pixel)
        )
        product.GetAttribute("omni:rtx:pt:denoising:enabled").Set(self.samples_per_pixel > 1)
        product.GetAttribute("omni:rtx:dlss:frameGeneration").Set(False)
        self._add_studio()

    def _render_and_display(self) -> None:
        """Render until the path tracer converges to samples_per_pixel; save_screenshot() reads the result."""
        if self._rtx is None or self._should_close:
            return
        # Reset sensor history after pose/visibility changes so static snapshots
        # accumulate samples against one scene instead of stale motion history.
        self._rtx.reset()
        for attempt in range(self.samples_per_pixel + 2):
            self._render_products = self._rtx.step(
                render_products={self._render_product_path},
                delta_time=1.0 / self.fps if attempt == 0 else 0.0,
            )
            frames = [frame for product in self._render_products.values() for frame in product.frames]
            if frames and all(frame.converged for frame in frames):
                break
        else:
            raise RuntimeError(
                f"RTX did not converge to the requested sample count: {[frame.progression for frame in frames]}"
            )

    def _capture_screenshot_pixels(self) -> np.ndarray:
        """Read LdrColor from the last render; ovrtx may prefix the render-var key with its path."""
        from ovrtx import Device

        for product in self._render_products.values():
            for frame in product.frames:
                for key, render_var in frame.render_vars.items():
                    if key == "LdrColor" or key.endswith("/LdrColor"):
                        with render_var.map(device=Device.CPU) as mapping:
                            return np.array(np.from_dlpack(mapping), copy=True)
        raise RuntimeError("save_screenshot() could not find the LdrColor render output")

    def _apply_ground_material(self) -> None:
        """Skip Newton's ground material; _add_studio binds the backdrop material to the ground."""

    def _add_studio(self) -> None:
        """Author the studio backdrop mesh and key/fill/rim/ambient lights, excluding shadow_excluded_mesh_names from their shadows."""
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        material = UsdShade.Material.Define(self.stage, "/root/Studio/Matte")
        shader = UsdShade.Shader.Define(self.stage, "/root/Studio/Matte/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*STUDIO_COLOR))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(STUDIO_BACKDROP_EMISSION))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        bend = STUDIO_BACKDROP_START
        radius_y, radius_z = STUDIO_BACKDROP_RADIUS
        span = max(20.0, 4.0 * (bend + radius_y))
        profile = [(-span, 0.0), (bend, 0.0)]
        normals = [(0.0, 0.0, 1.0)] * 2
        for angle in np.linspace(0.0, np.pi / 2, 33)[1:]:
            profile.append((bend + radius_y * np.sin(angle), radius_z * (1.0 - np.cos(angle))))
            normal = np.asarray((0.0, -radius_z * np.sin(angle), radius_y * np.cos(angle)))
            normals.append(tuple(normal / np.linalg.norm(normal)))
        profile.append((bend + radius_y, span))
        normals.append((0.0, -1.0, 0.0))
        vertices = [(float(x), float(y), float(STUDIO_FLOOR_HEIGHT + z)) for y, z in profile for x in (-span, span)]
        mesh = UsdGeom.Mesh.Define(self.stage, "/root/Studio/Backdrop")
        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr([4] * (len(profile) - 1))
        mesh.CreateFaceVertexIndicesAttr(
            [index for row in range(len(profile) - 1) for index in (2 * row, 2 * row + 1, 2 * row + 3, 2 * row + 2)]
        )
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateNormalsAttr([Gf.Vec3f(*normal) for normal in normals for _ in range(2)])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        for name, ground in self._meshes.items():
            if "plane" in name.lower():
                UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material)

        rod_material = UsdShade.Material.Define(self.stage, "/root/Studio/Rod")
        rod_surface = UsdShade.Shader.Define(self.stage, "/root/Studio/Rod/Surface")
        rod_surface.CreateIdAttr("UsdPreviewSurface")
        color = UsdShade.Shader.Define(self.stage, "/root/Studio/Rod/Color")
        color.CreateIdAttr("UsdPrimvarReader_float3")
        color.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        color.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        rod_surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            color.ConnectableAPI(), "result"
        )
        rod_surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(STUDIO_ROD_ROUGHNESS)
        rod_surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        rod_material.CreateSurfaceOutput().ConnectToSource(rod_surface.ConnectableAPI(), "surface")
        for name, prototype in self._meshes.items():
            if "capsule" in name.lower():
                UsdShade.MaterialBindingAPI.Apply(prototype.GetPrim()).Bind(rod_material)

        dome = UsdLux.DomeLight.Define(self.stage, "/root/Studio/Ambient")
        dome.CreateColorAttr(Gf.Vec3f(1.0))
        dome.CreateIntensityAttr(STUDIO_AMBIENT)
        key = UsdLux.DistantLight.Define(self.stage, "/root/Studio/Key")
        key.CreateColorAttr(Gf.Vec3f(1.0))
        key.CreateIntensityAttr(STUDIO_KEY)
        key.CreateAngleAttr(STUDIO_KEY_ANGLE)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(*PAPER_STUDIO.key_rotation_degrees))
        for name, rotation, intensity in (
            ("Fill", PAPER_STUDIO.fill_rotation_degrees, STUDIO_FILL),
            ("Rim", PAPER_STUDIO.rim_rotation_degrees, STUDIO_RIM),
        ):
            # Distant fill lights cannot appear as dark spheres in the camera.
            light = UsdLux.DistantLight.Define(self.stage, f"/root/Studio/{name}")
            light.CreateColorAttr(Gf.Vec3f(1.0))
            light.CreateIntensityAttr(intensity)
            light.CreateAngleAttr(PAPER_STUDIO.fill_angle_degrees)
            UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(*rotation))

        mesh_paths = [
            self._mesh_prim_paths[self._qualify(name)]
            for name in self.shadow_excluded_mesh_names
            if self._qualify(name) in self._mesh_prim_paths
        ]
        for light_path in ("/root/Studio/Ambient", "/root/Studio/Key", "/root/Studio/Fill", "/root/Studio/Rim"):
            excludes = UsdLux.LightAPI(self.stage.GetPrimAtPath(light_path)).GetShadowLinkCollectionAPI().CreateExcludesRel()
            for mesh_path in mesh_paths:
                excludes.AddTarget(Sdf.Path(mesh_path))
