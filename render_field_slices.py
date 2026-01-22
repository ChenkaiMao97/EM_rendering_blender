import bpy
import bmesh
import numpy as np
import math
from mathutils import Vector
import glob
import os

from utils.util import clear_scene, ensure_cycles, make_camera_light_autoframe, load_field, bbox_world, make_camera_autoframe
from utils.image import prepare_slices
from utils.material import make_emissive_image_material

NPY_PATH = "data/random1/src.npy"   # complex 3D array
CMAP = 'seismic'

AXIS_ORDER = "XYZ"               # "XYZ" means arr[x,y,z], "ZYX" means arr[z,y,x]
VOXEL_SIZE = 0.02
CENTER = True

#   "abs" | "real" | "imag"
FIELD_COMPONENT = "real"

# Slice indices (None -> use middle slice)
SLICE_X = None  # index along x (plane is YZ)
SLICE_Y = None  # index along y (plane is XZ)
SLICE_Z = None  # index along z (plane is XY)

# If True, use percentiles to avoid extreme outliers dominating the range
USE_PERCENTILE_RANGE = True
PCT_LOW = 1
PCT_HIGH = 99

# Emission/alpha shaping
EMISSION_STRENGTH = 1.5   # global multiplier (Cycles: this matters a lot)
GAMMA_TRANSPARENCY = 1.0 # >1 boosts extremes (brighter/less transparent only at high |value|)
GAMMA_COLOR = 0.3 # >1 boosts extremes (less color for background)
ALPHA_MIN = 0.0          # most transparent
ALPHA_MAX = 1.0          # most opaque

# Render settings
USE_CYCLES = True
SAMPLES = 128
RENDER_PATH = "results/field_slices.png"
BLEND_PATH = "blender_files/field_slices_scene.blend"

slice_centers = [(0,0,0), (0,0,0), (0,0,0)]
slice_angles = [(0,0,0), (90,0,0), (0,90,0)]

def main():
    clear_scene()
    ensure_cycles(USE_CYCLES, SAMPLES)

    field_s = load_field(NPY_PATH, AXIS_ORDER, FIELD_COMPONENT)
    nx, ny, nz = field_s.shape
    offx, offy, offz = 0, 0, 0

    # prepare slices
    prepare_slices(slice_centers, slice_angles, field_s, USE_PERCENTILE_RANGE, PCT_LOW, PCT_HIGH, FIELD_COMPONENT, GAMMA_TRANSPARENCY, GAMMA_COLOR, ALPHA_MIN, ALPHA_MAX, CENTER, VOXEL_SIZE, EMISSION_STRENGTH, CMAP=CMAP)

    # # Create a hidden "bounds" cube to frame the whole volume
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0,0,0))
    bounds = bpy.context.object
    bounds.name = "BoundsVolume"
    bounds.scale = (nx * VOXEL_SIZE, ny * VOXEL_SIZE, nz * VOXEL_SIZE)

    dx,dy,dz = nx*VOXEL_SIZE, ny*VOXEL_SIZE, nz*VOXEL_SIZE
    print("expected r:", 0.5 * (dx*dx + dy*dy + dz*dz)**0.5)

    bpy.context.view_layer.update()

    make_camera_autoframe(bounds)
    # make_camera_light_autoframe(bounds)

    bounds.hide_render = True
    bounds.hide_viewport = True

    # Transparent background
    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    scene.view_settings.exposure = 0.0
    scene.view_settings.look = 'None'

    # Optional: set resolution
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100

    # render
    print("start rendering")
    scene.view_settings.view_transform = "Khronos PBR Neutral"
    bpy.context.scene.render.filepath = RENDER_PATH
    bpy.ops.render.render(write_still=True)
    print("Rendered to:", RENDER_PATH)

    # Save .blend
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
    print("Saved blend file to:", BLEND_PATH)

    # remove the png files stored:
    for f in glob.glob("*.png"):
        os.remove(f)

if __name__ == "__main__":
    main()
