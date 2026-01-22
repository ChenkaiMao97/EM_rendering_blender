import bpy
import bmesh
import numpy as np
import math
import glob
import os
from mathutils import Vector
import matplotlib.pyplot as plt

from utils.util import clear_scene, ensure_cycles, make_camera_light_autoframe, load_eps, load_field, assign_material, force_evaluate_scene
from utils.image import cmap_rgba, slice_to_rgba, ensure_image, write_image_pixels, add_plane, prepare_slices
from utils.geometry import prepare_mesh_from_voxel, assign_eps_face_attribute_from_volume
from utils.material import make_material, make_emissive_image_material


# NPY_PATH_EPS = "data/eps.npy"
# NPY_PATH = "data/solution.npy"   # complex 3D array

# NPY_PATH_EPS = "data/wgb_eps.npy"
# NPY_PATH = "data/wgb_ez.npy"   # complex 3D array

NPY_PATH_EPS = "data/superpixel_eps.npy"
NPY_PATH = "data/superpixel_ez.npy"

AXIS_ORDER = "XYZ"  # meaning eps[z,y,x]. Use "XYZ" if eps[x,y,z].
EPS_AIR = 1.0
EPS_THRESH = 1.05   # eps <= thresh -> air (empty)
VOXEL_SIZE = 0.02   # meters-ish; adjust scale
CENTER = True

# Appearance
MAKE_GLASSY = True
MATERIAL_TRANSMISSION_WEIGHT = 0.1 if MAKE_GLASSY else 0.0
MATERIAL_ROUGHNESS = 0.4
MATERIAL_METALLIC = 0.6
MATERIAL_IOR = 4.0

# Optional smoothing
ADD_REMESH = True
REMESH_VOXEL_SIZE = 0.03
ADD_SMOOTH = True
SMOOTH_ITERS = 3

# Render settings
USE_CYCLES = True
SAMPLES = 128
RENDER_PATH = "results/render_all_slices.png"
BLEND_PATH = "blender_files/scene_all.blend"

# Which scalar to visualize from the complex field:
#   "abs" | "real" | "imag"
FIELD_COMPONENT = "real"

# Slice indices (None -> use middle slice)
SLICE_X = None  # index along x (plane is YZ)
SLICE_Y = None  # index along y (plane is XZ)
SLICE_Z = None  # index along z (plane is XY)

# If True, use percentiles to avoid extreme outliers dominating the range
USE_PERCENTILE_RANGE = False
PCT_LOW = 1
PCT_HIGH = 99

# Emission/alpha shaping
EMISSION_STRENGTH = 20.0   # global multiplier (Cycles: this matters a lot)
GAMMA_TRANSPARENCY = 1.0 # >1 boosts extremes (brighter/less transparent only at high |value|)
GAMMA_COLOR = 1.0 # >1 boosts extremes (less color for background)
ALPHA_MIN = 0.0          # most transparent
ALPHA_MAX = 1.0          # most opaque

# slice_centers = [(0,0,0), (0,0,0), (0,0,0)]
# slice_angles = [(0,0,0), (0,90,90), (0,90,0)]

slice_centers = [(0,0,0), (0,0,0)]
slice_angles = [(0,0,0), (0,60,240)]

def main():
    clear_scene()
    ensure_cycles(USE_CYCLES, SAMPLES)

    # load eps and prepare mesh object
    eps_xyz, occ, eps_min, eps_max = load_eps(NPY_PATH_EPS, AXIS_ORDER, EPS_THRESH)
    obj = prepare_mesh_from_voxel(occ, eps_xyz, VOXEL_SIZE, CENTER, ADD_REMESH, REMESH_VOXEL_SIZE, SMOOTH_ITERS, ADD_SMOOTH)
    bbox = [Vector(i) for i in obj.bound_box]

    # after remesh and smoothing, assign the mesh attribute based on original eps values
    assign_eps_face_attribute_from_volume(obj, eps_xyz=eps_xyz, voxel_size=VOXEL_SIZE, centered=CENTER, attr_name="eps", sample="local_max", radius=1)

    # assign material
    mat = make_material(MATERIAL_ROUGHNESS, MATERIAL_METALLIC, MATERIAL_IOR, MATERIAL_TRANSMISSION_WEIGHT, attr_name="eps", eps_min=eps_min, eps_max=eps_max)
    assign_material(obj, mat)

    # add field
    field_s = load_field(NPY_PATH, AXIS_ORDER, FIELD_COMPONENT)

    # prepare slices
    prepare_slices(slice_centers, slice_angles, field_s, USE_PERCENTILE_RANGE, PCT_LOW, PCT_HIGH, FIELD_COMPONENT, GAMMA_TRANSPARENCY, GAMMA_COLOR, ALPHA_MIN, ALPHA_MAX, CENTER, VOXEL_SIZE, EMISSION_STRENGTH)

    # # Create a hidden "bounds" cube to frame the whole volume
    # bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0,0,0))
    # bounds = bpy.context.object
    # bounds.name = "BoundsVolume"
    # bounds.scale = (0.5 * nx * VOXEL_SIZE, 0.5 * ny * VOXEL_SIZE, 0.5 * nz * VOXEL_SIZE)
    # bounds.location = (offx + (nx * VOXEL_SIZE) * 0.5, offy + (ny * VOXEL_SIZE) * 0.5, offz + (nz * VOXEL_SIZE) * 0.5)
    # bounds.hide_render = True
    # bounds.hide_viewport = True

    make_camera_light_autoframe(obj)

    force_evaluate_scene()
    # Render
    print("start rendering")
    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    # scene.view_settings.view_transform = "Khronos PBR Neutral"
    # scene.view_settings.look = "Very High Contrast"
    bpy.context.scene.render.filepath = RENDER_PATH
    bpy.ops.render.render(write_still=True)

    # save file
    print("saving file..")
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
    print("Saved blend file to:", BLEND_PATH)

    # remove the png files stored:
    for f in glob.glob("*.png"):
        os.remove(f)

if __name__ == "__main__":
    main()
