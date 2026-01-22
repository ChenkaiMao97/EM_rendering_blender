import bpy
import bmesh
import numpy as np
import math
from mathutils import Vector

from utils.util import clear_scene, ensure_cycles, make_camera_light_autoframe, load_eps, assign_material
from utils.geometry import prepare_mesh_from_voxel, assign_eps_face_attribute_from_volume
from utils.material import make_material
from utils.image import prepare_slices

NPY_PATH = "data/random1/eps.npy"   # <-- change
AXIS_ORDER = "XYZ"  # meaning eps[z,y,x]. Use "XYZ" if eps[x,y,z].
EPS_THRESH = 1.05   # eps <= thresh -> air (empty)
VOXEL_SIZE = 0.02   # meters-ish; adjust scale
CENTER = True

# If True, use percentiles to avoid extreme outliers dominating the range
USE_PERCENTILE_RANGE = True
PCT_LOW = 1
PCT_HIGH = 99

# Appearance
MAKE_GLASSY = True
MATERIAL_TRANSMISSION_WEIGHT = 1.0 if MAKE_GLASSY else 0.0
MATERIAL_ROUGHNESS = 0.5
MATERIAL_METALLIC = 1.0
MATERIAL_IOR = 1.45
ALPHA = 1.0

# Optional smoothing
ADD_REMESH = True
REMESH_VOXEL_SIZE = 0.03
ADD_SMOOTH = False
SMOOTH_ITERS = 3

GAMMA_TRANSPARENCY = 1.0
GAMMA_COLOR = 4.0
ALPHA_MIN = 0.0
ALPHA_MAX = 1.0
EMISSION_STRENGTH = 1.5
CMAP = 'binary'

# Render settings
USE_CYCLES = True
SAMPLES = 128
RENDER_PATH = "results/voxel_render.png"
BLEND_PATH = "blender_files/epsilon_scene.blend"

slice_centers = [(0,0,0), (0,0,0), (0,0,0)]
slice_angles = [(0,0,0), (90,0,0), (0,90,0)]

def main():
    clear_scene()
    ensure_cycles(USE_CYCLES, SAMPLES)

    eps_xyz, occ, eps_min, eps_max = load_eps(NPY_PATH, AXIS_ORDER, EPS_THRESH)
    nx, ny, nz = eps_xyz.shape
    
    prepare_slices(slice_centers, slice_angles, eps_xyz, USE_PERCENTILE_RANGE, PCT_LOW, PCT_HIGH, 'abs', GAMMA_TRANSPARENCY, GAMMA_COLOR, ALPHA_MIN, ALPHA_MAX, CENTER, VOXEL_SIZE, EMISSION_STRENGTH, CMAP=CMAP)

    # # Create a hidden "bounds" cube to frame the whole volume
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0,0,0))
    bounds = bpy.context.object
    bounds.name = "BoundsVolume"
    bounds.scale = (nx * VOXEL_SIZE, ny * VOXEL_SIZE, nz * VOXEL_SIZE)

    dx,dy,dz = nx*VOXEL_SIZE, ny*VOXEL_SIZE, nz*VOXEL_SIZE
    print("expected r:", 0.5 * (dx*dx + dy*dy + dz*dz)**0.5)

    bpy.context.view_layer.update()

    make_camera_light_autoframe(bounds)

    bounds.hide_render = True
    bounds.hide_viewport = True

    scene = bpy.context.scene
    scene.render.film_transparent = True

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    # Render
    print("start rendering")
    bpy.context.scene.render.filepath = RENDER_PATH
    bpy.ops.render.render(write_still=True)
    print("Rendered to:", RENDER_PATH)

    # save file
    print("saving file..")
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
    print("Saved blend file to:", BLEND_PATH)

if __name__ == "__main__":
    main()
