import bpy
import numpy as np
import math
from mathutils import Vector

from utils.util import clear_scene, ensure_cycles, make_camera_light_autoframe
from utils.material import make_field_emission_material
from utils.geometry import marching_cubes, build_mesh_from_mc
from utils.image import set_colorramp_seismic

NPY_PATH = "data/solution.npy"   # complex 3D array
AXIS_ORDER = "XYZ"               # "XYZ" means arr[x,y,z], "ZYX" means arr[z,y,x]
VOXEL_SIZE = 0.02
CENTER = True

#   "abs" | "real" | "imag"
FIELD_COMPONENT = "real"

# Isosurface settings
N_ISO = 7

# If True, use percentiles to avoid extreme outliers dominating the range
USE_PERCENTILE_RANGE = True
PCT_LOW = 1.0
PCT_HIGH = 99.0

# If True, skip very small/empty shells automatically
SKIP_EMPTY_LEVELS = True

# Appearance
MAKE_GLASSY = False
MATERIAL_TRANSMISSION_WEIGHT = 1.0 if MAKE_GLASSY else 0.0
MATERIAL_ROUGHNESS = 1.0
MATERIAL_IOR = 1.45

# Optional cheap smoothing (much cheaper than voxel remesh)
ADD_SMOOTH = False
SMOOTH_ITERS = 10

# Render settings
USE_CYCLES = True
SAMPLES = 128
RENDER_PATH = "results/field_render.png"
BLEND_PATH = "blender_files/field_scene.blend"

def main():
    clear_scene()
    ensure_cycles(USE_CYCLES, SAMPLES)

    field_c = np.load(NPY_PATH).squeeze()  # complex

    if len(field_c.shape) == 4:
        field_c = field_c[..., 0]

    # Reorder to x,y,z
    if AXIS_ORDER == "ZYX":
        field_c_xyz = np.transpose(field_c, (2, 1, 0))
    elif AXIS_ORDER == "XYZ":
        field_c_xyz = field_c
    else:
        raise ValueError("AXIS_ORDER must be 'ZYX' or 'XYZ'.")

    # Choose scalar component
    if FIELD_COMPONENT == "abs":
        field_s = np.abs(field_c_xyz)
    elif FIELD_COMPONENT == "real":
        field_s = np.real(field_c_xyz)
    elif FIELD_COMPONENT == "imag":
        field_s = np.imag(field_c_xyz)
    else:
        raise ValueError("FIELD_COMPONENT must be 'abs', 'real', or 'imag'.")

    # Range for isosurfaces
    if USE_PERCENTILE_RANGE:
        vmin = float(np.percentile(field_s, PCT_LOW))
        vmax = float(np.percentile(field_s, PCT_HIGH))
    else:
        vmin = float(np.min(field_s))
        vmax = float(np.max(field_s))
    
    if FIELD_COMPONENT == "real" or FIELD_COMPONENT == "imag":
        vm = max(abs(vmin), abs(vmax))
        vmax = vm
        vmin = -vm

    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax <= vmin):
        raise RuntimeError(f"Bad field range: vmin={vmin}, vmax={vmax}")

    # Levels (avoid exact endpoints)
    levels = np.linspace(vmin, vmax, N_ISO + 2)[1:-1]

    mat = make_field_emission_material(
        FIELD_COMPONENT,
        attr_name="iso", 
        vmin=vmin, vmax=vmax,
        ALPHA_MIN=0.7,
        ALPHA_MAX=0.99,
        GAMMA=2.0
    )

    shell_objs = []
    nx, ny, nz = field_s.shape

    for i, lv in enumerate(levels):
        print(f"[MC] level {i+1}/{len(levels)} = {lv}")

        # Optional skip if level is outside actual min/max due to percentiles
        if lv <= float(np.min(field_s)) or lv >= float(np.max(field_s)):
            continue

        # Quick emptiness check (cheap)
        if SKIP_EMPTY_LEVELS:
            if not np.any(field_s >= lv):
                continue
            if np.all(field_s >= lv):
                continue

        verts, faces = marching_cubes(field_s.astype(np.float32), lv)
        if faces is None or len(faces) == 0:
            continue

        obj = build_mesh_from_mc(verts, faces, name=f"FieldIso_{i:02d}", iso_value=lv, attr_name="iso")

        # Scale and center like eps script
        obj.scale = (VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
        if CENTER:
            obj.location = (-0.5 * nx * VOXEL_SIZE, -0.5 * ny * VOXEL_SIZE, -0.5 * nz * VOXEL_SIZE)

        # Material
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        # Optional cheap smoothing
        if ADD_SMOOTH:
            sm = obj.modifiers.new("Smooth", type='SMOOTH')
            sm.iterations = SMOOTH_ITERS

        shell_objs.append(obj)

    if len(shell_objs) == 0:
        raise RuntimeError("No isosurfaces generated. Try lowering N_ISO or adjust percentile range.")

    # Autoframe based on one shell
    make_camera_light_autoframe(shell_objs[-1])

    # Transparent background
    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    scene.view_settings.exposure = 0.0     # try -1..+2 depending on taste
    scene.view_settings.look = 'None' 

    # Save blend (recommended workflow)
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
    print("Saved blend file to:", BLEND_PATH)

if __name__ == "__main__":
    main()
