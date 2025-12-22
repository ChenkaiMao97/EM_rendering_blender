import bpy
import bmesh
import numpy as np
import math
from mathutils import Vector

from utils.util import clear_scene, ensure_cycles, make_camera_light_autoframe
from utils.material import make_field_emission_material, make_material
from utils.geometry import marching_cubes, build_mesh_from_mc, naive_voxel_surface_quads, build_mesh_from_quads_with_eps
from utils.image import set_colorramp_seismic

# ----------------------------
# USER SETTINGS
# ----------------------------
NPY_PATH_EPS = "data/wgb_eps.npy"   # <-- change
NPY_PATH_FIELD = "data/wgb_ez.npy"
AXIS_ORDER = "XYZ"  # meaning eps[z,y,x]. Use "XYZ" if eps[x,y,z].
EPS_AIR = 1.0
EPS_THRESH = 1.05   # eps <= thresh -> air (empty)
VOXEL_SIZE = 0.02   # meters-ish; adjust scale
CENTER = True

# Appearance
MAKE_GLASSY = True
MATERIAL_TRANSMISSION_WEIGHT = 1.0 if MAKE_GLASSY else 0.0
# MATERIAL_BASE_COLOR = (0.75, 0.85, 1.0, 1.0)
MATERIAL_ROUGHNESS = 0.85
MATERIAL_METALLIC = 0.3
MATERIAL_IOR = 1.75

# Optional smoothing
ADD_REMESH = True
REMESH_VOXEL_SIZE = 0.03
ADD_SMOOTH = True
SMOOTH_ITERS = 3

# Render settings
USE_CYCLES = False
SAMPLES = 128
RENDER_PATH = "results/render_all.png"
BLEND_PATH = "blender_files/scene_volumetric.blend"

SKIP_EMPTY_LEVELS = True

USE_PERCENTILE_RANGE = True
PCT_LOW = 1.0
PCT_HIGH = 99.0

FIELD_COMPONENT = "real"
N_ISO = 15

def main():
    clear_scene()
    ensure_cycles(USE_CYCLES, SAMPLES)

    eps = np.load(NPY_PATH_EPS).squeeze()

    # Reorder to x,y,z for the mesher
    if AXIS_ORDER == "ZYX":
        # eps[z,y,x] -> occ[x,y,z]
        eps_xyz = np.transpose(eps, (2, 1, 0))
    elif AXIS_ORDER == "XYZ":
        eps_xyz = eps
    else:
        raise ValueError("AXIS_ORDER must be 'ZYX' or 'XYZ' (extend if needed).")

    occ = (eps_xyz > EPS_THRESH)

    eps_vals = eps_xyz[occ]
    eps_min = np.min(eps_vals)
    eps_max = np.max(eps_vals)

    if not np.any(occ):
        raise RuntimeError("No material voxels found (all eps <= EPS_THRESH). Try lowering EPS_THRESH.")

    print("started naive mesh")
    quads = naive_voxel_surface_quads(occ, eps_xyz)
    print("finished meshing")
    obj = build_mesh_from_quads_with_eps(quads, name="Dielectric")
    print("finished building mesh")

    obj.scale = (VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)

    if CENTER:
        nx, ny, nz = occ.shape
        obj.location = (-0.5 * nx * VOXEL_SIZE, -0.5 * ny * VOXEL_SIZE, -0.5 * nz * VOXEL_SIZE)

    # Material
    mat = make_material(
        MATERIAL_ROUGHNESS,
        MATERIAL_METALLIC,
        MATERIAL_IOR,
        MATERIAL_TRANSMISSION_WEIGHT,
        attr_name="eps",
        eps_min=eps_min, 
        eps_max=eps_max
    )
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    if ADD_REMESH:
        orig_scale = obj.scale.copy()

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        obj.scale.z *= 100
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # (C) Remesh in squashed space
        rem = obj.modifiers.new("Remesh", type='REMESH')
        rem.mode = 'VOXEL'
        rem.voxel_size = REMESH_VOXEL_SIZE
        rem.use_smooth_shade = True
        bpy.ops.object.modifier_apply(modifier=rem.name)

        # (D) Unsquash X and bake back
        obj.scale.z *= 0.01
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    if ADD_SMOOTH:
        print("adding smooth")
        sm = obj.modifiers.new("Smooth", type='SMOOTH')
        sm.iterations = SMOOTH_ITERS

    make_camera_light_autoframe(obj)

    scene = bpy.context.scene
    scene.render.film_transparent = True

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    # add field
    field_c = np.load(NPY_PATH_FIELD).squeeze()  # complex
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
        ALPHA_MIN=0.85,
        ALPHA_MAX=1.0,
        GAMMA=3.5
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
