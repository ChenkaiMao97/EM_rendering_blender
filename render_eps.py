import bpy
import bmesh
import numpy as np
import math
from mathutils import Vector

from utils.util import clear_scene, ensure_cycles, make_camera_light_autoframe
from utils.geometry import naive_voxel_surface_quads, build_mesh_from_quads_with_eps
from utils.material import make_material

NPY_PATH = "data/eps.npy"   # <-- change
AXIS_ORDER = "XYZ"  # meaning eps[z,y,x]. Use "XYZ" if eps[x,y,z].
EPS_THRESH = 1.05   # eps <= thresh -> air (empty)
VOXEL_SIZE = 0.02   # meters-ish; adjust scale
CENTER = True

# Appearance
MAKE_GLASSY = True
MATERIAL_TRANSMISSION_WEIGHT = 1.0 if MAKE_GLASSY else 0.0
MATERIAL_ROUGHNESS = 0.7
MATERIAL_METALLIC = 0.3
MATERIAL_IOR = 1.45

# Optional smoothing
ADD_REMESH = True
REMESH_VOXEL_SIZE = 0.03
ADD_SMOOTH = True
SMOOTH_ITERS = 3

# Render settings
USE_CYCLES = True
SAMPLES = 128
RENDER_PATH = "results/voxel_render.png"
BLEND_PATH = "blender_files/epsilon_scene.blend"

def main():
    clear_scene()
    ensure_cycles(USE_CYCLES, SAMPLES)

    eps = np.load(NPY_PATH).squeeze()

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

    # Optional smoothing to reduce “blocky voxel” feel
    if ADD_REMESH:
        orig_scale = obj.scale.copy()

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        obj.scale.z *= 100
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        rem = obj.modifiers.new("Remesh", type='REMESH')
        rem.mode = 'VOXEL'
        rem.voxel_size = REMESH_VOXEL_SIZE
        rem.use_smooth_shade = True
        bpy.ops.object.modifier_apply(modifier=rem.name)

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
