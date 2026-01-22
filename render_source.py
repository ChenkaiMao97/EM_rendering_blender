import bpy
import bmesh
import numpy as np
import math
from mathutils import Vector

from utils.util import clear_scene, ensure_cycles, make_camera_light_autoframe, load_source, assign_material
from utils.geometry import prepare_mesh_from_voxel, assign_eps_face_attribute_from_volume
from utils.material import make_material

NPY_PATH = "data/ring_resonator/src.npy"   # <-- change
AXIS_ORDER = "XYZ"  # meaning eps[z,y,x]. Use "XYZ" if eps[x,y,z].
SRC_THRESH = 0.0   # eps <= thresh -> air (empty)
VOXEL_SIZE = 0.02   # meters-ish; adjust scale
CENTER = True

# Appearance
MAKE_GLASSY = True
MATERIAL_TRANSMISSION_WEIGHT = 1.0 if MAKE_GLASSY else 0.0
MATERIAL_ROUGHNESS = 0.8
MATERIAL_METALLIC = 0.8
MATERIAL_IOR = 1.45
ALPHA = 1.0

# Optional smoothing
ADD_REMESH = True
REMESH_VOXEL_SIZE = 0.03
ADD_SMOOTH = True
SMOOTH_ITERS = 3

# Render settings
USE_CYCLES = True
SAMPLES = 128
RENDER_PATH = "results/voxel_render.png"
BLEND_PATH = "blender_files/source_scene.blend"

def main():
    clear_scene()
    ensure_cycles(USE_CYCLES, SAMPLES)

    src_xyz, occ, src_min, src_max = load_source(NPY_PATH, AXIS_ORDER, SRC_THRESH)
    obj = prepare_mesh_from_voxel(occ, src_xyz, VOXEL_SIZE, CENTER, ADD_REMESH, REMESH_VOXEL_SIZE, SMOOTH_ITERS, ADD_SMOOTH)
    bbox = [Vector(i) for i in obj.bound_box]
    print("bbox:", bbox)

    # after remesh and smoothing, assign the mesh attribute based on original src values
    # assign_src_face_attribute_from_volume(obj, src_xyz=src_xyz, voxel_size=VOXEL_SIZE, centered=CENTER, attr_name="src", sample="local_max")

    # assign material
    color1 = (1,0,0,1)
    color2 = (1,0,0,1)
    mat = make_material(MATERIAL_ROUGHNESS, MATERIAL_METALLIC, MATERIAL_IOR, MATERIAL_TRANSMISSION_WEIGHT, attr_name="src", eps_min=src_min, eps_max=src_max, ALPHA=ALPHA, color1=color1, color2=color2)
    assign_material(obj, mat)
    
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
