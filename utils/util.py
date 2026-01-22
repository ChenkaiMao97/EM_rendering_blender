import bpy
import bmesh
import numpy as np
import math
from mathutils import Vector
import matplotlib.pyplot as plt

def bbox_world(obj):
    # 8 corners in world space
    return [obj.matrix_world @ Vector(c) for c in obj.bound_box]

def bbox_center_and_radius(obj):
    pts = bbox_world(obj)
    c = sum(pts, Vector((0,0,0))) / 8.0
    r = max((p - c).length for p in pts)
    return c, r

def look_at(obj, target, track='-Z', up='Y'):
    direction = target - obj.location
    if direction.length == 0:
        return
    quat = direction.to_track_quat(track, up)
    obj.rotation_euler = quat.to_euler()

def set_node_input(node, names, value):
    """Try multiple socket names; set the first that exists."""
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Purge orphan data blocks
    for _ in range(3):
        bpy.ops.outliner.orphans_purge(do_recursive=True)

def force_evaluate_scene():
    # Ensure depsgraph is up-to-date
    bpy.context.view_layer.update()
    # Evaluate objects once via depsgraph (important for GN/modifiers/constraints)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    # Optional: if animation/drivers exist, make sure you're on the intended frame
    bpy.context.scene.frame_set(bpy.context.scene.frame_current)

def ensure_cycles(USE_CYCLES, SAMPLES):
    if USE_CYCLES:
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = SAMPLES
        # Apple Silicon: try Metal
        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences if 'cycles' in prefs.addons else None
        if cprefs:
            try:
                cprefs.compute_device_type = 'METAL'
                bpy.context.scene.cycles.device = 'GPU'
            except Exception:
                bpy.context.scene.cycles.device = 'CPU'

def make_camera_autoframe(target_obj, margin=1.25):
    c, r = bbox_center_and_radius(target_obj)
    r = max(r, 1e-6)

    cam_dist = r * 3.0 * margin
    cam_loc = c + Vector((1, -2, 2.0)).normalized() * cam_dist

    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    bpy.context.scene.camera = cam

    cam.data.clip_start = max(0.001, r * 0.01)
    cam.data.clip_end = r * 100.0
    cam.data.lens_unit = 'FOV'
    cam.data.angle = math.radians(65)
    look_at(cam, c)

    # World background (dim, but you can keep film transparent anyway)
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.02, 0.02, 0.02, 1.0)
        bg.inputs[1].default_value = 1.0


def make_camera_light_autoframe(target_obj, margin=1.25):
    c, r = bbox_center_and_radius(target_obj)
    print("c:", c, "r:", r)

    r = max(r, 1e-6)

    # --- Camera ---
    # Put camera on a nice diagonal view
    cam_dist = r * 3.0 * margin
    cam_loc = c + Vector((1, -2, 2.0)).normalized() * cam_dist

    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    bpy.context.scene.camera = cam

    # Set clipping relative to size
    cam.data.clip_start = max(0.001, r * 0.01)
    cam.data.clip_end = r * 100.0

    # fov
    cam.data.lens_unit = 'FOV'
    cam.data.angle = math.radians(65)

    look_at(cam, c)

    # --- Lights (3-point-ish) ---
    # Key: Area light
    key_dist = r * 6
    key_loc = c + Vector((1, 0, 1.0)).normalized() * key_dist
    bpy.ops.object.light_add(type='AREA', location=key_loc)
    key = bpy.context.object
    # key.data.shape = "RECTANGLE"
    key.data.size = r
    key.data.energy = 5e3 * (r*r)  # crude scaling; adjust if too bright/dim
    look_at(key, c)

    # Fill: Point
    fill_dist = r * 6.0
    fill_loc = c + Vector((-1, -1, 1)).normalized() * fill_dist
    bpy.ops.object.light_add(type='AREA', location=fill_loc)
    fill = bpy.context.object
    fill.data.size = 2*r
    fill.data.energy = 4e3 * (r*r)
    look_at(fill, c)

    # Rim/back: Area
    rim_dist = r * 7.5
    rim_loc = c + Vector((-1.0, 1.0, 3.0)).normalized() * rim_dist
    bpy.ops.object.light_add(type='AREA', location=rim_loc)
    rim = bpy.context.object
    rim.data.size = 1.0*r
    rim.data.energy = 50 * (r*r)
    look_at(rim, c)

    # World background
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        print("background light")
        bg.inputs[0].default_value = (0.02, 0.02, 0.02, 1.0)
        bg.inputs[1].default_value = 1.0


def load_eps(NPY_PATH_EPS, AXIS_ORDER, EPS_THRESH):
    eps = np.load(NPY_PATH_EPS).squeeze()
    if len(eps.shape) == 4:
        eps = eps[..., 0]

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

    return eps_xyz, occ, eps_min, eps_max

def load_source(NPY_PATH_SRC, AXIS_ORDER, SRC_THRESH):
    src = np.abs(np.load(NPY_PATH_SRC).squeeze())
    if len(src.shape) == 4:
        src = np.sum(src, axis=-1)

    # Reorder to x,y,z for the mesher
    if AXIS_ORDER == "ZYX":
        # src[z,y,x] -> occ[x,y,z]
        src_xyz = np.transpose(src, (2, 1, 0))
    elif AXIS_ORDER == "XYZ":
        src_xyz = src
    else:
        raise ValueError("AXIS_ORDER must be 'ZYX' or 'XYZ' (extend if needed).")

    occ = (np.abs(src_xyz) > SRC_THRESH)

    src_vals = src_xyz[occ]
    src_min = np.min(src_vals)
    src_max = np.max(src_vals)

    if not np.any(occ):
        raise RuntimeError("No material voxels found (all src <= src_THRESH). Try lowering src_THRESH.")

    return src_xyz, occ, src_min, src_max

def load_field(NPY_PATH, AXIS_ORDER, FIELD_COMPONENT):
    field_c = np.load(NPY_PATH).squeeze()
    if len(field_c.shape) == 4:
        field_c = field_c[..., -1]  # you had this; keep if your array has trailing channel

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

    return field_s

def assign_material(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)