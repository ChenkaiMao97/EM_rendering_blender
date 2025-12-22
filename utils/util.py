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
    cam_loc = c + Vector((1.2, -1.8, 1.1)).normalized() * cam_dist

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

    r = max(r, 1e-6)

    # --- Camera ---
    # Put camera on a nice diagonal view
    cam_dist = r * 3.0 * margin
    cam_loc = c + Vector((1.2, -1.8, 3)).normalized() * cam_dist

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
    key_dist = r * 2
    key_loc = c + Vector((1, 0, 1.0)).normalized() * key_dist
    bpy.ops.object.light_add(type='AREA', location=key_loc)
    key = bpy.context.object
    # key.data.shape = "RECTANGLE"
    key.data.size = r
    key.data.energy = 5e2 * (r*r)  # crude scaling; adjust if too bright/dim
    look_at(key, c)

    # Fill: Point
    fill_dist = r * 2.0
    fill_loc = c + Vector((-1, -1, 1)).normalized() * fill_dist
    bpy.ops.object.light_add(type='AREA', location=fill_loc)
    fill = bpy.context.object
    fill.data.size = 2*r
    fill.data.energy = 400 * (r*r)
    look_at(fill, c)

    # Rim/back: Area
    rim_dist = r * 2.5
    rim_loc = c + Vector((-1.0, 1.0, 3.0)).normalized() * rim_dist
    bpy.ops.object.light_add(type='AREA', location=rim_loc)
    rim = bpy.context.object
    rim.data.size = 1.0*r
    rim.data.energy = 5 * (r*r)
    look_at(rim, c)

    # World background
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        print("background light")
        bg.inputs[0].default_value = (0.02, 0.02, 0.02, 1.0)
        bg.inputs[1].default_value = 1.0