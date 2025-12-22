import bpy
import bmesh
import numpy as np
import math
from mathutils import Vector
import glob
import os

from utils.util import clear_scene, ensure_cycles, make_camera_autoframe
from utils.image import cmap_rgba, slice_to_rgba, ensure_image, write_image_pixels, add_plane
from utils.geometry import naive_voxel_surface_quads, build_mesh_from_quads_with_eps
from utils.material import make_emissive_image_material

NPY_PATH = "data/wgb_ez.npy"   # complex 3D array
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
USE_PERCENTILE_RANGE = False
PCT_LOW = 1
PCT_HIGH = 99

# Emission/alpha shaping
EMISSION_STRENGTH = 3.0   # global multiplier (Cycles: this matters a lot)
GAMMA_TRANSPARENCY = 2.0 # >1 boosts extremes (brighter/less transparent only at high |value|)
GAMMA_COLOR = 0.2 # >1 boosts extremes (less color for background)
ALPHA_MIN = 0.0          # most transparent
ALPHA_MAX = 1.0          # most opaque

# Render settings
USE_CYCLES = True
SAMPLES = 128
RENDER_PATH = "results/field_slices.png"
BLEND_PATH = "blender_files/field_slices_scene.blend"

def main():
    clear_scene()
    ensure_cycles(USE_CYCLES, SAMPLES)

    field_c = np.load(NPY_PATH).squeeze()
    if len(field_c.shape) == 4:
        field_c = field_c[..., 0]  # you had this; keep if your array has trailing channel

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

    nx, ny, nz = field_s.shape

    # Choose slice indices
    ix = (nx // 2) if (SLICE_X is None) else int(SLICE_X)
    iy = (ny // 2) if (SLICE_Y is None) else int(SLICE_Y)
    iz = (nz // 2) if (SLICE_Z is None) else int(SLICE_Z)
    ix = max(0, min(nx - 1, ix))
    iy = max(0, min(ny - 1, iy))
    iz = max(0, min(nz - 1, iz))

    # Range for colormap
    if USE_PERCENTILE_RANGE:
        vmin = float(np.percentile(field_s, PCT_LOW))
        vmax = float(np.percentile(field_s, PCT_HIGH))
    else:
        vmin = float(np.min(field_s))
        vmax = float(np.max(field_s))

    if FIELD_COMPONENT in ("real", "imag"):
        vm = max(abs(vmin), abs(vmax))
        vmax = vm
        vmin = -vm

    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax <= vmin and FIELD_COMPONENT == "abs"):
        raise RuntimeError(f"Bad field range: vmin={vmin}, vmax={vmax}")

    slice_x = field_s[ix, :, :]          # (ny, nz)
    slice_y = field_s[:, iy, :]          # (nx, nz)
    slice_z = field_s[:, :, iz]          # (nx, ny)

    # Convert to RGBA images
    rgba_x = slice_to_rgba(slice_x, vmin, vmax, FIELD_COMPONENT, GAMMA_TRANSPARENCY, GAMMA_COLOR, ALPHA_MIN, ALPHA_MAX)  # (ny,nz,4)
    rgba_y = slice_to_rgba(slice_y, vmin, vmax, FIELD_COMPONENT, GAMMA_TRANSPARENCY, GAMMA_COLOR, ALPHA_MIN, ALPHA_MAX)  # (nx,nz,4)
    rgba_z = slice_to_rgba(slice_z, vmin, vmax, FIELD_COMPONENT, GAMMA_TRANSPARENCY, GAMMA_COLOR, ALPHA_MIN, ALPHA_MAX)  # (nx,ny,4)

    # Create Blender images (note: Blender image size is (W,H))
    img_x = ensure_image("slice_x_YZ", w=ny, h=nz)
    img_y = ensure_image("slice_y_XZ", w=nx, h=nz)
    img_z = ensure_image("slice_z_XY", w=nx, h=ny)

    write_image_pixels(img_x, rgba_x, 'rgba_x')  # (H,W,4) = (ny,nz,4)
    write_image_pixels(img_y, rgba_y, 'rgba_y')  # (nx,nz,4)
    write_image_pixels(img_z, rgba_z, 'rgba_z')  # (nx,ny,4)

    # Materials
    mat_x = make_emissive_image_material("MAT_slice_x", img_x, emission_strength=EMISSION_STRENGTH)
    mat_y = make_emissive_image_material("MAT_slice_y", img_y, emission_strength=EMISSION_STRENGTH)
    mat_z = make_emissive_image_material("MAT_slice_z", img_z, emission_strength=EMISSION_STRENGTH)

    offx = (-0.5 * nx * VOXEL_SIZE) if CENTER else 0.0
    offy = (-0.5 * ny * VOXEL_SIZE) if CENTER else 0.0
    offz = (-0.5 * nz * VOXEL_SIZE) if CENTER else 0.0

    plane_z = add_plane("Slice_Z_XY", size_x=nx * VOXEL_SIZE, size_y=ny * VOXEL_SIZE)
    plane_z.location = (offx + (nx * VOXEL_SIZE) * 0.5, offy + (ny * VOXEL_SIZE) * 0.5, offz + iz * VOXEL_SIZE)
    
    if plane_z.data.materials:
        plane_z.data.materials[0] = mat_z
    else:
        plane_z.data.materials.append(mat_z)

    plane_x = add_plane("Slice_X_YZ", size_x=ny * VOXEL_SIZE, size_y=nz * VOXEL_SIZE)
    plane_x.rotation_euler = (math.radians(90.0), 0.0, math.radians(90.0))
    plane_x.location = (offx + ix * VOXEL_SIZE, offy + (ny * VOXEL_SIZE) * 0.5, offz + (nz * VOXEL_SIZE) * 0.5)

    if plane_x.data.materials:
        plane_x.data.materials[0] = mat_x
    else:
        plane_x.data.materials.append(mat_x)

    plane_y = add_plane("Slice_Y_XZ", size_x=nx * VOXEL_SIZE, size_y=nz * VOXEL_SIZE)
    plane_y.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    plane_y.location = (offx + (nx * VOXEL_SIZE) * 0.5, offy + iy * VOXEL_SIZE, offz + (nz * VOXEL_SIZE) * 0.5)

    if plane_y.data.materials:
        plane_y.data.materials[0] = mat_y
    else:
        plane_y.data.materials.append(mat_y)

    # Create a hidden "bounds" cube to frame the whole volume
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0,0,0))
    bounds = bpy.context.object
    bounds.name = "BoundsVolume"
    bounds.scale = (0.5 * nx * VOXEL_SIZE, 0.5 * ny * VOXEL_SIZE, 0.5 * nz * VOXEL_SIZE)
    bounds.location = (offx + (nx * VOXEL_SIZE) * 0.5, offy + (ny * VOXEL_SIZE) * 0.5, offz + (nz * VOXEL_SIZE) * 0.5)
    bounds.hide_render = True
    bounds.hide_viewport = True

    make_camera_autoframe(bounds)

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
    bpy.context.scene.render.filepath = RENDER_PATH
    bpy.ops.render.render(write_still=True)
    print("Rendered to:", RENDER_PATH)

    # Save .blend
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
    print("Saved blend file to:", BLEND_PATH)
    print(f"Slices: ix={ix}, iy={iy}, iz={iz}")
    print(f"Range: vmin={vmin}, vmax={vmax}")

    # remove the png files stored:
    for f in glob.glob("*.png"):
        os.remove(f)

if __name__ == "__main__":
    main()
