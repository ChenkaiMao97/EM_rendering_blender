# slice_render_with_boundaries_and_intersections.py
#
# Features added:
# (1) Per-slice boundary outline (black) between alpha==0 and alpha>0 regions in the generated RGBA image.
# (2) Intersections between slice planes drawn as black 3D lines (curve objects) clipped to the volume AABB.
#
# Assumes you already have:
#   - utils.color.cmaps  (dict: cmap_name -> list[(pos,(r,g,b,a))])
#   - utils.material.make_emissive_image_material(img, emission_strength=...)
#
# Usage:
#   - Call prepare_slices(...) as before.
#   - It will now also (a) outline boundaries in the texture, and (b) create intersection curves.
#
import numpy as np
import math
import torch
import torch.nn.functional as F
import bpy
from mathutils import Vector

from utils.color import cmaps
from utils.material import make_emissive_image_material


# ----------------------------
# Colormap helpers
# ----------------------------
def cmap_rgba(t, cmap="seismic"):
    """
    t: array in [0,1]
    returns rgba array (...,4)
    """
    t = np.clip(t, 0.0, 1.0).astype(np.float32)

    CMAP = cmaps[cmap]
    pos = np.array([p for p, _ in CMAP], dtype=np.float32)
    col = np.array([c for _, c in CMAP], dtype=np.float32)  # (K,4)

    idx = np.searchsorted(pos, t, side="right") - 1
    idx = np.clip(idx, 0, len(pos) - 2)

    p0 = pos[idx]
    p1 = pos[idx + 1]
    w = (t - p0) / np.maximum(p1 - p0, 1e-8)

    c0 = col[idx]
    c1 = col[idx + 1]
    return c0 * (1.0 - w[..., None]) + c1 * (w[..., None])


# ----------------------------
# Blender image IO
# ----------------------------
def ensure_image(name, w, h):
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.new(
            name=name, width=w, height=h, alpha=True, float_buffer=True
        )
    elif img.size[0] != w or img.size[1] != h:
        bpy.data.images.remove(img)
        img = bpy.data.images.new(
            name=name, width=w, height=h, alpha=True, float_buffer=True
        )
    img.colorspace_settings.name = "sRGB"
    return img


def write_image_pixels(img, rgba, name, save_dir="blender_files"):
    """
    rgba: (H, W, 4) float32 in [0,1]
    """
    rgba = np.clip(rgba, 0.0, 1.0).astype(np.float32)

    # IMPORTANT: Blender expects row-major flattened pixels, with X as fastest.
    # Your current code transposed; we keep it consistent with your previous workflow.
    # rgba = rgba.transpose(1, 0, 2)  # (W,H,4) then flatten

    flat = rgba.reshape(-1)

    # foreach_set is typically faster, but direct assignment works too.
    # img.pixels.foreach_set(flat)
    img.pixels = flat

    img.filepath_raw = f"//{name}.png"
    img.file_format = "PNG"

    # Save both in a folder and also locally (as you had)
    try:
        img.save(filepath=f"{save_dir}/{name}.png")
    except Exception as e:
        print(f"[write_image_pixels] Could not save to {save_dir}/{name}.png: {e}")
    try:
        img.save(filepath=f"./{name}.png")
    except Exception as e:
        print(f"[write_image_pixels] Could not save to ./{name}.png: {e}")


# ----------------------------
# Geometry: plane + sampling
# ----------------------------
def add_plane(name, size_x, size_y):
    """
    Adds a plane centered at origin, size_x in local X, size_y in local Y.
    Note: Blender plane primitive size=1 spans local [-0.5..0.5] in X/Y.
    """
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size_x, size_y, 1.0)
    return obj


def sample_plane_slice_from_volume(
    data_np: np.ndarray,
    plane_obj,
    out_h: int,
    out_w: int,
    voxel_size: float = 1.0,
    volume_origin_world=(0.0, 0.0, 0.0),
    axis_order: str = "XYZ",
    mode: str = "bilinear",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    data_np: 3D numpy array. axis_order tells how axes map to world (X,Y,Z).
             axis_order="XYZ" means data_np[x,y,z].
             axis_order="ZYX" means data_np[z,y,x], etc.
    plane_obj: Blender object. Uses plane_obj.matrix_world.
    out_h/out_w: output image resolution.
    voxel_size: world units per voxel.
    volume_origin_world: where the volume origin is in world coords.
    """

    assert data_np.ndim == 3, "data_np must be 3D"
    assert len(axis_order) == 3 and set(axis_order.upper()) == set(
        "XYZ"
    ), "axis_order must be permutation of XYZ"

    # ----- 1) local plane coordinates in [-0.5, 0.5] -----
    u = np.linspace(-0.5, 0.5, out_w, dtype=np.float32)
    v = np.linspace(-0.5, 0.5, out_h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="ij")  # (W,H)
    zeros = np.zeros_like(uu, dtype=np.float32)
    pts_local = np.stack([uu, vv, zeros], axis=-1)  # (W,H,3)

    # ----- 2) local -> world -----
    M = plane_obj.matrix_world
    pts_world = np.empty_like(pts_local, dtype=np.float32)

    # (W,H) loop (kept close to your original)
    for i in range(out_w):
        for j in range(out_h):
            p = Vector((float(pts_local[i, j, 0]), float(pts_local[i, j, 1]), 0.0))
            pw = M @ p
            pts_world[i, j, 0] = pw.x
            pts_world[i, j, 1] = pw.y
            pts_world[i, j, 2] = pw.z

    # ----- 3) world -> voxel index in (X,Y,Z) -----
    origin = np.array(volume_origin_world, dtype=np.float32)[None, None, :]  # (1,1,3)
    rel = pts_world - origin
    idx_xyz = rel / float(voxel_size)  # (W,H,3) in voxel units

    shape = data_np.shape
    axis_order = axis_order.upper()
    size_map = {axis_order[0]: shape[0], axis_order[1]: shape[1], axis_order[2]: shape[2]}
    nx, ny, nz = size_map["X"], size_map["Y"], size_map["Z"]

    # ----- 4) normalized grid -----
    def norm_coord(i, n):
        if n <= 1:
            return np.zeros_like(i, dtype=np.float32)
        return 2.0 * (i / (n - 1.0)) - 1.0

    x_norm = norm_coord(idx_xyz[..., 0], nx)
    y_norm = norm_coord(idx_xyz[..., 1], ny)
    z_norm = norm_coord(idx_xyz[..., 2], nz)

    grid = np.stack([x_norm, y_norm, z_norm], axis=-1).astype(np.float32)  # (W,H,3)
    grid_t = torch.from_numpy(grid).to(device=device, dtype=dtype)
    # grid_sample wants (N, outD, outH, outW, 3)
    # Our grid is currently (W,H,3) so transpose to (H,W,3)
    grid_t = grid_t.permute(1, 0, 2).unsqueeze(0).unsqueeze(1)  # (1,1,H,W,3)

    # ----- 5) permute volume to (D,H,W) == (Z,Y,X) -----
    axis_to_dim = {axis_order[0]: 0, axis_order[1]: 1, axis_order[2]: 2}
    x_dim = axis_to_dim["X"]
    y_dim = axis_to_dim["Y"]
    z_dim = axis_to_dim["Z"]

    data_zyx = np.transpose(data_np, (z_dim, y_dim, x_dim))  # (Z,Y,X)
    vol = torch.from_numpy(data_zyx).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

    sampled = F.grid_sample(
        vol,
        grid_t,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )  # (1,1,1,H,W)

    img = sampled[0, 0, 0].detach().cpu().numpy()  # (H,W)
    return img


# ----------------------------
# Slice -> RGBA + boundary outlining
# ----------------------------
def slice_to_rgba(
    slice2d,
    vmin,
    vmax,
    component,
    gamma_transparency,
    gamma_color,
    alpha_min,
    alpha_max,
    outline_boundary=True,
    outline_px=1,
    outline_rgba=(0.0, 0.0, 0.0, 1.0),
    cmap="seismic",
):
    """
    Returns rgba: (H,W,4) float32 in [0,1]
    Adds boundary outlines (black) where alpha transitions from 0 to >0.
    """
    s = slice2d.astype(np.float32)
    zero_mask = s == 0.0

    if component in ("real", "imag"):
        denom = max(float(vmax), 1e-8)

        t_col_norm = (s / denom)  # [-1,1]
        t_col_sign = np.sign(t_col_norm)
        t_col_norm = np.power(np.abs(t_col_norm), gamma_color) * t_col_sign
        t_col = t_col_norm * 0.5 + 0.5  # [0,1]

        rgba = cmap_rgba(t_col, cmap=cmap)
        t_mag = np.abs(s) / denom
    else:
        denom = max(float(vmax - vmin), 1e-12)
        t_col = (s - float(vmin)) / denom
        t_col = np.power(np.abs(t_col), gamma_color)

        rgba = cmap_rgba(t_col, cmap=cmap)
        t_mag = np.clip(t_col, 0.0, 1.0)

    # Transparency
    vis = np.power(t_mag, float(gamma_transparency))
    # alpha = float(alpha_min) + (float(alpha_max) - float(alpha_min)) * vis
    alpha = np.ones_like(t_mag, dtype=np.float32) * 0.7
    alpha[zero_mask] = 0.0
    rgba[..., 3] = np.clip(alpha, 0.0, 1.0)

    if outline_boundary:
        rgba = add_alpha_boundary_outline(
            rgba,
            alpha_thresh=1e-8,
            outline_px=int(outline_px),
            outline_rgba=outline_rgba,
        )

    return rgba


def add_alpha_boundary_outline(
    rgba: np.ndarray,
    alpha_thresh: float = 1e-8,
    outline_px: int = 1,
    outline_rgba=(0.0, 0.0, 0.0, 1.0),
):
    """
    Draw a black boundary (or any RGBA) between alpha==0 and alpha>0 regions.

    Implementation: morphological boundary from a binary mask (alpha>thresh),
    using 4-neighborhood differences, optionally dilated to thickness outline_px.
    """
    assert rgba.ndim == 3 and rgba.shape[-1] == 4
    a = rgba[..., 3]
    mask = a > float(alpha_thresh)

    # Edge where mask differs from any 4-neighbor
    m = mask
    edge = np.zeros_like(m, dtype=bool)
    edge[:, 1:] |= m[:, 1:] ^ m[:, :-1]
    edge[:, :-1] |= m[:, 1:] ^ m[:, :-1]
    edge[1:, :] |= m[1:, :] ^ m[:-1, :]
    edge[:-1, :] |= m[1:, :] ^ m[:-1, :]

    # Thicken edge by simple dilation (Chebyshev ball)
    if outline_px > 1:
        thick = edge.copy()
        for _ in range(outline_px - 1):
            t = thick.copy()
            t[:, 1:] |= thick[:, :-1]
            t[:, :-1] |= thick[:, 1:]
            t[1:, :] |= thick[:-1, :]
            t[:-1, :] |= thick[1:, :]
            thick = t
        edge = thick

    out = rgba.copy()
    out[edge, 0] = outline_rgba[0]
    out[edge, 1] = outline_rgba[1]
    out[edge, 2] = outline_rgba[2]
    out[edge, 3] = outline_rgba[3]
    return out


# ----------------------------
# Plane intersections as 3D lines
# ----------------------------
def plane_point_normal_world(obj):
    """
    Returns (point, normal) of the object's local Z=0 plane in world space.

    Blender plane primitive lies in local XY with normal +Z.
    """
    M = obj.matrix_world
    p0 = M @ Vector((0.0, 0.0, 0.0))
    n = (M.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
    return p0, n


def intersect_two_planes(p1, n1, p2, n2, eps=1e-10):
    """
    Intersection of two planes:
      plane1: n1·(x - p1)=0
      plane2: n2·(x - p2)=0

    Returns (point_on_line, direction) or (None, None) if parallel/almost-parallel.
    """
    d = n1.cross(n2)
    denom = d.length_squared
    if denom < eps:
        return None, None

    # Use formula from plane-plane intersection:
    # x0 = ( ( (p2·n2 - p1·n2) * n1 - (p2·n1 - p1·n1) * n2 ) × d ) / |d|^2 + p1
    # Equivalent robust-ish scalar form:
    c1 = n1.dot(p1)
    c2 = n2.dot(p2)

    # Compute a point on the line:
    x0 = (c1 * n2 - c2 * n1).cross(d) / denom
    return x0, d.normalized()


def clip_line_to_aabb(x0: Vector, d: Vector, aabb_min: Vector, aabb_max: Vector, eps=1e-12):
    """
    Clips infinite line x(t)=x0 + t d to AABB. Returns (p_enter, p_exit) or None.

    Uses slab method on param t.
    """
    tmin = -1e30
    tmax = 1e30
    for axis in range(3):
        o = x0[axis]
        v = d[axis]
        mn = aabb_min[axis]
        mx = aabb_max[axis]
        if abs(v) < eps:
            # line parallel to slab; must be within
            if o < mn or o > mx:
                return None
            continue
        t1 = (mn - o) / v
        t2 = (mx - o) / v
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return None
    pA = x0 + tmin * d
    pB = x0 + tmax * d
    return pA, pB


def get_or_create_black_material(mat_name="MAT_black_line", emission_strength=0.0):
    """
    Simple black material (optionally emissive if you set emission_strength>0).
    """
    mat = bpy.data.materials.get(mat_name)
    if mat is not None:
        return mat

    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)

    if emission_strength > 0.0:
        em = nt.nodes.new("ShaderNodeEmission")
        em.location = (0, 0)
        em.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        em.inputs["Strength"].default_value = float(emission_strength)
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    else:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bsdf.inputs["Roughness"].default_value = 1.0
        bsdf.inputs["Specular IOR Level"].default_value = 0.0
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    return mat


def add_intersection_curve(name, pA: Vector, pB: Vector, radius=0.002, material=None, collection=None):
    """
    Creates a 3D curve segment from pA to pB with bevel radius.
    """
    # Create curve data
    curve_data = bpy.data.curves.new(name=f"{name}_curve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 2

    spline = curve_data.splines.new("POLY")
    spline.points.add(1)  # total 2 points
    spline.points[0].co = (pA.x, pA.y, pA.z, 1.0)
    spline.points[1].co = (pB.x, pB.y, pB.z, 1.0)

    curve_data.bevel_depth = float(radius)
    curve_data.bevel_resolution = 4

    obj = bpy.data.objects.new(name, curve_data)
    if collection is None:
        bpy.context.collection.objects.link(obj)
    else:
        collection.objects.link(obj)

    if material is not None:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

    return obj


def add_all_plane_intersections_as_lines(
    planes,
    volume_aabb_min,
    volume_aabb_max,
    line_radius=0.003,
    material=None,
    name_prefix="PlaneIntersect",
):
    """
    For every pair of planes, add a clipped intersection line segment to the volume AABB.
    """
    if material is None:
        material = get_or_create_black_material("MAT_black_intersection", emission_strength=0.0)

    aabb_min = Vector(volume_aabb_min)
    aabb_max = Vector(volume_aabb_max)

    created = []
    for i in range(len(planes)):
        p1, n1 = plane_point_normal_world(planes[i])
        for j in range(i + 1, len(planes)):
            p2, n2 = plane_point_normal_world(planes[j])
            x0, d = intersect_two_planes(p1, n1, p2, n2)
            if x0 is None:
                continue

            seg = clip_line_to_aabb(x0, d, aabb_min, aabb_max)
            if seg is None:
                continue

            pA, pB = seg
            obj = add_intersection_curve(
                name=f"{name_prefix}_{i}_{j}",
                pA=pA,
                pB=pB,
                radius=line_radius,
                material=material,
            )
            created.append(obj)

    return created


# ----------------------------
# Main pipeline: prepare slices
# ----------------------------
def prepare_slices(
    slice_centers,
    slice_angles,
    field_s,
    USE_PERCENTILE_RANGE,
    PCT_LOW,
    PCT_HIGH,
    FIELD_COMPONENT,
    GAMMA_TRANSPARENCY,
    GAMMA_COLOR,
    ALPHA_MIN,
    ALPHA_MAX,
    CENTER,
    VOXEL_SIZE,
    EMISSION_STRENGTH,
    # New knobs:
    OUTLINE_BOUNDARY=True,
    OUTLINE_PX=1.5,
    OUTLINE_RGBA=(0.0, 0.0, 0.0, 1.0),
    DRAW_PLANE_INTERSECTIONS=True,
    INTERSECTION_LINE_RADIUS=0.03,
    CMAP='seismic'
):
    """
    slice_centers: list[(cx,cy,cz)] in *voxel index coords* where (0,0,0) is volume center if CENTER=True.
    slice_angles: list[(anglex,angley,anglez)] degrees
    field_s: 3D numpy array
    """

    assert len(slice_centers) == len(slice_angles)
    N_slices = len(slice_centers)
    nx, ny, nz = field_s.shape

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

    if not np.isfinite(vmin) or not np.isfinite(vmax) or (
        vmax <= vmin and FIELD_COMPONENT == "abs"
    ):
        raise RuntimeError(f"Bad field range: vmin={vmin}, vmax={vmax}")

    max_plane_size = round((nx**2 + ny**2 + nz**2) ** 0.5)

    # Offsets for placing plane centers in world
    offx = 0.0 if CENTER else 0.5 * nx * VOXEL_SIZE
    offy = 0.0 if CENTER else 0.5 * ny * VOXEL_SIZE
    offz = 0.0 if CENTER else 0.5 * nz * VOXEL_SIZE

    # World-space AABB of volume
    # You currently sample using volume_origin_world = (-0.5*nx*vs, -0.5*ny*vs, -0.5*nz*vs) (when CENTER=True)
    # so AABB is [origin, origin + (nx,ny,nz)*vs]
    volume_origin_world = (-0.5 * nx * VOXEL_SIZE, -0.5 * ny * VOXEL_SIZE, -0.5 * nz * VOXEL_SIZE)
    vol_min = Vector(volume_origin_world)
    vol_max = Vector(
        (
            volume_origin_world[0] + nx * VOXEL_SIZE,
            volume_origin_world[1] + ny * VOXEL_SIZE,
            volume_origin_world[2] + nz * VOXEL_SIZE,
        )
    )

    planes = []

    for i in range(N_slices):
        (cx, cy, cz), (anglex, angley, anglez) = slice_centers[i], slice_angles[i]
        slice_name = f"slice{i}"

        this_plane = add_plane(
            slice_name,
            size_x=max_plane_size * VOXEL_SIZE,
            size_y=max_plane_size * VOXEL_SIZE,
        )
        this_plane.rotation_euler = (
            math.radians(anglex),
            math.radians(angley),
            math.radians(anglez),
        )
        this_plane.location = (
            offx + cx * VOXEL_SIZE,
            offy + cy * VOXEL_SIZE,
            offz + cz * VOXEL_SIZE,
        )
        bpy.context.view_layer.update()

        # Sample slice
        field_slice = sample_plane_slice_from_volume(
            field_s,
            this_plane,
            max_plane_size,
            max_plane_size,
            voxel_size=VOXEL_SIZE,
            volume_origin_world=volume_origin_world,
            axis_order="XYZ",
            mode="bilinear",
            device="cpu",
            dtype=torch.float32,
        )

        # Convert to RGBA (+ boundary outline)
        rgba = slice_to_rgba(
            field_slice,
            vmin,
            vmax,
            FIELD_COMPONENT,
            GAMMA_TRANSPARENCY,
            GAMMA_COLOR,
            ALPHA_MIN,
            ALPHA_MAX,
            outline_boundary=OUTLINE_BOUNDARY,
            outline_px=OUTLINE_PX,
            outline_rgba=OUTLINE_RGBA,
            cmap=CMAP,
        )

        # Write image + assign emissive material
        this_img = ensure_image(slice_name, w=max_plane_size, h=max_plane_size)
        write_image_pixels(this_img, rgba, slice_name)

        this_mat = make_emissive_image_material(
            f"MAT_slice_{i}", this_img, emission_strength=EMISSION_STRENGTH
        )

        if this_plane.data.materials:
            this_plane.data.materials[0] = this_mat
        else:
            this_plane.data.materials.append(this_mat)

        planes.append(this_plane)

    # (2) plane-plane intersections as black 3D curve lines
    if DRAW_PLANE_INTERSECTIONS and len(planes) >= 2:
        black_mat = get_or_create_black_material("MAT_black_intersections", emission_strength=0.0)
        add_all_plane_intersections_as_lines(
            planes,
            volume_aabb_min=(vol_min.x, vol_min.y, vol_min.z),
            volume_aabb_max=(vol_max.x, vol_max.y, vol_max.z),
            line_radius=INTERSECTION_LINE_RADIUS,
            material=black_mat,
            name_prefix="PlaneIntersect",
        )

    return planes


# import numpy as np
# import math
# import torch
# import torch.nn.functional as F
# import bpy
# from mathutils import Vector

# from utils.color import cmaps
# from utils.material import make_emissive_image_material

# def cmap_rgba(t, cmap='seismic'):
#     """
#     t: array in [0,1]
#     returns rgba array (...,4)
#     """
#     t = np.clip(t, 0.0, 1.0).astype(np.float32)

#     CMAP = cmaps[cmap]
#     pos = np.array([p for p, _ in CMAP], dtype=np.float32)
#     col = np.array([c for _, c in CMAP], dtype=np.float32)  # (K,4)

#     # find segment index for each t
#     idx = np.searchsorted(pos, t, side='right') - 1
#     idx = np.clip(idx, 0, len(pos)-2)

#     p0 = pos[idx]
#     p1 = pos[idx+1]
#     w = (t - p0) / np.maximum(p1 - p0, 1e-8)

#     c0 = col[idx]
#     c1 = col[idx+1]
#     return c0 * (1.0 - w[..., None]) + c1 * (w[..., None])

# def ensure_image(name, w, h):
#     img = bpy.data.images.get(name)
#     if img is None:
#         img = bpy.data.images.new(
#             name=name,
#             width=w,
#             height=h,
#             alpha=True,
#             float_buffer=True
#         )
#     elif img.size[0] != w or img.size[1] != h:
#         bpy.data.images.remove(img)
#         img = bpy.data.images.new(
#             name=name,
#             width=w,
#             height=h,
#             alpha=True,
#             float_buffer=True
#         )

#     img.colorspace_settings.name = 'sRGB'
#     return img

# def write_image_pixels(img, rgba, name):
#     """
#     rgba: (H, W, 4) float32 in [0,1]
#     """
#     rgba = np.clip(rgba, 0.0, 1.0).astype(np.float32)
#     # rgba = np.flipud(rgba)   # optional, but recommended
#     rgba = rgba.transpose(1,0,2)
#     flat = rgba.reshape(-1)

#     # img.pixels.foreach_set(flat)
#     img.pixels = flat

#     # output_path = bpy.path.relpath(f"//{name}.png")
#     img.filepath_raw = f'//{name}.png'
#     img.file_format="PNG"
#     img.save(filepath=f'blender_files/{name}.png')
#     img.save(filepath=f'./{name}.png') # for rendering

# def add_plane(name, size_x, size_y):
#     """
#     Adds a plane centered at origin, size_x in local X, size_y in local Y.
#     """
#     bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
#     obj = bpy.context.object
#     obj.name = name
#     # default plane size=1 => spans [-0.5..0.5] in local X/Y. scale to match sizes.
#     obj.scale = (size_x, size_y, 1.0)
#     return obj

# def slice_to_rgba(slice2d, vmin, vmax, component, gamma_transparency, gamma_color, alpha_min, alpha_max):
#     """
#     slice2d: (H,W) float
#     For abs: color t = (val - vmin)/(vmax - vmin) mapped to [0,1]
#              alpha uses t^gamma
#     For real/imag: color t = (val + vmax)/(2*vmax) mapped to [0,1] (diverging around 0)
#                    alpha uses (|val|/vmax)^gamma
#     """
#     s = slice2d.astype(np.float32)
#     zero_mask = s == 0.0

#     if component in ("real", "imag"):
#         denom = max(float(vmax), 1e-8)
#         # assert (s < denom + 1e-6).all() and (s > -denom - 1e-6).all()

#         t_col_norm = (s / denom) # [-1, 1]
#         t_col_sign = np.sign(t_col_norm)
#         t_col_norm = np.power(np.abs(t_col_norm), gamma_color)
#         t_col_norm = t_col_norm * t_col_sign
#         t_col = t_col_norm * 0.5 + 0.5          # [0,1]
        
#         rgba = cmap_rgba(t_col, cmap='seismic')
#         t_mag = np.abs(s) / denom    
#     else:
#         denom = max(float(vmax - vmin), 1e-12)
#         t_col = (s - float(vmin)) / denom
#         t_col = np.power(np.abs(t_col), gamma_color)
#         print("min max: ", denom, np.min(t_col), np.max(t_col))
        
#         rgba = cmap_rgba(t_col, cmap='magma')
#         t_mag = np.clip(t_col, 0.0, 1.0)         # use same scale for visibility
    
#                # [0,1]
#     vis = np.power(t_mag, float(gamma_transparency))
#     # alpha = float(alpha_min) + (float(alpha_max) - float(alpha_min)) * vis
#     alpha = np.ones_like(t_mag)*0.9
#     alpha[zero_mask] = 0.0
#     rgba[..., 3] = np.clip(alpha, 0.0, 1.0)
    
#     return rgba


# def sample_plane_slice_from_volume(
#     data_np: np.ndarray,
#     plane_obj,
#     out_h: int,
#     out_w: int,
#     voxel_size: float = 1.0,
#     volume_origin_world=(0.0, 0.0, 0.0),
#     axis_order: str = "XYZ",
#     mode: str = "bilinear",
#     device: str | torch.device = "cpu",
#     dtype: torch.dtype = torch.float32,
# ):
#     """
#     data_np: 3D numpy array. axis_order tells how axes map to world (X,Y,Z).
#              axis_order="XYZ" means data_np[x,y,z].
#              axis_order="ZYX" means data_np[z,y,x], etc.
#     plane_obj: Blender object (your plane). Uses plane_obj.matrix_world (correct Blender transform order).
#     out_h/out_w: output image resolution (pixels).
#     voxel_size: world units per voxel (assumed isotropic).
#     volume_origin_world: where the *volume* origin is in world coords.
#     mode: 'bilinear' gives trilinear sampling for 3D input in grid_sample.
#     """

#     assert data_np.ndim == 3, "data_np must be 3D"
#     assert len(axis_order) == 3 and set(axis_order.upper()) == set("XYZ"), "axis_order must be permutation of XYZ"

#     # ----- 1) Make a grid of local plane coordinates (u,v,0) -----
#     # Blender primitive plane (size=1) is corners at (-1,-1,0) ... (1,1,0) in local space.
#     # Object scale stretches that; using matrix_world is enough, but we still generate in [-1,1].
#     u = np.linspace(-0.5, 0.5, out_w, dtype=np.float32)
#     v = np.linspace(-0.5, 0.5, out_h, dtype=np.float32)
#     uu, vv = np.meshgrid(u, v, indexing="ij")  # uu: (H,W), vv: (H,W)
#     zeros = np.zeros_like(uu, dtype=np.float32)

#     # local points in plane space (H,W,3)
#     pts_local = np.stack([uu, vv, zeros], axis=-1)

#     # ----- 2) Transform local points -> world points using Blender's matrix_world -----
#     M = plane_obj.matrix_world  # includes scale, rotation, translation in Blender's correct order
#     pts_world = np.empty_like(pts_local, dtype=np.float32)

#     # Loop is fine for moderate sizes; vectorizing via numpy is possible but more annoying with mathutils.
#     for i in range(out_w):
#         for j in range(out_h):
#             p = Vector((float(pts_local[i, j, 0]), float(pts_local[i, j, 1]), 0.0))
#             pw = M @ p  # applying full object transform
#             pts_world[i, j, 0] = pw.x
#             pts_world[i, j, 1] = pw.y
#             pts_world[i, j, 2] = pw.z

#     # ----- 3) Map world coords -> voxel index coords (ix,iy,iz in "XYZ" meaning world X,Y,Z) -----
#     origin = np.array(volume_origin_world, dtype=np.float32)[None, None, :]  # (1,1,3)
#     rel = pts_world - origin  # (H,W,3)

#     # Convert to "index units": world / voxel_size
#     idx_xyz = rel / float(voxel_size)  # (H,W,3) in voxel units

#     # If the volume is centered at origin, shift so that origin maps to center index.
#     # center index is (n-1)/2 along each axis.
#     shape = data_np.shape
#     # Interpret shape according to axis_order: data_np[axis_order[0], axis_order[1], axis_order[2]]
#     # We want sizes for X, Y, Z in world terms.
#     axis_order = axis_order.upper()
#     size_map = {axis_order[0]: shape[0], axis_order[1]: shape[1], axis_order[2]: shape[2]}
#     nx, ny, nz = size_map["X"], size_map["Y"], size_map["Z"]

#     # Now idx_xyz is voxel indices in X,Y,Z.
#     # grid_sample wants normalized coords in order (x, y, z) in [-1,1] where x maps W, y maps H, z maps D.

#     # ----- 4) Build normalized grid for grid_sample -----
#     # With align_corners=True:
#     #   x_norm = 2 * (ix / (W-1)) - 1
#     # similarly for y,z
#     def norm_coord(i, n):
#         # n is size along that dimension
#         if n <= 1:
#             return np.zeros_like(i, dtype=np.float32)
#         return 2.0 * (i / (n - 1.0)) - 1.0

#     x_norm = norm_coord(idx_xyz[..., 0], nx)
#     y_norm = norm_coord(idx_xyz[..., 1], ny)
#     z_norm = norm_coord(idx_xyz[..., 2], nz)

#     grid = np.stack([x_norm, y_norm, z_norm], axis=-1).astype(np.float32)  # (H,W,3)
#     grid_t = torch.from_numpy(grid).to(device=device, dtype=dtype)
#     grid_t = grid_t.unsqueeze(0).unsqueeze(1)  # (N=1, outD=1, outH=H, outW=W, 3)

#     # ----- 5) Prepare input volume for grid_sample -----
#     # grid_sample 3D expects input (N,C,D,H,W) where coordinates correspond to (x->W, y->H, z->D)
#     # We currently have data_np in some axis_order. We need to permute it into (Z,Y,X) == (D,H,W).
#     # axis_order tells how data_np axes map to world X,Y,Z.
#     # Example axis_order="XYZ": data_np[x,y,z] -> to (z,y,x) => transpose (2,1,0)
#     axis_to_dim = {axis_order[0]: 0, axis_order[1]: 1, axis_order[2]: 2}
#     x_dim = axis_to_dim["X"]
#     y_dim = axis_to_dim["Y"]
#     z_dim = axis_to_dim["Z"]

#     data_zyx = np.transpose(data_np, (z_dim, y_dim, x_dim))  # (Z,Y,X) -> (D,H,W)
#     vol = torch.from_numpy(data_zyx).to(device=device, dtype=dtype)
#     vol = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

#     # ----- 6) Sample -----
#     # mode='bilinear' on 5D input performs trilinear interpolation.
#     # padding_mode='zeros' makes out-of-bounds -> 0.
#     sampled = F.grid_sample(
#         vol,
#         grid_t,
#         mode=mode,
#         padding_mode="zeros",
#         align_corners=True,
#     )  # (1,1,1,H,W)

#     img = sampled[0, 0, 0].detach().cpu().numpy()  # (H,W)
#     return img


# def prepare_slices(slice_centers, slice_angles, field_s, USE_PERCENTILE_RANGE, PCT_LOW, PCT_HIGH, FIELD_COMPONENT, GAMMA_TRANSPARENCY, GAMMA_COLOR, ALPHA_MIN, ALPHA_MAX, CENTER, VOXEL_SIZE, EMISSION_STRENGTH):
#     assert len(slice_centers) == len(slice_angles)
#     N_slices = len(slice_centers)
#     nx, ny, nz = field_s.shape

#     # Range for colormap
#     if USE_PERCENTILE_RANGE:
#         vmin = float(np.percentile(field_s, PCT_LOW))
#         vmax = float(np.percentile(field_s, PCT_HIGH))
#     else:
#         vmin = float(np.min(field_s))
#         vmax = float(np.max(field_s))

#     if FIELD_COMPONENT in ("real", "imag"):
#         vm = max(abs(vmin), abs(vmax))
#         vmax = vm
#         vmin = -vm

#     if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax <= vmin and FIELD_COMPONENT == "abs"):
#         raise RuntimeError(f"Bad field range: vmin={vmin}, vmax={vmax}")

#     max_plane_size = round((nx**2 + ny**2 + nz**2)**.5)
#     print("max_plane_size: ", max_plane_size, "VOXEL_SIZE: ", VOXEL_SIZE)
#     # Helpful offsets if CENTER
#     offx = 0.0 if CENTER else 0.5 * nx * VOXEL_SIZE
#     offy = 0.0 if CENTER else 0.5 * ny * VOXEL_SIZE
#     offz = 0.0 if CENTER else 0.5 * nz * VOXEL_SIZE

#     for i in range(N_slices):
#         (cx, cy, cz), (anglex, angley, anglez) = slice_centers[i], slice_angles[i]
#         slice_name = f"slice{i}"
#         this_plane = add_plane(slice_name, size_x=max_plane_size * VOXEL_SIZE, size_y=max_plane_size * VOXEL_SIZE)
#         this_plane.rotation_euler = (math.radians(anglex), math.radians(angley), math.radians(anglez))
#         this_plane.location = (offx + cx * VOXEL_SIZE, offy + cy * VOXEL_SIZE, offz + cz * VOXEL_SIZE)
#         print("plane location: ", this_plane.location)
#         bpy.context.view_layer.update()

#         field_slice = sample_plane_slice_from_volume(field_s, this_plane, max_plane_size, max_plane_size, voxel_size=VOXEL_SIZE, volume_origin_world=(-0.5 * nx * VOXEL_SIZE, -0.5 * ny * VOXEL_SIZE, -0.5 * nz * VOXEL_SIZE), axis_order="XYZ", mode="bilinear", device="cpu", dtype=torch.float32)

#         rgba = slice_to_rgba(field_slice, vmin, vmax, FIELD_COMPONENT, GAMMA_TRANSPARENCY, GAMMA_COLOR, ALPHA_MIN, ALPHA_MAX)
#         this_img = ensure_image(slice_name, w=max_plane_size, h=max_plane_size)
#         write_image_pixels(this_img, rgba, slice_name)

#         this_mat = make_emissive_image_material(f"MAT_slice_{i}", this_img, emission_strength=EMISSION_STRENGTH)

#         if this_plane.data.materials:
#             this_plane.data.materials[0] = this_mat
#         else:
#             this_plane.data.materials.append(this_mat)
