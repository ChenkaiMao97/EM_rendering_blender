import bpy
import bmesh
import numpy as np
import math
from mathutils import Vector
import matplotlib.pyplot as plt

def naive_voxel_surface_quads(occ, eps_xyz):
    """
    occ: bool array (nx, ny, nz)
    returns quads like greedy_mesh does: [((p0,p1,p2,p3), n), ...]
    all points are integer voxel-corner coords
    """
    occ = occ.astype(bool)
    nx, ny, nz = occ.shape
    quads = []

    # For each filled voxel, add any face whose neighbor is empty/outside.
    # Face vertices are in voxel-corner coordinates.
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if not occ[x, y, z]:
                    continue

                eps_val = float(eps_xyz[x, y, z])

                # helper: check neighbor occupancy (outside => empty)
                def filled(x2, y2, z2):
                    if 0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz:
                        return occ[x2, y2, z2]
                    return False

                # -X face (normal -x): plane at x
                if not filled(x - 1, y, z):
                    p0 = (x,   y,   z)
                    p1 = (x,   y,   z+1)
                    p2 = (x,   y+1, z+1)
                    p3 = (x,   y+1, z)
                    quads.append(((p0,p1,p2,p3), (-1,0,0), eps_val))

                # +X face (normal +x): plane at x+1
                if not filled(x + 1, y, z):
                    p0 = (x+1, y,   z)
                    p1 = (x+1, y+1, z)
                    p2 = (x+1, y+1, z+1)
                    p3 = (x+1, y,   z+1)
                    quads.append(((p0,p1,p2,p3), (1,0,0), eps_val))

                # -Y face
                if not filled(x, y - 1, z):
                    p0 = (x,   y, z)
                    p1 = (x+1, y, z)
                    p2 = (x+1, y, z+1)
                    p3 = (x,   y, z+1)
                    quads.append(((p0,p1,p2,p3), (0,-1,0), eps_val))

                # +Y face
                if not filled(x, y + 1, z):
                    p0 = (x,   y+1, z)
                    p1 = (x,   y+1, z+1)
                    p2 = (x+1, y+1, z+1)
                    p3 = (x+1, y+1, z)
                    quads.append(((p0,p1,p2,p3), (0,1,0), eps_val))

                # -Z face
                if not filled(x, y, z - 1):
                    p0 = (x,   y,   z)
                    p1 = (x,   y+1, z)
                    p2 = (x+1, y+1, z)
                    p3 = (x+1, y,   z)
                    quads.append(((p0,p1,p2,p3), (0,0,-1), eps_val))

                # +Z face
                if not filled(x, y, z + 1):
                    p0 = (x,   y,   z+1)
                    p1 = (x+1, y,   z+1)
                    p2 = (x+1, y+1, z+1)
                    p3 = (x,   y+1, z+1)
                    quads.append(((p0,p1,p2,p3), (0,0,1), eps_val))

    return quads

def build_mesh_from_quads_with_eps(
    quads, name="DielectricMesh", scale=1.0, origin=(0,0,0), attr_name="eps"
):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    vert_cache = {}
    origin = Vector(origin)

    # Keep eps values aligned with faces we successfully create
    face_eps_vals = []

    def get_vert(p):
        if p in vert_cache:
            return vert_cache[p]
        v = bm.verts.new(origin + scale * Vector(p))
        vert_cache[p] = v
        return v

    for quad, _n, eps_val in quads:
        vs = [get_vert(p) for p in quad]
        try:
            f = bm.faces.new(vs)
            face_eps_vals.append(float(eps_val))
        except ValueError:
            # face already exists (duplicate); skip
            pass

    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    # --- Write eps to a real Mesh attribute (reliable in Blender 4/5) ---
    # Ensure correct size: mesh.polygons corresponds to FACE domain
    if len(mesh.polygons) != len(face_eps_vals):
        # This can happen if remove_doubles / topology ops changed face count.
        # In that case, do not silently write wrong alignment.
        print(f"[WARN] polygon count changed: {len(mesh.polygons)} polys vs {len(face_eps_vals)} eps vals")
        # Best-effort: truncate to min length
        n = min(len(mesh.polygons), len(face_eps_vals))
        face_eps_vals = face_eps_vals[:n]

    # Create or replace attribute
    if mesh.attributes.get(attr_name):
        mesh.attributes.remove(mesh.attributes.get(attr_name))
    layer = mesh.attributes.new(name=attr_name, type='FLOAT', domain='FACE')

    for i, v in enumerate(face_eps_vals):
        layer.data[i].value = v

    mesh.update()
    return obj


def assign_eps_face_attribute_from_volume(
    obj,
    eps_xyz,                 # numpy array shaped (nx,ny,nz) in XYZ order
    voxel_size,
    centered=True,
    attr_name="eps",
    sample="nearest",        # "nearest" or "trilinear" or "local_max"
):
    me = obj.data
    nx, ny, nz = eps_xyz.shape

    # Create/replace attribute
    if me.attributes.get(attr_name):
        me.attributes.remove(me.attributes.get(attr_name))
    layer = me.attributes.new(name=attr_name, type='FLOAT', domain='FACE')

    # World->voxel coordinate mapping
    # Your centering convention:
    # obj.location = (-0.5*nx*voxel_size, -0.5*ny*voxel_size, -0.5*nz*voxel_size)
    offx = (-0.5 * nx * voxel_size) if centered else 0.0
    offy = (-0.5 * ny * voxel_size) if centered else 0.0
    offz = (-0.5 * nz * voxel_size) if centered else 0.0

    def clamp(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    def sample_nearest(xf, yf, zf):
        xi = int(clamp(math.floor(xf), 0, nx - 1))
        yi = int(clamp(math.floor(yf), 0, ny - 1))
        zi = int(clamp(math.floor(zf), 0, nz - 1))
        return float(eps_xyz[xi, yi, zi])
    
    def sample_local_max(xf, yf, zf, radius = 5):
        x_s = int(clamp(math.floor(xf-radius), 0, nx - 1))
        x_e = int(clamp(math.floor(xf+radius), 0, nx - 1))

        y_s = int(clamp(math.floor(yf-radius), 0, ny - 1))
        y_e = int(clamp(math.floor(yf+radius), 0, ny - 1))

        z = int(clamp(math.floor(zf-1), 0, nz - 1))
        return float(np.max(eps_xyz[x_s:x_e, y_s:y_e, z]))

        # z_s = int(clamp(math.floor(zf-radius), 0, nz - 1))
        # z_e = int(clamp(math.floor(zf+radius), 0, nz - 1))
        # return float(np.max(eps_xyz[x_s:x_e, y_s:y_e, z_s:z_e]))

    def sample_trilinear(xf, yf, zf):
        # clamp inside valid interpolation range
        xf = clamp(xf, 0.0, nx - 1.000001)
        yf = clamp(yf, 0.0, ny - 1.000001)
        zf = clamp(zf, 0.0, nz - 1.000001)

        x0 = int(math.floor(xf)); x1 = min(x0 + 1, nx - 1)
        y0 = int(math.floor(yf)); y1 = min(y0 + 1, ny - 1)
        z0 = int(math.floor(zf)); z1 = min(z0 + 1, nz - 1)

        tx = xf - x0
        ty = yf - y0
        tz = zf - z0

        c000 = eps_xyz[x0,y0,z0]; c100 = eps_xyz[x1,y0,z0]
        c010 = eps_xyz[x0,y1,z0]; c110 = eps_xyz[x1,y1,z0]
        c001 = eps_xyz[x0,y0,z1]; c101 = eps_xyz[x1,y0,z1]
        c011 = eps_xyz[x0,y1,z1]; c111 = eps_xyz[x1,y1,z1]

        c00 = c000*(1-tx) + c100*tx
        c10 = c010*(1-tx) + c110*tx
        c01 = c001*(1-tx) + c101*tx
        c11 = c011*(1-tx) + c111*tx

        c0 = c00*(1-ty) + c10*ty
        c1 = c01*(1-ty) + c11*ty

        c = c0*(1-tz) + c1*tz
        return float(c)

    methods = {
        'nearest': sample_nearest,
        'trilinear': sample_trilinear,
        'local_max': sample_local_max
    }
    sampler = methods[sample]

    # Fill per-face
    # poly.center is in object local coordinates; convert to world for consistency
    M = obj.matrix_world
    for i, poly in enumerate(me.polygons):
        cw = M @ poly.center
        xf = (cw.x - offx) / voxel_size
        yf = (cw.y - offy) / voxel_size
        zf = (cw.z - offz) / voxel_size
        layer.data[i].value = sampler(xf, yf, zf)

    me.update()

def marching_cubes(volume_xyz, level):
    """
    Returns verts, faces in voxel-index coordinates (x,y,z) with float verts.
    Tries scikit-image first; falls back to PyMCubes if available.
    """
    try:
        from skimage import measure
        # measure.marching_cubes expects volume in (z, y, x) indexing in many examples,
        # but it's actually just array indexing order. We'll keep our coordinates consistent
        # by swapping axes in/out so that returned verts are (x,y,z).
        #
        # Easiest: run marching cubes on a (z,y,x) view, then flip returned coords.
        vol_zyx = np.transpose(volume_xyz, (2, 1, 0))  # (x,y,z)->(z,y,x)
        verts_zyx, faces, _normals, _values = measure.marching_cubes(vol_zyx, level=float(level))
        # verts_zyx columns are (z,y,x) -> convert to (x,y,z)
        verts_xyz = np.stack([verts_zyx[:, 2], verts_zyx[:, 1], verts_zyx[:, 0]], axis=1)
        return verts_xyz, faces
    except Exception:
        pass

    try:
        import mcubes  # PyMCubes
        # mcubes expects array indexing; we'll use the same trick
        vol_zyx = np.transpose(volume_xyz, (2, 1, 0))
        verts_zyx, faces = mcubes.marching_cubes(vol_zyx, float(level))
        verts_xyz = np.stack([verts_zyx[:, 2], verts_zyx[:, 1], verts_zyx[:, 0]], axis=1)
        return verts_xyz, faces
    except Exception as e:
        raise RuntimeError(
            "No marching cubes backend found.\n"
            "Install one of:\n"
            "  - scikit-image (recommended)\n"
            "  - PyMCubes (mcubes)\n\n"
            "In Blender's Python console you can run:\n"
            "  import sys, subprocess\n"
            "  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-image'])\n"
            "or:\n"
            "  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'PyMCubes'])\n"
        ) from e

def build_mesh_from_mc(verts, faces, name="IsoMesh", iso_value=0.0, attr_name="iso"):
    """
    Build a Blender mesh from marching-cubes verts/faces.
    Stores a per-face float attribute attr_name set to iso_value.
    """
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Fast creation
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()

    # Face attribute (Blender 3.0+)
    if hasattr(mesh, "attributes"):
        if attr_name in mesh.attributes:
            attr = mesh.attributes[attr_name]
        else:
            attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='FACE')
        for i in range(len(mesh.polygons)):
            attr.data[i].value = float(iso_value)

    # Smooth shading (cheap)
    for p in mesh.polygons:
        p.use_smooth = True

    return obj