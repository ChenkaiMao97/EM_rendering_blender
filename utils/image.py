from utils.const import cmaps
import numpy as np
import bpy

def set_colorramp_seismic(ramp_node):
    """
    Configure a ShaderNodeValToRGB (ColorRamp) to approximate matplotlib 'seismic'.
    Uses 11 samples from the official colormap.
    """
    cr = ramp_node.color_ramp

    # (pos, (r,g,b,a)) sampled from matplotlib.cm.get_cmap("seismic") at 0..1
    stops = [
        (0.0, (0.0,    0.0,    0.3,    1.0)),
        (0.1, (0.0,    0.0,    0.5745, 1.0)),
        (0.2, (0.0,    0.0,    0.86,   1.0)),
        (0.3, (0.1922, 0.1922, 1.0,    1.0)),
        (0.4, (0.6,    0.6,    1.0,    1.0)),
        (0.5, (1.0,    0.9922, 0.9922, 1.0)),
        (0.6, (1.0,    0.6,    0.6,    1.0)),
        (0.7, (1.0,    0.1922, 0.1922, 1.0)),
        (0.8, (0.9,    0.0,    0.0,    1.0)),
        (0.9, (0.6961, 0.0,    0.0,    1.0)),
        (1.0, (0.5,    0.0,    0.0,    1.0)),
    ]

    # Ensure exact number of elements
    while len(cr.elements) < len(stops):
        cr.elements.new(0.5)
    while len(cr.elements) > len(stops):
        cr.elements.remove(cr.elements[-1])

    for elem, (pos, col) in zip(cr.elements, stops):
        elem.position = float(pos)
        elem.color = tuple(float(x) for x in col)

def cmap_rgba(t, cmap='seismic'):
    """
    t: array in [0,1]
    returns rgba array (...,4)
    """
    t = np.clip(t, 0.0, 1.0).astype(np.float32)

    CMAP = cmaps[cmap]
    pos = np.array([p for p, _ in CMAP], dtype=np.float32)
    col = np.array([c for _, c in CMAP], dtype=np.float32)  # (K,4)

    # find segment index for each t
    idx = np.searchsorted(pos, t, side='right') - 1
    idx = np.clip(idx, 0, len(pos)-2)

    p0 = pos[idx]
    p1 = pos[idx+1]
    w = (t - p0) / np.maximum(p1 - p0, 1e-8)

    c0 = col[idx]
    c1 = col[idx+1]
    return c0 * (1.0 - w[..., None]) + c1 * (w[..., None])

def ensure_image(name, w, h):
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.new(
            name=name,
            width=w,
            height=h,
            alpha=True,
            float_buffer=True
        )
    elif img.size[0] != w or img.size[1] != h:
        bpy.data.images.remove(img)
        img = bpy.data.images.new(
            name=name,
            width=w,
            height=h,
            alpha=True,
            float_buffer=True
        )

    img.colorspace_settings.name = 'sRGB'
    return img

def write_image_pixels(img, rgba, name):
    """
    rgba: (H, W, 4) float32 in [0,1]
    """
    rgba = np.clip(rgba, 0.0, 1.0).astype(np.float32)
    # rgba = np.flipud(rgba)   # optional, but recommended
    rgba = rgba.transpose(1,0,2)
    flat = rgba.reshape(-1)

    # img.pixels.foreach_set(flat)
    img.pixels = flat

    # output_path = bpy.path.relpath(f"//{name}.png")
    img.filepath_raw = f'//{name}.png'
    img.file_format="PNG"
    img.save(filepath=f'blender_files/{name}.png')
    img.save(filepath=f'./{name}.png') # for rendering

def add_plane(name, size_x, size_y):
    """
    Adds a plane centered at origin, size_x in local X, size_y in local Y.
    """
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.object
    obj.name = name
    # default plane size=1 => spans [-0.5..0.5] in local X/Y. scale to match sizes.
    obj.scale = (size_x, size_y, 1.0)
    return obj

def slice_to_rgba(slice2d, vmin, vmax, component, gamma_transparency, gamma_color, alpha_min, alpha_max):
    """
    slice2d: (H,W) float
    For abs: color t = (val - vmin)/(vmax - vmin) mapped to [0,1]
             alpha uses t^gamma
    For real/imag: color t = (val + vmax)/(2*vmax) mapped to [0,1] (diverging around 0)
                   alpha uses (|val|/vmax)^gamma
    """
    s = slice2d.astype(np.float32)

    if component in ("real", "imag"):
        denom = max(float(vmax), 1e-8)
        assert (s < denom).all() and (s > -denom).all()

        t_col_norm = (s / denom) # [-1, 1]
        t_col_sign = np.sign(t_col_norm)
        t_col_norm = np.power(np.abs(t_col_norm), gamma_color)
        t_col_norm = t_col_norm * t_col_sign
        t_col = t_col_norm * 0.5 + 0.5          # [0,1]
        
        rgba = cmap_rgba(t_col, cmap='seismic')
        t_mag = np.abs(s) / denom    
    else:
        denom = max(float(vmax - vmin), 1e-12)
        t_col = (s - float(vmin)) / denom
        t_col = np.power(np.abs(t_col), gamma_color)
        print("min max: ", denom, np.min(t_col), np.max(t_col))
        
        rgba = cmap_rgba(t_col, cmap='magma')
        t_mag = np.clip(t_col, 0.0, 1.0)         # use same scale for visibility
    
               # [0,1]
    vis = np.power(t_mag, float(gamma_transparency))
    alpha = float(alpha_min) + (float(alpha_max) - float(alpha_min)) * vis
    rgba[..., 3] = np.clip(alpha, 0.0, 1.0)
    
    return rgba