import bpy
import bmesh
import numpy as np
import math
from mathutils import Vector
import matplotlib.pyplot as plt

from utils.util import set_node_input
from utils.image import set_colorramp_seismic

def make_material(
    MATERIAL_ROUGHNESS,
    MATERIAL_METALLIC,
    MATERIAL_IOR,
    MATERIAL_TRANSMISSION_WEIGHT,
    name="Dielectric", 
    attr_name="eps", 
    eps_min=1.0, 
    eps_max=8.0,
    MAKE_GLASSY=True,
    ALPHA=0.75
):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in nt.nodes:
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    nt.links.new(bsdf.outputs[0], out.inputs[0])

    # Read attribute
    attr = nt.nodes.new("ShaderNodeAttribute")  # works in many Blender versions
    attr.location = (0, 0)
    attr.attribute_name = attr_name

    # Map Range eps -> 0..1
    mapr = nt.nodes.new("ShaderNodeMapRange")
    mapr.location = (220, 0)
    mapr.inputs["From Min"].default_value = eps_min
    mapr.inputs["From Max"].default_value = eps_max
    mapr.inputs["To Min"].default_value = 0.0
    mapr.inputs["To Max"].default_value = 1.0
    mapr.clamp = True

    # ColorRamp
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (450, 0)

    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.5, 0.5, 0.5, 1.0)   # black, RGBA
    cr.elements[1].position = 1.0
    cr.elements[1].color = (0.0, 0.0, 0.0, 1.0)   # white, RGBA

    nt.links.new(attr.outputs["Fac"], mapr.inputs["Value"])
    nt.links.new(mapr.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # set_node_input(bsdf, ["Base Color"], MATERIAL_BASE_COLOR)
    set_node_input(bsdf, ["Roughness"], MATERIAL_ROUGHNESS)
    set_node_input(bsdf, ["Metallic"], MATERIAL_METALLIC)
    set_node_input(bsdf, ["IOR"], MATERIAL_IOR)
    set_node_input(bsdf, ["Transmission Weight", "Transmission"], MATERIAL_TRANSMISSION_WEIGHT)

    # Make sure it's not fully opaque if glassy (some versions use Alpha)
    set_node_input(bsdf, ["Alpha"], ALPHA)

    # For Eevee transparency (if you switch engines later)
    mat.blend_method = 'BLEND' if MAKE_GLASSY else 'OPAQUE'
    # mat.shadow_method = 'HASHED'

    return mat


def make_emissive_image_material(name, image, emission_strength=1.0):
    """
    Emission colored by image RGB; alpha from image A (Transparent mix).
    """
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.location = (0, 0)
    tex.image = image
    tex.interpolation = 'Linear'

    emission = nt.nodes.new("ShaderNodeEmission")
    emission.location = (350, 120)
    nt.links.new(tex.outputs["Color"], emission.inputs["Color"])
    emission.inputs["Strength"].default_value = float(emission_strength)

    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (350, -120)

    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.location = (650, 0)

    # Fac = alpha: 0 -> transparent, 1 -> emission
    # (MixShader: Fac=0 gives input[1], Fac=1 gives input[2])
    nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    nt.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emission.outputs["Emission"], mix.inputs[2])

    nt.links.new(mix.outputs[0], out.inputs["Surface"])

    mat.blend_method = 'BLEND'
    try:
        mat.shadow_method = 'NONE'
    except Exception:
        pass

    return mat

def make_field_material_emissive(
    name="FieldEmissive",
    attr_name="iso",                 # your face float attribute
    vmin=0.0,
    vmax=1.0,
    field_component="abs",           # "abs" | "real" | "imag"
    gamma=2.0,                       # power exponent (higher = emphasize extremes more)
    emission_strength=1.0,          # global multiplier for emission
    alpha_min=0.03,                  # most transparent
    alpha_max=0.85,                  # most opaque
    use_seismic=True                 # optional: approximate matplotlib seismic
):
    """
    Builds a material:
      - abs: map [vmin,vmax] -> [0,1]
      - real/imag: map [vmin,vmax] -> [-1,1] then abs() -> [0,1]
      - apply pow(gamma)
      - use that to control Emission strength and opacity (via mix with Transparent)
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)

    # --- Attribute (your per-face float "iso") ---
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.location = (0, 0)
    attr.attribute_name = attr_name

    # --- Map Range ---
    mapr = nt.nodes.new("ShaderNodeMapRange")
    mapr.location = (200, 0)
    mapr.inputs["From Min"].default_value = float(vmin)
    mapr.inputs["From Max"].default_value = float(vmax)
    mapr.clamp = True

    nt.links.new(attr.outputs.get("Fac"), mapr.inputs["Value"])

    # abs: map -> [0,1]
    # real/imag: map -> [-1,1] then abs()
    if field_component in ("real", "imag"):
        mapr.inputs["To Min"].default_value = -1.0
        mapr.inputs["To Max"].default_value =  1.0

        abs_node = nt.nodes.new("ShaderNodeMath")
        abs_node.location = (400, 0)
        abs_node.operation = 'ABSOLUTE'
        nt.links.new(mapr.outputs["Result"], abs_node.inputs[0])

        mag = abs_node  # node whose output is now [0,1]
    else:
        mapr.inputs["To Min"].default_value = 0.0
        mapr.inputs["To Max"].default_value = 1.0
        mag = mapr

    # --- Power / Gamma ---
    pow_node = nt.nodes.new("ShaderNodeMath")
    pow_node.location = (600, 0)
    pow_node.operation = 'POWER'
    # base = mag, exponent = gamma
    nt.links.new(mag.outputs[0], pow_node.inputs[0])
    pow_node.inputs[1].default_value = float(gamma)

    # This is our “visibility factor” in [0,1] (after power)
    vis = pow_node

    # --- Color map: seismic-like (blue -> white -> red) ---
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (600, -250)
    set_colorramp_seismic(ramp)

    nt.links.new(vis.outputs[0], ramp.inputs["Fac"])

    # --- Emission shader ---
    emission = nt.nodes.new("ShaderNodeEmission")
    emission.location = (800, 120)
    nt.links.new(ramp.outputs["Color"], emission.inputs["Color"])

    # Strength = vis * emission_strength
    strength_mul = nt.nodes.new("ShaderNodeMath")
    strength_mul.location = (800, 260)
    strength_mul.operation = 'MULTIPLY'
    nt.links.new(vis.outputs[0], strength_mul.inputs[0])
    strength_mul.inputs[1].default_value = float(emission_strength)
    nt.links.new(strength_mul.outputs[0], emission.inputs["Strength"])

    # --- Transparency control ---
    # We want: low |field| => more transparent, high |field| => more opaque.
    # A nice mapping: alpha = alpha_min + (alpha_max-alpha_min)*vis
    alpha_scale = nt.nodes.new("ShaderNodeMath")
    alpha_scale.location = (800, -60)
    alpha_scale.operation = 'MULTIPLY'
    nt.links.new(vis.outputs[0], alpha_scale.inputs[0])
    alpha_scale.inputs[1].default_value = float(alpha_max - alpha_min)

    alpha_add = nt.nodes.new("ShaderNodeMath")
    alpha_add.location = (980, -60)
    alpha_add.operation = 'ADD'
    nt.links.new(alpha_scale.outputs[0], alpha_add.inputs[0])
    alpha_add.inputs[1].default_value = float(alpha_min)

    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (800, -220)

    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.location = (1120, 0)

    # Mix factor: 0 => Transparent, 1 => Emission
    # Using alpha_add as factor makes high vis “more emissive / less transparent”.
    nt.links.new(alpha_add.outputs[0], mix.inputs["Fac"])
    nt.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emission.outputs["Emission"], mix.inputs[2])

    nt.links.new(mix.outputs[0], out.inputs[0])

    # Enable blending in Eevee; Cycles handles transparency too.
    mat.blend_method = 'BLEND'
    try:
        mat.shadow_method = 'NONE'
    except Exception:
        pass

    return mat


def make_field_emission_material(
    FIELD_COMPONENT,
    name="FieldEmission",
    attr_name="iso",
    vmin=0.0,
    vmax=1.0,
    # Emission strength mapping (in Cycles these numbers matter a lot)
    EMIT_MIN=0.0,
    EMIT_MAX=2.0,
    # Transparency mapping (0 = opaque, 1 = fully transparent)
    ALPHA_MIN=0.7,   # high-value shells (more visible)
    ALPHA_MAX=0.99,   # low-value shells (mostly transparent)
    # Nonlinearity to boost extremes:
    # 1.0 = linear, >1 makes high values much brighter / low values dimmer
    GAMMA=2.0,
):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)

    # Attribute "iso"
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.location = (0, 0)
    attr.attribute_name = attr_name

    # Map iso -> t in [0,1]
    map01 = nt.nodes.new("ShaderNodeMapRange")
    map01.location = (200, 0)
    map01.inputs["From Min"].default_value = float(vmin)
    map01.inputs["From Max"].default_value = float(vmax)
    map01.inputs["To Min"].default_value = 0.0
    map01.inputs["To Max"].default_value = 1.0
    map01.clamp = True
    nt.links.new(attr.outputs.get("Fac"), map01.inputs["Value"])

    if FIELD_COMPONENT in ("real", "imag"):
        alpha_add = nt.nodes.new("ShaderNodeMath")
        alpha_add.location = (980, -60)
        alpha_add.operation = 'ADD'
        nt.links.new(map01.outputs["Result"], alpha_add.inputs[0])
        alpha_add.inputs[1].default_value = -0.5

        strength_mul = nt.nodes.new("ShaderNodeMath")
        strength_mul.location = (800, 260)
        strength_mul.operation = 'MULTIPLY'
        nt.links.new(alpha_add.outputs[0], strength_mul.inputs[0])
        strength_mul.inputs[1].default_value = 2

        abs_node = nt.nodes.new("ShaderNodeMath")
        abs_node.location = (400, 0)
        abs_node.operation = 'ABSOLUTE'
        nt.links.new(strength_mul.outputs[0], abs_node.inputs[0])

        mag = abs_node  # node whose output is now [0,1]
    else:
        mag = map01

    # Optional nonlinearity: t_gamma = t^GAMMA (boost extremes)
    # Use Math(Power)
    pow_node = nt.nodes.new("ShaderNodeMath")
    pow_node.location = (380, 0)
    pow_node.operation = 'POWER'

    nt.links.new(mag.outputs[0], pow_node.inputs[0])
    pow_node.inputs[1].default_value = float(GAMMA)

    # ColorRamp blue -> red driven by t (linear t looks better for color)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    set_colorramp_seismic(ramp)
    ramp.location = (380, -220)
    # while len(ramp.color_ramp.elements) > 2:
    #     ramp.color_ramp.elements.remove(ramp.color_ramp.elements[-1])
    # e0 = ramp.color_ramp.elements[0]
    # e1 = ramp.color_ramp.elements[1]
    # e0.position = 0.0
    # e1.position = 1.0
    # e0.color = (0.0, 0.2, 1.0, 1.0)  # blue
    # e1.color = (1.0, 0.1, 0.0, 1.0)  # red
    nt.links.new(map01.outputs["Result"], ramp.inputs["Fac"])

    # Emission shader
    emission = nt.nodes.new("ShaderNodeEmission")
    emission.location = (650, 120)
    nt.links.new(ramp.outputs["Color"], emission.inputs["Color"])

    # Emission strength: map t_gamma -> [EMIT_MIN, EMIT_MAX]
    emit_map = nt.nodes.new("ShaderNodeMapRange")
    emit_map.location = (650, 300)
    emit_map.inputs["From Min"].default_value = 0.0
    emit_map.inputs["From Max"].default_value = 1.0
    emit_map.inputs["To Min"].default_value = float(EMIT_MIN)
    emit_map.inputs["To Max"].default_value = float(EMIT_MAX)
    emit_map.clamp = True
    nt.links.new(pow_node.outputs["Value"], emit_map.inputs["Value"])
    nt.links.new(emit_map.outputs["Result"], emission.inputs["Strength"])

    # Transparency: low iso => more transparent
    # alpha(t_gamma): map t_gamma -> [ALPHA_MAX (low), ALPHA_MIN (high)]  (note inversion)
    alpha_map = nt.nodes.new("ShaderNodeMapRange")
    alpha_map.location = (650, -60)
    alpha_map.inputs["From Min"].default_value = 0.0
    alpha_map.inputs["From Max"].default_value = 1.0
    alpha_map.inputs["To Min"].default_value = float(ALPHA_MAX)  # low iso => high transparency
    alpha_map.inputs["To Max"].default_value = float(ALPHA_MIN)  # high iso => low transparency
    alpha_map.clamp = True
    nt.links.new(pow_node.outputs["Value"], alpha_map.inputs["Value"])

    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (650, -220)

    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.location = (830, 0)
    # Fac = alpha (0 opaque emission, 1 transparent)
    nt.links.new(alpha_map.outputs["Result"], mix.inputs["Fac"])
    nt.links.new(emission.outputs["Emission"], mix.inputs[1])
    nt.links.new(transparent.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    # Important for transparency
    mat.blend_method = 'BLEND'
    try:
        mat.shadow_method = 'NONE'
    except Exception:
        pass

    return mat