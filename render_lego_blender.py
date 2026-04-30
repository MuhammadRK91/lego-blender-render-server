import bpy
import json
import math
import os
import sys
from mathutils import Vector


DEFAULT_INPUT_PATH = "/tmp/blender_input.json"
DEFAULT_OUTPUT_PATH = "/tmp/lego_render.png"

# LEGO-style dimensions
STUD_SIZE = 1.0
PLATE_HEIGHT = 0.32
BRICK_HEIGHT = 0.96
STUD_RADIUS = 0.28
STUD_HEIGHT = 0.10

# Safety controls
MAX_PARTS_TO_RENDER = 25000
ADD_STUDS = True
ADD_BEVELS = True


def get_script_args():
    """
    Allows running:
    blender --background --python render_lego_blender.py -- --input /tmp/blender_input.json --output /tmp/lego_render.png
    """
    input_path = DEFAULT_INPUT_PATH
    output_path = DEFAULT_OUTPUT_PATH

    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    else:
        args = []

    for i, arg in enumerate(args):
        if arg == "--input" and i + 1 < len(args):
            input_path = args[i + 1]
        elif arg == "--output" and i + 1 < len(args):
            output_path = args[i + 1]

    return input_path, output_path


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def create_material(name, rgb, roughness=0.42):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    r, g, b = rgb
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0

    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0

    return mat


def get_material(cache, color_name, rgb):
    key = color_name

    if key in cache:
        return cache[key]

    mat = create_material(color_name, rgb)
    cache[key] = mat
    return mat


def add_bevel_and_normals(obj, bevel_width=0.035, bevel_segments=2):
    if not ADD_BEVELS:
        return

    bevel = obj.modifiers.new("soft_plastic_bevel", "BEVEL")
    bevel.width = bevel_width
    bevel.segments = bevel_segments

    normals = obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    return obj


def create_brick_body(part, material, grid_width, grid_height):
    """
    Creates a rectangular LEGO-style brick/plate body.

    Input part fields:
    x, y, z_plate, width, depth, height_plates
    """
    x = part["x"]
    y = part["y"]
    z_plate = part["z_plate"]
    width = part["width"]
    depth = part["depth"]
    height_plates = part["height_plates"]

    height = height_plates * PLATE_HEIGHT

    # Center model around world origin
    world_x = (x + width / 2.0) - (grid_width / 2.0)
    world_y = (y + depth / 2.0) - (grid_height / 2.0)
    world_z = (z_plate * PLATE_HEIGHT) + (height / 2.0)

    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(world_x, world_y, world_z)
    )

    obj = bpy.context.object
    obj.name = f"{part['part']}_{part['color']}"

    obj.dimensions = (
        width * STUD_SIZE * 0.96,
        depth * STUD_SIZE * 0.96,
        height
    )

    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    obj.data.materials.append(material)
    add_bevel_and_normals(obj, bevel_width=0.035, bevel_segments=2)

    return obj


def create_stud(world_x, world_y, top_z, material):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24,
        radius=STUD_RADIUS,
        depth=STUD_HEIGHT,
        location=(world_x, world_y, top_z + STUD_HEIGHT / 2.0)
    )

    stud = bpy.context.object
    stud.name = "lego_stud"
    stud.data.materials.append(material)

    add_bevel_and_normals(stud, bevel_width=0.018, bevel_segments=2)

    return stud


def build_top_plate_map(parts):
    """
    Finds the highest plate level per x/y position.
    This helps us add studs only on visible top surfaces,
    not inside hidden layers.
    """
    top_map = {}

    for part in parts:
        x = int(part["x"])
        y = int(part["y"])
        width = int(part["width"])
        depth = int(part["depth"])

        z_plate = int(part["z_plate"])
        height_plates = int(part["height_plates"])
        top_plate = z_plate + height_plates

        for xx in range(x, x + width):
            for yy in range(y, y + depth):
                key = (xx, yy)
                if key not in top_map or top_plate > top_map[key]:
                    top_map[key] = top_plate

    return top_map


def should_add_studs_for_part(part, top_map):
    x = int(part["x"])
    y = int(part["y"])
    width = int(part["width"])
    depth = int(part["depth"])

    top_plate = int(part["z_plate"]) + int(part["height_plates"])

    visible_count = 0
    total_count = 0

    for xx in range(x, x + width):
        for yy in range(y, y + depth):
            total_count += 1
            if top_map.get((xx, yy)) == top_plate:
                visible_count += 1

    return visible_count > 0 and visible_count >= max(1, total_count * 0.35)


def add_studs_for_part(part, material, grid_width, grid_height, top_map):
    if not ADD_STUDS:
        return

    if not should_add_studs_for_part(part, top_map):
        return

    x = int(part["x"])
    y = int(part["y"])
    width = int(part["width"])
    depth = int(part["depth"])

    top_z = (int(part["z_plate"]) + int(part["height_plates"])) * PLATE_HEIGHT

    # Put studs on every stud position inside this part.
    # For large parts this gives a LEGO-like top surface.
    for xx in range(x, x + width):
        for yy in range(y, y + depth):
            if top_map.get((xx, yy)) != int(part["z_plate"]) + int(part["height_plates"]):
                continue

            world_x = (xx + 0.5) - (grid_width / 2.0)
            world_y = (yy + 0.5) - (grid_height / 2.0)

            create_stud(world_x, world_y, top_z, material)


def create_baseplate(grid_width, grid_height):
    mat = create_material("baseplate_dark_gray", [55, 55, 55], roughness=0.5)

    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(0, 0, -0.08)
    )

    base = bpy.context.object
    base.name = "baseplate"

    base.dimensions = (
        grid_width * STUD_SIZE,
        grid_height * STUD_SIZE,
        0.16
    )

    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    base.data.materials.append(mat)
    add_bevel_and_normals(base, bevel_width=0.04, bevel_segments=1)

    return base


def setup_lighting(grid_width, grid_height):
    # Key area light
    bpy.ops.object.light_add(
        type="AREA",
        location=(-grid_width * 0.35, -grid_height * 0.55, 90)
    )
    key = bpy.context.object
    key.name = "large_softbox_key_light"
    key.data.energy = 8000
    key.data.size = 70

    # Fill light
    bpy.ops.object.light_add(
        type="AREA",
        location=(grid_width * 0.45, grid_height * 0.35, 50)
    )
    fill = bpy.context.object
    fill.name = "soft_fill_light"
    fill.data.energy = 1300
    fill.data.size = 80

    # Small rim light
    bpy.ops.object.light_add(
        type="POINT",
        location=(0, grid_height * 0.65, 40)
    )
    rim = bpy.context.object
    rim.name = "soft_rim_light"
    rim.data.energy = 450


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(grid_width, grid_height):
    """
    Camera is angled to show LEGO height, studs, and shadows.
    """
    cam_x = 0
    cam_y = -grid_height * 1.15
    cam_z = max(grid_width, grid_height) * 0.55

    bpy.ops.object.camera_add(
        location=(cam_x, cam_y, cam_z)
    )

    camera = bpy.context.object
    camera.name = "render_camera"
    bpy.context.scene.camera = camera

    target = (0, 0, 5)
    look_at(camera, target)

    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(grid_width, grid_height) * 1.05

    return camera


def setup_world():
    world = bpy.context.scene.world
    world.color = (1, 1, 1)


def setup_render(output_path):
    scene = bpy.context.scene

    # Cycles gives better shadows. If it fails, Blender may still fallback depending on environment.
    scene.render.engine = "CYCLES"

    try:
        scene.cycles.samples = 48
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 4
    except Exception:
        pass

    scene.render.resolution_x = 1536
    scene.render.resolution_y = 1024
    scene.render.film_transparent = False

    # Color management
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1

    # Output
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = "PNG"


def create_ground_plane(grid_width, grid_height):
    mat = create_material("matte_white_ground", [245, 245, 245], roughness=0.65)

    bpy.ops.mesh.primitive_plane_add(
        size=1,
        location=(0, 0, -0.18)
    )

    plane = bpy.context.object
    plane.name = "matte_ground_plane"
    plane.dimensions = (
        grid_width * 1.25,
        grid_height * 1.25,
        1
    )

    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    plane.data.materials.append(mat)

    return plane


def main():
    input_path, output_path = get_script_args()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parts = data.get("blender_optimized_parts", [])
    config = data.get("blender_render_config", {})
    grid_size = data.get("grid_size", {})

    grid_width = int(config.get("grid_width", grid_size.get("width", 256)))
    grid_height = int(config.get("grid_height", grid_size.get("height", 192)))

    if not parts:
        raise ValueError("No blender_optimized_parts found in input JSON.")

    if len(parts) > MAX_PARTS_TO_RENDER:
        print(f"WARNING: Part count {len(parts)} is high. Rendering first {MAX_PARTS_TO_RENDER} parts.")
        parts = parts[:MAX_PARTS_TO_RENDER]

    clear_scene()
    setup_world()

    material_cache = {}
    top_map = build_top_plate_map(parts)

    create_ground_plane(grid_width, grid_height)
    create_baseplate(grid_width, grid_height)

    rendered_count = 0

    for part in parts:
        color_name = part.get("color", "light_bluish_gray")
        rgb = part.get("rgb", [160, 165, 169])

        material = get_material(material_cache, color_name, rgb)

        create_brick_body(
            part=part,
            material=material,
            grid_width=grid_width,
            grid_height=grid_height
        )

        add_studs_for_part(
            part=part,
            material=material,
            grid_width=grid_width,
            grid_height=grid_height,
            top_map=top_map
        )

        rendered_count += 1

        if rendered_count % 1000 == 0:
            print(f"Created {rendered_count} parts...")

    setup_lighting(grid_width, grid_height)
    setup_camera(grid_width, grid_height)
    setup_render(output_path)

    # Save .blend file too, useful for debugging later.
    blend_path = output_path.replace(".png", ".blend")
    try:
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    except Exception as e:
        print(f"Could not save blend file: {e}")

    bpy.ops.render.render(write_still=True)

    print(f"Rendered LEGO model successfully.")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Parts rendered: {rendered_count}")
    print(f"Materials: {len(material_cache)}")


if __name__ == "__main__":
    main()
