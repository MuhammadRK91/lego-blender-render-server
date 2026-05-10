from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import time
import uuid
import numpy as np
import trimesh
from collections import defaultdict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LEGO-style dimensions
STUD_W = 0.8
STUD_D = 0.8
PLATE_H = 0.24
BLOCK_GAP = 0.025

# Safety limit
MAX_PLACEMENTS = 30000

# Supported optimized plate parts.
# We only use common rectangular plates for now.
OPTIMIZED_PLATE_PARTS = {
    (1, 1): {
        "part_name": "Plate 1x1",
        "part_num": "3024"
    },
    (1, 2): {
        "part_name": "Plate 1x2",
        "part_num": "3023"
    },
    (1, 3): {
        "part_name": "Plate 1x3",
        "part_num": "3623"
    },
    (1, 4): {
        "part_name": "Plate 1x4",
        "part_num": "3710"
    },
    (2, 2): {
        "part_name": "Plate 2x2",
        "part_num": "3022"
    },
    (2, 3): {
        "part_name": "Plate 2x3",
        "part_num": "3021"
    },
    (2, 4): {
        "part_name": "Plate 2x4",
        "part_num": "3020"
    },
}

# Greedy placement priority.
# Larger parts first, then smaller parts.
# Important: this does NOT reduce image quality.
# It only merges adjacent cells that already have the same color and same height.
OPTIMIZATION_SHAPES = [
    (2, 4),
    (4, 2),
    (2, 3),
    (3, 2),
    (2, 2),
    (1, 4),
    (4, 1),
    (1, 3),
    (3, 1),
    (1, 2),
    (2, 1),
    (1, 1),
]


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "lego-model-generator",
        "message": "Send POST request to /generate-glb with image_geometry. Response includes GLB preview, optimized LEGO parts, build data, and product tier/pricing.",
        "uses_blender": False,
        "output": "glb_base64 + optimized_lego_model + product_tier"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "lego-model-generator",
        "mode": "image_geometry_to_glb_optimized_parts_and_product_tier",
        "uses_blender": False
    }


def normalize_hex(hex_value, fallback="CCCCCC"):
    if not hex_value:
        return fallback

    clean = str(hex_value).replace("#", "").strip()

    if len(clean) == 3:
        clean = "".join([c + c for c in clean])

    if len(clean) != 6:
        return fallback

    try:
        int(clean, 16)
        return clean.upper()
    except Exception:
        return fallback


def hex_to_rgba(hex_value, alpha=255):
    clean = normalize_hex(hex_value)

    return np.array(
        [
            int(clean[0:2], 16),
            int(clean[2:4], 16),
            int(clean[4:6], 16),
            alpha
        ],
        dtype=np.uint8
    )


def source_rgb_to_rgba(source_rgb, alpha=255):
    try:
        return np.array(
            [
                int(source_rgb.get("r", 160)),
                int(source_rgb.get("g", 165)),
                int(source_rgb.get("b", 169)),
                alpha
            ],
            dtype=np.uint8
        )
    except Exception:
        return np.array([160, 165, 169, alpha], dtype=np.uint8)


def extract_geometry(data):
    """
    Accepts these formats:

    1. Direct image_geometry:
    {
      "width": 160,
      "height": 90,
      "placements": [...]
    }

    2. Wrapped:
    {
      "image_geometry": {
        "width": 160,
        "height": 90,
        "placements": [...]
      }
    }

    3. n8n/server array:
    [
      {
        "image_geometry": {...}
      }
    ]

    4. n8n json wrapper:
    [
      {
        "json": {
          "width": 160,
          "height": 90,
          "placements": [...]
        }
      }
    ]
    """

    if isinstance(data, list) and data:
        first = data[0]

        if isinstance(first, dict):
            if "image_geometry" in first:
                return first["image_geometry"]

            if "json" in first:
                return extract_geometry(first["json"])

    if not isinstance(data, dict):
        return None

    if "image_geometry" in data:
        return data["image_geometry"]

    if (
        "width" in data
        and "height" in data
        and "placements" in data
    ):
        return data

    if "json" in data:
        return extract_geometry(data["json"])

    return None


def get_color_key(p):
    """
    Groups placements by actual LEGO color.
    Rebrickable color ID is preferred because names can vary.
    """
    color_id = p.get("rebrickable_color_id")
    color_name = p.get("rebrickable_color_name") or p.get("color") or "Unknown"
    color_rgb = normalize_hex(p.get("rebrickable_color_rgb"), fallback=None)

    if color_id is not None:
        return f"id:{color_id}"

    if color_rgb:
        return f"rgb:{color_rgb}"

    return f"name:{color_name}"


def get_cell_signature(p):
    """
    Only cells with the same signature can be combined into larger plates.

    This protects product/image quality:
    - no color simplification
    - no height simplification
    - no detail smoothing
    - no pixel removal
    """
    return (
        get_color_key(p),
        int(p.get("height_plates", 1)),
        p.get("rebrickable_color_id"),
        p.get("rebrickable_color_name") or p.get("color") or "Unknown",
        normalize_hex(p.get("rebrickable_color_rgb")),
    )


def normalized_part_size(w, h):
    """
    Rebrickable part keys are normalized as smaller side first.
    Example: 4x2 should use Plate 2x4 with orientation 90.
    """
    if (w, h) in OPTIMIZED_PLATE_PARTS:
        return w, h, 0

    if (h, w) in OPTIMIZED_PLATE_PARTS:
        return h, w, 90

    return 1, 1, 0


def can_place_rectangle(grid, used, x, y, w, h, signature, width, height):
    if x + w > width or y + h > height:
        return False

    for yy in range(y, y + h):
        for xx in range(x, x + w):
            if used[yy][xx]:
                return False

            cell = grid[yy][xx]
            if cell is None:
                return False

            if get_cell_signature(cell) != signature:
                return False

    return True


def mark_used(used, x, y, w, h):
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            used[yy][xx] = True


def optimize_image_geometry(geometry):
    """
    Converts the raw 1x1 placement grid into larger real LEGO plate placements
    wherever possible.

    Important:
    This optimization is quality-safe.
    It only combines adjacent cells when they already share the exact same:
    - color ID / color
    - Rebrickable RGB
    - height_plates

    It does not lower resolution.
    It does not merge similar colors.
    It does not remove details.
    """

    width = int(geometry.get("width", 0))
    height = int(geometry.get("height", 0))
    placements = geometry.get("placements", [])

    if width <= 0 or height <= 0:
        raise ValueError("image_geometry.width and image_geometry.height are required.")

    if not isinstance(placements, list) or not placements:
        raise ValueError("image_geometry.placements[] is required.")

    grid = [[None for _ in range(width)] for _ in range(height)]

    for p in placements:
        x = int(p.get("x", 0))
        y = int(p.get("y", 0))

        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = p

    used = [[False for _ in range(width)] for _ in range(height)]
    optimized = []

    for y in range(height):
        for x in range(width):
            if used[y][x]:
                continue

            cell = grid[y][x]

            if cell is None:
                continue

            signature = get_cell_signature(cell)
            chosen_w = 1
            chosen_h = 1

            for shape_w, shape_h in OPTIMIZATION_SHAPES:
                if can_place_rectangle(
                    grid=grid,
                    used=used,
                    x=x,
                    y=y,
                    w=shape_w,
                    h=shape_h,
                    signature=signature,
                    width=width,
                    height=height
                ):
                    chosen_w = shape_w
                    chosen_h = shape_h
                    break

            mark_used(used, x, y, chosen_w, chosen_h)

            normalized_w, normalized_h, orientation = normalized_part_size(chosen_w, chosen_h)
            part_info = OPTIMIZED_PLATE_PARTS.get(
                (normalized_w, normalized_h),
                OPTIMIZED_PLATE_PARTS[(1, 1)]
            )

            optimized.append({
                "x": x,
                "y": y,
                "z": int(cell.get("z", 0)),
                "w": chosen_w,
                "h": chosen_h,
                "part_name": part_info["part_name"],
                "part_num": part_info["part_num"],
                "color": cell.get("color"),
                "rebrickable_color_id": cell.get("rebrickable_color_id"),
                "rebrickable_color_name": cell.get("rebrickable_color_name"),
                "rebrickable_color_rgb": normalize_hex(cell.get("rebrickable_color_rgb")),
                "height_plates": max(1, min(12, int(cell.get("height_plates", 1)))),
                "orientation": orientation,
                "source_rgb": cell.get("source_rgb", {})
            })

    return optimized


def create_parts_summary(optimized_placements):
    summary = {}

    for p in optimized_placements:
        color_id = p.get("rebrickable_color_id")
        color_name = p.get("rebrickable_color_name") or p.get("color") or "Unknown"
        part_num = p.get("part_num")
        part_name = p.get("part_name")
        height_plates = int(p.get("height_plates", 1))

        key = f"{part_num}-{color_id}-h{height_plates}"

        if key not in summary:
            summary[key] = {
                "part_num": part_num,
                "part_name": part_name,
                "color_id": color_id,
                "color_name": color_name,
                "height_plates": height_plates,
                "quantity": 0
            }

        summary[key]["quantity"] += 1

    return summary


def create_rebrickable_parts_export(parts_summary):
    rows = []

    for item in parts_summary.values():
        if item.get("part_num") and item.get("color_id") is not None:
            rows.append({
                "part_num": item["part_num"],
                "color_id": item["color_id"],
                "color_name": item["color_name"],
                "quantity": item["quantity"],
                "height_plates": item["height_plates"]
            })

    return rows


def create_basic_xml_export(parts_summary):
    xml = "<INVENTORY>\n"

    for item in parts_summary.values():
        part_num = item.get("part_num")
        color_id = item.get("color_id")

        if not part_num or color_id is None:
            continue

        xml += "  <ITEM>\n"
        xml += f"    <PARTNUM>{part_num}</PARTNUM>\n"
        xml += f"    <COLORID>{color_id}</COLORID>\n"
        xml += f"    <COLORNAME>{item.get('color_name')}</COLORNAME>\n"
        xml += f"    <QTY>{item.get('quantity')}</QTY>\n"
        xml += f"    <HEIGHT_PLATES>{item.get('height_plates')}</HEIGHT_PLATES>\n"
        xml += "  </ITEM>\n"

    xml += "</INVENTORY>"
    return xml


def create_build_layers(optimized_placements):
    """
    Simple build sequence:
    group by height_plates.

    This is not a final instruction booklet yet.
    It is structured data needed to create instructions later.
    """
    layers = defaultdict(list)

    for p in optimized_placements:
        layer_num = int(p.get("height_plates", 1))
        layers[layer_num].append(p)

    build_layers = []

    for layer_num in sorted(layers.keys()):
        layer_placements = sorted(
            layers[layer_num],
            key=lambda p: (int(p.get("y", 0)), int(p.get("x", 0)))
        )

        build_layers.append({
            "layer": layer_num,
            "placement_count": len(layer_placements),
            "placements": layer_placements
        })

    return build_layers


def classify_product_tier(optimized_part_count, original_stud_count, reduction_percent):
    """
    Classifies the product tier based on final optimized part count.

    Important:
    This does not reduce quality.
    It only classifies the final model for pricing and selling.
    """

    if optimized_part_count <= 1500:
        return {
            "tier": "Basic",
            "price_level": "Low",
            "cost_level": "Low",
            "quality_level": "Good",
            "market_position": "Affordable entry-level custom LEGO-style product",
            "suggested_price_range_usd": {
                "min": 49,
                "max": 99
            },
            "recommended_for": [
                "simple portraits",
                "small gifts",
                "lower-cost customer option",
                "quick production"
            ],
            "notes": [
                "This tier is suitable for smaller models with fewer parts.",
                "Quality is acceptable, but detail level is naturally lower than Standard or Premium."
            ]
        }

    if optimized_part_count <= 4000:
        return {
            "tier": "Standard",
            "price_level": "Medium",
            "cost_level": "Medium",
            "quality_level": "High",
            "market_position": "Balanced custom LEGO-style product with strong detail and reasonable cost",
            "suggested_price_range_usd": {
                "min": 100,
                "max": 179
            },
            "recommended_for": [
                "good-quality portraits",
                "balanced price/detail products",
                "most customers",
                "standard catalog offering"
            ],
            "notes": [
                "This tier gives a strong balance between visual detail and part cost.",
                "Recommended as the default commercial option when quality and price both matter."
            ]
        }

    if optimized_part_count <= 8000:
        return {
            "tier": "Premium",
            "price_level": "High",
            "cost_level": "High",
            "quality_level": "Very High",
            "market_position": "High-detail custom LEGO-style display product",
            "suggested_price_range_usd": {
                "min": 180,
                "max": 349
            },
            "recommended_for": [
                "high-detail pet portraits",
                "premium gifts",
                "large display models",
                "customers who care about accuracy"
            ],
            "notes": [
                "This tier preserves strong visual accuracy and detail.",
                "Part count is higher, but the model quality is better.",
                "This is suitable when image quality should not be compromised."
            ]
        }

    return {
        "tier": "Ultra / Manual Review",
        "price_level": "Very High",
        "cost_level": "Very High",
        "quality_level": "Maximum",
        "market_position": "Large advanced custom LEGO-style model requiring manual review",
        "suggested_price_range_usd": {
            "min": 350,
            "max": 700
        },
        "recommended_for": [
            "large premium commissions",
            "complex images",
            "collector display models",
            "manual designer review"
        ],
        "notes": [
            "This model is very large and may be expensive to produce.",
            "Manual review is recommended before selling.",
            "Do not automatically reduce quality unless the customer requests a cheaper version."
        ]
    }


def create_quality_preservation_report(original_placement_count, optimized_placements, reduction_percent):
    """
    Explains how the optimizer reduced part count without damaging image quality.
    """
    optimized_part_count = len(optimized_placements)

    return {
        "quality_preserved": True,
        "optimization_method": "lossless_grid_merge",
        "description": "The optimizer only combines adjacent cells when they have the same LEGO color and same height_plates value.",
        "does_not_change": [
            "image resolution",
            "color mapping",
            "height/depth values",
            "pixel/stud positions",
            "important visual details"
        ],
        "what_changed": [
            "multiple adjacent matching 1x1 plates can become larger plates",
            "part count is reduced where safe",
            "build becomes easier and cheaper without changing the visual grid"
        ],
        "original_placement_count": original_placement_count,
        "optimized_part_count": optimized_part_count,
        "reduction_percent": reduction_percent,
        "warning": "Further reduction should be done only by generating separate Basic or Standard versions from the first server, not by damaging this high-quality version."
    }


def create_commercial_summary(product_tier, optimized_part_count, original_placement_count, reduction_percent):
    return {
        "tier": product_tier.get("tier"),
        "recommended_selling_position": product_tier.get("market_position"),
        "quality_level": product_tier.get("quality_level"),
        "price_level": product_tier.get("price_level"),
        "cost_level": product_tier.get("cost_level"),
        "optimized_part_count": optimized_part_count,
        "original_placement_count": original_placement_count,
        "part_reduction_percent": reduction_percent,
        "suggested_price_range_usd": product_tier.get("suggested_price_range_usd"),
        "production_note": "This classification is based on optimized part count. It does not reduce model quality.",
        "recommended_next_action": "Use this version as the quality-preserved product. Create cheaper Basic/Standard variants separately only if the customer needs a lower price."
    }


def create_box_mesh(center, extents, rgba):
    mesh = trimesh.creation.box(extents=extents)

    transform = np.eye(4)
    transform[:3, 3] = center
    mesh.apply_transform(transform)

    mesh.visual.face_colors = rgba

    return mesh


def generate_glb_from_optimized_geometry(geometry, optimized_placements):
    width = int(geometry.get("width", 0))
    height = int(geometry.get("height", 0))

    if width <= 0 or height <= 0:
        raise ValueError("image_geometry.width and image_geometry.height are required.")

    if not isinstance(optimized_placements, list) or not optimized_placements:
        raise ValueError("optimized_placements[] is required.")

    model_w = width * STUD_W
    model_d = height * STUD_D

    meshes = []

    # Base plate
    base_mesh = create_box_mesh(
        center=[0, -0.12, 0],
        extents=[model_w + 2.0, 0.22, model_d + 2.0],
        rgba=np.array([230, 230, 224, 255], dtype=np.uint8)
    )
    meshes.append(base_mesh)

    # Thin green bottom border
    border_mesh = create_box_mesh(
        center=[0, -0.30, 0],
        extents=[model_w + 2.2, 0.14, model_d + 2.2],
        rgba=np.array([35, 122, 61, 255], dtype=np.uint8)
    )
    meshes.append(border_mesh)

    for p in optimized_placements:
        x = int(p.get("x", 0))
        y = int(p.get("y", 0))
        w = int(p.get("w", 1))
        h = int(p.get("h", 1))

        height_plates = int(p.get("height_plates", 1))
        height_plates = max(1, min(12, height_plates))

        block_h = height_plates * PLATE_H

        # Center model around origin.
        world_x = ((x + (w / 2)) * STUD_W) - (model_w / 2)
        world_z = ((y + (h / 2)) * STUD_D) - (model_d / 2)
        world_y = block_h / 2

        color_hex = p.get("rebrickable_color_rgb")

        if color_hex:
            rgba = hex_to_rgba(color_hex)
        else:
            rgba = source_rgb_to_rgba(p.get("source_rgb", {}))

        block_mesh = create_box_mesh(
            center=[world_x, world_y, world_z],
            extents=[
                (w * STUD_W) - BLOCK_GAP,
                block_h,
                (h * STUD_D) - BLOCK_GAP
            ],
            rgba=rgba
        )

        meshes.append(block_mesh)

    combined = trimesh.util.concatenate(meshes)

    scene = trimesh.Scene()
    scene.add_geometry(combined, node_name="optimized_lego_relief_model")

    glb_bytes = scene.export(file_type="glb")

    return {
        "glb_bytes": glb_bytes,
        "width": width,
        "height": height,
        "placement_count": len(optimized_placements)
    }


@app.post("/generate-glb")
async def generate_glb(data: dict):
    job_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        geometry = extract_geometry(data)

        if not geometry:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Missing image_geometry. Send either { image_geometry: {...} } or direct { width, height, placements }."
                }
            )

        width = int(geometry.get("width", 0))
        height = int(geometry.get("height", 0))
        original_placements = geometry.get("placements", [])

        if width <= 0 or height <= 0:
            raise ValueError("image_geometry.width and image_geometry.height are required.")

        if not isinstance(original_placements, list) or not original_placements:
            raise ValueError("image_geometry.placements[] is required.")

        original_placement_count_before_truncation = len(original_placements)

        if len(original_placements) > MAX_PLACEMENTS:
            geometry = dict(geometry)
            geometry["placements"] = original_placements[:MAX_PLACEMENTS]
            original_placements = geometry["placements"]

        original_placement_count = len(original_placements)

        optimized_placements = optimize_image_geometry(geometry)

        parts_summary = create_parts_summary(optimized_placements)
        rebrickable_parts_export = create_rebrickable_parts_export(parts_summary)
        basic_parts_xml_export = create_basic_xml_export(parts_summary)
        build_layers = create_build_layers(optimized_placements)

        optimized_part_count = len(optimized_placements)
        reduction_count = original_placement_count - optimized_part_count
        reduction_percent = round(
            (reduction_count / original_placement_count) * 100,
            2
        ) if original_placement_count else 0

        product_tier = classify_product_tier(
            optimized_part_count=optimized_part_count,
            original_stud_count=width * height,
            reduction_percent=reduction_percent
        )

        quality_preservation_report = create_quality_preservation_report(
            original_placement_count=original_placement_count,
            optimized_placements=optimized_placements,
            reduction_percent=reduction_percent
        )

        commercial_summary = create_commercial_summary(
            product_tier=product_tier,
            optimized_part_count=optimized_part_count,
            original_placement_count=original_placement_count,
            reduction_percent=reduction_percent
        )

        glb_result = generate_glb_from_optimized_geometry(
            geometry=geometry,
            optimized_placements=optimized_placements
        )

        glb_base64 = base64.b64encode(glb_result["glb_bytes"]).decode("utf-8")
        generation_time_seconds = round(time.time() - start_time, 2)

        return {
            "success": True,
            "message": "GLB model, optimized LEGO parts, and product tier generated successfully",
            "job_id": job_id,

            "glb": {
                "filename": f"lego_model_{job_id}.glb",
                "mimeType": "model/gltf-binary",
                "glb_base64": glb_base64
            },

            # Backward compatibility for your existing n8n Code node
            "filename": f"lego_model_{job_id}.glb",
            "mimeType": "model/gltf-binary",
            "glb_base64": glb_base64,

            "lego_model": {
                "type": "optimized_mosaic_relief_lego_model",
                "width": width,
                "height": height,
                "original_stud_count": width * height,
                "original_placement_count": original_placement_count,
                "original_placement_count_before_truncation": original_placement_count_before_truncation,
                "optimized_part_count": optimized_part_count,
                "reduction_count": reduction_count,
                "reduction_percent": reduction_percent,

                "product_tier": product_tier,
                "commercial_summary": commercial_summary,
                "quality_preservation_report": quality_preservation_report,

                "optimized_placements": optimized_placements,
                "parts_summary": parts_summary,
                "rebrickable_parts_export": rebrickable_parts_export,
                "basic_parts_xml_export": basic_parts_xml_export,
                "build_layers": build_layers
            },

            # Convenience fields for n8n
            "product_tier": product_tier,
            "commercial_summary": commercial_summary,
            "quality_preservation_report": quality_preservation_report,

            "width": width,
            "height": height,
            "placement_count": optimized_part_count,
            "original_placement_count": original_placement_count,
            "truncated": original_placement_count_before_truncation > original_placement_count,
            "generation_time_seconds": generation_time_seconds
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "generation_time_seconds": round(time.time() - start_time, 2)
            }
        )
