from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import json
import time
import uuid
import numpy as np
import trimesh
from collections import defaultdict
from typing import Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# LEGO-style dimensions
# ---------------------------------------------------------------------
STUD_W = 0.8
STUD_D = 0.8
PLATE_H = 0.24
BLOCK_GAP = 0.025

# ---------------------------------------------------------------------
# MVP quality mode
# ---------------------------------------------------------------------
# For MVP, we do NOT merge pixels into larger parts.
# Every Server 1 placement becomes one visible 1x1 LEGO-style part.
# This preserves image detail and avoids quality loss.
MVP_HIGH_QUALITY_MODE = True

# Higher limit for MVP quality mode.
# You can reduce this later for production.
MAX_PLACEMENTS = 100000

# ---------------------------------------------------------------------
# Practical support strategy
# ---------------------------------------------------------------------
# The visual GLB keeps the full relief height, but the purchase BOM should
# not count hidden support plates under every single raised tile.
PRACTICAL_SUPPORT_MODE = True
BACKING_SUPPORT_RATIO = 0.10
SELECTIVE_RISER_RATIO = 0.08
EDGE_REINFORCEMENT_RATIO = 0.03
MIN_PRACTICAL_SUPPORT_PARTS = 80
MAX_PRACTICAL_SUPPORT_RATIO_OF_VISIBLE = 0.28
SUPPORT_COLOR_ID = 71
SUPPORT_COLOR_NAME = "Light Bluish Gray"
SUPPORT_COLOR_RGB = "A0A5A9"


# ---------------------------------------------------------------------
# Fallback optimizer catalog
# Used only if Server 1 does not send candidate_parts_catalog.optimizer_ready_parts.
# ---------------------------------------------------------------------
FALLBACK_OPTIMIZER_READY_PARTS = [
    # Surface tiles
    {"part_name": "Tile 1x1", "part_num": "3070b", "official_part_name": "Tile 1 x 1 with Groove", "w": 1, "h": 1, "role": "surface", "optimizer_ready": True},
    {"part_name": "Tile 1x2", "part_num": "3069b", "official_part_name": "Tile 1 x 2 with Groove", "w": 1, "h": 2, "role": "surface", "optimizer_ready": True},
    {"part_name": "Tile 1x3", "part_num": "63864", "official_part_name": "Tile 1 x 3", "w": 1, "h": 3, "role": "surface", "optimizer_ready": True},
    {"part_name": "Tile 1x4", "part_num": "2431", "official_part_name": "Tile 1 x 4 with Groove", "w": 1, "h": 4, "role": "surface", "optimizer_ready": True},
    {"part_name": "Tile 1x6", "part_num": "6636", "official_part_name": "Tile 1 x 6 with Groove", "w": 1, "h": 6, "role": "surface", "optimizer_ready": True},
    {"part_name": "Tile 1x8", "part_num": "4162", "official_part_name": "Tile 1 x 8 with Groove", "w": 1, "h": 8, "role": "surface", "optimizer_ready": True},
    {"part_name": "Tile 2x2", "part_num": "3068b", "official_part_name": "Tile 2 x 2 with Groove", "w": 2, "h": 2, "role": "surface", "optimizer_ready": True},
    {"part_name": "Tile 2x3", "part_num": "26603", "official_part_name": "Tile 2 x 3", "w": 2, "h": 3, "role": "surface", "optimizer_ready": True},
    {"part_name": "Tile 2x4", "part_num": "87079", "official_part_name": "Tile 2 x 4 with Groove", "w": 2, "h": 4, "role": "surface", "optimizer_ready": True},

    # Detail parts
    {"part_name": "Tile Round 1x1", "part_num": "98138", "official_part_name": "Tile Round 1 x 1", "w": 1, "h": 1, "role": "detail", "optimizer_ready": True},
    {"part_name": "Tile Round 1x1 Half Circle", "part_num": "24246", "official_part_name": "Tile Round 1 x 1 Half Circle", "w": 1, "h": 1, "role": "detail", "optimizer_ready": True},
    {"part_name": "Tile Round 1x1 Quarter", "part_num": "25269", "official_part_name": "Tile Round 1 x 1 Quarter", "w": 1, "h": 1, "role": "detail", "optimizer_ready": True},
    {"part_name": "Tile Round 2x2", "part_num": "14769", "official_part_name": "Tile Round 2 x 2 with Bottom Stud Holder", "w": 2, "h": 2, "role": "detail", "optimizer_ready": True},
    {"part_name": "Plate Round 1x1 Open Stud", "part_num": "85861", "official_part_name": "Plate Round 1 x 1 with Open Stud", "w": 1, "h": 1, "role": "detail", "optimizer_ready": True},
    {"part_name": "Plate Round 1x1 Solid Stud", "part_num": "6141", "official_part_name": "Plate Round 1 x 1 with Solid Stud", "w": 1, "h": 1, "role": "detail", "optimizer_ready": True},

    # Structure/support plates
    {"part_name": "Plate 1x1", "part_num": "3024", "official_part_name": "Plate 1 x 1", "w": 1, "h": 1, "role": "structure", "optimizer_ready": True},
    {"part_name": "Plate 1x2", "part_num": "3023", "official_part_name": "Plate 1 x 2", "w": 1, "h": 2, "role": "structure", "optimizer_ready": True},
    {"part_name": "Plate 1x3", "part_num": "3623", "official_part_name": "Plate 1 x 3", "w": 1, "h": 3, "role": "structure", "optimizer_ready": True},
    {"part_name": "Plate 1x4", "part_num": "3710", "official_part_name": "Plate 1 x 4", "w": 1, "h": 4, "role": "structure", "optimizer_ready": True},
    {"part_name": "Plate 2x2", "part_num": "3022", "official_part_name": "Plate 2 x 2", "w": 2, "h": 2, "role": "structure", "optimizer_ready": True},
    {"part_name": "Plate 2x3", "part_num": "3021", "official_part_name": "Plate 2 x 3", "w": 2, "h": 3, "role": "structure", "optimizer_ready": True},
    {"part_name": "Plate 2x4", "part_num": "3020", "official_part_name": "Plate 2 x 4", "w": 2, "h": 4, "role": "structure", "optimizer_ready": True},
]


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "service": "lego-model-generator",
        "message": "Send POST request to /generate-glb with image_geometry and candidate_parts_catalog from Server 1.",
        "uses_blender": False,
        "mvp_high_quality_mode": MVP_HIGH_QUALITY_MODE,
        "output": "glb_base64 + high-quality 1x1 LEGO model + practical BOM + pricing"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "lego-model-generator",
        "mode": "mvp_high_quality_1x1_no_merge" if MVP_HIGH_QUALITY_MODE else "image_geometry_to_glb_using_server_1_optimizer_ready_parts",
        "uses_blender": False,
        "supports_candidate_parts_catalog": True,
        "mvp_high_quality_mode": MVP_HIGH_QUALITY_MODE,
        "pricing_basis": "actual_physical_purchase_part_count_with_practical_support_bom"
    }


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
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


def normalize_role(role):
    role = str(role or "").lower().strip()

    if role in ["surface", "tile", "top"]:
        return "surface"

    if role in ["detail", "round", "highlight"]:
        return "detail"

    if role in ["structure", "support", "plate"]:
        return "structure"

    if role in ["shaping", "slope", "curve"]:
        return "shaping"

    return "other"


def safe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


# ---------------------------------------------------------------------
# Input extraction helpers
# ---------------------------------------------------------------------
def try_parse_json_string(value):
    """
    n8n sometimes sends nested objects as JSON strings.
    This helper converts those strings back into dict/list objects when possible.
    """
    if isinstance(value, str):
        clean = value.strip()
        if not clean:
            return value
        if (clean.startswith("{") and clean.endswith("}")) or (clean.startswith("[") and clean.endswith("]")):
            try:
                return json.loads(clean)
            except Exception:
                return value
    return value


def unwrap_n8n_json(data):
    """
    Handles common n8n wrappers, but keeps this function conservative.
    The recursive extractors below handle deeper wrappers.
    """
    data = try_parse_json_string(data)

    if isinstance(data, list) and data:
        first = try_parse_json_string(data[0])
        if isinstance(first, dict):
            if isinstance(first.get("json"), dict):
                return first["json"]
            return first

    if isinstance(data, dict):
        if isinstance(data.get("json"), dict):
            return data["json"]

    return data


def debug_received_input(data, geometry=None, candidate_parts_catalog=None):
    """
    Adds useful diagnostics to the response so we can see exactly
    how Render/FastAPI received the n8n HTTP Request body.
    """
    root = try_parse_json_string(data)

    debug = {
        "received_root_type": type(root).__name__,
        "received_root_keys": list(root.keys()) if isinstance(root, dict) else [],
        "geometry_detected": geometry is not None,
        "candidate_parts_catalog_detected": candidate_parts_catalog is not None,
        "candidate_parts_catalog_type": type(candidate_parts_catalog).__name__ if candidate_parts_catalog is not None else None,
        "has_direct_image_geometry": isinstance(root, dict) and "image_geometry" in root,
        "has_direct_candidate_parts_catalog": isinstance(root, dict) and "candidate_parts_catalog" in root,
        "has_json_wrapper": isinstance(root, dict) and "json" in root,
        "has_body_wrapper": isinstance(root, dict) and "body" in root,
        "has_data_wrapper": isinstance(root, dict) and "data" in root,
        "mvp_high_quality_mode": MVP_HIGH_QUALITY_MODE,
        "max_placements": MAX_PLACEMENTS,
    }

    if isinstance(candidate_parts_catalog, dict):
        optimizer_ready = candidate_parts_catalog.get("optimizer_ready_parts")
        debug["candidate_parts_catalog_keys"] = list(candidate_parts_catalog.keys())
        debug["optimizer_ready_parts_count_in_input"] = len(optimizer_ready) if isinstance(optimizer_ready, list) else 0
    else:
        debug["candidate_parts_catalog_keys"] = []
        debug["optimizer_ready_parts_count_in_input"] = 0

    return debug


def extract_geometry(data):
    """
    Extracts image_geometry from many possible n8n/body formats.

    Supported examples:
    { "image_geometry": {...} }
    { "json": { "image_geometry": {...} } }
    [ { "json": { "image_geometry": {...} } } ]
    { "data": [ { "image_geometry": {...} } ] }
    { "body": { "image_geometry": {...} } }
    Direct geometry: { "width": 160, "height": 90, "placements": [...] }
    Also supports image_geometry accidentally sent as a JSON string.
    """
    data = try_parse_json_string(data)

    # n8n array wrapper
    if isinstance(data, list) and data:
        found = extract_geometry(data[0])
        if found:
            return found
        return None

    if not isinstance(data, dict):
        return None

    # Direct wrapped image_geometry
    geometry = try_parse_json_string(data.get("image_geometry"))
    if isinstance(geometry, dict):
        if geometry.get("width") and geometry.get("height") and isinstance(geometry.get("placements"), list):
            return geometry

    # Direct geometry
    if data.get("width") and data.get("height") and isinstance(data.get("placements"), list):
        return data

    # Common wrappers
    for key in ["json", "body", "payload"]:
        if key in data:
            found = extract_geometry(data.get(key))
            if found:
                return found

    # data can be dict or list
    if "data" in data:
        found = extract_geometry(data.get("data"))
        if found:
            return found

    return None


def extract_candidate_parts_catalog(data):
    """
    Extracts candidate_parts_catalog from many possible n8n/body formats.

    Supported examples:
    { "candidate_parts_catalog": {...} }
    { "json": { "candidate_parts_catalog": {...} } }
    [ { "json": { "candidate_parts_catalog": {...} } } ]
    { "data": [ { "candidate_parts_catalog": {...} } ] }
    { "body": { "candidate_parts_catalog": {...} } }
    Also supports candidate_parts_catalog accidentally sent as a JSON string.
    """
    data = try_parse_json_string(data)

    # n8n array wrapper
    if isinstance(data, list) and data:
        found = extract_candidate_parts_catalog(data[0])
        if found:
            return found
        return None

    if not isinstance(data, dict):
        return None

    # Direct case
    catalog = try_parse_json_string(data.get("candidate_parts_catalog"))
    if isinstance(catalog, dict):
        return catalog

    # Some nodes may pass it under different names
    catalog = try_parse_json_string(data.get("parts_catalog"))
    if isinstance(catalog, dict):
        return catalog

    # Common wrappers
    for key in ["json", "body", "payload"]:
        if key in data:
            found = extract_candidate_parts_catalog(data.get(key))
            if found:
                return found

    # data can be dict or list
    if "data" in data:
        found = extract_candidate_parts_catalog(data.get("data"))
        if found:
            return found

    return None


def get_optimizer_ready_parts(candidate_parts_catalog):
    """
    Returns optimizer-ready parts from Server 1 candidate_parts_catalog.
    Falls back to safe default catalog if catalog is missing or invalid.
    """
    if not isinstance(candidate_parts_catalog, dict):
        return FALLBACK_OPTIMIZER_READY_PARTS

    parts = candidate_parts_catalog.get("optimizer_ready_parts")

    if isinstance(parts, list) and parts:
        clean_parts = []

        for p in parts:
            if not isinstance(p, dict):
                continue

            if p.get("optimizer_ready") is False:
                continue

            if p.get("product_safe") is False:
                continue

            w = safe_int(p.get("w"))
            h = safe_int(p.get("h"))

            if not w or not h:
                continue

            if w <= 0 or h <= 0:
                continue

            # Avoid very large parts in automatic optimizer.
            if w > 8 or h > 8:
                continue

            clean = dict(p)
            clean["w"] = w
            clean["h"] = h
            clean["role"] = normalize_role(clean.get("role"))
            clean_parts.append(clean)

        if clean_parts:
            return clean_parts

    return FALLBACK_OPTIMIZER_READY_PARTS


# ---------------------------------------------------------------------
# Dynamic optimizer catalog
# ---------------------------------------------------------------------
def build_optimizer_catalog(candidate_parts_catalog):
    """
    Builds a dynamic part catalog from Server 1 optimizer_ready_parts.

    In MVP_HIGH_QUALITY_MODE, this catalog is still used to choose the
    correct 1x1 visible part and 1x1 support part, but no grid merging happens.
    """
    optimizer_ready_parts = get_optimizer_ready_parts(candidate_parts_catalog)

    by_size_role = defaultdict(list)
    by_size_any = defaultdict(list)

    for p in optimizer_ready_parts:
        w = safe_int(p.get("w"))
        h = safe_int(p.get("h"))
        role = normalize_role(p.get("role"))

        if not w or not h:
            continue

        record = {
            "part_name": p.get("part_name") or p.get("official_part_name") or "Unknown Part",
            "part_num": str(p.get("part_num")),
            "official_part_name": p.get("official_part_name") or p.get("part_name"),
            "part_category": p.get("part_category"),
            "w": w,
            "h": h,
            "role": role,
            "source": p.get("source", "candidate_parts_catalog"),
            "preferred": bool(p.get("preferred", True)),
            "optimizer_ready": True,
        }

        key = (min(w, h), max(w, h))
        by_size_role[(key[0], key[1], role)].append(record)
        by_size_any[key].append(record)

    shapes = sorted(
        by_size_any.keys(),
        key=lambda size: (size[0] * size[1], max(size), min(size)),
        reverse=True
    )

    # Always ensure 1x1 exists as fallback.
    if (1, 1) not in by_size_any:
        fallback = {
            "part_name": "Plate 1x1",
            "part_num": "3024",
            "official_part_name": "Plate 1 x 1",
            "part_category": "Plates",
            "w": 1,
            "h": 1,
            "role": "structure",
            "source": "fallback",
            "preferred": True,
            "optimizer_ready": True
        }
        by_size_any[(1, 1)].append(fallback)
        by_size_role[(1, 1, "structure")].append(fallback)
        if (1, 1) not in shapes:
            shapes.append((1, 1))

    return {
        "optimizer_ready_parts": optimizer_ready_parts,
        "by_size_role": by_size_role,
        "by_size_any": by_size_any,
        "shapes": shapes,
        "source": "candidate_parts_catalog.optimizer_ready_parts" if candidate_parts_catalog else "fallback_catalog"
    }


def find_part_for_size(catalog, w, h, preferred_roles):
    """
    Finds the best part for a footprint.

    For a 4x2 area, it can use a 2x4 part with orientation 90.
    """
    key = (min(w, h), max(w, h))
    orientation = 0 if (w, h) == key else 90

    for role in preferred_roles:
        candidates = catalog["by_size_role"].get((key[0], key[1], role), [])
        if candidates:
            return candidates[0], orientation

    candidates = catalog["by_size_any"].get(key, [])
    if candidates:
        return candidates[0], orientation

    fallback = catalog["by_size_any"].get((1, 1), [FALLBACK_OPTIMIZER_READY_PARTS[-7]])[0]
    return fallback, 0


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
    Only cells with the same signature can be combined.

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


def create_high_quality_1x1_geometry(geometry, candidate_parts_catalog=None):
    """
    MVP high-quality mode.

    Every input placement from Server 1 becomes one visible 1x1 part.
    No merging.
    No part optimization.
    No reduction.
    No visual compromise.

    This is heavier, but it preserves the image quality.
    """
    catalog = build_optimizer_catalog(candidate_parts_catalog)

    width = int(geometry.get("width", 0))
    height = int(geometry.get("height", 0))
    placements = geometry.get("placements", [])

    if width <= 0 or height <= 0:
        raise ValueError("image_geometry.width and image_geometry.height are required.")

    if not isinstance(placements, list) or not placements:
        raise ValueError("image_geometry.placements[] is required.")

    optimized = []

    # Prefer 1x1 surface/detail pieces for visible quality.
    visible_part, orientation = find_part_for_size(
        catalog=catalog,
        w=1,
        h=1,
        preferred_roles=["surface", "detail", "structure"]
    )

    support_part, support_orientation = find_part_for_size(
        catalog=catalog,
        w=1,
        h=1,
        preferred_roles=["structure", "surface"]
    )

    for cell in placements:
        x = int(cell.get("x", 0))
        y = int(cell.get("y", 0))

        if not (0 <= x < width and 0 <= y < height):
            continue

        height_plates = max(1, min(12, int(cell.get("height_plates", 1))))

        optimized.append({
            "x": x,
            "y": y,
            "z": int(cell.get("z", 0)),
            "w": 1,
            "h": 1,

            "part_name": visible_part["part_name"],
            "part_num": visible_part["part_num"],
            "official_part_name": visible_part.get("official_part_name"),
            "part_role": visible_part.get("role"),
            "part_category": visible_part.get("part_category"),

            "support_part_name": support_part["part_name"],
            "support_part_num": support_part["part_num"],
            "support_official_part_name": support_part.get("official_part_name"),
            "support_part_role": support_part.get("role"),

            "color": cell.get("color"),
            "rebrickable_color_id": cell.get("rebrickable_color_id"),
            "rebrickable_color_name": cell.get("rebrickable_color_name"),
            "rebrickable_color_rgb": normalize_hex(cell.get("rebrickable_color_rgb")),
            "height_plates": height_plates,
            "orientation": orientation,
            "support_orientation": support_orientation,
            "source_rgb": cell.get("source_rgb", {}),

            "physical_parts_note": "MVP high-quality mode: every original image cell is preserved as one visible 1x1 part."
        })

    return optimized, catalog


def optimize_image_geometry(geometry, candidate_parts_catalog=None):
    """
    Converts raw 1x1 placement grid into larger realistic LEGO placements
    wherever possible.

    Important:
    This optimization is quality-safe, but for MVP it is disabled by
    MVP_HIGH_QUALITY_MODE=True.
    """
    catalog = build_optimizer_catalog(candidate_parts_catalog)

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

    # Try large shapes first.
    base_shapes = catalog["shapes"]

    # Include rotated versions during placement testing.
    test_shapes = []
    seen_shapes = set()

    for w, h in base_shapes:
        for shape in [(w, h), (h, w)]:
            if shape in seen_shapes:
                continue
            seen_shapes.add(shape)
            test_shapes.append(shape)

    test_shapes = sorted(
        test_shapes,
        key=lambda size: (size[0] * size[1], max(size), min(size)),
        reverse=True
    )

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

            for shape_w, shape_h in test_shapes:
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

            height_plates = max(1, min(12, int(cell.get("height_plates", 1))))

            # Visible part:
            # Prefer smooth surface tiles. Use detail pieces only for 1x1/2x2 detail footprints.
            if chosen_w * chosen_h <= 1:
                preferred_roles = ["detail", "surface", "structure"]
            else:
                preferred_roles = ["surface", "structure", "detail"]

            visible_part, orientation = find_part_for_size(
                catalog=catalog,
                w=chosen_w,
                h=chosen_h,
                preferred_roles=preferred_roles
            )

            support_part, support_orientation = find_part_for_size(
                catalog=catalog,
                w=chosen_w,
                h=chosen_h,
                preferred_roles=["structure", "surface"]
            )

            optimized.append({
                "x": x,
                "y": y,
                "z": int(cell.get("z", 0)),
                "w": chosen_w,
                "h": chosen_h,

                "part_name": visible_part["part_name"],
                "part_num": visible_part["part_num"],
                "official_part_name": visible_part.get("official_part_name"),
                "part_role": visible_part.get("role"),
                "part_category": visible_part.get("part_category"),

                "support_part_name": support_part["part_name"],
                "support_part_num": support_part["part_num"],
                "support_official_part_name": support_part.get("official_part_name"),
                "support_part_role": support_part.get("role"),

                "color": cell.get("color"),
                "rebrickable_color_id": cell.get("rebrickable_color_id"),
                "rebrickable_color_name": cell.get("rebrickable_color_name"),
                "rebrickable_color_rgb": normalize_hex(cell.get("rebrickable_color_rgb")),
                "height_plates": height_plates,
                "orientation": orientation,
                "support_orientation": support_orientation,
                "source_rgb": cell.get("source_rgb", {}),

                "physical_parts_note": "Physical purchase list counts this as a visible top piece plus support plates underneath when height_plates > 1."
            })

    return optimized, catalog


# ---------------------------------------------------------------------
# Parts list creation
# ---------------------------------------------------------------------
def add_purchase_part(agg, part_num, part_name, color_id, color_name, color_rgb, quantity, role, source_height_plates=None):
    if not part_num or quantity <= 0:
        return

    key = f"{part_num}-{color_id}-{role}"

    if key not in agg:
        agg[key] = {
            "part_num": part_num,
            "part_name": part_name,
            "color_id": color_id,
            "color_name": color_name,
            "color_rgb": color_rgb,
            "quantity": 0,
            "role": role,
            "source_height_plates": []
        }

    agg[key]["quantity"] += int(quantity)

    if source_height_plates is not None and source_height_plates not in agg[key]["source_height_plates"]:
        agg[key]["source_height_plates"].append(source_height_plates)


def create_instruction_parts_summary(optimized_placements):
    """
    Instruction-focused parts grouped by part_num + color + height_plates.
    This preserves the relief-height information for build instructions.
    """
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
                "quantity": 0,
                "list_type": "instruction_parts_by_height"
            }

        summary[key]["quantity"] += 1

    return summary


def get_best_support_part(optimized_placements):
    """
    Finds a realistic structure plate already selected by the optimizer.
    Prefer larger plates because they reduce hidden support quantity.

    In MVP_HIGH_QUALITY_MODE, placements are 1x1, so this will usually
    return a 1x1 support part.
    """
    candidates = {}

    for p in optimized_placements:
        part_num = p.get("support_part_num") or p.get("part_num")
        part_name = p.get("support_part_name") or p.get("part_name")
        w = safe_int(p.get("w"), 1) or 1
        h = safe_int(p.get("h"), 1) or 1

        if not part_num:
            continue

        key = str(part_num)
        area = max(1, w * h)

        if key not in candidates or area > candidates[key]["area"]:
            candidates[key] = {
                "part_num": key,
                "part_name": part_name or "Support Plate",
                "area": area,
                "w": w,
                "h": h,
            }

    if candidates:
        preferred_order = ["3020", "3021", "3022", "3710", "3623", "3023", "3024"]
        for part_num in preferred_order:
            if part_num in candidates:
                return candidates[part_num]

        return sorted(candidates.values(), key=lambda x: x["area"], reverse=True)[0]

    return {
        "part_num": "3024",
        "part_name": "Plate 1x1",
        "area": 1,
        "w": 1,
        "h": 1,
    }


def calculate_practical_support_quantities(optimized_placements):
    """
    Calculates a practical hidden support pack instead of stacking plates
    under every raised visible piece.
    """
    visible_count = len(optimized_placements)

    if visible_count <= 0:
        return {
            "backing_support_qty": 0,
            "selective_riser_qty": 0,
            "edge_reinforcement_qty": 0,
            "total_practical_support_qty": 0,
            "estimated_solid_stack_support_qty_avoided": 0,
            "support_reduction_percent_vs_solid_stack": 0,
        }

    heights = [max(1, min(12, int(p.get("height_plates", 1)))) for p in optimized_placements]
    sorted_heights = sorted(heights)
    median_height = sorted_heights[len(sorted_heights) // 2]

    elevated_count = sum(1 for h in heights if h > median_height)
    high_detail_count = sum(1 for h in heights if h >= median_height + 2)

    estimated_solid_stack_support_qty = sum(max(0, h - 1) for h in heights)

    backing_support_qty = int(round(visible_count * BACKING_SUPPORT_RATIO))
    selective_riser_qty = int(round((elevated_count + high_detail_count) * SELECTIVE_RISER_RATIO))
    edge_reinforcement_qty = int(round(visible_count * EDGE_REINFORCEMENT_RATIO))

    total = backing_support_qty + selective_riser_qty + edge_reinforcement_qty

    min_support = min(MIN_PRACTICAL_SUPPORT_PARTS, max(0, visible_count))
    max_support = int(round(visible_count * MAX_PRACTICAL_SUPPORT_RATIO_OF_VISIBLE))

    if total < min_support:
        total = min_support

    if max_support > 0 and total > max_support:
        total = max_support

    backing_support_qty = int(round(total * 0.55))
    selective_riser_qty = int(round(total * 0.30))
    edge_reinforcement_qty = max(0, total - backing_support_qty - selective_riser_qty)

    avoided = max(0, estimated_solid_stack_support_qty - total)
    reduction_percent = round((avoided / estimated_solid_stack_support_qty) * 100, 2) if estimated_solid_stack_support_qty else 0

    return {
        "backing_support_qty": backing_support_qty,
        "selective_riser_qty": selective_riser_qty,
        "edge_reinforcement_qty": edge_reinforcement_qty,
        "total_practical_support_qty": total,
        "median_height_plates": median_height,
        "elevated_visual_block_count": elevated_count,
        "high_detail_visual_block_count": high_detail_count,
        "estimated_solid_stack_support_qty_avoided": avoided,
        "support_reduction_percent_vs_solid_stack": reduction_percent,
    }


def create_purchase_parts_summary(optimized_placements):
    """
    Purchase-focused parts grouped by part_num + color_id.

    Practical BOM logic:
    - Count every visible top tile/part exactly once.
    - Do NOT stack hidden support plates under every raised tile.
    - Add a neutral practical support pack for backing, selective risers, and edge reinforcement.
    """
    agg = {}

    for p in optimized_placements:
        color_id = p.get("rebrickable_color_id")
        color_name = p.get("rebrickable_color_name") or p.get("color") or "Unknown"
        color_rgb = p.get("rebrickable_color_rgb")
        height_plates = max(1, min(12, int(p.get("height_plates", 1))))

        add_purchase_part(
            agg=agg,
            part_num=p.get("part_num"),
            part_name=p.get("part_name"),
            color_id=color_id,
            color_name=color_name,
            color_rgb=color_rgb,
            quantity=1,
            role="visible_surface",
            source_height_plates=height_plates
        )

    support_part = get_best_support_part(optimized_placements)
    support_quantities = calculate_practical_support_quantities(optimized_placements)

    if PRACTICAL_SUPPORT_MODE and support_quantities["total_practical_support_qty"] > 0:
        add_purchase_part(
            agg=agg,
            part_num=support_part["part_num"],
            part_name=support_part["part_name"],
            color_id=SUPPORT_COLOR_ID,
            color_name=SUPPORT_COLOR_NAME,
            color_rgb=SUPPORT_COLOR_RGB,
            quantity=support_quantities["backing_support_qty"],
            role="practical_backing_support",
            source_height_plates="mixed"
        )

        add_purchase_part(
            agg=agg,
            part_num=support_part["part_num"],
            part_name=support_part["part_name"],
            color_id=SUPPORT_COLOR_ID,
            color_name=SUPPORT_COLOR_NAME,
            color_rgb=SUPPORT_COLOR_RGB,
            quantity=support_quantities["selective_riser_qty"],
            role="selective_riser_support",
            source_height_plates="mixed"
        )

        add_purchase_part(
            agg=agg,
            part_num=support_part["part_num"],
            part_name=support_part["part_name"],
            color_id=SUPPORT_COLOR_ID,
            color_name=SUPPORT_COLOR_NAME,
            color_rgb=SUPPORT_COLOR_RGB,
            quantity=support_quantities["edge_reinforcement_qty"],
            role="edge_reinforcement_support",
            source_height_plates="mixed"
        )

    return agg


def create_support_strategy_summary(optimized_placements, purchase_parts_summary):
    visible_count = len(optimized_placements)
    support_quantities = calculate_practical_support_quantities(optimized_placements)
    physical_part_count = sum(int(item.get("quantity", 0)) for item in purchase_parts_summary.values())
    visible_top_piece_count = sum(int(item.get("quantity", 0)) for item in purchase_parts_summary.values() if item.get("role") == "visible_surface")
    support_piece_count = max(0, physical_part_count - visible_top_piece_count)

    return {
        "support_strategy": "practical_backing_and_selective_risers",
        "visual_quality_preserved": True,
        "glb_relief_height_preserved": True,
        "mvp_high_quality_mode": MVP_HIGH_QUALITY_MODE,
        "bom_support_mode": "practical_not_solid_stack",
        "visible_top_piece_count": visible_top_piece_count,
        "practical_support_piece_count": support_piece_count,
        "total_physical_purchase_part_count": physical_part_count,
        "old_solid_stack_support_qty_avoided": support_quantities.get("estimated_solid_stack_support_qty_avoided"),
        "support_reduction_percent_vs_old_solid_stack": support_quantities.get("support_reduction_percent_vs_solid_stack"),
        "backing_support_qty": support_quantities.get("backing_support_qty"),
        "selective_riser_qty": support_quantities.get("selective_riser_qty"),
        "edge_reinforcement_qty": support_quantities.get("edge_reinforcement_qty"),
        "explanation": "The visible design is unchanged. The BOM counts one visible top piece per visual placement plus a neutral backing/riser support pack, instead of stacking hidden plates under every raised tile.",
        "production_note": "For physical production, use a rigid backing/base and selective LEGO support pieces. This keeps the product buildable without creating an unrealistic solid-stack kit."
    }


def create_rebrickable_parts_export(purchase_parts_summary):
    rows = []

    for item in purchase_parts_summary.values():
        if item.get("part_num") and item.get("color_id") is not None:
            rows.append({
                "part_num": item["part_num"],
                "color_id": item["color_id"],
                "color_name": item["color_name"],
                "quantity": item["quantity"],
                "role": item.get("role")
            })

    rows.sort(key=lambda r: (str(r["part_num"]), int(r["color_id"]) if str(r["color_id"]).isdigit() else 9999))
    return rows


def create_basic_xml_export(purchase_parts_summary):
    """
    Generic XML export using purchase-focused real physical part quantities.
    """
    xml = "<INVENTORY>\n"

    for item in purchase_parts_summary.values():
        part_num = item.get("part_num")
        color_id = item.get("color_id")

        if not part_num or color_id is None:
            continue

        xml += "  <ITEM>\n"
        xml += f"    <PARTNUM>{part_num}</PARTNUM>\n"
        xml += f"    <COLORID>{color_id}</COLORID>\n"
        xml += f"    <COLORNAME>{item.get('color_name')}</COLORNAME>\n"
        xml += f"    <QTY>{item.get('quantity')}</QTY>\n"
        xml += "  </ITEM>\n"

    xml += "</INVENTORY>"
    return xml


def create_build_layers(optimized_placements):
    """
    Simple build sequence:
    group by height_plates.

    This is structured data for future instructions.
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


# ---------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------
def calculate_price_from_part_count(
    physical_part_count,
    min_parts,
    max_parts,
    min_price,
    max_price
):
    """
    Calculates one recommended price inside the tier range.
    Pricing is based on actual physical purchase part count.
    """
    if physical_part_count <= min_parts:
        return min_price

    if physical_part_count >= max_parts:
        return max_price

    position = (physical_part_count - min_parts) / (max_parts - min_parts)
    price = min_price + (position * (max_price - min_price))

    rounded = int(round(price / 10.0) * 10)

    if rounded <= 50:
        return max(min_price, rounded - 1)

    return max(min_price, min(max_price, rounded - 1))


def classify_product_tier(physical_part_count, optimized_visual_block_count, original_stud_count, reduction_percent):
    """
    Classifies price tier based on actual physical purchase part count.
    """
    if physical_part_count <= 1500:
        min_parts = 1
        max_parts = 1500
        min_price = 49
        max_price = 149

        recommended_price = calculate_price_from_part_count(
            physical_part_count,
            min_parts,
            max_parts,
            min_price,
            max_price
        )

        return {
            "tier": "Basic",
            "tier_basis": "actual_physical_purchase_part_count",
            "part_count_range": {"min": min_parts, "max": max_parts},
            "physical_part_count": physical_part_count,
            "optimized_visual_block_count": optimized_visual_block_count,
            "suggested_price_range_usd": {"min": min_price, "max": max_price},
            "recommended_price_usd": recommended_price,
            "price_explanation": f"Price is calculated from {physical_part_count} actual physical parts inside the Basic range.",
            "quality_note": "Quality was not reduced. Tier is based only on final physical part count.",
            "market_position": "Affordable small-size custom LEGO-style product"
        }

    if physical_part_count <= 4000:
        min_parts = 1501
        max_parts = 4000
        min_price = 150
        max_price = 349

        recommended_price = calculate_price_from_part_count(
            physical_part_count,
            min_parts,
            max_parts,
            min_price,
            max_price
        )

        return {
            "tier": "Standard",
            "tier_basis": "actual_physical_purchase_part_count",
            "part_count_range": {"min": min_parts, "max": max_parts},
            "physical_part_count": physical_part_count,
            "optimized_visual_block_count": optimized_visual_block_count,
            "suggested_price_range_usd": {"min": min_price, "max": max_price},
            "recommended_price_usd": recommended_price,
            "price_explanation": f"Price is calculated from {physical_part_count} actual physical parts inside the Standard range.",
            "quality_note": "Quality was not reduced. Tier is based only on final physical part count.",
            "market_position": "Medium-size custom LEGO-style product"
        }

    if physical_part_count <= 8000:
        min_parts = 4001
        max_parts = 8000
        min_price = 350
        max_price = 699

        recommended_price = calculate_price_from_part_count(
            physical_part_count,
            min_parts,
            max_parts,
            min_price,
            max_price
        )

        return {
            "tier": "Premium",
            "tier_basis": "actual_physical_purchase_part_count",
            "part_count_range": {"min": min_parts, "max": max_parts},
            "physical_part_count": physical_part_count,
            "optimized_visual_block_count": optimized_visual_block_count,
            "suggested_price_range_usd": {"min": min_price, "max": max_price},
            "recommended_price_usd": recommended_price,
            "price_explanation": f"Price is calculated from {physical_part_count} actual physical parts inside the Premium range.",
            "quality_note": "Quality was not reduced. Tier is based only on final physical part count.",
            "market_position": "Large custom LEGO-style display product"
        }

    extra_parts = physical_part_count - 8000
    raw_price = 699 + (extra_parts * 0.09)
    recommended_price = int(round(raw_price / 10.0) * 10) - 1

    if recommended_price < 700:
        recommended_price = 700

    manual_quote_recommended = physical_part_count > 18000

    return {
        "tier": "Ultra / Manual Review",
        "tier_basis": "actual_physical_purchase_part_count",
        "part_count_range": {"min": 8001, "max": None},
        "physical_part_count": physical_part_count,
        "optimized_visual_block_count": optimized_visual_block_count,
        "suggested_price_range_usd": {"min": 700, "max": None},
        "recommended_price_usd": recommended_price,
        "price_formula": "$699 + $0.09 per physical part above 8,000",
        "price_explanation": f"Price is calculated from {physical_part_count} actual physical parts. Ultra pricing continues above the Premium range.",
        "manual_quote_recommended": manual_quote_recommended,
        "quality_note": "Quality was not reduced. Tier is based only on final physical part count.",
        "market_position": "Very large custom LEGO-style model"
    }


def create_quality_preservation_report(
    original_placement_count,
    optimized_visual_block_count,
    physical_part_count,
    reduction_percent,
    catalog_source,
    support_strategy_summary=None
):
    """
    Explains how image quality was handled.
    """
    return {
        "quality_preserved": True,
        "mvp_high_quality_mode": MVP_HIGH_QUALITY_MODE,
        "optimization_method": "mvp_high_quality_1x1_no_merge" if MVP_HIGH_QUALITY_MODE else "lossless_grid_merge_with_realistic_part_catalog",
        "catalog_source": catalog_source,
        "description": "MVP mode keeps every original image placement as a separate 1x1 visible part. No merging is applied." if MVP_HIGH_QUALITY_MODE else "The optimizer only combines adjacent cells when they have the same LEGO color and same height_plates value.",
        "does_not_change": [
            "image resolution",
            "color mapping",
            "height/depth values",
            "pixel/stud positions",
            "important visual details"
        ],
        "what_changed": [
            "MVP high-quality mode keeps the original image geometry as 1x1 visible placements",
            "no adjacent cells are merged into larger parts",
            "purchase_parts_list groups real physical parts by part number and color",
            "hidden support is counted using a practical backing/selective-riser strategy instead of full solid stacking"
        ] if MVP_HIGH_QUALITY_MODE else [
            "multiple adjacent matching 1x1 grid cells can become larger footprint parts",
            "Server 2 uses optimizer_ready_parts from Server 1 instead of only hardcoded plates",
            "purchase_parts_list groups real physical parts by part number and color",
            "hidden support is counted using a practical backing/selective-riser strategy instead of full solid stacking"
        ],
        "original_placement_count": original_placement_count,
        "optimized_visual_block_count": optimized_visual_block_count,
        "actual_physical_purchase_part_count": physical_part_count,
        "visual_block_reduction_percent": reduction_percent,
        "support_strategy_summary": support_strategy_summary or {},
        "warning": "For MVP, quality is prioritized over part-count optimization. Production optimization can be enabled later by setting MVP_HIGH_QUALITY_MODE=False."
    }


def create_commercial_summary(product_tier, physical_part_count, optimized_visual_block_count, original_placement_count, reduction_percent):
    return {
        "pricing_basis": "actual_physical_purchase_part_count_with_practical_support_bom",
        "mvp_high_quality_mode": MVP_HIGH_QUALITY_MODE,
        "tier": product_tier.get("tier"),
        "recommended_selling_position": product_tier.get("market_position"),
        "physical_part_count": physical_part_count,
        "optimized_visual_block_count": optimized_visual_block_count,
        "original_placement_count": original_placement_count,
        "visual_block_reduction_percent": reduction_percent,
        "suggested_price_range_usd": product_tier.get("suggested_price_range_usd"),
        "recommended_price_usd": product_tier.get("recommended_price_usd"),
        "price_explanation": product_tier.get("price_explanation"),
        "quality_note": "Basic, Standard, Premium, and Ultra are based on actual physical part count only. They do not mean different generation quality.",
        "production_note": "MVP mode keeps every original image placement as a 1x1 visible part. This prioritizes image quality over production part optimization.",
        "recommended_next_action": "Use this for MVP visual validation. Later, enable optimization for production by setting MVP_HIGH_QUALITY_MODE=False."
    }


# ---------------------------------------------------------------------
# GLB generation
# ---------------------------------------------------------------------
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

    # Thin bottom border
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
    scene.add_geometry(combined, node_name="lego_relief_model")

    glb_bytes = scene.export(file_type="glb")

    return {
        "glb_bytes": glb_bytes,
        "width": width,
        "height": height,
        "placement_count": len(optimized_placements)
    }


# ---------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------
@app.post("/generate-glb")
async def generate_glb(data: Any = Body(...)):
    job_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        geometry = extract_geometry(data)
        candidate_parts_catalog = extract_candidate_parts_catalog(data)
        debug_input = debug_received_input(
            data=data,
            geometry=geometry,
            candidate_parts_catalog=candidate_parts_catalog
        )

        if not geometry:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Missing image_geometry. Send { image_geometry: {...}, candidate_parts_catalog: {...} }.",
                    "debug_input": debug_input
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

        if MVP_HIGH_QUALITY_MODE:
            optimized_placements, optimizer_catalog = create_high_quality_1x1_geometry(
                geometry=geometry,
                candidate_parts_catalog=candidate_parts_catalog
            )
        else:
            optimized_placements, optimizer_catalog = optimize_image_geometry(
                geometry=geometry,
                candidate_parts_catalog=candidate_parts_catalog
            )

        instruction_parts_summary = create_instruction_parts_summary(optimized_placements)
        purchase_parts_summary = create_purchase_parts_summary(optimized_placements)
        support_strategy_summary = create_support_strategy_summary(optimized_placements, purchase_parts_summary)

        purchase_parts_list = list(purchase_parts_summary.values())
        purchase_parts_list.sort(
            key=lambda x: (
                str(x.get("part_num")),
                int(x.get("color_id")) if str(x.get("color_id")).isdigit() else 9999,
                str(x.get("role"))
            )
        )

        rebrickable_parts_export = create_rebrickable_parts_export(purchase_parts_summary)
        basic_parts_xml_export = create_basic_xml_export(purchase_parts_summary)
        build_layers = create_build_layers(optimized_placements)

        optimized_visual_block_count = len(optimized_placements)
        physical_part_count = sum(int(item.get("quantity", 0)) for item in purchase_parts_summary.values())

        reduction_count = original_placement_count - optimized_visual_block_count
        reduction_percent = round(
            (reduction_count / original_placement_count) * 100,
            2
        ) if original_placement_count else 0

        product_tier = classify_product_tier(
            physical_part_count=physical_part_count,
            optimized_visual_block_count=optimized_visual_block_count,
            original_stud_count=width * height,
            reduction_percent=reduction_percent
        )

        quality_preservation_report = create_quality_preservation_report(
            original_placement_count=original_placement_count,
            optimized_visual_block_count=optimized_visual_block_count,
            physical_part_count=physical_part_count,
            reduction_percent=reduction_percent,
            catalog_source=optimizer_catalog.get("source"),
            support_strategy_summary=support_strategy_summary
        )

        commercial_summary = create_commercial_summary(
            product_tier=product_tier,
            physical_part_count=physical_part_count,
            optimized_visual_block_count=optimized_visual_block_count,
            original_placement_count=original_placement_count,
            reduction_percent=reduction_percent
        )

        glb_result = generate_glb_from_optimized_geometry(
            geometry=geometry,
            optimized_placements=optimized_placements
        )

        glb_base64 = base64.b64encode(glb_result["glb_bytes"]).decode("utf-8")
        generation_time_seconds = round(time.time() - start_time, 2)

        candidate_catalog_summary = {
            "received_candidate_parts_catalog": candidate_parts_catalog is not None,
            "catalog_source_used": optimizer_catalog.get("source"),
            "optimizer_ready_part_count_used": len(optimizer_catalog.get("optimizer_ready_parts", [])),
            "shape_count_used": len(optimizer_catalog.get("shapes", [])),
            "shapes_used": [
                {"w": w, "h": h}
                for w, h in optimizer_catalog.get("shapes", [])
            ]
        }

        return {
            "success": True,
            "message": "GLB model generated successfully in MVP high-quality mode. Original image placements are preserved as 1x1 visible parts with no visual optimization merge." if MVP_HIGH_QUALITY_MODE else "GLB model, optimized LEGO parts, purchase BOM, instructions, and brick-count pricing generated successfully.",
            "job_id": job_id,

            "glb": {
                "filename": f"lego_model_{job_id}.glb",
                "mimeType": "model/gltf-binary",
                "glb_base64": glb_base64
            },

            # Backward compatibility for existing n8n GLB Code node
            "filename": f"lego_model_{job_id}.glb",
            "mimeType": "model/gltf-binary",
            "glb_base64": glb_base64,

            "lego_model": {
                "type": "mvp_high_quality_1x1_mosaic_relief_lego_model" if MVP_HIGH_QUALITY_MODE else "optimized_mosaic_relief_lego_model",
                "mvp_high_quality_mode": MVP_HIGH_QUALITY_MODE,
                "width": width,
                "height": height,
                "original_stud_count": width * height,
                "original_placement_count": original_placement_count,
                "original_placement_count_before_truncation": original_placement_count_before_truncation,

                "optimized_visual_block_count": optimized_visual_block_count,
                "actual_physical_purchase_part_count": physical_part_count,

                # Backward-compatible field name, now using real physical count
                "optimized_part_count": physical_part_count,

                "reduction_count": reduction_count,
                "reduction_percent": reduction_percent,

                "candidate_catalog_summary": candidate_catalog_summary,
                "product_tier": product_tier,
                "pricing": product_tier,
                "commercial_summary": commercial_summary,
                "quality_preservation_report": quality_preservation_report,
                "support_strategy_summary": support_strategy_summary,

                "optimized_placements": optimized_placements,

                # Instruction-focused summary keeps height.
                "instruction_parts_summary": instruction_parts_summary,

                # Purchase-focused summary gives real physical BOM.
                "purchase_parts_summary": purchase_parts_summary,
                "purchase_parts_list": purchase_parts_list,

                # Backward compatibility: parts_summary points to purchase summary.
                "parts_summary": purchase_parts_summary,

                "rebrickable_parts_export": rebrickable_parts_export,
                "basic_parts_xml_export": basic_parts_xml_export,
                "build_layers": build_layers
            },

            # Convenience fields for n8n
            "debug_input": debug_input,
            "candidate_catalog_summary": candidate_catalog_summary,
            "product_tier": product_tier,
            "pricing": product_tier,
            "commercial_summary": commercial_summary,
            "quality_preservation_report": quality_preservation_report,
            "support_strategy_summary": support_strategy_summary,
            "recommended_price_usd": product_tier.get("recommended_price_usd"),

            "width": width,
            "height": height,

            # Compatibility + clearer count fields
            "placement_count": optimized_visual_block_count,
            "optimized_visual_block_count": optimized_visual_block_count,
            "optimized_part_count": physical_part_count,
            "actual_physical_purchase_part_count": physical_part_count,

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
                "debug_input": locals().get("debug_input"),
                "generation_time_seconds": round(time.time() - start_time, 2)
            }
        )
