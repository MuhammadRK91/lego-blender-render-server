from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import time
import uuid
import numpy as np
import trimesh

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


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "lego-glb-generator",
        "message": "Send POST request to /generate-glb with image_geometry",
        "uses_blender": False,
        "output": "glb_base64"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "lego-glb-generator",
        "mode": "image_geometry_to_glb",
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


def create_box_mesh(center, extents, rgba):
    mesh = trimesh.creation.box(extents=extents)

    transform = np.eye(4)
    transform[:3, 3] = center
    mesh.apply_transform(transform)

    mesh.visual.face_colors = rgba

    return mesh


def generate_glb_from_image_geometry(geometry):
    width = int(geometry.get("width", 0))
    height = int(geometry.get("height", 0))
    placements = geometry.get("placements", [])

    if width <= 0 or height <= 0:
        raise ValueError("image_geometry.width and image_geometry.height are required.")

    if not isinstance(placements, list) or not placements:
        raise ValueError("image_geometry.placements[] is required.")

    original_placement_count = len(placements)

    if len(placements) > MAX_PLACEMENTS:
        placements = placements[:MAX_PLACEMENTS]

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

    for p in placements:
        x = int(p.get("x", 0))
        y = int(p.get("y", 0))

        height_plates = int(p.get("height_plates", 1))
        height_plates = max(1, min(12, height_plates))

        block_h = height_plates * PLATE_H

        # Center model around origin
        world_x = (x * STUD_W + STUD_W / 2) - (model_w / 2)
        world_z = (y * STUD_D + STUD_D / 2) - (model_d / 2)
        world_y = block_h / 2

        color_hex = p.get("rebrickable_color_rgb")

        if color_hex:
            rgba = hex_to_rgba(color_hex)
        else:
            rgba = source_rgb_to_rgba(p.get("source_rgb", {}))

        block_mesh = create_box_mesh(
            center=[world_x, world_y, world_z],
            extents=[
                STUD_W - BLOCK_GAP,
                block_h,
                STUD_D - BLOCK_GAP
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
        "placement_count": len(placements),
        "original_placement_count": original_placement_count,
        "truncated": original_placement_count > len(placements)
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

        result = generate_glb_from_image_geometry(geometry)

        glb_base64 = base64.b64encode(result["glb_bytes"]).decode("utf-8")
        generation_time_seconds = round(time.time() - start_time, 2)

        return {
            "success": True,
            "message": "GLB model generated successfully",
            "job_id": job_id,
            "filename": f"lego_model_{job_id}.glb",
            "mimeType": "model/gltf-binary",
            "glb_base64": glb_base64,
            "width": result["width"],
            "height": result["height"],
            "placement_count": result["placement_count"],
            "original_placement_count": result["original_placement_count"],
            "truncated": result["truncated"],
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
