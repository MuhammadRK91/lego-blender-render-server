from fastapi import FastAPI
from fastapi.responses import JSONResponse
import base64
import json
import os
import subprocess
import time
import uuid

app = FastAPI()

BASE_DIR = "/tmp"
SCRIPT_PATH = "/app/render_lego_blender.py"


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "lego-blender-render-server",
        "message": "Send POST request to /render-lego"
    }


@app.get("/health")
def health():
    blender_check = subprocess.run(
        ["blender", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return {
        "status": "ok",
        "blender_available": blender_check.returncode == 0,
        "blender_version": blender_check.stdout.split("\n")[0] if blender_check.stdout else None
    }


@app.post("/render-lego")
async def render_lego(data: dict):
    job_id = str(uuid.uuid4())

    input_path = os.path.join(BASE_DIR, f"blender_input_{job_id}.json")
    output_path = os.path.join(BASE_DIR, f"lego_render_{job_id}.png")
    blend_path = output_path.replace(".png", ".blend")

    try:
        if "blender_optimized_parts" not in data:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Missing blender_optimized_parts in request body"
                }
            )

        part_count = len(data.get("blender_optimized_parts", []))

        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        start_time = time.time()

        command = [
            "blender",
            "--background",
            "--python",
            SCRIPT_PATH,
            "--",
            "--input",
            input_path,
            "--output",
            output_path
        ]

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=900
        )

        render_time_seconds = round(time.time() - start_time, 2)

        if process.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Blender render failed",
                    "return_code": process.returncode,
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                    "part_count": part_count,
                    "render_time_seconds": render_time_seconds
                }
            )

        if not os.path.exists(output_path):
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Blender finished but output image was not created",
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                    "part_count": part_count,
                    "render_time_seconds": render_time_seconds
                }
            )

        with open(output_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "success": True,
            "message": "LEGO render completed successfully",
            "filename": "lego_render.png",
            "mimeType": "image/png",
            "image_base64": image_base64,
            "part_count": part_count,
            "render_time_seconds": render_time_seconds,
            "stdout_tail": process.stdout[-2000:]
        }

    except subprocess.TimeoutExpired:
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": "Blender render timed out",
                "part_count": len(data.get("blender_optimized_parts", []))
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "part_count": len(data.get("blender_optimized_parts", [])) if isinstance(data, dict) else None
            }
        )

    finally:
        for path in [input_path, output_path, blend_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
