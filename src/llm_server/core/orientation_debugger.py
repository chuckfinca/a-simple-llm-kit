"""
Simple image orientation debugger module.
Helps verify image orientation issues during development.
Requires 'pillow-heif' to be installed for HEIC support: pip install pillow-heif
"""

import base64
import binascii
import io
import json
import os
import uuid
from collections import deque
from datetime import datetime
from typing import Annotated, Optional

from PIL import Image

# Ensure pillow-heif is registered if installed (it usually does this automatically on import)
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass  # pillow-heif not installed, HEIC support will be missing

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from llm_server.core import logging
from llm_server.core.config import get_settings
from llm_server.core.security import (
    get_api_key,  # Assuming get_api_key handles security
)


class OrientationDebugger:
    """
    Simple in-memory image orientation debugger.
    Only stores images in memory during the current server session.
    Requires 'pillow-heif' for HEIC support.
    """

    # Maximum number of images to keep in memory
    MAX_IMAGES: int = 20

    # Debug mode enabled flag - set by setup_orientation_debugger
    ENABLED: bool = False

    # In-memory storage of images
    _images = deque(maxlen=MAX_IMAGES)

    # Stats counter
    _stats = {
        "total_images": 0,
        "by_orientation": dict.fromkeys(range(1, 9), 0),  # Initialize all orientations
    }

    @classmethod
    def is_valid_key(cls, key: str) -> bool:
        """Verify API key against existing LLM_SERVER_API_KEY"""
        # Note: This duplicates logic from get_api_key but might be needed
        # if routes don't use the dependency correctly. Prefer using Depends(get_api_key).
        settings = get_settings()
        return key == settings.llm_server_api_key

    @classmethod
    def capture_image(
        cls, image_bytes: bytes, endpoint: str, metadata: Optional[dict] = None
    ) -> Optional[str]:
        """
        Capture an image for orientation debugging.
        Returns an ID if successful, None otherwise.
        """
        if not cls.ENABLED:
            return None

        try:
            # Generate ID and timestamp
            image_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()

            logging.debug(
                f"OrientationDebugger: Trying to open image bytes (length: {len(image_bytes)}) for ID {image_id}"
            )
            # Open image to get EXIF data
            img = Image.open(io.BytesIO(image_bytes))
            # Force loading image data to catch potential errors early
            img.load()
            logging.debug(
                f"OrientationDebugger: Image opened successfully. Format: {img.format}, Mode: {img.mode}, Size: {img.size}"
            )

            # Extract EXIF orientation
            orientation = 1  # Default (normal orientation)
            try:
                exif_data = img.getexif()
                if exif_data:
                    orientation = exif_data.get(
                        0x0112, 1
                    )  # 0x0112 is the EXIF Orientation tag
            except Exception as exif_err:
                logging.warning(
                    f"OrientationDebugger: Could not read EXIF data for {image_id}: {exif_err}"
                )

            logging.debug(f"OrientationDebugger: Extracted orientation: {orientation}")

            # Create thumbnails - both original and EXIF-corrected
            logging.debug("OrientationDebugger: Creating thumbnails...")

            # Create original (unrotated) thumbnail
            orig_thumb_img = img.copy()
            orig_thumb_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            if orig_thumb_img.mode not in ["RGB", "RGBA"]:
                logging.debug(
                    f"OrientationDebugger: Converting original thumbnail mode from {orig_thumb_img.mode} to RGB"
                )
                orig_thumb_img = orig_thumb_img.convert("RGB")

            orig_thumbnail_buffer = io.BytesIO()
            orig_thumb_img.save(orig_thumbnail_buffer, format="JPEG", quality=85)
            orig_thumbnail_base64 = base64.b64encode(
                orig_thumbnail_buffer.getvalue()
            ).decode("utf-8")

            # Create EXIF-corrected thumbnail
            corrected_thumb_img = cls._apply_orientation(img.copy(), orientation)
            corrected_thumb_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            if corrected_thumb_img.mode not in ["RGB", "RGBA"]:
                logging.debug(
                    f"OrientationDebugger: Converting corrected thumbnail mode from {corrected_thumb_img.mode} to RGB"
                )
                corrected_thumb_img = corrected_thumb_img.convert("RGB")

            corrected_thumbnail_buffer = io.BytesIO()
            corrected_thumb_img.save(
                corrected_thumbnail_buffer, format="JPEG", quality=85
            )
            corrected_thumbnail_base64 = base64.b64encode(
                corrected_thumbnail_buffer.getvalue()
            ).decode("utf-8")

            logging.debug(
                f"OrientationDebugger: Thumbnails created (Original Base64 length: {len(orig_thumbnail_base64)}, Corrected Base64 length: {len(corrected_thumbnail_base64)})"
            )

            # Create entry
            entry = {
                "id": image_id,
                "timestamp": timestamp,
                "endpoint": endpoint,
                "orientation": orientation,
                "thumbnail_original": orig_thumbnail_base64,  # Original unrotated thumbnail
                "thumbnail_corrected": corrected_thumbnail_base64,  # EXIF-corrected thumbnail
                "has_exif_orientation": orientation
                != 1,  # Flag to indicate if image had EXIF orientation data
                "metadata": metadata or {},
            }

            # Update storage and stats
            cls._images.appendleft(entry)  # Prepend to show newest first
            cls._stats["total_images"] += 1
            cls._stats["by_orientation"][orientation] = (
                cls._stats["by_orientation"].get(orientation, 0) + 1
            )

            logging.info(
                f"Captured image: ID={image_id}, orientation={orientation}, endpoint={endpoint}"
            )

            return image_id

        except Exception as e:
            # Enhanced error logging
            logging.error(
                f"OrientationDebugger: Error processing image for endpoint {endpoint}: {type(e).__name__} - {str(e)}",
                exc_info=True,
            )
            return None

    @classmethod
    def _apply_orientation(cls, image: Image.Image, orientation: int) -> Image.Image:
        """Apply EXIF orientation to the image"""
        if orientation == 1:
            # Normal orientation, no change needed
            return image
        elif orientation == 2:
            # Mirror horizontal
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotate 180
            return image.transpose(Image.Transpose.ROTATE_180)
        elif orientation == 4:
            # Mirror vertical
            return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            # Mirror horizontal and rotate 270 CW
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(
                Image.Transpose.ROTATE_270
            )
        elif orientation == 6:
            # Rotate 90 CW
            return image.transpose(Image.Transpose.ROTATE_270)
        elif orientation == 7:
            # Mirror horizontal and rotate 90 CW
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(
                Image.Transpose.ROTATE_90
            )
        elif orientation == 8:
            # Rotate 270 CW
            return image.transpose(Image.Transpose.ROTATE_90)
        return image  # Default: return original image

    @classmethod
    def get_images(cls) -> list[dict]:
        """Get the current list of captured images (newest first)."""
        return list(cls._images)

    @classmethod
    def get_stats(cls) -> dict:
        """Get current statistics."""
        return cls._stats.copy()

    @classmethod
    def clear_images(cls) -> None:
        """Clear all captured images and reset counts."""
        cls._images.clear()
        # Reset stats
        total = cls._stats[
            "total_images"
        ]  # Keep total overall count if desired, or set to 0
        cls._stats = {
            "total_images": total,
            "by_orientation": dict.fromkeys(range(1, 9), 0),
        }
        logging.info("OrientationDebugger: Cleared captured images.")

    @classmethod
    def generate_debug_html(cls) -> str:
        """
        Generate a simple HTML page for orientation debugging.
        """
        images = cls.get_images()
        stats = cls.get_stats()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Image Orientation Debug</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: system-ui, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; background-color: #f9f9f9; color: #333; }}
                h1, h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 10px; color: #111; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }}
                .stat-box {{ background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .stat-box div {{ margin-bottom: 5px; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
                .image-card {{ background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; flex-direction: column; }}
                .image-card img {{ width: 100%; height: 250px; object-fit: contain; background: #eee; border-bottom: 1px solid #eee; }}
                .image-info {{ padding: 15px; font-size: 0.9em; flex-grow: 1; }}
                .image-info div {{ margin-bottom: 5px; word-wrap: break-word; }}
                .controls {{ margin: 20px 0; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
                button {{ padding: 10px 18px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 1em; transition: background-color 0.2s; }}
                button:hover {{ background: #0056b3; }}
                .clear-btn {{ background: #dc3545; }}
                .clear-btn:hover {{ background: #c82333; }}
                .notice {{ background: #fff3cd; color: #856404; border: 1px solid #ffeeba; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .orientation-guide {{ margin: 20px 0; padding: 15px; background: #e9ecef; border-radius: 5px; }}
                .orientation-guide ul {{ padding-left: 20px; margin-top: 10px; }}
                .toggle-container {{ margin: 20px 0; background: #e9ecef; padding: 15px; border-radius: 5px; display: flex; align-items: center; gap: 10px; }}
                .toggle-switch {{ position: relative; display: inline-block; width: 60px; height: 34px; }}
                .toggle-switch input {{ opacity: 0; width: 0; height: 0; }}
                .slider {{ position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; border-radius: 34px; }}
                .slider:before {{ position: absolute; content: ""; height: 26px; width: 26px; left: 4px; bottom: 4px; background-color: white; transition: .4s; border-radius: 50%; }}
                input:checked + .slider {{ background-color: #2196F3; }}
                input:checked + .slider:before {{ transform: translateX(26px); }}
                .no-exif-badge {{ display: inline-block; padding: 3px 8px; background-color: #dc3545; color: white; border-radius: 12px; font-size: 0.7em; margin-left: 8px; }}
                .exif-normal-badge {{ display: inline-block; padding: 3px 8px; background-color: #28a745; color: white; border-radius: 12px; font-size: 0.7em; margin-left: 8px; }}
                .hidden {{ display: none; }}
            </style>
        </head>
        <body>
            <h1>Image Orientation Debug</h1>
            <p>Generated: {datetime.now().isoformat(timespec="seconds")}</p>

            <div class="notice">
                <strong>Note:</strong> Include your API key in the request headers as 'X-API-Key' or via the ?key=YOUR_API_KEY query parameter to view this page and use controls. Requires `pillow-heif` for HEIC support.
            </div>

            <div class="controls">
                <button onclick="window.location.reload()">Refresh</button>
                <button class="clear-btn" onclick="clearImages()">Clear Images</button>
            </div>
            
            <div class="toggle-container">
                <span><strong>Display Mode:</strong></span>
                <label class="toggle-switch">
                    <input type="checkbox" id="orientationToggle" checked>
                    <span class="slider"></span>
                </label>
                <span id="toggleLabel"><strong>Correctly Oriented</strong> (EXIF-corrected)</span>
            </div>

            <h2>Statistics</h2>
            <div class="stats">
                <div class="stat-box">
                    <div>Total Images Processed: <strong>{stats["total_images"]}</strong></div>
                     <div>Images Currently Displayed: <strong>{len(images)}</strong> (Max: {cls.MAX_IMAGES})</div>
                </div>

                <div class="stat-box">
                    <div>By EXIF Orientation</div>
        """

        # Add orientation stats
        orientation_desc = {
            1: "Normal (0°)",
            2: "Mirrored horizontal",
            3: "Rotated 180°",
            4: "Mirrored vertical",
            5: "Mirrored horizontal, rotated 270° CW",
            6: "Rotated 90° CW",
            7: "Mirrored horizontal, rotated 90° CW",
            8: "Rotated 270° CW",
        }
        for orient in range(1, 9):
            count = stats["by_orientation"].get(orient, 0)
            if count > 0:
                desc = orientation_desc.get(orient, f"Value {orient}")
                html += f"<div>{orient} ({desc}): <strong>{count}</strong></div>"

        html += """
                </div>
            </div>

            <div class="orientation-guide">
                <p><strong>EXIF Orientation Value Meanings:</strong></p>
                <ul>
                    <li><strong>1:</strong> Normal</li>
                    <li><strong>3:</strong> Rotated 180°</li>
                    <li><strong>6:</strong> Rotated 90° Clockwise (CW) / 270° Counter-Clockwise (CCW)</li>
                    <li><strong>8:</strong> Rotated 270° Clockwise (CW) / 90° Counter-Clockwise (CCW)</li>
                    <li>2, 4, 5, 7 involve mirroring/flipping.</li>
                </ul>
                <p>Toggle the switch above to view images with or without EXIF orientation correction applied.</p>
                <p><span class="no-exif-badge">No EXIF</span> indicates images without orientation data.</p>
                <p><span class="exif-normal-badge">Normal</span> indicates images with orientation = 1 (normal).</p>
            </div>

            <h2>Recent Images (Newest First)</h2>
        """

        if not images:
            html += "<p>No images captured yet. Send images via POST to configured endpoints (e.g., /v1/extract-contact) with JSON body containing base64 image.</p>"
        else:
            html += """<div class="image-grid">"""

            # Add images
            for img_data in images:  # Iterate directly as deque stores in order
                badge = ""
                if img_data["orientation"] == 1:
                    badge = '<span class="exif-normal-badge">Normal</span>'
                elif not img_data.get("has_exif_orientation", False):
                    badge = '<span class="no-exif-badge">No EXIF</span>'

                html += f"""
                    <div class="image-card">
                        <img src="data:image/jpeg;base64,{img_data["thumbnail_corrected"]}"
                             alt="Image {img_data["id"]}"
                             class="corrected-img"
                             data-original="data:image/jpeg;base64,{img_data["thumbnail_original"]}"
                             data-corrected="data:image/jpeg;base64,{img_data["thumbnail_corrected"]}">
                        <div class="image-info">
                            <div><strong>ID:</strong> {img_data["id"]}</div>
                            <div><strong>Endpoint:</strong> {img_data["endpoint"]}</div>
                            <div><strong>Orientation Tag:</strong> {img_data["orientation"]} ({orientation_desc.get(img_data["orientation"], "Unknown")}) {badge}</div>
                            <div><strong>Time:</strong> {img_data["timestamp"].replace("T", " ").split(".")[0]}</div>
                        </div>
                    </div>
                """

            html += "</div>"

        html += """
            <script>
                // Handle orientation toggle
                const orientationToggle = document.getElementById('orientationToggle');
                const toggleLabel = document.getElementById('toggleLabel');
                const images = document.querySelectorAll('.image-card img');
                
                // Initial state set to corrected (checked)
                updateImageDisplay();
                
                orientationToggle.addEventListener('change', updateImageDisplay);
                
                function updateImageDisplay() {
                    const isCorrected = orientationToggle.checked;
                    
                    toggleLabel.innerHTML = isCorrected ?
                        "<strong>Correctly Oriented</strong> (EXIF-corrected)" :
                        "<strong>Original Orientation</strong> (as received)";
                        
                    images.forEach(img => {
                        if (isCorrected) {
                            img.src = img.dataset.corrected;
                        } else {
                            img.src = img.dataset.original;
                        }
                    });
                }
                
                function clearImages() {
                    // Try to get key from URL first for convenience
                    const urlParams = new URLSearchParams(window.location.search);
                    let key = urlParams.get('key');

                    if (!key) {
                        key = prompt("Enter your API Key to clear images:");
                    }

                    if (!key) {
                         alert("API Key is required to clear images.");
                         return;
                    }

                    fetch('/debug/orientation/clear', {
                        method: 'POST',
                        headers: { 'X-API-Key': key }
                    })
                    .then(response => {
                        if (response.ok) {
                            alert('Images cleared successfully.');
                            window.location.reload();
                        } else {
                            response.text().then(text => {
                                alert('Failed to clear images. Status: ' + response.status + '\\nReason: ' + (text || 'Invalid API key or server error.'));
                            });
                        }
                    })
                    .catch(err => alert('Error clearing images: ' + err.message));
                }
            </script>
        </body>
        </html>
        """

        return html


# Middleware to capture images for orientation debugging
class OrientationDebugMiddleware:
    """ASGI Middleware to capture images before they reach the endpoint."""

    WATCHED_PATHS = ["/v1/extract-contact", "/v1/upload", "/v1/image"]

    def __init__(self, app: ASGIApp):
        self.app = app

    async def _try_capture_image_from_request(
        self, request: Request
    ) -> Optional[bytes]:
        """Reads the request body and attempts to capture an image for debugging."""
        body_bytes: Optional[bytes] = None
        try:
            body_bytes = await request.body()
            content_type = request.headers.get("content-type", "").lower()

            if "application/json" not in content_type:
                logging.debug(
                    "Skipping image capture: Content-Type is not application/json."
                )
                return body_bytes

            data = json.loads(body_bytes)
            # Standardized check for image content in the request
            req_data = data.get("request", {})
            content = (
                req_data.get("content")
                if req_data.get("media_type") == "image"
                else None
            )

            if not content or not isinstance(content, str):
                logging.debug("No suitable image content found in JSON payload.")
                return body_bytes

            # Handle data URI prefix if present
            if "base64," in content:
                content = content.split("base64,")[1]

            image_bytes_for_debug = base64.b64decode(content)
            OrientationDebugger.capture_image(
                image_bytes_for_debug,
                request.url.path,
                {"content_type": content_type},
            )
            return body_bytes

        except (json.JSONDecodeError, binascii.Error) as e:
            logging.warning(f"Could not process request for image capture: {e}")
            return body_bytes  # Return original bytes if parsing fails
        except Exception as e:
            logging.error(f"Unexpected error during image capture: {e}", exc_info=True)
            # If an unknown error occurs, we might not have the body bytes, so return None
            return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)

        # Guard Clauses to exit early
        if not OrientationDebugger.ENABLED or request.method != "POST":
            await self.app(scope, receive, send)
            return

        is_watched_path = any(
            request.url.path.startswith(p) for p in self.WATCHED_PATHS
        )
        if not is_watched_path:
            await self.app(scope, receive, send)
            return

        # The core logic is now cleaner
        body_bytes = await self._try_capture_image_from_request(request)

        # Re-create the receive channel so the endpoint can read the body
        async def cached_receive() -> dict:
            return {
                "type": "http.request",
                "body": body_bytes or b"",
                "more_body": False,
            }

        await self.app(scope, cached_receive, send)


# Function to add the debug routes to the FastAPI app
def create_orientation_debug_routes(app: FastAPI):
    """
    Add orientation debugging routes to a FastAPI app.
    """
    debug_router = APIRouter(prefix="/debug/orientation")

    @debug_router.get("", response_class=HTMLResponse, include_in_schema=False)
    async def orientation_debug_page(
        request: Request,
        key: Optional[str] = None,  # Allow key via query param
    ):
        """Serve the debug HTML page"""
        settings = get_settings()
        api_key_to_check = key or request.headers.get("X-API-Key")

        if not api_key_to_check or api_key_to_check != settings.llm_server_api_key:
            logging.warning("Unauthorized access attempt to /debug/orientation.")
            raise HTTPException(status_code=403, detail="Invalid or missing API key")

        html = OrientationDebugger.generate_debug_html()
        return HTMLResponse(content=html)

    @debug_router.get("/images", response_class=JSONResponse, include_in_schema=False)
    async def get_debug_images(api_key: Annotated[str, Depends(get_api_key)]):
        """Get debug images as JSON (Requires API Key in header)"""
        return JSONResponse(
            content={"success": True, "images": OrientationDebugger.get_images()}
        )

    @debug_router.post("/clear", include_in_schema=False)
    async def clear_debug_images(api_key: Annotated[str, Depends(get_api_key)]):
        """Clear all debug images (Requires API Key in header)"""
        OrientationDebugger.clear_images()
        return JSONResponse(content={"success": True, "message": "Images cleared"})

    # Add the router to the app
    app.include_router(debug_router)


# Main setup function called from llm_server.main
def setup_orientation_debugger(app: FastAPI):
    """
    Set up the orientation debugger with the application.

    Args:
        app: The FastAPI application

    Returns:
        The modified app with debugging middleware and routes (if enabled)
    """
    # Allow overriding via environment variable, default to True in non-prod
    env = os.getenv("APP_ENV", "development").lower()
    debug_enabled_env = os.getenv(
        "ORIENTATION_DEBUGGER_ENABLED", "true" if env != "production" else "false"
    ).lower()
    OrientationDebugger.ENABLED = debug_enabled_env in ["true", "1", "yes"]

    if OrientationDebugger.ENABLED:
        OrientationDebugger.MAX_IMAGES = int(
            os.getenv("ORIENTATION_DEBUG_MAX_IMAGES", "20")
        )

        # Add debug routes first
        create_orientation_debug_routes(app)
        # Add the middleware CLASS
        app.add_middleware(OrientationDebugMiddleware)

        logging.info("=" * 50)
        logging.info("🔍 Image orientation debugging is ENABLED")
        logging.info("   View at: /debug/orientation (use API Key)")
        logging.info(
            f"   Watching paths starting with: {OrientationDebugMiddleware.WATCHED_PATHS}"
        )
        logging.info(
            f"   Keeping last {OrientationDebugger.MAX_IMAGES} images in memory."
        )
        logging.info("   Requires 'pillow-heif' for HEIC support.")
        logging.info("=" * 50)
    else:
        logging.debug("🔒 Image orientation debugging is DISABLED")

    return app
