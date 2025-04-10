"""
Simple image orientation debugger module.
Helps verify image orientation issues during development.
Requires 'pillow-heif' to be installed for HEIC support: pip install pillow-heif
"""
import io
import uuid
import os
import json
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

from PIL import Image, ExifTags
# Ensure pillow-heif is registered if installed (it usually does this automatically on import)
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass # pillow-heif not installed, HEIC support will be missing

from fastapi import Request, FastAPI, APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import get_settings
from app.core.security import get_api_key # Assuming get_api_key handles security
from app.core import logging

class OrientationDebugger:
    """
    Simple in-memory image orientation debugger.
    Only stores images in memory during the current server session.
    Requires 'pillow-heif' for HEIC support.
    """
    # Maximum number of images to keep in memory
    MAX_IMAGES = 20

    # Debug mode enabled flag - set by setup_orientation_debugger
    ENABLED = False

    # In-memory storage of images
    _images = deque(maxlen=MAX_IMAGES)

    # Stats counter
    _stats = {
        "total_images": 0,
        "by_orientation": {i: 0 for i in range(1, 9)} # Initialize all orientations
    }

    @classmethod
    def is_valid_key(cls, key: str) -> bool:
        """Verify API key against existing LLM_SERVER_API_KEY"""
        # Note: This duplicates logic from get_api_key but might be needed
        # if routes don't use the dependency correctly. Prefer using Depends(get_api_key).
        settings = get_settings()
        return key == settings.llm_server_api_key

    @classmethod
    def capture_image(cls, image_bytes: bytes, endpoint: str, metadata: Optional[Dict] = None) -> Optional[str]:
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

            logging.debug(f"OrientationDebugger: Trying to open image bytes (length: {len(image_bytes)}) for ID {image_id}")
            # Open image to get EXIF data
            img = Image.open(io.BytesIO(image_bytes))
            # Force loading image data to catch potential errors early
            img.load()
            logging.debug(f"OrientationDebugger: Image opened successfully. Format: {img.format}, Mode: {img.mode}, Size: {img.size}")

            # Extract EXIF orientation
            orientation = 1  # Default (normal orientation)
            try:
                exif_data = img.getexif()
                if exif_data:
                    orientation = exif_data.get(0x0112, 1) # 0x0112 is the EXIF Orientation tag
            except Exception as exif_err:
                 logging.warning(f"OrientationDebugger: Could not read EXIF data for {image_id}: {exif_err}")

            logging.debug(f"OrientationDebugger: Extracted orientation: {orientation}")

            # Create small thumbnail for display
            logging.debug(f"OrientationDebugger: Creating thumbnail...")
            # Use a copy for thumbnail to avoid modifying original img object if needed later
            thumb_img = img.copy()
            thumb_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            thumbnail_buffer = io.BytesIO()
            # Ensure thumbnail is saved in a web-friendly format like JPEG
            if thumb_img.mode not in ['RGB', 'RGBA']:
                logging.debug(f"OrientationDebugger: Converting thumbnail mode from {thumb_img.mode} to RGB")
                thumb_img = thumb_img.convert('RGB')
            # Save as JPEG for wider browser compatibility
            thumb_img.save(thumbnail_buffer, format='JPEG', quality=85)
            thumbnail_base64 = base64.b64encode(thumbnail_buffer.getvalue()).decode('utf-8')
            logging.debug(f"OrientationDebugger: Thumbnail created (Base64 length: {len(thumbnail_base64)})")

            # Create entry
            entry = {
                "id": image_id,
                "timestamp": timestamp,
                "endpoint": endpoint,
                "orientation": orientation,
                "thumbnail": thumbnail_base64, # JPEG thumbnail
                "metadata": metadata or {}
            }

            # Update storage and stats
            cls._images.appendleft(entry) # Prepend to show newest first
            cls._stats["total_images"] += 1
            cls._stats["by_orientation"][orientation] = cls._stats["by_orientation"].get(orientation, 0) + 1

            logging.info(f"Captured image: ID={image_id}, orientation={orientation}, endpoint={endpoint}")

            return image_id

        except Exception as e:
            # Enhanced error logging
            logging.error(f"OrientationDebugger: Error processing image for endpoint {endpoint}: {type(e).__name__} - {str(e)}", exc_info=True)
            return None

    @classmethod
    def get_images(cls) -> List[Dict]:
        """Get the current list of captured images (newest first)."""
        return list(cls._images)

    @classmethod
    def get_stats(cls) -> Dict:
        """Get current statistics."""
        return cls._stats.copy()

    @classmethod
    def clear_images(cls) -> None:
        """Clear all captured images and reset counts."""
        cls._images.clear()
        # Reset stats
        total = cls._stats["total_images"] # Keep total overall count if desired, or set to 0
        cls._stats = {
            "total_images": total,
            "by_orientation": {i: 0 for i in range(1, 9)}
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
            </style>
        </head>
        <body>
            <h1>Image Orientation Debug</h1>
            <p>Generated: {datetime.now().isoformat(timespec='seconds')}</p>

            <div class="notice">
                <strong>Note:</strong> Include your API key in the request headers as 'X-API-Key' or via the ?key=YOUR_API_KEY query parameter to view this page and use controls. Requires `pillow-heif` for HEIC support.
            </div>

            <div class="controls">
                <button onclick="window.location.reload()">Refresh</button>
                <button class="clear-btn" onclick="clearImages()">Clear Images</button>
            </div>

            <h2>Statistics</h2>
            <div class="stats">
                <div class="stat-box">
                    <div>Total Images Processed: <strong>{stats['total_images']}</strong></div>
                     <div>Images Currently Displayed: <strong>{len(images)}</strong> (Max: {cls.MAX_IMAGES})</div>
                </div>

                <div class="stat-box">
                    <div>By EXIF Orientation</div>
        """

        # Add orientation stats
        orientation_desc = {
            1: "Normal (0°)", 2: "Mirrored horizontal", 3: "Rotated 180°", 4: "Mirrored vertical",
            5: "Mirrored horizontal, rotated 270° CW", 6: "Rotated 90° CW",
            7: "Mirrored horizontal, rotated 90° CW", 8: "Rotated 270° CW"
        }
        for orient in range(1, 9):
            count = stats['by_orientation'].get(orient, 0)
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
                <p>Thumbnails are shown *without* applying EXIF orientation correction to reflect what the server received initially.</p>
            </div>

            <h2>Recent Images (Newest First)</h2>
        """

        if not images:
            html += "<p>No images captured yet. Send images via POST to configured endpoints (e.g., /v1/extract-contact) with JSON body containing base64 image.</p>"
        else:
            html += """<div class="image-grid">"""

            # Add images
            for img_data in images: # Iterate directly as deque stores in order
                html += f"""
                    <div class="image-card">
                        <img src="data:image/jpeg;base64,{img_data['thumbnail']}" alt="Image {img_data['id']}">
                        <div class="image-info">
                            <div><strong>ID:</strong> {img_data['id']}</div>
                            <div><strong>Endpoint:</strong> {img_data['endpoint']}</div>
                            <div><strong>Orientation Tag:</strong> {img_data['orientation']} ({orientation_desc.get(img_data['orientation'], 'Unknown')})</div>
                            <div><strong>Time:</strong> {img_data['timestamp'].replace('T', ' ').split('.')[0]}</div>
                        </div>
                    </div>
                """

            html += "</div>"

        html += f"""
            <script>
                function clearImages() {{
                    // Try to get key from URL first for convenience
                    const urlParams = new URLSearchParams(window.location.search);
                    let key = urlParams.get('key');

                    if (!key) {{
                        key = prompt("Enter your API Key to clear images:");
                    }}

                    if (!key) {{
                         alert("API Key is required to clear images.");
                         return;
                    }}

                    fetch('/debug/orientation/clear', {{
                        method: 'POST',
                        headers: {{ 'X-API-Key': key }}
                    }})
                    .then(response => {{
                        if (response.ok) {{
                            alert('Images cleared successfully.');
                            window.location.reload();
                        }} else {{
                            response.text().then(text => {{
                                alert('Failed to clear images. Status: ' + response.status + '\\nReason: ' + (text || 'Invalid API key or server error.'));
                            }});
                        }}
                    }})
                    .catch(err => alert('Error clearing images: ' + err.message));
                }}
            </script>
        </body>
        </html>
        """

        return html

# Middleware to capture images for orientation debugging
class OrientationDebugMiddleware:
    """ASGI Middleware to capture images before they reach the endpoint."""
    # Define paths to watch for image uploads (Ensure these match your actual routes)
    WATCHED_PATHS = ['/v1/extract-contact', '/v1/upload', '/v1/image'] # Corrected paths

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        request_body_bytes = None # Store body if read

        # --- Logic moved BEFORE calling the app ---
        if OrientationDebugger.ENABLED and request.method == "POST" and \
           any(request.url.path.startswith(watch_path) for watch_path in self.WATCHED_PATHS):

            try:
                # Read the body *before* calling the app
                request_body_bytes = await request.body()

                # Now process the image using the read body_bytes
                content_type = request.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    try:
                        data = json.loads(request_body_bytes)
                        content = None
                        # Look for image content (same logic as before)
                        if "request" in data and isinstance(data["request"], dict):
                           req_data = data["request"]
                           if req_data.get("media_type") == "image":
                               content = req_data.get("content")
                        if content is None:
                           # Fallback checks
                           for field in ["image", "content", "image_data"]:
                               if field in data:
                                   content = data.get(field)
                                   break

                        if content and isinstance(content, str):
                           logging.debug(f"OrientationDebugMiddleware: Found image content in JSON key.")
                           # Handle data URI prefix if present
                           if "base64," in content:
                               content = content.split("base64,")[1]

                           try:
                               image_bytes_for_debug = base64.b64decode(content)
                               OrientationDebugger.capture_image(
                                   image_bytes_for_debug,
                                   request.url.path,
                                   {"content_type": content_type}
                               )
                           except base64.binascii.Error as b64_err:
                               logging.error(f"OrientationDebugMiddleware: Invalid Base64 data received: {b64_err}")
                           except Exception as capture_err:
                                logging.error(f"OrientationDebugMiddleware: Error during OrientationDebugger.capture_image: {capture_err}", exc_info=True)

                        else:
                            logging.debug(f"OrientationDebugMiddleware: No suitable image content found in JSON payload for path {request.url.path}.")

                    except json.JSONDecodeError:
                         logging.warning(f"OrientationDebugMiddleware: Request body for {request.url.path} is not valid JSON.")
                    except Exception as e:
                        # Log errors during image finding/decoding phase specifically
                        logging.error(f"OrientationDebugMiddleware: Error extracting/processing image from JSON body: {type(e).__name__} - {str(e)}", exc_info=False)
                else:
                     logging.debug(f"OrientationDebugMiddleware: Request Content-Type is not application/json ({content_type}), skipping image capture.")

            except Exception as e:
                 # Log errors during the initial body read or middleware logic
                 logging.error(f"OrientationDebugMiddleware: Error processing request: {type(e).__name__} - {str(e)}", exc_info=True)


            # --- Make the read body available again for the actual endpoint ---
            async def cached_receive() -> dict:
                # Send the cached body in one chunk
                return {"type": "http.request", "body": request_body_bytes if request_body_bytes is not None else b"", "more_body": False}

            # Replace the original receive channel
            receive = cached_receive
        # --- End of middleware pre-processing ---

        # Call the main app (or next middleware) with the original or replaced 'receive'
        await self.app(scope, receive, send)

# Function to add the debug routes to the FastAPI app
def create_orientation_debug_routes(app: FastAPI):
    """
    Add orientation debugging routes to a FastAPI app.
    """
    debug_router = APIRouter(prefix="/debug/orientation")

    @debug_router.get("", response_class=HTMLResponse, include_in_schema=False)
    async def orientation_debug_page(
        request: Request,
        key: Optional[str] = None # Allow key via query param
    ):
        """Serve the debug HTML page"""
        settings = get_settings()
        api_key_to_check = key or request.headers.get("X-API-Key")

        if not api_key_to_check or api_key_to_check != settings.llm_server_api_key:
             logging.warning(f"Unauthorized access attempt to /debug/orientation.")
             raise HTTPException(status_code=403, detail="Invalid or missing API key")

        html = OrientationDebugger.generate_debug_html()
        return HTMLResponse(content=html)

    @debug_router.get("/images", response_class=JSONResponse, include_in_schema=False)
    async def get_debug_images(api_key: str = Depends(get_api_key)):
        """Get debug images as JSON (Requires API Key in header)"""
        return JSONResponse(content={"success": True, "images": OrientationDebugger.get_images()})

    @debug_router.post("/clear", include_in_schema=False)
    async def clear_debug_images(api_key: str = Depends(get_api_key)):
        """Clear all debug images (Requires API Key in header)"""
        OrientationDebugger.clear_images()
        return JSONResponse(content={"success": True, "message": "Images cleared"})

    # Add the router to the app
    app.include_router(debug_router)

# Main setup function called from app.main
def setup_orientation_debugger(app: FastAPI):
    """
    Set up the orientation debugger with the application.

    Args:
        app: The FastAPI application

    Returns:
        The modified app with debugging middleware and routes (if enabled)
    """
    # Check environment settings
    settings = get_settings()
    # Allow overriding via environment variable, default to True in non-prod
    env = os.getenv("APP_ENV", "development").lower()
    debug_enabled_env = os.getenv("ORIENTATION_DEBUGGER_ENABLED", "true" if env != "production" else "false").lower()
    OrientationDebugger.ENABLED = debug_enabled_env in ["true", "1", "yes"]

    if OrientationDebugger.ENABLED:
        OrientationDebugger.MAX_IMAGES = int(os.getenv("ORIENTATION_DEBUG_MAX_IMAGES", "20"))

        # Add debug routes first
        create_orientation_debug_routes(app)
        # Add the middleware CLASS
        app.add_middleware(OrientationDebugMiddleware)

        logging.info("="*50)
        logging.info("🔍 Image orientation debugging is ENABLED")
        logging.info(f"   View at: /debug/orientation (use API Key)")
        logging.info(f"   Watching paths starting with: {OrientationDebugMiddleware.WATCHED_PATHS}")
        logging.info(f"   Keeping last {OrientationDebugger.MAX_IMAGES} images in memory.")
        logging.info(f"   Requires 'pillow-heif' for HEIC support.")
        logging.info("="*50)
    else:
        logging.debug("🔒 Image orientation debugging is DISABLED")

    return app