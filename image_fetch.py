import json
import os
import base64
from urllib.parse import urlparse

HAR_FILE = "network.har"
OUTPUT_DIR = "snapshots_all"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(HAR_FILE, "r", encoding="utf-8") as f:
    har = json.load(f)

count = 0

for entry in har["log"]["entries"]:
    request_url = entry["request"]["url"]
    response = entry.get("response", {})
    content = response.get("content", {})

    mime_type = content.get("mimeType", "")
    text = content.get("text")
    encoding = content.get("encoding")

    if not text:
        continue

    is_image = mime_type.startswith("image/")
    url_has_image_ext = any(ext in request_url.lower() for ext in [".png", ".jpg", ".jpeg", ".webp"])

    if not (is_image or url_has_image_ext):
        continue

    parsed = urlparse(request_url)
    filename = os.path.basename(parsed.path)

    if not filename:
        ext = mime_type.split("/")[-1] if "/" in mime_type else "png"
        filename = f"snapshot_{count + 1}.{ext}"

    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        if encoding == "base64":
            image_bytes = base64.b64decode(text)
        else:
            image_bytes = text.encode("utf-8")

        with open(output_path, "wb") as img_file:
            img_file.write(image_bytes)

        count += 1
        print("Saved:", filename, "Size:", len(image_bytes))

    except Exception as e:
        print("Failed:", filename, e)

print(f"\nDone. Extracted {count} images.")