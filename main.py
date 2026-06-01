import base64
import io
import json
import os
import re
from typing import Optional, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from google import genai
from google.genai import types


app = FastAPI(title="Gemini Crop A4 Document API")

A4_WIDTH_PX = 1240
A4_HEIGHT_PX = 1754


class CropRequest(BaseModel):
    filename: Optional[str] = "image.jpg"
    mimeType: Optional[str] = "image/jpeg"
    fileBase64: str
    output: Optional[Dict[str, Any]] = None


@app.get("/")
def health_check():
    return {
        "ok": True,
        "service": "Gemini Crop A4 Document API"
    }


@app.post("/crop-a4")
def crop_a4(payload: CropRequest):
    try:
        image_bytes = base64.b64decode(payload.fileBase64)
        image = decode_image(image_bytes)

        if image is None:
            raise HTTPException(status_code=400, detail="Không đọc được ảnh.")

        output_options = payload.output or {}
        width_px = int(output_options.get("widthPx", A4_WIDTH_PX))
        height_px = int(output_options.get("heightPx", A4_HEIGHT_PX))
        enhance_text = bool(output_options.get("enhanceText", True))

        cropped, detected, method, gemini_data = crop_document_with_gemini(
            image=image,
            image_bytes=image_bytes,
            mime_type=payload.mimeType or "image/jpeg",
            output_width=width_px,
            output_height=height_px,
            enhance_text=enhance_text
        )

        output_bytes = encode_jpeg(cropped, quality=92)
        output_base64 = base64.b64encode(output_bytes).decode("utf-8")

        return {
            "filename": make_output_name(payload.filename or "image.jpg"),
            "mimeType": "image/jpeg",
            "fileBase64": output_base64,
            "detectedDocument": detected,
            "method": method,
            "gemini": gemini_data
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def decode_image(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is not None:
        return image

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.array(pil_image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def crop_document_with_gemini(
    image,
    image_bytes: bytes,
    mime_type: str,
    output_width: int,
    output_height: int,
    enhance_text: bool
):
    h, w = image.shape[:2]

    gemini_result = find_document_area_with_gemini(
        image_bytes=image_bytes,
        mime_type=mime_type,
        image_width=w,
        image_height=h
    )

    if gemini_result.get("document_found") is True:
        corners = gemini_result.get("corners")

        if is_valid_corners(corners, w, h):
            points = np.array(corners, dtype="float32")

            warped = four_point_transform(
                image,
                points,
                output_width,
                output_height
            )

            if enhance_text:
                warped = enhance_document(warped)

            return warped, True, "gemini_corners", gemini_result

        box = gemini_result.get("box_2d")

        if is_valid_box(box, w, h):
            x1, y1, x2, y2 = box_to_pixels(box, w, h)
            cropped = image[y1:y2, x1:x2]

            warped = resize_to_a4(cropped, output_width, output_height)

            if enhance_text:
                warped = enhance_document(warped)

            return warped, True, "gemini_box", gemini_result

    # Fallback an toàn: không crop bừa, đưa nguyên ảnh vào nền A4
    safe = fit_image_to_a4_canvas(image, output_width, output_height)

    if enhance_text:
        safe = enhance_document(safe)

    return safe, False, "safe_fit_full_image", gemini_result


def find_document_area_with_gemini(
    image_bytes: bytes,
    mime_type: str,
    image_width: int,
    image_height: int
):
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return {
            "document_found": False,
            "reason": "Missing GEMINI_API_KEY"
        }

    client = genai.Client(api_key=api_key)

    prompt = f"""
Bạn là hệ thống phát hiện vùng tài liệu trong ảnh chụp.

Nhiệm vụ:
- Tìm vùng tờ giấy / tài liệu chính trong ảnh.
- Ưu tiên trả về 4 góc thật của tài liệu nếu nhìn thấy.
- Không chọn nền bàn, tay, bóng đổ, màn hình, hoặc vật khác.
- Nếu không chắc, document_found phải là false.

Ảnh có kích thước:
width={image_width}
height={image_height}

Chỉ trả về JSON hợp lệ, không markdown, không giải thích.

Schema:
{{
  "document_found": true hoặc false,
  "confidence": số từ 0 đến 1,
  "corners": [
    [x_top_left, y_top_left],
    [x_top_right, y_top_right],
    [x_bottom_right, y_bottom_right],
    [x_bottom_left, y_bottom_left]
  ],
  "box_2d": [ymin, xmin, ymax, xmax],
  "reason": "ngắn gọn"
}}

Yêu cầu tọa độ:
- corners dùng tọa độ pixel thật theo ảnh gốc.
- box_2d dùng tọa độ pixel thật theo ảnh gốc, thứ tự [ymin, xmin, ymax, xmax].
- Nếu chỉ thấy một phần tài liệu hoặc không chắc mép giấy, document_found=false.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            ),
            prompt
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json"
        )
    )

    text = response.text or "{}"

    try:
        data = json.loads(text)
    except Exception:
        data = parse_json_from_text(text)

    if not isinstance(data, dict):
        return {
            "document_found": False,
            "reason": "Gemini returned invalid JSON",
            "raw": text[:500]
        }

    confidence = float(data.get("confidence", 0) or 0)

    if confidence < 0.55:
        data["document_found"] = False
        data["reason"] = "Confidence too low: " + str(confidence)

    return data


def parse_json_from_text(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return {}

    return json.loads(match.group(0))


def is_valid_corners(corners, image_width: int, image_height: int):
    if not isinstance(corners, list) or len(corners) != 4:
        return False

    try:
        points = np.array(corners, dtype="float32")
    except Exception:
        return False

    if points.shape != (4, 2):
        return False

    for x, y in points:
        if x < 0 or y < 0 or x > image_width or y > image_height:
            return False

    area = cv2.contourArea(points)

    if area < image_width * image_height * 0.08:
        return False

    return True


def is_valid_box(box, image_width: int, image_height: int):
    if not isinstance(box, list) or len(box) != 4:
        return False

    try:
        ymin, xmin, ymax, xmax = [float(v) for v in box]
    except Exception:
        return False

    if xmax <= xmin or ymax <= ymin:
        return False

    if xmin < 0 or ymin < 0 or xmax > image_width or ymax > image_height:
        return False

    area = (xmax - xmin) * (ymax - ymin)

    if area < image_width * image_height * 0.08:
        return False

    return True


def box_to_pixels(box, image_width: int, image_height: int):
    ymin, xmin, ymax, xmax = [int(round(float(v))) for v in box]

    xmin = max(0, min(xmin, image_width - 1))
    xmax = max(1, min(xmax, image_width))
    ymin = max(0, min(ymin, image_height - 1))
    ymax = max(1, min(ymax, image_height))

    return xmin, ymin, xmax, ymax


def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


def four_point_transform(image, points, output_width: int, output_height: int):
    rect = order_points(points)

    dst = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (output_width, output_height))

    return warped


def resize_to_a4(image, output_width: int, output_height: int):
    return cv2.resize(
        image,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA
    )


def fit_image_to_a4_canvas(image, output_width: int, output_height: int):
    canvas = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255

    h, w = image.shape[:2]
    scale = min(output_width / w, output_height / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x = (output_width - new_w) // 2
    y = (output_height - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def enhance_document(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced


def encode_jpeg(image, quality: int = 92):
    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )

    if not success:
        raise ValueError("Không encode được ảnh JPG.")

    return buffer.tobytes()


def make_output_name(filename: str):
    name = filename.rsplit(".", 1)[0]
    return f"{name}_a4.jpg"