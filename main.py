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

# A4 nhẹ hơn để chạy nhanh, đủ đọc văn bản
# Apps Script vẫn có thể truyền widthPx / heightPx để ghi đè
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

    gemini_result = find_document_corners_with_gemini(
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

    # QUAN TRỌNG:
    # Không dùng bounding box để crop vì Gemini hay chọn vùng chữ.
    # Nếu không có đủ 4 góc ngoài của tờ giấy thì giữ nguyên ảnh vào nền A4.
    safe = fit_image_to_a4_canvas(image, output_width, output_height)

    if enhance_text:
        safe = enhance_document(safe)

    return safe, False, "safe_fit_full_image_no_valid_corners", gemini_result


def find_document_corners_with_gemini(
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
Bạn là hệ thống phát hiện mép ngoài của tờ giấy/tài liệu trong ảnh chụp.

Nhiệm vụ chính:
- Chỉ tìm 4 góc ngoài cùng của toàn bộ tờ giấy hoặc tài liệu chính.
- Mục tiêu là lấy nguyên cả trang giấy, bao gồm cả lề trắng của giấy nếu có.
- Không được chọn vùng chữ.
- Không được chọn tiêu đề.
- Không được chọn đoạn văn bản.
- Không được chọn logo.
- Không được chọn bảng biểu.
- Không được chọn nội dung nằm bên trong giấy.
- Không được crop sát chữ.
- Không được chọn vùng chỉ chứa chữ lớn.
- Không được chọn một phần trang giấy.

Điều kiện bắt buộc:
- Chỉ trả document_found=true nếu nhìn thấy rõ đủ 4 góc ngoài của tờ giấy.
- Nếu chỉ thấy nội dung chữ nhưng không thấy rõ mép giấy, document_found=false.
- Nếu nền và giấy trùng màu khiến mép giấy không rõ, document_found=false.
- Nếu ảnh bị cắt mất góc giấy, document_found=false.
- Nếu có nhiều vật thể khác trong ảnh, chỉ chọn tờ giấy/tài liệu chính.
- Nếu không chắc chắn, document_found=false.

Ảnh có kích thước:
width={image_width}
height={image_height}

Chỉ trả về JSON hợp lệ.
Không markdown.
Không giải thích ngoài JSON.

Schema bắt buộc:
{{
  "document_found": true,
  "confidence": 0.0,
  "corners": [
    [x_top_left, y_top_left],
    [x_top_right, y_top_right],
    [x_bottom_right, y_bottom_right],
    [x_bottom_left, y_bottom_left]
  ],
  "reason": "ngắn gọn"
}}

Quy định tọa độ:
- corners là 4 góc ngoài của toàn bộ tờ giấy.
- corners dùng tọa độ pixel thật theo ảnh gốc.
- Tọa độ x nằm trong khoảng 0 đến {image_width}.
- Tọa độ y nằm trong khoảng 0 đến {image_height}.
- Nếu document_found=false thì corners có thể là [].
- Nếu không chắc, confidence phải dưới 0.55 và document_found=false.

Ví dụ đúng:
- Trả 4 góc ngoài của trang giấy.
- Bao gồm cả lề trắng của tờ giấy.

Ví dụ sai:
- Chọn khung quanh chữ.
- Chọn khung quanh tiêu đề.
- Chọn vùng nội dung ở giữa trang.
- Chọn vùng crop sát chữ.
"""

    models_to_try = [
        "gemini-3.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash"
    ]

    last_error = None

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
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
                last_error = f"{model_name}: invalid JSON"
                continue

            data["model_used"] = model_name

            confidence = float(data.get("confidence", 0) or 0)

            if confidence < 0.60:
                data["document_found"] = False
                data["reason"] = "Confidence too low: " + str(confidence)

            # Không cho dùng corners rỗng
            if data.get("document_found") is True:
                corners = data.get("corners")
                if not corners or not isinstance(corners, list) or len(corners) != 4:
                    data["document_found"] = False
                    data["reason"] = "No valid 4 document corners returned"

            return data

        except Exception as exc:
            last_error = f"{model_name}: {str(exc)}"
            continue

    return {
        "document_found": False,
        "reason": "All Gemini models unavailable or failed",
        "last_error": str(last_error)
    }


def parse_json_from_text(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


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

    # Nếu vùng quá nhỏ, có thể Gemini đã chọn vùng chữ
    if area < image_width * image_height * 0.18:
        return False

    if not is_reasonable_quad(points):
        return False

    return True


def is_reasonable_quad(points):
    rect = order_points(points)

    tl, tr, br, bl = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    if min(width_top, width_bottom, height_left, height_right) <= 1:
        return False

    width_ratio = max(width_top, width_bottom) / max(min(width_top, width_bottom), 1)
    height_ratio = max(height_left, height_right) / max(min(height_left, height_right), 1)

    # Nếu 4 góc tạo ra hình quá méo thì không tin
    if width_ratio > 2.5:
        return False

    if height_ratio > 2.5:
        return False

    avg_width = (width_top + width_bottom) / 2.0
    avg_height = (height_left + height_right) / 2.0

    doc_ratio = avg_width / max(avg_height, 1)

    # A4 đứng khoảng 0.707, A4 ngang khoảng 1.414.
    # Cho rộng hơn để chấp nhận ảnh chụp nghiêng.
    if not (0.35 <= doc_ratio <= 2.50):
        return False

    return True


def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]      # top-left
    rect[2] = points[np.argmax(s)]      # bottom-right

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]   # top-right
    rect[3] = points[np.argmax(diff)]   # bottom-left

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
    # Tăng tương phản nhẹ để chữ rõ hơn nhưng không làm mất màu/chi tiết
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
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