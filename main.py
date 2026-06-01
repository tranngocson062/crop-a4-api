import base64
import io
from typing import Optional, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image


app = FastAPI(title="Crop A4 Document API")

A4_WIDTH_PX = 2480
A4_HEIGHT_PX = 3508


class CropRequest(BaseModel):
    filename: Optional[str] = "image.jpg"
    mimeType: Optional[str] = "image/jpeg"
    fileBase64: str
    output: Optional[Dict[str, Any]] = None


@app.get("/")
def health_check():
    return {
        "ok": True,
        "service": "Crop A4 Document API"
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

        cropped, detected = scan_document_to_a4(
            image=image,
            output_width=width_px,
            output_height=height_px,
            enhance_text=enhance_text
        )

        output_bytes = encode_jpeg(cropped, quality=95)
        output_base64 = base64.b64encode(output_bytes).decode("utf-8")

        return {
            "filename": make_output_name(payload.filename or "image.jpg"),
            "mimeType": "image/jpeg",
            "fileBase64": output_base64,
            "detectedDocument": detected
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


def scan_document_to_a4(image, output_width: int, output_height: int, enhance_text: bool):
    original = image.copy()

    ratio = image.shape[0] / 700.0
    resized = resize_to_height(image, 700)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 75, 200)

    contours, _ = cv2.findContours(
        edged.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    document_contour = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            document_contour = approx.reshape(4, 2)
            break

    if document_contour is not None:
        points = document_contour.astype("float32") * ratio
        warped = four_point_transform(
            original,
            points,
            output_width,
            output_height
        )
        detected = True
    else:
        warped = center_crop_to_a4(original, output_width, output_height)
        detected = False

    if enhance_text:
        warped = enhance_document(warped)

    return warped, detected


def resize_to_height(image, height: int):
    h, w = image.shape[:2]
    ratio = height / float(h)
    width = int(w * ratio)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


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


def center_crop_to_a4(image, output_width: int, output_height: int):
    target_ratio = output_width / output_height
    h, w = image.shape[:2]
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x1 = max((w - new_w) // 2, 0)
        cropped = image[:, x1:x1 + new_w]
    else:
        new_h = int(w / target_ratio)
        y1 = max((h - new_h) // 2, 0)
        cropped = image[y1:y1 + new_h, :]

    return cv2.resize(cropped, (output_width, output_height), interpolation=cv2.INTER_AREA)


def enhance_document(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced


def encode_jpeg(image, quality: int = 95):
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