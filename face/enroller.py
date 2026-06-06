"""
FaceID/enrollor.py
==================
Register a person into the FaceID offender database.
Give it a clear photo (passport-style or any frontal face photo).

Usage:
    python FaceID/enrollor.py --image path/to/photo.jpg \
                               --name "Full Name" \
                               --email "email@example.com" \
                               --phone "9901967521" \
                               --uid   "UID123456"
"""

import argparse
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

from FaceID.database import init_db, save_offender


def load_model() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def enroll(image_path: str, name: str, email: str,
           phone: str, uid: str) -> None:
    init_db()
    app = load_model()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    faces = app.get(img)
    if not faces:
        print("[Enrollor] ❌ No face detected in the image. Use a clearer photo.")
        return
    if len(faces) > 1:
        print(f"[Enrollor] ⚠️  {len(faces)} faces found — using the largest one.")

    # Pick the largest face by bounding box area
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    embedding = face.embedding.astype(np.float32)

    offender_id = save_offender(name, email, phone, uid, embedding)
    print(f"[Enrollor] ✅ Registered '{name}' with ID={offender_id}")
    print(f"           email={email}  phone={phone}  uid={uid}")
    print(f"           embedding shape={embedding.shape}  norm={np.linalg.norm(embedding):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--name",  required=True)
    parser.add_argument("--email", default="")
    parser.add_argument("--phone", default="")
    parser.add_argument("--uid",   default="")
    args = parser.parse_args()

    enroll(args.image, args.name, args.email, args.phone, args.uid)