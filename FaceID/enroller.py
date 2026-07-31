"""FaceID/enroller.py — Register a person into the offender face database.

Usage:
    python -m FaceID.enroller --image path/to/photo.jpg \\
        --name "Full Name" --email "x@y.com" --phone "9901967521" \\
        --uid "UID123456" --address "Bengaluru, KA"
"""

import argparse
import cv2
import numpy as np

from FaceID.database import init_db, save_offender


def _load_model():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def enroll(image_path: str, name: str, email: str,
           phone: str, uid: str, address: str = "") -> None:
    init_db()
    app = _load_model()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    faces = app.get(img)
    if not faces:
        print("[Enroller] ❌ No face detected.")
        return
    if len(faces) > 1:
        print(f"[Enroller] ⚠ {len(faces)} faces — using largest.")
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    embedding = face.embedding.astype(np.float32)
    oid = save_offender(name, email, phone, uid, embedding, address)
    print(f"[Enroller] ✅ Registered '{name}' id={oid}  norm={np.linalg.norm(embedding):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   required=True)
    parser.add_argument("--name",    required=True)
    parser.add_argument("--email",   default="")
    parser.add_argument("--phone",   default="")
    parser.add_argument("--uid",     default="")
    parser.add_argument("--address", default="")
    args = parser.parse_args()
    enroll(args.image, args.name, args.email, args.phone, args.uid, args.address)