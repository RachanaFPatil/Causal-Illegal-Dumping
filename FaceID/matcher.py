"""FaceID/matcher.py — Cosine similarity face matching against offender DB."""

import cv2
import numpy as np
from typing import Optional
from FaceID.database import load_all_offenders, increment_violation_count

SIMILARITY_THRESHOLD = 0.45   # tuned for buffalo_l 512-dim embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b))


def match_face(embedding: np.ndarray) -> Optional[dict]:
    offenders = load_all_offenders()
    if not offenders:
        return None
    best_match, best_score = None, -1.0
    for person in offenders:
        score = cosine_similarity(embedding, person["embedding"])
        if score > best_score:
            best_score = score
            best_match = person
    if best_score >= SIMILARITY_THRESHOLD:
        increment_violation_count(best_match["id"])
        return {
            "id":         best_match["id"],
            "name":       best_match["name"],
            "email":      best_match["email"],
            "phone":      best_match["phone"],
            "address":    best_match.get("address", ""),
            "uid_ref":    best_match["uid_ref"],
            "similarity": round(best_score, 4),
        }
    return None


def get_embedding(app, frame: np.ndarray) -> Optional[tuple]:
    """Returns (embedding, bbox, face_crop) or None if no face detected."""
    faces = app.get(frame)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    embedding = face.embedding.astype(np.float32)
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    face_crop = frame[y1:y2, x1:x2]
    return embedding, face.bbox, face_crop