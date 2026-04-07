"""
Face Recognition Engine - Wrapper around InsightFace for face detection and embedding extraction.
Uses ArcFace model which produces 512-dimensional face embeddings.
"""

import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FaceEngine:
    """Singleton wrapper around InsightFace FaceAnalysis."""

    _instance: Optional["FaceEngine"] = None

    def __init__(self):
        logger.info("Initializing InsightFace model (buffalo_l)...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        # det_size: kích thước ảnh đầu vào cho face detection
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace model loaded successfully.")

    @classmethod
    def get_instance(cls) -> "FaceEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _read_image(self, image_bytes: bytes) -> np.ndarray:
        """Convert raw bytes to OpenCV image (BGR)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không thể đọc ảnh. Vui lòng kiểm tra file ảnh.")
        return img

    def detect_faces(self, image_bytes: bytes) -> list[dict]:
        """
        Detect all faces in an image.
        Returns list of dicts with bbox, confidence, landmarks.
        """
        img = self._read_image(image_bytes)
        faces = self.app.get(img)

        results = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            results.append({
                "bbox": {
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                },
                "confidence": float(face.det_score),
                "age": int(face.age) if hasattr(face, "age") and face.age else None,
                "gender": "male" if hasattr(face, "gender") and face.gender == 1 else "female" if hasattr(face, "gender") and face.gender is not None else None,
            })

        return results

    def extract_embedding(self, image_bytes: bytes) -> dict:
        """
        Extract 512-dim face embedding from image.
        Expects exactly 1 face. Raises if 0 or >1 faces found.
        """
        img = self._read_image(image_bytes)
        faces = self.app.get(img)

        if len(faces) == 0:
            raise ValueError("Không phát hiện khuôn mặt nào trong ảnh.")
        if len(faces) > 1:
            raise ValueError(
                f"Phát hiện {len(faces)} khuôn mặt. Vui lòng upload ảnh chỉ có 1 khuôn mặt."
            )

        face = faces[0]
        embedding = face.normed_embedding.tolist()  # 512-dim normalized vector
        bbox = face.bbox.astype(int).tolist()

        return {
            "embedding": embedding,
            "confidence": float(face.det_score),
            "bbox": {
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
            },
        }

    def extract_all_embeddings(self, image_bytes: bytes) -> list[dict]:
        """
        Extract embeddings for ALL faces in image (for group photo attendance).
        """
        img = self._read_image(image_bytes)
        faces = self.app.get(img)

        if len(faces) == 0:
            raise ValueError("Không phát hiện khuôn mặt nào trong ảnh.")

        results = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            results.append({
                "embedding": face.normed_embedding.tolist(),
                "confidence": float(face.det_score),
                "bbox": {
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                },
            })

        return results

    def compare_embeddings(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """Compute cosine similarity between two 512-dim embeddings."""
        e1 = np.array(embedding1).reshape(1, -1)
        e2 = np.array(embedding2).reshape(1, -1)
        similarity = cosine_similarity(e1, e2)[0][0]
        return float(similarity)

    def identify(
        self,
        image_bytes: bytes,
        known_embeddings: list[dict],
        threshold: float = 0.6,
    ) -> list[dict]:
        """
        Identify faces in image against a list of known embeddings.

        Args:
            image_bytes: raw image bytes
            known_embeddings: list of {"id": ..., "embedding": [...512 floats]}
            threshold: minimum cosine similarity to consider a match

        Returns:
            List of matched results with student info and similarity score.
        """
        img = self._read_image(image_bytes)
        faces = self.app.get(img)

        if len(faces) == 0:
            raise ValueError("Không phát hiện khuôn mặt nào trong ảnh.")

        if not known_embeddings:
            raise ValueError("Danh sách embeddings trống. Chưa có sinh viên nào đăng ký khuôn mặt.")

        # Build matrix of known embeddings for vectorized comparison
        known_ids = [item["id"] for item in known_embeddings]
        known_student_ids = [item["student_id"] for item in known_embeddings]
        known_matrix = np.array([item["embedding"] for item in known_embeddings])

        results = []
        for face in faces:
            query = face.normed_embedding.reshape(1, -1)
            similarities = cosine_similarity(query, known_matrix)[0]

            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            bbox = face.bbox.astype(int).tolist()

            if best_score >= threshold:
                results.append({
                    "matched": True,
                    "face_id": known_ids[best_idx],
                    "student_id": known_student_ids[best_idx],
                    "similarity": best_score,
                    "confidence": float(face.det_score),
                    "bbox": {
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],
                    },
                })
            else:
                results.append({
                    "matched": False,
                    "best_similarity": best_score,
                    "confidence": float(face.det_score),
                    "bbox": {
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],
                    },
                })

        return results
