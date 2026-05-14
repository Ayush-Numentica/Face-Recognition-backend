"""
recognition.py
--------------
Core Face Recognition Engine.

Pipeline
--------
1. Detect faces with MTCNN (gives accurate bounding boxes + landmarks).
2. Crop & preprocess each face to 160×160 RGB.
3. Generate a 512-D embedding with the pre-trained FaceNet model
   (keras-facenet – NOT trained from scratch).
4. Compare new embeddings against all stored embeddings via Euclidean
   distance.  Average distance per person → pick the closest.
5. If the closest average distance exceeds THRESHOLD → "Unknown".
"""

import os
import pickle
import shutil
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

logger = logging.getLogger(__name__)


class FaceRecognitionEngine:
    """
    Wraps MTCNN + FaceNet into a simple recognize / add-person / delete-person
    API that Flask routes can call directly.
    """

    # Minimum confidence (0–1) required to accept a match.
    # confidence = exp(−euclidean_distance)
    # 0.50 ≈ distance 0.69  |  0.55 ≈ distance 0.60  |  0.60 ≈ distance 0.51
    CONFIDENCE_THRESHOLD: float = 0.45

    # FaceNet input size (fixed by the model architecture).
    FACE_SIZE: int = 160

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self, dataset_dir: str, embeddings_dir: str) -> None:
        self.dataset_dir   = dataset_dir
        self.embeddings_dir = embeddings_dir
        self.embeddings_file = os.path.join(embeddings_dir, "embeddings.pkl")

        # { "Ayush": [emb_array_1, emb_array_2, ...], ... }
        self.known_embeddings: Dict[str, List[np.ndarray]] = {}

        # Precomputed mean embedding per person — used at inference time
        # so matching is a single vector comparison instead of N comparisons.
        # { "Ayush": mean_emb_array }
        self._mean_embeddings: Dict[str, np.ndarray] = {}

        logger.info("Loading MTCNN face detector …")
        self.detector = MTCNN()

        logger.info("Loading FaceNet (keras-facenet) …")
        self.facenet = FaceNet()

        logger.info("FaceRecognitionEngine ready.")

    # ------------------------------------------------------------------ #
    # Low-level helpers                                                    #
    # ------------------------------------------------------------------ #

    def _crop_face(
        self,
        rgb: np.ndarray,
        img_shape: tuple,
        box: list,
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
        """Crop and pad one MTCNN bounding box from an RGB image."""
        x, y, w, h = box
        x, y = max(0, x), max(0, y)
        w = min(img_shape[1] - x, w)
        h = min(img_shape[0] - y, h)
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_shape[1], x + w + pad)
        y2 = min(img_shape[0], y + h + pad)
        face_rgb = rgb[y1:y2, x1:x2]
        if face_rgb.size == 0:
            return None, (x1, y1, x2 - x1, y2 - y1)
        return face_rgb, (x1, y1, x2 - x1, y2 - y1)

    def _detect_face(
        self, bgr_image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Detect the single highest-confidence face in *bgr_image*."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb)
        if not results:
            return None, None
        best = max(results, key=lambda r: r["confidence"])
        face_rgb, box = self._crop_face(rgb, bgr_image.shape, best["box"])
        if face_rgb is None:
            return None, None
        return face_rgb, box

    def _detect_all_faces(
        self, bgr_image: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect ALL faces in *bgr_image* with MTCNN confidence ≥ 0.90.

        Returns
        -------
        List of (face_rgb, (x, y, w, h)), one per detected face.
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb)
        faces = []
        for det in results:
            if det["confidence"] < 0.70:
                continue
            face_rgb, box = self._crop_face(rgb, bgr_image.shape, det["box"])
            if face_rgb is not None:
                faces.append((face_rgb, box))
        return faces

    def _preprocess(self, face_rgb: np.ndarray) -> np.ndarray:
        """Resize face to 160×160 and return as float32 array."""
        resized = cv2.resize(face_rgb, (self.FACE_SIZE, self.FACE_SIZE))
        return resized.astype(np.float32)

    def _embed(self, face_float: np.ndarray) -> np.ndarray:
        """
        Run FaceNet on a single preprocessed face and return the 512-D
        embedding.  keras-facenet handles its own internal normalisation.
        """
        batch = np.expand_dims(face_float, axis=0)   # (1, 160, 160, 3)
        return self.facenet.embeddings(batch)[0]      # (512,)

    @staticmethod
    def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    # ------------------------------------------------------------------ #
    # Public API – called by Flask routes                                  #
    # ------------------------------------------------------------------ #

    def recognize(self, bgr_image: np.ndarray) -> dict:
        """
        Detect and identify the most prominent face in *bgr_image*.

        Returns a dict suitable for JSON serialisation:
        {
            "detected":   bool,
            "name":       str,      # person name or "Unknown"
            "confidence": float,    # 0-1  (exp(-distance))
            "box":        [x,y,w,h] | null
        }
        """
        face, box = self._detect_face(bgr_image)

        if face is None:
            return {"detected": False, "name": "No face detected",
                    "confidence": 0.0, "box": None}

        embedding = self._embed(self._preprocess(face))

        if not self.known_embeddings:
            return {"detected": True, "name": "Unknown",
                    "confidence": 0.0, "box": list(box)}

        match = self._match_embedding(embedding)
        logger.info(
            "Recognised: %s  dist=%.4f  conf=%.4f  threshold=%.2f",
            match["name"], match["distance"], match["confidence"], self.CONFIDENCE_THRESHOLD,
        )
        return {
            "detected":   True,
            "name":       match["name"],
            "confidence": match["confidence"],
            "box":        list(box),
        }

    def _match_embedding(self, embedding: np.ndarray) -> dict:
        """
        Compare one embedding against the mean embedding of each known person.
        Using the mean is O(persons) instead of O(persons × images) — much faster.
        """
        best_name       = "Unknown"
        best_distance   = float("inf")
        best_confidence = 0.0

        for name, mean_emb in self._mean_embeddings.items():
            distance   = self._euclidean(embedding, mean_emb)
            confidence = float(np.exp(-distance))

            if confidence > best_confidence:
                best_confidence = confidence
                best_distance   = distance
                best_name       = name

        if best_confidence < self.CONFIDENCE_THRESHOLD:
            best_name = "Unknown"

        return {
            "name":       best_name,
            "confidence": round(best_confidence, 4),
            "distance":   round(best_distance, 4),
        }

    def recognize_all(self, bgr_image: np.ndarray) -> List[dict]:
        """
        Detect and identify ALL faces in *bgr_image*.

        FaceNet is called once with a batch containing all detected faces,
        which is significantly faster than one call per face.

        Returns a list of result dicts sorted by confidence descending.
        """
        faces = self._detect_all_faces(bgr_image)
        if not faces:
            return []

        # ── Batch embed all detected faces in one FaceNet forward pass ──
        face_arrays = [self._preprocess(f) for f, _ in faces]
        batch       = np.stack(face_arrays, axis=0)          # (N, 160, 160, 3)
        embeddings  = self.facenet.embeddings(batch)          # (N, 512)

        results = []
        for (_, box), embedding in zip(faces, embeddings):
            if not self._mean_embeddings:
                results.append({
                    "detected":   True,
                    "name":       "Unknown",
                    "confidence": 0.0,
                    "box":        list(box),
                })
                continue

            match = self._match_embedding(embedding)
            logger.info(
                "Face at box=%s → %s  conf=%.4f  dist=%.4f",
                box, match["name"], match["confidence"], match["distance"],
            )
            results.append({
                "detected":   True,
                "name":       match["name"],
                "confidence": match["confidence"],
                "box":        list(box),
            })

        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results

    # ------------------------------------------------------------------ #
    # Dataset / embedding management                                       #
    # ------------------------------------------------------------------ #

    def _generate_embeddings_for_person(self, name: str) -> List[np.ndarray]:
        """
        Process every image inside *dataset/<name>/* with MTCNN + FaceNet
        and return the resulting list of embeddings.
        """
        person_dir = os.path.join(self.dataset_dir, name)
        if not os.path.isdir(person_dir):
            logger.warning("Dataset folder missing for '%s'", name)
            return []

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [
            f for f in os.listdir(person_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]

        if not image_files:
            logger.warning("No images found for '%s'", name)
            return []

        embeddings: List[np.ndarray] = []
        for filename in image_files:
            path  = os.path.join(person_dir, filename)
            image = cv2.imread(path)
            if image is None:
                logger.warning("Cannot read '%s' — skipping.", path)
                continue

            face, _ = self._detect_face(image)
            if face is None:
                logger.warning("No face detected in '%s' — skipping.", filename)
                continue

            emb = self._embed(self._preprocess(face))
            embeddings.append(emb)
            logger.info("  Embedded %s / %s", name, filename)

        return embeddings

    def _rebuild_mean_embeddings(self) -> None:
        """Recompute _mean_embeddings from known_embeddings."""
        self._mean_embeddings = {
            name: np.mean(embs, axis=0)
            for name, embs in self.known_embeddings.items()
            if embs
        }

    def add_person(self, name: str) -> bool:
        """
        (Re-)build embeddings for *name* from images already saved in
        *dataset/<name>/*.

        Returns True on success, False if no embeddings could be generated.
        """
        embeddings = self._generate_embeddings_for_person(name)
        if not embeddings:
            return False

        self.known_embeddings[name] = embeddings
        self._rebuild_mean_embeddings()
        self._save_embeddings()
        logger.info("Added '%s' with %d embedding(s).", name, len(embeddings))
        return True

    def delete_person(self, name: str) -> bool:
        """
        Remove *name* from in-memory embeddings, disk embeddings file, and
        the dataset folder.

        Returns True if found and removed, False if the name was unknown.
        """
        if name not in self.known_embeddings:
            return False

        del self.known_embeddings[name]
        self._rebuild_mean_embeddings()
        self._save_embeddings()

        person_dir = os.path.join(self.dataset_dir, name)
        if os.path.isdir(person_dir):
            shutil.rmtree(person_dir)

        logger.info("Deleted '%s'.", name)
        return True

    def rebuild_all_embeddings(self) -> int:
        """
        Scan the entire *dataset/* directory and rebuild all embeddings.

        Returns the number of persons successfully processed.
        """
        self.known_embeddings = {}
        if not os.path.isdir(self.dataset_dir):
            return 0

        count = 0
        for person_name in sorted(os.listdir(self.dataset_dir)):
            person_path = os.path.join(self.dataset_dir, person_name)
            if not os.path.isdir(person_path):
                continue
            embeddings = self._generate_embeddings_for_person(person_name)
            if embeddings:
                self.known_embeddings[person_name] = embeddings
                count += 1

        self._rebuild_mean_embeddings()
        self._save_embeddings()
        logger.info("Rebuilt embeddings for %d person(s).", count)
        return count

    def get_known_persons(self) -> List[dict]:
        """Return a list of dicts describing every known person."""
        result = []
        for name, embs in self.known_embeddings.items():
            person_dir = os.path.join(self.dataset_dir, name)
            img_count  = 0
            if os.path.isdir(person_dir):
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                img_count  = sum(
                    1 for f in os.listdir(person_dir)
                    if os.path.splitext(f)[1].lower() in valid_exts
                )
            result.append({
                "name":             name,
                "images_count":     img_count,
                "embeddings_count": len(embs),
            })
        return result

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _save_embeddings(self) -> None:
        os.makedirs(self.embeddings_dir, exist_ok=True)
        with open(self.embeddings_file, "wb") as fh:
            pickle.dump(self.known_embeddings, fh)
        logger.info(
            "Saved embeddings for %d person(s) → %s",
            len(self.known_embeddings), self.embeddings_file,
        )

    def load_embeddings(self) -> None:
        """
        Load embeddings from disk.  If none exist, attempt to build them
        from the dataset directory automatically.
        """
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, "rb") as fh:
                self.known_embeddings = pickle.load(fh)
            self._rebuild_mean_embeddings()
            logger.info(
                "Loaded embeddings for: %s",
                list(self.known_embeddings.keys()),
            )
        else:
            logger.info("No saved embeddings found.")
            # Auto-build from dataset if images are already present
            if os.path.isdir(self.dataset_dir) and os.listdir(self.dataset_dir):
                logger.info("Dataset found — building embeddings automatically …")
                self.rebuild_all_embeddings()
