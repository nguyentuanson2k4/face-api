"""
Face Recognition Microservice - FastAPI server
Provides endpoints for face detection, embedding extraction, and identification.
Designed to be called by the NestJS backend.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from face_engine import FaceEngine
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Face Recognition Service",
    description="Microservice for face detection and recognition using InsightFace (ArcFace)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== REQUEST / RESPONSE MODELS =====================


class CompareRequest(BaseModel):
    embedding1: list[float]
    embedding2: list[float]


class KnownEmbedding(BaseModel):
    id: int
    student_id: int
    embedding: list[float]


class IdentifyRequest(BaseModel):
    known_embeddings: list[KnownEmbedding]
    threshold: float = 0.6


# ===================== LIFECYCLE =====================


@app.on_event("startup")
async def startup():
    """Pre-load model on startup."""
    logger.info("Loading face recognition model...")
    FaceEngine.get_instance()
    logger.info("Face recognition model ready!")


# ===================== ENDPOINTS =====================


@app.get("/health")
async def health_check():
    return {"status": "ok", "model": "insightface/buffalo_l"}


@app.post("/detect-faces")
async def detect_faces(file: UploadFile = File(...)):
    """Detect all faces in an uploaded image."""
    try:
        image_bytes = await file.read()
        engine = FaceEngine.get_instance()
        faces = engine.detect_faces(image_bytes)
        return {
            "success": True,
            "face_count": len(faces),
            "faces": faces,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")


@app.post("/extract-embedding")
async def extract_embedding(file: UploadFile = File(...)):
    """Extract 512-dim face embedding from image with exactly 1 face."""
    try:
        image_bytes = await file.read()
        engine = FaceEngine.get_instance()
        result = engine.extract_embedding(image_bytes)
        return {
            "success": True,
            **result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")


@app.post("/extract-all-embeddings")
async def extract_all_embeddings(file: UploadFile = File(...)):
    """Extract embeddings for ALL faces in image (group photo)."""
    try:
        image_bytes = await file.read()
        engine = FaceEngine.get_instance()
        results = engine.extract_all_embeddings(image_bytes)
        return {
            "success": True,
            "face_count": len(results),
            "faces": results,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error extracting embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")


@app.post("/compare-faces")
async def compare_faces(request: CompareRequest):
    """Compare two face embeddings and return cosine similarity."""
    try:
        if len(request.embedding1) != 512 or len(request.embedding2) != 512:
            raise ValueError("Mỗi embedding phải có đúng 512 chiều.")

        engine = FaceEngine.get_instance()
        similarity = engine.compare_embeddings(request.embedding1, request.embedding2)
        return {
            "success": True,
            "similarity": similarity,
            "is_same_person": similarity >= 0.6,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error comparing faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/identify")
async def identify(
    file: UploadFile = File(...),
    known_embeddings: str = "",
    threshold: float = 0.6,
):
    """
    Identify faces in image against known embeddings.
    known_embeddings should be JSON string of [{id, student_id, embedding}, ...]
    """
    import json

    try:
        image_bytes = await file.read()

        if not known_embeddings:
            raise ValueError("known_embeddings is required")

        embeddings_data = json.loads(known_embeddings)
        engine = FaceEngine.get_instance()
        results = engine.identify(image_bytes, embeddings_data, threshold)

        return {
            "success": True,
            "face_count": len(results),
            "results": results,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in known_embeddings")
    except Exception as e:
        logger.error(f"Error identifying: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi nhận diện: {str(e)}")


@app.post("/identify-json")
async def identify_json(file: UploadFile = File(...)):
    """
    Alternative identify endpoint - embeddings sent as form data.
    This is the preferred endpoint called from NestJS.
    """
    # This endpoint is handled by the NestJS service which sends
    # the image to extract-embedding first, then does matching in NestJS
    pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
