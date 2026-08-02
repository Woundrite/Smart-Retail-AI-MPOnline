from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.auth import verify_api_key
from app.schemas import (
    FaceEnrollResponse,
    FaceRecognitionResponse,
    ProductClassificationResponse,
    ProductEnrollResponse,
)
from app.services.cv_service import (
    EnrollmentError,
    ModelNotTrainedError,
    face_recognition_service,
    product_classifier,
)

router = APIRouter(tags=["Computer Vision"])


@router.post("/classify-product", response_model=ProductClassificationResponse, dependencies=[Depends(verify_api_key)])
async def classify_product(file: UploadFile = File(..., description="Product image (jpg/png)")):
    image_bytes = await file.read()
    try:
        label, confidence, all_scores = product_classifier.predict(image_bytes)
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ProductClassificationResponse(category=label, confidence=confidence, all_scores=all_scores)


@router.post("/recognize-face", response_model=FaceRecognitionResponse, dependencies=[Depends(verify_api_key)])
async def recognize_face(file: UploadFile = File(..., description="Frame/photo containing a customer's face")):
    image_bytes = await file.read()
    try:
        result = face_recognition_service.recognize(image_bytes)
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return FaceRecognitionResponse(**result)


@router.post("/enroll-product", response_model=ProductEnrollResponse, dependencies=[Depends(verify_api_key)])
async def enroll_product(
    label: str = Form(..., description="Name of the new (or existing online-enrolled) product type"),
    files: List[UploadFile] = File(..., description="At least a few sample photos of the product"),
):
    """
    Add a product type on the fly — no retraining or redeploying the model.
    Embeds each photo with the frozen MobileNetV2 backbone and folds it into
    a running per-class centroid, so this is cheap even for frequent updates.
    """
    image_bytes_list = [await f.read() for f in files]
    try:
        result = product_classifier.enroll(label, image_bytes_list)
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (EnrollmentError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ProductEnrollResponse(**result)


@router.post("/enroll-face", response_model=FaceEnrollResponse, dependencies=[Depends(verify_api_key)])
async def enroll_face(
    person_name: str = Form(..., description="Name/ID of the new (or existing) person"),
    files: List[UploadFile] = File(..., description="At least a few clear, front-facing photos"),
):
    """
    Add a person on the fly using LBPH's native incremental update — no
    full retrain of face_db.pkl required.
    """
    image_bytes_list = [await f.read() for f in files]
    try:
        result = face_recognition_service.enroll(person_name, image_bytes_list)
    except (EnrollmentError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return FaceEnrollResponse(**result)
