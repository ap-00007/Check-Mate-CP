from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.verification import VerificationResponse
from app.services.verification_service import verify_handwriting
from app.core.config import settings

router = APIRouter()

MAX_BYTES = settings.max_upload_mb * 1024 * 1024

@router.post("/verify", response_model=VerificationResponse)
async def verify(
    reference_image: UploadFile = File(...),
    comparison_image: UploadFile = File(...),
):
    allowed = {"image/png", "image/jpeg", "image/webp"}
    if reference_image.content_type not in allowed or comparison_image.content_type not in allowed:
        raise HTTPException(status_code=415, detail="Upload PNG, JPEG, or WEBP images.")

    ref_bytes = await reference_image.read()
    cmp_bytes = await comparison_image.read()

    if not ref_bytes or not cmp_bytes:
        raise HTTPException(status_code=400, detail="Both images are required.")

    if len(ref_bytes) > MAX_BYTES or len(cmp_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Each image must be under {settings.max_upload_mb} MB."
        )

    result = await verify_handwriting(ref_bytes, cmp_bytes)
    return result
