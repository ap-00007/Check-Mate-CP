from app.ml.pipeline import analyze_image_pair

async def verify_handwriting(reference_bytes: bytes, comparison_bytes: bytes):
    result = analyze_image_pair(reference_bytes, comparison_bytes)
    return result
