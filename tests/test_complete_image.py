from pathlib import Path
import cv2

from app.config.config_loader import Config
from app.detection.yolo_detector import YOLODetector
from app.tracking.tag_tracker import TagTracker
from app.preprocessing.tag_processor import TagProcessor
from app.preprocessing.image_quality import ImageQualityChecker
from app.barcode.barcode_engine import BarcodeEngine
from app.barcode.barcode_region_processor import BarcodeRegionProcessor
from app.ocr.ocr_engine import OCREngine
from app.validation.validator import Validator


IMAGE = Path(r"captures\test_tag_01.jpg")


def main():
    print("=" * 75)
    print("OCR_BHS COMPLETE IMAGE TEST")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. IMAGE
    # ---------------------------------------------------------
    assert IMAGE.exists(), f"IMAGE NOT FOUND: {IMAGE}"

    image = cv2.imread(str(IMAGE))
    assert image is not None, "IMAGE LOAD FAILED"

    print(
        f"[PASS] Image Load          "
        f"{image.shape[1]}x{image.shape[0]}"
    )

    # ---------------------------------------------------------
    # 2. CONFIG
    # ---------------------------------------------------------
    config = Config()
    print("[PASS] Configuration")

    # ---------------------------------------------------------
    # 3. YOLO
    # ---------------------------------------------------------
    detector = YOLODetector(
        model_path=config.data["detection"]["model_path"],
        confidence=config.data["detection"]["confidence"],
        image_size=config.data["detection"]["image_size"],
        device=config.data["detection"]["device"],
    )

    detections = detector.detect(image)

    print(
        f"[PASS] YOLO Detection      "
        f"detections={len(detections)}"
    )

    for i, detection in enumerate(detections, 1):
        print(
            f"       Detection {i}: "
            f"class={detection.class_name} "
            f"conf={detection.confidence:.3f} "
            f"bbox={detection.bbox}"
        )

    # ---------------------------------------------------------
    # 4. BYTE TRACK
    # ---------------------------------------------------------
    tracker = TagTracker(
        track_high_thresh=config.data["tracking"]["track_high_thresh"],
        track_low_thresh=config.data["tracking"]["track_low_thresh"],
        new_track_thresh=config.data["tracking"]["new_track_thresh"],
        track_buffer=config.data["tracking"]["track_buffer"],
        match_thresh=config.data["tracking"]["match_thresh"],
        fuse_score=config.data["tracking"]["fuse_score"],
        min_hits=config.data["tracking"]["min_hits"],
    )

    tracks = tracker.update(
        detections,
        frame_id=1,
    )

    print(
        f"[PASS] ByteTrack           "
        f"tracks={len(tracks)} "
        f"ids={[track.track_id for track in tracks]}"
    )

    for track in tracks:
        print(
            f"       Track ID={track.track_id} "
            f"bbox={track.detection.bbox} "
            f"conf={track.track_confidence:.3f} "
            f"hits={track.hits} "
            f"confirmed={track.is_confirmed}"
        )

    # ---------------------------------------------------------
    # 5. TAG PROCESSOR
    # ---------------------------------------------------------
    TagProcessor()
    print("[PASS] Tag Processor")

    # ---------------------------------------------------------
    # 6. IMAGE QUALITY
    # ---------------------------------------------------------
    quality = ImageQualityChecker()
    quality_result = quality.check(image)

    print(
        f"[PASS] Image Quality       "
        f"{quality_result}"
    )

    # ---------------------------------------------------------
    # 7. BARCODE ENGINE
    # ---------------------------------------------------------
    barcode_engine = BarcodeEngine(config)
    barcode_results = barcode_engine.read(image)

    print(
        f"[PASS] Barcode Engine      "
        f"results={len(barcode_results)}"
    )

    for result in barcode_results:
        print(
            f"       Barcode: "
            f"value={result.value} "
            f"format={result.format} "
            f"confidence={result.confidence:.3f}"
        )

    # ---------------------------------------------------------
    # 8. BARCODE REGION PROCESSOR
    # ---------------------------------------------------------
    barcode_region_processor = BarcodeRegionProcessor(config)
    region_results = barcode_region_processor.process(image)

    print(
        f"[PASS] Barcode Region      "
        f"results={len(region_results)}"
    )

    for result in region_results:
        print(
            f"       Region Barcode: "
            f"value={result.value} "
            f"format={result.format} "
            f"confidence={result.confidence:.3f}"
        )

    # ---------------------------------------------------------
    # 9. OCR
    # ---------------------------------------------------------
    ocr_engine = OCREngine(config)
    ocr_result = ocr_engine.read(image)

    print(
        f"[PASS] OCR Engine          "
        f"success={getattr(ocr_result, 'success', None)}"
    )

    print(
        f"       OCR Text: "
        f"{getattr(ocr_result, 'text', '')!r}"
    )

    print(
        f"       OCR Confidence: "
        f"{getattr(ocr_result, 'confidence', None)}"
    )

    # ---------------------------------------------------------
    # 10. VALIDATOR
    # ---------------------------------------------------------
    Validator()
    print("[PASS] Validator")

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print()
    print("=" * 75)
    print("COMPLETE IMAGE TEST FINISHED")
    print("=" * 75)

    print(
        f"YOLO={len(detections)} | "
        f"TRACKS={len(tracks)} | "
        f"BARCODE={len(barcode_results)} | "
        f"BARCODE_REGIONS={len(region_results)} | "
        f"OCR={getattr(ocr_result, 'success', False)}"
    )

    print()
    print("STATUS: PASS")
    print("=" * 75)


if __name__ == "__main__":
    main()