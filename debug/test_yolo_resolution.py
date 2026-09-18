from pathlib import Path
import sys
import cv2

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.config_loader import load_config
from app.detection.yolo_detector import YOLODetector


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YOLO26 GOLDEN IMAGE - RESOLUTION TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    config = load_config()

    model_path = config.model_path
    device = config.device

    print(f"Model      : {model_path}")
    print(f"Device     : {device}")

    # --------------------------------------------------------
    # GOLDEN IMAGE
    # --------------------------------------------------------

    IMAGE_PATH = PROJECT_ROOT / "captures" / "tags" / "golden_iata.jpg"

    if not IMAGE_PATH.exists():
        print(f"\nERROR: Image not found:")
        print(IMAGE_PATH)
        return

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print(f"\nERROR: Could not load image:")
        print(IMAGE_PATH)
        return

    print(f"Test image : {IMAGE_PATH}")
    print(f"Resolution : {image.shape[1]}x{image.shape[0]}")

    # --------------------------------------------------------
    # TEST SETTINGS
    # --------------------------------------------------------

    test_resolutions = [
        512,
        640,
        832,
        1024,
        1280,
    ]

    confidence = 0.01

    print(f"Confidence : {confidence}")
    print()
    print("-" * 70)

    # --------------------------------------------------------
    # TEST EACH RESOLUTION
    # --------------------------------------------------------

    results = []

    for resolution in test_resolutions:

        print()
        print(f"TESTING IMAGE SIZE: {resolution}")
        print("-" * 50)

        try:

            detector = YOLODetector(
                model_path=model_path,
                confidence=confidence,
                image_size=resolution,
                device=device,
            )

            detections = detector.detect(image)

            count = len(detections)

            print(f"Detections : {count}")

            if count > 0:

                for i, detection in enumerate(detections, start=1):

                    print(
                        f"  Detection {i}: "
                        f"class={detection.class_name} "
                        f"conf={detection.confidence:.4f} "
                        f"bbox={detection.bbox}"
                    )

            else:
                print("  No detections.")

            results.append(
                {
                    "resolution": resolution,
                    "detections": count,
                }
            )

        except Exception as exc:

            print(f"ERROR at resolution {resolution}:")
            print(exc)

            results.append(
                {
                    "resolution": resolution,
                    "detections": -1,
                }
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESOLUTION TEST SUMMARY")
    print("=" * 70)

    print(f"{'Resolution':<15}{'Detections':<15}")
    print("-" * 30)

    for result in results:

        resolution = result["resolution"]
        detections = result["detections"]

        print(f"{resolution:<15}{detections:<15}")

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()