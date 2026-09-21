import cv2
import os
import time
from datetime import datetime

# =========================
# CONFIGURATION
# =========================

RTSP_URL = "rtsp://beumer:Beumer!123@172.20.45.131:554/video/live?channel=1&subtype=1"

OUTPUT_DIR = "rtsp_frames"

# Capture one frame every N seconds
CAPTURE_INTERVAL = 1.0

# Save JPEG quality: 0-100
JPEG_QUALITY = 95

# Reconnect automatically if RTSP disconnects
RECONNECT_DELAY = 3

# =========================
# SETUP
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = None
last_capture_time = 0

print("=" * 60)
print("RTSP FRAME CAPTURE")
print("=" * 60)
print(f"Output directory : {os.path.abspath(OUTPUT_DIR)}")
print(f"Capture interval : {CAPTURE_INTERVAL} sec")
print("Press Q to quit")
print("=" * 60)


def connect_camera():
    global cap

    if cap is not None:
        cap.release()

    print("\nConnecting to RTSP...")

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    # Reduce internal buffering for live RTSP
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print("RTSP connected")
        print(f"Resolution : {width}x{height}")
        print(f"FPS        : {fps:.2f}")

        return True

    print("RTSP connection failed")
    return False


# =========================
# MAIN LOOP
# =========================

while True:

    if cap is None or not cap.isOpened():

        if not connect_camera():
            time.sleep(RECONNECT_DELAY)
            continue

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Frame read failed. Reconnecting...")
        cap.release()
        cap = None
        time.sleep(RECONNECT_DELAY)
        continue

    # --------------------------------
    # Save frame periodically
    # --------------------------------

    current_time = time.time()

    if current_time - last_capture_time >= CAPTURE_INTERVAL:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        filename = os.path.join(
            OUTPUT_DIR,
            f"frame_{timestamp}.jpg"
        )

        success = cv2.imwrite(
            filename,
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        if success:
            print(f"[SAVED] {filename}")

        last_capture_time = current_time

    # --------------------------------
    # Display live stream
    # --------------------------------

    cv2.imshow("RTSP Live", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# =========================
# CLEANUP
# =========================

if cap is not None:
    cap.release()

cv2.destroyAllWindows()

print("\nRTSP capture stopped.")