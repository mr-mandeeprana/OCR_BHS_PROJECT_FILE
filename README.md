# OCR_BHS

Production-oriented multi-camera OCR and barcode recognition pipeline for IATA tags used in baggage handling systems.

## Overview

OCR_BHS combines camera/video input, YOLO-based IATA tag detection, ByteTrack tracking, asynchronous OCR and barcode processing, validation, and SQL Server persistence into a single pipeline.

### Main pipeline

```text
Camera / RTSP / Video
        │
        ▼
      YOLO
 IATA Tag Detection
        │
        ▼
    ByteTrack
      Tracking
        │
        ▼
 Best Frame / Tag Crop
        │
   ┌────┴────┐
   ▼         ▼
  OCR      Barcode
 Worker     Worker
   │         │
   └────┬────┘
        ▼
 Result Manager
        │
        ▼
   Validation
        │
        ▼
 SQL Server
```

The live path is designed so OCR and barcode processing run asynchronously and do not unnecessarily block camera capture and detection.

---

## Features

- Multi-camera architecture
- RTSP camera support
- Recorded video testing
- YOLO IATA-tag detection
- ByteTrack object tracking
- Continuous camera capture
- Asynchronous OCR processing
- Asynchronous barcode processing
- Best-frame selection for recognition
- Image quality checking
- OCR validation
- Barcode validation
- SQL Server persistence
- Central YAML configuration
- Environment-variable based secrets
- Structured logging
- Worker queues
- Production-oriented modular architecture

---

## Project Structure

```text
OCR_BHS_PROJECT/
│
├── config.yaml
├── requirements.txt
├── .env
├── .env.example
├── README.md
├── main.py
│
├── app/
│   ├── config/
│   │   └── config_loader.py
│   │
│   ├── camera/
│   │   └── camera.py
│   │
│   ├── detection/
│   │   └── yolo_detector.py
│   │
│   ├── tracking/
│   │   └── tag_tracker.py
│   │
│   ├── preprocessing/
│   │   ├── tag_processor.py
│   │   └── image_quality.py
│   │
│   ├── ocr/
│   │   └── ocr_engine.py
│   │
│   ├── barcode/
│   │   ├── barcode_engine.py
│   │   └── barcode_region_processor.py
│   │
│   ├── validation/
│   │   └── validator.py
│   │
│   ├── pipeline/
│   │   ├── pipeline.py
│   │   ├── processing_queue.py
│   │   ├── worker_manager.py
│   │   ├── barcode_worker.py
│   │   ├── result_manager.py
│   │   └── best_frame_selector.py
│   │
│   └── database/
│       ├── database_manager.py
│       └── result_persistence.py
│
├── models/
│   └── yolo/
│       └── iata_tag_yolo26n_best.pt
│
├── videos/
│   ├── input/
│   └── output/
│
├── captures/
├── failed/
├── logs/
├── output/
├── debug/
├── dataset/
└── tests/
```

---

## Requirements

Recommended environment:

- Windows
- Python 3.11.x
- OpenCV
- Ultralytics
- PaddleOCR
- PaddlePaddle
- PyZbar
- SQL Server
- FFmpeg for camera/video testing

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

If a virtual environment does not exist:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration

All major runtime settings should be controlled through:

```text
config.yaml
```

Sensitive database credentials must be stored in:

```text
.env
```

Do not commit `.env` to Git.

Example:

```text
DB_HOST=YOUR_SQL_SERVER
DB_NAME=OCR_BHS_PROJECT
DB_USERNAME=YOUR_USERNAME
DB_PASSWORD=YOUR_PASSWORD
```

Use `.env.example` as the template for other machines.

---

## YOLO Model

The application uses a trained YOLO model for IATA tag detection.

Default model:

```text
models/yolo/iata_tag_yolo26n_best.pt
```

The detector is configured from `config.yaml`.

Example:

```yaml
detection:
  enabled: true
  model_path: "models/yolo/iata_tag_yolo26n_best.pt"
  confidence: 0.03
  image_size: 640
  device: "cpu"
  classes:
    - 0
  class_names:
    0: "iata_tag"
  half: false
  max_det: 20
```

The class used by this project is:

```text
0 = IATA_TAG
```

---

## Tracking

ByteTrack is used to maintain persistent identities for detected IATA tags.

Example configuration:

```yaml
tracking:
  enabled: true
  tracker: "bytetrack"
  track_high_thresh: 0.10
  track_low_thresh: 0.05
  new_track_thresh: 0.10
  track_buffer: 45
  match_thresh: 0.85
  fuse_score: true
```

Tracking is performed after YOLO detection.

The same tracking system is intended to be shared by the pipeline rather than creating a second independent tracking pipeline.

---

## Camera Sources

The project supports recorded video and RTSP sources.

### Recorded video

Example:

```yaml
camera_1:
  enabled: true
  source_type: "video"
  video_source: "videos/input/camera_1_test.mp4"
  name: "Camera 1"
  reconnect: true
  reconnect_delay: 2
  buffer_size: 1
  target_fps: 0
  video_loop: true
```

### RTSP

For RTSP, keep the camera URL in environment/configuration rather than hard-coding credentials into source files.

The application should resolve the configured camera source before starting the pipeline.

---

## Running the Application

From the project root:

```powershell
python main.py
```

The application will:

1. Load configuration.
2. Initialize SQL Server.
3. Load the YOLO model.
4. Create camera objects.
5. Create ByteTrack trackers.
6. Start OCR and barcode workers.
7. Start camera capture.
8. Run YOLO detection.
9. Track IATA tags.
10. Submit recognition jobs asynchronously.
11. Validate recognition results.
12. Persist results to SQL Server.

---

## Testing With Recorded Video

Place the test video in:

```text
videos/input/
```

Then configure:

```yaml
source_type: "video"
video_source: "videos/input/camera_1_test.mp4"
```

Run:

```powershell
python main.py
```

For looping playback:

```yaml
video_loop: true
```

For a single playback:

```yaml
video_loop: false
```

---

## Testing the Camera Separately

Camera connectivity should be tested independently before debugging YOLO, OCR, or barcode processing.

Typical checks include:

- Source opens successfully
- Frame is received
- Resolution is correct
- FPS is available
- RTSP reconnect works
- Video does not continuously return empty frames

For an RTSP source, verify that the camera provides usable frames before debugging downstream processing.

---

## OCR

OCR is performed asynchronously using the OCR worker system.

The intended processing flow is:

```text
YOLO detection
      ↓
ByteTrack track
      ↓
Best frame selection
      ↓
IATA tag crop
      ↓
Image preprocessing / quality checks
      ↓
OCR
      ↓
Validation
      ↓
Result Manager
```

OCR should operate on the selected IATA-tag crop instead of unnecessarily processing the complete camera frame.

---

## Barcode

Barcode processing follows a similar asynchronous path:

```text
YOLO detection
      ↓
ByteTrack track
      ↓
Best frame
      ↓
Tag / barcode region
      ↓
Barcode engine
      ↓
Validation
      ↓
Result Manager
```

Barcode processing should not block the live camera/detection path.

---

## Worker Architecture

The worker manager provides background processing for OCR and barcode jobs.

```text
Main Pipeline
     │
     ├── OCR Queue ──────► OCR Worker 1
     │                    OCR Worker 2
     │
     └── Barcode Queue ──► Barcode Worker 1
                          Barcode Worker 2
```

Results are returned through the result-management system and can then be persisted.

---

## Database

The project uses SQL Server for result persistence.

The database configuration is provided through environment variables.

Example database:

```text
OCR_BHS_PROJECT
```

The application initializes the required database/schema through the database management layer.

Database processing is separated from the live vision pipeline so database operations do not unnecessarily block camera processing.

---

## Logging

Runtime logs should be stored under:

```text
logs/
```

Useful information includes:

- Camera connection
- Resolution and FPS
- YOLO detections
- Detection confidence
- ByteTrack tracking
- OCR jobs
- Barcode jobs
- Worker status
- Database status
- Pipeline statistics
- Runtime errors

Example detection output:

```text
[YOLO] frame=593 detections=4
```

Example tracking output:

```text
[TagTracker] frame 630: 11/11 accepted
```

---

## Troubleshooting

### 1. Camera does not open

Check:

```text
source_type
video_source
RTSP configuration
camera availability
```

For a recorded video, verify the file exists.

---

### 2. YOLO gives zero detections

Check:

```text
model_path
confidence
image_size
device
class configuration
input frame resolution
```

Run the detector against a known-good image before debugging the complete pipeline.

---

### 3. YOLO detects but tracking is empty

Check:

```text
YOLO detections
Detection object format
ByteTrack input format
tracking thresholds
track buffer
```

Do not immediately replace ByteTrack. First verify that the exact detections returned by YOLO are being passed to the tracker.

---

### 4. Tracking works but OCR does not run

Check:

```text
best-frame selection
OCR queue submission
WorkerManager.submit_ocr()
OCR worker startup
OCR task structure
```

The pipeline statistics should show OCR jobs being submitted.

---

### 5. Tracking works but barcode does not run

Check:

```text
Barcode queue submission
WorkerManager.submit_barcode()
BarcodeTask structure
Barcode worker startup
Barcode result handling
```

The pipeline and worker method signatures must match exactly.

---

### 6. Database does not save results

Check:

```text
DB_HOST
DB_NAME
DB_USERNAME
DB_PASSWORD
SQL Server service
database permissions
database schema
```

Database errors should be checked separately from the vision pipeline.

---

## Pipeline Statistics

At shutdown, the application reports statistics such as:

```text
frames_processed
detection_runs
tracks_seen
active_tracks
latest_frame_id
latest_detection_count
ocr_jobs_submitted
ocr_jobs_failed
barcode_jobs_submitted
barcode_jobs_failed
ocr_results_received
barcode_results_received
results_received
detection_errors
result_errors
```

These values are useful for locating failures between:

```text
Camera
  ↓
YOLO
  ↓
ByteTrack
  ↓
OCR / Barcode
  ↓
Results
  ↓
Database
```

---

## Performance

The application is designed around the following principles:

- Camera capture should remain continuous.
- Detection can run at a configured interval.
- OCR and barcode recognition should run asynchronously.
- Only useful/best frames should be submitted for expensive recognition.
- Database writes should not block camera capture.
- A small camera buffer helps reduce live latency.
- CPU and GPU can be selected through configuration.

For CPU operation:

```yaml
device: "cpu"
half: false
```

For a supported CUDA environment, the device can be changed through configuration.

---

## Development Guidelines

When modifying the project:

1. Keep configuration in `config.yaml`.
2. Keep credentials in `.env`.
3. Do not hard-code camera credentials.
4. Do not create a second YOLO pipeline.
5. Do not create a second ByteTrack pipeline.
6. Keep OCR asynchronous.
7. Keep barcode processing asynchronous.
8. Keep database operations separated from camera capture.
9. Test individual components before testing the full pipeline.
10. Preserve the existing data contract between pipeline components.

---

## Recommended Test Order

When debugging the application, use this order:

```text
1. Python environment
       ↓
2. Configuration
       ↓
3. Camera / video
       ↓
4. YOLO model
       ↓
5. YOLO detections
       ↓
6. ByteTrack
       ↓
7. Best-frame selection
       ↓
8. OCR
       ↓
9. Barcode
       ↓
10. Result Manager
       ↓
11. Database
       ↓
12. Complete application
```

This prevents downstream components from being blamed when an upstream component is the actual source of the problem.

---

## Git

Initialize the repository:

```powershell
git init
```

Add all project files:

```powershell
git add .
```

Commit:

```powershell
git commit -m "Initial OCR BHS project"
```

Set the main branch:

```powershell
git branch -M main
```

Add the remote:

```powershell
git remote add origin https://github.com/mr-mandeeprana/OCR_BHS_PROJECT_FILE.git
```

Push:

```powershell
git push -u origin main
```

Before pushing, verify that `.env` is ignored:

```powershell
git status
```

Never commit database passwords, API keys, camera credentials, or other secrets.

---

## Current System Status

The project has been tested component-by-component for:

- Camera/video capture
- YOLO IATA tag detection
- ByteTrack tracking
- OCR
- Barcode recognition
- Database initialization

The complete application should be validated end-to-end after the pipeline worker submission/result-handling interfaces are aligned.

---

## Author

**Mandeep Rana**

OCR_BHS — Baggage Handling System OCR / Barcode Processing Project
