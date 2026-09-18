-- ============================================================
-- OCR_BHS PROJECT - SQL SERVER SCHEMA
-- ============================================================

-- ------------------------------------------------------------
-- OCR READINGS
-- ------------------------------------------------------------

IF OBJECT_ID(N'dbo.ocr_readings', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ocr_readings
    (
        id BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_ocr_readings PRIMARY KEY,

        event_id NVARCHAR(100) NULL,

        camera_id NVARCHAR(100) NOT NULL,

        track_id INT NULL,

        task_id NVARCHAR(100) NULL,

        frame_id BIGINT NULL,

        captured_at DATETIME2(3) NULL,

        text NVARCHAR(MAX) NULL,

        normalized_text NVARCHAR(MAX) NULL,

        confidence FLOAT NULL,

        engine NVARCHAR(100) NULL,

        rotation INT NULL,

        success BIT NOT NULL
            CONSTRAINT DF_ocr_readings_success DEFAULT (0),

        elapsed_ms FLOAT NULL,

        error NVARCHAR(MAX) NULL,

        details_json NVARCHAR(MAX) NULL,

        created_at DATETIME2(3) NOT NULL
            CONSTRAINT DF_ocr_readings_created_at
            DEFAULT (SYSUTCDATETIME())
    );
END;


-- ------------------------------------------------------------
-- BARCODE READINGS
-- ------------------------------------------------------------

IF OBJECT_ID(N'dbo.barcode_readings', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.barcode_readings
    (
        id BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_barcode_readings PRIMARY KEY,

        event_id NVARCHAR(100) NULL,

        camera_id NVARCHAR(100) NOT NULL,

        track_id INT NULL,

        task_id NVARCHAR(100) NULL,

        frame_id BIGINT NULL,

        captured_at DATETIME2(3) NULL,

        barcode_value NVARCHAR(500) NULL,

        barcode_type NVARCHAR(100) NULL,

        confidence FLOAT NULL,

        rotation INT NULL,

        variant NVARCHAR(100) NULL,

        success BIT NOT NULL
            CONSTRAINT DF_barcode_readings_success DEFAULT (0),

        elapsed_ms FLOAT NULL,

        error NVARCHAR(MAX) NULL,

        details_json NVARCHAR(MAX) NULL,

        created_at DATETIME2(3) NOT NULL
            CONSTRAINT DF_barcode_readings_created_at
            DEFAULT (SYSUTCDATETIME())
    );
END;


-- ------------------------------------------------------------
-- FINAL COMBINED OCR + BARCODE RESULTS
-- ------------------------------------------------------------

IF OBJECT_ID(N'dbo.ocr_barcode_results', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ocr_barcode_results
    (
        id BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_ocr_barcode_results PRIMARY KEY,

        event_id NVARCHAR(100) NOT NULL,

        camera_id NVARCHAR(100) NOT NULL,

        track_id INT NULL,

        task_id NVARCHAR(100) NULL,

        frame_id BIGINT NULL,

        captured_at DATETIME2(3) NULL,

        ocr_text NVARCHAR(MAX) NULL,

        ocr_normalized_text NVARCHAR(MAX) NULL,

        ocr_confidence FLOAT NULL,

        ocr_engine NVARCHAR(100) NULL,

        ocr_success BIT NOT NULL
            CONSTRAINT DF_results_ocr_success DEFAULT (0),

        barcode_value NVARCHAR(500) NULL,

        barcode_type NVARCHAR(100) NULL,

        barcode_confidence FLOAT NULL,

        barcode_success BIT NOT NULL
            CONSTRAINT DF_results_barcode_success DEFAULT (0),

        validation_valid BIT NOT NULL
            CONSTRAINT DF_results_validation_valid DEFAULT (0),

        validation_confidence FLOAT NULL,

        validation_reason NVARCHAR(MAX) NULL,

        identifier_match BIT NOT NULL
            CONSTRAINT DF_results_identifier_match DEFAULT (0),

        status NVARCHAR(50) NULL,

        processing_ms FLOAT NULL,

        raw_json NVARCHAR(MAX) NULL,

        created_at DATETIME2(3) NOT NULL
            CONSTRAINT DF_results_created_at
            DEFAULT (SYSUTCDATETIME())
    );
END;


-- ------------------------------------------------------------
-- INDEXES
-- ------------------------------------------------------------

IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_ocr_readings_camera_track'
      AND object_id = OBJECT_ID(N'dbo.ocr_readings')
)
BEGIN
    CREATE INDEX IX_ocr_readings_camera_track
        ON dbo.ocr_readings(camera_id, track_id);
END;


IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_barcode_readings_camera_track'
      AND object_id = OBJECT_ID(N'dbo.barcode_readings')
)
BEGIN
    CREATE INDEX IX_barcode_readings_camera_track
        ON dbo.barcode_readings(camera_id, track_id);
END;


IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_results_camera_track'
      AND object_id = OBJECT_ID(N'dbo.ocr_barcode_results')
)
BEGIN
    CREATE INDEX IX_results_camera_track
        ON dbo.ocr_barcode_results(camera_id, track_id);
END;


IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_results_event_id'
      AND object_id = OBJECT_ID(N'dbo.ocr_barcode_results')
)
BEGIN
    CREATE INDEX IX_results_event_id
        ON dbo.ocr_barcode_results(event_id);
END;


IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_results_created_at'
      AND object_id = OBJECT_ID(N'dbo.ocr_barcode_results')
)
BEGIN
    CREATE INDEX IX_results_created_at
        ON dbo.ocr_barcode_results(created_at);
END;