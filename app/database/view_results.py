from __future__ import annotations

from pathlib import Path

from app.database.database_manager import (
    DatabaseManager,
    load_database_config_from_yaml,
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = PROJECT_ROOT / "config.yaml"


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 120)
    print("OCR_BHS - SQL SERVER RESULTS")
    print("=" * 120)
    print()

    manager = None

    try:

        # ----------------------------------------------------
        # LOAD SAME CONFIG USED BY DATABASE MANAGER
        # ----------------------------------------------------

        config = load_database_config_from_yaml(
            CONFIG_FILE
        )

        print(
            f"Host      : {config.host}"
        )

        print(
            f"Database  : {config.database}"
        )

        print(
            f"Username  : {config.username}"
        )

        print(
            f"Driver    : {config.driver}"
        )

        print()

        if not config.host:
            raise RuntimeError(
                "DB_HOST is empty. Check your .env file."
            )

        if not config.database:
            raise RuntimeError(
                "DB_NAME is empty. Check your .env file."
            )

        if not config.username:
            raise RuntimeError(
                "DB_USERNAME is empty. Check your .env file."
            )

        # ----------------------------------------------------
        # CREATE DATABASE MANAGER
        # ----------------------------------------------------

        manager = DatabaseManager(
            config
        )

        # ----------------------------------------------------
        # CONNECTION TEST
        # ----------------------------------------------------

        if not manager.test_connection():

            raise RuntimeError(
                "SQL Server connection failed."
            )

        # ----------------------------------------------------
        # FETCH RESULTS
        # ----------------------------------------------------

        results = manager.fetch_latest(
            limit=50
        )

        print()

        if not results:

            print(
                "[INFO] No OCR/barcode results "
                "have been saved yet."
            )

            print()

            return

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print(
            f"[DB] Found {len(results)} result(s)."
        )

        print()

        for index, row in enumerate(
            results,
            start=1,
        ):

            print("-" * 120)

            print(
                f"Result #{index}"
            )

            print(
                f"ID                  : "
                f"{row.get('id')}"
            )

            print(
                f"Event ID            : "
                f"{row.get('event_id')}"
            )

            print(
                f"Camera              : "
                f"{row.get('camera_id')}"
            )

            print(
                f"Track ID            : "
                f"{row.get('track_id')}"
            )

            print(
                f"Frame ID            : "
                f"{row.get('frame_id')}"
            )

            print(
                f"Captured At         : "
                f"{row.get('captured_at')}"
            )

            print(
                f"OCR Text            : "
                f"{row.get('ocr_text')}"
            )

            print(
                f"OCR Confidence      : "
                f"{row.get('ocr_confidence')}"
            )

            print(
                f"Barcode             : "
                f"{row.get('barcode_value')}"
            )

            print(
                f"Barcode Type        : "
                f"{row.get('barcode_type')}"
            )

            print(
                f"Barcode Confidence  : "
                f"{row.get('barcode_confidence')}"
            )

            print(
                f"Validation Valid    : "
                f"{row.get('validation_valid')}"
            )

            print(
                f"Validation Confidence: "
                f"{row.get('validation_confidence')}"
            )

            print(
                f"Validation Reason   : "
                f"{row.get('validation_reason')}"
            )

            print(
                f"Identifier Match    : "
                f"{row.get('identifier_match')}"
            )

            print(
                f"Status              : "
                f"{row.get('status')}"
            )

            print(
                f"Processing Time     : "
                f"{row.get('processing_ms')} ms"
            )

            print(
                f"Created At          : "
                f"{row.get('created_at')}"
            )

        print()

        print("=" * 120)
        print("RESULT QUERY COMPLETE")
        print("=" * 120)
        print()

    except Exception as exc:

        print()
        print(
            f"[FAIL] {exc}"
        )

        raise

    finally:

        if manager is not None:

            manager.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()