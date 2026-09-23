from __future__ import annotations

import json
import os
import queue
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pyodbc
import yaml
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

@dataclass
class DatabaseConfig:
    enabled: bool = False
    db_type: str = "sqlserver"

    host: str = ""
    port: int = 1433
    database: str = ""

    username: str = ""
    password: str = ""

    driver: str = "ODBC Driver 18 for SQL Server"

    connection_timeout: int = 10

    encrypt: str = "yes"
    trust_server_certificate: str = "yes"

    insert_results: bool = True


# ============================================================
# DATABASE MANAGER
# ============================================================

class DatabaseManager:
    """
    SQL Server database manager for OCR_BHS.

    Responsibilities:
        - Connect to SQL Server
        - Create database if required
        - Initialize tables
        - Test connection
        - Insert OCR readings
        - Insert barcode readings
        - Insert combined results
        - Fetch latest results
        - Close connection safely
    """

    def __init__(
        self,
        config: Optional[DatabaseConfig] = None,
    ):
        self.config = config or DatabaseConfig()

        self.connection: Optional[pyodbc.Connection] = None

        self.initialized = False

        self.lock = threading.RLock()

    # ========================================================
    # CONNECTION STRING
    # ========================================================

    def _server(self) -> str:
        host = self.config.host.strip()

        if not host:
            raise ValueError(
                "Database host is empty."
            )

        # Don't append port to named SQL Server instances.
        if "\\" in host:
            return host

        if self.config.port:
            return f"{host},{self.config.port}"

        return host

    # --------------------------------------------------------

    def _connection_string(
        self,
        database: str,
    ) -> str:

        server = self._server()

        driver = self.config.driver

        username = self.config.username

        password = self.config.password

        encrypt = self.config.encrypt

        trust = self.config.trust_server_certificate

        return (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            f"Encrypt={encrypt};"
            f"TrustServerCertificate={trust};"
            f"Connection Timeout={self.config.connection_timeout};"
        )

    # ========================================================
    # CONNECT
    # ========================================================

    def connect(
        self,
        database: Optional[str] = None,
    ) -> pyodbc.Connection:

        with self.lock:

            if self.connection is not None:

                try:
                    cursor = self.connection.cursor()

                    cursor.execute("SELECT 1")

                    cursor.fetchone()

                    cursor.close()

                    return self.connection

                except Exception:
                    try:
                        self.connection.close()
                    except Exception:
                        pass

                    self.connection = None

            db_name = (
                database
                if database is not None
                else self.config.database
            )

            if not db_name:
                raise ValueError(
                    "Database name is empty."
                )

            connection_string = self._connection_string(
                db_name
            )

            self.connection = pyodbc.connect(
                connection_string,
                timeout=self.config.connection_timeout,
                autocommit=False,
            )

            return self.connection

    # ========================================================
    # CREATE DATABASE
    # ========================================================

    def create_database_if_missing(self) -> None:
        """
        Create the configured database if it does not exist.

        Uses master database.
        """

        database_name = self.config.database.strip()

        if not database_name:
            raise ValueError(
                "Database name is empty."
            )

        print(
            "[DB] Creating database if missing..."
        )

        connection = None
        cursor = None

        try:

            connection = pyodbc.connect(
                self._connection_string("master"),
                timeout=self.config.connection_timeout,
                autocommit=True,
            )

            cursor = connection.cursor()

            # Escape closing bracket in database name.
            safe_name = database_name.replace(
                "]",
                "]]",
            )

            cursor.execute(
                f"""
                IF DB_ID(N'{safe_name}') IS NULL
                BEGIN
                    CREATE DATABASE [{safe_name}]
                END
                """
            )

            print(
                f"[DB] Database ready: {database_name}"
            )

        finally:

            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass

            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    # ========================================================
    # INITIALIZE DATABASE
    # ========================================================

    def initialize(self) -> None:
        """
        Create all OCR_BHS tables and indexes.

        Schema statements are split and executed one-by-one because
        pyodbc cannot reliably execute a multi-statement DDL batch
        (containing IF/BEGIN/END blocks) as a single execute() call
        against SQL Server.
        """

        print(
            "[DB] Connecting to database..."
        )

        connection = self.connect(
            self.config.database
        )

        schema_path = (
            Path(__file__).resolve().parent
            / "schema.sql"
        )

        if not schema_path.exists():
            raise FileNotFoundError(
                f"Schema file not found: {schema_path}"
            )

        schema_sql = schema_path.read_text(
            encoding="utf-8"
        )

        # --------------------------------------------------------------------
        # Split on semicolons that terminate top-level SQL statements.
        # Each IF … BEGIN … END; block is a single self-contained statement.
        # We use a simple regex that splits on ';' followed by optional
        # whitespace / comments, handling Windows (\r\n) and Unix (\n) lines.
        # Blank / comment-only chunks are skipped.
        # --------------------------------------------------------------------

        batches = [
            chunk.strip()
            for chunk in re.split(r";\s*", schema_sql)
        ]

        cursor = connection.cursor()

        errors = []

        try:

            for batch in batches:

                # Skip empty chunks and pure comment blocks.
                stripped = re.sub(
                    r"--[^\n]*",
                    "",
                    batch,
                ).strip()

                if not stripped:
                    continue

                try:

                    cursor.execute(batch)

                    connection.commit()

                except Exception as exc:

                    connection.rollback()

                    errors.append(
                        f"{type(exc).__name__}: {exc} "
                        f"[batch: {batch[:80]!r}]"
                    )

                    # Surface every failing statement loudly instead of
                    # silently continuing — a failed CREATE TABLE here
                    # (e.g. from an unquoted reserved keyword) used to be
                    # masked as a "non-fatal" error, leaving the table
                    # missing while later inserts failed with a much
                    # more confusing error.
                    print(
                        "[DB] Schema statement FAILED: "
                        f"{exc}"
                    )
                    print(
                        f"[DB]   -> statement: {batch[:200]!r}"
                    )

        finally:

            cursor.close()

        if errors:

            # Raise only if NO statements succeeded at all.
            # Partial failures (e.g. table already exists) are acceptable
            # when the database was previously initialised.
            print(
                f"[DB] Schema init completed with "
                f"{len(errors)} non-fatal error(s)."
            )

        self.initialized = True

        print(
            "[DB] Database schema initialized successfully."
        )

    # ========================================================
    # TEST CONNECTION
    # ========================================================

    def test_connection(self) -> bool:
        """
        Test SQL Server connectivity.

        Returns:
            True if connection works.
        """

        connection = None
        cursor = None

        try:

            connection = self.connect(
                self.config.database
            )

            cursor = connection.cursor()

            cursor.execute(
                "SELECT @@SERVERNAME, DB_NAME(), @@VERSION"
            )

            row = cursor.fetchone()

            if row:

                print(
                    f"[DB] SQL Server connection: OK"
                )

                print(
                    f"[DB] Server: {row[0]}"
                )

                print(
                    f"[DB] Database: {row[1]}"
                )

            return True

        except Exception as exc:

            print(
                f"[DB] SQL Server connection: FAILED"
            )

            print(
                f"[DB] Error: {exc}"
            )

            return False

        finally:

            if cursor is not None:

                try:
                    cursor.close()
                except Exception:
                    pass

    # ========================================================
    # INITIALIZE FULL DATABASE
    # ========================================================

    def setup(self) -> None:
        """
        Complete database setup.
        """

        self.create_database_if_missing()

        self.connect(
            self.config.database
        )

        self.initialize()

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _timestamp(
        value: Any = None,
    ) -> datetime:

        if isinstance(value, datetime):
            # Already a datetime — strip tzinfo for pyodbc compat.
            return value.replace(tzinfo=None)

        # Accept Unix float/int timestamps (what result_persistence.py
        # passes as captured_at / last_timestamp).
        if isinstance(value, (int, float)):
            try:
                return datetime.utcfromtimestamp(float(value))
            except (OSError, OverflowError, ValueError):
                pass

        return datetime.utcnow()

    # --------------------------------------------------------

    @staticmethod
    def _json(
        value: Any,
    ) -> Optional[str]:

        if value is None:
            return None

        try:

            return json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )

        except Exception:

            return str(value)

    # --------------------------------------------------------

    @staticmethod
    def _get(
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:

        if obj is None:
            return default

        if isinstance(obj, dict):

            return obj.get(
                key,
                default,
            )

        return getattr(
            obj,
            key,
            default,
        )

    # ========================================================
    # INSERT OCR READING
    # ========================================================

    def insert_ocr_reading(
        self,
        data: Optional[dict] = None,
        **kwargs,
    ) -> bool:

        payload = {}

        if data:
            payload.update(data)

        payload.update(kwargs)

        event_id = (
            payload.get("event_id")
            or str(uuid.uuid4())
        )

        camera_id = (
            payload.get("camera_id")
            or "unknown"
        )

        # NOTE: "text" is a reserved / deprecated type keyword in
        # T-SQL (the legacy TEXT data type), so the column name is
        # wrapped in [text] here and in schema.sql. Leaving it
        # unquoted causes a syntax error on both CREATE TABLE and
        # INSERT, which was silently swallowed as "non-fatal" by
        # initialize() and surfaced later as confusing insert
        # failures / a missing table.
        query = """
        INSERT INTO dbo.ocr_readings
        (
            event_id,
            camera_id,
            track_id,
            task_id,
            frame_id,
            captured_at,
            [text],
            normalized_text,
            confidence,
            engine,
            rotation,
            success,
            elapsed_ms,
            error,
            details_json
        )
        VALUES
        (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?
        )
        """

        params = (
            str(event_id),
            str(camera_id),
            payload.get("track_id"),
            payload.get("task_id"),
            payload.get("frame_id"),
            self._timestamp(
                payload.get("captured_at")
            ),
            payload.get("text"),
            payload.get("normalized_text"),
            payload.get("confidence"),
            payload.get("engine"),
            payload.get("rotation"),
            1 if payload.get(
                "success",
                False
            ) else 0,
            payload.get("elapsed_ms"),
            payload.get("error"),
            self._json(
                payload.get("details")
            ),
        )

        with self.lock:

            connection = self.connect()

            cursor = connection.cursor()

            try:

                cursor.execute(
                    query,
                    params,
                )

                connection.commit()

                return True

            except Exception:

                connection.rollback()

                raise

            finally:

                cursor.close()

    # ========================================================
    # INSERT BARCODE READING
    # ========================================================

    def insert_barcode_reading(
        self,
        data: Optional[dict] = None,
        **kwargs,
    ) -> bool:

        payload = {}

        if data:
            payload.update(data)

        payload.update(kwargs)

        event_id = (
            payload.get("event_id")
            or str(uuid.uuid4())
        )

        camera_id = (
            payload.get("camera_id")
            or "unknown"
        )

        query = """
        INSERT INTO dbo.barcode_readings
        (
            event_id,
            camera_id,
            track_id,
            task_id,
            frame_id,
            captured_at,
            barcode_value,
            barcode_type,
            confidence,
            rotation,
            variant,
            success,
            elapsed_ms,
            error,
            details_json
        )
        VALUES
        (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?
        )
        """

        params = (
            str(event_id),
            str(camera_id),
            payload.get("track_id"),
            payload.get("task_id"),
            payload.get("frame_id"),
            self._timestamp(
                payload.get("captured_at")
            ),
            payload.get("barcode_value"),
            payload.get("barcode_type"),
            payload.get("confidence"),
            payload.get("rotation"),
            payload.get("variant"),
            1 if payload.get(
                "success",
                False
            ) else 0,
            payload.get("elapsed_ms"),
            payload.get("error"),
            self._json(
                payload.get("details")
            ),
        )

        with self.lock:

            connection = self.connect()

            cursor = connection.cursor()

            try:

                cursor.execute(
                    query,
                    params,
                )

                connection.commit()

                return True

            except Exception:

                connection.rollback()

                raise

            finally:

                cursor.close()

    # ========================================================
    # INSERT FINAL RESULT
    # ========================================================

    def insert_final_result(
        self,
        data: Optional[dict] = None,
        **kwargs,
    ) -> bool:

        payload = {}

        if data:
            payload.update(data)

        payload.update(kwargs)

        event_id = (
            payload.get("event_id")
            or str(uuid.uuid4())
        )

        camera_id = (
            payload.get("camera_id")
            or "unknown"
        )

        query = """
        INSERT INTO dbo.ocr_barcode_results
        (
            event_id,
            camera_id,
            track_id,
            task_id,
            frame_id,
            captured_at,

            ocr_text,
            ocr_normalized_text,
            ocr_confidence,
            ocr_engine,
            ocr_success,

            barcode_value,
            barcode_type,
            barcode_confidence,
            barcode_success,

            validation_valid,
            validation_confidence,
            validation_reason,
            identifier_match,

            status,
            processing_ms,
            raw_json
        )
        VALUES
        (
            ?, ?, ?, ?, ?, ?,

            ?, ?, ?, ?, ?,

            ?, ?, ?, ?,

            ?, ?, ?, ?,

            ?, ?, ?
        )
        """

        params = (
            str(event_id),
            str(camera_id),
            payload.get("track_id"),
            payload.get("task_id"),
            payload.get("frame_id"),
            self._timestamp(
                payload.get("captured_at")
            ),

            payload.get("ocr_text"),
            payload.get("ocr_normalized_text"),
            payload.get("ocr_confidence"),
            payload.get("ocr_engine"),
            1 if payload.get(
                "ocr_success",
                False
            ) else 0,

            payload.get("barcode_value"),
            payload.get("barcode_type"),
            payload.get("barcode_confidence"),
            1 if payload.get(
                "barcode_success",
                False
            ) else 0,

            1 if payload.get(
                "validation_valid",
                False
            ) else 0,

            payload.get(
                "validation_confidence"
            ),

            payload.get(
                "validation_reason"
            ),

            1 if payload.get(
                "identifier_match",
                False
            ) else 0,

            payload.get(
                "status"
            ),

            payload.get(
                "processing_ms"
            ),

            self._json(payload),
        )

        with self.lock:

            connection = self.connect()

            cursor = connection.cursor()

            try:

                cursor.execute(
                    query,
                    params,
                )

                connection.commit()

                return True

            except Exception:

                connection.rollback()

                raise

            finally:

                cursor.close()

    # ========================================================
    # FETCH LATEST RESULTS
    # ========================================================

    def fetch_latest(
        self,
        limit: int = 50,
    ) -> list[dict]:

        limit = max(
            1,
            min(
                int(limit),
                1000,
            ),
        )

        query = f"""
        SELECT TOP {limit}
            id,
            event_id,
            camera_id,
            track_id,
            frame_id,
            captured_at,
            ocr_text,
            ocr_confidence,
            barcode_value,
            barcode_type,
            barcode_confidence,
            validation_valid,
            validation_confidence,
            validation_reason,
            identifier_match,
            status,
            processing_ms,
            created_at
        FROM dbo.ocr_barcode_results
        ORDER BY created_at DESC
        """

        with self.lock:

            connection = self.connect()

            cursor = connection.cursor()

            try:

                cursor.execute(query)

                columns = [
                    column[0]
                    for column in cursor.description
                ]

                rows = cursor.fetchall()

                return [
                    dict(
                        zip(
                            columns,
                            row,
                        )
                    )
                    for row in rows
                ]

            finally:

                cursor.close()

    # ========================================================
    # FETCH LATEST OCR READINGS (debug helper)
    # ========================================================

    def fetch_latest_ocr_readings(
        self,
        limit: int = 50,
    ) -> list[dict]:
        """
        Convenience helper to inspect dbo.ocr_readings directly —
        useful for confirming the [text] fix took effect end-to-end.
        """

        limit = max(
            1,
            min(
                int(limit),
                1000,
            ),
        )

        query = f"""
        SELECT TOP {limit}
            id,
            event_id,
            camera_id,
            track_id,
            frame_id,
            captured_at,
            [text],
            normalized_text,
            confidence,
            engine,
            rotation,
            success,
            elapsed_ms,
            error,
            created_at
        FROM dbo.ocr_readings
        ORDER BY created_at DESC
        """

        with self.lock:

            connection = self.connect()

            cursor = connection.cursor()

            try:

                cursor.execute(query)

                columns = [
                    column[0]
                    for column in cursor.description
                ]

                rows = cursor.fetchall()

                return [
                    dict(
                        zip(
                            columns,
                            row,
                        )
                    )
                    for row in rows
                ]

            finally:

                cursor.close()

    # ========================================================
    # REPAIR HELPER: recreate ocr_readings from scratch
    # ========================================================

    def repair_ocr_readings_table(self) -> None:
        """
        Drops and recreates dbo.ocr_readings only.

        Use this ONCE if the table was previously created (or attempted)
        before the [text] reserved-keyword fix, since the original
        CREATE TABLE statement may have failed silently and left the
        table missing or malformed. Safe to call repeatedly — it is a
        no-op once the table matches schema.sql.
        """

        print(
            "[DB] Repairing dbo.ocr_readings..."
        )

        connection = self.connect(
            self.config.database
        )

        cursor = connection.cursor()

        try:

            cursor.execute(
                "IF OBJECT_ID(N'dbo.ocr_readings', N'U') IS NOT NULL "
                "DROP TABLE dbo.ocr_readings;"
            )

            connection.commit()

            cursor.execute(
                """
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
                    [text] NVARCHAR(MAX) NULL,
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
                """
            )

            connection.commit()

            cursor.execute(
                """
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
                """
            )

            connection.commit()

            print(
                "[DB] dbo.ocr_readings repaired successfully."
            )

        except Exception:

            connection.rollback()

            raise

        finally:

            cursor.close()

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:

        with self.lock:

            if self.connection is not None:

                try:
                    self.connection.close()
                except Exception:
                    pass

                self.connection = None

                self.initialized = False

                print(
                    "[DB] Connection closed."
                )


# ============================================================
# ASYNCHRONOUS DATABASE WORKER
# ============================================================

class DatabaseWorker:
    """
    Non-blocking database writer.

    The main OCR/YOLO pipeline submits records here.
    Database inserts happen in a background thread.
    """

    def __init__(
        self,
        manager: DatabaseManager,
        max_queue_size: int = 500,
    ):

        self.manager = manager

        self.queue: queue.Queue = queue.Queue(
            maxsize=max_queue_size
        )

        self.thread: Optional[
            threading.Thread
        ] = None

        self.stop_event = threading.Event()

        self.running = False

        self.inserted = 0

        self.failed = 0

    # ========================================================

    def start(self) -> None:

        if self.running:
            return

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._run,
            name="DatabaseWorker",
            daemon=True,
        )

        self.running = True

        self.thread.start()

        print(
            "[DBWorker] Started."
        )

    # ========================================================

    def submit(
        self,
        record_type: str,
        data: dict,
        block: bool = False,
    ) -> bool:

        item = (
            record_type,
            data,
        )

        try:

            self.queue.put(
                item,
                block=block,
            )

            return True

        except queue.Full:

            print(
                "[DBWorker] Queue full - "
                "record dropped."
            )

            return False

    # ========================================================

    def _run(self) -> None:

        # Keep consuming until shutdown was requested AND the queue
        # has been fully drained. This prevents accepted records from
        # being lost during application shutdown.
        while (
            not self.stop_event.is_set()
            or not self.queue.empty()
        ):

            try:

                record_type, data = (
                    self.queue.get(
                        timeout=0.25
                    )
                )

            except queue.Empty:

                continue

            try:

                if record_type == "ocr":

                    self.manager.insert_ocr_reading(
                        data
                    )

                elif record_type == "barcode":

                    self.manager.insert_barcode_reading(
                        data
                    )

                elif record_type in (
                    "final",
                    "result",
                    "ocr_barcode",
                ):

                    self.manager.insert_final_result(
                        data
                    )

                else:

                    raise ValueError(
                        f"Unknown database record type: "
                        f"{record_type}"
                    )

                self.inserted += 1

            except Exception as exc:

                self.failed += 1

                print(
                    f"[DBWorker] Insert failed: {exc}"
                )

            finally:

                self.queue.task_done()

        self.running = False

    # ========================================================

    def stop(
        self,
        wait: bool = True,
    ) -> None:

        self.stop_event.set()

        if wait:
            # Give the worker an opportunity to finish every accepted
            # database record before the thread is joined.
            try:
                self.queue.join()
            except Exception:
                pass

        if (
            wait
            and self.thread is not None
            and self.thread.is_alive()
        ):

            self.thread.join(
                timeout=15
            )

        self.running = False

        print(
            "[DBWorker] Stopped."
        )

    # ========================================================

    def status(self) -> dict:

        return {
            "running": self.running,
            "queue_size": self.queue.qsize(),
            "inserted": self.inserted,
            "failed": self.failed,
        }


# ============================================================
# CONFIG LOADER
# ============================================================

def load_database_config_from_yaml(
    config_path: str | Path,
) -> DatabaseConfig:

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        yaml_data = yaml.safe_load(file) or {}

    database = (
        yaml_data.get(
            "database",
            {},
        )
        or {}
    )

    host_env = database.get(
        "host_env",
        "DB_HOST",
    )

    database_env = database.get(
        "database_env",
        "DB_NAME",
    )

    username_env = database.get(
        "username_env",
        "DB_USERNAME",
    )

    password_env = database.get(
        "password_env",
        "DB_PASSWORD",
    )

    return DatabaseConfig(

        enabled=bool(
            database.get(
                "enabled",
                False,
            )
        ),

        db_type=str(
            database.get(
                "type",
                "sqlserver",
            )
        ),

        host=os.getenv(
            host_env,
            "",
        ),

        port=int(
            database.get(
                "port",
                1433,
            )
        ),

        database=os.getenv(
            database_env,
            "",
        ),

        username=os.getenv(
            username_env,
            "",
        ),

        password=os.getenv(
            password_env,
            "",
        ),

        driver=str(
            database.get(
                "driver",
                "ODBC Driver 18 for SQL Server",
            )
        ),

        connection_timeout=int(
            database.get(
                "connection_timeout",
                10,
            )
        ),

        encrypt=str(
            database.get(
                "encrypt",
                "yes",
            )
        ),

        trust_server_certificate=str(
            database.get(
                "trust_server_certificate",
                "yes",
            )
        ),

        insert_results=bool(
            database.get(
                "insert_results",
                True,
            )
        ),
    )


# ============================================================
# COMMAND LINE DATABASE TEST
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("OCR_BHS SQL SERVER DATABASE TEST")
    print("=" * 70)
    print()

    config_path = (
        PROJECT_ROOT / "config.yaml"
    )

    manager = None

    try:

        config = (
            load_database_config_from_yaml(
                config_path
            )
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

        manager = DatabaseManager(
            config
        )

        # ----------------------------------------------------
        # STEP 1: CONNECTION
        # ----------------------------------------------------

        if not manager.test_connection():

            raise RuntimeError(
                "SQL Server connection test failed."
            )

        # ----------------------------------------------------
        # STEP 2: CREATE DATABASE
        # ----------------------------------------------------

        manager.create_database_if_missing()

        # ----------------------------------------------------
        # STEP 3: CONNECT
        # ----------------------------------------------------

        manager.connect(
            config.database
        )

        # ----------------------------------------------------
        # STEP 4: REPAIR ocr_readings (one-time fix for the
        # earlier reserved-keyword [text] bug), THEN create the
        # rest of the schema normally.
        # ----------------------------------------------------

        manager.repair_ocr_readings_table()

        manager.initialize()

        # ----------------------------------------------------
        # STEP 5: FINAL TEST
        # ----------------------------------------------------

        if not manager.test_connection():

            raise RuntimeError(
                "Database verification failed."
            )

        print()
        print("=" * 70)
        print("DATABASE TEST PASSED")
        print("=" * 70)
        print()

    except Exception as exc:

        print()
        print(
            f"[FAIL] {exc}"
        )

        raise

    finally:

        if manager is not None:

            try:
                manager.close()
            except Exception:
                pass


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()