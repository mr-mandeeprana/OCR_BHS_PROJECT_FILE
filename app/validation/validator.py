from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    """
    Final validation result for one IATA tag.
    """

    valid: bool

    confidence: float

    reason: str

    normalized_ocr: str = ""

    normalized_barcode: str = ""

    identifier_match: bool = False

    def to_dict(self) -> dict:

        return {

            "valid": self.valid,

            "confidence": (
                self.confidence
            ),

            "reason": (
                self.reason
            ),

            "normalized_ocr": (
                self.normalized_ocr
            ),

            "normalized_barcode": (
                self.normalized_barcode
            ),

            "identifier_match": (
                self.identifier_match
            ),
        }


class Validator:
    """
    Validates OCR and barcode output
    for an IATA baggage tag.

    Validation is intentionally conservative.
    """

    AIRPORT_CODES = {

        "DEL",
        "BLR",
        "BOM",
        "MAA",
        "HYD",
        "CCU",
        "PNQ",
        "GOI",
        "COK",
        "AMD",
        "JAI",
        "LKO",

        "DXB",
        "SIN",
        "LHR",
        "CDG",
        "FRA",
        "AMS",
        "DOH",
        "AUH",

        "JFK",
        "ORD",
        "SFO",
        "LAX",

        "BKK",
        "HKG",
        "IST",
        "ICN",
        "SYD",
        "MEL",
    }

    TAG_WORDS = {

        "BAG",
        "BAGS",
        "IATA",
        "FLIGHT",
        "DATE",
        "SEQ",
        "BOARDING",
        "PASS",

        "DELHI",
        "BENGALURU",
        "MUMBAI",
        "CHENNAI",
        "INDIGO",
    }

    def __init__(
        self,
        minimum_ocr_confidence: float = 0.70,
        minimum_barcode_confidence: float = 0.50,
    ) -> None:

        self.minimum_ocr_confidence = float(
            minimum_ocr_confidence
        )

        self.minimum_barcode_confidence = float(
            minimum_barcode_confidence
        )

    # ============================================================
    # NORMALIZE
    # ============================================================

    @staticmethod
    def normalize(
        value: Optional[str],
    ) -> str:

        if not value:

            return ""

        value = str(
            value
        ).upper()

        value = re.sub(
            r"[^A-Z0-9]+",
            " ",
            value,
        )

        return " ".join(
            value.split()
        )

    # ============================================================
    # IDENTIFIER MATCH
    # ============================================================

    def identifiers_match(
        self,
        ocr_text: str,
        barcode: str,
    ) -> bool:

        ocr = self.normalize(
            ocr_text
        )

        code = self.normalize(
            barcode
        )

        if not ocr or not code:

            return False

        compact_ocr = (
            ocr.replace(
                " ",
                "",
            )
        )

        compact_code = (
            code.replace(
                " ",
                "",
            )
        )

        return (
            compact_code
            in compact_ocr
        )

    # ============================================================
    # IATA CONTENT
    # ============================================================

    def has_iata_content(
        self,
        text: str,
    ) -> bool:

        normalized = self.normalize(
            text
        )

        if not normalized:

            return False

        tokens = set(
            normalized.split()
        )

        airport_matches = (
            tokens
            & self.AIRPORT_CODES
        )

        if airport_matches:

            return True

        tag_matches = (
            tokens
            & self.TAG_WORDS
        )

        if tag_matches:

            return True

        identifiers = re.findall(
            r"\b[A-Z0-9]{8,}\b",
            normalized,
        )

        return bool(
            identifiers
        )

    # ============================================================
    # VALIDATE
    # ============================================================

    def validate(
        self,
        ocr_text: str = "",
        ocr_confidence: float = 0.0,
        barcode_data: str = "",
        barcode_confidence: float = 0.0,
    ) -> ValidationResult:

        ocr = self.normalize(
            ocr_text
        )

        barcode = self.normalize(
            barcode_data
        )

        try:

            ocr_confidence = float(
                ocr_confidence
            )

        except (
            TypeError,
            ValueError,
        ):

            ocr_confidence = 0.0

        try:

            barcode_confidence = float(
                barcode_confidence
            )

        except (
            TypeError,
            ValueError,
        ):

            barcode_confidence = 0.0

        match = (
            self.identifiers_match(
                ocr,
                barcode,
            )
        )

        iata_content = (
            self.has_iata_content(
                ocr
            )
        )

        # ========================================================
        # OCR + BARCODE AGREEMENT
        # ========================================================

        if (
            ocr
            and barcode
            and match
            and iata_content
        ):

            confidence = min(
                0.99,
                (
                    ocr_confidence
                    + barcode_confidence
                ) / 2.0,
            )

            return ValidationResult(

                valid=True,

                confidence=confidence,

                reason=(
                    "OCR_BARCODE_AGREEMENT"
                ),

                normalized_ocr=ocr,

                normalized_barcode=barcode,

                identifier_match=True,
            )

        # ========================================================
        # STRONG BARCODE
        # ========================================================

        if (
            barcode
            and barcode_confidence
            >= self.minimum_barcode_confidence
        ):

            return ValidationResult(

                valid=True,

                confidence=min(
                    0.99,
                    barcode_confidence,
                ),

                reason=(
                    "BARCODE_VALID"
                ),

                normalized_ocr=ocr,

                normalized_barcode=barcode,

                identifier_match=match,
            )

        # ========================================================
        # STRONG OCR
        # ========================================================

        if (
            ocr
            and ocr_confidence
            >= self.minimum_ocr_confidence
            and iata_content
        ):

            return ValidationResult(

                valid=True,

                confidence=min(
                    0.99,
                    ocr_confidence,
                ),

                reason=(
                    "OCR_VALID"
                ),

                normalized_ocr=ocr,

                normalized_barcode=barcode,

                identifier_match=match,
            )

        # ========================================================
        # OCR / BARCODE DISAGREEMENT
        # ========================================================

        if (
            ocr
            and barcode
            and not match
        ):

            return ValidationResult(

                valid=False,

                confidence=0.0,

                reason=(
                    "OCR_BARCODE_DISAGREEMENT"
                ),

                normalized_ocr=ocr,

                normalized_barcode=barcode,

                identifier_match=False,
            )

        # ========================================================
        # NOT VALIDATED
        # ========================================================

        return ValidationResult(

            valid=False,

            confidence=0.0,

            reason=(
                "IATA_CONTENT_NOT_VALIDATED"
            ),

            normalized_ocr=ocr,

            normalized_barcode=barcode,

            identifier_match=match,
        )


# ======================================================================
# TEST
# ======================================================================

if __name__ == "__main__":

    print("=" * 72)

    print(
        "OCR_BHS VALIDATOR TEST"
    )

    print("=" * 72)

    validator = Validator()

    result = validator.validate(

        ocr_text=(
            "BAG DEL BLR FLIGHT "
            "5756567890AER"
        ),

        ocr_confidence=0.92,

        barcode_data=(
            "5756567890AER"
        ),

        barcode_confidence=1.0,
    )

    print(
        "Valid      :",
        result.valid,
    )

    print(
        "Confidence :",
        result.confidence,
    )

    print(
        "Reason     :",
        result.reason,
    )

    print(
        "OCR        :",
        result.normalized_ocr,
    )

    print(
        "Barcode    :",
        result.normalized_barcode,
    )

    assert result.valid

    assert result.identifier_match

    print()

    print(
        "[PASS] Validator"
    )

    print("=" * 72)

    print(
        "VALIDATOR TEST PASSED"
    )

    print("=" * 72)