"""PDF text extraction and evidence-first field extraction.

Digital PDFs are read with pypdf. If the text layer is missing or too small,
an optional OCR backend (NVIDIA nemotron-ocr-v2) recovers line-level text plus
normalized bounding boxes and confidence. Every field keeps the page, line and
source text that produced it, and OCR confidence is stored alongside the regex
match confidence.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .ocr import (
    OCRError,
    field_recovery_enabled,
    min_text_chars,
    ocr_enabled,
    ocr_mode,
    ocr_pdf,
)
from .schemas import Evidence, ExtractedField, ExtractionResult
from .store import utc_now

FIELD_LABELS = {
    "supplier": "Supplier",
    "account_number": "Account number",
    "meter_number": "Meter number",
    "billing_period": "Billing period",
    "billing_period_start": "Billing period start",
    "billing_period_end": "Billing period end",
    "consumption_kwh": "Consumption",
    "demand_kw": "Maximum demand",
    "unit_rate": "Unit rate",
    "standing_charge": "Standing charge",
    "tax": "Tax / levy",
    "total_cost": "Total amount due",
    "currency": "Currency",
    "tariff": "Tariff",
}

CURRENCY_SYMBOLS = {"£": "GBP", "$": "USD", "€": "EUR"}


def _clean_number(raw: str) -> float | None:
    value = raw.strip().replace(",", "").replace(" ", "")
    value = re.sub(r"[^\d.\-]", "", value)
    try:
        return float(value)
    except ValueError:
        return None


def _parse_date(raw: str) -> str | None:
    value = raw.strip().replace(".", "/").replace("-", "/")
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _find_match(
    pages: list[list[str]], pattern: re.Pattern[str]
) -> tuple[re.Match[str], int, int, str] | None:
    for page_number, lines in enumerate(pages, start=1):
        for line_number, line in enumerate(lines, start=1):
            match = pattern.search(line)
            if match:
                return match, page_number, line_number, line
    return None


def _make_field(
    name: str,
    match: tuple[re.Match[str], int, int, str],
    *,
    value: float | str | None,
    unit: str | None,
    raw_value: str | None = None,
    confidence: float = 0.92,
    note: str | None = None,
) -> ExtractedField:
    _, page_number, line_number, source_line = match
    evidence = Evidence(
        id=f"ev_{name}_{page_number}_{line_number}",
        page=page_number,
        line=line_number,
        source_text=source_line.strip(),
        method="regex",
        confidence=confidence,
    )
    return ExtractedField(
        name=name,
        label=FIELD_LABELS.get(name, name.replace("_", " ").title()),
        value=value,
        unit=unit,
        confidence=confidence,
        needs_review=confidence < 0.8,
        raw_value=raw_value,
        evidence=[evidence],
        notes=[note] if note else [],
    )


def _extract_field(
    pages: list[list[str]],
    name: str,
    pattern: str,
    *,
    group: int = 1,
    unit: str | None = None,
    numeric: bool = True,
    value_from_group: bool = True,
    confidence: float = 0.92,
) -> ExtractedField | None:
    match = _find_match(pages, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
    if not match:
        return None

    raw = match[0][group] if value_from_group else match[0][0]
    value: float | str | None
    if numeric:
        value = _clean_number(raw)
        if value is None:
            return None
    else:
        value = raw.strip().rstrip(" .,;:")

    return _make_field(
        name,
        match,
        value=value,
        unit=unit,
        raw_value=raw.strip(),
        confidence=confidence,
    )


def _extract_dates(pages: list[list[str]]) -> dict[str, ExtractedField]:
    pattern = re.compile(
        r"(?:billing\s+period|period|statement\s+period)\s*[:\-]?\s*"
        r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})"
        r"\s*(?:to|until|through|–|-)\s*"
        r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})",
        re.IGNORECASE,
    )
    match = _find_match(pages, pattern)
    if not match:
        return {}

    found_match, page_number, line_number, source_line = match
    start_raw, end_raw = found_match.group(1), found_match.group(2)
    start = _parse_date(start_raw)
    end = _parse_date(end_raw)
    if not start or not end:
        return {}

    evidence = Evidence(
        id=f"ev_billing_period_{page_number}_{line_number}",
        page=page_number,
        line=line_number,
        source_text=source_line.strip(),
        method="regex",
        confidence=0.9,
    )
    period = ExtractedField(
        name="billing_period",
        label=FIELD_LABELS["billing_period"],
        value=f"{start} to {end}",
        unit=None,
        confidence=0.9,
        needs_review=False,
        raw_value=f"{start_raw} to {end_raw}",
        evidence=[evidence],
    )
    start_field = period.model_copy(
        update={
            "name": "billing_period_start",
            "label": FIELD_LABELS["billing_period_start"],
            "value": start,
        }
    )
    end_field = period.model_copy(
        update={
            "name": "billing_period_end",
            "label": FIELD_LABELS["billing_period_end"],
            "value": end,
        }
    )
    return {
        "billing_period": period,
        "billing_period_start": start_field,
        "billing_period_end": end_field,
    }


def _extract_currency(pages: list[list[str]]) -> ExtractedField | None:
    for page_number, lines in enumerate(pages, start=1):
        for line_number, line in enumerate(lines, start=1):
            for symbol, code in CURRENCY_SYMBOLS.items():
                if symbol in line:
                    evidence = Evidence(
                        id=f"ev_currency_{page_number}_{line_number}",
                        page=page_number,
                        line=line_number,
                        source_text=line.strip(),
                        method="symbol",
                        confidence=0.85,
                    )
                    return ExtractedField(
                        name="currency",
                        label=FIELD_LABELS["currency"],
                        value=code,
                        unit=None,
                        confidence=0.85,
                        needs_review=False,
                        raw_value=symbol,
                        evidence=[evidence],
                    )
    return None


def _extract_supplier(pages: list[list[str]]) -> ExtractedField | None:
    labelled = _extract_field(
        pages,
        "supplier",
        r"supplier\s*[:\-]\s*(.+)",
        numeric=False,
        confidence=0.9,
    )
    if labelled:
        return labelled

    # Conservative fallback: the first short line that looks like a company
    # name.  Low confidence so it is always flagged for review.
    if not pages or not pages[0]:
        return None
    for line_number, line in enumerate(pages[0][:5], start=1):
        candidate = line.strip()
        if (
            2 <= len(candidate) <= 60
            and not re.search(r"\d", candidate)
            and not re.search(
                r"statement|invoice|bill|electricity|account|period|page",
                candidate,
                re.IGNORECASE,
            )
        ):
            match = (re.match(r"(.+)", candidate), 1, line_number, line)  # type: ignore[arg-type]
            return _make_field(
                "supplier",
                match,  # type: ignore[arg-type]
                value=candidate,
                unit=None,
                raw_value=candidate,
                confidence=0.45,
                note="Inferred from the first line; please confirm.",
            )
    return None


def _extract_fields(pages: list[list[str]]) -> dict[str, ExtractedField]:
    """Apply the bill regexes to a set of page lines."""
    fields: dict[str, ExtractedField] = {}

    for name, value in _extract_dates(pages).items():
        fields[name] = value

    patterns = [
        (
            "account_number",
            r"account(?:\s*(?:number|no\.?|#))?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{4,})",
            None,
            False,
        ),
        (
            "meter_number",
            r"(?:meter|mpan|mprn)(?:\s*(?:number|no\.?|#))?\s*[:\-]?\s*"
            r"(?:mpan|mprn)?\s*([A-Z0-9]*\d[A-Z0-9\-/]{3,})",
            None,
            False,
        ),
        (
            "consumption_kwh",
            r"(?:total\s+)?(?:consumption|usage|energy\s+used|electricity\s+used)\s*"
            r"[:\-]?\s*([\d,]+(?:\.\d+)?)\s*kwh",
            "kWh",
            True,
        ),
        (
            "demand_kw",
            r"(?:maximum\s+demand|max\s+demand|demand)\s*[:\-]?\s*"
            r"([\d,]+(?:\.\d+)?)\s*kw",
            "kW",
            True,
        ),
        (
            "total_cost",
            r"(?:total\s+(?:amount|charges?|due)|amount\s+due|balance\s+due)\s*"
            r"[:\-]?\s*[£$€]?\s*([\d,]+(?:\.\d{1,4})?)",
            "currency",
            True,
        ),
        (
            "tax",
            r"(?:vat|tax|lev(?:y|ies))\s*[:\-]?\s*[£$€]?\s*([\d,]+(?:\.\d{1,4})?)",
            "currency",
            True,
        ),
        (
            "tariff",
            r"(?:tariff|plan|product)\s*[:\-]\s*(.+)",
            None,
            False,
        ),
    ]

    for name, pattern, unit, numeric in patterns:
        extracted = _extract_field(
            pages,
            name,
            pattern,
            unit=unit,
            numeric=numeric,
            confidence=0.92,
        )
        if extracted:
            fields[name] = extracted

    # Unit rate: pence-per-kWh or currency-per-kWh.
    rate_match = _find_match(
        pages,
        re.compile(
            r"([\d,]+(?:\.\d+)?)\s*(p|pence|cents?|c)\s*(?:/|per)\s*kwh",
            re.IGNORECASE,
        ),
    )
    if rate_match:
        value = _clean_number(rate_match[0].group(1))
        if value is not None:
            fields["unit_rate"] = _make_field(
                "unit_rate",
                rate_match,
                value=value,
                unit="p/kWh",
                raw_value=rate_match[0].group(0).strip(),
                confidence=0.92,
            )
    else:
        rate_match = _find_match(
            pages,
            re.compile(
                r"([£$€]\s*[\d,]+(?:\.\d+)?)\s*(?:/|per)\s*kwh",
                re.IGNORECASE,
            ),
        )
        if rate_match:
            value = _clean_number(rate_match[0].group(1))
            if value is not None:
                fields["unit_rate"] = _make_field(
                    "unit_rate",
                    rate_match,
                    value=value,
                    unit="currency/kWh",
                    raw_value=rate_match[0].group(0).strip(),
                    confidence=0.92,
                )

    # Standing charge: pence/day or currency/day.
    standing_match = _find_match(
        pages,
        re.compile(
            r"standing\s+charge\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*"
            r"(p|pence|cents?|c|£|\$|€)?\s*(?:/|per)?\s*(day|month)?",
            re.IGNORECASE,
        ),
    )
    if standing_match:
        value = _clean_number(standing_match[0].group(1))
        if value is not None:
            unit_token = (standing_match[0].group(2) or "").lower()
            period = (standing_match[0].group(3) or "day").lower()
            if unit_token in {"p", "pence", "cent", "cents", "c"}:
                unit = f"p/{period}"
            else:
                unit = f"currency/{period}"
            fields["standing_charge"] = _make_field(
                "standing_charge",
                standing_match,
                value=value,
                unit=unit,
                raw_value=standing_match[0].group(0).strip(),
                confidence=0.9,
            )

    currency = _extract_currency(pages)
    if currency:
        fields["currency"] = currency
        currency_code = str(currency.value)
        for name in ("total_cost", "tax"):
            if name in fields and fields[name].unit == "currency":
                fields[name].unit = currency_code
        for name in ("unit_rate", "standing_charge"):
            field = fields.get(name)
            if field and field.unit:
                field.unit = field.unit.replace("currency", currency_code)

    supplier = _extract_supplier(pages)
    if supplier:
        fields["supplier"] = supplier

    return fields


def _apply_ocr_evidence(
    fields: dict[str, ExtractedField],
    ocr_pages: dict[int, list],
) -> None:
    """Attach OCR bbox/confidence and provenance to evidence built from OCR text."""
    for field in fields.values():
        for index, evidence in enumerate(field.evidence):
            page_lines = ocr_pages.get(evidence.page)
            if not page_lines or evidence.line is None:
                continue
            if not 1 <= evidence.line <= len(page_lines):
                continue
            line = page_lines[evidence.line - 1]
            field.evidence[index] = evidence.model_copy(
                update={
                    "bbox": list(line.bbox),
                    "ocr_confidence": line.confidence,
                    "method": f"{evidence.method}+ocr",
                    "confidence": round(
                        min(evidence.confidence, max(0.5, line.confidence)), 3
                    ),
                }
            )


def _run_ocr(
    pdf_path: Path,
    reason: str,
    warnings: list[str],
) -> tuple[list[list[str]] | None, dict[int, list]]:
    """Run the configured OCR provider and return page text plus line boxes."""
    try:
        result = ocr_pdf(pdf_path)
    except OCRError as exc:
        warnings.append(f"OCR was requested ({reason}) but did not run: {exc}")
        return None, {}
    if not any(result.pages):
        warnings.append(f"OCR ran ({reason}) but returned no text.")
        return None, {}
    pages = [[line.text for line in page] for page in result.pages]
    warnings.extend(result.warnings)
    return pages, {index: page for index, page in enumerate(result.pages, start=1)}


def extract_pdf(document_id: str, pdf_path: Path) -> ExtractionResult:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "pypdf is required for PDF extraction. Add pypdf to requirements.txt."
        ) from exc

    reader = PdfReader(str(pdf_path))
    digital_pages: list[list[str]] = []
    text_chars = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        text_chars += len(text)
        digital_pages.append(text.splitlines())

    warnings: list[str] = []
    ocr_used = False
    ocr_provider: str | None = None
    ocr_lines: dict[int, list] = {}
    pages = digital_pages

    enabled = ocr_enabled()
    mode = ocr_mode()
    low_text = text_chars < min_text_chars()

    if enabled and mode == "always":
        ocr_text, ocr_lines = _run_ocr(pdf_path, "OCR_MODE=always", warnings)
        if ocr_text is not None:
            pages, ocr_used, ocr_provider = ocr_text, True, "nemotron-ocr-v2"
            text_chars = sum(len(line) for page in ocr_text for line in page)
    elif enabled and mode == "auto" and (low_text or text_chars < 40):
        ocr_text, ocr_lines = _run_ocr(pdf_path, "low text layer", warnings)
        if ocr_text is not None:
            pages, ocr_used, ocr_provider = ocr_text, True, "nemotron-ocr-v2"
            text_chars = sum(len(line) for page in ocr_text for line in page)
    elif low_text and mode != "never":
        warnings.append(
            "This PDF has little or no text layer and OCR is not enabled. "
            "Set OCR_PROVIDER=nemotron and an NVIDIA API key to read scanned bills."
        )

    fields = _extract_fields(pages)

    # Optional second pass: a scan with a sparse text layer but missing key fields.
    if enabled and field_recovery_enabled() and not ocr_used:
        required = ("consumption_kwh", "total_cost", "billing_period", "unit_rate")
        missing = [name for name in required if name not in fields]
        if missing:
            reason = "missing fields: " + ", ".join(missing)
            ocr_text, recovered_lines = _run_ocr(pdf_path, reason, warnings)
            if ocr_text is not None:
                pages = ocr_text
                ocr_lines = recovered_lines
                ocr_used = True
                ocr_provider = "nemotron-ocr-v2"
                text_chars = sum(len(line) for page in ocr_text for line in page)
                fields.update(_extract_fields(pages))

    if ocr_used:
        _apply_ocr_evidence(fields, ocr_lines)

    for required, message in (
        ("consumption_kwh", "Consumption (kWh) was not found."),
        ("total_cost", "Total amount due was not found."),
        ("billing_period", "Billing period was not found."),
        ("unit_rate", "Unit rate was not found."),
    ):
        if required not in fields:
            warnings.append(message)

    status = "needs_review"
    if not fields:
        warnings.append(
            "No bill fields could be extracted. The document may be scanned, "
            "password-protected or use an unsupported layout."
        )

    now = utc_now()
    page_count = max(len(reader.pages), len(ocr_lines) if ocr_used else 0)
    return ExtractionResult(
        document_id=document_id,
        status=status,
        fields=fields,
        warnings=warnings,
        page_count=page_count,
        text_chars=text_chars,
        extractor="pypdf+nemotron-ocr+regex" if ocr_used else "pypdf+regex",
        ocr_used=ocr_used,
        ocr_provider=ocr_provider,
        created_at=now,
        updated_at=now,
    )
