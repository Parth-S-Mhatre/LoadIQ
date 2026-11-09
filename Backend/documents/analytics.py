"""Deterministic analytics, evidence-linked insights and recommendations.

The LLM (or the UI) may narrate these results later, but the numbers are
computed here.  Every insight and recommendation references evidence IDs from
the extracted fields.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from .schemas import (
    AnalyticsResult,
    Evidence,
    ExtractedField,
    ExtractionResult,
    Insight,
    Recommendation,
)
from .store import utc_now

CURRENCY_SYMBOLS = {"GBP": "£", "USD": "$", "EUR": "€"}


def _numeric(field: ExtractedField | None) -> float | None:
    if not field or field.value is None:
        return None
    try:
        return float(field.value)
    except (TypeError, ValueError):
        return None


def _text(field: ExtractedField | None) -> str | None:
    if not field or field.value is None:
        return None
    return str(field.value)


def _evidence_ids(*fields: ExtractedField | None) -> list[str]:
    ids: list[str] = []
    for field in fields:
        if not field:
            continue
        for evidence in field.evidence:
            if evidence.id not in ids:
                ids.append(evidence.id)
    return ids


def _currency_symbol(fields: dict[str, ExtractedField]) -> str:
    code = _text(fields.get("currency")) or "GBP"
    return CURRENCY_SYMBOLS.get(code.upper(), code)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _billing_days(fields: dict[str, ExtractedField]) -> int | None:
    start = _parse_date(_text(fields.get("billing_period_start")))
    end = _parse_date(_text(fields.get("billing_period_end")))
    if not start or not end:
        return None
    days = (end - start).days
    return days if days > 0 else None


def _unit_rate_per_kwh(field: ExtractedField | None) -> float | None:
    value = _numeric(field)
    if value is None:
        return None
    unit = (field.unit or "").lower() if field else ""
    if "p/" in unit or unit.startswith("p"):
        return value / 100.0
    return value


def _standing_per_day(field: ExtractedField | None) -> float | None:
    value = _numeric(field)
    if value is None:
        return None
    unit = (field.unit or "").lower() if field else ""
    if "p/" in unit or unit.startswith("p"):
        value = value / 100.0
    if "/month" in unit:
        value = value / 30.0
    return value


def _money(symbol: str, value: float | None) -> str | None:
    if value is None:
        return None
    return f"{symbol}{value:,.2f}"


def _energy_cost(fields: dict[str, ExtractedField]) -> float | None:
    consumption = _numeric(fields.get("consumption_kwh"))
    rate = _unit_rate_per_kwh(fields.get("unit_rate"))
    if consumption is None or rate is None:
        return None
    return consumption * rate


def _standing_cost(fields: dict[str, ExtractedField]) -> float | None:
    per_day = _standing_per_day(fields.get("standing_charge"))
    days = _billing_days(fields)
    if per_day is None or days is None:
        return None
    return per_day * days


def _validate(fields: dict[str, ExtractedField]) -> list[str]:
    warnings: list[str] = []
    consumption = _numeric(fields.get("consumption_kwh"))
    total = _numeric(fields.get("total_cost"))
    unit_rate = _unit_rate_per_kwh(fields.get("unit_rate"))
    standing = _standing_per_day(fields.get("standing_charge"))
    days = _billing_days(fields)
    tax = _numeric(fields.get("tax"))

    if consumption is not None and consumption <= 0:
        warnings.append("Consumption must be greater than zero.")
    if total is not None and total <= 0:
        warnings.append("Total amount due must be greater than zero.")
    if unit_rate is not None and (unit_rate <= 0 or unit_rate > 2.0):
        warnings.append(
            "Unit rate is outside the expected range (0–2 per kWh); check the unit."
        )
    if standing is not None and (standing < 0 or standing > 10):
        warnings.append(
            "Standing charge is outside the expected range (0–10 per day); check the unit."
        )

    start = _parse_date(_text(fields.get("billing_period_start")))
    end = _parse_date(_text(fields.get("billing_period_end")))
    if start and end and start >= end:
        warnings.append("Billing period start must be before the end date.")

    if total is not None and consumption is not None and unit_rate is not None:
        expected = consumption * unit_rate
        if standing is not None and days:
            expected += standing * days
        if tax is not None:
            expected += tax
        if expected > 0:
            difference = abs(total - expected) / expected
            if difference > 0.10:
                warnings.append(
                    "The bill total does not reconcile with consumption × unit rate "
                    f"(expected about {expected:,.2f}, found {total:,.2f}). "
                    "There may be additional charges or an extraction error."
                )
            elif expected > total * 1.01:
                warnings.append(
                    "The extracted energy, standing and tax components add up to more "
                    f"than the total ({expected:,.2f} vs {total:,.2f}). "
                    "One of the values may include a different unit or period."
                )

    return warnings


def _build_history(
    current: ExtractionResult, history: list[ExtractionResult]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = [current, *history]
    for extraction in candidates:
        fields = extraction.fields
        consumption = _numeric(fields.get("consumption_kwh"))
        total = _numeric(fields.get("total_cost"))
        period_end = _text(fields.get("billing_period_end")) or extraction.created_at[:10]
        period_start = _text(fields.get("billing_period_start"))
        if consumption is None and total is None:
            continue
        rows.append(
            {
                "document_id": extraction.document_id,
                "period": period_end,
                "period_start": period_start,
                "period_end": period_end,
                "consumption_kwh": consumption,
                "total_cost": total,
            }
        )
    rows.sort(key=lambda row: row["period"])
    return rows


def build_analytics(
    extraction: ExtractionResult,
    *,
    history: list[ExtractionResult] | None = None,
) -> AnalyticsResult:
    fields = extraction.fields
    symbol = _currency_symbol(fields)

    consumption = _numeric(fields.get("consumption_kwh"))
    total_cost = _numeric(fields.get("total_cost"))
    unit_rate = _unit_rate_per_kwh(fields.get("unit_rate"))
    standing_per_day = _standing_per_day(fields.get("standing_charge"))
    demand = _numeric(fields.get("demand_kw"))
    tax = _numeric(fields.get("tax"))
    days = _billing_days(fields)

    energy_cost = _energy_cost(fields)
    standing_cost = _standing_cost(fields)
    other_cost = None
    if total_cost is not None:
        known = (energy_cost or 0.0) + (standing_cost or 0.0) + (tax or 0.0)
        if known > 0:
            other_cost = max(total_cost - known, 0.0)

    avg_daily_kwh = consumption / days if consumption is not None and days else None
    avg_daily_cost = total_cost / days if total_cost is not None and days else None
    effective_rate = (
        total_cost / consumption
        if total_cost is not None and consumption not in (None, 0)
        else None
    )
    annualized_kwh = (
        consumption * 365 / days if consumption is not None and days else None
    )
    annualized_cost = (
        total_cost * 365 / days if total_cost is not None and days else None
    )
    average_kw = (
        consumption / (days * 24) if consumption is not None and days else None
    )
    load_factor = (
        average_kw / demand
        if average_kw is not None and demand not in (None, 0)
        else None
    )

    charts: dict[str, Any] = {
        "cost_breakdown": [],
        "usage_history": _build_history(extraction, history or []),
    }
    for name, value in (
        ("Energy", energy_cost),
        ("Standing charge", standing_cost),
        ("Tax / levy", tax),
        ("Other charges", other_cost),
    ):
        if value is not None and value > 0:
            charts["cost_breakdown"].append({"name": name, "value": round(value, 2)})

    insights: list[Insight] = []
    warnings = [*extraction.warnings, *_validate(fields)]

    if days:
        insights.append(
            Insight(
                id="billing_period",
                title=f"Billing period covers {days} days",
                detail=(
                    f"The bill runs from {_text(fields.get('billing_period_start'))} "
                    f"to {_text(fields.get('billing_period_end'))}."
                ),
                severity="info",
                evidence_ids=_evidence_ids(
                    fields.get("billing_period"),
                    fields.get("billing_period_start"),
                    fields.get("billing_period_end"),
                ),
            )
        )

    if avg_daily_kwh is not None:
        insights.append(
            Insight(
                id="daily_usage",
                title=f"Average daily consumption: {avg_daily_kwh:,.1f} kWh",
                detail=(
                    "This is total consumption divided by the number of billing days. "
                    "Use it to compare bills with different period lengths."
                ),
                severity="info",
                evidence_ids=_evidence_ids(
                    fields.get("consumption_kwh"), fields.get("billing_period")
                ),
            )
        )

    if effective_rate is not None:
        insights.append(
            Insight(
                id="effective_rate",
                title=f"Effective cost: {symbol}{effective_rate:,.3f} per kWh",
                detail=(
                    "This includes energy, standing charges and taxes, so it is the "
                    "number to compare across tariffs."
                ),
                severity="info",
                evidence_ids=_evidence_ids(
                    fields.get("total_cost"), fields.get("consumption_kwh")
                ),
            )
        )

    if total_cost and tax is not None and tax > 0:
        tax_share = tax / total_cost
        if tax_share >= 0.08:
            insights.append(
                Insight(
                    id="tax_share",
                    title=f"Tax / levy is {tax_share:.1%} of the bill",
                    detail=(
                        "Check whether the applicable VAT or levy rate is correct for "
                        "your premises and tariff."
                    ),
                    severity="warning" if tax_share >= 0.15 else "info",
                    evidence_ids=_evidence_ids(fields.get("tax"), fields.get("total_cost")),
                )
            )

    if total_cost and standing_cost:
        standing_share = standing_cost / total_cost
        if standing_share >= 0.20:
            insights.append(
                Insight(
                    id="standing_share",
                    title=f"Standing charge is {standing_share:.1%} of the bill",
                    detail=(
                        "A high fixed charge matters more when consumption is low. "
                        "Compare fixed fees when switching tariffs."
                    ),
                    severity="warning",
                    evidence_ids=_evidence_ids(
                        fields.get("standing_charge"), fields.get("total_cost")
                    ),
                )
            )

    if load_factor is not None:
        severity = "warning" if load_factor < 0.35 else "info"
        insights.append(
            Insight(
                id="load_factor",
                title=f"Load factor: {load_factor:.0%}",
                detail=(
                    "Load factor is average demand divided by maximum demand. "
                    "Low values mean demand is spiky and may attract capacity charges."
                ),
                severity=severity,
                evidence_ids=_evidence_ids(
                    fields.get("consumption_kwh"),
                    fields.get("demand_kw"),
                    fields.get("billing_period"),
                ),
            )
        )

    history_rows = charts["usage_history"]
    if len(history_rows) >= 2:
        previous = history_rows[-2]
        current_row = history_rows[-1]
        if previous.get("consumption_kwh") and current_row.get("consumption_kwh"):
            change = (
                current_row["consumption_kwh"] - previous["consumption_kwh"]
            ) / previous["consumption_kwh"]
            insights.append(
                Insight(
                    id="usage_change",
                    title=f"Consumption changed {change:+.1%} versus the previous bill",
                    detail=(
                        f"Previous: {previous['consumption_kwh']:,.1f} kWh; "
                        f"current: {current_row['consumption_kwh']:,.1f} kWh."
                    ),
                    severity="warning" if abs(change) >= 0.15 else "info",
                    evidence_ids=_evidence_ids(fields.get("consumption_kwh")),
                )
            )

    if not warnings:
        insights.append(
            Insight(
                id="reconciles",
                title="Bill arithmetic appears consistent",
                detail=(
                    "The extracted consumption, unit rate, standing charge and total "
                    "are within the 10% reconciliation tolerance."
                ),
                severity="positive",
                evidence_ids=_evidence_ids(
                    fields.get("total_cost"),
                    fields.get("consumption_kwh"),
                    fields.get("unit_rate"),
                ),
            )
        )

    recommendations: list[Recommendation] = []
    rate_evidence = _evidence_ids(
        fields.get("unit_rate"),
        fields.get("consumption_kwh"),
        fields.get("billing_period"),
        fields.get("total_cost"),
    )

    if unit_rate is not None and annualized_kwh is not None:
        savings = annualized_kwh * 0.01
        recommendations.append(
            Recommendation(
                id="unit_rate_comparison",
                title="Compare the unit rate with current market tariffs",
                detail=(
                    f"A 1p/kWh reduction on this annualized usage would save about "
                    f"{_money(symbol, savings)} per year. Use the bill's actual "
                    "unit rate and contract end date when comparing offers."
                ),
                estimated_savings=round(savings, 2),
                currency=symbol,
                confidence="medium",
                evidence_ids=rate_evidence,
                assumptions=[
                    "Annualized usage is based on this billing period only.",
                    "Savings exclude changes to standing charges and taxes.",
                ],
            )
        )

    if load_factor is not None and load_factor < 0.35:
        recommendations.append(
            Recommendation(
                id="demand_management",
                title="Investigate peak demand and load shifting",
                detail=(
                    "The load factor is low, so maximum demand is high relative to "
                    "average use. Shifting flexible load away from peak periods can "
                    "reduce capacity charges, but the saving depends on the tariff."
                ),
                confidence="low",
                evidence_ids=_evidence_ids(
                    fields.get("demand_kw"),
                    fields.get("consumption_kwh"),
                    fields.get("billing_period"),
                ),
                assumptions=[
                    "The tariff includes a demand or capacity charge.",
                    "Operational flexibility exists to move load off peak.",
                ],
            )
        )

    if total_cost and standing_cost and standing_cost / total_cost >= 0.20:
        recommendations.append(
            Recommendation(
                id="fixed_charge_review",
                title="Review fixed charges and standing fees",
                detail=(
                    "Fixed charges are a large share of this bill. Include them when "
                    "comparing tariffs, especially if consumption is seasonal."
                ),
                confidence="medium",
                evidence_ids=_evidence_ids(
                    fields.get("standing_charge"), fields.get("total_cost")
                ),
                assumptions=["The bill period is representative of annual usage."],
            )
        )

    if warnings:
        recommendations.append(
            Recommendation(
                id="verify_bill",
                title="Verify the flagged bill details before acting",
                detail=(
                    "One or more extracted values failed validation. Confirm the "
                    "fields against the PDF, then re-run the analysis."
                ),
                confidence="high",
                evidence_ids=_evidence_ids(
                    fields.get("total_cost"),
                    fields.get("consumption_kwh"),
                    fields.get("unit_rate"),
                ),
            )
        )

    evidence_index: dict[str, Evidence] = {}
    for field in fields.values():
        for evidence in field.evidence:
            evidence_index[evidence.id] = evidence

    summary = {
        "billing_days": days,
        "consumption_kwh": consumption,
        "total_cost": total_cost,
        "currency": symbol,
        "unit_rate_per_kwh": round(unit_rate, 4) if unit_rate is not None else None,
        "standing_charge_per_day": (
            round(standing_per_day, 4) if standing_per_day is not None else None
        ),
        "energy_cost": round(energy_cost, 2) if energy_cost is not None else None,
        "standing_cost": round(standing_cost, 2) if standing_cost is not None else None,
        "tax": tax,
        "other_cost": round(other_cost, 2) if other_cost is not None else None,
        "average_daily_kwh": round(avg_daily_kwh, 2) if avg_daily_kwh is not None else None,
        "average_daily_cost": (
            round(avg_daily_cost, 2) if avg_daily_cost is not None else None
        ),
        "effective_cost_per_kwh": (
            round(effective_rate, 4) if effective_rate is not None else None
        ),
        "annualized_kwh": round(annualized_kwh, 1) if annualized_kwh is not None else None,
        "annualized_cost": (
            round(annualized_cost, 2) if annualized_cost is not None else None
        ),
        "max_demand_kw": demand,
        "average_kw": round(average_kw, 2) if average_kw is not None else None,
        "load_factor": round(load_factor, 4) if load_factor is not None else None,
        "field_count": len(fields),
        "needs_review_fields": [
            name for name, field in fields.items() if field.needs_review
        ],
    }

    return AnalyticsResult(
        document_id=extraction.document_id,
        billing_period={
            "start": _text(fields.get("billing_period_start")),
            "end": _text(fields.get("billing_period_end")),
            "raw": _text(fields.get("billing_period")),
        },
        summary=summary,
        charts=charts,
        insights=insights,
        recommendations=recommendations,
        evidence_index=evidence_index,
        warnings=list(dict.fromkeys(warnings)),
        generated_at=utc_now(),
    )
