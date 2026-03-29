WEIGHT_FUNDAMENTAL = 0.45
WEIGHT_TECHNICAL = 0.25
WEIGHT_SENTIMENT = 0.3
THRESHOLD_BUY = 6.5
THRESHOLD_SELL = 4.0


def compute_composite_score(
    fundamental_score: float, technical_score: float, sentiment_score: float
) -> float:
    composite = (
        fundamental_score * WEIGHT_FUNDAMENTAL
        + technical_score * WEIGHT_TECHNICAL
        + sentiment_score * WEIGHT_SENTIMENT
    )
    return round(composite, 2)


def compute_confidence(composite_score: float, verdict: str) -> float:
    if verdict == "BUY":
        distance = composite_score - THRESHOLD_BUY
        confidence = distance / (10.0 - THRESHOLD_BUY) * 100
    elif verdict == "SELL":
        distance = THRESHOLD_SELL - composite_score
        confidence = distance / THRESHOLD_SELL * 100
    else:
        midpoint = (THRESHOLD_BUY + THRESHOLD_SELL) / 2.0
        distance_from_mid = abs(composite_score - midpoint)
        max_hold_distance = (THRESHOLD_BUY - THRESHOLD_SELL) / 2.0
        confidence = (1.0 - distance_from_mid / max_hold_distance) * 100
    return round(min(100.0, max(0.0, confidence)), 1)


def determine_verdict(composite_score: float) -> str:
    if composite_score >= THRESHOLD_BUY:
        return "BUY"
    elif composite_score <= THRESHOLD_SELL:
        return "SELL"
    else:
        return "HOLD"


def run_master_node(
    fundamental_score: float, technical_score: float, sentiment_score: float
) -> dict:
    composite_score = compute_composite_score(
        fundamental_score, technical_score, sentiment_score
    )
    verdict = determine_verdict(composite_score)
    confidence = compute_confidence(composite_score, verdict)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "composite_score": composite_score,
        "weights_used": {
            "fundamental": WEIGHT_FUNDAMENTAL,
            "technical": WEIGHT_TECHNICAL,
            "sentiment": WEIGHT_SENTIMENT,
        },
    }
