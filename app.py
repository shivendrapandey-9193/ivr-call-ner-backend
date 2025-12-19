import os
import re
import time
import tempfile
import hashlib
import json
import traceback
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# === GROQ SETUP ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY required in .env")

groq_client = Groq(api_key=GROQ_API_KEY)

class Config:
    MAX_TEXT_LENGTH = 5000
    MIN_TEXT_LENGTH = 10
    CACHE_SIZE = 300
    CACHE_TTL = 900
    ALLOWED_AUDIO = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}
    TARGET_SR = 16000
    SILENCE_THRESHOLD = 0.01
    MIN_AUDIO_SECONDS = 1.0

class SentimentLabel(str, Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"
    URGENT_NEGATIVE = "urgent_negative"

class IntentLabel(str, Enum):
    ESCALATION = "ESCALATION"
    FRAUD = "FRAUD"
    REFUND_REQUEST = "REFUND_REQUEST"
    PAYMENT_ISSUE = "PAYMENT_ISSUE"
    ORDER_PROBLEM = "ORDER_PROBLEM"
    SERVICE_COMPLAINT = "SERVICE_COMPLAINT"
    INFORMATION_REQUEST = "INFORMATION_REQUEST"
    UNKNOWN = "UNKNOWN"

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    confidence: float
    source: str

@dataclass
class SentimentResult:
    label: str
    score: float
    emotion_distribution: Dict[str, float]
    reasoning: str

@dataclass
class IntentResult:
    intent: str
    confidence: float
    all_intents: Dict[str, float]

@dataclass
class RiskFlags:
    threat_detected: bool
    profanity_detected: bool
    threat_terms: List[str]
    profanity_terms: List[str]

class SimpleCache:
    def __init__(self, max_size: int, ttl: int):
        self.data = {}
        self.order = []
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key: str):
        item = self.data.get(key)
        if not item: return None
        value, ts = item
        if time.time() - ts > self.ttl:
            self.data.pop(key, None)
            if key in self.order: self.order.remove(key)
            return None
        return value

    def set(self, key: str, value: Any):
        if key in self.data: return
        if len(self.order) >= self.max_size:
            old = self.order.pop(0)
            self.data.pop(old, None)
        self.data[key] = (value, time.time())
        self.order.append(key)

cache = SimpleCache(Config.CACHE_SIZE, Config.CACHE_TTL)

def groq_llm(prompt: str) -> str:
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional IVR transcript translator. "
                        "Translate Hindi/Hinglish to natural English. "
                        "DO NOT change numbers, names, IDs, amounts, dates, or times."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=700
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("⚠️ Groq error:", str(e))
        return ""

def translate_to_english(text: str) -> str:
    if not any("\u0900" <= c <= "\u097F" for c in text):
        return text
    prompt = f"""
Translate this Hinglish/Hindi transcript to natural English. Keep all entity values (names, amounts, IDs, numbers, dates, times, emails, pincodes) exactly the same. Only translate the surrounding words.
Example:
Input: "मेरा नाम राहुल है और कल ₹500 कट गया pincode 110001"
Output: "my name is राहुल and yesterday ₹500 was deducted pincode 110001"
Transcript:
{text}
"""
    translated = groq_llm(prompt)
    return translated or text

def extract_customer_name(original_text: str) -> Optional[Dict]:
    patterns = [
        r"(?:mera|meri)\s+naam\s+([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+)?)",
        r"my\s+name\s+is\s+([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+)?)",
        r"main\s+([A-Za-z\u0900-\u097F]+)\s+bol\s+raha?\s+hoon",
        r"i\s+am\s+(?!talking|calling|speaking)([A-Za-z\u0900-\u097F]{2,})",
        r"naam\s+([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+)?)",
    ]
    lower_text = original_text.lower()
    for pat in patterns:
        match = re.search(pat, lower_text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            start = original_text.lower().find(name.lower())
            if start == -1:
                start = match.start()
            return {
                "text": name.title(),
                "label": "CUSTOMER_NAME",
                "confidence": 0.95,
                "source": "name_pattern",
                "start": start,
                "end": start + len(name)
            }
    return None

def extract_entities_english(english_text: str, original_text: str) -> List[Dict]:
    entities = []
    working_text = english_text.lower()
    original_lower = original_text.lower()

    patterns = [
        ("TXN_ID", re.compile(r"\b(txn|transaction\s*id)[\s:-]*([a-z0-9]{8,})\b", re.IGNORECASE)),
        ("REF_ID", re.compile(r"\b(ref|reference)[\s:-]*([a-z0-9]{8,})\b", re.IGNORECASE)),
        ("ACCOUNT_ID", re.compile(r"\b\d{11,18}\b")),
        ("PHONE_NUMBER", re.compile(r"\b[6-9]\d{9}\b")),
        ("AMOUNT", re.compile(r"(₹|rs\.?|rupees?)\s*([\d,]+(?:\.\d{2})?)", re.IGNORECASE)),
        ("PINCODE", re.compile(r"\b(pin\s*code|pincode)?\s*(\d{6})\b", re.IGNORECASE)),
        ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
        ("TIME", re.compile(r"\b(\d{1,2}:\d{2}\s?(am|pm)?|\d{1,2}\s?(baje|o'?clock)|morning|evening|night|subah|shaam|raat)\b", re.IGNORECASE)),
        ("DATE", re.compile(
            r"\b(yesterday|today|tomorrow|kal|parso|aaj|\d{1,2}(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d+\s+(days?|din|दिन)\s+(ago|before|pehle|पहले))\b",
            re.IGNORECASE
        )),
    ]

    for label, pattern in patterns:
        for match in pattern.finditer(working_text):
            if label in ["TXN_ID", "REF_ID"]:
                value = match.group(2).upper()  # 🔥 FIX: only the ID part, uppercase
            elif label == "AMOUNT":
                amount_match = re.search(r"[\d,]+(?:\.\d{2})?", match.group())
                if not amount_match:
                    continue
                numeric = amount_match.group().replace(",", "")
                value = f"₹{numeric}"
            elif label == "PINCODE":
                value = match.group(2)
            elif label == "EMAIL":
                value = match.group(0)
            else:
                value = match.group(0).strip()

            if len(value) < 3:
                continue

            start = original_lower.find(value.lower())
            if start == -1:
                start = match.start()
            end = start + len(value)

            entities.append({
                "text": value,
                "label": label,
                "confidence": 0.99,
                "source": "regex",
                "start": start,
                "end": end
            })
    return entities

def transcribe_with_groq(path: str) -> Dict[str, Any]:
    try:
        with open(path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                response_format="verbose_json",
                temperature=0.0,
            )

        text = transcription.text.strip()

        if not text:
            raise ValueError("No speech detected")

        language = getattr(transcription, "language", "hi")
        duration = getattr(transcription, "duration", None)
        language_confidence = 0.99

        return {
            "text": text,
            "language": language,
            "language_confidence": language_confidence,
            "duration_seconds": round(duration, 1) if duration else 0.0,
        }

    except Exception as e:
        print("Groq transcription error:", str(e))
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

def analyze_text(original_text: str) -> Dict[str, Any]:
    if len(original_text) < Config.MIN_TEXT_LENGTH:
        raise ValueError("Text too short")
    if len(original_text) > Config.MAX_TEXT_LENGTH:
        raise ValueError("Text too long")

    cache_key = hashlib.md5(original_text.encode()).hexdigest()
    cached = cache.get(cache_key)
    if cached:
        return cached

    name_entity = extract_customer_name(original_text)
    english_text = translate_to_english(original_text)
    entities_en = extract_entities_english(english_text, original_text)
    if name_entity:
        entities_en.append(name_entity)

    sentiment = analyze_sentiment(english_text)
    intent = detect_intent(english_text)
    risk = detect_risk(english_text)
    timeline = extract_timeline(english_text)
    sla_breached = timeline is not None and timeline.get("reported_days_ago", 0) > 3

    entities = [
        Entity(
            text=e["text"],
            label=e["label"],
            start=e["start"],
            end=e["end"],
            confidence=e["confidence"],
            source=e["source"]
        ) for e in entities_en
    ]

    summary = generate_summary(original_text, intent.intent, sentiment.label, entities)
    score = calculate_score(sentiment, intent, len(entities), risk)

    result = {
        "original_text": original_text,
        "language": "hi" if any("\u0900" <= c <= "\u097F" for c in original_text) else "en",
        "entities": [e.__dict__ for e in entities],
        "sentiment": sentiment.__dict__,
        "intent": intent.__dict__,
        "risk": risk.__dict__,
        "timeline": timeline,
        "sla_breached": sla_breached,
        "call_score": score,
        "key_phrases": extract_key_phrases(original_text),
        "summary": summary,
        "visualization": build_visualization(sentiment.label, intent.intent),
        "timestamp": datetime.utcnow().isoformat()
    }

    cache.set(cache_key, result)
    return result

NEGATIVE_WORDS = {"fail", "failed", "refund", "problem", "issue", "deducted", "not received", "wrong", "hold", "debit", "kat gaya", "कट गया", "रिफंड", "payment receive nahi hua", "amount debit dikh raha hai"}
POSITIVE_WORDS = {"thanks", "thank you", "good", "solved", "resolved", "received"}
ANGER_WORDS = {"court", "police", "legal", "complaint"}

def analyze_sentiment(text: str) -> SentimentResult:
    t = text.lower()
    neg = sum(1 for w in NEGATIVE_WORDS if w in t)
    pos = sum(1 for w in POSITIVE_WORDS if w in t)
    anger = sum(1 for w in ANGER_WORDS if w in t)
    if any(x in t for x in ["refund", "रिफंड", "deducted", "कट गया", "payment receive nahi hua"]):
        neg += 4

    if anger >= 1 or any(x in t for x in ["fraud", "hacked", "otp", "scam", "unauthorized"]):
        label = SentimentLabel.URGENT_NEGATIVE.value
        score = -0.95
        reasoning = "Fraud / legal escalation detected"
        frustration = 0.6
        anger_score = 0.3
        happiness = 0.05
        sadness = 0.05
    else:
        total = max(neg + pos + anger, 1)
        frustration = round(neg / total, 2)
        happiness = round(pos / total, 2)
        anger_score = round(anger / total, 2)
        remaining = max(0, 1.0 - frustration - happiness - anger_score)
        sadness = round(0.05 + remaining / 2, 2)
        happiness = round(happiness + remaining / 2, 2)

        if neg >= 3:
            label = SentimentLabel.NEGATIVE.value
            score = -0.7
            reasoning = "Payment/refund issue"
        elif neg >= 1:
            label = SentimentLabel.NEGATIVE.value
            score = -0.4
            reasoning = "Unhappy customer"
        elif pos >= 1:
            label = SentimentLabel.POSITIVE.value
            score = 0.7
            reasoning = "Customer satisfied"
        else:
            label = SentimentLabel.NEUTRAL.value
            score = 0.0
            reasoning = "Neutral tone"

    emotion_dist = {"frustration": frustration, "happiness": happiness, "anger": anger_score, "sadness": sadness}

    return SentimentResult(label, score, emotion_dist, reasoning)

def detect_intent(text: str) -> IntentResult:
    t = text.lower()
    scores = {}
    if any(x in t for x in ["court", "police", "legal", "consumer"]):
        scores[IntentLabel.ESCALATION] = 0.99
    if any(x in t for x in ["fraud", "otp", "cvv"]):
        scores[IntentLabel.FRAUD] = 0.98
    if any(x in t for x in ["refund", "रिफंड", "paise wapas", "पैसे वापस"]):
        scores[IntentLabel.REFUND_REQUEST] = 0.98
    if any(x in t for x in ["payment", "deducted", "कट गया", "debit", "hold", "failed but deducted"]):
        scores[IntentLabel.PAYMENT_ISSUE] = 0.97
    if any(x in t for x in ["order", "delivery", "cancel"]):
        scores[IntentLabel.ORDER_PROBLEM] = 0.88
    if any(x in t for x in ["complaint", "service"]):
        scores[IntentLabel.SERVICE_COMPLAINT] = 0.85
    if any(x in t for x in ["status", "when", "kab"]):
        scores[IntentLabel.INFORMATION_REQUEST] = 0.70

    if not scores:
        return IntentResult(IntentLabel.UNKNOWN.value, 0.0, {})

    primary = max(scores, key=scores.get)
    return IntentResult(primary, scores[primary], scores)

def detect_risk(text: str) -> RiskFlags:
    t = text.lower()
    threats = bool(any(w in t for w in ["court", "police", "legal", "consumer", "fraud"]))
    profanity = bool(any(w in t for w in ["idiot", "stupid", "bewakoof", "chutiya", "madarchod"]))
    return RiskFlags(threat_detected=threats, profanity_detected=profanity, threat_terms=[], profanity_terms=[])

def extract_timeline(text: str) -> Optional[Dict]:
    t = text.lower()
    if any(x in t for x in ["yesterday", "कल"]):
        return {"event": "issue_reported", "reported_days_ago": 1}
    if any(x in t for x in ["two days", "do din", "three days", "teen din"]):
        return {"event": "issue_reported", "reported_days_ago": 2}
    if any(x in t for x in ["today", "aaj"]):
        return {"event": "issue_reported", "reported_days_ago": 0}
    return None

def generate_summary(text: str, intent: str, sentiment: str, entities: List[Entity]) -> Dict[str, str]:
    amount = next((e.text for e in entities if e.label == "AMOUNT"), "")
    txn = next((e.text for e in entities if e.label in ["TXN_ID", "REF_ID"]), "")
    name = next((e.text for e in entities if e.label == "CUSTOMER_NAME"), "")
    date = next((e.text for e in entities if e.label == "DATE"), "")
    time = next((e.text for e in entities if e.label == "TIME"), "")

    intent_text = intent.replace("_", " ").lower()

    # 🔥 Special case for ESCALATION (legal/fraud threats)
    if intent == "ESCALATION":
        return {
            "en": "Customer raised a legal or fraud escalation.",
            "hi": "ग्राहक ने कानूनी या धोखाधड़ी से संबंधित शिकायत की।"
        }

    # Normal case – clean, readable English summary
    en_parts = []
    if name:
        en_parts.append(f"Customer {name}")
    else:
        en_parts.append("Customer")

    if amount:
        en_parts.append(f"reported amount {amount}")

    if txn:
        en_parts.append(f"with reference {txn}")

    if date:
        en_parts.append(f"on {date}")

    if time:
        en_parts.append(f"at {time}")

    en_parts.append(f"regarding a {intent_text}.")

    en = " ".join(en_parts).capitalize()

    # Hindi summary – clean with mappings
    DATE_MAP = {"yesterday": "कल", "today": "आज", "tomorrow": "कल"}
    TIME_MAP = {"morning": "सुबह", "evening": "शाम", "night": "रात"}

    date_hi = DATE_MAP.get(date.lower() if date else "", date or "")
    time_hi = TIME_MAP.get(time.lower() if time else "", time or "")

    hi_parts = []
    if name:
        hi_parts.append(f"ग्राहक {name}")
    else:
        hi_parts.append("ग्राहक")

    if amount:
        hi_parts.append(f"राशि {amount}")

    if txn:
        hi_parts.append(f"{txn} रेफरेंस के साथ")

    if date_hi:
        hi_parts.append(f"{date_hi} को")

    if time_hi:
        hi_parts.append(f"{time_hi} पर")

    hi_parts.append(f"{intent_text} के संबंध में शिकायत की")

    hi = " ".join(hi_parts)
    hi += "।"

    return {"en": en, "hi": hi}

def build_visualization(sentiment: str, intent: str) -> Dict[str, str]:
    if sentiment in ["urgent_negative", "very_negative", "negative"] or intent == "ESCALATION":
        return {"icon": "alert-circle", "theme": "critical_red", "color": "#dc2626"}
    if "negative" in sentiment:
        return {"icon": "frown", "theme": "negative_orange", "color": "#f97316"}
    if "positive" in sentiment:
        return {"icon": "smile", "theme": "positive_green", "color": "#22c55e"}
    return {"icon": "meh", "theme": "neutral_grey", "color": "#94a3b8"}

def build_recommendations(intent: IntentResult, sentiment: SentimentResult, risk: RiskFlags,
                          stt_conf: float, timeline: Optional[Dict], sla_breached: bool) -> Dict[str, Any]:
    alerts = []
    actions = []
    if stt_conf < 0.6:
        alerts.append({"title": "Low Audio Clarity", "severity": "Info", "description": "Manual review suggested"})
    if risk.threat_detected:
        alerts.append({"title": "Legal/Fraud Threat", "severity": "Warning"})
        actions.append("Escalate to Risk Team")
    if sentiment.label in ["urgent_negative", "very_negative", "negative"]:
        alerts.append({"title": "Unhappy Customer", "severity": "Warning"})
        actions.append("Call back within 1 hour")
    if intent.intent == "REFUND_REQUEST":
        actions.append("Verify and process refund")
    if sla_breached:
        alerts.append({"title": "SLA Breach", "severity": "Watch"})
    if not alerts:
        alerts.append({"title": "Routine Call", "severity": "Advisory"})
    return {"alerts": alerts, "actions": actions}

def extract_key_phrases(text: str) -> List[str]:
    sentences = re.split(r'[.!?]\s*', text)
    return [s.strip()[:120] + "..." for s in sentences if len(s) > 20 and any(k in s.lower() for k in ["refund", "kat gaya", "रिफंड", "कट गया"])][:4]

def calculate_score(sentiment: SentimentResult, intent: IntentResult, entities: int, risk: RiskFlags) -> int:
    score = 50
    score += int((sentiment.score + 1) * 25)
    score += int(intent.confidence * 20)
    score += min(entities * 4, 20)
    if risk.threat_detected: score -= 35
    if risk.profanity_detected: score -= 15

    if sentiment.label in ["urgent_negative", "very_negative", "negative"]:
        score = min(score, 65)

    return max(0, min(100, score))

def call_quality_score(stt_conf: float, duration: float) -> int:
    score = int(stt_conf * 75)
    score += 25 if duration > 15 else 10
    return min(100, score)

def is_repetitive_gibberish(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < 8:
        return False

    unique_ratio = len(set(tokens)) / len(tokens)
    if unique_ratio < 0.25:
        return True

    most_common = max(tokens.count(t) for t in set(tokens))
    if most_common / len(tokens) > 0.6:
        return True

    return False

def is_hallucinated_stt(text: str) -> bool:
    tokens = text.strip().split()

    if len(tokens) < 8:
        return True

    unique_ratio = len(set(tokens)) / len(tokens)
    if unique_ratio < 0.4:
        return True

    return is_repetitive_gibberish(text)

app = FastAPI(title="IVR Call Analyzer – v34 (Final Polish – Production Ready)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)

@app.get("/")
def root():
    return {"status": "running", "version": "v34-final", "message": "All entity, summary, and escalation issues fixed – ready for production"}

@app.post("/api/analyze")
def analyze_text_api(req: TextRequest):
    try:
        return analyze_text(req.text)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Error: {str(e)}")

@app.post("/api/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in Config.ALLOWED_AUDIO:
        raise HTTPException(400, f"Unsupported format: {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        path = tmp.name

    try:
        stt = transcribe_with_groq(path)

        gibberish = is_repetitive_gibberish(stt["text"])

        if gibberish:
            return {
                "analysis_valid": False,
                "audio_warning": "Gibberish / non-meaningful speech detected",
                "stt": stt,
                "call_quality_score": call_quality_score(0.99, stt["duration_seconds"]),
                "recommendations": {
                    "alerts": [{"title": "Manual Review Required", "severity": "Warning"}],
                    "actions": []
                }
            }

        analysis = analyze_text(stt["text"])
        analysis["stt"] = stt
        analysis["call_quality_score"] = call_quality_score(
            stt["language_confidence"],
            stt["duration_seconds"]
        )

        hallucinated = is_hallucinated_stt(stt["text"])
        if hallucinated:
            analysis["audio_warning"] = "Possible low-clarity audio – manual verification recommended"

        analysis["analysis_valid"] = True

        intent_obj = IntentResult(**analysis["intent"])
        sentiment_obj = SentimentResult(**analysis["sentiment"])
        risk_obj = RiskFlags(**analysis["risk"])

        analysis["recommendations"] = build_recommendations(
            intent_obj,
            sentiment_obj,
            risk_obj,
            stt["language_confidence"],
            analysis.get("timeline"),
            analysis["sla_breached"]
        )

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Error: {str(e)}")
    finally:
        if os.path.exists(path):
            os.unlink(path)

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)