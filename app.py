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
import numpy as np
from pydub import AudioSegment
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from faster_whisper import WhisperModel
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
    WHISPER_MODEL = "small"  # ~500MB download, ~2-3GB RAM peak with int8 – safest for low-memory systems
    WHISPER_DEVICE = "cuda" if os.getenv("USE_GPU", "0") == "1" else "cpu"
    WHISPER_COMPUTE = "int8"  # Maximum memory efficiency
    WHISPER_BEAM_SIZE = 5
    WHISPER_VAD_FILTER = True
    TARGET_SR = 16000
    SILENCE_THRESHOLD = 0.01
    MIN_AUDIO_SECONDS = 1.0

whisper_model = None
def get_whisper():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model (small + int8)... This may take a few minutes on first run.")
        whisper_model = WhisperModel(
            Config.WHISPER_MODEL,
            device=Config.WHISPER_DEVICE,
            compute_type=Config.WHISPER_COMPUTE
        )
        print("Whisper model loaded successfully!")
    return whisper_model

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
            model="llama-3.1-8b-instant",  # 🔥 fastest & perfect for translation
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
                value = f"{match.group(1).upper()}{match.group(2).upper()}"
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
    total = max(neg + pos + anger, 1)
    frustration = round(neg / total, 2)
    happiness = round(pos / total, 2)
    anger_score = round(anger / total, 2)
    remaining = max(0, 1.0 - frustration - happiness - anger_score)
    sadness = round(0.05 + remaining / 2, 2)
    happiness = round(happiness + remaining / 2, 2)
    emotion_dist = {"frustration": frustration, "happiness": happiness, "anger": anger_score, "sadness": sadness}

    if anger >= 2:
        label = SentimentLabel.URGENT_NEGATIVE.value
        score = -0.95
        reasoning = "Threat/legal language"
    elif neg >= 3:
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

    en_parts = []
    if name: en_parts.append(f"Customer {name}")
    if amount: en_parts.append(f"amount {amount}")
    if txn: en_parts.append(f"{txn}")
    if date: en_parts.append(f"on {date}")
    if time: en_parts.append(f"at {time}")
    en_parts.append(f"raised {intent.replace('_', ' ').lower()} issue")
    en = " ".join(en_parts).capitalize() + "." if en_parts else "Customer raised a complaint."

    hi_parts = []
    if name: hi_parts.append(f"ग्राहक {name}")
    if amount: hi_parts.append(f"राशि {amount}")
    if txn: hi_parts.append(f"{txn}")
    if date: hi_parts.append(f"{date} को")
    if time: hi_parts.append(f"{time} पर")
    hi_parts.append(f"ने {intent.replace('_', ' ').lower()} की शिकायत की")
    hi = " ".join(hi_parts) + "।" if hi_parts else "ग्राहक ने शिकायत की।"

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
    return unique_ratio < 0.25

    unique_ratio = len(set(tokens)) / len(tokens)

    # extreme repetition like "आप आप आप आप"
    if unique_ratio < 0.25:
        return True

    # same token repeated continuously
    most_common = max(tokens.count(t) for t in set(tokens))
    if most_common / len(tokens) > 0.6:
        return True

    return False

def is_hallucinated_stt(text: str) -> bool:
    tokens = text.strip().split()

    # Not enough words
    if len(tokens) < 8:
        return True

    # Low lexical diversity
    unique_ratio = len(set(tokens)) / len(tokens)
    if unique_ratio < 0.4:
        return True

    return is_repetitive_gibberish(text)

class WhisperSTT:
    def __init__(self):
        self.model = get_whisper()

    def transcribe(self, path: str) -> Dict[str, Any]:
        segments, info = self.model.transcribe(
            path,
            beam_size=3,
            vad_filter=True,
            language="en",
            temperature=0.0,
            best_of=2,
            condition_on_previous_text=False
        )

        text = " ".join(seg.text for seg in segments).strip()
        if not text:
            raise ValueError("No speech detected")

        return {
            "text": text,
            "language": info.language or "hi",
            "language_confidence": round(info.language_probability, 3),
            "duration_seconds": round(info.duration, 1)
        }

stt_engine = WhisperSTT()

app = FastAPI(title="IVR Call Analyzer – v31 (Ultra Low-Memory Safe + Groq Translation)")

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
    return {"status": "running", "version": "v31-groq", "message": "Using 'small' Whisper + Groq (llama-3.1-8b-instant) for ultra-fast Hindi→English translation"}

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
        # =========================
        # 1️⃣ Speech to Text
        # =========================
        stt = stt_engine.transcribe(path)

        hallucinated = is_hallucinated_stt(stt["text"])
        gibberish = is_repetitive_gibberish(stt["text"])
        low_conf = stt["language_confidence"] < 0.55

        # 🚫 HARD BLOCK: extreme noise
        if gibberish and low_conf:
            raise HTTPException(
                status_code=422,
                detail="No meaningful speech detected in audio"
            )

        # 🚫 SOFT BLOCK: gibberish but confident
        if gibberish:
            return {
                "analysis_valid": False,
                "audio_warning": "Gibberish / non-meaningful speech detected",
                "stt": stt,
                "call_quality_score": call_quality_score(
                    stt["language_confidence"],
                    stt["duration_seconds"]
                ),
                "recommendations": {
                    "alerts": [
                        {
                            "title": "Manual Review Required",
                            "severity": "Warning"
                        }
                    ],
                    "actions": []
                }
            }

        # =========================
        # 2️⃣ Text Analysis (ONLY for valid speech)
        # =========================
        analysis = analyze_text(stt["text"])
        analysis["stt"] = stt
        analysis["call_quality_score"] = call_quality_score(
            stt["language_confidence"],
            stt["duration_seconds"]
        )

        # =========================
        # 3️⃣ Audio Quality Flags
        # =========================
        if hallucinated or low_conf:
            analysis["audio_warning"] = "Low clarity IVR audio – verify manually"

        analysis["analysis_valid"] = True

        # =========================
        # 4️⃣ Recommendations
        # =========================
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
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)