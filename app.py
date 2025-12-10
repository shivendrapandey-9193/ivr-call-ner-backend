###############################################################
#  IVR NER ANALYZER – BACKEND (EN + HINDI + SCORING, IMPROVED+)
#  FastAPI + Groq Whisper Large-v3 + SpaCy + Rule-based AI
#  + Feature Engineering + SQLite + Analytics + RL Flows + Scoring
###############################################################

import os
import json
import tempfile
import re
import random
from datetime import datetime
from typing import List, Optional, Dict, Union
from collections import defaultdict, Counter

# ------------------------------------------------------------
# WARNING / LOG SUPPRESSION (clean console)
# ------------------------------------------------------------
import warnings
import logging
from transformers.utils.logging import set_verbosity_error

warnings.filterwarnings(
    "ignore",
    message="Valid config keys have changed in V2:",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Valid config keys have changed",
    category=UserWarning,
)

set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("pydantic").setLevel(logging.ERROR)

# ------------------------------------------------------------

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

import spacy
from spacy.util import minibatch, compounding
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from groq import Groq
from dotenv import load_dotenv
from spellchecker import SpellChecker
from langdetect import detect, LangDetectException


###############################################################
# ENVIRONMENT + GROQ KEY
###############################################################

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

_groq_client: Optional[Groq] = None


def get_groq_client() -> Groq:
    """
    Lazily create and cache the Groq client.
    """
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set. Cannot initialize Groq client.")
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


if not GROQ_API_KEY:
    print("⚠ WARNING: GROQ_API_KEY not set. Whisper STT endpoints will fail.")


###############################################################
# DATABASE (SQLite via SQLAlchemy)
###############################################################

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ivr_ner.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class CallAnalysis(Base):
    __tablename__ = "call_analysis"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    input_type = Column(String(20))  # "audio" or "text"
    transcript = Column(Text)
    entities_json = Column(Text)
    relationships_json = Column(Text)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


###############################################################
# Pydantic Schemas
###############################################################

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    source: str  # spacy | bert | merged | rule


class Relationship(BaseModel):
    subject: str
    predicate: str
    object: str


class TranscribeAudioResponse(BaseModel):
    transcript: str
    language: Optional[str] = None
    duration: Optional[float] = None


class AnalyzeTextRequest(BaseModel):
    text: str


class SentimentEmotion(BaseModel):
    sentiment_label: str
    sentiment_score: float
    primary_emotion: Optional[str] = None
    emotions: Dict[str, float] = Field(default_factory=dict)


class IntentDetection(BaseModel):
    primary_intent: str
    confidence: float
    intents: Dict[str, float] = Field(default_factory=dict)


class ComplianceCheck(BaseModel):
    overall_score: float
    passed: bool
    warnings: List[str] = Field(default_factory=list)


class ThreatProfanity(BaseModel):
    threat_detected: bool
    profanity_detected: bool
    threat_terms: List[str] = Field(default_factory=list)
    profanity_terms: List[str] = Field(default_factory=list)


class AgentAssist(BaseModel):
    suggestions: List[str] = Field(default_factory=list)
    next_best_action: Optional[str] = None
    call_flow_action: Optional[str] = None


class AnalyzeTextResponse(BaseModel):
    text: str
    language: Optional[str]
    entities: List[Entity]
    relationships: List[Relationship]
    sentiment: SentimentEmotion
    intents: IntentDetection
    summary: str
    agent_assist: AgentAssist
    compliance: ComplianceCheck
    risk_flags: ThreatProfanity
    call_score: int


class AnalyzeAudioResponse(BaseModel):
    transcript: str
    language: Optional[str]
    duration: Optional[float]
    entities: List[Entity]
    relationships: List[Relationship]
    sentiment: SentimentEmotion
    intents: IntentDetection
    summary: str
    agent_assist: AgentAssist
    compliance: ComplianceCheck
    risk_flags: ThreatProfanity
    call_score: int


class HistoryItem(BaseModel):
    id: int
    created_at: datetime
    input_type: str
    transcript: str
    entities: List[Entity]
    relationships: List[Relationship]

    class Config:
        orm_mode = True
        from_attributes = True


class FlowFeedbackRequest(BaseModel):
    call_id: Optional[int] = None
    intent: Optional[str] = None
    chosen_action: str
    reward: float


###############################################################
# NER MODELS – SpaCy + BERT
###############################################################

# ✅ Auto-download en_core_web_sm if missing
try:
    nlp_spacy = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    print("⚠ SpaCy model 'en_core_web_sm' not found. Downloading...")
    download("en_core_web_sm")
    nlp_spacy = spacy.load("en_core_web_sm")

BERT_MODEL_NAME = "dslim/bert-base-NER"

try:
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModelForTokenClassification.from_pretrained(BERT_MODEL_NAME)
    nlp_bert = pipeline(
        "ner",
        model=bert_model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    BERT_AVAILABLE = True
except Exception:
    nlp_bert = None
    BERT_AVAILABLE = False


###############################################################
# FEATURE ENGINEERING – ADVANCED TEXT CLEANING
###############################################################

spell = SpellChecker()

FILLER_WORDS = {
    "uh", "umm", "hmm", "like", "basically", "actually", "you know",
    "i mean", "so yeah", "okay so", "well", "right", "listen",
}

IVR_BOILERPLATE = [
    "this call may be recorded for quality purposes",
    "press 1 to continue",
    "press 2",
    "press 3",
    "please wait while we connect your call",
    "your call is important to us",
    "welcome to the interactive voice response system",
    "welcome to our customer care",
    "welcome to customer support",
]


###############################################################
# SPOKEN NUMBER → DIGIT CONVERSION (EN + HINDI)
###############################################################

NUMBER_WORDS_EN = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

NUMBER_WORDS_HI = {
    "एक": "1",
    "दो": "2",
    "तीन": "3",
    "चार": "4",
    "पांच": "5",
    "पाँच": "5",
    "छः": "6",
    "छह": "6",
    "सात": "7",
    "आठ": "8",
    "नौ": "9",
    "शून्य": "0",
}


def _normalize_word_token(token: Optional[str]) -> str:
    """
    Lowercase and strip punctuation, keep Devanagari for Hindi.
    """
    if token is None:
        return ""
    return re.sub(r"[^\w\u0900-\u097F]", "", token.lower())


def spoken_numbers_to_digits(text: str) -> str:
    """
    Convert spoken digit sequences in EN / HI to numeric strings.
    """
    if not text:
        return ""

    words = text.split()
    output: List[str] = []
    i = 0

    while i < len(words):
        raw = words[i]
        w = _normalize_word_token(raw)

        if w in NUMBER_WORDS_EN:
            digit_str = NUMBER_WORDS_EN[w]
            i += 1
            while i < len(words) and _normalize_word_token(words[i]) in NUMBER_WORDS_EN:
                digit_str += NUMBER_WORDS_EN[_normalize_word_token(words[i])]
                i += 1
            output.append(digit_str)
            continue

        if w in NUMBER_WORDS_HI:
            digit_str = NUMBER_WORDS_HI[w]
            i += 1
            while i < len(words) and _normalize_word_token(words[i]) in NUMBER_WORDS_HI:
                digit_str += NUMBER_WORDS_HI[_normalize_word_token(words[i])]
                i += 1
            output.append(digit_str)
            continue

        output.append(raw)
        i += 1

    merged = " ".join(output)

    # EN digit + EN word
    pattern_en = re.compile(
        r"\b(?P<num>\d{5,})\s+(?P<word>zero|one|two|three|four|five|six|seven|eight|nine)\b",
        re.IGNORECASE,
    )

    def _merge_match_en(m: re.Match) -> str:
        num = m.group("num")
        word = m.group("word").lower()
        return num + NUMBER_WORDS_EN[word]

    while True:
        new_merged, count = pattern_en.subn(_merge_match_en, merged)
        if count == 0:
            break
        merged = new_merged

    # EN digit + HI word
    pattern_hi = re.compile(
        r"\b(?P<num>\d{5,})\s+(?P<word>एक|दो|तीन|चार|पांच|पाँच|छः|छह|सात|आठ|नौ|शून्य)\b",
        re.IGNORECASE,
    )

    def _merge_match_hi(m: re.Match) -> str:
        num = m.group("num")
        word = m.group("word")
        return num + NUMBER_WORDS_HI.get(word, "")

    while True:
        new_merged, count = pattern_hi.subn(_merge_match_hi, merged)
        if count == 0:
            break
        merged = new_merged

    return merged


def feature_engineer_text(text: str) -> str:
    """
    Advanced text cleanup, noise removal, normalization & feature engineering.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    temp = text.lower()

    for phrase in IVR_BOILERPLATE:
        temp = temp.replace(phrase, " ")

    for fw in FILLER_WORDS:
        temp = re.sub(rf"\b{re.escape(fw)}\b", " ", temp)

    temp = re.sub(r"\s+", " ", temp).strip()
    temp = re.sub(r"[!?.,]{2,}", ".", temp)
    temp = temp.replace("..", ".")
    temp = re.sub(r"\s+([.,!?])", r"\1", temp)

    temp = spoken_numbers_to_digits(temp)

    corrected_words: List[str] = []
    for word in temp.split():
        if re.fullmatch(r"\d{3,12}", word):
            corrected_words.append(word)
            continue
        if re.search(r"[₹$€£]|,\d{3}", word):
            corrected_words.append(word)
            continue
        if len(word) < 3:
            corrected_words.append(word)
            continue
        # keep Hindi tokens untouched
        if re.search(r"[\u0900-\u097F]", word):
            corrected_words.append(word)
            continue

        try:
            corrected = spell.correction(word)
            if corrected is None:
                corrected = word
        except Exception:
            corrected = word

        corrected_words.append(corrected)

    cleaned = " ".join(corrected_words)

    sentences = [s.strip() for s in re.split(r"[.]", cleaned) if s.strip()]
    capped_sentences: List[str] = []
    for s in sentences:
        if not s:
            continue
        if re.search(r"[A-Za-z]", s):
            capped_sentences.append(s[0].upper() + s[1:])
        else:
            capped_sentences.append(s)

    cleaned = ". ".join(capped_sentences)
    return cleaned.strip()


###############################################################
# AUDIO VALIDATION
###############################################################

ALLOWED_AUDIO_EXT = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg",
    ".mp4", ".webm", ".mpeg", ".mpga",
}
MAX_AUDIO_MB = 25


def validate_audio(filename: str, size: int):
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO_EXT:
        raise ValueError(
            f"Unsupported audio format: {ext}. "
            f"Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXT))}"
        )

    if size > MAX_AUDIO_MB * 1024 * 1024:
        raise ValueError(f"File too large (> {MAX_AUDIO_MB} MB).")


###############################################################
# GROQ WHISPER LARGE-v3 TRANSCRIPTION
###############################################################

def transcribe_audio(file_bytes: bytes, filename: str) -> Dict[str, Optional[Union[str, float]]]:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured on server.")

    client = get_groq_client()

    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp.flush()

        with open(tmp.name, "rb") as audio:
            response = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio,
                response_format="verbose_json",
            )

    if isinstance(response, dict):
        text = response.get("text", "") or ""
        language = response.get("language")
        duration = response.get("duration")
    else:
        text = getattr(response, "text", "") or ""
        language = getattr(response, "language", None)
        duration = getattr(response, "duration", None)

    return {
        "transcript": text,
        "language": language,
        "duration": duration,
    }


###############################################################
# HELPER: HINDI TEXT DETECTION
###############################################################

def is_hindi_text(text: str) -> bool:
    """
    True if text contains Devanagari characters.
    """
    return bool(re.search(r"[\u0900-\u097F]", text))


###############################################################
# NER PIPELINE (SpaCy + BERT + Rule-based)
###############################################################

def run_spacy_ner(text: str) -> List[Dict]:
    doc = nlp_spacy(text)
    ents: List[Dict] = []
    for ent in doc.ents:
        ents.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "spacy",
            }
        )
    return ents


def _normalize_bert_label(label: str) -> str:
    mapping = {
        "PER": "PERSON",
        "ORG": "ORG",
        "LOC": "LOC",
        "MISC": "MISC",
    }
    return mapping.get(label, label)


def run_bert_ner(text: str) -> List[Dict]:
    if not BERT_AVAILABLE or nlp_bert is None:
        return []

    try:
        results = nlp_bert(text)
    except Exception as e:
        print(f"⚠ BERT NER failed at runtime, falling back to spaCy only: {e}")
        return []

    ents: List[Dict] = []
    for r in results:
        ents.append(
            {
                "text": r.get("word", ""),
                "label": _normalize_bert_label(r.get("entity_group", "")),
                "start": int(r.get("start", 0)),
                "end": int(r.get("end", 0)),
                "source": "bert",
            }
        )
    return ents


def merge_entities(spacy_entities: List[Dict], bert_entities: List[Dict]) -> List[Dict]:
    merged: Dict[tuple, Dict] = {}

    for e in spacy_entities:
        key = (e["start"], e["end"], e["label"])
        merged[key] = e.copy()

    for e in bert_entities:
        key = (e["start"], e["end"], e["label"])
        if key in merged:
            merged[key]["source"] = "merged"
        else:
            merged[key] = e.copy()

    return list(merged.values())


def add_rule_based_entities(text: str, entities: List[Dict]) -> List[Dict]:
    """
    Add/adjust rule-based entities like ACCOUNT_ID and ISSUE_TYPE.
    - For 8–12 digit numbers, override CARDINAL → ACCOUNT_ID.
    """
    new_entities = [e.copy() for e in entities]

    # 1) Upgrade numeric IDs to ACCOUNT_ID
    for e in new_entities:
        if e.get("label") in {"CARDINAL", "DATE"}:
            if re.fullmatch(r"\d{8,12}", e.get("text", "").strip()):
                e["label"] = "ACCOUNT_ID"
                e["source"] = "rule"

    # 2) Ensure any 8-12 digit sequence in text is captured as ACCOUNT_ID
    for m in re.finditer(r"\b\d{8,12}\b", text):
        start, end = m.start(), m.end()
        span_text = m.group(0)
        span_found = False
        for e in new_entities:
            if e["start"] == start and e["end"] == end:
                # Already there – force label
                e["label"] = "ACCOUNT_ID"
                e["source"] = "rule"
                span_found = True
                break
        if not span_found:
            new_entities.append(
                {
                    "text": span_text,
                    "label": "ACCOUNT_ID",
                    "start": start,
                    "end": end,
                    "source": "rule",
                }
            )

    # 3) ISSUE_TYPE based on common phrases (EN + HI)
    issue_keywords = [
        "payment failure",
        "payment failed",
        "failed payment",
        "payment error",
        "payment did not go through",
        "double payment",
        "wrong deduction",
        "refund",
        "login issue",
        "password reset",
        "account blocked",
        "card blocked",
        "transaction failed",
        "balance mismatch",
        "भुगतान में समस्या",
        "पैसा दो बार कट गया",
        "रिफंड चाहिए",
    ]

    lowered = text.lower()
    for kw in issue_keywords:
        idx = lowered.find(kw)
        while idx != -1:
            start, end = idx, idx + len(kw)
            new_entities.append(
                {
                    "text": text[start:end],
                    "label": "ISSUE_TYPE",
                    "start": start,
                    "end": end,
                    "source": "rule",
                }
            )
            idx = lowered.find(kw, end)

    return new_entities


def _is_bad_date_entity(text: str) -> bool:
    t = text.strip().lower()
    if re.fullmatch(r"\d{8,12}", t):
        return True
    if t in {"id is", "account id", "acc id"}:
        return True
    return False


def _postprocess_ner_for_hindi(text: str, entities: List[Dict]) -> List[Dict]:
    """
    Clean up SpaCy noise on pure Hindi spans:
    - Drop PRODUCT / ORG / NORP / CARDINAL when text is Hindi-only,
      except when CARDINAL contains digits (we keep numbers).
    """
    if not is_hindi_text(text):
        return entities

    cleaned: List[Dict] = []
    for e in entities:
        span = e.get("text", "")
        label = e.get("label", "")
        source = e.get("source", "")

        if source == "spacy" and label in {"PRODUCT", "ORG", "NORP", "CARDINAL"}:
            # If it's a Hindi-only span with no digits, drop it
            if is_hindi_text(span) and not re.search(r"\d", span):
                continue

        cleaned.append(e)

    return cleaned


def analyze_text_ner(text: str) -> List[Dict]:
    try:
        spacy_ents = run_spacy_ner(text)
    except Exception as e:
        print(f"❌ SpaCy NER failed: {e}")
        spacy_ents = []

    bert_ents = run_bert_ner(text)
    merged = merge_entities(spacy_ents, bert_ents)

    filtered: List[Dict] = []
    for e in merged:
        if e["label"] == "DATE" and _is_bad_date_entity(e["text"]):
            continue
        filtered.append(e)

    final = add_rule_based_entities(text, filtered)

    clean_ents: List[Dict] = []
    for e in final:
        if e["label"] == "DATE" and _is_bad_date_entity(e["text"]):
            continue
        if e["label"] == "CARDINAL" and _normalize_word_token(e["text"]) in NUMBER_WORDS_EN:
            continue
        clean_ents.append(e)

    # Hindi-specific cleanup (drop noisy SpaCy entities on Hindi spans)
    clean_ents = _postprocess_ner_for_hindi(text, clean_ents)

    return clean_ents


###############################################################
# RELATIONSHIP EXTRACTION
###############################################################

def extract_relationships(text: str, ents: List[Dict]) -> List[Dict]:
    relationships: List[Dict] = []
    subject = "Customer"

    issues = [e for e in ents if e["label"] == "ISSUE_TYPE"]
    accounts = [e for e in ents if e["label"] == "ACCOUNT_ID"]
    dates = [e for e in ents if e["label"] in ("DATE", "TIME")]

    for issue in issues:
        relationships.append(
            {
                "subject": subject,
                "predicate": "reports",
                "object": issue["text"],
            }
        )

    for acc in accounts:
        relationships.append(
            {
                "subject": subject,
                "predicate": "has_account",
                "object": acc["text"],
            }
        )

    if dates:
        relationships.append(
            {
                "subject": subject,
                "predicate": "called_on",
                "object": dates[0]["text"],
            }
        )

    return relationships


###############################################################
# MULTI-LANGUAGE DETECTION (TEXT)
###############################################################

def detect_language_text(text: str) -> Optional[str]:
    try:
        cleaned = text.strip()
        if not cleaned:
            return None
        return detect(cleaned)
    except LangDetectException:
        return None
    except Exception:
        return None


###############################################################
# SENTIMENT + EMOTION ANALYSIS (LEXICON-BASED, EN + HI)
###############################################################

POSITIVE_WORDS = {
    # English
    "good", "great", "awesome", "excellent", "happy", "satisfied",
    "resolved", "thank", "thanks", "helpful", "love", "wonderful",
    "nice",
    # Hindi positive
    "धन्यवाद", "शुक्रिया", "अच्छा", "बहुत अच्छा", "शानदार", "संतुष्ट",
    "खुश", "मंगलमय", "शुभ", "सहायता", "मदद",
}

NEGATIVE_WORDS = {
    # English
    "bad", "terrible", "horrible", "angry", "upset", "sad", "unhappy",
    "frustrated", "annoyed", "disappointed", "worst", "issue", "problem",
    "complaint", "escalate", "escalation",
    # Hindi negatives
    "समस्या", "दिक्कत", "गलत", "खराब", "नाराज", "नाराज़", "गुस्सा",
    "परेशान", "तुरंत",  # often used in tense / escalated tone
}

EMOTION_LEXICON = {
    "anger": {"angry", "furious", "mad", "annoyed", "irritated",
              "frustrated", "गुस्सा", "नाराज", "नाराज़"},
    "sadness": {"sad", "upset", "unhappy", "depressed", "crying", "disappointed"},
    "fear": {"scared", "afraid", "worried", "anxious", "fear", "nervous"},
    "joy": {"happy", "glad", "pleased", "satisfied", "great", "awesome",
            "good", "खुश", "शानदार"},
    "surprise": {"surprised", "shocked", "amazed", "wow"},
}


def _analyze_sentiment_emotion_generic(text: str) -> SentimentEmotion:
    tokens = re.findall(r"\w+|\w+[\u0900-\u097F]+", text.lower())
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    total = pos + neg

    if total == 0:
        score = 0.0
    else:
        score = (pos - neg) / float(total)

    if score >= 0.4:
        label = "very_positive"
    elif score >= 0.1:
        label = "positive"
    elif score <= -0.4:
        label = "very_negative"
    elif score <= -0.1:
        label = "negative"
    else:
        label = "neutral"

    emotion_counts: Dict[str, int] = {k: 0 for k in EMOTION_LEXICON}
    for tok in tokens:
        for emo, words in EMOTION_LEXICON.items():
            if tok in words:
                emotion_counts[emo] += 1

    total_emo = sum(emotion_counts.values())
    if total_emo == 0:
        primary_emo = None
        emo_dist = {}
    else:
        primary_emo = max(emotion_counts.items(), key=lambda x: x[1])[0]
        emo_dist = {k: v / total_emo for k, v in emotion_counts.items() if v > 0}

    return SentimentEmotion(
        sentiment_label=label,
        sentiment_score=score,
        primary_emotion=primary_emo,
        emotions=emo_dist,
    )


def _analyze_hindi_sentiment(text: str) -> SentimentEmotion:
    """
    Hindi-specific sentiment booster:
    - Detects gratitude / trust / hope as positive
    - Detects strong negative complaint words
    """
    lower_text = text.lower()

    positive_keywords = [
        "धन्यवाद", "शुक्रिया", "आभारी", "सहायता", "मदद",
        "विश्वास", "भरोसा", "उम्मीद", "आपकी सहायता के लिए धन्यवाद",
        "आपकी मदद के लिए धन्यवाद", "आपका दिन शुभ हो",
    ]

    negative_keywords = [
        "गुस्सा", "नाराज", "नाराज़", "खराब सेवा", "बहुत बुरा",
        "धोखा", "परेशान", "शिकायत", "निराश", "बहुत खराब",
    ]

    has_pos = any(pk in text for pk in positive_keywords)
    has_neg = any(nk in text for nk in negative_keywords)

    if has_pos and not has_neg:
        return SentimentEmotion(
            sentiment_label="positive",
            sentiment_score=0.5,
            primary_emotion="gratitude",
            emotions={"gratitude": 1.0},
        )

    if has_neg and not has_pos:
        return SentimentEmotion(
            sentiment_label="negative",
            sentiment_score=-0.5,
            primary_emotion="anger",
            emotions={"anger": 1.0},
        )

    if has_pos and has_neg:
        # mixed feeling
        return SentimentEmotion(
            sentiment_label="neutral",
            sentiment_score=0.0,
            primary_emotion=None,
            emotions={},
        )

    # no strong Hindi cues → neutral
    return SentimentEmotion(
        sentiment_label="neutral",
        sentiment_score=0.0,
        primary_emotion=None,
        emotions={},
    )


def analyze_sentiment_emotion(text: str, language: Optional[str] = None) -> SentimentEmotion:
    """
    Final sentiment router:
    - English / non-Hindi → generic lexicon model
    - Hindi / Hinglish (contains Devanagari or lang=hi) → combine generic + Hindi booster
    """
    generic = _analyze_sentiment_emotion_generic(text)

    if language == "hi" or is_hindi_text(text):
        hi = _analyze_hindi_sentiment(text)

        # If Hindi booster gives a non-neutral signal, prefer it
        if hi.sentiment_label != "neutral" or hi.sentiment_score != 0.0:
            # merge in generic emotions if Hindi has none
            if not hi.emotions and generic.emotions:
                hi.emotions = generic.emotions
            if hi.primary_emotion is None:
                hi.primary_emotion = generic.primary_emotion
            return hi

    return generic


###############################################################
# INTENT DETECTION (RULE-BASED AI)
###############################################################

INTENT_KEYWORDS = {
    "payment_issue": [
        "payment failure", "payment failed", "failed payment",
        "payment error", "payment did not go through", "double payment",
        "wrong deduction", "refund", "charged twice", "money deducted",
        "भुगतान में समस्या", "पैसा दो बार कट गया", "रिफंड चाहिए",
        "payment दो बार कट", "payment 2 बार कट",
    ],
    "login_issue": [
        "login issue", "password reset", "forgot password",
        "cannot login", "unable to login", "account blocked", "card blocked",
    ],
    "balance_inquiry": [
        "balance", "available balance", "account balance", "remaining balance",
        "बैलेंस", "खाते में कितना है",
    ],
    "card_lost": [
        "lost card", "stolen card", "block my card", "card stolen",
    ],
    "general_complaint": [
        "complaint", "not happy", "bad service", "very bad", "worst service",
        "raise a complaint", "escalate", "escalation", "खराब सेवा",
    ],
    "information_query": [
        "want to know", "need information", "details about", "how to",
        "explain", "clarify",
    ],
    "greeting_smalltalk": [
        "hello", "hi", "good morning", "good evening", "how are you",
        "नमस्ते", "नमस्कार", "हेलो", "हैलो",
        "ग्राहक सहायता में आपका स्वागत है",
        "ग्राहक सेवा में आपका स्वागत है",
    ],
}


def detect_intents(text: str, ents: Optional[List[Dict]] = None) -> IntentDetection:
    if ents is None:
        ents = []

    lowered = text.lower()
    scores: Dict[str, float] = defaultdict(float)

    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in lowered:
                scores[intent] += 1.0

    for e in ents:
        if e.get("label") == "ISSUE_TYPE":
            scores["payment_issue"] += 0.5
            scores["general_complaint"] += 0.5
        if e.get("label") == "ACCOUNT_ID":
            scores["balance_inquiry"] += 0.3

    if not scores:
        return IntentDetection(
            primary_intent="unknown",
            confidence=0.0,
            intents={},
        )

    max_score = max(scores.values())
    intents_norm = {k: v / max_score for k, v in scores.items()}
    primary_intent = max(intents_norm.items(), key=lambda x: x[1])[0]
    confidence = intents_norm[primary_intent]

    # Auto-improve intent: if ISSUE_TYPE exists, favor payment_issue
    if any(e.get("label") == "ISSUE_TYPE" for e in ents) and "payment_issue" in intents_norm:
        primary_intent = "payment_issue"
        confidence = intents_norm["payment_issue"]

    return IntentDetection(
        primary_intent=primary_intent,
        confidence=confidence,
        intents=intents_norm,
    )


###############################################################
# THREAT & PROFANITY DETECTION
###############################################################

PROFANITY_WORDS = {
    "shit", "damn", "bloody", "bastard", "idiot", "stupid",
}

THREAT_WORDS = {
    "kill", "hurt", "attack", "lawsuit", "legal action", "court case",
    "file a case", "police complaint", "police case",
}


def detect_threats_profanity(text: str) -> ThreatProfanity:
    lowered = text.lower()
    tokens = set(re.findall(r"\w+|\w+[\u0900-\u097F]+", lowered))

    prof_terms = sorted({w for w in tokens if w in PROFANITY_WORDS})
    threat_terms = []

    for phrase in THREAT_WORDS:
        if " " in phrase:
            if phrase in lowered:
                threat_terms.append(phrase)
        else:
            if phrase in tokens:
                threat_terms.append(phrase)

    return ThreatProfanity(
        threat_detected=bool(threat_terms),
        profanity_detected=bool(prof_terms),
        threat_terms=sorted(set(threat_terms)),
        profanity_terms=prof_terms,
    )


###############################################################
# COMPLIANCE CHECKER (EN + HINDI AWARE)
###############################################################

def check_compliance(text: str) -> ComplianceCheck:
    """
    Heuristic compliance check with Hindi-aware rules:
      - Greeting
      - Identity / verification
      - Mandatory disclosure
      - Proper closing
    """
    lowered = text.lower()

    # 1) Greeting
    greeting_ok = False
    greeting_phrases_en = [
        "hello", "hi", "good morning", "good afternoon", "good evening",
        "welcome", "thank you for calling",
    ]
    greeting_phrases_hi = [
        "नमस्ते",
        "नमस्कार",
        "हेलो",
        "हैलो",
        "ग्राहक सहायता में आपका स्वागत है",
        "ग्राहक सेवा में आपका स्वागत है",
    ]

    for g in greeting_phrases_en:
        if g in lowered:
            greeting_ok = True
            break
    if not greeting_ok:
        for g in greeting_phrases_hi:
            if g in text:
                greeting_ok = True
                break

    # 2) Identity verification
    id_ok = bool(
        re.search(
            r"(account number|customer id|verify your identity|registered mobile|date of birth|अकाउंट नंबर)",
            lowered,
        )
    )

    # 3) Mandatory disclosure (EN + HI)
    disclosure_ok = False
    disclosure_phrases_en = [
        "this call may be recorded",
        "recorded for quality",
        "monitored or recorded",
        "terms and conditions apply",
    ]
    disclosure_phrases_hi = [
        "यह कॉल गुणवत्ता और प्रशिक्षण उद्देश्यों के लिए रिकॉर्ड की जा सकती है",
        "यह कॉल रिकॉर्ड की जा सकती है",
        "यह कॉल रिकॉर्ड की जाएगी",
        "गुणवत्ता और प्रशिक्षण उद्देश्यों के लिए रिकॉर्ड",
    ]

    for d in disclosure_phrases_en:
        if d in lowered:
            disclosure_ok = True
            break
    if not disclosure_ok:
        for d in disclosure_phrases_hi:
            if d in text:
                disclosure_ok = True
                break

    if not disclosure_ok and ("कॉल" in text and "रिकॉर्ड" in text):
        disclosure_ok = True

    # 4) Proper closing
    closing_ok = bool(
        re.search(
            r"(thank you for calling|have a nice day|have a great day|is there anything else i can help|धन्यवाद)",
            lowered,
        )
    )

    passed_checks = 0
    total_checks = 4
    warnings: List[str] = []

    if greeting_ok:
        passed_checks += 1
    else:
        warnings.append("Missing proper greeting.")

    if id_ok:
        passed_checks += 1
    else:
        warnings.append("Customer identity not clearly verified.")

    if disclosure_ok:
        passed_checks += 1
    else:
        warnings.append("Mandatory disclosure about recording/terms is missing.")

    if closing_ok:
        passed_checks += 1
    else:
        warnings.append("No proper closing phrase / thanks.")

    score = passed_checks / float(total_checks)
    passed = score >= 0.75

    return ComplianceCheck(
        overall_score=score,
        passed=passed,
        warnings=warnings,
    )


###############################################################
# AUTO CALL SUMMARY GENERATION
###############################################################

def generate_call_summary(
    text: str,
    ents: List[Dict],
    intents: IntentDetection,
    sentiment: SentimentEmotion,
    language: Optional[str],
) -> str:
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    short_sentences = sentences[:3]

    issue_types = [e["text"] for e in ents if e.get("label") == "ISSUE_TYPE"]
    accounts = [e["text"] for e in ents if e.get("label") == "ACCOUNT_ID"]

    parts = []

    if language:
        parts.append(f"Language detected: {language}.")
    if intents.primary_intent != "unknown":
        parts.append(f"Primary intent: {intents.primary_intent.replace('_', ' ')}.")
    parts.append(f"Sentiment: {sentiment.sentiment_label} (score={sentiment.sentiment_score:.2f}).")

    if issue_types:
        parts.append(f"Issue types mentioned: {', '.join(set(issue_types))}.")
    if accounts:
        parts.append(f"Account references: {', '.join(set(accounts))}.")

    if short_sentences:
        parts.append("Call context: " + " ".join(short_sentences))

    return " ".join(parts)


###############################################################
# REINFORCEMENT LEARNING STYLE ADAPTIVE CALL FLOWS
###############################################################

class FlowPolicy:
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.stats: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {"success": 0.0, "count": 0.0})
        )

    def _candidate_actions(self, intent: str) -> List[str]:
        if intent == "payment_issue":
            return [
                "Ask for last 4 digits of account/card",
                "Verify recent transaction details",
                "Offer to check payment gateway logs",
            ]
        if intent == "login_issue":
            return [
                "Guide password reset flow",
                "Verify registered email or mobile",
            ]
        if intent == "balance_inquiry":
            return [
                "Authenticate customer and read current balance",
                "Offer SMS of current balance",
            ]
        if intent == "card_lost":
            return [
                "Authenticate and block card immediately",
                "Offer to reissue card and provide timeline",
            ]
        if intent == "general_complaint":
            return [
                "Acknowledge issue and empathize",
                "Offer escalation to supervisor",
            ]
        if intent == "information_query":
            return [
                "Provide requested information clearly",
                "Offer to send summary via SMS/email",
            ]
        return [
            "Ask clarifying question",
            "Transfer to human agent",
        ]

    def choose_action(self, intent: str) -> str:
        actions = self._candidate_actions(intent)
        if not actions:
            return "Ask clarifying question"

        state = intent or "unknown"
        for a in actions:
            _ = self.stats[state][a]

        if random.random() < self.epsilon:
            return random.choice(actions)

        best_action = None
        best_score = -1e9
        for a in actions:
            data = self.stats[state][a]
            count = data["count"]
            success = data["success"]
            avg_reward = success / count if count > 0 else 0.0
            if avg_reward > best_score:
                best_score = avg_reward
                best_action = a

        return best_action or random.choice(actions)

    def update(self, intent: str, action: str, reward: float):
        state = intent or "unknown"
        data = self.stats[state][action]
        data["count"] += 1.0
        data["success"] += float(reward)


flow_policy = FlowPolicy(epsilon=0.15)


###############################################################
# REAL-TIME AGENT ASSIST SUGGESTIONS
###############################################################

def build_agent_assist(
    text: str,
    ents: List[Dict],
    intents: IntentDetection,
    sentiment: SentimentEmotion,
    compliance: ComplianceCheck,
    risk: ThreatProfanity,
) -> AgentAssist:
    suggestions: List[str] = []

    intent = intents.primary_intent
    if intent == "payment_issue":
        suggestions.append("Confirm transaction date, amount, and mode of payment.")
        suggestions.append("Check for duplicate or pending transactions.")
    elif intent == "login_issue":
        suggestions.append("Guide the customer through secure password reset.")
        suggestions.append("Confirm username / registered mobile or email.")
    elif intent == "balance_inquiry":
        suggestions.append("Authenticate customer and share current balance.")
    elif intent == "card_lost":
        suggestions.append("Immediately block the card and confirm last known transactions.")
    elif intent == "general_complaint":
        suggestions.append("Acknowledge the issue and apologize for the inconvenience.")
        suggestions.append("Offer escalation if the customer remains dissatisfied.")
    elif intent == "information_query":
        suggestions.append("Provide concise information and confirm if it answers the query.")
    else:
        suggestions.append("Ask a clarifying question to better understand the issue.")

    if sentiment.sentiment_label in {"negative", "very_negative"}:
        suggestions.append("Use empathetic phrases and reassure the customer.")
    elif sentiment.sentiment_label in {"very_positive", "positive"}:
        suggestions.append("Maintain positive tone and confirm if anything else is needed.")

    for w in compliance.warnings:
        suggestions.append(f"Compliance gap: {w}")

    if risk.threat_detected:
        suggestions.append("Consider escalating due to threat / legal language.")
    if risk.profanity_detected:
        suggestions.append("Maintain calm tone; avoid mirroring customer's language.")

    call_flow_action = flow_policy.choose_action(intent)
    nba = suggestions[0] if suggestions else call_flow_action

    return AgentAssist(
        suggestions=list(dict.fromkeys(suggestions)),
        next_best_action=nba,
        call_flow_action=call_flow_action,
    )


###############################################################
# CALL SCORE SYSTEM (IMPROVED)
###############################################################

def compute_call_score(intents: IntentDetection,
                       sentiment: SentimentEmotion,
                       compliance: ComplianceCheck,
                       risk: ThreatProfanity) -> int:
    # Intent strength (0–30)
    intent_score = intents.confidence * 30

    # Sentiment (−1 to +1) → 0–20
    sentiment_scaled = (1 + sentiment.sentiment_score) / 2
    sentiment_score = sentiment_scaled * 20

    # Compliance (0–1 → 0–40)
    compliance_score = compliance.overall_score * 40

    # Risk penalty
    penalty = 0
    if risk.threat_detected:
        penalty -= 40
    if risk.profanity_detected:
        penalty -= 25

    # Soften penalty if intent is clear & some compliance exists
    if intents.primary_intent != "unknown" and compliance.overall_score >= 0.25:
        penalty *= 0.6

    final_score = intent_score + sentiment_score + compliance_score + penalty
    final_score = max(0, min(100, round(final_score)))
    return final_score


###############################################################
# FASTAPI APP SETUP
###############################################################

app = FastAPI(title="IVR NER Analyzer – All-in-One Backend", version="2.4.0")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


###############################################################
# API ENDPOINTS – CORE
###############################################################

@app.post("/api/transcribe-audio", response_model=TranscribeAudioResponse)
async def api_transcribe_audio(file: UploadFile = File(...)):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured on server.")

    data = await file.read()
    try:
        validate_audio(file.filename, len(data))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = transcribe_audio(data, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    try:
        cleaned_transcript = feature_engineer_text(result.get("transcript", "") or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text cleaning failed: {e}")

    if not cleaned_transcript:
        raise HTTPException(status_code=500, detail="Transcription returned empty or unusable text.")

    return TranscribeAudioResponse(
        transcript=cleaned_transcript,
        language=result.get("language"),
        duration=result.get("duration"),
    )


@app.post("/api/analyze-text", response_model=AnalyzeTextResponse)
async def api_analyze_text(req: AnalyzeTextRequest, db: Session = Depends(get_db)):
    try:
        cleaned_text = feature_engineer_text(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text cleaning failed: {e}")

    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Text is empty after cleaning.")

    language = detect_language_text(cleaned_text)

    try:
        ents = analyze_text_ner(cleaned_text)
        rels = extract_relationships(cleaned_text, ents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NER analysis failed: {e}")

    sentiment = analyze_sentiment_emotion(cleaned_text, language)
    intents = detect_intents(cleaned_text, ents)
    risk = detect_threats_profanity(cleaned_text)
    compliance = check_compliance(cleaned_text)
    summary = generate_call_summary(cleaned_text, ents, intents, sentiment, language)
    agent_assist = build_agent_assist(cleaned_text, ents, intents, sentiment, compliance, risk)
    score = compute_call_score(intents, sentiment, compliance, risk)

    try:
        record = CallAnalysis(
            input_type="text",
            transcript=cleaned_text,
            entities_json=json.dumps(ents, ensure_ascii=False),
            relationships_json=json.dumps(rels, ensure_ascii=False),
        )
        db.add(record)
        db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database save failed: {e}")

    return AnalyzeTextResponse(
        text=cleaned_text,
        language=language,
        entities=ents,
        relationships=rels,
        sentiment=sentiment,
        intents=intents,
        summary=summary,
        agent_assist=agent_assist,
        compliance=compliance,
        risk_flags=risk,
        call_score=score,
    )


@app.post("/api/analyze-audio", response_model=AnalyzeAudioResponse)
async def api_analyze_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured on server.")

    data = await file.read()
    try:
        validate_audio(file.filename, len(data))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        t = transcribe_audio(data, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    try:
        cleaned_transcript = feature_engineer_text(t.get("transcript", "") or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text cleaning failed: {e}")

    if not cleaned_transcript:
        raise HTTPException(status_code=500, detail="Transcription returned empty or unusable text.")

    language = t.get("language") or detect_language_text(cleaned_transcript)

    try:
        ents = analyze_text_ner(cleaned_transcript)
        rels = extract_relationships(cleaned_transcript, ents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NER analysis failed: {e}")

    sentiment = analyze_sentiment_emotion(cleaned_transcript, language)
    intents = detect_intents(cleaned_transcript, ents)
    risk = detect_threats_profanity(cleaned_transcript)
    compliance = check_compliance(cleaned_transcript)
    summary = generate_call_summary(cleaned_transcript, ents, intents, sentiment, language)
    agent_assist = build_agent_assist(cleaned_transcript, ents, intents, sentiment, compliance, risk)
    score = compute_call_score(intents, sentiment, compliance, risk)

    try:
        record = CallAnalysis(
            input_type="audio",
            transcript=cleaned_transcript,
            entities_json=json.dumps(ents, ensure_ascii=False),
            relationships_json=json.dumps(rels, ensure_ascii=False),
        )
        db.add(record)
        db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database save failed: {e}")

    return AnalyzeAudioResponse(
        transcript=cleaned_transcript,
        language=language,
        duration=t.get("duration"),
        entities=ents,
        relationships=rels,
        sentiment=sentiment,
        intents=intents,
        summary=summary,
        agent_assist=agent_assist,
        compliance=compliance,
        risk_flags=risk,
        call_score=score,
    )


@app.get("/api/history", response_model=List[HistoryItem])
async def api_history(limit: int = 50, db: Session = Depends(get_db)):
    limit = max(1, min(200, limit))
    rows = (
        db.query(CallAnalysis)
        .order_by(CallAnalysis.created_at.desc())
        .limit(limit)
        .all()
    )

    items: List[HistoryItem] = []
    for r in rows:
        ents = json.loads(r.entities_json or "[]")
        rels = json.loads(r.relationships_json or "[]")
        items.append(
            HistoryItem(
                id=r.id,
                created_at=r.created_at,
                input_type=r.input_type,
                transcript=r.transcript,
                entities=ents,
                relationships=rels,
            )
        )
    return items


###############################################################
# CALL ANALYTICS DASHBOARD ENDPOINT
###############################################################

@app.get("/api/analytics-dashboard")
async def api_analytics_dashboard(limit: int = 200, db: Session = Depends(get_db)):
    limit = max(1, min(1000, limit))
    rows = (
        db.query(CallAnalysis)
        .order_by(CallAnalysis.created_at.desc())
        .limit(limit)
        .all()
    )

    total_calls = len(rows)
    by_type = Counter()
    by_language = Counter()
    by_intent = Counter()
    issue_type_counts = Counter()
    sentiment_scores: List[float] = []
    threat_count = 0
    profanity_count = 0

    for r in rows:
        by_type[r.input_type or "unknown"] += 1
        transcript = r.transcript or ""
        ents = json.loads(r.entities_json or "[]")

        lang = detect_language_text(transcript)
        if lang:
            by_language[lang] += 1

        sentiment = analyze_sentiment_emotion(transcript, lang)
        sentiment_scores.append(sentiment.sentiment_score)

        intents = detect_intents(transcript, ents)
        by_intent[intents.primary_intent] += 1

        risk = detect_threats_profanity(transcript)
        if risk.threat_detected:
            threat_count += 1
        if risk.profanity_detected:
            profanity_count += 1

        for e in ents:
            if e.get("label") == "ISSUE_TYPE":
                issue_type_counts[e["text"].lower()] += 1

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

    return {
        "total_calls": total_calls,
        "by_input_type": dict(by_type),
        "by_language": dict(by_language),
        "by_primary_intent": dict(by_intent),
        "avg_sentiment_score": avg_sentiment,
        "threat_call_count": threat_count,
        "profanity_call_count": profanity_count,
        "issue_type_counts": dict(issue_type_counts),
    }


###############################################################
# RL FLOW FEEDBACK ENDPOINT
###############################################################

@app.post("/api/flow-feedback")
async def api_flow_feedback(req: FlowFeedbackRequest):
    try:
        intent = req.intent or "unknown"
        flow_policy.update(intent=intent, action=req.chosen_action, reward=req.reward)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update flow policy: {e}")


###############################################################
# ROOT (Optional Health Check)
###############################################################

@app.get("/")
def root():
    return {"status": "ok", "message": "IVR AI Backend running 🚀"}


###############################################################
# ENTRYPOINT
###############################################################

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
