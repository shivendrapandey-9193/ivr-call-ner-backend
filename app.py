# app.py
"""
IVR NER Analyzer - Improved single-file FastAPI backend
Features:
 - FastAPI endpoints for text/audio analysis
 - SpaCy NER (en_core_web_sm)
 - Optional BERT NER (transformers) if ENABLE_BERT=1
 - Groq Whisper transcription (requires GROQ_API_KEY)
 - Rule-based enhancements (ACCOUNT_ID, ISSUE_TYPE)
 - Hindi-aware postprocessing
 - SQLite persistence via SQLAlchemy
"""

import os
import re
import json
import tempfile
import random
import logging
from datetime import datetime
from typing import List, Optional, Dict, Union
from collections import defaultdict, Counter

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# Optional / third-party imports that should be in requirements.txt
try:
    import spacy
except Exception as e:
    spacy = None

try:
    from spellchecker import SpellChecker
except Exception:
    SpellChecker = None

try:
    from langdetect import detect, LangDetectException
except Exception:
    detect = None
    LangDetectException = Exception

# transformers are optional; load if env says so
ENABLE_BERT = os.getenv("ENABLE_BERT", "0") in {"1", "true", "True"}
TRANSFORMERS_AVAILABLE = False
if ENABLE_BERT:
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        TRANSFORMERS_AVAILABLE = True
    except Exception:
        TRANSFORMERS_AVAILABLE = False

# Groq client optional
try:
    from groq import Groq
except Exception:
    Groq = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ivr-ner")

# Load environment
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ivr_ner.db")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

# SQLAlchemy setup
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
    input_type = Column(String(20))  # audio | text
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


# Pydantic schemas
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


# -------------------------
# Models: SpaCy + optional BERT
# -------------------------
nlp_spacy = None
nlp_bert = None

def try_load_spacy():
    global nlp_spacy
    if nlp_spacy is not None:
        return
    if spacy is None:
        logger.error("spaCy is not installed. Please install spacy and the language wheel (en_core_web_sm).")
        return
    try:
        # do NOT auto-download inside runtime on hosted platforms; expect model installed in build.
        nlp_spacy = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model: en_core_web_sm")
    except Exception as e:
        logger.error("Could not load spaCy model 'en_core_web_sm'. Install the model before starting. Error: %s", e)
        nlp_spacy = None


def try_load_bert():
    global nlp_bert
    if not TRANSFORMERS_AVAILABLE:
        logger.info("Transformers not enabled or not available. BERT disabled.")
        return
    try:
        BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "dslim/bert-base-NER")
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert_model = AutoModelForTokenClassification.from_pretrained(BERT_MODEL_NAME)
        nlp_bert = pipeline("ner", model=bert_model, tokenizer=tokenizer, aggregation_strategy="simple")
        logger.info("Loaded BERT NER pipeline: %s", BERT_MODEL_NAME)
    except Exception as e:
        logger.warning("Failed to init BERT pipeline: %s", e)
        nlp_bert = None


# call loads at startup
try_load_spacy()
if ENABLE_BERT:
    try_load_bert()

# -------------------------
# Groq client (lazy)
# -------------------------
_groq_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    if Groq is None:
        raise RuntimeError("groq python SDK not installed.")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured.")
    _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

# -------------------------
# Text cleaning + numbers
# -------------------------
# small Hindi/English number maps (extend as needed)
NUMBER_WORDS_EN = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9"
}
NUMBER_WORDS_HI = {
    "एक":"1","दो":"2","तीन":"3","चार":"4","पांच":"5","पाँच":"5","छह":"6","छः":"6",
    "सात":"7","आठ":"8","नौ":"9","शून्य":"0"
}

def _normalize_word_token(token: Optional[str]) -> str:
    if token is None:
        return ""
    return re.sub(r"[^\w\u0900-\u097F]", "", token.lower())

def spoken_numbers_to_digits(text: str) -> str:
    if not text:
        return ""
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        w_raw = words[i]
        w = _normalize_word_token(w_raw)
        if w in NUMBER_WORDS_EN:
            digits = NUMBER_WORDS_EN[w]
            i += 1
            while i < len(words) and _normalize_word_token(words[i]) in NUMBER_WORDS_EN:
                digits += NUMBER_WORDS_EN[_normalize_word_token(words[i])]
                i += 1
            result.append(digits)
            continue
        if w in NUMBER_WORDS_HI:
            digits = NUMBER_WORDS_HI[w]
            i += 1
            while i < len(words) and _normalize_word_token(words[i]) in NUMBER_WORDS_HI:
                digits += NUMBER_WORDS_HI[_normalize_word_token(words[i])]
                i += 1
            result.append(digits)
            continue
        result.append(w_raw)
        i += 1
    merged = " ".join(result)

    # Merge patterns like "12345 five" -> "123455"
    merged = re.sub(r"\b(\d{5,})\s+(zero|one|two|three|four|five|six|seven|eight|nine)\b",
                    lambda m: m.group(1) + NUMBER_WORDS_EN[m.group(2)], merged, flags=re.I)
    return merged

# Spell checker optional
spell = None
if SpellChecker is not None:
    try:
        spell = SpellChecker()
    except Exception:
        spell = None

FILLER_WORDS = {"uh","umm","hmm","like","basically","actually","you know","i mean","so yeah","okay so","well","right","listen"}
IVR_BOILERPLATE = [
    "this call may be recorded for quality purposes",
    "press 1 to continue",
    "please wait while we connect your call",
    "your call is important to us",
    "welcome to the interactive voice response system",
    "welcome to customer support",
]

def feature_engineer_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n"," ").replace("\n"," ")
    text = re.sub(r"\s+"," ", text).strip()
    temp = text.lower()
    for phrase in IVR_BOILERPLATE:
        temp = temp.replace(phrase, " ")
    for fw in FILLER_WORDS:
        temp = re.sub(rf"\b{re.escape(fw)}\b", " ", temp)
    temp = re.sub(r"\s+([.,!?])", r"\1", temp)
    temp = spoken_numbers_to_digits(temp)
    tokens = []
    for tok in temp.split():
        # keep short tokens, numbers, currency or Devanagari tokens untouched
        if re.fullmatch(r"\d{1,12}", tok) or re.search(r"[₹$€£]|,\d{3}", tok) or len(tok) < 3 or re.search(r"[\u0900-\u097F]", tok):
            tokens.append(tok)
            continue
        if spell:
            try:
                corrected = spell.correction(tok) or tok
            except Exception:
                corrected = tok
            tokens.append(corrected)
        else:
            tokens.append(tok)
    cleaned = " ".join(tokens)
    # sentence casing for english spans
    sentences = [s.strip() for s in re.split(r"[.!?]", cleaned) if s.strip()]
    out_sents = []
    for s in sentences:
        if re.search(r"[A-Za-z]", s):
            out_sents.append(s[0].upper() + s[1:])
        else:
            out_sents.append(s)
    return ". ".join(out_sents).strip()

# -------------------------
# NER pipeline helpers
# -------------------------
def run_spacy_ner(text: str) -> List[Dict]:
    if nlp_spacy is None:
        return []
    doc = nlp_spacy(text)
    ents = []
    for ent in doc.ents:
        ents.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "source": "spacy"
        })
    return ents

def _normalize_bert_label(label: str) -> str:
    mapping = {"PER":"PERSON","ORG":"ORG","LOC":"LOC","MISC":"MISC"}
    return mapping.get(label, label)

def run_bert_ner(text: str) -> List[Dict]:
    if not TRANSFORMERS_AVAILABLE or nlp_bert is None:
        return []
    try:
        results = nlp_bert(text)
    except Exception as e:
        logger.warning("BERT NER failed: %s", e)
        return []
    ents = []
    for r in results:
        ents.append({
            "text": r.get("word",""),
            "label": _normalize_bert_label(r.get("entity_group","")),
            "start": int(r.get("start",0)),
            "end": int(r.get("end",0)),
            "source": "bert"
        })
    return ents

def merge_entities(spacy_entities: List[Dict], bert_entities: List[Dict]) -> List[Dict]:
    merged = {}
    # use (start,end,label) as key but prefer longer spans
    for e in spacy_entities + bert_entities:
        key = (e.get("start",0), e.get("end",0), e.get("label",""))
        if key in merged:
            # keep the longest text span (defensive)
            existing = merged[key]
            if len(e.get("text","")) > len(existing.get("text","")):
                merged[key] = e
            else:
                # merge sources
                existing["source"] = "merged"
        else:
            merged[key] = e.copy()
    return list(merged.values())

# rule-based entity upgrades
def add_rule_based_entities(text: str, entities: List[Dict]) -> List[Dict]:
    new = [e.copy() for e in entities]
    # upgrade 8-12 digit numbers to ACCOUNT_ID
    for e in new:
        if e.get("label") in {"CARDINAL", "DATE"}:
            if re.fullmatch(r"\d{8,12}", e.get("text","").strip()):
                e["label"] = "ACCOUNT_ID"
                e["source"] = "rule"
    # find any 8-12 digit in text
    for m in re.finditer(r"\b\d{8,12}\b", text):
        s, eidx = m.start(), m.end()
        span = m.group(0)
        found = False
        for ent in new:
            if ent["start"] == s and ent["end"] == eidx:
                ent["label"] = "ACCOUNT_ID"
                ent["source"] = "rule"
                found = True
                break
        if not found:
            new.append({"text":span,"label":"ACCOUNT_ID","start":s,"end":eidx,"source":"rule"})
    # issue phrases (english + some hindi)
    issue_keywords = [
        "payment failure","payment failed","failed payment","payment error","double payment",
        "refund","transaction failed","wrong deduction","charged twice","money deducted",
        "भुगतान में समस्या","पैसा दो बार कट गया","रिफंड चाहिए","payment 2 बार"
    ]
    lowered = text.lower()
    for kw in issue_keywords:
        idx = lowered.find(kw)
        while idx != -1:
            s = idx
            eidx = idx + len(kw)
            new.append({"text":text[s:eidx],"label":"ISSUE_TYPE","start":s,"end":eidx,"source":"rule"})
            idx = lowered.find(kw, eidx)
    # dedupe closely overlapping entities (keep rule-labeled ones)
    def overlap(a,b):
        return not (a["end"] <= b["start"] or b["end"] <= a["start"])
    compressed = []
    for ent in sorted(new, key=lambda x:(x["start"], - (x["end"]-x["start"]))):
        keep = True
        for existing in compressed:
            if overlap(ent, existing) and ent["label"] == existing["label"]:
                # prefer rule or longer
                if existing.get("source") == "rule":
                    keep = False
                    break
                if ent.get("source") == "rule" or (ent["end"]-ent["start"]) > (existing["end"]-existing["start"]):
                    # replace
                    compressed.remove(existing)
                    break
        if keep:
            compressed.append(ent)
    return compressed

# Hindi-specific postprocessing to remove spacy noise on Devanagari-only spans
def is_hindi_text(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text))

def _postprocess_ner_for_hindi(text: str, entities: List[Dict]) -> List[Dict]:
    if not is_hindi_text(text):
        return entities
    cleaned = []
    for e in entities:
        span = e.get("text","")
        label = e.get("label","")
        src = e.get("source","")
        # drop SpaCy-labeled PRODUCT/ORG/NORP if purely Hindi and no digits
        if src == "spacy" and label in {"PRODUCT","ORG","NORP","CARDINAL"}:
            if is_hindi_text(span) and not re.search(r"\d", span):
                continue
        cleaned.append(e)
    return cleaned

def _is_bad_date_entity(text: str) -> bool:
    t = text.strip().lower()
    if re.fullmatch(r"\d{8,12}", t):
        return True
    if t in {"id is", "account id", "acc id"}:
        return True
    return False

def analyze_text_ner(text: str) -> List[Dict]:
    try:
        spacy_ents = run_spacy_ner(text)
    except Exception as e:
        logger.warning("SpaCy NER failed: %s", e)
        spacy_ents = []
    bert_ents = run_bert_ner(text) if TRANSFORMERS_AVAILABLE else []
    merged = merge_entities(spacy_ents, bert_ents)
    filtered = []
    for e in merged:
        if e.get("label") == "DATE" and _is_bad_date_entity(e.get("text","")):
            continue
        filtered.append(e)
    final = add_rule_based_entities(text, filtered)
    clean = []
    for e in final:
        if e.get("label") == "DATE" and _is_bad_date_entity(e.get("text","")):
            continue
        # avoid small spelled-out numbers being mislabeled
        if e.get("label") == "CARDINAL" and _normalize_word_token(e.get("text","")) in NUMBER_WORDS_EN:
            continue
        clean.append(e)
    clean = _postprocess_ner_for_hindi(text, clean)
    # sort by start position
    clean = sorted(clean, key=lambda x: x.get("start",0))
    return clean

# -------------------------
# Relationships
# -------------------------
def extract_relationships(text: str, ents: List[Dict]) -> List[Dict]:
    relationships = []
    subj = "Customer"
    issues = [e for e in ents if e.get("label") == "ISSUE_TYPE"]
    accounts = [e for e in ents if e.get("label") == "ACCOUNT_ID"]
    dates = [e for e in ents if e.get("label") in ("DATE","TIME")]
    for issue in issues:
        relationships.append({"subject":subj,"predicate":"reports","object":issue["text"]})
    for acc in accounts:
        relationships.append({"subject":subj,"predicate":"has_account","object":acc["text"]})
    if dates:
        relationships.append({"subject":subj,"predicate":"called_on","object":dates[0]["text"]})
    return relationships

# -------------------------
# Language detection
# -------------------------
def detect_language_text(text: str) -> Optional[str]:
    if not text:
        return None
    if detect is None:
        # fall back to Devanagari check
        return "hi" if is_hindi_text(text) else "en"
    try:
        return detect(text)
    except Exception:
        return "hi" if is_hindi_text(text) else "en"

# -------------------------
# Sentiment / Emotion (lexicon-ish)
# -------------------------
POSITIVE_WORDS = {"good","great","awesome","excellent","happy","satisfied","resolved","thank","thanks","helpful","love","wonderful","nice","धन्यवाद","शुक्रिया","अच्छा","सहायता","मदद"}
NEGATIVE_WORDS = {"bad","terrible","horrible","angry","upset","sad","frustrated","annoyed","disappointed","issue","problem","complaint","escalate","समस्या","दिक्कत","नाराज","गुस्सा"}
EMOTION_LEXICON = {
    "anger":{"angry","furious","mad","annoyed","irritated","frustrated","गुस्सा","नाराज"},
    "joy":{"happy","glad","satisfied","good","खुश"},
}

def _analyze_sentiment_emotion_generic(text: str) -> SentimentEmotion:
    tokens = re.findall(r"\w+|\w+[\u0900-\u097F]+", text.lower())
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    total = pos + neg
    score = 0.0 if total == 0 else (pos - neg) / float(total)
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
    emotion_counts = {k:0 for k in EMOTION_LEXICON}
    for tok in tokens:
        for emo, words in EMOTION_LEXICON.items():
            if tok in words:
                emotion_counts[emo] += 1
    total_emo = sum(emotion_counts.values())
    if total_emo == 0:
        primary = None
        emo_dist = {}
    else:
        primary = max(emotion_counts.items(), key=lambda x:x[1])[0]
        emo_dist = {k: v/total_emo for k,v in emotion_counts.items() if v>0}
    return SentimentEmotion(sentiment_label=label, sentiment_score=score, primary_emotion=primary, emotions=emo_dist)

def _analyze_hindi_sentiment(text: str) -> SentimentEmotion:
    lower = text.lower()
    pos_keys = ["धन्यवाद","शुक्रिया","आभारी","सहायता","मदद"]
    neg_keys = ["गुस्सा","नाराज","परेशान","शिकायत","समस्या"]
    has_pos = any(k in lower for k in pos_keys)
    has_neg = any(k in lower for k in neg_keys)
    if has_pos and not has_neg:
        return SentimentEmotion(sentiment_label="positive", sentiment_score=0.5, primary_emotion="gratitude", emotions={"gratitude":1.0})
    if has_neg and not has_pos:
        return SentimentEmotion(sentiment_label="negative", sentiment_score=-0.5, primary_emotion="anger", emotions={"anger":1.0})
    if has_pos and has_neg:
        return SentimentEmotion(sentiment_label="neutral", sentiment_score=0.0, primary_emotion=None, emotions={})
    return SentimentEmotion(sentiment_label="neutral", sentiment_score=0.0, primary_emotion=None, emotions={})

def analyze_sentiment_emotion(text: str, language: Optional[str]=None) -> SentimentEmotion:
    generic = _analyze_sentiment_emotion_generic(text)
    if language == "hi" or is_hindi_text(text):
        hi = _analyze_hindi_sentiment(text)
        if hi.sentiment_label != "neutral" or hi.sentiment_score != 0.0:
            if not hi.emotions and generic.emotions:
                hi.emotions = generic.emotions
            if hi.primary_emotion is None:
                hi.primary_emotion = generic.primary_emotion
            return hi
    return generic

# -------------------------
# Intent detection (rule-based)
# -------------------------
INTENT_KEYWORDS = {
    "payment_issue":[ "payment failure","payment failed","refund","double payment","failed payment","transaction failed","payment error","पैसा","रिफंड","पैसा दो बार" ],
    "login_issue":[ "login issue","password reset","cannot login","account blocked"],
    "balance_inquiry":[ "balance","account balance","बैलेंस" ],
    "card_lost":[ "lost card","stolen card","block my card" ],
    "general_complaint":[ "complaint","not happy","bad service","escalate","शिकायत" ],
    "greeting_smalltalk":[ "hello","hi","thank you","नमस्ते","नमस्कार" ],
}

def detect_intents(text: str, ents: Optional[List[Dict]] = None) -> IntentDetection:
    if ents is None: ents = []
    lowered = text.lower()
    scores = defaultdict(float)
    for intent,kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in lowered:
                scores[intent] += 1.0
    for e in ents:
        if e.get("label") == "ISSUE_TYPE":
            scores["payment_issue"] += 0.5
            scores["general_complaint"] += 0.5
        if e.get("label") == "ACCOUNT_ID":
            scores["balance_inquiry"] += 0.2
    if not scores:
        return IntentDetection(primary_intent="unknown", confidence=0.0, intents={})
    max_score = max(scores.values())
    norm = {k: v/max_score for k,v in scores.items()}
    primary = max(norm.items(), key=lambda x:x[1])[0]
    conf = norm[primary]
    # prefer payment_issue if ISSUE_TYPE present
    if any(e.get("label")=="ISSUE_TYPE" for e in ents) and "payment_issue" in norm:
        primary = "payment_issue"
        conf = norm["payment_issue"]
    return IntentDetection(primary_intent=primary, confidence=conf, intents=norm)

# -------------------------
# Threat & profanity detection
# -------------------------
PROFANITY_WORDS = {"shit","damn","bloody","bastard","idiot","stupid"}
THREAT_WORDS = {"kill","hurt","attack","lawsuit","police complaint","file a case"}

def detect_threats_profanity(text: str) -> ThreatProfanity:
    lowered = text.lower()
    tokens = set(re.findall(r"\w+|\w+[\u0900-\u097F]+", lowered))
    prof = sorted({w for w in tokens if w in PROFANITY_WORDS})
    threats = []
    for phrase in THREAT_WORDS:
        if " " in phrase:
            if phrase in lowered:
                threats.append(phrase)
        else:
            if phrase in tokens:
                threats.append(phrase)
    return ThreatProfanity(threat_detected=bool(threats), profanity_detected=bool(prof), threat_terms=threats, profanity_terms=prof)

# -------------------------
# Compliance checker
# -------------------------
def check_compliance(text: str) -> ComplianceCheck:
    lowered = text.lower()
    greeting_ok = any(g in lowered for g in ["hello","hi","good morning","thank you","नमस्ते"])
    id_ok = bool(re.search(r"(account number|customer id|registered mobile|date of birth|अकाउंट नंबर|account no|acc no)", lowered))
    disclosure_ok = any(d in lowered for d in ["this call may be recorded","recorded for quality","कॉल" ]) or ("कॉल" in text and "रिकॉर्ड" in text)
    closing_ok = bool(re.search(r"(thank you for calling|have a nice day|धन्यवाद|thank you)", lowered))
    warnings = []
    passed = 0
    total = 4
    if greeting_ok: passed += 1
    else: warnings.append("Missing proper greeting.")
    if id_ok: passed += 1
    else: warnings.append("Customer identity not clearly verified.")
    if disclosure_ok: passed += 1
    else: warnings.append("Mandatory disclosure about recording/terms is missing.")
    if closing_ok: passed += 1
    else: warnings.append("No proper closing phrase / thanks.")
    score = passed/float(total)
    return ComplianceCheck(overall_score=score, passed=score>=0.75, warnings=warnings)

# -------------------------
# Summary / agent assist / scoring
# -------------------------
def generate_call_summary(text: str, ents: List[Dict], intents: IntentDetection, sentiment: SentimentEmotion, language: Optional[str]) -> str:
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    short = sentences[:3]
    issue_types = [e["text"] for e in ents if e.get("label")=="ISSUE_TYPE"]
    accounts = [e["text"] for e in ents if e.get("label")=="ACCOUNT_ID"]
    parts = []
    if language: parts.append(f"Language detected: {language}.")
    if intents.primary_intent != "unknown": parts.append(f"Primary intent: {intents.primary_intent.replace('_',' ')}.")
    parts.append(f"Sentiment: {sentiment.sentiment_label} (score={sentiment.sentiment_score:.2f}).")
    if issue_types: parts.append(f"Issue types: {', '.join(set(issue_types))}.")
    if accounts: parts.append(f"Account refs: {', '.join(set(accounts))}.")
    if short: parts.append("Call context: " + " ".join(short))
    return " ".join(parts)

class FlowPolicy:
    def __init__(self, epsilon=0.15):
        self.epsilon = epsilon
        self.stats = defaultdict(lambda: defaultdict(lambda: {"success":0.0,"count":0.0}))
    def _candidate_actions(self, intent):
        if intent=="payment_issue":
            return ["Ask for last 4 digits of account/card","Verify recent transaction details","Offer to check payment gateway logs"]
        if intent=="login_issue":
            return ["Guide password reset flow","Verify registered email or mobile"]
        if intent=="balance_inquiry":
            return ["Authenticate and read current balance","Offer SMS of current balance"]
        if intent=="card_lost":
            return ["Authenticate and block card immediately","Offer to reissue card"]
        if intent=="general_complaint":
            return ["Acknowledge and empathize","Offer escalation to supervisor"]
        return ["Ask clarifying question","Transfer to human agent"]
    def choose_action(self, intent):
        actions = self._candidate_actions(intent)
        state = intent or "unknown"
        for a in actions: _=self.stats[state][a]
        if random.random() < self.epsilon:
            return random.choice(actions)
        best = None
        best_score = -1e9
        for a in actions:
            d = self.stats[state][a]
            avg = d["success"]/d["count"] if d["count"]>0 else 0.0
            if avg > best_score:
                best_score = avg
                best = a
        return best or random.choice(actions)
    def update(self,intent,action,reward):
        state = intent or "unknown"
        d = self.stats[state][action]
        d["count"] += 1.0
        d["success"] += float(reward)

flow_policy = FlowPolicy()

def build_agent_assist(text: str, ents: List[Dict], intents: IntentDetection, sentiment: SentimentEmotion, compliance: ComplianceCheck, risk: ThreatProfanity) -> AgentAssist:
    suggestions = []
    intent = intents.primary_intent
    if intent == "payment_issue":
        suggestions += ["Confirm transaction date, amount, and mode of payment.","Check for duplicate or pending transactions."]
    elif intent == "login_issue":
        suggestions += ["Guide the customer through secure password reset.","Confirm username / registered mobile or email."]
    elif intent == "balance_inquiry":
        suggestions += ["Authenticate customer and share current balance."]
    elif intent == "card_lost":
        suggestions += ["Immediately block the card and confirm last known transactions."]
    elif intent == "general_complaint":
        suggestions += ["Acknowledge the issue and apologize for the inconvenience.","Offer escalation if necessary."]
    else:
        suggestions += ["Ask a clarifying question to better understand the issue."]
    if sentiment.sentiment_label in {"negative","very_negative"}:
        suggestions.append("Use empathetic phrases and reassure the customer.")
    elif sentiment.sentiment_label in {"positive","very_positive"}:
        suggestions.append("Maintain positive tone and confirm if anything else is needed.")
    for w in compliance.warnings:
        suggestions.append(f"Compliance gap: {w}")
    if risk.threat_detected:
        suggestions.append("Consider escalating due to threat/legal language.")
    if risk.profanity_detected:
        suggestions.append("Maintain calm tone; avoid mirroring customer's language.")
    call_flow_action = flow_policy.choose_action(intent)
    nba = suggestions[0] if suggestions else call_flow_action
    return AgentAssist(suggestions=list(dict.fromkeys(suggestions)), next_best_action=nba, call_flow_action=call_flow_action)

def compute_call_score(intents: IntentDetection, sentiment: SentimentEmotion, compliance: ComplianceCheck, risk: ThreatProfanity) -> int:
    intent_score = intents.confidence * 30
    sentiment_scaled = (1 + sentiment.sentiment_score) / 2
    sentiment_score = sentiment_scaled * 20
    compliance_score = compliance.overall_score * 40
    penalty = 0
    if risk.threat_detected: penalty -= 40
    if risk.profanity_detected: penalty -= 25
    if intents.primary_intent != "unknown" and compliance.overall_score >= 0.25:
        penalty *= 0.6
    final_score = intent_score + sentiment_score + compliance_score + penalty
    final_score = max(0, min(100, round(final_score)))
    return final_score

# -------------------------
# Transcription wrapper (Groq Whisper)
# -------------------------
ALLOWED_AUDIO_EXT = {".wav",".mp3",".m4a",".flac",".ogg",".mp4",".webm",".mpeg",".mpga"}
MAX_AUDIO_MB = int(os.getenv("MAX_AUDIO_MB", "25"))

def validate_audio(filename: str, size: int):
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO_EXT:
        raise ValueError(f"Unsupported audio format: {ext}. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXT))}")
    if size > MAX_AUDIO_MB * 1024 * 1024:
        raise ValueError(f"File too large (> {MAX_AUDIO_MB} MB).")

def transcribe_audio(file_bytes: bytes, filename: str) -> Dict[str, Optional[Union[str,float]]]:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured on server.")
    client = get_groq_client()
    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        with open(tmp.name, "rb") as audio:
            resp = client.audio.transcriptions.create(model="whisper-large-v3", file=audio, response_format="verbose_json")
    if isinstance(resp, dict):
        text = resp.get("text","") or ""
        language = resp.get("language")
        duration = resp.get("duration")
    else:
        text = getattr(resp,"text","") or ""
        language = getattr(resp,"language",None)
        duration = getattr(resp,"duration",None)
    return {"transcript": text, "language": language, "duration": duration}

# -------------------------
# FastAPI app & endpoints
# -------------------------
app = FastAPI(title="IVR NER Analyzer (improved)", version="2.5.0")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup_event():
    try_load_spacy()
    if ENABLE_BERT:
        try_load_bert()
    logger.info("App startup complete. spaCy loaded=%s, BERT enabled=%s", nlp_spacy is not None, TRANSFORMERS_AVAILABLE)

@app.get("/", tags=["health"])
def root(request: Request = None):
    return {"status":"ok","message":"IVR AI Backend running 🚀"}

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
        cleaned_transcript = feature_engineer_text(result.get("transcript","") or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text cleaning failed: {e}")
    if not cleaned_transcript:
        raise HTTPException(status_code=500, detail="Transcription returned empty or unusable text.")
    return TranscribeAudioResponse(transcript=cleaned_transcript, language=result.get("language"), duration=result.get("duration"))

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
        record = CallAnalysis(input_type="text", transcript=cleaned_text, entities_json=json.dumps(ents, ensure_ascii=False), relationships_json=json.dumps(rels, ensure_ascii=False))
        db.add(record)
        db.commit()
    except Exception as e:
        logger.warning("DB save failed: %s", e)
    return AnalyzeTextResponse(text=cleaned_text, language=language, entities=ents, relationships=rels, sentiment=sentiment, intents=intents, summary=summary, agent_assist=agent_assist, compliance=compliance, risk_flags=risk, call_score=score)

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
        cleaned_transcript = feature_engineer_text(t.get("transcript","") or "")
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
        record = CallAnalysis(input_type="audio", transcript=cleaned_transcript, entities_json=json.dumps(ents, ensure_ascii=False), relationships_json=json.dumps(rels, ensure_ascii=False))
        db.add(record)
        db.commit()
    except Exception as e:
        logger.warning("DB save failed: %s", e)
    return AnalyzeAudioResponse(transcript=cleaned_transcript, language=language, duration=t.get("duration"), entities=ents, relationships=rels, sentiment=sentiment, intents=intents, summary=summary, agent_assist=agent_assist, compliance=compliance, risk_flags=risk, call_score=score)

@app.get("/api/history", response_model=List[Dict])
async def api_history(limit: int = 50, db: Session = Depends(get_db)):
    limit = max(1, min(200, limit))
    rows = db.query(CallAnalysis).order_by(CallAnalysis.created_at.desc()).limit(limit).all()
    out = []
    for r in rows:
        ents = json.loads(r.entities_json or "[]")
        rels = json.loads(r.relationships_json or "[]")
        out.append({"id": r.id, "created_at": r.created_at.isoformat(), "input_type": r.input_type, "transcript": r.transcript, "entities": ents, "relationships": rels})
    return out

@app.get("/api/analytics-dashboard")
async def api_analytics_dashboard(limit: int = 200, db: Session = Depends(get_db)):
    limit = max(1, min(1000, limit))
    rows = db.query(CallAnalysis).order_by(CallAnalysis.created_at.desc()).limit(limit).all()
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
        if lang: by_language[lang] += 1
        sentiment = analyze_sentiment_emotion(transcript, lang)
        sentiment_scores.append(sentiment.sentiment_score)
        intents = detect_intents(transcript, ents)
        by_intent[intents.primary_intent] += 1
        risk = detect_threats_profanity(transcript)
        if risk.threat_detected: threat_count += 1
        if risk.profanity_detected: profanity_count += 1
        for e in ents:
            if e.get("label") == "ISSUE_TYPE":
                issue_type_counts[e["text"].lower()] += 1
    avg_sentiment = sum(sentiment_scores)/len(sentiment_scores) if sentiment_scores else 0.0
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

@app.post("/api/flow-feedback")
async def api_flow_feedback(payload: Dict):
    try:
        intent = payload.get("intent") or "unknown"
        action = payload["chosen_action"]
        reward = float(payload.get("reward", 0.0))
        flow_policy.update(intent=intent, action=action, reward=reward)
        return {"status":"ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update flow policy: {e}")

# Entrypoint for local run
if __name__ == "__main__":
    import uvicorn
    # host/port can be configured with env vars
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=True)
