"""
IVR NER ANALYZER - PRODUCTION READY V6.0
FIXED ALL CRITICAL NER ACCURACY ISSUES + IMPROVED STT
"""
import os
import json
import re
import tempfile
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Database
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# Environment
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CONFIGURATION

class Config:
    """Configuration settings"""
    MAX_AUDIO_SIZE_MB = 10
    MAX_TEXT_LENGTH = 5000
    ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a'}
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ivr_ner.db")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PORT = int(os.getenv("PORT", 8000))
  
    # NER confidence thresholds
    MIN_CONFIDENCE = 0.4
    ENTITY_PRIORITIES = {
        'TRANSACTION_ID': 100,
        'ACCOUNT_NUMBER': 95,
        'PHONE_NUMBER': 90,
        'CARD_NUMBER': 85,
        'AADHAAR': 80,
        'PAN': 75,
        'IFSC': 70,
        'AMOUNT': 65,
        'ISSUE_TYPE': 90,
        'DATE': 60,
        'TIME': 55,
        'EMAIL': 50,
        'PERSON': 40,
        'ORGANIZATION': 35,
        'LOCATION': 30,
        'OTP': 25,
    }

# LAZY LOADING FUNCTIONS

class LazyLoader:
    """Lazy load heavy dependencies"""
  
    _spacy_model = None
    _groq_client = None
    _groq_llm_client = None
  
    @staticmethod
    def get_spacy_model():
        """Lazy load spaCy model"""
        if LazyLoader._spacy_model is None:
            try:
                import spacy
                logger.info("Loading spaCy model...")
                LazyLoader._spacy_model = spacy.load(
                    "en_core_web_sm",
                    disable=["parser", "tagger", "lemmatizer", "attribute_ruler"]
                )
                LazyLoader._spacy_model.max_length = 200000
                logger.info("spaCy model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load spaCy model: {e}")
                raise
        return LazyLoader._spacy_model
  
    @staticmethod
    def get_groq_client():
        """Lazy load Groq client for Whisper"""
        if LazyLoader._groq_client is None:
            try:
                from groq import Groq
                api_key = Config.GROQ_API_KEY
                if not api_key:
                    raise RuntimeError("GROQ_API_KEY not set")
                LazyLoader._groq_client = Groq(api_key=api_key)
                logger.info("Groq Whisper client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                raise
        return LazyLoader._groq_client
  
    @staticmethod
    def get_groq_llm_client():
        """Lazy load Groq client for LLM"""
        if LazyLoader._groq_llm_client is None:
            try:
                from groq import Groq
                api_key = Config.GROQ_API_KEY
                if not api_key:
                    return None
                LazyLoader._groq_llm_client = Groq(api_key=api_key)
                logger.info("Groq LLM client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq LLM client: {e}")
                LazyLoader._groq_llm_client = None
        return LazyLoader._groq_llm_client

# DATABASE SETUP

Base = declarative_base()

class CallAnalysis(Base):
    __tablename__ = "call_analysis"
  
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    input_type = Column(String(20))
    transcript = Column(Text)
    entities_json = Column(Text)
    relationships_json = Column(Text)
    sentiment = Column(String(20))
    intent = Column(String(50))
    call_score = Column(Integer)
    language = Column(String(10))
    processing_time_ms = Column(Integer)

engine_kwargs = {
    "echo": False
}

if Config.DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(
    Config.DATABASE_URL,
    **engine_kwargs
)


Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# PYDANTIC MODELS

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    source: str = Field(default="rule")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

class Relationship(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

class SentimentEmotion(BaseModel):
    sentiment_label: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    primary_emotion: Optional[str] = None
    reasoning: Optional[str] = None

class IntentDetection(BaseModel):
    primary_intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    intents: Dict[str, float] = Field(default_factory=dict)

class ComplianceCheck(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    warnings: List[str] = Field(default_factory=list)
    note: Optional[str] = None

class ThreatProfanity(BaseModel):
    threat_detected: bool
    profanity_detected: bool
    threat_terms: List[str] = Field(default_factory=list)
    profanity_terms: List[str] = Field(default_factory=list)

class AgentAssist(BaseModel):
    suggestions: List[str] = Field(default_factory=list)
    next_best_action: Optional[str] = None
    call_flow_action: Optional[str] = None

class AnalyzeTextRequest(BaseModel):
    text: str

class TranscribeResponse(BaseModel):
    transcript: str
    language: Optional[str] = None
    duration: Optional[float] = None
    success: bool


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
    call_score: int = Field(ge=0, le=100)
    processing_time_ms: int

# TEXT PREPROCESSOR

class TextPreprocessor:
    """Text preprocessing"""

    IVR_PHRASES = [
        r"this call may be recorded.*",
        r"press \d+ to continue",
        r"please wait while we connect",
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        if not text or not text.strip():
            return ""

        for phrase in cls.IVR_PHRASES:
            text = re.sub(phrase, '', text, flags=re.IGNORECASE)

        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @classmethod
    def detect_language(cls, text: str) -> str:
        """Simple language detection"""
        if not text:
            return "en"

        if re.search(r'[\u0900-\u097F]', text):
            return "hi"

        return "en"

# FIXED NER ENGINE - ALL CRITICAL ISSUES RESOLVED

class NEREngine:
    """Fixed NER engine with all accuracy issues resolved"""
  
    # Phone number context patterns - COMPREHENSIVE
    PHONE_CONTEXT_PATTERNS = [
        r'my\s+mobile',
        r'mobile\s+number',
        r'registered\s+mobile',
        r'phone\s+number',
        r'contact\s+number',
        r'my\s+phone',
        r'my\s+contact',
        r'call\s+me\s+at',
        r'sms\s+me\s+at',
        r'whatsapp\s+number',
        r'alternate\s+number',
        r'emergency\s+contact',
        r'primary\s+phone',
        r'secondary\s+phone',
        r'contact\s+details',
        r'mobile\s+no\s*[\.:]',
        r'phone\s+no\s*[\.:]',
        r'contact\s+no\s*[\.:]',
    ]
  
    # Account number context patterns
    ACCOUNT_CONTEXT_PATTERNS = [
        r'account\s+number',
        r'acc\s+no',
        r'savings\s+account',
        r'current\s+account',
        r'bank\s+account',
        r'account\s+details',
        r'my\s+account',
        r'account\s+no\s*[\.:]',
        r'acc\s+no\s*[\.:]',
        r'account\s+id',
        r'customer\s+id',
    ]
  
    # Pattern configuration with ALL FIXES APPLIED
    PATTERNS = {
        'PHONE_NUMBER': {
            'patterns': [
                r'\b\d{10}\b',
                r'\b\+91[-\s]?\d{10}\b',
                r'\b\d{5}[-\s]?\d{5}\b',
                r'\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b',
            ],
            'base_confidence': 0.85,
            'context_patterns': PHONE_CONTEXT_PATTERNS,
            'validation': lambda x, txt: NEREngine._is_phone_number(x, txt),
        },
      
        'ACCOUNT_NUMBER': {
            'patterns': [
                r'\b\d{11,18}\b',  # FIXED: Changed from 9-18 to 11-18 (account numbers > 10 digits)
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4,10}\b',
                r'\b\d{3}[-\s]?\d{4}[-\s]?\d{4,7}\b',
            ],
            'base_confidence': 0.8,
            'context_patterns': ACCOUNT_CONTEXT_PATTERNS,
            'validation': lambda x, txt: NEREngine._is_account_number(x, txt),  # FIXED VALIDATION
        },
      
        'TRANSACTION_ID': {
            'patterns': [
                # FIXED: Transaction IDs must have numbers and be longer
                r'\b(?:txn|trxn|trans)[-_]?(?:id|ref)?[_\s-]?[A-Z0-9]{8,20}\b',
                r'\b(?:ref|reference)[-_]?(?:id|no)?[_\s-]?[A-Z0-9]{8,20}\b',
                r'\b(?:order|ord)[-_]?(?:id|no)?[_\s-]?[A-Z0-9]{8,20}\b',
                r'\b[A-Z]{2,4}\d{6,12}\b',  # Common pattern: AB12345678
                r'\b\d{8,15}[A-Z]{2,4}\b',  # Common pattern: 12345678ABC
            ],
            'base_confidence': 0.88,
            'context_patterns': ['transaction', 'txn', 'reference', 'ref', 'order', 'payment'],
            'validation': lambda x, txt: NEREngine._is_transaction_id(x, txt),  # FIXED VALIDATION
        },
      
        'CARD_NUMBER': {
            'patterns': [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                r'\b\d{16}\b',
                r'\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b',  # Some card patterns
            ],
            'base_confidence': 0.92,
            'context_patterns': ['card', 'credit', 'debit', 'visa', 'mastercard', 'rupay'],
            'validation': lambda x, txt: True,
        },
      
        'AMOUNT': {
            'patterns': [
                r'₹\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b',
                r'rs\.?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b',
                r'rupees?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b',
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s?(?:rs|rupees|₹)\b',
                r'\bINR\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b',
            ],
            'base_confidence': 0.9,
            'context_patterns': ['amount', 'rs', 'rupees', '₹', 'paid', 'charge', 'fee'],
            'validation': lambda x, txt: True,
        },
      
        'OTP': {
            'patterns': [
                r'\b\d{4,6}\s*(?:otp|pin|code)\b',
                r'\b(?:otp|pin|code)\s*\d{4,6}\b',
                r'\b\d{4,6}\b(?=\s*(?:for|is|otp|verification))',
            ],
            'base_confidence': 0.85,
            'context_patterns': ['otp', 'pin', 'verification', 'code'],
            'validation': lambda x, txt: len(''.join(filter(str.isdigit, x))) in [4, 6],
        },
      
        'AADHAAR': {
            'patterns': [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                r'\b\d{12}\b',
            ],
            'base_confidence': 0.8,
            'context_patterns': ['aadhaar', 'aadhar', 'uid', 'unique id'],
            'validation': lambda x, txt: True,
        },
      
        'PAN': {
            'patterns': [
                r'\b[A-Z]{5}\d{4}[A-Z]{1}\b',
            ],
            'base_confidence': 0.95,
            'context_patterns': ['pan', 'permanent account number'],
            'validation': lambda x, txt: True,
        },

        'EMAIL': {
            'patterns': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
            ],
            'base_confidence': 0.9,
            'context_patterns': ['email', 'mail'],
            'validation': lambda x, txt: True,
        },
      
        'IFSC': {
            'patterns': [
                r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
            ],
            'base_confidence': 0.95,
            'context_patterns': ['ifsc', 'code', 'bank code'],
            'validation': lambda x, txt: True,
        },

    }
  
    @staticmethod
    def _is_phone_number(entity_text: str, full_text: str) -> bool:
        """Phone number validation - FIXED: 10-digit only"""
        digits = ''.join(filter(str.isdigit, entity_text))
        
        # MUST be exactly 10 digits
        if len(digits) != 10:
            return False
      
        text_lower = full_text.lower()
        
        # Check for explicit phone context
        for pattern in NEREngine.PHONE_CONTEXT_PATTERNS:
            if re.search(pattern, text_lower):
                return True
      
        # Check for standalone indicators
        standalone_indicators = [
            r'\bphone\b',
            r'\bmobile\b',
            r'\bcontact\b',
            r'\bnumber\b',
            r'\bwhatsapp\b',
            r'\bcall\b',
            r'\bsms\b',
        ]
        
        for indicator in standalone_indicators:
            if re.search(indicator, text_lower):
                # Check if indicator is near the number
                window_start = max(0, full_text.lower().find(digits) - 50)
                window_end = min(len(full_text), full_text.lower().find(digits) + 50)
                window = full_text[window_start:window_end].lower()
                if re.search(indicator, window):
                    return True
      
        # Check for explicit ACCOUNT context - if present, it's NOT a phone
        for pattern in NEREngine.ACCOUNT_CONTEXT_PATTERNS:
            if re.search(pattern, text_lower):
                # Check if account context is near the number
                window_start = max(0, full_text.lower().find(digits) - 50)
                window_end = min(len(full_text), full_text.lower().find(digits) + 50)
                window = full_text[window_start:window_end].lower()
                if re.search(pattern, window):
                    return False
      
        # Default: 10-digit without clear context = phone
        return True
  
    @staticmethod
    def _is_account_number(entity_text: str, full_text: str) -> bool:
        """Account number validation - FIXED: >10 digits or explicit context"""
        digits = ''.join(filter(str.isdigit, entity_text))
        
        # FIXED CRITICAL ISSUE 1: Account numbers > 10 digits
        if len(digits) <= 10:
            # 10-digit or less numbers need STRONG account context
            text_lower = full_text.lower()
            
            # Check for EXPLICIT account context near the number
            window_start = max(0, full_text.lower().find(entity_text) - 50)
            window_end = min(len(full_text), full_text.lower().find(entity_text) + 50)
            window = full_text[window_start:window_end].lower()
            
            has_account_context = False
            for pattern in NEREngine.ACCOUNT_CONTEXT_PATTERNS:
                if re.search(pattern, window):
                    has_account_context = True
                    break
            
            # FIXED: If phone context exists, NEVER mark as account
            has_phone_context = False
            for pattern in NEREngine.PHONE_CONTEXT_PATTERNS:
                if re.search(pattern, window):
                    has_phone_context = True
                    break
            
            if has_phone_context:
                return False  # CRITICAL FIX
            
            return has_account_context
        else:
            # >10 digits = account number (high confidence)
            return True
  
    @staticmethod
    def _is_transaction_id(entity_text: str, full_text: str) -> bool:
        """Transaction ID validation - FIXED CRITICAL ISSUE 2"""
        # Must have minimum length
        if len(entity_text) < 8:
            return False
        
        # FIXED: Must contain digits
        digit_count = sum(c.isdigit() for c in entity_text)
        if digit_count < 4:  # At least 4 digits
            return False
        
        # Must contain at least some alphanumeric characters
        if not any(c.isalnum() for c in entity_text):
            return False
        
        # Check for common transaction ID patterns
        patterns = [
            r'^[A-Z]{2,}\d{6,}$',  # AB123456
            r'^\d{8,}[A-Z]{2,}$',  # 12345678AB
            r'^TXN\d{6,}$',        # TXN123456
            r'^REF\d{6,}$',        # REF123456
            r'^TR\d{8,}$',         # TR12345678
        ]
        
        for pattern in patterns:
            if re.match(pattern, entity_text, re.IGNORECASE):
                return True
        
        # Check context for transaction-related words
        text_lower = full_text.lower()
        transaction_context = any(
            word in text_lower 
            for word in ['transaction', 'txn', 'reference', 'ref', 'payment', 'order']
        )
        
        return transaction_context

  
    @classmethod
    def extract_entities(cls, text: str) -> List[Dict[str, Any]]:
        """Extract entities with all fixes applied"""
        if not text:
            return []
      
        text_lower = text.lower()
        all_entities = []
      
        # Extract rule-based entities in priority order
        priority_order = ['PHONE_NUMBER', 'ACCOUNT_NUMBER', 'TRANSACTION_ID', 'CARD_NUMBER', 
                         'AMOUNT', 'OTP', 'AADHAAR', 'PAN','EMAIL', 'IFSC']
        
        for label in priority_order:
            if label in cls.PATTERNS:
                config = cls.PATTERNS[label]
                for pattern in config['patterns']:
                    try:
                        for match in re.finditer(pattern, text, re.IGNORECASE):
                            entity_text = match.group()
                          
                            confidence = cls._calculate_entity_confidence(
                                config['base_confidence'],
                                entity_text,
                                label,
                                text,
                                text_lower,
                                config['context_patterns'],
                                config['validation']
                            )
                          
                            if confidence >= Config.MIN_CONFIDENCE:
                                entity = {
                                    'text': entity_text,
                                    'label': label,
                                    'start': match.start(),
                                    'end': match.end(),
                                    'source': 'rule',
                                    'confidence': confidence,
                                }
                                all_entities.append(entity)
                    except Exception as e:
                        logger.warning(f"Pattern error {label}: {e}")
                        continue
      
        # Extract spaCy entities with FIXED blacklists
        spacy_entities = cls._extract_spacy_entities(text)
        all_entities.extend(spacy_entities)
      
        # Resolve conflicts with improved logic
        resolved = cls._resolve_entity_conflicts(all_entities, text_lower)
      
        # Final confidence adjustment based on context
        resolved = cls._adjust_confidence_by_context(resolved, text_lower)
      
        # Sort by priority and confidence
        resolved.sort(
            key=lambda x: (
                Config.ENTITY_PRIORITIES.get(x['label'], 0),
                x['confidence']
            ),
            reverse=True
        )
      
        return resolved
  
    @classmethod
    def _calculate_entity_confidence(cls, base_confidence: float, entity_text: str,
                                   label: str, original_text: str, text_lower: str,
                                   context_patterns: List[str], validation_func) -> float:
        """Calculate confidence with context"""
      
        if not validation_func(entity_text, original_text):
            return 0.0
      
        confidence = min(base_confidence, 0.9)  # Cap at 0.9
      
        # Context adjustment
        context_score = cls._calculate_context_score(
            entity_text, label, original_text, text_lower, context_patterns
        )
        confidence += context_score * 0.2
      
        # Special adjustments
        if label == 'PHONE_NUMBER':
            digits = ''.join(filter(str.isdigit, entity_text))
            if len(digits) == 10:
                # Check for clear phone context
                phone_context = any(
                    re.search(pattern, text_lower) 
                    for pattern in cls.PHONE_CONTEXT_PATTERNS
                )
                if phone_context:
                    confidence += 0.1
          
        elif label == 'ACCOUNT_NUMBER':
            digits = ''.join(filter(str.isdigit, entity_text))
            if len(digits) > 10:
                confidence += 0.15  # Boost for >10 digit accounts
          
        elif label == 'TRANSACTION_ID':
            # Transaction IDs with common prefixes get confidence boost
            prefixes = ['txn', 'trxn', 'trans', 'ref', 'ord', 'tr']
            if any(entity_text.lower().startswith(prefix) for prefix in prefixes):
                confidence = min(0.9, confidence + 0.05)
      
        return max(Config.MIN_CONFIDENCE, min(0.9, round(confidence, 2)))
  
    @classmethod
    def _calculate_context_score(cls, entity_text: str, label: str, original_text: str,
                               text_lower: str, context_patterns: List[str]) -> float:
        """Calculate context score"""
        # Find first occurrence
        occurrences = []
        escaped_text = re.escape(entity_text)
        for match in re.finditer(escaped_text, original_text, re.IGNORECASE):
            occurrences.append((match.start(), match.end()))
      
        if not occurrences:
            return 0.0
      
        start, end = occurrences[0]
      
        # Get context window (100 characters before and after)
        window_start = max(0, start - 100)
        window_end = min(len(original_text), end + 100)
        context_window = text_lower[window_start:window_end]
      
        # Check for context patterns
        context_matches = 0
        for pattern in context_patterns:
            if re.search(pattern, context_window):
                context_matches += 1
      
        # Normalize score
        return min(1.0, context_matches * 0.2)
  
    @classmethod
    def _extract_spacy_entities(cls, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy - FIXED CRITICAL ISSUE 3"""
        try:
            # 🚫 STOP spaCy for Hindi / Hinglish
            if (
                re.search(r'[\u0900-\u097F]', text) or
                re.search(
                    r'\b(hai|ho|gaya|tha|thi|mera|meri|mere|aur|lekin|paisa|kat|cut|kar|karo|wala|wali)\b',
                    text.lower()
                )
            ):
                return []
        
            nlp = LazyLoader.get_spacy_model()
            doc = nlp(text[:1000])
        
            entities = []
            label_mapping = {
                'PERSON': 'PERSON',
                'ORG': 'ORGANIZATION',
                'GPE': 'LOCATION',
                'LOC': 'LOCATION',
                'DATE': 'DATE',
                'TIME': 'TIME',
                'MONEY': 'AMOUNT',
                'CARDINAL': 'NUMBER',
            }
        
            # FIXED: Blacklists for spaCy false positives
            person_blacklist = [
                'transaction', 'payment', 'account', 'card',
                'id', 'upi', 'refund', 'balance', 'loan',
                'amount', 'charge', 'fee', 'interest', 'otp',
                'emi', 'customer', 'service', 'support',
            ]
            
            org_blacklist = [  # FIXED CRITICAL ISSUE 3
                'otp', 'upi', 'atm', 'ivr', 'sms', 'call',
    'bank', 'branch', 'center', 'customer',
    'dsa', 'data structures', 'algorithms',
    'full stack', 'development', 'course', 'program'
            ]
        
            for ent in doc.ents:
                mapped_label = label_mapping.get(ent.label_, ent.label_)
                ent_lower = ent.text.lower()
            
                # FIXED: Filter false PERSON entities
                if mapped_label == 'PERSON':
                    if any(word in ent_lower for word in person_blacklist):
                        continue
                    if len(ent.text.split()) == 1 and len(ent.text) < 4:
                        continue
                    if re.search(r'\b(txn|trxn|trans|ref|id|otp)\b', ent_lower):
                        continue
            
                # FIXED: Filter false ORGANIZATION entities
                if mapped_label == 'ORGANIZATION':
                    if ent_lower in org_blacklist:
                        continue
                    if len(ent.text) < 3:
                        continue
            
                # Filter other obvious false positives
                if mapped_label == 'CARDINAL':
                    # Check if it's actually an entity we care about
                    digits = ''.join(filter(str.isdigit, ent.text))
                    if len(digits) >= 4:
                        # Could be account/phone/etc - let rule engine handle
                        continue
            
                entities.append({
                    'text': ent.text,
                    'label': mapped_label,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'source': 'spacy',
                    'confidence': 0.6,
                })
        
            return entities
        
        except Exception as e:
            logger.error(f"spaCy NER failed: {e}")
            return []
  
    @classmethod
    def _resolve_entity_conflicts(cls, entities: List[Dict], text_lower: str) -> List[Dict]:
        """Resolve overlapping entity conflicts with improved logic"""
        if not entities:
            return []
      
        entities.sort(key=lambda x: x['start'])
        resolved = []
      
        for entity in entities:
            if not resolved:
                resolved.append(entity)
                continue
            
            last_entity = resolved[-1]
            
            # Check for overlap
            if entity['start'] < last_entity['end']:
                # Conflict - choose better entity
                last_priority = Config.ENTITY_PRIORITIES.get(last_entity['label'], 0)
                entity_priority = Config.ENTITY_PRIORITIES.get(entity['label'], 0)
                
                # FIXED: Special handling for phone vs account
                if (last_entity['label'] == 'PHONE_NUMBER' and entity['label'] == 'ACCOUNT_NUMBER' or
                    entity['label'] == 'PHONE_NUMBER' and last_entity['label'] == 'ACCOUNT_NUMBER'):
                    
                    digits_last = ''.join(filter(str.isdigit, last_entity['text']))
                    digits_entity = ''.join(filter(str.isdigit, entity['text']))
                    
                    # Both are 10-digit - check context
                    if len(digits_last) == 10 and len(digits_entity) == 10:
                        # Get context window
                        start_pos = min(last_entity['start'], entity['start'])
                        end_pos = max(last_entity['end'], entity['end'])
                        window_start = max(0, start_pos - 30)
                        window_end = min(len(text_lower), end_pos + 30)
                        context = text_lower[window_start:window_end]
                        
                        # Check for phone context
                        phone_context = False
                        for pattern in cls.PHONE_CONTEXT_PATTERNS:
                            if re.search(pattern, context):
                                phone_context = True
                                break
                        
                        # Check for account context
                        account_context = False
                        for pattern in cls.ACCOUNT_CONTEXT_PATTERNS:
                            if re.search(pattern, context):
                                account_context = True
                                break
                        
                        # Decision logic
                        if phone_context and not account_context:
                            # Keep phone
                            if last_entity['label'] == 'PHONE_NUMBER':
                                resolved[-1] = last_entity
                            else:
                                resolved[-1] = entity
                        elif account_context and not phone_context:
                            # Keep account
                            if last_entity['label'] == 'ACCOUNT_NUMBER':
                                resolved[-1] = last_entity
                            else:
                                resolved[-1] = entity
                        else:
                            # Default: phone has higher priority for 10-digit
                            if last_entity['label'] == 'PHONE_NUMBER':
                                resolved[-1] = last_entity
                            else:
                                resolved[-1] = entity
                    else:
                        # Different lengths - choose by priority
                        if entity_priority > last_priority:
                            resolved[-1] = entity
                        elif entity_priority == last_priority:
                            if entity['confidence'] > last_entity['confidence']:
                                resolved[-1] = entity
                else:
                    # Normal conflict resolution
                    if entity_priority > last_priority:
                        resolved[-1] = entity
                    elif entity_priority == last_priority:
                        if entity['confidence'] > last_entity['confidence']:
                            resolved[-1] = entity
            else:
                resolved.append(entity)
      
        return resolved
  
    @classmethod
    def _adjust_confidence_by_context(cls, entities: List[Dict], text_lower: str) -> List[Dict]:
        """Adjust confidence based on context"""
        for entity in entities:
            label = entity['label']
            
            # Get context window
            start_pos = entity['start']
            end_pos = entity['end']
            window_start = max(0, start_pos - 50)
            window_end = min(len(text_lower), end_pos + 50)
            context = text_lower[window_start:window_end]
            
            # Phone number confidence adjustments
            if label == 'PHONE_NUMBER':
                # Boost confidence for clear phone context
                phone_context = False
                for pattern in cls.PHONE_CONTEXT_PATTERNS:
                    if re.search(pattern, context):
                        phone_context = True
                        break
                
                if phone_context:
                    entity['confidence'] = min(0.95, entity['confidence'] + 0.15)
            
            # Account number confidence adjustments
            elif label == 'ACCOUNT_NUMBER':
                # Boost for >10 digits
                digits = ''.join(filter(str.isdigit, entity['text']))
                if len(digits) > 10:
                    entity['confidence'] = min(0.95, entity['confidence'] + 0.1)
                
                # Boost for clear account context
                account_context = False
                for pattern in cls.ACCOUNT_CONTEXT_PATTERNS:
                    if re.search(pattern, context):
                        account_context = True
                        break
                
                if account_context:
                    entity['confidence'] = min(0.95, entity['confidence'] + 0.1)
        
        return entities

# RELATIONSHIP EXTRACTOR

class RelationshipExtractor:
    """Extract semantic relationships"""
  
    @staticmethod
    def extract(entities: List[Dict], text: str) -> List[Dict]:
        """Extract relationships"""
        if not entities:
            return []
      
        relationships = []
        customer = "Customer"
      
        # Map entity labels to predicate names
        predicate_map = {
            'PHONE_NUMBER': 'has_phone',
            'ACCOUNT_NUMBER': 'has_account',
            'CARD_NUMBER': 'has_card',
            'EMAIL': 'has_email',
            'TRANSACTION_ID': 'has_transaction',
            'AMOUNT': 'has_amount',
            'OTP': 'has_otp',
            'AADHAAR': 'has_aadhaar',
            'PAN': 'has_pan',
            'IFSC': 'has_ifsc',
        }
      
        # Basic relationships
        for entity in entities:
            label = entity['label']
            if label in predicate_map:
                predicate = predicate_map[label]
                relationships.append({
                    'subject': customer,
                    'predicate': predicate,
                    'object': entity['text'],
                    'confidence': min(0.9, entity.get('confidence', 0.7) * 0.9),
                })
      
        # Link amounts to transactions
        transactions = [e for e in entities if e['label'] == 'TRANSACTION_ID']
        amounts = [e for e in entities if e['label'] == 'AMOUNT']
      
        if transactions and amounts:
            # Link first transaction to closest amount
            txn = transactions[0]
            closest_amount = None
            min_distance = float('inf')
          
            for amt in amounts:
                distance = abs(txn['start'] - amt['start'])
                if distance < min_distance and distance < 150:
                    min_distance = distance
                    closest_amount = amt
          
            if closest_amount:
                relationships.append({
                    'subject': txn['text'],
                    'predicate': 'has_amount',
                    'object': closest_amount['text'],
                    'confidence': 0.6,
                })
      
        # Remove duplicates
        unique_rels = []
        seen = set()
      
        for rel in relationships:
            key = (rel['subject'], rel['predicate'], rel['object'])
            if key not in seen:
                unique_rels.append(rel)
                seen.add(key)
      
        return unique_rels

# FIXED SENTIMENT ANALYZER

class SentimentAnalyzer:
    """Sentiment analysis with Groq LLM fallback"""
  
    POSITIVE_INDICATORS = {
        'good', 'great', 'excellent', 'awesome', 'fantastic',
        'happy', 'satisfied', 'pleased', 'delighted',
        'thank', 'thanks', 'thankful', 'grateful', 'appreciate',
        'helpful', 'resolved', 'perfect', 'quick', 'fast',
        'easy', 'simple', 'convenient', 'working', 'successful',
    }
  
    NEGATIVE_INDICATORS = {
        'bad', 'terrible', 'horrible', 'awful', 'poor', 'worst',
        'angry', 'upset', 'furious', 'frustrated', 'annoyed', 'irritated',
        'disappointed', 'dissatisfied', 'unhappy', 'sad', 'depressed',
        'failed', 'failure', 'error', 'issue', 'problem', 'bug', 'glitch',
        'broken', 'not working', 'slow', 'late', 'delayed', 'pending',
        'complicated', 'confusing', 'difficult', 'complex', 'expensive',
        'complaint', 'escalate', 'supervisor', 'manager', 'legal',
        'urgent', 'emergency', 'immediate', 'asap', 'now',
        'lost', 'stolen', 'blocked', 'hacked', 'fraud',
    }
  
    @classmethod
    def analyze(cls, text: str, language: str = "en") -> SentimentEmotion:
        """Analyze sentiment with Groq LLM enhancement"""
        if not text or len(text.strip()) < 10:
            return SentimentEmotion(
                sentiment_label='neutral',
                sentiment_score=0.0,
                primary_emotion=None,
                reasoning="Text too short"
            )
      
        # Get rule-based analysis
        rule_based = cls._rule_based_analysis(text, language)
      
        # Use LLM for longer or complex text
        if len(text) > 80 or abs(rule_based.sentiment_score) < 0.2:
            llm_analysis = cls._llm_analysis(text, language)
            if llm_analysis:
                return llm_analysis
      
        return rule_based
  
    @classmethod
    def _rule_based_analysis(cls, text: str, language: str) -> SentimentEmotion:
        """Rule-based sentiment analysis"""
        text_lower = text.lower()
      
        # Extract words
        words = re.findall(r'\b\w+\b', text_lower)
      
        pos_count = 0
        neg_count = 0
      
        for word in words:
            if word in cls.POSITIVE_INDICATORS:
                pos_count += 1
            elif word in cls.NEGATIVE_INDICATORS:
                neg_count += 1
      
        total = pos_count + neg_count
      
        if total == 0:
            score = 0.0
        else:
            score = (pos_count - neg_count) / total
      
        # Determine label
        if score >= 0.6:
            label = "very_positive"
        elif score >= 0.3:
            label = "positive"
        elif score <= -0.6:
            label = "very_negative"
        elif score <= -0.3:
            label = "negative"
        else:
            label = "neutral"
      
        # Check for urgency
        urgency_terms = ['urgent', 'emergency', 'immediate', 'asap', 'now']
        if label in ['negative', 'very_negative'] and any(term in text_lower for term in urgency_terms):
            label = "urgent_negative"
      
        # Emotion detection
        emotion = cls._detect_emotion(text_lower, score)
      
        reasoning = f"Rule-based: {pos_count} positive, {neg_count} negative indicators"
        if emotion:
            reasoning += f", emotion: {emotion}"
      
        return SentimentEmotion(
            sentiment_label=label,
            sentiment_score=round(score, 3),
            primary_emotion=emotion,
            reasoning=reasoning
        )
  
    @classmethod
    def _llm_analysis(cls, text: str, language: str) -> Optional[SentimentEmotion]:
        """Use Groq LLM for sentiment analysis"""
        try:
            client = LazyLoader.get_groq_llm_client()
            if not client:
                return None
          
            prompt = f"""Analyze customer service call sentiment:
          
            Text: "{text[:400]}"
            Language: {language}
          
            Provide JSON analysis:
            {{
                "sentiment_label": "very_positive|positive|neutral|negative|very_negative|urgent_negative",
                "sentiment_score": -1.0 to 1.0,
                "primary_emotion": "anger|frustration|sadness|anxiety|satisfaction|gratitude|relief|urgency|neutral",
                "reasoning": "brief explanation"
            }}
          
            Consider banking context: payment failures are negative."""
          
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
          
            result_text = response.choices[0].message.content.strip()
          
            # Extract JSON
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    result['sentiment_score'] = max(-1.0, min(1.0, float(result['sentiment_score'])))
                    return SentimentEmotion(**result)
                except:
                    pass
          
        except Exception as e:
            logger.warning(f"Groq sentiment analysis failed: {e}")
      
        return None
  
    @staticmethod
    def _detect_emotion(text_lower: str, sentiment_score: float) -> Optional[str]:
        """Detect primary emotion"""
        emotion_patterns = {
            'anger': ['angry', 'furious', 'mad', 'rage', 'irate', 'livid'],
            'frustration': ['frustrated', 'annoyed', 'irritated', 'fed up'],
            'sadness': ['sad', 'upset', 'disappointed', 'depressed', 'unhappy'],
            'anxiety': ['worried', 'anxious', 'concerned', 'nervous', 'tense'],
            'urgency': ['urgent', 'emergency', 'immediate', 'asap', 'now', 'quickly'],
            'gratitude': ['thank', 'thanks', 'grateful', 'appreciate', 'thankful'],
            'satisfaction': ['satisfied', 'happy', 'pleased', 'delighted', 'content'],
            'relief': ['relieved', 'relief', 'glad', 'thank god'],
        }
      
        if sentiment_score < -0.5:
            for emotion, patterns in emotion_patterns.items():
                if any(pattern in text_lower for pattern in patterns):
                    return emotion
            return 'anger'
      
        elif sentiment_score > 0.5:
            for emotion, patterns in emotion_patterns.items():
                if emotion in ['gratitude', 'satisfaction', 'relief']:
                    if any(pattern in text_lower for pattern in patterns):
                        return emotion
            return 'satisfaction'
      
        return None

# SUMMARY GENERATOR

class SummaryGenerator:
    """Generate summaries using Groq LLM"""
  
    @staticmethod
    def generate(text: str, entities: List[Dict], intent: IntentDetection,
                sentiment: SentimentEmotion, language: str) -> str:
        """Generate summary with Groq LLM enhancement"""
      
        # Try LLM for better summaries
        llm_summary = SummaryGenerator._llm_summary(text, entities, intent, sentiment, language)
        if llm_summary:
            return llm_summary
      
        # Fallback to rule-based
        return SummaryGenerator._rule_based_summary(text, entities, intent, sentiment, language)
  
    @staticmethod
    def _llm_summary(text: str, entities: List[Dict], intent: IntentDetection,
                    sentiment: SentimentEmotion, language: str) -> Optional[str]:
        """Generate summary using Groq LLM"""
        try:
            client = LazyLoader.get_groq_llm_client()
            if not client:
                return None
          
            # Format key entities
            key_entities = []
            for entity in entities[:6]:
                key_entities.append(f"{entity['label']}: {entity['text']}")
          
            prompt = f"""Create concise summary of this customer service call:
            Call: "{text[:350]}"
            Intent: {intent.primary_intent}
            Sentiment: {sentiment.sentiment_label}
            Primary Emotion: {sentiment.primary_emotion or 'Not specified'}
            Key Details: {', '.join(key_entities)}
            
            2-3 sentence summary focusing on:
            1. Customer's main issue
            2. Key information mentioned
            3. Emotional tone
            Summary:"""
          
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=150,
            )
          
            summary = response.choices[0].message.content.strip()
          
            # Clean up
            summary = re.sub(r'^Summary:\s*', '', summary)
            summary = re.sub(r'\s+', ' ', summary).strip()
          
            if summary and len(summary) > 30:
                return summary
          
        except Exception as e:
            logger.warning(f"Groq summary generation failed: {e}")
      
        return None
  
    @staticmethod
    def _rule_based_summary(text: str, entities: List[Dict], intent: IntentDetection,
                           sentiment: SentimentEmotion, language: str) -> str:
        """Rule-based summary generation"""
        parts = []
      
        # Intent and sentiment
        intent_desc = intent.primary_intent.replace('_', ' ').lower()
        if intent.primary_intent == "UNKNOWN":
            parts.append("The call was informational and not related to a specific service issue.")
        else:
            parts.append(f"Customer contacted regarding {intent_desc}.")

        if sentiment.primary_emotion:
            parts.append(f"Sentiment is {sentiment.sentiment_label} with {sentiment.primary_emotion} emotion.")
        else:
            parts.append(f"Overall sentiment is {sentiment.sentiment_label}.")
      
        # Key entities
        important_labels = ['PHONE_NUMBER', 'ACCOUNT_NUMBER', 'TRANSACTION_ID', 'AMOUNT', 'CARD_NUMBER']
        important_entities = [
            e for e in entities
            if e['label'] in important_labels
        ]
      
        if important_entities:
            entity_desc = []
            for entity in important_entities[:4]:
                label = entity['label'].replace('_', ' ').lower()
                entity_desc.append(f"{label}: {entity['text']}")
          
            if entity_desc:
                parts.append(f"Key details mentioned: {', '.join(entity_desc)}.")
      
        # Urgency indicator
        if sentiment.sentiment_label == 'urgent_negative':
            parts.append("Customer expressed urgency in resolution.")
      
        return " ".join(parts)

# FIXED INTENT DETECTOR - CRITICAL ISSUE 5

class IntentDetector:
    """Fixed intent detection with entity-based boosting"""
  
    @staticmethod
    def detect(text: str, entities: List[Dict]) -> IntentDetection:
        """Detect intent with FIXED entity-based boosting"""
        text_lower = text.lower()
        intents = {}
      
        # Base regex patterns
        if re.search(r'payment\s+(?:failed|failure|not\s+going)', text_lower):
            intents['PAYMENT_ISSUE'] = 0.85
        if re.search(r'transaction\s+(?:failed|failure|not\s+completed)', text_lower):
            intents['PAYMENT_ISSUE'] = max(intents.get('PAYMENT_ISSUE', 0), 0.8)
        if re.search(r'refund\s+(?:not\s+received|pending|request)', text_lower):
            intents['REFUND_REQUEST'] = 0.8
        if re.search(r'check\s+balance', text_lower):
            intents['BALANCE_INQUIRY'] = 0.7
        if re.search(r'card\s+(?:lost|stolen|blocked|hacked)', text_lower):
            intents['CARD_ISSUE'] = 0.85
        if re.search(r'login\s+(?:problem|issue|failed|not\s+working)', text_lower):
            intents['LOGIN_ISSUE'] = 0.7
        if re.search(r'otp\s+(?:not\s+received|problem|issue)', text_lower):
            intents['OTP_ISSUE'] = 0.75
        if re.search(r'(payment|paisa).*(fail|kat|cut)', text_lower):
            intents['PAYMENT_ISSUE'] = max(intents.get('PAYMENT_ISSUE', 0), 0.9)
        if re.search(r'refund.*(karo|kar|dijiye|chahiye)', text_lower):
            intents['REFUND_REQUEST'] = max(intents.get('REFUND_REQUEST', 0), 0.85)
      
        # FIXED CRITICAL ISSUE 5: Entity-based intent boosting
        for entity in entities:
            if entity['label'] == 'AMOUNT' and any(word in text_lower for word in ['failed', 'deducted', 'not']):
                intents['PAYMENT_ISSUE'] = max(intents.get('PAYMENT_ISSUE', 0), 0.9)
            
            if entity['label'] == 'TRANSACTION_ID' and any(word in text_lower for word in ['failed', 'issue', 'problem']):
                intents['PAYMENT_ISSUE'] = max(intents.get('PAYMENT_ISSUE', 0), 0.88)
        
        # FIXED: New intent categories
        if 'emi' in text_lower or 'loan' in text_lower:
            if any(word in text_lower for word in ['failed', 'bounce', 'due']):
                intents['LOAN_EMI_ISSUE'] = 0.85
        
        if 'aadhaar' in text_lower or 'pan' in text_lower:
            if any(word in text_lower for word in ['link', 'update', 'kyc']):
                intents['KYC_UPDATE'] = 0.75
        
        if 'mobile' in text_lower and 'update' in text_lower:
            intents['CONTACT_UPDATE'] = 0.7
        
        if 'statement' in text_lower or 'passbook' in text_lower:
            intents['STATEMENT_REQUEST'] = 0.65
      
        if intents:
            # Sort by confidence
            sorted_intents = sorted(intents.items(), key=lambda x: x[1], reverse=True)
            primary_intent = sorted_intents[0]
          
            return IntentDetection(
                primary_intent=primary_intent[0],
                confidence=min(primary_intent[1], 0.95),
                intents=intents
            )
      
        return IntentDetection(
            primary_intent='UNKNOWN',
            confidence=0.0,
            intents={}
        )

# FIXED THREAT DETECTOR - CRITICAL ISSUE 4

class ThreatDetector:
    """Fixed threat detection with contextual patterns"""
  
    @staticmethod
    def detect(text: str) -> ThreatProfanity:
        """Detect threats and profanity with FIXED patterns"""
        text_lower = text.lower()
      
        # FIXED CRITICAL ISSUE 4: Contextual threat patterns
        threat_patterns = [
            r'file\s+(a\s+)?(case|complaint|fir|charges)',
            r'go\s+to\s+(court|police|lawyer|legal)',
            r'sue\s+(you|bank|them)',
            r'legal\s+(action|notice|case)',
            r'police\s+(complaint|case|fir)',
            r'court\s+(case|notice|order)',
            r'lodge\s+(complaint|fir)',
            r'register\s+(case|complaint)',
            r'take\s+(legal|police)\s+action',
            r'consumer\s+forum',
            r'banking\s+ombudsman',
            r'rb[i]\s+complaint',
        ]
      
        # FIXED: Profanity with context
        profanity_patterns = [
            r'\bstupid\s+(bank|service|people)\b',
            r'\bidiot\s+(staff|customer\s+service)\b',
            r'\bfoolish\s+(system|process)\b',
            r'\bdamn\s+(bank|service)\b',
            r'\bbloody\s+(hell|service)\b',
            r'\bcrappy\s+(service|support)\b',
            r'\bworthless\s+(service|support)\b',
            r'\bhorrible\s+(experience|service)\b',
            r'\bterrible\s+(service|support)\b',
        ]
      
        threat_terms = []
        profanity_terms = []
      
        # Check threat patterns
        for pattern in threat_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                threat_terms.append(match.group())
      
        # Check profanity patterns
        for pattern in profanity_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                profanity_terms.append(match.group())
      
        # Remove duplicates
        threat_terms = list(set(threat_terms))
        profanity_terms = list(set(profanity_terms))
      
        return ThreatProfanity(
            threat_detected=len(threat_terms) > 0,
            profanity_detected=len(profanity_terms) > 0,
            threat_terms=threat_terms,
            profanity_terms=profanity_terms
        )

# COMPLIANCE CHECKER

class ComplianceChecker:
    """Check compliance"""
  
    @classmethod
    def check(cls, text: str) -> ComplianceCheck:
        """Check compliance"""
        if not text:
            return ComplianceCheck(
                overall_score=0.0,
                passed=False,
                warnings=['Empty text']
            )
      
        text_lower = text.lower()
        warnings = []
        score = 0.8
      
        # Check for professionalism
        agent_indicators = ['welcome', 'thank you for calling', 'how can i help', 'please provide']
        customer_indicators = ['my', 'i have', 'i want', 'problem', 'issue', 'help']
      
        agent_count = sum(1 for indicator in agent_indicators if indicator in text_lower)
        customer_count = sum(1 for indicator in customer_indicators if indicator in text_lower)
      
        if agent_count > customer_count:
            if not any(phrase in text_lower for phrase in ['call may be recorded', 'for quality']):
                warnings.append('Agent speech missing recording disclosure')
                score -= 0.2
          
            if not any(word in text_lower for word in ['hello', 'hi', 'thank you']):
                warnings.append('Agent speech missing professional greeting/closing')
                score -= 0.1
      
        return ComplianceCheck(
            overall_score=max(0.0, min(1.0, round(score, 2))),
            passed=score >= 0.6,
            warnings=warnings,
            note="Automated compliance check"
        )

# AGENT ASSIST GENERATOR

class AgentAssistGenerator:
    """Agent assistance"""
  
    @staticmethod
    def generate(intent: str, sentiment: str, entities: List[Dict]) -> AgentAssist:
        """Generate agent assistance"""
        suggestions = []
      
        # Intent-based suggestions
        if intent == 'PAYMENT_ISSUE':
            suggestions.append("Ask for transaction ID, amount, and payment method")
            suggestions.append("Verify the customer's account/card details")
            suggestions.append("Check payment gateway status and error codes")
            suggestions.append("Explain refund timeline if payment failed but money deducted")
      
        elif intent == 'REFUND_REQUEST':
            suggestions.append("Verify original transaction details")
            suggestions.append("Check refund policy and processing timeline")
            suggestions.append("Explain any charges or deductions")
            suggestions.append("Provide reference number for tracking")
      
        elif intent == 'BALANCE_INQUIRY':
            suggestions.append("Authenticate customer using registered mobile")
            suggestions.append("Provide current balance and available limit")
            suggestions.append("Offer mini-statement if requested")
            suggestions.append("Explain any pending transactions")
      
        elif intent == 'CARD_ISSUE':
            suggestions.append("Verify customer identity thoroughly")
            suggestions.append("Block card immediately if lost/stolen")
            suggestions.append("Explain card replacement process and timeline")
            suggestions.append("Check for any unauthorized transactions")
      
        elif intent == 'LOAN_EMI_ISSUE':
            suggestions.append("Check loan account status and due date")
            suggestions.append("Verify EMI amount and due date")
            suggestions.append("Explain bounce charges and grace period")
            suggestions.append("Check for any technical issues with auto-debit")
      
        else:
            suggestions.append("Ask clarifying questions to understand the issue")
            suggestions.append("Verify customer identity")
            suggestions.append("Document all details for follow-up")
            suggestions.append("Provide reference number for the interaction")
      
        # Sentiment-based suggestions
        if sentiment in ['negative', 'very_negative', 'urgent_negative']:
            suggestions.append("Use empathetic language and acknowledge frustration")
            suggestions.append("Avoid technical jargon, explain in simple terms")
            suggestions.append("Focus on solution, not blame")
            suggestions.append("Set clear expectations for resolution timeline")
      
        # Entity-based suggestions
        has_account = any(e['label'] == 'ACCOUNT_NUMBER' for e in entities)
        has_phone = any(e['label'] == 'PHONE_NUMBER' for e in entities)
      
        if not has_account and intent in ['PAYMENT_ISSUE', 'BALANCE_INQUIRY', 'REFUND_REQUEST']:
            suggestions.append("Ask for account number or registered mobile number")
      
        if not has_phone:
            suggestions.append("Verify or collect contact number for updates")
      
        return AgentAssist(
            suggestions=suggestions[:5],
            next_best_action=suggestions[0] if suggestions else "Listen carefully and understand the issue",
            call_flow_action="Follow standard escalation protocol if needed"
        )

# CALL SCORE CALCULATOR

class CallScoreCalculator:
    """Call score calculation"""
  
    @staticmethod
    def calculate(sentiment_score: float, intent_confidence: float,
                 compliance_score: float, entity_count: int,
                 has_threats: bool) -> int:
        """Calculate call score"""
      
        # Base scores with weights
        sentiment_weight = 0.35
        intent_weight = 0.25
        compliance_weight = 0.20
        entity_weight = 0.20
      
        # Normalize sentiment score from -1..1 to 0..1
        sentiment_normalized = (sentiment_score + 1) / 2
      
        # Calculate components
        sentiment_component = 100 * sentiment_weight * sentiment_normalized
        intent_component = 100 * intent_weight * intent_confidence
        compliance_component = 100 * compliance_weight * compliance_score
        entity_component = 100 * entity_weight * min(1.0, entity_count / 10.0)
      
        score = sentiment_component + intent_component + compliance_component + entity_component
      
        # Penalties
        if has_threats:
            score *= 0.7
      
        # Ensure within bounds and round
        final_score = max(0, min(100, int(round(score))))
      
        return final_score

# IMPROVED STT ALGORITHM (FINAL – SAFE & WORKING)

class AudioProcessor:
    """Audio transcription with improved STT accuracy"""

    @staticmethod
    def validate_file(filename: str, size: int) -> None:
        """Validate audio file"""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in Config.ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                f"Allowed: {', '.join(Config.ALLOWED_AUDIO_EXTENSIONS)}"
            )

        max_size = Config.MAX_AUDIO_SIZE_MB * 1024 * 1024
        if size > max_size:
            raise ValueError(
                f"File too large ({size / (1024*1024):.1f} MB). "
                f"Max: {Config.MAX_AUDIO_SIZE_MB} MB"
            )

    @staticmethod
    def transcribe(file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Transcribe audio using Groq Whisper (FINAL FIXED VERSION)"""

        if not Config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not configured")

        client = LazyLoader.get_groq_client()

        transcript = ""
        language = "en"
        duration = None

        # Create temp file
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in Config.ALLOWED_AUDIO_EXTENSIONS:
            suffix = ".wav"

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            with open(tmp_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    language="en",
                    prompt=(
                        "This is a customer service call about banking, payments, "
                        "transactions, accounts, cards, refunds, balance inquiry, "
                        "UPI, mobile banking, net banking, loans, EMI, "
                        "customer support, IVR, call center."
                    )
                )

            # SAFE RESPONSE EXTRACTION
            if isinstance(response, dict):
                transcript = response.get("text", "")
                language = response.get("language", "en")
                duration = response.get("duration")
            else:
                transcript = getattr(response, "text", "")
                language = getattr(response, "language", "en")
                duration = getattr(response, "duration", None)

            # Post-process transcript
            if transcript:
                transcript = AudioProcessor._enhanced_post_process(transcript)

            return {
                "transcript": transcript,
                "language": language,
                "duration": duration,
                "success": bool(transcript),
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return AudioProcessor._fallback_transcription(file_bytes)

        finally:
            try:
                if "tmp_path" in locals():
                    os.unlink(tmp_path)
            except Exception:
                pass

    @staticmethod
    def _enhanced_post_process(transcript: str) -> str:
        """Enhanced post-processing for better accuracy"""
        if not transcript:
            return ""

        transcript = re.sub(r"\s+", " ", transcript).strip()
        
        transcript = re.sub(
            r'(\w+)\s+dot\s+(\w+)\s+at\s+(\w+)\s+dot\s+(\w+)',
            r'\1.\2@\3.\4',
            transcript,
            flags=re.IGNORECASE
        )
        
        transcript = re.sub(
            r'(\w+)\s+at\s+(\w+)\s+dot\s+(\w+)',
            r'\1@\2.\3',
            transcript,
            flags=re.IGNORECASE
        )

        # Capitalize sentences
        sentences = re.split(r"([.!?]\s+)", transcript)
        processed = []
        capitalize_next = True

        for part in sentences:
            if capitalize_next and part.strip():
                part = part.strip()
                part = part[0].upper() + part[1:]
                capitalize_next = False
            if part.endswith((".", "!", "?")):
                capitalize_next = True
            processed.append(part)

        transcript = "".join(processed)

        # Fix banking terms
        replacements = {
            r"\bu\s*\.?\s*p\s*\.?\s*i\b": "UPI",
            r"\ba\s*\.?\s*t\s*\.?\s*m\b": "ATM",
            r"\btransaction\s+i\s*d\b": "transaction ID",
            r"\breference\s+number\b": "reference number",
            r"\bmobile\s+banking\b": "mobile banking",
            r"\bnet\s+banking\b": "net banking",
            r"\bcredit\s+card\b": "credit card",
            r"\bdebit\s+card\b": "debit card",
        }

        for pattern, repl in replacements.items():
            transcript = re.sub(pattern, repl, transcript, flags=re.IGNORECASE)

        return transcript.strip()

    @staticmethod
    def _fallback_transcription(file_bytes: bytes) -> Dict[str, Any]:
        """Fallback transcription"""
        logger.warning("Using fallback transcription")
        return {
            "transcript": "",
            "language": "en",
            "duration": None,
            "success": False,
        }

# MAIN PROCESSING PIPELINE

class IVRPipeline:
    """Main IVR processing pipeline"""

    @staticmethod
    def process_text(text: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Process text through complete pipeline"""

        # 🔒 SAFETY CHECK FOR TEXT LENGTH
        if len(text) > Config.MAX_TEXT_LENGTH:
            raise ValueError("Text exceeds maximum allowed length")

        start_time = time.time()

        # Step 1: Clean
        cleaned_text = TextPreprocessor.clean(text)
        if not cleaned_text or len(cleaned_text.strip()) < 3:
            raise ValueError("Text too short")

        # Step 2: Detect language
        if not language:
            language = TextPreprocessor.detect_language(cleaned_text)

        # Step 3: Extract entities
        entities = NEREngine.extract_entities(cleaned_text)

        # Step 4: Extract relationships
        relationships = RelationshipExtractor.extract(entities, cleaned_text)

        # Step 5: Analyze sentiment
        sentiment = SentimentAnalyzer.analyze(cleaned_text, language)

        # Step 6: Detect intent
        intent = IntentDetector.detect(cleaned_text, entities)

        # Step 7: Check compliance
        compliance = ComplianceChecker.check(cleaned_text)

        # Step 8: Detect threats
        risk = ThreatDetector.detect(cleaned_text)

        # Step 9: Generate summary
        summary = SummaryGenerator.generate(
            cleaned_text, entities, intent, sentiment, language
        )

        # Step 10: Generate agent assistance
        agent_assist = AgentAssistGenerator.generate(
            intent.primary_intent,
            sentiment.sentiment_label,
            entities
        )

        # Step 11: Calculate call score
        call_score = CallScoreCalculator.calculate(
            sentiment_score=sentiment.sentiment_score,
            intent_confidence=intent.confidence,
            compliance_score=compliance.overall_score,
            entity_count=len(entities),
            has_threats=risk.threat_detected
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return {
            "text": cleaned_text,
            "language": language,
            "entities": entities,
            "relationships": relationships,
            "sentiment": sentiment,
            "intents": intent,
            "compliance": compliance,
            "risk_flags": risk,
            "summary": summary,
            "agent_assist": agent_assist,
            "call_score": call_score,
            "processing_time_ms": processing_time_ms,
        }

# DATABASE HELPER

class DatabaseHelper:
    """Database operations"""
  
    @staticmethod
    def save_analysis(
        db: Session,
        input_type: str,
        transcript: str,
        result: Dict[str, Any]
    ) -> int:
        """Save analysis to database"""
      
        record = CallAnalysis(
            input_type=input_type,
            transcript=transcript[:2000],
            entities_json=json.dumps(result['entities'], ensure_ascii=False),
            relationships_json=json.dumps(result['relationships'], ensure_ascii=False),
            sentiment=result['sentiment'].sentiment_label,
            intent=result['intents'].primary_intent,
            call_score=result['call_score'],
            language=result.get('language', 'unknown'),
            processing_time_ms=result['processing_time_ms'],
        )
      
        db.add(record)
        db.commit()
      
        return record.id

# FASTAPI SETUP
app = FastAPI(
    title="IVR NER Analyzer API",
    version="6.0.0",
    description="Production-grade IVR Call Analytics with NER, Sentiment, Intent, STT"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API ENDPOINTS
@app.get("/")
async def root():
    return {"status": "healthy", "version": "6.0.0"}

@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file with improved STT
    """
    try:
        file_bytes = await file.read()
        AudioProcessor.validate_file(file.filename, len(file_bytes))
      
        result = AudioProcessor.transcribe(file_bytes, file.filename)
      
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail="Transcription failed")
      
        return TranscribeResponse(
            transcript=result["transcript"],
            language=result["language"],
            duration=result["duration"],
            success=result["success"],
            )

      
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/analyze/text", response_model=AnalyzeTextResponse)
async def analyze_text(
    request: AnalyzeTextRequest,
    db: Session = Depends(get_db)
):
    """
    Analyze text with fixed NER accuracy
    """

    # 🔒 ENFORCE MAX TEXT LENGTH
    if len(request.text) > Config.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail="Input text exceeds maximum allowed length"
        )

    try:
        result = IVRPipeline.process_text(request.text)

        
      
        # Save to database
        record_id = DatabaseHelper.save_analysis(
            db, "text", request.text, result
        )
      
        # Convert to response
        entities_response = [
            Entity(
                text=e['text'],
                label=e['label'],
                start=e['start'],
                end=e['end'],
                source=e.get('source', 'unknown'),
                confidence=round(e.get('confidence', 0.5), 2)
            )
            for e in result['entities']
        ]
      
        relationships_response = [
            Relationship(
                subject=r['subject'],
                predicate=r['predicate'],
                object=r['object'],
                confidence=round(r.get('confidence', 0.5), 2)
            )
            for r in result['relationships']
        ]
      
        return AnalyzeTextResponse(
            text=result['text'],
            language=result['language'],
            entities=entities_response,
            relationships=relationships_response,
            sentiment=result['sentiment'],
            intents=result['intents'],
            summary=result['summary'],
            agent_assist=result['agent_assist'],
            compliance=result['compliance'],
            risk_flags=result['risk_flags'],
            call_score=result['call_score'],
            processing_time_ms=result['processing_time_ms'],
        )
      
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze/audio", response_model=AnalyzeTextResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Transcribe and analyze audio with improved STT
    """
    try:
        file_bytes = await file.read()
        AudioProcessor.validate_file(file.filename, len(file_bytes))
      
        transcription = AudioProcessor.transcribe(file_bytes, file.filename)
      
        if not transcription.get("transcript"):
            raise HTTPException(status_code=400, detail="Transcription returned empty text")
      
        result = IVRPipeline.process_text(
            transcription["transcript"],
            transcription.get("language")
        )
      
        # Save to database
        record_id = DatabaseHelper.save_analysis(
            db, "audio", transcription["transcript"], result
        )
      
        # Convert to response
        entities_response = [
            Entity(
                text=e['text'],
                label=e['label'],
                start=e['start'],
                end=e['end'],
                source=e.get('source', 'unknown'),
                confidence=round(e.get('confidence', 0.5), 2)
            )
            for e in result['entities']
        ]
      
        relationships_response = [
            Relationship(
                subject=r['subject'],
                predicate=r['predicate'],
                object=r['object'],
                confidence=round(r.get('confidence', 0.5), 2)
            )
            for r in result['relationships']
        ]
      
        return AnalyzeTextResponse(
            text=result['text'],
            language=result['language'],
            entities=entities_response,
            relationships=relationships_response,
            sentiment=result['sentiment'],
            intents=result['intents'],
            summary=result['summary'],
            agent_assist=result['agent_assist'],
            compliance=result['compliance'],
            risk_flags=result['risk_flags'],
            call_score=result['call_score'],
            processing_time_ms=result['processing_time_ms'],
        )
      
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/test/ner")
async def test_ner(text: str = Query(..., description="Text to test NER")):
    """Test NER accuracy with the fixes"""

    # 🔒 LIMIT TEST INPUT SIZE
    if len(text) > Config.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail="Text too long for NER test"
        )

    try:
        entities = NEREngine.extract_entities(text)

      
        # Debug info for phone vs account classification
        debug_info = []
        for entity in entities:
            if entity['label'] in ['PHONE_NUMBER', 'ACCOUNT_NUMBER']:
                digits = ''.join(filter(str.isdigit, entity['text']))
                debug_info.append({
                    'text': entity['text'],
                    'label': entity['label'],
                    'confidence': entity['confidence'],
                    'digits': digits,
                    'digit_count': len(digits),
                })
      
        return {
            "text": text,
            "entities": entities,
            "debug_info": debug_info,
            "statistics": {
                "total_entities": len(entities),
                "phone_numbers": sum(1 for e in entities if e['label'] == 'PHONE_NUMBER'),
                "account_numbers": sum(1 for e in entities if e['label'] == 'ACCOUNT_NUMBER'),
                "transaction_ids": sum(1 for e in entities if e['label'] == 'TRANSACTION_ID'),
                "avg_confidence": round(sum(e.get('confidence', 0) for e in entities) / len(entities), 2) if entities else 0,
            }
        }
      
    except Exception as e:
        logger.error(f"NER test error: {e}")
        raise HTTPException(status_code=500, detail=f"NER test failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=Config.PORT,
        workers=1,
        log_level="warning",
    )