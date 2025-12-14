"""
IVR NER ANALYZER - PRODUCTION READY V5.0
Fixed critical NER accuracy issues + Improved STT
"""
import os
import json
import re
import tempfile
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
import logging

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Database
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
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

# ============================================================================
# CONFIGURATION
# ============================================================================
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
        'OTP': 20,
    }

# ============================================================================
# LAZY LOADING FUNCTIONS
# ============================================================================
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
                LazyLoader._spacy_model.max_length = 1000000
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

# ============================================================================
# DATABASE SETUP
# ============================================================================
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

engine = create_engine(
    Config.DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=None,
    echo=False
)

# Drop and recreate to ensure schema is correct
try:
    Base.metadata.drop_all(bind=engine)
except:
    pass
Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
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

# ============================================================================
# TEXT PREPROCESSOR
# ============================================================================
class TextPreprocessor:
    """Text preprocessing"""
  
    IVR_PHRASES = [
        r"this call may be recorded.*",
        r"press \d+ to continue",
        r"please wait while we connect",
    ]
  
    @classmethod
    def clean(cls, text: str) -> str:
        """Clean text"""
        if not text or not text.strip():
            return ""
      
        text = text.lower()
      
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

# ============================================================================
# FIXED NER ENGINE - CRITICAL ACCURACY FIXES APPLIED
# ============================================================================
class NEREngine:
    """Advanced NER with all critical fixes applied"""
  
    # Phone number context patterns - FIXED ISSUE 1
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
    ]
  
    # Pattern configuration with FIXED validations
    PATTERNS = {
        'PHONE_NUMBER': {
            'patterns': [
                r'\b\d{10}\b',
                r'\b\+91[-\s]?\d{10}\b',
                r'\b\d{5}[-\s]?\d{5}\b',
            ],
            'base_confidence': 0.8,
            'context_patterns': PHONE_CONTEXT_PATTERNS,
            'validation': lambda x, txt: NEREngine._is_phone_number(x, txt),
        },
      
        'ACCOUNT_NUMBER': {
            'patterns': [
                r'\b\d{9,18}\b',
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4,6}\b',
            ],
            'base_confidence': 0.7,
            'context_patterns': ACCOUNT_CONTEXT_PATTERNS,
            'validation': lambda x, txt: NEREngine._is_account_number(x, txt),
        },
      
        'TRANSACTION_ID': {
            'patterns': [
                r'\b(?:txn|trxn|trans)[-_]?(?:id|ref)?[_\s-]?[A-Z0-9]{6,20}\b',
                r'\b(?:ref|reference)[-_]?(?:id|no)?[_\s-]?[A-Z0-9]{6,20}\b',
                r'\b(?:order|ord)[-_]?(?:id|no)?[_\s-]?[A-Z0-9]{6,20}\b',
            ],
            'base_confidence': 0.85,  # Slightly reduced from 0.9
            'context_patterns': ['transaction', 'txn', 'reference', 'ref', 'order'],
            'validation': lambda x, txt: True,
        },
      
        'CARD_NUMBER': {
            'patterns': [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                r'\b\d{16}\b',
            ],
            'base_confidence': 0.9,
            'context_patterns': ['card', 'credit', 'debit', 'visa', 'mastercard'],
            'validation': lambda x, txt: True,
        },
      
        'AMOUNT': {
            'patterns': [
                r'₹\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b',
                r'rs\.?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b',
                r'rupees?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b',
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s?(?:rs|rupees|₹)\b',
            ],
            'base_confidence': 0.85,
            'context_patterns': ['amount', 'rs', 'rupees', '₹', 'paid', 'charge'],
            'validation': lambda x, txt: True,
        },
      
        'EMAIL': {
            'patterns': [
                r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            ],
            'base_confidence': 0.95,
            'context_patterns': ['email', 'mail', 'id'],
            'validation': lambda x, txt: True,
        },
    }
  
    @staticmethod
    def _is_phone_number(entity_text: str, full_text: str) -> bool:
        """Improved phone number validation - FIXED ISSUE 1"""
        digits = ''.join(filter(str.isdigit, entity_text))
        if len(digits) != 10:
            return False
      
        # Check for phone context patterns - FIXED regex
        text_lower = full_text.lower()
        phone_context = False
      
        for pattern in NEREngine.PHONE_CONTEXT_PATTERNS:
            if re.search(pattern, text_lower):
                phone_context = True
                break
      
        # Also check for standalone phone indicators
        standalone_indicators = [
            r'\bphone\b',
            r'\bmobile\b',
            r'\bcontact\b',
            r'\bnumber\b.*\d{10}',
            r'\d{10}.*\bnumber\b',
        ]
      
        for indicator in standalone_indicators:
            if re.search(indicator, text_lower):
                phone_context = True
                break
      
        # If we have clear phone context, it's definitely a phone
        if phone_context:
            return True
      
        # If we have account context, it's probably not a phone
        account_context = False
        for pattern in NEREngine.ACCOUNT_CONTEXT_PATTERNS:
            if re.search(pattern, text_lower):
                account_context = True
                break
      
        if account_context:
            return False
      
        # Default: 10-digit numbers without context are phones
        return True
  
    @staticmethod
    def _is_account_number(entity_text: str, full_text: str) -> bool:
        """Improved account number validation"""
        digits = ''.join(filter(str.isdigit, entity_text))
        if not (9 <= len(digits) <= 18):
            return False
      
        # Check for account context
        text_lower = full_text.lower()
        account_context = False
      
        for pattern in NEREngine.ACCOUNT_CONTEXT_PATTERNS:
            if re.search(pattern, text_lower):
                account_context = True
                break
      
        # Account numbers usually have more than 10 digits
        if len(digits) > 10:
            return True
      
        # 10-digit numbers need clear account context
        if len(digits) == 10 and account_context:
            return True
      
        # 9-10 digit numbers without clear context might be account or phone
        if len(digits) in [9, 10] and not account_context:
            return False
      
        return account_context
  
    @classmethod
    def extract_entities(cls, text: str) -> List[Dict[str, Any]]:
        """Extract entities with all fixes applied"""
        if not text:
            return []
      
        text_lower = text.lower()
        all_entities = []
      
        # Extract rule-based entities
        for label, config in cls.PATTERNS.items():
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
                    logger.warning(f"Pattern error {pattern}: {e}")
                    continue
      
        # Extract spaCy entities with FIXED ISSUE 2
        spacy_entities = cls._extract_spacy_entities(text)
        all_entities.extend(spacy_entities)
      
        # Resolve conflicts
        resolved = cls._resolve_entity_conflicts(all_entities, text_lower)
      
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
      
        confidence = min(base_confidence, 0.9)  # FIXED: Cap at 0.9
      
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
          
        elif label == 'TRANSACTION_ID':
            # Transaction IDs with common prefixes get confidence boost
            prefixes = ['txn', 'trxn', 'trans', 'ref', 'ord']
            if any(entity_text.lower().startswith(prefix) for prefix in prefixes):
                confidence = min(0.9, confidence + 0.05)
      
        return max(Config.MIN_CONFIDENCE, min(0.9, round(confidence, 2)))  # Cap at 0.9
  
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
        """Extract entities using spaCy - FIXED ISSUE 2"""
        try:
            nlp = LazyLoader.get_spacy_model()
            doc = nlp(text[:1000])  # Limit length
          
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
          
            # Words that should NOT be tagged as PERSON
            person_blacklist = [
                'transaction', 'payment', 'account', 'card',
                'id', 'upi', 'refund', 'balance', 'loan',
                'amount', 'charge', 'fee', 'interest',
            ]
          
            for ent in doc.ents:
                mapped_label = label_mapping.get(ent.label_, ent.label_)
              
                # FIXED ISSUE 2: Filter false PERSON entities
                if mapped_label == 'PERSON':
                    ent_lower = ent.text.lower()
                    # Skip if contains blacklisted words
                    if any(word in ent_lower for word in person_blacklist):
                        continue
                    # Skip single words that are too short
                    if len(ent.text.split()) == 1 and len(ent.text) < 3:
                        continue
                    # Skip if looks like a transaction reference
                    if re.search(r'\b(txn|trxn|trans|ref|id)\b', ent_lower):
                        continue
              
                # Filter other obvious false positives
                if mapped_label == 'ORG' and len(ent.text) < 3:
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
        """Resolve overlapping entity conflicts"""
        if not entities:
            return []
      
        entities.sort(key=lambda x: x['start'])
        resolved = []
        current = None
      
        for entity in entities:
            if current is None:
                current = entity
                continue
          
            # Check for overlap
            if entity['start'] < current['end']:
                # Conflict - choose better entity
                current_priority = Config.ENTITY_PRIORITIES.get(current['label'], 0)
                entity_priority = Config.ENTITY_PRIORITIES.get(entity['label'], 0)
              
                if entity_priority > current_priority:
                    current = entity
                elif entity_priority == current_priority:
                    # Same priority, choose higher confidence
                    if entity['confidence'] > current['confidence']:
                        current = entity
                    # Special case: phone vs account for 10-digit numbers
                    elif (current['label'] == 'PHONE_NUMBER' and 
                          entity['label'] == 'ACCOUNT_NUMBER' and
                          len(''.join(filter(str.isdigit, current['text']))) == 10):
                        # Check context to decide
                        if re.search(r'\b(mobile|phone|contact)\b', text_lower):
                            current = current  # Keep as phone
                        elif re.search(r'\b(account|acc)\b', text_lower):
                            current = entity  # Change to account
            else:
                resolved.append(current)
                current = entity
      
        if current:
            resolved.append(current)
      
        return resolved

# ============================================================================
# IMPROVED RELATIONSHIP EXTRACTOR - FIXED ISSUE 3
# ============================================================================
class RelationshipExtractor:
    """Extract semantic relationships with consistent predicates"""
  
    @staticmethod
    def extract(entities: List[Dict], text: str) -> List[Dict]:
        """Extract relationships with FIXED predicate naming"""
        if not entities:
            return []
      
        relationships = []
        customer = "Customer"
      
        # Map entity labels to predicate names - FIXED ISSUE 3
        predicate_map = {
            'PHONE_NUMBER': 'has_phone',
            'ACCOUNT_NUMBER': 'has_account',
            'CARD_NUMBER': 'has_card',
            'EMAIL': 'has_email',
            'TRANSACTION_ID': 'has_transaction',
            'AMOUNT': 'has_amount',
        }
      
        # Basic relationships
        for entity in entities:
            label = entity['label']
            if label in predicate_map:
                # FIXED: Use consistent predicate names
                predicate = predicate_map[label]
                relationships.append({
                    'subject': customer,
                    'predicate': predicate,
                    'object': entity['text'],
                    'confidence': min(0.9, entity.get('confidence', 0.7) * 0.9),  # Slightly reduce confidence
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
                if distance < min_distance and distance < 150:  # Max 150 chars apart
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

# ============================================================================
# IMPROVED SENTIMENT ANALYZER
# ============================================================================
class SentimentAnalyzer:
    """Advanced sentiment analysis with enhanced emotion detection"""
  
    # Enhanced indicators
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
        """Analyze sentiment with enhanced emotion detection"""
        if not text or len(text.strip()) < 10:
            return SentimentEmotion(
                sentiment_label='neutral',
                sentiment_score=0.0,
                primary_emotion=None,
                reasoning="Text too short"
            )
      
        # Get enhanced rule-based analysis
        rule_based = cls._enhanced_rule_based_analysis(text, language)
      
        # Use LLM for longer or complex text
        if len(text) > 80 or abs(rule_based.sentiment_score) < 0.2:
            llm_analysis = cls._llm_analysis(text, language)
            if llm_analysis:
                return llm_analysis
      
        return rule_based
  
    @classmethod
    def _enhanced_rule_based_analysis(cls, text: str, language: str) -> SentimentEmotion:
        """Enhanced rule-based sentiment analysis"""
        text_lower = text.lower()
      
        # Extract words with stemming
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
      
        # Enhanced label determination
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
        urgency_terms = ['urgent', 'emergency', 'immediate', 'asap', 'now', 'तुरंत', 'जल्दी']
        if label in ['negative', 'very_negative'] and any(term in text_lower for term in urgency_terms):
            label = "urgent_negative"
      
        # Enhanced emotion detection
        emotion = cls._detect_enhanced_emotion(text_lower, score)
      
        reasoning = f"Enhanced analysis: {pos_count} positive, {neg_count} negative indicators"
        if emotion:
            reasoning += f", primary emotion: {emotion}"
      
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
                "reasoning": "brief explanation focusing on customer emotions and urgency"
            }}
          
            Consider banking context: payment failures are negative, refund requests indicate dissatisfaction."""
          
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
                    # Validate sentiment score range
                    result['sentiment_score'] = max(-1.0, min(1.0, float(result['sentiment_score'])))
                    return SentimentEmotion(**result)
                except:
                    pass
          
        except Exception as e:
            logger.warning(f"Groq sentiment analysis failed: {e}")
      
        return None
  
    @staticmethod
    def _detect_enhanced_emotion(text_lower: str, sentiment_score: float) -> Optional[str]:
        """Enhanced emotion detection"""
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
            # Check for specific emotions in negative sentiment
            for emotion, patterns in emotion_patterns.items():
                if any(pattern in text_lower for pattern in patterns):
                    return emotion
            return 'anger'  # Default for strong negative
      
        elif sentiment_score > 0.5:
            # Check for specific emotions in positive sentiment
            for emotion, patterns in emotion_patterns.items():
                if emotion in ['gratitude', 'satisfaction', 'relief']:
                    if any(pattern in text_lower for pattern in patterns):
                        return emotion
            return 'satisfaction'  # Default for strong positive
      
        return None

# ============================================================================
# IMPROVED SUMMARY GENERATOR
# ============================================================================
class SummaryGenerator:
    """Generate summaries with improved quality"""
  
    @staticmethod
    def generate(text: str, entities: List[Dict], intent: IntentDetection,
                sentiment: SentimentEmotion, language: str) -> str:
        """Generate summary with improved quality"""
      
        # Always try LLM first for better quality
        llm_summary = SummaryGenerator._improved_llm_summary(text, entities, intent, sentiment, language)
        if llm_summary:
            return llm_summary
      
        # Fallback to improved rule-based
        return SummaryGenerator._improved_rule_summary(text, entities, intent, sentiment, language)
  
    @staticmethod
    def _improved_llm_summary(text: str, entities: List[Dict], intent: IntentDetection,
                             sentiment: SentimentEmotion, language: str) -> Optional[str]:
        """Improved LLM summary generation"""
        try:
            client = LazyLoader.get_groq_llm_client()
            if not client:
                return None
          
            # Format key entities
            key_entities = []
            for entity in entities[:6]:  # Increased to 6 entities
                key_entities.append(f"{entity['label']}: {entity['text']}")
          
            prompt = f"""Create a professional summary for IVR analytics:
          
            Customer Call: "{text[:350]}"
            Primary Intent: {intent.primary_intent}
            Sentiment: {sentiment.sentiment_label}
            Primary Emotion: {sentiment.primary_emotion or 'Not specified'}
            Key Details: {', '.join(key_entities)}
          
            Please provide a 2-3 sentence summary that:
            1. States the customer's main issue or request clearly
            2. Mentions key identifiers (account, phone, transaction, amount)
            3. Notes the emotional tone and urgency level
            4. Provides context for agent assistance
          
            Summary:"""
          
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=180,
            )
          
            summary = response.choices[0].message.content.strip()
          
            # Clean and validate
            summary = re.sub(r'^(Summary|Here is the summary|Analysis):\s*', '', summary, flags=re.IGNORECASE)
            summary = re.sub(r'\s+', ' ', summary).strip()
          
            if summary and len(summary) > 30 and len(summary) <= 250:
                return summary
          
        except Exception as e:
            logger.warning(f"Groq summary generation failed: {e}")
      
        return None
  
    @staticmethod
    def _improved_rule_summary(text: str, entities: List[Dict], intent: IntentDetection,
                              sentiment: SentimentEmotion, language: str) -> str:
        """Improved rule-based summary"""
        parts = []
      
        # Intent and sentiment
        intent_desc = intent.primary_intent.replace('_', ' ').lower()
        parts.append(f"Customer contacted regarding {intent_desc}.")
      
        # Sentiment with emotion
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
            for entity in important_entities[:4]:  # Increased to 4 entities
                label = entity['label'].replace('_', ' ').lower()
                entity_desc.append(f"{label}: {entity['text']}")
          
            if entity_desc:
                parts.append(f"Key details mentioned: {', '.join(entity_desc)}.")
      
        # Urgency indicator
        if sentiment.sentiment_label == 'urgent_negative':
            parts.append("Customer expressed urgency in resolution.")
      
        # First meaningful sentence for context
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if sentences and len(sentences[0].split()) >= 4:
            context = sentences[0][:120]
            if len(sentences[0]) > 120:
                context += "..."
            parts.append(f"Context: {context}")
      
        return " ".join(parts)

# ============================================================================
# IMPROVED AUDIO PROCESSOR (STT)
# ============================================================================
class AudioProcessor:
    """Audio transcription with improved STT"""
  
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
        """Transcribe audio using Groq Whisper with improved handling"""
        if not Config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not configured")
      
        try:
            client = LazyLoader.get_groq_client()
          
            # Create temp file with appropriate extension
            suffix = os.path.splitext(filename)[1].lower()
            if suffix not in Config.ALLOWED_AUDIO_EXTENSIONS:
                suffix = '.wav'
          
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
          
            try:
                # Open audio file
                with open(tmp_path, "rb") as audio_file:
                    # Improved transcription with better parameters
                    response = client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=audio_file,
                        response_format="verbose_json",
                        language="en",  # Specify language for better accuracy
                        temperature=0.0,  # Reduce randomness
                        prompt="IVR customer service call about banking, payments, transactions, accounts, cards, refunds, balance inquiry",  # Context prompt
                    )
              
                # Extract response
                if hasattr(response, 'text'):
                    transcript = response.text
                    language = getattr(response, 'language', 'en')
                    duration = getattr(response, 'duration', None)
                elif isinstance(response, dict):
                    transcript = response.get("text", "")
                    language = response.get("language", "en")
                    duration = response.get("duration")
                else:
                    transcript = ""
                    language = "en"
                    duration = None
              
                # Post-process transcript for better quality
                if transcript:
                    transcript = AudioProcessor._post_process_transcript(transcript)
              
                return {
                    "transcript": transcript or "",
                    "language": language,
                    "duration": duration,
                    "success": bool(transcript),
                }
              
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
              
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {str(e)}")
  
    @staticmethod
    def _post_process_transcript(transcript: str) -> str:
        """Post-process transcript for better quality"""
        if not transcript:
            return ""
      
        # Remove excessive whitespace
        transcript = re.sub(r'\s+', ' ', transcript)
      
        # Capitalize sentences
        sentences = re.split(r'([.!?]\s+)', transcript)
        processed = []
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i]
                if sentence:
                    # Capitalize first letter
                    sentence = sentence.strip()
                    if sentence:
                        sentence = sentence[0].upper() + sentence[1:]
                        processed.append(sentence)
              
                # Add punctuation back
                if i + 1 < len(sentences):
                    processed.append(sentences[i + 1])
      
        transcript = ''.join(processed)
      
        # Fix common IVR/call center artifacts
        replacements = [
            (r'\bi\s*\.\s*v\s*\.\s*r\b', 'IVR'),
            (r'\bu\s*\.\s*p\s*\.\s*i\b', 'UPI'),
            (r'\ba\s*\.\s*t\s*\.\s*m\b', 'ATM'),
            (r'\baccount\s+number\b', 'account number'),
            (r'\bphone\s+number\b', 'phone number'),
            (r'\btransaction\s+id\b', 'transaction ID'),
            (r'\bref\s*\.', 'ref'),
            (r'\btxn\s*\.', 'txn'),
        ]
      
        for pattern, replacement in replacements:
            transcript = re.sub(pattern, replacement, transcript, flags=re.IGNORECASE)
      
        return transcript.strip()

# ============================================================================
# OTHER COMPONENTS (Simplified)
# ============================================================================
class IntentDetector:
    """Intent detection"""
  
    @staticmethod
    def detect(text: str, entities: List[Dict]) -> IntentDetection:
        """Detect intent"""
        text_lower = text.lower()
        intents = {}
      
        if re.search(r'payment\s+(?:failed|failure)', text_lower):
            intents['PAYMENT_ISSUE'] = 0.9
        if re.search(r'transaction\s+(?:failed|failure)', text_lower):
            intents['PAYMENT_ISSUE'] = max(intents.get('PAYMENT_ISSUE', 0), 0.85)
        if re.search(r'refund', text_lower):
            intents['REFUND_REQUEST'] = 0.8
        if re.search(r'check\s+balance', text_lower):
            intents['BALANCE_INQUIRY'] = 0.7
        if re.search(r'card\s+(?:lost|stolen|blocked)', text_lower):
            intents['CARD_ISSUE'] = 0.85
        if re.search(r'login\s+(?:problem|issue|failed)', text_lower):
            intents['LOGIN_ISSUE'] = 0.7
      
        # Entity-based intent boosting
        for entity in entities:
            if entity['label'] == 'ISSUE_TYPE':
                entity_text = entity['text'].lower()
                if 'payment' in entity_text:
                    intents['PAYMENT_ISSUE'] = intents.get('PAYMENT_ISSUE', 0) + 0.05
                elif 'refund' in entity_text:
                    intents['REFUND_REQUEST'] = intents.get('REFUND_REQUEST', 0) + 0.05
      
        if intents:
            primary_intent = max(intents.items(), key=lambda x: x[1])
            return IntentDetection(
                primary_intent=primary_intent[0],
                confidence=min(primary_intent[1], 0.95),  # Cap confidence
                intents=intents
            )
      
        return IntentDetection(
            primary_intent='UNKNOWN',
            confidence=0.0,
            intents={}
        )

class ThreatDetector:
    """Threat and profanity detection"""
  
    @staticmethod
    def detect(text: str) -> ThreatProfanity:
        """Detect threats and profanity"""
        text_lower = text.lower()
      
        threat_terms = []
        profanity_terms = []
      
        threat_words = ['police', 'complaint', 'case', 'court', 'sue', 'legal', 'lawyer', 'file']
        profanity_words = ['stupid', 'idiot', 'fool', 'damn', 'hell', 'crap', 'bloody']
      
        for word in threat_words:
            if word in text_lower:
                threat_terms.append(word)
      
        for word in profanity_words:
            if word in text_lower:
                profanity_terms.append(word)
      
        return ThreatProfanity(
            threat_detected=len(threat_terms) > 0,
            profanity_detected=len(profanity_terms) > 0,
            threat_terms=threat_terms,
            profanity_terms=profanity_terms
        )

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
        score = 0.8  # Default good score for customer speech
      
        # Check for professionalism in agent-like speech
        agent_indicators = ['welcome', 'thank you for calling', 'how can i help', 'please provide']
        customer_indicators = ['my', 'i have', 'i want', 'problem', 'issue', 'help']
      
        agent_count = sum(1 for indicator in agent_indicators if indicator in text_lower)
        customer_count = sum(1 for indicator in customer_indicators if indicator in text_lower)
      
        if agent_count > customer_count:
            # This might be agent speech, check for compliance phrases
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
            suggestions=suggestions[:5],  # Top 5 suggestions
            next_best_action=suggestions[0] if suggestions else "Listen carefully and understand the issue",
            call_flow_action="Follow standard escalation protocol if needed"
        )

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
        entity_component = 100 * entity_weight * min(1.0, entity_count / 10.0)  # Cap at 10 entities
      
        score = sentiment_component + intent_component + compliance_component + entity_component
      
        # Penalties
        if has_threats:
            score *= 0.7  # 30% penalty for threats
      
        # Ensure within bounds and round
        final_score = max(0, min(100, int(round(score))))
      
        # Quality bands
        if final_score >= 85:
            logger.debug(f"High quality call: {final_score}")
        elif final_score >= 60:
            logger.debug(f"Medium quality call: {final_score}")
        else:
            logger.debug(f"Low quality call: {final_score}")
      
        return final_score

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================
class IVRPipeline:
    """Main IVR processing pipeline"""
  
    @staticmethod
    def process_text(text: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Process text through complete pipeline"""
        start_time = time.time()
      
        # Step 1: Clean
        cleaned_text = TextPreprocessor.clean(text)
        if not cleaned_text or len(cleaned_text.strip()) < 3:
            raise ValueError("Text too short")
      
        # Step 2: Detect language
        if not language:
            language = TextPreprocessor.detect_language(cleaned_text)
      
        # Step 3: Extract entities (FOCUS)
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

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================
app = FastAPI(
    title="IVR NER Analyzer API V5.0",
    description="Fixed NER accuracy + Improved STT + Enhanced sentiment analysis",
    version="5.0.0",
    docs_url="/docs",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 IVR NER Analyzer V5.0 starting up...")
    logger.info("✅ All critical NER accuracy fixes applied")
    logger.info("✅ Improved STT algorithm")
    logger.info("✅ Enhanced sentiment and emotion detection")

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/")
async def root():
    return {"status": "healthy", "version": "5.0.0"}

@app.post("/api/analyze/text", response_model=AnalyzeTextResponse)
async def analyze_text(
    request: AnalyzeTextRequest,
    db: Session = Depends(get_db)
):
    """Analyze text with fixed NER accuracy"""
    try:
        result = IVRPipeline.process_text(request.text)
      
        # Save to database
        from sqlalchemy.orm import Session
        record = CallAnalysis(
            input_type="text",
            transcript=request.text[:2000],
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
    """Transcribe and analyze audio with improved STT"""
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
        record = CallAnalysis(
            input_type="audio",
            transcript=transcription["transcript"][:2000],
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

@app.post("/api/test/ner")
async def test_ner(text: str = Query(..., description="Text to test NER")):
    """Test NER accuracy with the fixes"""
    try:
        entities = NEREngine.extract_entities(text)
      
        # Debug info for phone vs account classification
        debug_info = []
        for entity in entities:
            if entity['label'] in ['PHONE_NUMBER', 'ACCOUNT_NUMBER']:
                debug_info.append({
                    'text': entity['text'],
                    'label': entity['label'],
                    'confidence': entity['confidence'],
                    'digits': ''.join(filter(str.isdigit, entity['text'])),
                })
      
        return {
            "text": text,
            "entities": entities,
            "debug_info": debug_info,
            "statistics": {
                "total_entities": len(entities),
                "phone_numbers": sum(1 for e in entities if e['label'] == 'PHONE_NUMBER'),
                "account_numbers": sum(1 for e in entities if e['label'] == 'ACCOUNT_NUMBER'),
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