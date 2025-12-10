# 🚀 IVR NER Analyzer – AI-Powered Backend  
### **FastAPI + Groq Whisper Large-v3 + SpaCy + BERT + Rule-Based AI + RL Call Flows + Analytics**

This backend provides a full AI pipeline for IVR call analysis.  
It supports **English + Hindi**, handles both **audio and text**, and performs:

- 🎧 **Audio Transcription (Groq Whisper Large-v3)**
- 🧠 **Hybrid NER (SpaCy + BERT + Rule-based)**
- 📌 **Account Number Detection (8–12 digits)**
- 📑 **Issue Type Extraction (EN + HI)**
- 😊 **Sentiment + Emotion Analysis**
- 🎯 **Intent Detection**
- 🛡 **Threat + Profanity Detection**
- 📚 **Compliance Check (EN + HI)**
- 🧩 **Relationship Extraction**
- 📝 **Automatic Call Summary**
- 🤖 **Reinforcement Learning Call Flow Engine**
- 🧠 **Real-Time Agent Assist Suggestions**
- 📊 **Analytics Dashboard API**
- 💾 **SQLite Call Storage**

---

# 📁 Project Structure

/project
│── ivr_backend.py # Full FastAPI backend code
│── requirements.txt # Python dependency list
│── .env # Groq API key
│── ivr_ner.db # SQLite DB (auto-created)
│── README.md # Documentation file

yaml
Copy code

---

# 🔧 Features Breakdown

## ✔ Audio Processing
- Whisper Large-v3 via Groq API  
- Automatic language identification  
- Cleans transcripts with noise removal + number conversion  

## ✔ Hybrid NER — Production Quality
- SpaCy `en_core_web_sm`
- BERT NER (`dslim/bert-base-NER`)
- Rule-based upgrade (ACCOUNT_ID, ISSUE_TYPE)
- Hindi NER cleanup (remove SpaCy noise)
- Spoken number → digit conversion (EN + HI)

## ✔ Sentiment & Emotion Detection
- Lexicon model (EN + HI)
- Smart Hindi booster for gratitude/anger cues  
- Emotion categories: anger, sadness, fear, joy, surprise  

## ✔ Intent Detection
- Keyword probability scoring  
- Auto-strengthening based on ISSUE_TYPE  
- Supports bilingual intents  

## ✔ Compliance Engine
Checks for:
- Greeting  
- Identity verification  
- Mandatory disclosure  
- Closing statements  

## ✔ Risk Detection
- Profanity  
- Threat phrases (legal/police/court)  

## ✔ Reinforcement Learning (RL)
- Epsilon-greedy model  
- Learns best next action per intent  
- Supports feedback from `/api/flow-feedback`  

## ✔ Analytics Dashboard
Aggregates:
- Languages  
- Intents  
- Issue types  
- Sentiment  
- Risk calls  
- Input type distribution  

---

# 🧪 API Endpoints

## **1️⃣ POST /api/transcribe-audio**
Input: Audio file  
Output: Transcript + Language + Duration

## **2️⃣ POST /api/analyze-text**
Input: Raw text  
Output:  
- NER  
- Intent  
- Sentiment + Emotion  
- Compliance  
- Threat detection  
- Relationships  
- Summary  
- Agent Assist  
- Call Score  

## **3️⃣ POST /api/analyze-audio**
Audio → Transcription → Full AI pipeline

## **4️⃣ GET /api/history**
Returns past call analyses from SQLite

## **5️⃣ GET /api/analytics-dashboard**
Full aggregated call analytics

## **6️⃣ POST /api/flow-feedback**
Updates RL model with reward signal

## **7️⃣ GET /**
Health check

---

# 📦 Installation & Execution Guide

## **1️⃣ Clone Repo**
```bash
git clone https://github.com/your-username/ivr-ner-backend.git
cd ivr-ner-backend
2️⃣ Create Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
3️⃣ Install Requirements
Create a file:

requirements.txt
nginx
Copy code
fastapi
uvicorn
sqlalchemy
groq
python-dotenv
transformers
torch
spacy
langdetect
pyspellchecker
pydantic
Install packages:

bash
Copy code
pip install -r requirements.txt
4️⃣ Download SpaCy Model
bash
Copy code
python -m spacy download en_core_web_sm
5️⃣ Configure .env
Create a file:

.env
ini
Copy code
GROQ_API_KEY=your_api_key_here
6️⃣ Run the Server
bash
Copy code
uvicorn ivr_backend:app --host 0.0.0.0 --port 8000 --reload
Server URL:

👉 http://localhost:8000
👉 Docs: http://localhost:8000/docs
👉 Redoc: http://localhost:8000/redoc

🔥 Example Request (Analyze Text)
Request:
json
Copy code
{
  "text": "Hello I am facing a payment failure. Money was deducted twice and my account number is 987654321."
}
Response (shortened):
json
Copy code
{
  "language": "en",
  "entities": [...],
  "intents": {...},
  "sentiment": {...},
  "summary": "Primary intent: payment issue...",
  "call_score": 84
}
🚀 Deployment Ready
For production:
bash
Copy code
uvicorn ivr_backend:app --host 0.0.0.0 --port 8000
Works on:

Render