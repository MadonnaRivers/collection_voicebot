# Collection Voicebot - Fast Deployment Guide

## 1) Server prep

```bash
sudo apt update && sudo apt install -y git python3 python3-venv ffmpeg
```

## 2) Clone repo

```bash
cd /opt
sudo git clone https://github.com/MadonnaRivers/collection_voicebot.git
sudo chown -R $USER:$USER /opt/collection_voicebot
cd /opt/collection_voicebot
```

## 3) Create venv and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Configure environment

Create `.env` in repo root:

```env
OPENAI_API_KEY=...
SARVAM_API_KEY=...
SARVAM_VOICE=simran

PLIVO_AUTH_ID=...
PLIVO_AUTH_TOKEN=...
PLIVO_PHONE_NUMBER=...

NGROK_URL=https://<your-public-url>
PORT=5050

CALL_SUMMARY_WEBHOOK_URL=https://web-n8n.easyhomefinance.in/webhook/push_data
AUDIO_TRANSCRIPT_WEBHOOK_URL=https://web-n8n.easyhomefinance.in/webhook/audio_and_transcripts
```

## 5) Start API server (recommended for curl /make-call)

```bash
source .venv/bin/activate
uvicorn routes:app --host 0.0.0.0 --port 5050
```

### Run in background

```bash
nohup .venv/bin/uvicorn routes:app --host 0.0.0.0 --port 5050 > app.log 2>&1 &
```

## 6) Health check

```bash
curl http://<server-ip>:5050/health
```

## 7) Trigger outbound call via curl

Primary curl used for triggering calls:

```bash
curl --location 'https://d559-103-58-152-59.ngrok-free.app/make-call' \
--header 'Content-Type: application/json' \
--data '{
  "phone_number":     "+919876543210",
  "customer_name":    "Rahul Sharma",
  "loan_id":          "EH12345",
  "emi_overdue_amt":  "8,500",
  "emi_overdue_date": "5 March 2026",
  "min_partial":      "1,500",
  "payment_deadline": ""
}'
```

Generic template:

```bash
curl --location 'https://<your-public-url>/make-call' \
--header 'Content-Type: application/json' \
--data '{
  "phone_number": "+917977365303",
  "customer_name": "KJ",
  "loan_id": "EH12345",
  "emi_overdue_amt": "12,000",
  "emi_overdue_date": "5 March 2026",
  "min_partial": "1,500",
  "payment_deadline": "12 May 2026"
}'
```

## Important note about `main.py`

- Current `main.py` requires a phone number argument and exits if missing.
- So this command is for direct dial startup, not API-only mode:

```bash
python main.py +91XXXXXXXXXX
```

- If you need to run curl on `/make-call`, use:

```bash
uvicorn routes:app --host 0.0.0.0 --port 5050
```

