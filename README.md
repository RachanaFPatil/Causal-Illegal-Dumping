# A Causal Event-Driven Agentic AI Framework for Offline Illegal Dumping Detection in Video

The project is a modular surveillance framework that detects illegal dumping in public spaces using computer vision, temporal reasoning, and automated enforcement workflows.

The system processes live CCTV streams or recorded footage, tracks people and objects across frames, determines whether disposal behaviour is legal or illegal, and automatically generates a challan (penalty notice) with evidence extraction and email delivery.

Designed as a production-oriented multi-layer pipeline with independent perception, tracking, reasoning, and enforcement modules.

---

## ✨ Features

- Real-time illegal dumping detection
- Multi-object tracking with identity preservation
- Trash throw and slow-drop detection
- Bin-aware disposal reasoning
- Temporal memory & behavioural analysis
- ROI recovery for lost tracks
- Vehicle plate OCR & evidence extraction
- Automated PDF challan generation
- Email notification & escalation workflow
- Modular layer-based architecture

---

# 🏗️ System Architecture

```text
Video Stream / CCTV Feed
            │
            ▼
┌──────────────────────────────────────────────┐
│ Layer 1 — Perception & Detection             │
│ RT-DETR + Trash Detection + Bin Detection    │
└──────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│ Layer 2 — Multi-Object Tracking              │
│ ByteTrack + ReID + ROI Recovery              │
└──────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│ Layer 3 — Memory & Feature Extraction        │
│ Sliding Window + Bin Interaction Features    │
└──────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│ Layer 4 — Dumping Inference                  │
│ Context-Aware Temporal Evaluation            │
└──────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│ Layer 5 — Agentic Perception Controller      │
│ Intent Analysis + Behaviour Validation       │
└──────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│ Enforcement Subsystem                        │
│ OCR + PDF Challan + Email Delivery           │
└──────────────────────────────────────────────┘
```

---

# 🧠 Core Pipeline

## Layer 1 — Perception & Detection
- RT-DETR based object detection
- Slow-drop and fast-throw trash detection
- Dedicated trash-bin detector
- Frame calibration & preprocessing

## Layer 2 — Multi-Object Tracking
- Pure NumPy ByteTrack implementation
- ReID embeddings for identity consistency
- ROI recovery for uncertain tracks
- Stable trash and bin tracking

## Layer 3 — Memory & Feature Extraction
- Sliding-window temporal memory
- Trash↔bin interaction modelling
- Behavioural feature vector extraction

## Layer 4 — Dumping Inference
Classifies events into:
- `legal_disposal`
- `illegal_dumping`
- `pending`

Uses:
- release behaviour
- bin proximity
- object trajectory
- post-release motion analysis

## Layer 5 — Agentic Perception Controller
Final reasoning layer that validates or overrides Layer 4 decisions using:
- motion coupling
- possession analysis
- divergence detection
- trajectory intent scoring
- confidence-based arbitration

---

# 🚨 Enforcement Pipeline

## Enhancer
- Super-resolution & sharpening
- Evidence extraction
- Vehicle plate OCR

## Penalty Manager
- SQLite-based challan management
- Automated PDF generation
- Dynamic UPI QR integration
- Escalation handling

## Delivery Agent
- Email notification system
- Escalation reminders
- Retry mechanism for failed sends

---

# 🚀 Setup

## Prerequisites

- Python 3.10+
- OpenCV compatible environment
- CUDA GPU recommended

---

## Installation

```bash
git clone https://github.com/Poojitha20-B/Illegal_dumping.git

cd Illegal_dumping

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

pip install -r requirements.txt
```

---

# 📦 Model Weights

Place the required weights inside the `weights/` directory.

```text
weights/
├── rtdetr-l.pt
└── trash_bin_detector.pt
```

---

# ▶️ Running the Project

### Process video input
```bash
python run_pipeline.py --source test2.mp4
```
### Save processed output video
```bash
python run_pipeline.py --source test2.mp4 --save
```
### Run with custom violation location
```bash
python run_pipeline.py --source test2.mp4 --save --location "MG Road, Bengaluru"
```

# ⚡ Penalty Escalation Simulation

Simulate overdue challan escalation directly from terminal:

```bash
python -c "
from penalty_manager import PenaltyManager
pm = PenaltyManager()
pm.simulate_days_passed('BBMP-VH-KA05KK5546-1DE0619F', 10)
"
```

This applies escalation rules and updates the challan amount based on overdue duration.

---

---

# 📂 Project Structure

```text
Illegal_dumping/
│
├── run_pipeline.py
├── enhancer.py
├── penalty_manager.py
├── delivery_agent.py
│
├── Layer1/
├── Layer2/
├── Layer3/
├── Layer4/
├── Layer5/
│
└── weights/
```

---

# 🧪 Tech Stack

| Component | Technology |
|---|---|
| Detection | RT-DETR |
| Tracking | ByteTrack |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| ReID | TorchReID |
| OCR | EasyOCR |
| Database | SQLite |
| PDF Generation | ReportLab |
| Scheduling | APScheduler |

---

# 🌍 Applications

- Smart city surveillance
- Municipal sanitation monitoring
- Public space monitoring
- Railway & bus station surveillance
- Campus & gated community monitoring

---

# ⚠️ Note

Detection accuracy and enforcement reliability may vary depending on:
- video quality
- lighting conditions
- camera angle
- object visibility
- environmental occlusions
