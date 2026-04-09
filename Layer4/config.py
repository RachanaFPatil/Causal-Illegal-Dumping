# Layer4/config.py

# ── Input (must match Layer 3) ────────────────────────────
SEQUENCE_LENGTH = 24
FEATURE_DIM     = 9

# ── Regularization ────────────────────────────────────────
DROPPATH_RATE   = 0.1
LABEL_SMOOTHING = 0.05

# ── Model architecture ────────────────────────────────────
D_MODEL         = 32
NHEAD           = 4
NUM_LAYERS      = 2
DIM_FEEDFORWARD = 64
DROPOUT         = 0.2

# ── Training ──────────────────────────────────────────────
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 16
EPOCHS          = 80
TRAIN_SPLIT     = 0.8

# ── Class imbalance ───────────────────────────────────────
POS_WEIGHT      = 5.6

# ── Inference ─────────────────────────────────────────────
DUMP_THRESHOLD  = 0.4
INFER_EVERY_N   = 5

# ── Paths ─────────────────────────────────────────────────
DATA_DIR        = "data/sequences"
MODEL_SAVE_PATH = "Layer4/weights/model.pt"

# ── NEW: Hybrid Agent ─────────────────────────────────────
# Blend weights for: transformer, ML model, rule-based score
HYBRID_ALPHA      = 0.50   # transformer weight
HYBRID_BETA       = 0.30   # ML model weight
HYBRID_GAMMA      = 0.20   # rule-based weight
HYBRID_THRESHOLD  = 0.45   # final blended score threshold for alert

# ── NEW: ML Scoring Model ─────────────────────────────────
ML_MODEL_PATH     = "Layer4/weights/ml_model.pkl"

# ── NEW: Logging ──────────────────────────────────────────
LOG_DIR           = "logs"
LOG_FLUSH_EVERY   = 30     # frames between disk flushes

# ── NEW: Inference pipeline flags ─────────────────────────
ENABLE_ML_MODEL   = True   # set False to skip ML scoring (rule + transformer only)
ENABLE_LOGGING    = True   # set False to disable JSON logging
ENABLE_HYBRID     = True   # set False to use transformer-only decision (legacy mode)