# Layer4/config.py

# ── Input (must match Layer 3) ────────────────────────────
SEQUENCE_LENGTH = 24
FEATURE_DIM     = 9

# ── Regularization ────────────────────────────────────────
DROPPATH_RATE   = 0.1      # stochastic depth — keep low, already well-regularized
LABEL_SMOOTHING = 0.05     # reduced from 0.1 — don't over-smooth with small data

# ── Model architecture ────────────────────────────────────
D_MODEL         = 32
NHEAD           = 4
NUM_LAYERS      = 2
DIM_FEEDFORWARD = 64
DROPOUT         = 0.2      # reduced from 0.3 — we are in underfitting territory

# ── Training ──────────────────────────────────────────────
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 16       # smaller → more gradient steps per epoch with 139 samples
EPOCHS          = 80       # was 30 — loss still descending at epoch 30
TRAIN_SPLIT     = 0.8

# ── Class imbalance ───────────────────────────────────────
# 118 normal / 21 dump = 5.6x imbalance
# Trainer recomputes this automatically from actual data counts.
# This is only the fallback default.
POS_WEIGHT      = 5.6

# ── Inference ─────────────────────────────────────────────
# Lowered from 0.5 — with class imbalance the model under-predicts positives.
# Prefer higher recall (catch more dumps) over precision.
DUMP_THRESHOLD  = 0.4
INFER_EVERY_N   = 5

# ── Paths ─────────────────────────────────────────────────
DATA_DIR        = "data/sequences"
MODEL_SAVE_PATH = "Layer4/weights/model.pt"