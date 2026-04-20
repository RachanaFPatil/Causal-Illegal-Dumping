# ─────────────────────────────────────────────────────────────────────────────
#  Layer 4 / config.py  —  Temporal Intent Evaluator
# ─────────────────────────────────────────────────────────────────────────────

# ── Throw / hold thresholds ───────────────────────────────────────────────────
MIN_HELD_FRAMES       = 3     # frames object must have been held before release counts
MIN_POST_RELEASE      = 4     # frames of independent motion required after release
MIN_SEQ_LEN           = 6     # skip evaluation if sequence shorter than this

# ── Velocity thresholds (px / frame) ─────────────────────────────────────────
MOTION_VEL_THRESHOLD  = 2.5   # minimum speed after release to count as a throw
THROW_VEL_THRESHOLD   = 5.0   # speed that yields full release confidence score

# ── Bin proximity ─────────────────────────────────────────────────────────────
BIN_NEAR_PX           = 120   # object within this many px of bin → legal_disposal

# ── Confidence weights (must sum to 1.0) ─────────────────────────────────────
W_HELD        = 0.20
W_RELEASE     = 0.35
W_MOTION      = 0.25
W_BIN_SPATIAL = 0.20

# ── State management ─────────────────────────────────────────────────────────
MAX_RESULT_AGE        = 60    # frames before an idle pair state is purged
