# ─────────────────────────────────────────────────────────────────────────────
#  Layer 2 — Production ByteTrack Configuration/config.py
# ─────────────────────────────────────────────────────────────────────────────

# ── Detection confidence pools ────────────────────────────────────────────────
# FIX: All thresholds lowered to match the new Layer1 CONF_THRESH=0.20.
# On CPU, non-person objects arrive at conf 0.20–0.30.
# Original thresholds (HIGH=0.35, LOW=0.15, NEW=0.25) meant handbag detections
# at conf=0.22 were in the "low" pool but never birthed (NEW_TRACK_THRESH=0.25).

TRACK_HIGH_THRESH   = 0.25   # was 0.35 — objects at 0.25+ go into high pool
TRACK_LOW_THRESH    = 0.10   # was 0.15 — objects at 0.10+ go into low pool
TRACK_MATCH_THRESH  = 0.20   # was 0.25 — IoU match threshold (kept low for small objects)
TRACK_SECOND_THRESH = 0.10   # was 0.15
NEW_TRACK_THRESH    = 0.20   # was 0.25 — CRITICAL: objects below this are NEVER born
                              #            a handbag at conf=0.22 was silently dropped

# ── Track lifecycle ───────────────────────────────────────────────────────────
MAX_TIME_LOST       = 90
MIN_CONFIRM_FRAMES  = 2
MIN_TRACK_FRAMES    = 2

# ── Occlusion / re-identification window ─────────────────────────────────────
PREDICT_FRAMES      = 10

# ── Velocity & trajectory ─────────────────────────────────────────────────────
VELOCITY_ALPHA      = 0.4
TRAIL_MAXLEN        = 60

# ── Class-consistency lock ────────────────────────────────────────────────────
LOCK_CLASS_ON_CONFIRM = True

# ── Matching guards ───────────────────────────────────────────────────────────
MAX_MATCH_DISTANCE_PX = 280

# ── Ghost-throw cooldown ──────────────────────────────────────────────────────
GHOST_COOLDOWN_FRAMES = 40

# ── Visualisation ─────────────────────────────────────────────────────────────
TRACK_ID_COLOR      = (255, 255, 0)
TRAIL_COLOR         = (200, 200, 0)
TRAIL_LENGTH        = 30
SHOW_TRAILS         = True
SHOW_VELOCITY       = False
SHOW_TRACK_STATUS   = False

# ─────────────────────────────────────────────────────────────────────────────
#  Failure Detection Parameters
# ─────────────────────────────────────────────────────────────────────────────

KALMAN_PROCESS_NOISE      = 1e-2
KALMAN_MEASUREMENT_NOISE  = 1e-1

UNCERTAINTY_THRESHOLD     = 0.60
MOTION_ERROR_WEIGHT       = 0.40
MISSED_FRAMES_WEIGHT      = 0.40
SUDDEN_LOSS_WEIGHT        = 0.20

MISSED_FRAMES_SPIKE       = 3
MOTION_ERROR_THRESH_PX    = 30.0

# ─────────────────────────────────────────────────────────────────────────────
#  ROI Recovery Parameters
# ─────────────────────────────────────────────────────────────────────────────

ROI_EXPANSION_FACTOR      = 1.8
ROI_VELOCITY_SCALE        = 3.0
ROI_MIN_SIZE              = 64
ROI_MAX_SIZE              = 640
ROI_MATCH_IOU_THRESH      = 0.20
ROI_MATCH_DIST_THRESH     = 80.0
MAX_RECOVERABLE_FRAMES    = 10
MAX_TRACK_HISTORY         = 60

# ─────────────────────────────────────────────────────────────────────────────
#  ReID Appearance Embedding Parameters
# ─────────────────────────────────────────────────────────────────────────────

REID_WEIGHT             = 0.25
MOTION_WEIGHT           = 0.70
IOU_WEIGHT              = 0.20

REID_MAX_COSINE_DIST    = 0.75
REID_FALLBACK_IOU       = 0.30
REID_ENABLED            = True
REID_MIN_CROP_SIZE      = 20