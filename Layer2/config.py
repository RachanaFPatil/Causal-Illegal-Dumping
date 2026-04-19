# ─────────────────────────────────────────────────────────────────────────────
#  Layer 2 — Production ByteTrack Configuration/config.py
# ─────────────────────────────────────────────────────────────────────────────

# ── Detection confidence pools ────────────────────────────────────────────────
TRACK_HIGH_THRESH   = 0.35
TRACK_LOW_THRESH    = 0.15
TRACK_MATCH_THRESH  = 0.25
TRACK_SECOND_THRESH = 0.15
NEW_TRACK_THRESH    = 0.25

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
#  NEW — ReID Appearance Embedding Parameters
# ─────────────────────────────────────────────────────────────────────────────

# Weight blend for the combined matching cost:
#   total_cost = REID_WEIGHT * appearance_dist
#              + MOTION_WEIGHT * (1 - iou)
#              + IOU_WEIGHT * (1 - iou)          [redundant but explicit]
# Must sum to 1.0
REID_WEIGHT             = 0.25   # appearance embedding (primary for viewpoint)
MOTION_WEIGHT           = 0.70   # IoU + Kalman motion (secondary)
IOU_WEIGHT              = 0.20   # raw IoU (weak cue, tiebreaker)

# Cosine distance threshold: above this → appearance too different → no match
REID_MAX_COSINE_DIST    = 0.75   # 0=identical 1=opposite; 0.55 is permissive

# Min IoU to still allow a match even if embedding missing (fallback mode)
REID_FALLBACK_IOU       = 0.30

# Whether to use ReID at all (set False to disable and revert to IoU-only)
REID_ENABLED            = True

# Minimum crop size (pixels) to bother extracting an embedding
REID_MIN_CROP_SIZE      = 20