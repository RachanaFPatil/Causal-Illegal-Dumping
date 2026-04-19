# ─────────────────────────────────────────
#  Layer 3 — Feature Extraction Config
# ─────────────────────────────────────────

# Bin region scaling factors
# bin_region = shrink the bin bbox by this factor (inner zone = actually inside bin)
BIN_REGION_SHRINK   = 0.6
# bin_zone   = expand the bin bbox by this factor (proximity zone around bin)
BIN_ZONE_EXPAND     = 1.5

# Maximum distance (pixels) between a trash centroid and a bin centroid
# to be considered associated.  Trash beyond this distance is ignored.
BIN_ASSOCIATION_MAX_DIST = 400   # px — tune to your camera's FOV

# Temporal sliding window — how many frames of feature history to keep per pair
SEQUENCE_WINDOW = 30             # frames  (~1 sec at 30fps)

# Entry event signal thresholds
# downward velocity threshold (pixels/frame) to count as "moving toward bin"
ENTRY_VY_THRESHOLD  = 1.5
# minimum frames inside bin_region to confirm a "stay" event
ENTRY_MIN_FRAMES    = 4

# Debug visualisation colours (BGR)
DBG_REGION_COLOR    = (0,  200,  0)    # green  — inner bin_region
DBG_ZONE_COLOR      = (0,  200, 255)   # yellow — outer bin_zone
DBG_TRAIL_COLOR     = (200, 80, 255)   # purple — trash trajectory
DBG_LINE_COLOR      = (255, 140,   0)  # orange — trash→bin connector
DBG_SCORE_COLOR     = (255, 255, 255)  # white  — entry event score text