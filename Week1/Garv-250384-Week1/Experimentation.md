## Preprocessing: 
**Simple box blur (3x3 average):**

Tried this first to reduce noise before gradient computation. Works, but tends to blur edges themselves along with the noise since every pixel gets equal weight.

**Gaussian blur (5x5):**

Weights the center more heavily, so noise gets smoothed without destroying edge sharpness as much. Standard preprocessing step before gradient-based edge detection .

## Gradient Operators

**Sobel (3x3):**

Classic gradient kernel.

**Scharr (3x3):**
Better rotational symmetry, so gradient direction estimates are more accurate. Noticeably better angle accuracy when computing `arctan2(g_y, g_x)` for edge orientation.

## Thresholding

First attempt — just clip anything below the 90th percentile of gradient magnitude. Gave noisy, patchy edges since it doesn't account for edge continuity, a strong edge can have a weak segment that gets discarded, breaking the line.

**Double threshold (no hysteresis):**

Tried using two thresholds but without connectivity tracing between them, pixels between `low` and `high` just get dropped. Cleaner than single threshold but edges still have visible gaps wherever gradient dips mid-edge.

**Non-max suppression:**

Added proper NMS, the edges were clearer, reduced noise and patchy edges
