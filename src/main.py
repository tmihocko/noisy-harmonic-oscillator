"""
Damped Harmonic Oscillator — Interactive Visualization
mx'' + cx' + kx = 0

Drag the slider to explore underdamped, critically damped, and overdamped regimes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec

# ── Parameters ────────────────────────────────────────────────────────────────
M = 1.0   # mass
K = 4.0   # spring constant
C_CRIT = 2 * np.sqrt(M * K)   # critical damping: c* = 2√(mk)
T = np.linspace(0, 12, 2000)

# ── ODE solver (simple Euler — fast enough for interactive use) ────────────────
def solve(c, t):
    dt = t[1] - t[0]
    x, v = 1.0, 0.0
    xs = np.empty(len(t))
    xs[0] = x
    for i in range(1, len(t)):
        a = -(c / M) * v - (K / M) * x
        v += a * dt
        x += v * dt
        xs[i] = x
    return xs

# Pre-compute critical reference
x_crit = solve(C_CRIT, T)

# ── Regime helpers ─────────────────────────────────────────────────────────────
def regime(c):
    D = c**2 - 4 * M * K
    if abs(D) < 0.3:
        return "critically damped", "#3B6D11", "#EAF3DE"
    elif D < 0:
        return "underdamped", "#185FA5", "#E6F1FB"
    else:
        return "overdamped", "#993C1D", "#FAECE7"

DESCRIPTIONS = {
    "underdamped":    "Underdamped (D < 0)\nOscillates with decaying amplitude.\nSystem overshoots equilibrium — bouncy.",
    "critically damped": "Critically Damped (D = 0)\nFastest return with no oscillation.\nThe engineering 'Goldilocks' zone.",
    "overdamped":     "Overdamped (D > 0)\nNo oscillation — sluggish creep\nback to equilibrium.",
}

# ── Layout ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 6.5), facecolor="#fafafa")
gs = GridSpec(3, 1, figure=fig, height_ratios=[5, 0.6, 0.9],
              hspace=0.05, left=0.1, right=0.92, top=0.92, bottom=0.08)

ax   = fig.add_subplot(gs[0])
ax_s = fig.add_subplot(gs[1])
ax_i = fig.add_subplot(gs[2])   # info strip

ax_i.axis("off")
ax_s.set_facecolor("#fafafa")
for spine in ax_s.spines.values():
    spine.set_visible(False)
ax_s.set_yticks([])

# ── Main axes setup ────────────────────────────────────────────────────────────
ax.set_xlim(0, 12)
ax.set_ylim(-1.45, 1.45)
ax.set_xlabel("Time (s)", fontsize=11, color="#555")
ax.set_ylabel("Displacement  x(t)", fontsize=11, color="#555")
ax.set_facecolor("#fafafa")
ax.axhline(0, color="#cccccc", lw=1, zorder=0)
ax.tick_params(colors="#777")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color("#dddddd")

# ── Plot elements ──────────────────────────────────────────────────────────────
c0 = 3.0
x0 = solve(c0, T)
label0, color0, _ = regime(c0)

line_crit, = ax.plot(T, x_crit, color="#639922", lw=1.5,
                     linestyle="--", alpha=0.7, label="Critical reference (c*)")
line_cur,  = ax.plot(T, x0,    color=color0,   lw=2.5, label="Current response")

# Discriminant shading (first half of x-axis as indicator strip)
fill_bg = ax.axhspan(-1.45, 1.45, alpha=0.0)   # placeholder, redrawn on update

# Legend
leg = ax.legend(loc="upper right", framealpha=0.85, fontsize=10,
                facecolor="white", edgecolor="#dddddd")

# Title
title = ax.set_title(
    f"Damped Harmonic Oscillator   |   m={M:.0f}, k={K:.0f}",
    fontsize=13, color="#333", pad=10, loc="left"
)

# Info strip
badge_ax = ax_i
badge_text = badge_ax.text(
    0.02, 0.5, "", transform=badge_ax.transAxes,
    fontsize=11, va="center", ha="left",
    bbox=dict(boxstyle="round,pad=0.4", fc="#E6F1FB", ec="#185FA5", lw=1.2),
    color="#185FA5", fontweight="bold"
)
disc_text = badge_ax.text(
    0.98, 0.5, "", transform=badge_ax.transAxes,
    fontsize=11, va="center", ha="right", color="#555", family="monospace"
)

# ── Slider ─────────────────────────────────────────────────────────────────────
slider = Slider(ax_s, "  c  ", 0.0, 12.0, valinit=c0, valstep=0.05,
                color="#378ADD", initcolor="none")
slider.label.set_fontsize(11)
slider.label.set_color("#555")
slider.valtext.set_fontsize(11)
slider.valtext.set_color("#333")

# Vertical marker for c*
ax_s.axvline(C_CRIT, color="#639922", lw=1.5, linestyle="--")
ax_s.text(C_CRIT, 1.5, "c*", transform=ax_s.get_xaxis_transform(),
          fontsize=9, color="#639922", ha="center")

# ── Update callback ────────────────────────────────────────────────────────────
def update(val):
    c = slider.val
    D = c**2 - 4 * M * K
    label, color, bg = regime(c)
    xs = solve(c, T)

    line_cur.set_ydata(xs)
    line_cur.set_color(color)

    badge_text.set_text(f"  {label.title()}  ")
    badge_text.get_bbox_patch().set_facecolor(bg)
    badge_text.get_bbox_patch().set_edgecolor(color)
    badge_text.set_color(color)

    disc_text.set_text(f"D = {D:+.2f}   |   c = {c:.2f}   |   c* = {C_CRIT:.2f}")

    fig.canvas.draw_idle()

slider.on_changed(update)
update(c0)   # init display

plt.savefig("/mnt/user-data/outputs/damped_oscillator_preview.png",
            dpi=140, bbox_inches="tight", facecolor="#fafafa")

print("Interactive window: drag the slider to explore all three regimes.")
plt.show()