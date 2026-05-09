import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons, Button
from scipy.stats import levy_stable


# Fixed simulation settings
MASS = 1.0
DT = 0.01
T_END = 40.0
N = int(T_END / DT)
T_GRID = np.linspace(0, T_END, N) # time array used for x(t) plot x-axis

# Default slider values shown on startup
DEF_C = 0.20 # damping coefficient c [kg/s]
DEF_K = 4.00 # spring constant k
DEF_SCALE = 0.30 # noise intensity sigma
DEF_ALPHA = 1.50 # Levy stability index (alpha=2 is Gaussian)
DEF_BETA = 0.00 # Levy skewness (beta=0 is symmetric)

BG = "#0d1117" # figure background
PANEL = "#161b22" # axes background
GRID_COL = "#21262d" # gridlines and spines
SL_BG = "#21262d" # slider track
SL_COL = "#58a6ff" # slider handle / active colour
TICK_COL = "#8b949e" # secondary text (ticks, labels)
TEXT_COL = "#e6edf3" # primary text
C_X = "#58a6ff" # x(t) line colour
C_PP = "#f78166" # phase portrait line colour

# Advances the oscillator one step using Euler-Maruyama:
#	x_{i+1} = x_i + v_i * dt
#	v_{i+1} = v_i + drift * dt + dW/m
# where drift = -(c/m)*v - (k/m)*x (spring force + damping)
def euler_maruyama_step(x_i, v_i, c, k, dt, dW):
	# drift = a = F/M = (-cv - kx)/m
	drift  = -(c / MASS) * v_i - (k / MASS) * x_i
	
	x_next = x_i + (v_i * dt)
	v_next = v_i + (drift * dt) + (dW / MASS)

	return x_next, v_next


# Creates full x(t) and v(t) arrays.
def simulate(c, k, alpha, beta, scale, noise_type="levy"):
	x = np.zeros(N)
	v = np.zeros(N)
	x[0] = 0.5
	v[0] = 0.0

	if noise_type == "gaussian":
		noise_arr = np.random.normal(0, scale * np.sqrt(DT), N)
	elif noise_type == "levy":
		noise_arr = levy_stable.rvs(alpha, beta, scale=scale * DT**(1/alpha), size=N)

	for i in range(N - 1):		
		x[i + 1], v[i + 1] = euler_maruyama_step(x[i], v[i], c, k, DT, noise_arr[i])

	return x, v


# Computes three quantities shown in the info panel:
#	f0: natural frequency in Hz: sqrt(k/m) / (2*pi)
#	c_crit: critical damping: 2 * sqrt(k*m)
#	regime: whether the system is underdamped, overdamped, or critical
def compute_physical_quantities(c, k):
	f0 = np.sqrt(k / MASS) / (2 * np.pi)
	c_crit = 2.0 * np.sqrt(k * MASS)

	if c < c_crit:
		regime = "underdamped"
	elif c > c_crit:
		regime = "overdamped"
	else:
		regime = "critically damped"

	return f0, c_crit, regime


# Applies dark theme styling to a single axes: background, ticks, spines, grid.
def apply_axis_style(ax):
	ax.set_facecolor(PANEL)
	ax.tick_params(colors=TICK_COL, labelsize=9)
	for spine in ax.spines.values():
		spine.set_edgecolor(GRID_COL)
	ax.grid(True, color=GRID_COL, linewidth=0.6)


# Creates the figure and the two plot panels (x(t) left, phase portrait right).
# Returns the figure and both axes so draw_initial_plots can populate them.
def build_figure():
	fig = plt.figure(figsize=(14, 6), facecolor=BG)
	fig.suptitle("Noisy Harmonic Oscillator", color=TEXT_COL, fontsize=14, fontweight="bold", y=0.98)

	# GridSpec positions the two plot panels in the left 70% of the figure
	gs = gridspec.GridSpec(1, 2, left=0.06, right=0.70, top=0.91, bottom=0.12, wspace=0.30)

	ax_x = fig.add_subplot(gs[0, 0]) # x(t) panel
	ax_pp = fig.add_subplot(gs[0, 1]) # phase portrait panel

	apply_axis_style(ax_x)
	apply_axis_style(ax_pp)

	ax_x.set_xlabel("time  [s]", color=TICK_COL, fontsize=10)
	ax_x.set_ylabel("displacement  x(t)  [m]", color=TEXT_COL, fontsize=10)
	ax_x.set_title("Displacement vs Time", color=TEXT_COL, fontsize=10)

	ax_pp.set_xlabel("displacement  x  [m]", color=TICK_COL, fontsize=10)
	ax_pp.set_ylabel("velocity  v  [m/s]", color=TICK_COL, fontsize=10)
	ax_pp.set_title("Phase Portrait", color=TEXT_COL, fontsize=10)

	return fig, ax_x, ax_pp


# Draws x(t) and the phase portrait from the first simulation run.
# Returns the line objects so they can be updated in-place later
# without rebuilding the figure from scratch.
def draw_initial_plots(fig, ax_x, ax_pp, x, v):
	(line_x,) = ax_x.plot(T_GRID, x, color=C_X, lw=0.9)
	(line_pp,) = ax_pp.plot(x, v, color=C_PP, lw=0.5, alpha=0.85)

	# White dot marks where the trajectory starts on the phase portrait
	(start_dot,) = ax_pp.plot(x[0], v[0], "o", color="white", markersize=5, zorder=5, label="start")
	ax_pp.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT_COL, edgecolor=GRID_COL)

	return line_x, line_pp, start_dot


# Helper that creates a single styled slider at the given figure position.
# rect = [left, bottom, width, height] in figure coordinates (0 to 1).
def make_slider(fig, rect, label, vmin, vmax, vinit, valfmt="%.2f"):
	ax_s = fig.add_axes(rect, facecolor=SL_BG)
	slider = Slider(ax_s, label, vmin, vmax, valinit=vinit, valfmt=valfmt, color=SL_COL)
	slider.label.set_color(TEXT_COL)
	slider.label.set_fontsize(8)
	slider.valtext.set_color(TEXT_COL)
	return slider


# Builds the full right-hand widget panel: section labels, all five sliders,
# noise-type radio buttons, Re-run button, and the info text annotation.
# Returns everything main() needs to wire up the callbacks.
def build_widget_panel(fig):
	RX = 0.73 # left edge of widget column in figure coordinates
	RW = 0.22 # slider width
	RH = 0.03 # slider height

	fig.text(RX + RW / 2, 0.875, "Physical parameters", ha="center", color=TICK_COL, fontsize=7)
	fig.text(RX + RW / 2, 0.685, "Noise parameters", ha="center", color=TICK_COL, fontsize=7)

	sl_c = make_slider(fig, [RX, 0.82, RW, RH], r"damping  $c$", 0.01, 3.0, DEF_C)
	sl_k = make_slider(fig, [RX, 0.75, RW, RH], r"spring   $k$",  0.50, 12.0, DEF_K)
	sl_scale = make_slider(fig, [RX, 0.64, RW, RH], "noise scale", 0.01, 2.0, DEF_SCALE)
	sl_alpha = make_slider(fig, [RX, 0.57, RW, RH], r"Levy  $\alpha$", 0.50, 2.0, DEF_ALPHA)
	sl_beta = make_slider(fig, [RX, 0.50, RW, RH], r"Levy  $\beta$", -1.0, 1.0, DEF_BETA)

	# Radio buttons to switch between Levy and Gaussian noise
	ax_radio = fig.add_axes([RX + 0.02, 0.36, RW - 0.04, 0.10], facecolor=PANEL)
	ax_radio.set_title("Noise type", color=TEXT_COL, fontsize=8, pad=3)
	radio = RadioButtons(ax_radio, ["Levy stable", r"Gaussian ($\alpha=2$, $\beta=0$)"], activecolor=SL_COL)
	for lbl in radio.labels:
		lbl.set_color(TEXT_COL)
		lbl.set_fontsize(8)

	# Re-run button -- redraws plots with a fresh random seed
	ax_btn = fig.add_axes([RX + 0.05, 0.27, RW - 0.10, 0.05])
	btn = Button(ax_btn, "Re-run  (new seed)", color="#238636", hovercolor="#2ea043")
	btn.label.set_color("white")
	btn.label.set_fontsize(8)

	# Info text showing f0, c_crit, and current regime -- updated on each refresh
	f0_init, c_crit_init, regime_init = compute_physical_quantities(DEF_C, DEF_K)
	info_text = fig.text(
		RX, 0.21,
		rf"$f_0 \approx {f0_init:.3f}\,\mathrm{{Hz}}"
		rf"\quad|\quad"
		rf"c_{{\mathrm{{crit}}}} \approx {c_crit_init:.3f}"
		rf"\quad|\quad"
		rf"\mathrm{{Regime:}}\ \mathrm{{{regime_init}}}$",
		color=TICK_COL, fontsize=7.5, va="top", fontfamily="monospace"
	)

	sliders = {
		"c": sl_c, 
		"k": sl_k,
		"scale": sl_scale, 
		"alpha": sl_alpha, 
		"beta": sl_beta
	}

	return sliders, radio, btn, info_text


# Uses set_xdata / set_ydata + relim + autoscale_view to avoid redrawing
# the entire figure, which is much faster than clearing and replotting.
def update_plots(line_x, line_pp, start_dot, ax_x, ax_pp, x, v):
	line_x.set_ydata(x)
	ax_x.relim()
	ax_x.autoscale_view()

	line_pp.set_xdata(x)
	line_pp.set_ydata(v)
	start_dot.set_xdata([x[0]])
	start_dot.set_ydata([v[0]])
	ax_pp.relim()
	ax_pp.autoscale_view()


# Refreshes the info text below the sliders with current f0, c_crit, and regime.
def update_info_text(info_text, c, k):
	f0, c_crit, regime = compute_physical_quantities(c, k)
	# info_text.set_text(
		# rf"f0 ~ {f0:.3f} Hz  |  c_crit ~ {c_crit:.3f}  |  Regime: {regime}"
	info_text.set_text(
		rf"$f_0 \approx {f0:.3f}\,\mathrm{{Hz}}"
		rf"\quad|\quad"
		rf"c_{{\mathrm{{crit}}}} \approx {c_crit:.3f}"
		rf"\quad|\quad"
		rf"\mathrm{{Regime:}}\ \mathrm{{{regime}}}$"
	)

def main():
	x0, v0 = simulate(DEF_C, DEF_K, DEF_ALPHA, DEF_BETA, DEF_SCALE, noise_type="levy")

	fig, ax_x, ax_pp = build_figure()
	line_x, line_pp, start_dot = draw_initial_plots(fig, ax_x, ax_pp, x0, v0)

	sliders, radio, btn, info_text = build_widget_panel(fig)

	# Dict so the nested refresh() callback can modify noise_state in place
	noise_state = {"type": "levy"}

	# Connected to every slider and the Re-run button.
	# _ because on_click event sends useless stuff
	def refresh(_=None):
		c = sliders["c"].val
		k = sliders["k"].val
		scale = sliders["scale"].val
		alpha = sliders["alpha"].val
		beta = sliders["beta"].val
		ntype = noise_state["type"]

		x, v = simulate(c, k, alpha, beta, scale, noise_type=ntype)

		# Push new data into both panels
		update_plots(line_x, line_pp, start_dot, ax_x, ax_pp, x, v)
		update_info_text(info_text, c, k)

		fig.canvas.draw_idle()

	def on_radio(label):
		noise_state["type"] = "levy" if "Levy" in label else "gaussian"
		refresh()

	for sl in sliders.values():
		sl.on_changed(refresh)
	radio.on_clicked(on_radio)
	btn.on_clicked(refresh)

	plt.show()


if __name__ == "__main__":
	main()