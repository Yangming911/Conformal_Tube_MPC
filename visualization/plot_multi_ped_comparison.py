import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_CSV = Path(__file__).resolve().parent / "multi_ped_results.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
	df = pd.read_csv(DATA_CSV)
	# Normalize columns
	df["ped_num"] = df["ped_num"].astype(int)
	# Ensure numeric
	df["collision_rate_percent"] = pd.to_numeric(df["collision_rate_percent"], errors="coerce")
	df["speed_m_s"] = pd.to_numeric(df["speed_m_s"], errors="coerce")
	df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
	# Keep only rows with meaningful collision_rate
	df = df.dropna(subset=["collision_rate_percent"], how="any")
	return df


def style_matplotlib():
	# Use pure Matplotlib defaults
	plt.rcParams.update(plt.rcParamsDefault)
	# Unified font sizes via rcParams
	plt.rcParams["font.size"] = 14
	plt.rcParams["axes.labelsize"] = 16
	plt.rcParams["xtick.labelsize"] = 14
	plt.rcParams["ytick.labelsize"] = 14
	plt.rcParams["legend.fontsize"] = 10


def plot_figure(df: pd.DataFrame):
	# Prepare alpha as string while keeping original for labels
	df = df.copy()
	df["alpha_str"] = df["alpha"].astype(str).str.strip()

	# Aggregate duplicates per (alpha, ped_num) by mean
	agg = (
		df.groupby(["alpha_str", "ped_num"], as_index=False)["collision_rate_percent"]
		.mean()
	)

	# Order: Random first, then numeric alphas ascending
	def alpha_sort_key(a):
		if a == "Random":
			return (-1, -1.0)
		try:
			return (0, float(a))
		except ValueError:
			return (1, float("inf"))

	unique_alphas = sorted(agg["alpha_str"].unique().tolist(), key=alpha_sort_key)

	# Figure
	fig, ax = plt.subplots(figsize=(5.0, 3.8))

	# Fixed color mapping per alpha
	def color_for_alpha(a: str):
		if a in ["1", "1.0"]:
			return "#1f77b4"  # blue
		if a == "0.25":
			return "#ff7f0e"  # orange
		if a == "0.5":
			return "#2ca02c"  # green
		if a == "0.15":
			return "#d62728"  # red
		if a == "APF":
			return "#bdbdbd"  # light gray
		if a == "Random":
			return "#9467bd"  # purple
		# fallback to default cycle
		cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
		return cycle[0] if cycle else "#000000"

	# Plot order: APF first (base layer), then other methods, Random last
	others = [a for a in unique_alphas if a not in ["APF", "Random"]]
	plot_order = []
	if "APF" in unique_alphas:
		plot_order.append("APF")
	plot_order.extend(others)
	if "Random" in unique_alphas:
		plot_order.append("Random")

	for a in plot_order:
		grp = agg[agg["alpha_str"] == a].sort_values("ped_num")
		color = color_for_alpha(a)
		# Label naming
		if a == "Random":
			label = "Random"
		elif a in ["1", "1.0"]:
			label = "Vanila CBF (alpha=1)"
		elif a == "APF":
			label = "APF"
		else:
			label = f"CP+CBF (alpha={a})"
		z = 1 if a == "APF" else 2
		ax.plot(grp["ped_num"], grp["collision_rate_percent"], "o-", label=label, color=color, zorder=z)

	ax.set_xlabel("Number of pedestrians")
	ax.set_ylabel("Collision rate (%)")
	# ax.set_title("Collision rate vs pedestrians")
	ax.legend(frameon=True)
	# Ensure x-axis shows 1, 3, 5, 7, 9
	ax.set_xticks([1, 3, 5, 7, 9])
	fig.tight_layout()
	return fig


def plot_speed_figure(df: pd.DataFrame):
	# Prepare alpha as string while keeping original for labels
	df = df.copy()
	df["alpha_str"] = df["alpha"].astype(str).str.strip()

	# Aggregate duplicates per (alpha, ped_num) by mean for speed
	agg = (
		df.groupby(["alpha_str", "ped_num"], as_index=False)["speed_m_s"]
		.mean()
	)

	# Order: Random first, then numeric alphas ascending
	def alpha_sort_key(a):
		if a == "Random":
			return (-1, -1.0)
		try:
			return (0, float(a))
		except ValueError:
			return (1, float("inf"))

	unique_alphas = sorted(agg["alpha_str"].unique().tolist(), key=alpha_sort_key)
	# Exclude Random from speed plot; include APF and alphas
	unique_alphas = [a for a in unique_alphas if a != "Random"]

	# Figure
	fig, ax = plt.subplots(figsize=(5.0, 3.8))

	# Fixed color mapping per alpha (same as collision figure)
	alpha_color = {
		"1": "#1f77b4",
		"1.0": "#1f77b4",
		"0.25": "#ff7f0e",
		"0.5": "#2ca02c",
		"0.15": "#d62728",
		"APF": "#bdbdbd",
		"Random": "#9467bd",
	}

	# Draw APF first if present
	ordered = []
	if "APF" in unique_alphas:
		ordered.append("APF")
	ordered.extend([a for a in unique_alphas if a != "APF"])

	for idx, a in enumerate(ordered):
		grp = agg[agg["alpha_str"] == a].sort_values("ped_num")
		color = alpha_color.get(a, plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#000000"])[0])
		if a == "Random":
			label = "Random"
		elif a in ["1", "1.0"]:
			label = "Vanila CBF (alpha=1)"
		elif a == "APF":
			label = "APF"
		else:
			label = f"CP+CBF (alpha={a})"
		z = 1 if a == "APF" else 2
		ax.plot(grp["ped_num"], grp["speed_m_s"], "s--", label=label, color=color, zorder=z)

	ax.set_xlabel("Number of pedestrians")
	ax.set_ylabel("Average speed (m/s)")
	# ax.set_title("Average speed vs pedestrians")
	ax.legend(frameon=True)
	# Ensure x-axis shows 1, 3, 5, 7, 9
	ax.set_xticks([1, 3, 5, 7, 9])
	fig.tight_layout()
	return fig


def plot_time_figure(df: pd.DataFrame):
	# Prepare alpha as string while keeping original for labels
	df = df.copy()
	df["alpha_str"] = df["alpha"].astype(str).str.strip()

	# Aggregate duplicates per (alpha, ped_num) by mean for time
	agg = (
		df.groupby(["alpha_str", "ped_num"], as_index=False)["time_ms"]
		.mean()
	)

	# Order: Random first, then numeric alphas ascending, then remove Random
	def alpha_sort_key(a):
		if a == "Random":
			return (-1, -1.0)
		try:
			return (0, float(a))
		except ValueError:
			return (1, float("inf"))

	unique_alphas = sorted(agg["alpha_str"].unique().tolist(), key=alpha_sort_key)
	unique_alphas = [a for a in unique_alphas if a != "Random"]

	# Figure
	fig, ax = plt.subplots(figsize=(5.0, 3.8))

	# Fixed color mapping per alpha (same as other figures)
	alpha_color = {
		"1": "#1f77b4",
		"1.0": "#1f77b4",
		"0.25": "#ff7f0e",
		"0.5": "#2ca02c",
		"0.15": "#d62728",
		"APF": "#bdbdbd",
		"Random": "#9467bd",
	}

	# Draw APF first if present
	ordered = []
	if "APF" in unique_alphas:
		ordered.append("APF")
	ordered.extend([a for a in unique_alphas if a != "APF"])

	for a in ordered:
		grp = agg[agg["alpha_str"] == a].sort_values("ped_num")
		color = alpha_color.get(a, plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#000000"])[0])
		if a in ["1", "1.0"]:
			label = "Vanila CBF (alpha=1)"
		elif a == "APF":
			label = "APF"
		else:
			label = f"CP+CBF (alpha={a})"
		z = 1 if a == "APF" else 2
		ax.plot(grp["ped_num"], grp["time_ms"], "^-.", label=label, color=color, zorder=z)

	ax.set_xlabel("Number of pedestrians")
	ax.set_ylabel("Calculation time (ms)")
	# ax.set_title("Calculation time vs pedestrians")
	# Place legend at bottom-right, slightly lifted to avoid overlap
	ax.legend(frameon=True, loc="lower right", bbox_to_anchor=(1.0, 0.07), bbox_transform=ax.transAxes)
	# Ensure x-axis shows 1, 3, 5, 7, 9
	ax.set_xticks([1, 3, 5, 7, 9])
	fig.tight_layout()
	return fig


def main():
	style_matplotlib()
	df = load_data()
	fig = plot_figure(df)
	fig_spd = plot_speed_figure(df)
	fig_time = plot_time_figure(df)
	# Save collision figure
	png_path = RESULTS_DIR / "multi_pedestrian_comparison.png"
	pdf_path = RESULTS_DIR / "multi_pedestrian_comparison.pdf"
	fig.savefig(png_path, bbox_inches="tight")
	fig.savefig(pdf_path, bbox_inches="tight")
	print(f"Saved: {png_path}")
	print(f"Saved: {pdf_path}")
	# Save speed figure
	png_spd = RESULTS_DIR / "multi_pedestrian_speed.png"
	pdf_spd = RESULTS_DIR / "multi_pedestrian_speed.pdf"
	fig_spd.savefig(png_spd, bbox_inches="tight")
	fig_spd.savefig(pdf_spd, bbox_inches="tight")
	print(f"Saved: {png_spd}")
	print(f"Saved: {pdf_spd}")
	# Save calculation time figure
	png_time = RESULTS_DIR / "multi_pedestrian_time.png"
	pdf_time = RESULTS_DIR / "multi_pedestrian_time.pdf"
	fig_time.savefig(png_time, bbox_inches="tight")
	fig_time.savefig(pdf_time, bbox_inches="tight")
	print(f"Saved: {png_time}")
	print(f"Saved: {pdf_time}")


if __name__ == "__main__":
	main()
