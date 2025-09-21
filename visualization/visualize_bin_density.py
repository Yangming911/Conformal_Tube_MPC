import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load bin count data
count_path = "assets/conformal_grid_counts.pkl"
with open(count_path, "rb") as f:
    bin_count = pickle.load(f)

# Initialize heatmaps
heatmap_aperp_zone0 = np.zeros((3, 15))
heatmap_aperp_zone1 = np.zeros((3, 15))
heatmap_apar_zone0 = np.zeros((1, 15))
heatmap_apar_zone1 = np.zeros((1, 15))

# Fill in counts
for (i, j, k, m), count in bin_count.items():
    # a_perp divided into 3 segments (j), a_par divided into 1 segment (k=0)
    if m == 0:
        heatmap_aperp_zone0[j, i] += count
        heatmap_apar_zone0[k, i] += count
    else:
        heatmap_aperp_zone1[j, i] += count
        heatmap_apar_zone1[k, i] += count

# Set axis labels
car_labels = [f"[{i}-{i+1})" for i in range(15)]
aperp_labels = ["[0–1)", "[1–2)", "[2–3)"]
apar_labels = ["[0–1)"]  # Only one segment

# ===== Plot: a_perp zone 0 =====
plt.figure(figsize=(9, 3))
plt.imshow(heatmap_aperp_zone0, cmap='Blues', interpolation='nearest', origin='lower')
plt.colorbar(label='Sample Count')
plt.xticks(np.arange(15), car_labels, rotation=45)
plt.yticks(np.arange(3), aperp_labels)
plt.xlabel('Car Speed Range (m/s)', fontsize=12)
plt.ylabel(r'$a_{\perp}$ Range (m/s)', fontsize=12)
plt.title(r'Sample Count Heatmap of $a_{\perp}$ vs Car Speed', fontsize=13)
plt.tight_layout()
plt.show()

# ===== Plot: a_perp zone 1 =====
plt.figure(figsize=(9, 3))
plt.imshow(heatmap_aperp_zone1, cmap='Blues', interpolation='nearest', origin='lower')
plt.colorbar(label='Sample Count')
plt.xticks(np.arange(15), car_labels, rotation=45)
plt.yticks(np.arange(3), aperp_labels)
plt.xlabel('Car Speed Range (m/s)', fontsize=12)
plt.ylabel(r'$a_{\perp}$ Range (m/s)', fontsize=12)
plt.title(r'Sample Count Heatmap of $a_{\perp}$ vs Car Speed (zone 1)', fontsize=13)
plt.tight_layout()
plt.show()

# # ===== Plot: a_par zone 0 =====
# plt.figure(figsize=(9, 2.5))
# plt.imshow(heatmap_apar_zone0, cmap='Reds', interpolation='nearest', origin='lower')
# plt.colorbar(label='Sample Count')
# plt.xticks(np.arange(15), car_labels, rotation=45)
# plt.yticks(np.arange(1), apar_labels)
# plt.xlabel('Car Speed Range (dm/s)', fontsize=12)
# plt.ylabel(r'$a_{\parallel}$ Range (dm/s)', fontsize=12)
# plt.title(r'Sample Count Heatmap of $a_{\parallel}$ vs Car Speed (zone 0)', fontsize=13)
# plt.tight_layout()
# plt.show()

# # ===== Plot: a_par zone 1 =====
# plt.figure(figsize=(9, 2.5))
# plt.imshow(heatmap_apar_zone1, cmap='Reds', interpolation='nearest', origin='lower')
# plt.colorbar(label='Sample Count')
# plt.xticks(np.arange(15), car_labels, rotation=45)
# plt.yticks(np.arange(1), apar_labels)
# plt.xlabel('Car Speed Range (dm/s)', fontsize=12)
# plt.ylabel(r'$a_{\parallel}$ Range (dm/s)', fontsize=12)
# plt.title(r'Sample Count Heatmap of $a_{\parallel}$ vs Car Speed (zone 1)', fontsize=13)
# plt.tight_layout()
# plt.show()