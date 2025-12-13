# %% load and filter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

processed_dir = Path("../data/processed")
stats = pd.read_csv(processed_dir / "player_stats_norm_hard_2018_2024.csv")

features = [
    "aces_per_svpt",
    "df_rate",
    "first_serve_in_pct",
    "first_serve_won_pct",
    "second_serve_won_pct",
    "bp_faced_per_match",
    "bp_saved_pct",
    "return_points_won_pct",
    "bp_converted_pct",
]

X = stats[[c + "_zscore" for c in features]]

# %% scale (so PCA doesn't get influence by scale difference between stats)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% PCA

pca = PCA(n_components=0.8)  # keep enough components to explain 80% of variance
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)

# %% silhouette analysis
k_range = range(2, 10)
sil_means = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    sil_means.append(sil)
    print(f"k={k}: silhouette:{sil:.3f}")

best_k = k_range[int(np.argmax(sil_means))]
print(f"Best k by mean silhouette: {best_k}")
# %% silhouette plot for chosen k
km = KMeans(n_clusters=best_k, random_state=42, n_init=50)
labels = km.fit_predict(X_pca)
sample_sil = silhouette_samples(X_pca, labels)

fig, ax = plt.subplots(figsize=(7, 5))
y_lower = 0
for i in range(best_k):
    vals = np.sort(sample_sil[labels == i])
    size = len(vals)
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size, str(i))
    y_lower = y_upper

ax.axvline(np.mean(sample_sil), color="red", linestyle="--", label="mean silhouette")
ax.set_xlabel("Silhouette coefficient")
ax.set_ylabel("Cluster")
ax.set_title(f"Silhouette plot (k={best_k})")
ax.legend()
plt.tight_layout()
plt.show()

# %% k-means

km = KMeans(n_clusters=best_k, random_state=42, n_init=50)
labels = km.fit_predict(X_pca)

stats["cluster"] = labels

# %% visualize in PCA space
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
# for i, name in enumerate(stats["player_name"]):
#     if stats.loc[i, "num_of_matches"] > 200:  # label only big names to avoid clutter
#         plt.text(X_pca[i, 0], X_pca[i, 1], name, fontsize=8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Player Archetypes on Hard Court (2018-2024)")
plt.colorbar(scatter, label="Cluster")
plt.show()

# %% interpret

cluster_profiles = stats.groupby("cluster")[features + ["win_rate", "avg_rank"]].mean()
print(cluster_profiles)
# %% feature importance

df = pd.DataFrame(X_scaled, columns=features)
y = labels

# centroid shifts
global_mean = df.mean()
cluster_means = df.groupby(y).mean()
shift_score = (cluster_means - global_mean).abs().max().sort_values(ascending=False)

print(shift_score)
