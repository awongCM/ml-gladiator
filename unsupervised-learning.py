from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def run():
  wine = load_wine()
  scaled = StandardScaler().fit_transform(wine.data)

  pca = PCA(n_components=2, random_state=42)
  pca.fit(scaled)

  k_candidates = range(2, 6)
  silhouette_by_k = {}
  for k in k_candidates:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    silhouette_by_k[k] = silhouette_score(scaled, labels)

  best_k = max(silhouette_by_k, key=silhouette_by_k.get)
  model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
  clusters = model.fit_predict(scaled)

  print("Unsupervised learning: PCA + KMeans on Wine")
  print(f"Explained variance (2 components): {pca.explained_variance_ratio_.sum():.3f}")
  print("Silhouette by k (on scaled features, not the 2D PCA plane):")
  for k in k_candidates:
    marker = " <-- selected" if k == best_k else ""
    print(f"  k={k}: {silhouette_by_k[k]:.3f}{marker}")
  print(f"Silhouette score (k={best_k}): {silhouette_by_k[best_k]:.3f}")
  # Labels are an external check only; they were not used to choose k or fit KMeans.
  print(
    f"Adjusted Rand Index vs true labels (external check): "
    f"{adjusted_rand_score(wine.target, clusters):.3f}"
  )
  print("Cluster sizes:")
  for cluster_id in range(model.n_clusters):
    count = (clusters == cluster_id).sum()
    print(f"  cluster {cluster_id}: {count}")
