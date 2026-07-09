from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def run():
  wine = load_wine()
  scaled = StandardScaler().fit_transform(wine.data)

  pca = PCA(n_components=2, random_state=42)
  reduced = pca.fit_transform(scaled)

  model = KMeans(n_clusters=3, random_state=42, n_init=10)
  clusters = model.fit_predict(reduced)

  print("Unsupervised learning: PCA + KMeans on Wine")
  print(f"Explained variance (2 components): {pca.explained_variance_ratio_.sum():.3f}")
  print(f"Silhouette score: {silhouette_score(reduced, clusters):.3f}")
  print(f"Adjusted Rand Index vs true labels: {adjusted_rand_score(wine.target, clusters):.3f}")
  print("Cluster sizes:")
  for cluster_id in range(model.n_clusters):
    count = (clusters == cluster_id).sum()
    print(f"  cluster {cluster_id}: {count}")
