import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"

def cluster_countries(df : pd.DataFrame, k=4):
    agg = df.groupby('country', as_index = False).agg(
    total_waste = ('total_waste_(tons)','sum'),
    economic_loss = ('economic_loss_(million_$)','sum'),
    per_capita_waste_kg = ('per_capita_waste_kg','mean'),
    household_waste_pct = ('household_waste_(%)','mean')  
    ).fillna(0)

    #Select relevant features for clustering
    X = agg[['total_waste','economic_loss','per_capita_waste_kg','household_waste_pct']].values

    Xs = StandardScaler().fit_transform(X)

    #KMeans clustering
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)

    #Assign clusters
    agg['cluster'] = km.labels_
    return agg, km

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(BASE_DIR,"data","processed","food_waste_clean.csv")
    output_path = os.path.join(BASE_DIR,"outputs","country_clusters.csv")

    print(f"Loading dataset from : {input_path}")
    df = pd.read_csv(input_path)

    #Elbow method to find best k
    print("Running Elbow Method : ")
    inertia = []
    K = range(2,11)
    agg = df.groupby('country', as_index = False).agg(
    total_waste = ('total_waste_(tons)','sum'),
    economic_loss = ('economic_loss_(million_$)','sum'),
    per_capita_waste_kg = ('per_capita_waste_kg','mean'),
    household_waste_pct = ('household_waste_(%)','mean')  
    ).fillna(0)

    X = agg[['total_waste','economic_loss','per_capita_waste_kg','household_waste_pct']].values
    Xs = StandardScaler().fit_transform(X)

    for k in K:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        inertia.append(km.inertia_)
    plt.figure(figsize=(8,6))
    plt.plot(K, inertia, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.savefig(os.path.join(BASE_DIR, "outputs", "elbow_plot.png"))
    plt.show()

    clusters, kmodel = cluster_countries(df, k=4)
    score = silhouette_score(Xs, clusters['cluster'])
    print("Clustering is done with k=4")
    print(f"Silhouette score is : {score:.2f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clusters.to_csv(output_path, index=False)
    print(f"Clustered data saved at: {output_path}")

    print(clusters.head())