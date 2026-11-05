from sklearn.datasets import load_iris
import pandas as pd
import os

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/iris.csv", index=False)

print("Dataset Iris généré avec succès")
print("Nombre de lignes :", len(df))
