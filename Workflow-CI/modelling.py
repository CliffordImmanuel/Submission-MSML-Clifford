import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    # Mengambil lokasi file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "churn_clean.csv")
    
    if not os.path.exists(data_path):
        print(f"ERROR: File {data_path} TIDAK DITEMUKAN!")
        return

    # 1. Load Data
    df = pd.read_csv(data_path)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 2. MLflow Autolog (Sangat disarankan oleh reviewer)
    mlflow.sklearn.autolog()

    # 3. Training Model
    # Tidak perlu mlflow.start_run() karena sudah otomatis dijalankan oleh 'mlflow run'
    print("Sedang melatih model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    acc = accuracy_score(y, model.predict(X))
    print(f"Berhasil! Akurasi: {acc:.2f}")

if __name__ == "__main__":
    train_model()